// The command stream: draws, and with BBHOST_STREAM_SUBMIT the command
// buffers themselves, recorded on their own thread (BBHOST_RECORDER, on
// unless 0).
//
// Of each draw's ~11 us on the command processor, ~2.5 went to the driver:
// vkUpdateDescriptorSets, then the binds, dynamic state and draw call. A draw
// puts those in a packet, and a recorder thread (bb-record) replays them into
// the command buffer in order while the command processor resolves the next
// draw. The packets travel in one ordered stream of blocks; a block is an
// arena of typed op records (host/stream_ops.h) - a draw's packet is one op,
// kDrawState, and every other command the stream carries is an op of its own:
// barriers, fills, copies, query resets, timestamps, dispatches, and the
// command buffer's begin and its end-and-submit.
//
// With BBHOST_STREAM_SUBMIT (on unless 0, or with the recorder off) bb-record
// owns the command pool's buffers: it begins each one (kBegin), records into
// it, ends it and hands the submission to bb-submit (kEndSubmit). The submit
// path's barriers, label fills and copies are ops, so a submission no longer
// waits for the recorder to finish the draws before it, and the command
// processor makes no driver call for it at all. The presenter's blit goes
// into the stream too (kForeignSubmit), behind the frame it shows. Anything
// still recorded in place takes the command buffer through g_cmd(), which
// publishes what is open and waits for the recorder to replay every block
// published so far: one ordered stream of commands, whichever thread
// records them. BBHOST_STREAM_SUBMIT=0: the command processor begins, ends
// and submits, and every op is recorded in place, as before the stream.
//
// Only the thread holding g.mu appends, publishes or drains. The recorder
// takes no lock: it owns the command buffer between a publish and the drain
// that follows. The descriptor sets it writes belong to the packet or op
// (vkUpdateDescriptorSets needs only the set synchronised, not its pool, so
// the command processor keeps allocating). The buffer shadow's copies go to
// a command buffer of their own pool, ended on the command processor.
//
// Write combining. AMD's Windows driver records into
// write-combined memory, and x86 orders a core's write-combining buffers
// against later stores only with a fence or a locked instruction. So every
// hand-over fences on the writing core: the recorder after each block it
// replays (R1); the producer before it publishes (R2: the command processor's
// writes to host-visible memory the GPU reads after a submission the
// recorder makes, and the presenter's own command buffer;
// BBHOST_STREAM_FENCE=kick fences only at the submissions and after an
// in-place section); the producer before its first publish after it
// recorded in place (R3); and the recorder again before a command buffer goes
// to bb-submit (R4).

#include "host/gpu_internal.h"
#include "host/stream_ops.h"

#include <algorithm>
#include "core/host_clock.h"
#include "core/portable.h"

#include "log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace gpu {

const char* stream::op_name(stream::Op op) {
    static const char* const names[] = {"begin",          "end+submit",   "foreign submit", "draw",       "barrier",       "transfer barrier",
                                        "copy order",     "pass barrier", "end rendering",  "begin rendering", "fill",     "copy",
                                        "copy to image",  "copy to buffer", "clear",        "reset queries", "begin query", "end query",
                                        "copy queries",   "timestamp",    "bind pipeline",  "bind sets",     "push constants", "update sets",
                                        "dispatch",       "clear depth",  "copy image",     "dispatch indirect", "marker"};
    static_assert(sizeof(names) / sizeof(names[0]) == static_cast<std::size_t>(stream::Op::kCount), "a name for every op");
    const auto k = static_cast<std::size_t>(op);
    return k < static_cast<std::size_t>(stream::Op::kCount) ? names[k] : "?";
}

StreamStats g_stream_stats;

namespace {

using stream::Op;

inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#endif
}

struct alignas(64) DrawPacket {  // a line of its own: the recorder reads the one before
    std::uint64_t number = 0;  // its place in the packet ring's sequence
    // Descriptor writes, applied first and in order. Each write's info
    // pointer is an index into buffers / images until the replay.
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    std::vector<std::int32_t> write_buffer, write_image;  // first info of each write, or -1
    // take_sets: a draw's writes, their pointers already final.
    std::vector<VkWriteDescriptorSet> taken_writes;
    std::vector<VkDescriptorImageInfo> taken_images;
    std::vector<VkDescriptorBufferInfo> taken_buffers;
    VkDescriptorBufferInfo extra[2] = {};

    VkPipeline pipeline = VK_NULL_HANDLE;  // null: keep the bound one
    bool library = false;
    DrawLibraryState lib{};
    std::uint32_t library_fields = kLibraryAll;
    bool has_depth_bounds = false;
    bool bind_sets = false;
    bool bindless = false;  // set 3, the global views and samplers, too
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet sets[2] = {};
    std::uint32_t dynamic_count = 0, dynamic[2] = {};
    std::uint32_t pushes = 0;  // constant buffers pushed into set 2 after the binds: binding and range
    std::uint32_t push_binding[2 * gcn::kMaxBuffers];
    VkDescriptorBufferInfo push_infos[2 * gcn::kMaxBuffers];
    bool viewport = false, scissor = false, bounds = false, stencil = false, blend = false;
    VkViewport vp{};
    VkRect2D sc{};
    float depth_bounds[2] = {};
    std::uint32_t stencil_words[6] = {};
    float blend_const[4] = {};
    bool index = false;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceSize index_offset = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
    std::uint32_t vertex_buffers = 0;
    VkBuffer vb[kMaxVertexBindings] = {};
    VkDeviceSize vb_offset[kMaxVertexBindings] = {};
    DrawCall call{};
    bool draw = false;
    // AMD's buffer markers just before the draw, on a start after a lost
    // device (gpu_write_markers): here, so such a start still records its
    // draws on the recorder.
    bool marker = false;
    std::uint32_t marker_value = 0;
    // Pass changes, replayed after the descriptor writes and before the binds.
    bool end_pass = false, begin_pass = false;
    bool end_pass_barrier = true;  // the end's barrier with it (lazy pass barriers leave it out)
    bool pass_barrier = false;     // a pass end's barrier owed from earlier, after end_pass
    // A transfer batch, between the pass end and the pass begin: the barrier
    // into the transfer stage, buffer copies, the barrier out of it.
    bool xfer_begin = false, xfer_end = false;
    // A copy-version batch is ~25 copies; past this many the rest are
    // ops of the stream (or, without it, recorded in place).
    static constexpr std::uint32_t kCopies = 32;
    std::uint32_t ncopies = 0;
    std::uint32_t copy_orders = 0;  // bit k: the copies before copy k finish first (copy_order)
    VkBuffer copy_src[kCopies] = {}, copy_dst[kCopies] = {};
    VkBufferCopy copies[kCopies] = {};
    // Region copies (textures.cpp rt_copied_view), after the transfer batch:
    // a draw binds up to a few of a glare pyramid's levels.
    static constexpr std::uint32_t kRegionCopies = 4;
    std::uint32_t nregions = 0;
    struct RegionCopy {
        VkImage src, dst;
        std::uint32_t width, height;
        bool dst_initialised;
    } regions[kRegionCopies] = {};
    VkRenderingInfo rendering{};
    VkRenderingAttachmentInfo colors[8]{}, depth_att{}, stencil_att{};
    bool has_depth = false, has_stencil = false;

    void reset() {
        taken_writes.clear();
        writes.clear();
        buffers.clear();
        images.clear();
        write_buffer.clear();
        write_image.clear();
        pipeline = VK_NULL_HANDLE;
        library = bind_sets = viewport = scissor = bounds = stencil = blend = index = draw = marker = false;
        end_pass = begin_pass = false;
        end_pass_barrier = true;
        pass_barrier = false;
        xfer_begin = xfer_end = false;
        ncopies = 0;
        copy_orders = 0;
        nregions = 0;
        pushes = 0;
        vertex_buffers = 0;
    }
};

// Nothing a pass change must precede is in the packet yet.
bool before_binds(const DrawPacket& p) {
    return !p.pipeline && !p.library && !p.bind_sets && !p.pushes && !p.viewport && !p.scissor && !p.bounds && !p.stencil && !p.blend &&
           !p.index && !p.vertex_buffers && !p.draw;
}

// A block of the stream: op records, in the order they were appended.
struct alignas(64) Block {  // the same
    VkCommandBuffer cmd = VK_NULL_HANDLE;  // the command buffer it records into
    std::vector<std::uint64_t> words;      // the records; grown, never shrunk
    std::size_t used = 0;                  // words in use
    std::uint32_t ops = 0;
    bool kick = false;
};

// Blocks in flight: about two frames of draws here (one a draw).
constexpr std::uint64_t kBlocks = 2048;
// Draw packets in flight. A packet stays open while its draw resolves, so
// it lives beside the blocks; its kDrawState op goes into the block open
// when the draw is done.
constexpr std::uint64_t kPackets = 1024;
// A block whose records pass this many words is handed over at once.
constexpr std::size_t kBlockWords = 4096;

enum Mode : int { kModeOff, kModeInline, kModeThread };

Block* g_blocks = nullptr;
Block g_inline_block;  // kModeInline: replayed as it is published
DrawPacket* g_packets = nullptr;
// Each counter on a cache line of its own: the producer writes g_published
// at every draw and the recorder g_done and g_pk_done after every block, and
// sharing a line with them made every publish a miss across the cores (two
// core complexes on a Ryzen AI Max: ~600 ns a draw packet in the self-test,
// ~100 once apart). The producer reads the recorder's counters only when its
// own copy says a ring may be full.
alignas(64) std::atomic<std::uint64_t> g_published{0};  // blocks handed over (written under g.mu)
alignas(64) std::atomic<std::uint64_t> g_done{0};       // blocks replayed (written by the recorder)
alignas(64) std::atomic<std::uint64_t> g_pk_done{0};    // packets replayed: the last one's number + 1
alignas(64) std::atomic<bool> g_sleeping{false};        // written by the recorder only as it sleeps and wakes
alignas(64) std::uint64_t g_pk_taken = 0;               // packets taken (under g.mu)
std::uint64_t g_done_seen = 0, g_pk_done_seen = 0;      // the producer's last reads of g_done and g_pk_done
bool g_block_open = false;                  // the block at g_published is being appended to (under g.mu)
bool g_inplace_dirty = false;               // recorded in place since the last publish (R3)
DrawCmds* g_open = nullptr;  // under g.mu
bool g_thread_started = false;

// Producer counters (under g.mu).
struct Counts {
    std::uint64_t drains = 0, drain_waits = 0, drain_ns = 0;
    std::uint64_t full_waits = 0, packet_waits = 0;
    std::uint64_t submitted_waits = 0, submitted_wait_ns = 0;
    std::uint64_t blocks = 0, draws = 0;
    std::uint64_t ops[static_cast<std::size_t>(Op::kCount)] = {};
};
Counts g_counts;

// The recorder's own counters (written by it alone), and what it replayed
// last, for the device-loss report: lines of their own, away from what the
// producer touches per draw.
struct alignas(64) RecorderOwn {
    std::atomic<std::uint64_t> spin_ns{0}, replay_ns{0}, sleeps{0};
    std::atomic<std::uint32_t> last_kind{0xffff};
    std::atomic<std::uint64_t> last_block{0}, last_serial{~0ull};
};
RecorderOwn g_own;
std::atomic<std::uint64_t>& g_spin_ns = g_own.spin_ns;
std::atomic<std::uint64_t>& g_replay_ns = g_own.replay_ns;
std::atomic<std::uint64_t>& g_sleeps = g_own.sleeps;
std::atomic<std::uint32_t>& g_last_kind = g_own.last_kind;
std::atomic<std::uint64_t>& g_last_block = g_own.last_block;
std::atomic<std::uint64_t>& g_last_serial = g_own.last_serial;
inline void add_relaxed(std::atomic<std::uint64_t>& c, std::uint64_t n) { c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed); }

int g_mode = [] {
    const char* e = std::getenv("BBHOST_RECORDER");
    if (e && e[0] == '0') return kModeOff;
    // BBHOST_RECORDER_INLINE=1 (a diagnostic): the blocks are replayed on the
    // publishing thread at once, no recorder thread - the stream's command
    // order, recorded in place.
    const char* i = std::getenv("BBHOST_RECORDER_INLINE");
    return i && i[0] == '1' ? kModeInline : kModeThread;
}();
// BBHOST_STREAM_SUBMIT=0 (or BBHOST_STREAM=0): the command processor begins,
// ends and submits, and every op is recorded in place.
bool g_submit_on = [] {
    const char* e = std::getenv("BBHOST_STREAM_SUBMIT");
    const char* s = std::getenv("BBHOST_STREAM");
    return !(e && e[0] == '0') && !(s && s[0] == '0');
}();
// BBHOST_STREAM_FENCE=kick: the producer fences only at the submissions and
// after an in-place section, not at every publish (R2).
const bool g_fence_all = [] {
    const char* e = std::getenv("BBHOST_STREAM_FENCE");
    return !(e && std::strcmp(e, "kick") == 0);
}();
// BBHOST_CMD_CENSUS=1: the in-place recording sites, counted.
const bool g_census_on = [] {
    const char* e = std::getenv("BBHOST_CMD_CENSUS");
    return e && e[0] == '1';
}();
std::unordered_map<void*, std::uint64_t> g_census;  // under g.mu

// How long the recorder spins for the next block before it sleeps, in
// pauses (BBHOST_RECORDER_SPIN, 4000). That rides the few microseconds between
// a burst's draws. On a Steam Deck 61% of the recorder's time is this loop and
// 12% its work, but spinning less did not give its GPU the power: 300 and 50
// pauses, windowed, 56.7 and 56.6 fps against 57.0 - the command processor
// waited longer on the recorder's wake-ups instead.
const int g_recorder_spin = [] {
    const char* e = std::getenv("BBHOST_RECORDER_SPIN");
    return e && *e ? std::max(0, std::atoi(e)) : 4000;
}();

void replay_packet(DrawPacket& p, VkCommandBuffer cmd) {
    if (!p.taken_writes.empty()) {
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(p.taken_writes.size()), p.taken_writes.data(), 0, nullptr);
    }
    if (!p.writes.empty()) {
        for (std::size_t i = 0; i < p.writes.size(); ++i) {
            VkWriteDescriptorSet& w = p.writes[i];
            if (p.write_buffer[i] >= 0) w.pBufferInfo = &p.buffers[static_cast<std::size_t>(p.write_buffer[i])];
            if (p.write_image[i] >= 0) w.pImageInfo = &p.images[static_cast<std::size_t>(p.write_image[i])];
        }
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(p.writes.size()), p.writes.data(), 0, nullptr);
    }
    if (p.end_pass) record_end_rendering(cmd, p.end_pass_barrier);
    if (p.pass_barrier) record_pass_barrier(cmd);
    if (p.xfer_begin) record_transfer_barrier(cmd, true);
    for (std::uint32_t k = 0; k < p.ncopies; ++k) {
        if (p.copy_orders >> k & 1) record_copy_order_barrier(cmd);
        vkCmdCopyBuffer(cmd, p.copy_src[k], p.copy_dst[k], 1, &p.copies[k]);
    }
    if (p.xfer_end) record_transfer_barrier(cmd, false);
    for (std::uint32_t k = 0; k < p.nregions; ++k) {
        const DrawPacket::RegionCopy& r = p.regions[k];
        record_region_copy(cmd, r.src, r.dst, r.width, r.height, r.dst_initialised);
    }
    if (p.begin_pass) {
        p.rendering.pColorAttachments = p.colors;
        p.rendering.pDepthAttachment = p.has_depth ? &p.depth_att : nullptr;
        p.rendering.pStencilAttachment = p.has_stencil ? &p.stencil_att : nullptr;
        vkCmdBeginRendering(cmd, &p.rendering);
    }
    if (p.pipeline) vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
    if (p.library) record_library_state(cmd, p.lib, p.has_depth_bounds, p.library_fields);
    if (p.bind_sets) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 2, p.sets, p.dynamic_count, p.dynamic);
    if (p.bind_sets && p.bindless) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 3, 1, &g.bindless_set, 0, nullptr);
    if (p.pushes) {
        VkWriteDescriptorSet writes[2 * gcn::kMaxBuffers];
        for (std::uint32_t k = 0; k < p.pushes; ++k) {
            writes[k] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[k].dstBinding = p.push_binding[k];
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &p.push_infos[k];
        }
        g.cmd_push_descriptor_set(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 2, p.pushes, writes);
    }
    if (p.viewport) vkCmdSetViewport(cmd, 0, 1, &p.vp);
    if (p.scissor) vkCmdSetScissor(cmd, 0, 1, &p.sc);
    if (p.bounds) vkCmdSetDepthBounds(cmd, p.depth_bounds[0], p.depth_bounds[1]);
    if (p.stencil) record_stencil_words(cmd, p.stencil_words);
    if (p.blend) vkCmdSetBlendConstants(cmd, p.blend_const);
    if (p.index) vkCmdBindIndexBuffer(cmd, p.index_buffer, p.index_offset, p.index_type);
    if (p.vertex_buffers) vkCmdBindVertexBuffers(cmd, 0, p.vertex_buffers, p.vb, p.vb_offset);
    if (p.marker) gpu_write_markers(cmd, p.marker_value);
    if (p.draw) record_draw_call(cmd, p.call);
}

// ---- reading records ----

constexpr std::size_t round8(std::size_t n) { return (n + 7) & ~static_cast<std::size_t>(7); }

template <class T> const T* tail(const std::uint8_t*& at, std::size_t n) {
    const T* t = reinterpret_cast<const T*>(at);
    at += round8(n * sizeof(T));
    return t;
}

void replay_block(const Block& b) {
    const VkCommandBuffer cmd = b.cmd;
    std::size_t at = 0;
    while (at < b.used) {
        const auto* h = reinterpret_cast<const stream::Header*>(&b.words[at]);
        const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(h + 1);
        g_last_kind.store(static_cast<std::uint32_t>(h->kind), std::memory_order_relaxed);
        switch (h->kind) {
        case Op::kBegin: {
            const auto* r = reinterpret_cast<const stream::Begin*>(p);
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            const std::uint64_t t0 = host_clock_monotonic_ns();
            vkBeginCommandBuffer(r->cmd, &bi);
            add_relaxed(g_stream_stats.endbegin_ns, host_clock_monotonic_ns() - t0);
            break;
        }
        case Op::kEndSubmit: {
            const auto* r = reinterpret_cast<const stream::EndSubmit*>(p);
            const std::uint64_t t0 = host_clock_monotonic_ns();
            vkEndCommandBuffer(r->cmds[r->ncmds - 1]);
            add_relaxed(g_stream_stats.endbegin_ns, host_clock_monotonic_ns() - t0);
            add_relaxed(g_stream_stats.endbegin_n, 1);
            // R4: this core's commands are in memory before another thread
            // submits them (enqueue's mutex would also drain them).
            write_combine_fence();
            g_last_serial.store(r->serial, std::memory_order_relaxed);
            stream_submit_job(r->cmds, r->ncmds, r->wait, r->wait_stage, VK_NULL_HANDLE, r->fence, r->items, r->serial, 0);
            break;
        }
        case Op::kForeignSubmit: {
            const auto* r = reinterpret_cast<const stream::ForeignSubmit*>(p);
            stream_submit_job(&r->cmd, 1, r->wait, r->wait_stage, r->signal, r->fence, 0, ~0ull, r->id);
            break;
        }
        case Op::kDrawState: {
            const auto* r = reinterpret_cast<const stream::DrawState*>(p);
            replay_packet(g_packets[r->packet % kPackets], cmd);
            g_pk_done.store(r->packet + 1, std::memory_order_release);
            break;
        }
        case Op::kBarrier: {
            const auto* r = reinterpret_cast<const stream::Barrier*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            const auto* mem = tail<VkMemoryBarrier>(q, r->nmem);
            const auto* buf = tail<VkBufferMemoryBarrier>(q, r->nbuf);
            const auto* img = tail<VkImageMemoryBarrier>(q, r->nimg);
            vkCmdPipelineBarrier(cmd, r->src, r->dst, r->dep, r->nmem, r->nmem ? mem : nullptr, r->nbuf, r->nbuf ? buf : nullptr, r->nimg,
                                 r->nimg ? img : nullptr);
            break;
        }
        case Op::kTransferBarrier: record_transfer_barrier(cmd, reinterpret_cast<const stream::Flag*>(p)->value != 0); break;
        case Op::kCopyOrderBarrier: record_copy_order_barrier(cmd); break;
        case Op::kPassBarrier: record_pass_barrier(cmd); break;
        case Op::kEndRendering: record_end_rendering(cmd, reinterpret_cast<const stream::Flag*>(p)->value != 0); break;
        case Op::kBeginRendering: {
            const auto* r = reinterpret_cast<const stream::BeginRendering*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            const auto* att = tail<VkRenderingAttachmentInfo>(q, r->ncolor + r->has_depth + r->has_stencil);
            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.flags = r->flags;
            ri.renderArea = r->area;
            ri.layerCount = r->layers;
            ri.viewMask = r->view_mask;
            ri.colorAttachmentCount = r->ncolor;
            ri.pColorAttachments = r->ncolor ? att : nullptr;
            ri.pDepthAttachment = r->has_depth ? att + r->ncolor : nullptr;
            ri.pStencilAttachment = r->has_stencil ? att + r->ncolor + r->has_depth : nullptr;
            vkCmdBeginRendering(cmd, &ri);
            break;
        }
        case Op::kFill: {
            const auto* r = reinterpret_cast<const stream::Fill*>(p);
            vkCmdFillBuffer(cmd, r->buffer, r->offset, r->size, r->data);
            break;
        }
        case Op::kCopyBuffer: {
            const auto* r = reinterpret_cast<const stream::CopyBuffer*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            vkCmdCopyBuffer(cmd, r->src, r->dst, r->n, tail<VkBufferCopy>(q, r->n));
            break;
        }
        case Op::kCopyBufferToImage: {
            const auto* r = reinterpret_cast<const stream::CopyBufferToImage*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            vkCmdCopyBufferToImage(cmd, r->buffer, r->image, r->layout, r->n, tail<VkBufferImageCopy>(q, r->n));
            break;
        }
        case Op::kCopyImageToBuffer: {
            const auto* r = reinterpret_cast<const stream::CopyImageToBuffer*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            vkCmdCopyImageToBuffer(cmd, r->image, r->layout, r->buffer, r->n, tail<VkBufferImageCopy>(q, r->n));
            break;
        }
        case Op::kClearColor: {
            const auto* r = reinterpret_cast<const stream::ClearColor*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            vkCmdClearColorImage(cmd, r->image, r->layout, &r->color, r->n, tail<VkImageSubresourceRange>(q, r->n));
            break;
        }
        case Op::kResetQueries: {
            const auto* r = reinterpret_cast<const stream::Queries*>(p);
            vkCmdResetQueryPool(cmd, r->pool, r->first, r->count);
            break;
        }
        case Op::kBeginQuery: {
            const auto* r = reinterpret_cast<const stream::Queries*>(p);
            vkCmdBeginQuery(cmd, r->pool, r->first, r->flags);
            break;
        }
        case Op::kEndQuery: {
            const auto* r = reinterpret_cast<const stream::Queries*>(p);
            vkCmdEndQuery(cmd, r->pool, r->first);
            break;
        }
        case Op::kCopyQueryResults: {
            const auto* r = reinterpret_cast<const stream::CopyQueryResults*>(p);
            vkCmdCopyQueryPoolResults(cmd, r->pool, r->first, r->count, r->dst, r->offset, r->stride, r->flags);
            break;
        }
        case Op::kTimestamp: {
            const auto* r = reinterpret_cast<const stream::Timestamp*>(p);
            vkCmdWriteTimestamp(cmd, r->stage, r->pool, r->query);
            break;
        }
        case Op::kBindPipeline: {
            const auto* r = reinterpret_cast<const stream::BindPipeline*>(p);
            vkCmdBindPipeline(cmd, r->bind_point, r->pipeline);
            break;
        }
        case Op::kBindSets: {
            const auto* r = reinterpret_cast<const stream::BindSets*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            const auto* sets = tail<VkDescriptorSet>(q, r->n);
            const auto* dyn = tail<std::uint32_t>(q, r->ndynamic);
            vkCmdBindDescriptorSets(cmd, r->bind_point, r->layout, r->first, r->n, sets, r->ndynamic, r->ndynamic ? dyn : nullptr);
            break;
        }
        case Op::kPushConstants: {
            const auto* r = reinterpret_cast<const stream::PushConstants*>(p);
            vkCmdPushConstants(cmd, r->layout, r->stages, r->offset, r->size, p + round8(sizeof(*r)));
            break;
        }
        case Op::kUpdateSets: {
            const auto* r = reinterpret_cast<const stream::UpdateSets*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            const auto* ws = tail<VkWriteDescriptorSet>(q, r->n);
            const auto* fb = tail<std::int32_t>(q, r->n);
            const auto* fi = tail<std::int32_t>(q, r->n);
            const auto* bufs = tail<VkDescriptorBufferInfo>(q, r->nbuffers);
            const auto* imgs = tail<VkDescriptorImageInfo>(q, r->nimages);
            VkWriteDescriptorSet local[16];
            std::vector<VkWriteDescriptorSet> many;
            VkWriteDescriptorSet* w = local;
            if (r->n > 16) {
                many.resize(r->n);
                w = many.data();
            }
            for (std::uint32_t k = 0; k < r->n; ++k) {
                w[k] = ws[k];
                if (fb[k] >= 0) w[k].pBufferInfo = bufs + fb[k];
                if (fi[k] >= 0) w[k].pImageInfo = imgs + fi[k];
            }
            vkUpdateDescriptorSets(g.device, r->n, w, 0, nullptr);
            break;
        }
        case Op::kDispatch: {
            const auto* r = reinterpret_cast<const stream::Dispatch*>(p);
            vkCmdDispatch(cmd, r->x, r->y, r->z);
            break;
        }
        case Op::kClearDepth: {
            const auto* r = reinterpret_cast<const stream::ClearDepth*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            vkCmdClearDepthStencilImage(cmd, r->image, r->layout, &r->value, r->n, tail<VkImageSubresourceRange>(q, r->n));
            break;
        }
        case Op::kMarker: {
            gpu_write_markers(cmd, reinterpret_cast<const stream::Marker*>(p)->value);
            break;
        }
        case Op::kCopyImage: {
            const auto* r = reinterpret_cast<const stream::CopyImage*>(p);
            const std::uint8_t* q = p + round8(sizeof(*r));
            vkCmdCopyImage(cmd, r->src, r->src_layout, r->dst, r->dst_layout, r->n, tail<VkImageCopy>(q, r->n));
            break;
        }
        case Op::kDispatchIndirect: {
            const auto* r = reinterpret_cast<const stream::DispatchIndirect*>(p);
            vkCmdDispatchIndirect(cmd, r->buffer, r->offset);
            break;
        }
        case Op::kCount: break;
        }
        at += h->bytes / 8;
    }
}

void recorder_thread() {
    host_thread_set_name("bb-record");
    host_thread_set_class(HostThreadClass::GpuFeed, "bb-record");  // with the command processor it serves
    const int spin = g_recorder_spin;
    std::uint64_t done = g_done.load(std::memory_order_acquire);
    for (;;) {
        std::uint64_t published = g_published.load(std::memory_order_acquire);
        if (published == done) {
            // Draws come in bursts a few microseconds apart: spin through
            // those, sleep through the gaps between frames.
            const std::uint64_t t0 = host_clock_monotonic_ns();
            for (int i = 0; i < spin && published == done; ++i) {
                cpu_relax();
                published = g_published.load(std::memory_order_acquire);
            }
            add_relaxed(g_spin_ns, host_clock_monotonic_ns() - t0);
            if (published == done) {
                add_relaxed(g_sleeps, 1);
                g_sleeping.store(true, std::memory_order_seq_cst);
                g_published.wait(done, std::memory_order_seq_cst);
                g_sleeping.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        const std::uint64_t t1 = host_clock_monotonic_ns();
        while (done < published) {
            replay_block(g_blocks[done % kBlocks]);
            g_last_block.store(done, std::memory_order_relaxed);
            ++done;
            // The driver may record into write-combined memory (AMD's
            // Windows driver does: its command chunks are GPU memory the
            // CPU writes through). Write-combining buffers are per core and
            // x86's ordering does not cover them - a release store is a
            // plain store - so without a store fence the command processor
            // could end and submit this command buffer while some of the
            // block's commands still sat in this core's buffers, and the
            // GPU ran whatever was in memory: a device loss in the title's
            // first frames on a Radeon 8060S, every run, while in-place
            // recording (the same commands, one thread) never was.
            write_combine_fence();
            g_done.store(done, std::memory_order_release);
        }
        add_relaxed(g_replay_ns, host_clock_monotonic_ns() - t1);
    }
}

void ensure_rings() {
    if (g_blocks) return;
    g_blocks = new Block[kBlocks];      // never freed: the thread runs until exit
    g_packets = new DrawPacket[kPackets];
}

void ensure_started() {
    ensure_rings();
    if (g_mode == kModeInline) {
        static const bool said = [] {
            host_log("recorder: blocks replayed in place as they are published (BBHOST_RECORDER_INLINE)%s",
                     g_submit_on ? "; the stream begins, ends and hands over the command buffers" : "");
            return true;
        }();
        (void)said;
    }
    if (g_mode != kModeThread || g_thread_started) return;
    g_thread_started = true;
    std::thread(recorder_thread).detach();
    host_log("recorder: draws are recorded on their own thread (BBHOST_RECORDER=0 records them in place)");
    if (g_submit_on) {
        host_log("recorder: it also begins, ends and hands over the command buffers (BBHOST_STREAM_SUBMIT=0: the command processor does); "
                 "%s",
                 g_fence_all ? "a store fence at every hand-over (BBHOST_STREAM_FENCE=kick: at submissions only)"
                             : "a store fence at submissions and after in-place recording (BBHOST_STREAM_FENCE=kick)");
    } else {
        host_log("recorder: the command processor begins, ends and submits the command buffers and records everything but the draws in "
                 "place (BBHOST_STREAM_SUBMIT=0)");
    }
}

// Waits for `ready`, waking a recorder that went to sleep with work unseen
// (publish's wake is not fenced: see there).
template <class F> std::uint64_t wait_recorder(F ready) {
    if (ready()) return 0;
    const std::uint64_t t0 = host_clock_monotonic_ns();
    for (std::uint64_t spins = 0; !ready(); ++spins) {
        if ((spins & 1023) == 0 && g_sleeping.load(std::memory_order_seq_cst)) g_published.notify_one();
        if (spins < (1u << 16)) {
            cpu_relax();
        } else {
            std::this_thread::yield();
        }
    }
    return host_clock_monotonic_ns() - t0;
}

Block& current_block() { return g_mode == kModeInline ? g_inline_block : g_blocks[g_published.load(std::memory_order_relaxed) % kBlocks]; }

// The block ops go into, opened when there is none (under g.mu).
Block& open_block() {
    if (g_block_open) return current_block();
    ensure_started();
    if (g_mode == kModeThread) {
        const std::uint64_t published = g_published.load(std::memory_order_relaxed);
        if (published - g_done_seen >= kBlocks) g_done_seen = g_done.load(std::memory_order_acquire);
        if (published - g_done_seen >= kBlocks) {
            ++g_counts.full_waits;
            wait_recorder([published] { return published - g_done.load(std::memory_order_acquire) < kBlocks; });
            g_done_seen = g_done.load(std::memory_order_acquire);
        }
    }
    Block& b = current_block();
    b.used = 0;
    b.ops = 0;
    b.kick = false;
    g_block_open = true;
    return b;
}

// Hands the open block over (under g.mu).
void publish_block(bool kick) {
    if (!g_block_open) return;
    Block& b = current_block();
    g_block_open = false;
    b.cmd = g.cmd_;
    b.kick = kick;
    ++g_counts.blocks;
    if (g_mode == kModeInline) {
        replay_block(b);
        return;
    }
    // R2/R3: what this core wrote for the GPU and for the recorder - host-
    // visible memory a submission reads, commands recorded in place, the
    // presenter's command buffer - leaves its write-combining buffers before
    // the recorder can act on this block.
    if (g_fence_all || kick || g_inplace_dirty) write_combine_fence();
    g_inplace_dirty = false;
    // A release store, not a locked add: that would wait here for this
    // draw's stores (its params blocks among them) to drain.
    g_published.store(g_published.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    if (kick) {
        // A submission is waited for (stream_wait_submitted, the presenter):
        // the recorder must not sleep through it. Against its store of
        // g_sleeping and load of g_published, both sequentially consistent.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (g_sleeping.load(std::memory_order_relaxed)) g_published.notify_one();
    } else if (g_sleeping.load(std::memory_order_relaxed)) {
        // Unfenced: the recorder may go to sleep with this block unseen; the
        // next publish or the drain that needs it wakes it.
        g_published.notify_one();
    }
}

// Before an op: a packet holding commands goes first, so the op lands after
// them (under g.mu).
void before_op(void* site) {
    if (g.profile_gaps) profile_note_site_locked(site);  // BBHOST_GPU_PROFILE=2: an op is recorded by its caller too
    if (g_open && g_open->holds_commands()) g_open->publish();
}

// A record of `payload` bytes appended to the open block; its payload.
std::uint8_t* append_record(Op kind, std::size_t payload) {
    Block& b = open_block();
    const std::size_t bytes = sizeof(stream::Header) + round8(payload);
    const std::size_t words = bytes / 8;
    if (b.used + words > b.words.size()) b.words.resize(std::max<std::size_t>(b.words.size() * 2, b.used + words + 256));
    auto* h = reinterpret_cast<stream::Header*>(&b.words[b.used]);
    h->kind = kind;
    h->site = 0;
    h->bytes = static_cast<std::uint32_t>(bytes);
    b.used += words;
    ++b.ops;
    ++g_counts.ops[static_cast<std::size_t>(kind)];
    return reinterpret_cast<std::uint8_t*>(h + 1);
}

// After a non-kick op: a block grown past kBlockWords goes now.
void after_op() {
    if (g_block_open && current_block().used >= kBlockWords) publish_block(false);
}

template <class T> T* put_struct(std::uint8_t*& at) {
    T* t = reinterpret_cast<T*>(at);
    at += round8(sizeof(T));
    return t;
}
template <class T> void put_array(std::uint8_t*& at, const T* src, std::size_t n) {
    if (n) std::memcpy(at, src, n * sizeof(T));
    at += round8(n * sizeof(T));
}

// The command buffer, for recording in place: everything published first and
// replayed (under g.mu). `site` is who records, for the census and
// BBHOST_GPU_PROFILE=2's gaps.
VkCommandBuffer in_place(void* site) {
    if (g.profile_gaps) profile_note_site_locked(site);
    if (g_census_on) ++g_census[site];
    if (g_mode != kModeOff && g_blocks) {
        // An open draw whose packet already holds commands goes first.
        // Descriptor writes alone need not: they only have to land before
        // the draw's own binds, which come after them in the same packet.
        if (g_open && g_open->holds_commands()) g_open->publish();
        publish_block(false);
        recorder_drain();
        if (g_mode == kModeThread) g_inplace_dirty = true;
    }
    return g.cmd_;
}

bool warn_pnext(const char* what) {
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 4) host_log("stream: %s with a pNext chain; recorded in place", what);
    return true;
}

}  // namespace

bool stream_on() { return g_mode != kModeOff && g_submit_on; }
bool recorder_enabled() { return g_mode != kModeOff; }

void recorder_drain() {
    if (g_mode != kModeThread || !g_blocks) return;
    const std::uint64_t published = g_published.load(std::memory_order_relaxed);  // only the g.mu holder publishes
    ++g_counts.drains;
    if (g_done.load(std::memory_order_acquire) == published) return;
    ++g_counts.drain_waits;
    g_counts.drain_ns += wait_recorder([published] { return g_done.load(std::memory_order_acquire) == published; });
}

void stream_drain_published() {
    if (g_mode != kModeThread || !g_blocks) return;
    const std::uint64_t published = g_published.load(std::memory_order_acquire);
    wait_recorder([published] { return g_done.load(std::memory_order_acquire) >= published; });
}

void stream_wait_submitted(std::uint64_t serial) {
    if (stream_submitted_serial() > serial) return;
    // The submission's kEndSubmit was published with a kick; the recorder
    // replays it and bb-submit submits it. With 16 slots this rarely waits,
    // but a flush waits for the submission it just made. The same holds when
    // the command processor hands its submissions to bb-submit itself
    // (BBHOST_STREAM_SUBMIT=0, BBHOST_RECORDER=0): a fence waited for while
    // bb-submit was still submitting it was used on two threads at once
    // (vkQueueSubmit's fence must be externally synchronised), which the
    // validation layer reports and AMD's driver answered once with a
    // 10-second timeout in the self-test.
    ++g_counts.submitted_waits;
    const std::uint64_t t0 = host_clock_monotonic_ns();
    for (std::uint64_t spins = 0; stream_submitted_serial() <= serial; ++spins) {
        if ((spins & 1023) == 0 && g_mode == kModeThread && g_sleeping.load(std::memory_order_seq_cst)) g_published.notify_one();
        if (spins < 4096) {
            cpu_relax();
        } else if (spins < 8192) {
            std::this_thread::yield();
        } else {
            host_sleep_us(20);
        }
    }
    g_counts.submitted_wait_ns += host_clock_monotonic_ns() - t0;
}

std::string stream_last_op() {
    if (!g_blocks || g_mode == kModeOff) return {};
    const std::uint32_t k = g_last_kind.load(std::memory_order_relaxed);
    char buf[200];
    std::snprintf(buf, sizeof(buf), "the recorder's last op: %s in block %llu (%llu published, %llu replayed), last submission handed over %lld",
                  k < static_cast<std::uint32_t>(Op::kCount) ? stream::op_name(static_cast<Op>(k)) : "none",
                  static_cast<unsigned long long>(g_last_block.load()), static_cast<unsigned long long>(g_published.load()),
                  static_cast<unsigned long long>(g_done.load()), static_cast<long long>(g_last_serial.load()));
    return buf;
}

bool stream_test_set_mode(int mode, bool submit) {
    // Only between rounds of the self-test: nothing open, nothing recording.
    if (g_open || g_block_open || g.recording) return false;
    if (g_mode == kModeThread && g_blocks) stream_drain_published();
    g_mode = mode == 0 ? kModeOff : mode == 1 ? kModeInline : kModeThread;
    g_submit_on = submit && g_mode != kModeOff;
    ensure_started();
    return true;
}

StreamTestCounts stream_test_counts() {
    StreamTestCounts c;
    c.blocks = g_counts.blocks;
    c.draws = g_counts.ops[static_cast<std::size_t>(Op::kDrawState)];
    for (std::uint64_t n : g_counts.ops) c.ops += n;
    c.submits = g_counts.ops[static_cast<std::size_t>(Op::kEndSubmit)];
    c.drains = g_counts.drains;
    c.drain_waits = g_counts.drain_waits;
    return c;
}

VkCommandBuffer g_cmd() { return in_place(__builtin_return_address(0)); }

std::string recorder_report() {
    if (!g_blocks) return std::string();
    struct Last {
        Counts c;
        std::uint64_t spin = 0, replay = 0, sleeps = 0, endbegin_ns = 0, endbegin_n = 0, retire_ns = 0, retires = 0, fence_ns = 0, submits = 0;
        std::uint64_t flip = 0;
    };
    static Last last;
    const Counts& c = g_counts;
    const Counts& l = last.c;
    const auto ull = [](std::uint64_t v) { return static_cast<unsigned long long>(v); };
    const auto op = [&](Op k) { return c.ops[static_cast<std::size_t>(k)] - l.ops[static_cast<std::size_t>(k)]; };
    std::uint64_t all = 0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(Op::kCount); ++k) all += c.ops[k] - l.ops[k];
    const std::uint64_t fills = op(Op::kFill);
    const std::uint64_t barriers = op(Op::kBarrier) + op(Op::kTransferBarrier) + op(Op::kCopyOrderBarrier) + op(Op::kPassBarrier);
    const std::uint64_t copies = op(Op::kCopyBuffer) + op(Op::kCopyBufferToImage) + op(Op::kCopyImageToBuffer);
    const std::uint64_t passes = op(Op::kEndRendering) + op(Op::kBeginRendering);
    const std::uint64_t queries = op(Op::kResetQueries) + op(Op::kBeginQuery) + op(Op::kEndQuery) + op(Op::kCopyQueryResults) + op(Op::kTimestamp);
    const std::uint64_t submits = op(Op::kEndSubmit) + op(Op::kForeignSubmit) + op(Op::kBegin);
    const std::uint64_t draws = op(Op::kDrawState);
    const std::uint64_t spin = g_spin_ns.load(), replay = g_replay_ns.load(), sleeps = g_sleeps.load();
    const StreamStats& s = g_stream_stats;
    const std::uint64_t eb_ns = s.endbegin_ns.load(), eb_n = s.endbegin_n.load(), r_ns = s.retire_ns.load(), r_n = s.retires.load(),
                        f_ns = s.fence_ns.load(), subs = s.submits.load();
    const std::uint64_t flip = hle_video_flip_count();
    char buf[1400];
    int n = std::snprintf(
        buf, sizeof(buf),
        "recorder: %llu draws replayed; %llu drains, %llu of them waited; %llu waits for a free packet. This window: %llu blocks "
        "(%llu draws), %llu ops (fill %llu, barrier %llu, copy %llu, pass %llu, query %llu, begin/submit %llu, other %llu); drains %llu "
        "(%llu waited, %llu us); submitted-waits %llu (%llu us); waits for a free block %llu / packet %llu; recorder spin %llu ms, "
        "replay %llu ms, sleeps %llu%s",
        ull(g_pk_done.load()), ull(c.drains), ull(c.drain_waits), ull(c.packet_waits), ull(c.blocks - l.blocks), ull(draws), ull(all),
        ull(fills), ull(barriers), ull(copies), ull(passes), ull(queries), ull(submits),
        ull(all - fills - barriers - copies - passes - queries - submits - draws), ull(c.drains - l.drains), ull(c.drain_waits - l.drain_waits),
        ull((c.drain_ns - l.drain_ns) / 1000), ull(c.submitted_waits - l.submitted_waits), ull((c.submitted_wait_ns - l.submitted_wait_ns) / 1000),
        ull(c.full_waits - l.full_waits), ull(c.packet_waits - l.packet_waits), ull((spin - last.spin) / 1000000),
        ull((replay - last.replay) / 1000000), ull(sleeps - last.sleeps),
        g_mode == kModeOff ? " (recorder off)" : g_mode == kModeInline ? " (inline)" : stream_on() ? "" : " (BBHOST_STREAM_SUBMIT=0)");
    // What a submission costs on its way, wherever it runs.
    const std::uint64_t ns = subs - last.submits;
    if (n > 0 && static_cast<std::size_t>(n) < sizeof(buf)) {
        n += std::snprintf(buf + n, sizeof(buf) - static_cast<std::size_t>(n),
                           "\n[bbhost]   submit: %llu, end+begin %llu us (%.1f us each), retire %llu us (%llu retired; fence %llu us)", ull(ns),
                           ull((eb_ns - last.endbegin_ns) / 1000),
                           eb_n > last.endbegin_n ? static_cast<double>(eb_ns - last.endbegin_ns) / 1000.0 / static_cast<double>(eb_n - last.endbegin_n) : 0.0,
                           ull((r_ns - last.retire_ns) / 1000), ull(r_n - last.retires), ull((f_ns - last.fence_ns) / 1000));
    }
    std::string out = buf;
    if (g_census_on && !g_census.empty()) {
        // The in-place recording sites, the top twelve, as offsets in
        // bbhost's image (tools/gpu_gaps.py --census names them).
        std::vector<std::pair<void*, std::uint64_t>> v(g_census.begin(), g_census.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        const double flips = flip > last.flip ? static_cast<double>(flip - last.flip) : 1.0;
        out += "\n[bbhost]   cmd census:";
        for (std::size_t i = 0; i < v.size() && i < 12; ++i) {
            char e[64];
            std::snprintf(e, sizeof(e), " %llx=%.1f/flip", ull(reinterpret_cast<std::uintptr_t>(v[i].first) - host_image_base()),
                          static_cast<double>(v[i].second) / flips);
            out += e;
        }
        g_census.clear();
    }
    last.c = c;
    last.spin = spin;
    last.replay = replay;
    last.sleeps = sleeps;
    last.endbegin_ns = eb_ns;
    last.endbegin_n = eb_n;
    last.retire_ns = r_ns;
    last.retires = r_n;
    last.fence_ns = f_ns;
    last.submits = subs;
    last.flip = flip;
    return out;
}


// ---- Rec: the command processor's vkCmd* calls, as ops of the stream ----
//
// Each mirrors its vkCmd* call without the command buffer. With the stream
// on it appends a record (the arrays copied in); otherwise it records in
// place on g_cmd()'s command buffer, so one call site serves both.

#define REC_SITE __builtin_return_address(0)

Rec& rec() {
    static Rec r;
    return r;
}

void Rec::begin(VkCommandBuffer cmd) {
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kBegin, sizeof(stream::Begin));
    put_struct<stream::Begin>(at)->cmd = cmd;
    after_op();
}

void Rec::end_submit(const stream::EndSubmit& s) {
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kEndSubmit, sizeof(stream::EndSubmit));
    *put_struct<stream::EndSubmit>(at) = s;
    publish_block(true);
}

std::uint64_t Rec::foreign_submit(VkCommandBuffer cmd, VkSemaphore wait, VkPipelineStageFlags wait_stage, VkSemaphore signal, VkFence fence) {
    static std::uint64_t next = 0;  // under g.mu
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kForeignSubmit, sizeof(stream::ForeignSubmit));
    stream::ForeignSubmit* r = put_struct<stream::ForeignSubmit>(at);
    *r = stream::ForeignSubmit{};
    r->cmd = cmd;
    r->wait = wait;
    r->wait_stage = wait_stage;
    r->signal = signal;
    r->fence = fence;
    r->id = ++next;
    const std::uint64_t id = r->id;
    publish_block(true);  // (inline: replayed here, and the record may be reused after)
    return id;
}

void Rec::pipeline_barrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags dep, std::uint32_t nmem,
                           const VkMemoryBarrier* mem, std::uint32_t nbuf, const VkBufferMemoryBarrier* buf, std::uint32_t nimg,
                           const VkImageMemoryBarrier* img) {
    bool chained = false;
    for (std::uint32_t k = 0; k < nmem; ++k) chained |= mem[k].pNext != nullptr;
    for (std::uint32_t k = 0; k < nbuf; ++k) chained |= buf[k].pNext != nullptr;
    for (std::uint32_t k = 0; k < nimg; ++k) chained |= img[k].pNext != nullptr;
    if (!stream_on() || (chained && warn_pnext("a barrier"))) {
        vkCmdPipelineBarrier(in_place(REC_SITE), src, dst, dep, nmem, mem, nbuf, buf, nimg, img);
        return;
    }
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kBarrier, round8(sizeof(stream::Barrier)) + round8(nmem * sizeof(VkMemoryBarrier)) +
                                                       round8(nbuf * sizeof(VkBufferMemoryBarrier)) + round8(nimg * sizeof(VkImageMemoryBarrier)));
    *put_struct<stream::Barrier>(at) = stream::Barrier{src, dst, dep, nmem, nbuf, nimg};
    put_array(at, mem, nmem);
    put_array(at, buf, nbuf);
    put_array(at, img, nimg);
    after_op();
}

namespace {
void flag_op(Op kind, std::uint32_t value, void* site) {
    before_op(site);
    std::uint8_t* at = append_record(kind, sizeof(stream::Flag));
    *put_struct<stream::Flag>(at) = stream::Flag{value, 0};
    after_op();
}
}  // namespace

void Rec::transfer_barrier(bool begin) {
    if (!stream_on()) return record_transfer_barrier(in_place(REC_SITE), begin);
    flag_op(Op::kTransferBarrier, begin ? 1 : 0, REC_SITE);
}

void Rec::copy_order_barrier() {
    if (!stream_on()) return record_copy_order_barrier(in_place(REC_SITE));
    flag_op(Op::kCopyOrderBarrier, 0, REC_SITE);
}

void Rec::pass_barrier() {
    if (!stream_on()) return record_pass_barrier(in_place(REC_SITE));
    flag_op(Op::kPassBarrier, 0, REC_SITE);
}

void Rec::end_rendering(bool barrier) {
    if (!stream_on()) return record_end_rendering(in_place(REC_SITE), barrier);
    flag_op(Op::kEndRendering, barrier ? 1 : 0, REC_SITE);
}

void Rec::begin_rendering(const VkRenderingInfo& ri) {
    bool chained = ri.pNext != nullptr;
    for (std::uint32_t k = 0; k < ri.colorAttachmentCount; ++k) chained |= ri.pColorAttachments[k].pNext != nullptr;
    if (ri.pDepthAttachment) chained |= ri.pDepthAttachment->pNext != nullptr;
    if (ri.pStencilAttachment) chained |= ri.pStencilAttachment->pNext != nullptr;
    if (!stream_on() || (chained && warn_pnext("a rendering begin"))) {
        vkCmdBeginRendering(in_place(REC_SITE), &ri);
        return;
    }
    const std::uint32_t natt = ri.colorAttachmentCount + (ri.pDepthAttachment ? 1 : 0) + (ri.pStencilAttachment ? 1 : 0);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kBeginRendering, round8(sizeof(stream::BeginRendering)) + round8(natt * sizeof(VkRenderingAttachmentInfo)));
    stream::BeginRendering* r = put_struct<stream::BeginRendering>(at);
    *r = stream::BeginRendering{};
    r->flags = ri.flags;
    r->layers = ri.layerCount;
    r->view_mask = ri.viewMask;
    r->ncolor = ri.colorAttachmentCount;
    r->area = ri.renderArea;
    r->has_depth = ri.pDepthAttachment ? 1 : 0;
    r->has_stencil = ri.pStencilAttachment ? 1 : 0;
    // The attachments one after another: colours, depth, stencil.
    if (ri.colorAttachmentCount) std::memcpy(at, ri.pColorAttachments, ri.colorAttachmentCount * sizeof(VkRenderingAttachmentInfo));
    auto* next = reinterpret_cast<VkRenderingAttachmentInfo*>(at) + ri.colorAttachmentCount;
    if (ri.pDepthAttachment) *next++ = *ri.pDepthAttachment;
    if (ri.pStencilAttachment) *next++ = *ri.pStencilAttachment;
    after_op();
}

void Rec::fill_buffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data) {
    if (!stream_on()) return vkCmdFillBuffer(in_place(REC_SITE), buffer, offset, size, data);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kFill, sizeof(stream::Fill));
    *put_struct<stream::Fill>(at) = stream::Fill{buffer, offset, size, data, 0};
    after_op();
}

void Rec::copy_buffer(VkBuffer src, VkBuffer dst, std::uint32_t n, const VkBufferCopy* regions) {
    if (!stream_on()) return vkCmdCopyBuffer(in_place(REC_SITE), src, dst, n, regions);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kCopyBuffer, round8(sizeof(stream::CopyBuffer)) + round8(n * sizeof(VkBufferCopy)));
    *put_struct<stream::CopyBuffer>(at) = stream::CopyBuffer{src, dst, n, 0};
    put_array(at, regions, n);
    after_op();
}

void Rec::copy_buffer_to_image(VkBuffer buffer, VkImage image, VkImageLayout layout, std::uint32_t n, const VkBufferImageCopy* regions) {
    if (!stream_on()) return vkCmdCopyBufferToImage(in_place(REC_SITE), buffer, image, layout, n, regions);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kCopyBufferToImage, round8(sizeof(stream::CopyBufferToImage)) + round8(n * sizeof(VkBufferImageCopy)));
    *put_struct<stream::CopyBufferToImage>(at) = stream::CopyBufferToImage{buffer, image, layout, n};
    put_array(at, regions, n);
    after_op();
}

void Rec::copy_image_to_buffer(VkImage image, VkImageLayout layout, VkBuffer buffer, std::uint32_t n, const VkBufferImageCopy* regions) {
    if (!stream_on()) return vkCmdCopyImageToBuffer(in_place(REC_SITE), image, layout, buffer, n, regions);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kCopyImageToBuffer, round8(sizeof(stream::CopyImageToBuffer)) + round8(n * sizeof(VkBufferImageCopy)));
    *put_struct<stream::CopyImageToBuffer>(at) = stream::CopyImageToBuffer{image, buffer, layout, n};
    put_array(at, regions, n);
    after_op();
}

void Rec::clear_color_image(VkImage image, VkImageLayout layout, const VkClearColorValue& color, std::uint32_t n,
                            const VkImageSubresourceRange* ranges) {
    if (!stream_on()) return vkCmdClearColorImage(in_place(REC_SITE), image, layout, &color, n, ranges);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kClearColor, round8(sizeof(stream::ClearColor)) + round8(n * sizeof(VkImageSubresourceRange)));
    *put_struct<stream::ClearColor>(at) = stream::ClearColor{image, layout, n, color};
    put_array(at, ranges, n);
    after_op();
}

void Rec::reset_query_pool(VkQueryPool pool, std::uint32_t first, std::uint32_t count) {
    if (!stream_on()) return vkCmdResetQueryPool(in_place(REC_SITE), pool, first, count);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kResetQueries, sizeof(stream::Queries));
    *put_struct<stream::Queries>(at) = stream::Queries{pool, first, count, 0, 0};
    after_op();
}

void Rec::begin_query(VkQueryPool pool, std::uint32_t query, VkQueryControlFlags flags) {
    if (!stream_on()) return vkCmdBeginQuery(in_place(REC_SITE), pool, query, flags);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kBeginQuery, sizeof(stream::Queries));
    *put_struct<stream::Queries>(at) = stream::Queries{pool, query, 1, flags, 0};
    after_op();
}

void Rec::end_query(VkQueryPool pool, std::uint32_t query) {
    if (!stream_on()) return vkCmdEndQuery(in_place(REC_SITE), pool, query);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kEndQuery, sizeof(stream::Queries));
    *put_struct<stream::Queries>(at) = stream::Queries{pool, query, 1, 0, 0};
    after_op();
}

void Rec::copy_query_results(VkQueryPool pool, std::uint32_t first, std::uint32_t count, VkBuffer dst, VkDeviceSize offset, VkDeviceSize stride,
                             VkQueryResultFlags flags) {
    if (!stream_on()) return vkCmdCopyQueryPoolResults(in_place(REC_SITE), pool, first, count, dst, offset, stride, flags);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kCopyQueryResults, sizeof(stream::CopyQueryResults));
    *put_struct<stream::CopyQueryResults>(at) = stream::CopyQueryResults{pool, first, count, dst, offset, stride, flags, 0};
    after_op();
}

void Rec::write_timestamp(VkPipelineStageFlagBits stage, VkQueryPool pool, std::uint32_t query) {
    if (!stream_on()) return vkCmdWriteTimestamp(in_place(REC_SITE), stage, pool, query);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kTimestamp, sizeof(stream::Timestamp));
    *put_struct<stream::Timestamp>(at) = stream::Timestamp{pool, stage, query};
    after_op();
}

void Rec::bind_pipeline(VkPipelineBindPoint bind_point, VkPipeline pipeline) {
    if (!stream_on()) return vkCmdBindPipeline(in_place(REC_SITE), bind_point, pipeline);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kBindPipeline, sizeof(stream::BindPipeline));
    *put_struct<stream::BindPipeline>(at) = stream::BindPipeline{pipeline, bind_point, 0};
    after_op();
}

void Rec::bind_sets(VkPipelineBindPoint bind_point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t n, const VkDescriptorSet* sets,
                    std::uint32_t ndynamic, const std::uint32_t* dynamic) {
    if (!stream_on()) return vkCmdBindDescriptorSets(in_place(REC_SITE), bind_point, layout, first, n, sets, ndynamic, dynamic);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kBindSets, round8(sizeof(stream::BindSets)) + round8(n * sizeof(VkDescriptorSet)) +
                                                        round8(ndynamic * sizeof(std::uint32_t)));
    *put_struct<stream::BindSets>(at) = stream::BindSets{layout, bind_point, first, n, ndynamic};
    put_array(at, sets, n);
    put_array(at, dynamic, ndynamic);
    after_op();
}

void Rec::push_constants(VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* data) {
    if (!stream_on()) return vkCmdPushConstants(in_place(REC_SITE), layout, stages, offset, size, data);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kPushConstants, round8(sizeof(stream::PushConstants)) + round8(size));
    *put_struct<stream::PushConstants>(at) = stream::PushConstants{layout, stages, offset, size, 0};
    put_array(at, static_cast<const std::uint8_t*>(data), size);
    after_op();
}

void Rec::update_sets(std::uint32_t n, const VkWriteDescriptorSet* writes) {
    if (!n) return;
    bool chained = false, texel = false;
    std::uint32_t nbuf = 0, nimg = 0;
    for (std::uint32_t k = 0; k < n; ++k) {
        chained |= writes[k].pNext != nullptr;
        texel |= writes[k].pTexelBufferView != nullptr;
        if (writes[k].pBufferInfo) nbuf += writes[k].descriptorCount;
        if (writes[k].pImageInfo) nimg += writes[k].descriptorCount;
    }
    if (!stream_on() || texel || (chained && warn_pnext("a descriptor write"))) {
        vkUpdateDescriptorSets(g.device, n, writes, 0, nullptr);
        return;
    }
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kUpdateSets, round8(sizeof(stream::UpdateSets)) + round8(n * sizeof(VkWriteDescriptorSet)) +
                                                          2 * round8(n * sizeof(std::int32_t)) + round8(nbuf * sizeof(VkDescriptorBufferInfo)) +
                                                          round8(nimg * sizeof(VkDescriptorImageInfo)));
    *put_struct<stream::UpdateSets>(at) = stream::UpdateSets{n, nbuf, nimg, 0};
    auto* ws = reinterpret_cast<VkWriteDescriptorSet*>(at);
    put_array(at, writes, n);
    auto* fb = reinterpret_cast<std::int32_t*>(at);
    at += round8(n * sizeof(std::int32_t));
    auto* fi = reinterpret_cast<std::int32_t*>(at);
    at += round8(n * sizeof(std::int32_t));
    auto* bufs = reinterpret_cast<VkDescriptorBufferInfo*>(at);
    at += round8(nbuf * sizeof(VkDescriptorBufferInfo));
    auto* imgs = reinterpret_cast<VkDescriptorImageInfo*>(at);
    std::uint32_t b = 0, im = 0;
    for (std::uint32_t k = 0; k < n; ++k) {
        fb[k] = fi[k] = -1;
        if (writes[k].pBufferInfo) {
            fb[k] = static_cast<std::int32_t>(b);
            std::memcpy(bufs + b, writes[k].pBufferInfo, writes[k].descriptorCount * sizeof(VkDescriptorBufferInfo));
            b += writes[k].descriptorCount;
        }
        if (writes[k].pImageInfo) {
            fi[k] = static_cast<std::int32_t>(im);
            std::memcpy(imgs + im, writes[k].pImageInfo, writes[k].descriptorCount * sizeof(VkDescriptorImageInfo));
            im += writes[k].descriptorCount;
        }
        ws[k].pBufferInfo = nullptr;
        ws[k].pImageInfo = nullptr;
    }
    after_op();
}

void Rec::dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (!stream_on()) return vkCmdDispatch(in_place(REC_SITE), x, y, z);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kDispatch, sizeof(stream::Dispatch));
    *put_struct<stream::Dispatch>(at) = stream::Dispatch{x, y, z, 0};
    after_op();
}

void Rec::marker(std::uint32_t value) {
    if (!stream_on()) return gpu_write_markers(in_place(REC_SITE), value);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kMarker, sizeof(stream::Marker));
    *put_struct<stream::Marker>(at) = stream::Marker{value, 0};
    after_op();
}

void Rec::clear_depth_stencil_image(VkImage image, VkImageLayout layout, const VkClearDepthStencilValue& value, std::uint32_t n,
                                    const VkImageSubresourceRange* ranges) {
    if (!stream_on()) return vkCmdClearDepthStencilImage(in_place(REC_SITE), image, layout, &value, n, ranges);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kClearDepth, round8(sizeof(stream::ClearDepth)) + round8(n * sizeof(VkImageSubresourceRange)));
    *put_struct<stream::ClearDepth>(at) = stream::ClearDepth{image, layout, n, value};
    put_array(at, ranges, n);
    after_op();
}

void Rec::copy_image(VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, std::uint32_t n, const VkImageCopy* regions) {
    if (!stream_on()) return vkCmdCopyImage(in_place(REC_SITE), src, src_layout, dst, dst_layout, n, regions);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kCopyImage, round8(sizeof(stream::CopyImage)) + round8(n * sizeof(VkImageCopy)));
    *put_struct<stream::CopyImage>(at) = stream::CopyImage{src, dst, src_layout, dst_layout, n, 0};
    put_array(at, regions, n);
    after_op();
}

void Rec::dispatch_indirect(VkBuffer buffer, VkDeviceSize offset) {
    if (!stream_on()) return vkCmdDispatchIndirect(in_place(REC_SITE), buffer, offset);
    before_op(REC_SITE);
    std::uint8_t* at = append_record(Op::kDispatchIndirect, sizeof(stream::DispatchIndirect));
    *put_struct<stream::DispatchIndirect>(at) = stream::DispatchIndirect{buffer, offset};
    after_op();
}

#undef REC_SITE

}  // namespace gpu

std::string host_gpu_recorder_report() {
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    std::string r = gpu::recorder_report();
    if (gpu::set_cache_on()) r += "; " + gpu::set_cache_report();
    if (const std::string cv = gpu::copy_versions_report(); !cv.empty()) r += "; " + cv;
    if (const std::string pb = gpu::render_barriers_report(); !pb.empty()) r += "; " + pb;
    return r;
}

namespace gpu {

// ---- DrawCmds: a draw's commands, packed or recorded in place ----

DrawCmds::DrawCmds(bool deferred) { start(deferred); }

void DrawCmds::start(bool deferred) {
    if (packet_) {
        if (!deferred) publish();  // started early, and the draw records in place after all
        return;
    }
    if (!deferred || g_mode == kModeOff) return;
    ensure_started();
    // One packet open at a time: one another draw left open goes first.
    if (g_open && g_open != this) g_open->publish();
    if (g_pk_taken - g_pk_done_seen >= kPackets) g_pk_done_seen = g_pk_done.load(std::memory_order_acquire);
    if (g_pk_taken - g_pk_done_seen >= kPackets) {
        // Every packet taken before this one is in a published block.
        ++g_counts.packet_waits;
        const std::uint64_t taken = g_pk_taken;
        wait_recorder([taken] { return taken - g_pk_done.load(std::memory_order_acquire) < kPackets; });
        g_pk_done_seen = g_pk_done.load(std::memory_order_acquire);
    }
    DrawPacket& p = g_packets[g_pk_taken % kPackets];
    p.reset();
    p.number = g_pk_taken++;
    packet_ = &p;
    g_open = this;
}

DrawCmds* DrawCmds::open() { return g_open; }

bool DrawCmds::holds_commands() const {
    if (!packet_) return false;
    const DrawPacket& p = *static_cast<const DrawPacket*>(packet_);
    return !before_binds(p) || p.end_pass || p.pass_barrier || p.begin_pass || p.xfer_begin || p.xfer_end || p.ncopies || p.nregions;
}

DrawCmds::~DrawCmds() { publish(); }

void DrawCmds::publish() {
    if (!packet_) return;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    packet_ = nullptr;
    if (g_open == this) g_open = nullptr;
    // The packet's place in the stream: after every op appended before it
    // (they were appended while it held no commands, or it went first).
    std::uint8_t* at = append_record(Op::kDrawState, sizeof(stream::DrawState));
    put_struct<stream::DrawState>(at)->packet = p.number;
    publish_block(false);
}

void DrawCmds::update_sets(const VkWriteDescriptorSet* writes, std::size_t n) {
    if (!n) return;
    if (!packet_) {
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(n), writes, 0, nullptr);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    for (std::size_t i = 0; i < n; ++i) {
        VkWriteDescriptorSet w = writes[i];
        std::int32_t b = -1, im = -1;
        if (w.pBufferInfo) {
            b = static_cast<std::int32_t>(p.buffers.size());
            p.buffers.insert(p.buffers.end(), w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
            w.pBufferInfo = nullptr;
        }
        if (w.pImageInfo) {
            im = static_cast<std::int32_t>(p.images.size());
            p.images.insert(p.images.end(), w.pImageInfo, w.pImageInfo + w.descriptorCount);
            w.pImageInfo = nullptr;
        }
        p.writes.push_back(w);
        p.write_buffer.push_back(b);
        p.write_image.push_back(im);
    }
}

void DrawCmds::take_sets(std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& images,
                         const std::vector<VkDescriptorBufferInfo>& buffers, const VkDescriptorBufferInfo* extra, std::size_t n_extra) {
    if (writes.empty()) return;
    if (!packet_) {
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    const VkDescriptorBufferInfo* const b0 = buffers.data();
    const VkDescriptorBufferInfo* const b1 = b0 + buffers.size();
    const VkDescriptorImageInfo* const i0 = images.data();
    const VkDescriptorImageInfo* const i1 = i0 + images.size();
    // Taken only when every info is one of these and nothing came before
    // (the packet's writes are applied in the order they came).
    bool takeable = n_extra <= 2 && p.writes.empty() && p.taken_writes.empty();
    for (std::size_t k = 0; takeable && k < writes.size(); ++k) {
        const VkWriteDescriptorSet& w = writes[k];
        if (w.pBufferInfo) {
            takeable = (w.pBufferInfo >= b0 && w.pBufferInfo + w.descriptorCount <= b1) ||
                       (w.pBufferInfo >= extra && w.pBufferInfo + w.descriptorCount <= extra + n_extra);
        }
        if (takeable && w.pImageInfo) takeable = w.pImageInfo >= i0 && w.pImageInfo + w.descriptorCount <= i1;
        if (takeable && w.pTexelBufferView) takeable = false;
    }
    if (!takeable) {
        update_sets(writes.data(), writes.size());
        return;
    }
    p.taken_buffers.assign(buffers.begin(), buffers.end());
    for (std::size_t k = 0; k < n_extra; ++k) p.extra[k] = extra[k];
    for (VkWriteDescriptorSet& w : writes) {
        if (!w.pBufferInfo) continue;
        if (w.pBufferInfo >= b0 && w.pBufferInfo < b1) {
            w.pBufferInfo = p.taken_buffers.data() + (w.pBufferInfo - b0);
        } else {
            w.pBufferInfo = p.extra + (w.pBufferInfo - extra);
        }
    }
    p.taken_writes.swap(writes);  // image infos keep their buffer, and the writes' pointers into it
    p.taken_images.swap(images);
}

void DrawCmds::bind_pipeline(VkPipeline pipeline) {
    if (!packet_) {
        vkCmdBindPipeline(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        return;
    }
    static_cast<DrawPacket*>(packet_)->pipeline = pipeline;
}

void DrawCmds::library_state(const DrawLibraryState& s, bool has_depth_bounds, std::uint32_t fields) {
    if (!fields) return;
    if (!packet_) {
        record_library_state(g_cmd(), s, has_depth_bounds, fields);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.library_fields = p.library ? p.library_fields | fields : fields;
    p.library = true;
    p.lib = s;
    p.has_depth_bounds = has_depth_bounds;
}

void DrawCmds::bind_sets(VkPipelineLayout layout, const VkDescriptorSet sets[2], const std::uint32_t* dynamic, bool bindless) {
    if (!packet_) {
        vkCmdBindDescriptorSets(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, dynamic ? 2 : 0, dynamic);
        if (bindless) vkCmdBindDescriptorSets(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 3, 1, &g.bindless_set, 0, nullptr);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.bind_sets = true;
    p.bindless = bindless;
    p.layout = layout;
    p.sets[0] = sets[0];
    p.sets[1] = sets[1];
    p.dynamic_count = dynamic ? 2 : 0;
    if (dynamic) {
        p.dynamic[0] = dynamic[0];
        p.dynamic[1] = dynamic[1];
    }
}

void DrawCmds::push_buffers(VkPipelineLayout layout, const std::uint32_t* bindings, const VkDescriptorBufferInfo* infos, std::uint32_t n) {
    if (!n) return;
    if (!packet_) {
        VkWriteDescriptorSet writes[2 * gcn::kMaxBuffers];
        for (std::uint32_t k = 0; k < n; ++k) {
            writes[k] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[k].dstBinding = bindings[k];
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &infos[k];
        }
        g.cmd_push_descriptor_set(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, n, writes);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.layout = layout;
    p.pushes = n;
    std::memcpy(p.push_binding, bindings, n * sizeof(std::uint32_t));
    std::memcpy(p.push_infos, infos, n * sizeof(VkDescriptorBufferInfo));
}

void DrawCmds::viewport(const VkViewport& vp) {
    if (!packet_) {
        vkCmdSetViewport(g_cmd(), 0, 1, &vp);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.viewport = true;
    p.vp = vp;
}

void DrawCmds::scissor(const VkRect2D& sc) {
    if (!packet_) {
        vkCmdSetScissor(g_cmd(), 0, 1, &sc);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.scissor = true;
    p.sc = sc;
}

void DrawCmds::depth_bounds(float lo, float hi) {
    if (!packet_) {
        vkCmdSetDepthBounds(g_cmd(), lo, hi);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.bounds = true;
    p.depth_bounds[0] = lo;
    p.depth_bounds[1] = hi;
}

void DrawCmds::stencil(const std::uint32_t words[6]) {
    if (!packet_) {
        record_stencil_words(g_cmd(), words);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.stencil = true;
    std::memcpy(p.stencil_words, words, sizeof(p.stencil_words));
}

void DrawCmds::blend_constants(const float c[4]) {
    if (!packet_) {
        vkCmdSetBlendConstants(g_cmd(), c);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.blend = true;
    std::memcpy(p.blend_const, c, sizeof(p.blend_const));
}

void DrawCmds::index_buffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    if (!packet_) {
        vkCmdBindIndexBuffer(g_cmd(), buffer, offset, type);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.index = true;
    p.index_buffer = buffer;
    p.index_offset = offset;
    p.index_type = type;
}

void DrawCmds::marker(std::uint32_t value) {
    if (!packet_) {
        gpu_write_markers(g_cmd(), value);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.marker = true;
    p.marker_value = value;
}

void DrawCmds::vertex_buffers(std::uint32_t n, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    if (!n) return;
    if (!packet_ || n > kMaxVertexBindings) {
        if (packet_) {
            publish();  // too many to pack: this draw records the rest in place
        }
        vkCmdBindVertexBuffers(g_cmd(), 0, n, buffers, offsets);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.vertex_buffers = n;
    std::memcpy(p.vb, buffers, n * sizeof(VkBuffer));
    std::memcpy(p.vb_offset, offsets, n * sizeof(VkDeviceSize));
}

void DrawCmds::draw(const DrawCall& call) {
    if (!packet_) {
        record_draw_call(g_cmd(), call);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.draw = true;
    p.call = call;
}

bool DrawCmds::end_rendering(bool barrier) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.end_pass || p.nregions) return false;
    p.end_pass = true;
    p.end_pass_barrier = barrier;
    return true;
}

bool DrawCmds::pass_barrier() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.pass_barrier || p.xfer_begin || p.xfer_end || p.ncopies || p.nregions) return false;
    p.pass_barrier = true;
    return true;
}

// The transfer batch's parts go between the pass end and the pass begin, so
// a packet takes them only while nothing after that point is in it.
bool DrawCmds::transfer_begin() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_begin || p.xfer_end || p.ncopies || p.nregions) return false;
    p.xfer_begin = true;
    return true;
}

bool DrawCmds::copy_buffer(VkBuffer src, VkBuffer dst, const VkBufferCopy& region) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_end || p.nregions || p.ncopies >= DrawPacket::kCopies) return false;
    p.copy_src[p.ncopies] = src;
    p.copy_dst[p.ncopies] = dst;
    p.copies[p.ncopies] = region;
    ++p.ncopies;
    return true;
}

bool DrawCmds::copy_order() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_end || p.nregions || p.ncopies >= DrawPacket::kCopies) return false;
    p.copy_orders |= 1u << p.ncopies;
    return true;
}

bool DrawCmds::transfer_end() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_end || p.nregions) return false;
    p.xfer_end = true;
    return true;
}

bool DrawCmds::copy_region(VkImage src, VkImage dst, std::uint32_t width, std::uint32_t height, bool dst_initialised) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    // After a transfer batch's start its end must come first (replay order).
    if (!before_binds(p) || p.begin_pass || (p.xfer_begin && !p.xfer_end) || p.nregions >= DrawPacket::kRegionCopies) return false;
    p.regions[p.nregions++] = {src, dst, width, height, dst_initialised};
    return true;
}

bool DrawCmds::begin_rendering(const VkRenderingInfo& ri) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || ri.colorAttachmentCount > 8 || ri.pNext) return false;
    p.begin_pass = true;
    p.rendering = ri;
    for (std::uint32_t k = 0; k < ri.colorAttachmentCount; ++k) p.colors[k] = ri.pColorAttachments[k];
    p.has_depth = ri.pDepthAttachment != nullptr;
    if (p.has_depth) p.depth_att = *ri.pDepthAttachment;
    p.has_stencil = ri.pStencilAttachment != nullptr;
    if (p.has_stencil) p.stencil_att = *ri.pStencilAttachment;
    return true;
}

// ---- the commands themselves, shared by both paths ----

std::uint32_t draw_library_changes(const DrawLibraryState& a, const DrawLibraryState& b) {
    std::uint32_t fields = 0;
    if (a.cull != b.cull) fields |= kLibraryCull;
    if (a.front != b.front) fields |= kLibraryFront;
    if (a.depth_test != b.depth_test) fields |= kLibraryDepthTest;
    if (a.depth_write != b.depth_write) fields |= kLibraryDepthWrite;
    if (a.depth_compare != b.depth_compare) fields |= kLibraryDepthCompare;
    if (a.bounds_test != b.bounds_test) fields |= kLibraryBoundsTest;
    if (a.stencil_test != b.stencil_test) fields |= kLibraryStencilTest;
    auto ops_differ = [](const VkStencilOpState& x, const VkStencilOpState& y) {
        return x.failOp != y.failOp || x.passOp != y.passOp || x.depthFailOp != y.depthFailOp || x.compareOp != y.compareOp;
    };
    if (ops_differ(a.front_ops, b.front_ops)) fields |= kLibraryFrontOps;
    if (ops_differ(a.back_ops, b.back_ops)) fields |= kLibraryBackOps;
    if (std::memcmp(&a.bias_constant, &b.bias_constant, sizeof(float)) ||
        std::memcmp(&a.bias_clamp, &b.bias_clamp, sizeof(float)) ||
        std::memcmp(&a.bias_slope, &b.bias_slope, sizeof(float))) fields |= kLibraryBias;
    if (a.depth_clamp != b.depth_clamp) fields |= kLibraryDepthClamp;
    return fields;
}

void record_library_state(VkCommandBuffer cmd, const DrawLibraryState& s, bool has_depth_bounds, std::uint32_t fields) {
    if (fields & kLibraryCull) vkCmdSetCullMode(cmd, s.cull);
    if (fields & kLibraryFront) vkCmdSetFrontFace(cmd, s.front);
    if (fields & kLibraryDepthTest) vkCmdSetDepthTestEnable(cmd, s.depth_test);
    if (fields & kLibraryDepthWrite) vkCmdSetDepthWriteEnable(cmd, s.depth_write);
    if (fields & kLibraryDepthCompare) vkCmdSetDepthCompareOp(cmd, s.depth_compare);
    if (has_depth_bounds && (fields & kLibraryBoundsTest)) vkCmdSetDepthBoundsTestEnable(cmd, s.bounds_test);
    if (fields & kLibraryStencilTest) vkCmdSetStencilTestEnable(cmd, s.stencil_test);
    if (fields & kLibraryFrontOps)
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, s.front_ops.failOp, s.front_ops.passOp, s.front_ops.depthFailOp, s.front_ops.compareOp);
    if (fields & kLibraryBackOps)
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, s.back_ops.failOp, s.back_ops.passOp, s.back_ops.depthFailOp, s.back_ops.compareOp);
    if (fields & kLibraryBias) vkCmdSetDepthBias(cmd, s.bias_constant, s.bias_clamp, s.bias_slope);
    if (g.dynamic_depth_clamp && (fields & kLibraryDepthClamp)) g.cmd_set_depth_clamp_enable(cmd, s.depth_clamp);
}

void record_stencil_words(VkCommandBuffer cmd, const std::uint32_t w[6]) {
    vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, w[0]);
    vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, w[1]);
    vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, w[2]);
    vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, w[3]);
    vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, w[4]);
    vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, w[5]);
}

// BBHOST_RENDER_MIN=1: a test client. Everything is recorded but the draws
// and dispatches themselves, so the game sees its labels, flips and
// readbacks and the GPU does next to nothing - several instances on one
// GPU for the online tests. The screen is whatever the clears leave.
const bool g_render_min = [] {
    const char* e = std::getenv("BBHOST_RENDER_MIN");
    return e && e[0] == '1';
}();

void record_draw_call(VkCommandBuffer cmd, const DrawCall& c) {
    if (g_render_min) return;
    if (c.query_pool) vkCmdBeginQuery(cmd, c.query_pool, c.query, c.query_flags);
    if (c.cond_buffer) occlusion_record_cond(cmd, c, true);
    switch (c.kind) {
    case DrawCall::kIndexedIndirect: vkCmdDrawIndexedIndirect(cmd, c.buffer, c.offset, 1, 20); break;
    case DrawCall::kIndirect: vkCmdDrawIndirect(cmd, c.buffer, c.offset, 1, 16); break;
    case DrawCall::kIndexed: vkCmdDrawIndexed(cmd, c.count, c.instances, 0, c.vertex_offset, 0); break;
    case DrawCall::kDirect: vkCmdDraw(cmd, c.count, c.instances, static_cast<std::uint32_t>(c.vertex_offset), 0); break;
    }
    if (c.cond_buffer) occlusion_record_cond(cmd, c, false);
    if (c.query_pool) vkCmdEndQuery(cmd, c.query_pool, c.query);
}

}  // namespace gpu
