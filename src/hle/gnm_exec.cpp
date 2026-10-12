#include "host/reflex.h"
// CPU-visible side effects of a Gnm command stream. Until Phase 3 there is
// no GPU, but the engine still waits on labels, fences and timestamps that
// EVENT_WRITE_EOP / RELEASE_MEM / WRITE_DATA / DMA_DATA would produce. Walk
// the PM4 packets and perform exactly those writes; everything else is
// skipped. Draw/dispatch state is ignored here (the renderer owns it).
#include "hle/common.h"
#include "core/portable.h"
#include "engine/yebis_bind.h"
#include "host/frame_stats.h"
#include "hle/modules.h"
#include "host/gpu.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <algorithm>
#include <map>
#include <set>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace {

constexpr std::uint32_t kNop = 0x10;
constexpr std::uint32_t kIndirectBufferConst = 0x33;
constexpr std::uint32_t kWriteData = 0x37;
constexpr std::uint32_t kWaitRegMem = 0x3C;
constexpr std::uint32_t kIndirectBuffer = 0x3F;
constexpr std::uint32_t kEventWrite = 0x46;
constexpr std::uint32_t kEventWriteEop = 0x47;
constexpr std::uint32_t kSetPredication = 0x20;
constexpr unsigned kPixelPipeStatDump = 57;   // EventType::PixelPipeStatDump
constexpr unsigned kEventIndexZpassDone = 1;  // EventIndex::ZpassDone: the packet carries an address
constexpr std::uint32_t kEventWriteEos = 0x48;
constexpr std::uint32_t kReleaseMem = 0x49;
constexpr std::uint32_t kDmaData = 0x50;
constexpr std::uint32_t kDispatchDirect = 0x15;
constexpr std::uint32_t kDrawIndex2 = 0x27;
constexpr std::uint32_t kDrawIndexAuto = 0x2D;
constexpr std::uint32_t kDrawIndexOffset2 = 0x35;
constexpr std::uint32_t kNumInstances = 0x2F;
constexpr std::uint32_t kIndexType = 0x2A;
constexpr std::uint32_t kIndexBase = 0x26;
constexpr std::uint32_t kIndexBufferSize = 0x13;
constexpr std::uint32_t kSetBase = 0x11;
constexpr std::uint32_t kDispatchIndirect = 0x16;
constexpr std::uint32_t kDrawIndirect = 0x24;
constexpr std::uint32_t kDrawIndexIndirect = 0x25;
constexpr std::uint32_t kWriteConstRam = 0x81;
constexpr std::uint32_t kDumpConstRam = 0x83;
constexpr std::uint32_t kIncrementCeCounter = 0x84;
constexpr std::uint32_t kIncrementDeCounter = 0x85;
constexpr std::uint32_t kWaitOnCeCounter = 0x86;
constexpr std::uint32_t kWaitOnDeCounterDiff = 0x88;
constexpr std::uint32_t kSetContextReg = 0x69;
constexpr std::uint32_t kSetShReg = 0x76;
constexpr std::uint32_t kSetUconfigReg = 0x79;

// Register file as the command processor sees it: SH registers (0x2C00
// base, per-stage shader state), context registers (0xA000 base) and
// user-config registers (0xC000 base). One copy per CP thread.
struct CpRegs {
    std::uint32_t sh[0x400];
    std::uint32_t ctx[0x1000];
    std::uint32_t uconfig[0x400];
};
thread_local CpRegs t_regs;

// The constant engine's scratch RAM (48 KiB on this hardware). The CCB writes
// shader constants into it and dumps them into memory for the DCB's draws to
// read; Bloodborne builds most of its constant buffers this way, so ignoring
// these packets left those buffers holding whatever was in memory. One copy per
// CP thread: a submission's CCB and DCB are walked by the same thread.
constexpr std::size_t kConstRamBytes = 48 * 1024;
thread_local std::uint8_t t_const_ram[kConstRamBytes];
std::atomic<std::uint64_t> g_dispatches{0}, g_dispatch_failed{0}, g_draws{0};
const bool g_trace_draws = [] {
    const char* e = std::getenv("BBHOST_TRACE_DRAWS");
    return e && e[0] == '1';
}();
// With every draw a token, the CP neither keeps the context and
// user-config register files nor draws from draw packets, unless a shadow or
// packet-path mode needs them (hle_gx_shadow_wanted) or the switches keep them.
// SH registers are still kept: the compute dispatches read them.
bool cp_registers() {
    static const bool v = [] {
        const char* e = std::getenv("BBHOST_CP_REGISTER_FILE");
        return (e && e[0] == '1') || hle_gx_shadow_wanted();
    }();
    return v;
}
bool cp_draw_packets() {
    static const bool v = [] {
        const char* e = std::getenv("BBHOST_CP_DRAW_PACKETS");
        return (e && e[0] == '1') || hle_gx_shadow_wanted();
    }();
    return v;
}
// The dispatch packets run unless the dispatches come from tokens instead.
bool cp_dispatch_packets() {
    static const bool v = [] {
        const char* e = std::getenv("BBHOST_CP_DISPATCH_PACKETS");
        return (e && e[0] == '1') || !hle_gx_dispatch_native();
    }();
    return v;
}
std::atomic<std::uint64_t> g_dispatch_packets_dropped[2] = {};
enum DroppedDraw { kDropIndex2, kDropAuto, kDropOffset2, kDropIndirect, kDropIndexIndirect, kDroppedDraws };
const char* const kDroppedDrawName[kDroppedDraws] = {"DRAW_INDEX_2", "DRAW_INDEX_AUTO", "DRAW_INDEX_OFFSET_2", "DRAW_INDIRECT",
                                                     "DRAW_INDEX_INDIRECT"};
std::atomic<std::uint64_t> g_draw_packets_dropped[kDroppedDraws] = {};
std::atomic<std::uint64_t> g_register_dwords_skipped{0};
bool drop_draw_packet(DroppedDraw kind, std::uint32_t count) {
    if (cp_draw_packets()) return false;
    if (g_draw_packets_dropped[kind].fetch_add(1, std::memory_order_relaxed) < 4) {
        host_log("cp: %s of %u at flip %llu not drawn: every draw is a token and the register file is not kept "
                 "(BBHOST_CP_DRAW_PACKETS=1 draws it)", kDroppedDrawName[kind], count, static_cast<unsigned long long>(hle_video_flip_count()));
    }
    return true;
}
thread_local std::uint32_t t_cs_writers[32] = {};  // checks: per compute user-data slot, the last SET_SH_REG header and the marker before it
thread_local std::uint64_t t_index_base = 0;
thread_local std::uint32_t t_index_size = 0;
thread_local std::uint32_t t_index_type = 0;
thread_local std::uint32_t t_num_instances = 1;
// SET_BASE's DrawIndexIndirPatchTable: where the indirect draw and dispatch
// arguments live. The CP reads the counts from there instead of the packet.
thread_local std::uint64_t t_indirect_base = 0;
// The packet being executed (header address, opcode), for BBHOST_GX_TRACE.
thread_local std::uint64_t t_packet_va = 0;
thread_local std::uint32_t t_packet_op = 0;
thread_local const std::uint32_t* t_next_packet = nullptr;  // the packet after the one being executed

std::atomic<std::uint64_t> g_draw_failed{0};
std::atomic<std::uint64_t> g_indirect_zero_cpu{0};  // GX indirect draws whose count read 0 on the CPU, drawn for the GPU to decide
std::atomic<std::uint64_t> g_flush_by_op[256];
std::atomic<std::uint64_t> g_wait_timeouts{0};
std::atomic<std::uint64_t> g_wait_pending{0};  // waits satisfied by a recorded GPU write
std::atomic<std::uint64_t> g_resyncs{0};       // PM4 walk re-locks

// Hands a draw to the renderer; BBHOST_TRACE_DRAWS=1 also prints the state.
void indirect_log(const char* kind, std::uint64_t args, const std::uint32_t* a, int count) {
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) >= 8) return;
    host_log("gnm exec: %s args at 0x%llx: %u %u %u %u %u", kind, static_cast<unsigned long long>(args), a[0], a[1],
             count > 2 ? a[2] : 0u, count > 3 ? a[3] : 0u, count > 4 ? a[4] : 0u);
}

void bind_gds_counters(GpuDispatch& d);

void run_dispatch(const char* kind, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t indirect_va = 0) {
    GpuDispatch d;
    const std::uint32_t* sh = t_regs.sh;
    d.code_va = (static_cast<std::uint64_t>(sh[0x20D] & 0xff) << 40) | (static_cast<std::uint64_t>(sh[0x20C]) << 8);
    d.rsrc1 = sh[0x212];
    d.rsrc2 = sh[0x213];
    d.threads[0] = sh[0x207];
    d.threads[1] = sh[0x208];
    d.threads[2] = sh[0x209];
    std::memcpy(d.user_data, &sh[0x240], sizeof(d.user_data));
    d.dim[0] = x;
    d.dim[1] = y;
    d.dim[2] = z;
    d.indirect_va = indirect_va;
    d.depth_clear = t_regs.ctx[0x0B];
    d.stencil_clear = t_regs.ctx[0x0A];
    // A dispatch token in shadow mode: this packet is the dispatch it described.
    if (hle_gx_dispatch_pending()) hle_gx_dispatch_check(d, t_cs_writers);
    if (!cp_dispatch_packets()) {
        if (g_dispatch_packets_dropped[indirect_va ? 1 : 0].fetch_add(1, std::memory_order_relaxed) < 4) {
            host_log("cp: %s %ux%ux%u at flip %llu not run: every dispatch is a token and the register file is not kept "
                     "(BBHOST_CP_DISPATCH_PACKETS=1 runs it)", kind, x, y, z, static_cast<unsigned long long>(hle_video_flip_count()));
        }
        return;
    }
    g_dispatches.fetch_add(1);
    hle_gx_trace_dispatch(t_packet_op, t_packet_va, x, sh[0x20C]);
    if ((!x || !y || !z) && !indirect_va) return;
    if (!d.threads[0]) return;
    bind_gds_counters(d);
    if (!host_gpu_dispatch(d)) {
        g_dispatch_failed.fetch_add(1);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 6) {
            host_log("HLE fake: %s %ux%ux%u of shader 0x%llx (threads %ux%ux%u) skipped", kind, x, y, z,
                     static_cast<unsigned long long>(d.code_va), d.threads[0], d.threads[1], d.threads[2]);
        }
    }
}

// BBHOST_GX_NATIVE=2: a draw its GX method left as a token.
// Inputs, objects and geometry come from the call; the register file only
// supplies what the draw's own packets never write.
void native_draw(const GpuDrawInputs* in, const char* kind) {
    const GxDrawObjects* o = hle_gx_trace_objects();
    if (!o || !o->geometry || (!o->count && !o->indirect_va)) return;
    g_draws.fetch_add(1);
    GpuDraw d;
    d.gx = in;
    d.gx_render = true;
    d.gx_objects = o;
    d.gx_token = true;
    d.gx_token_kind = kind;
    d.registers = cp_registers();
    d.sh = t_regs.sh;
    d.ctx = t_regs.ctx;
    d.uconfig = t_regs.uconfig;
    d.index_count = o->count;
    d.instance_count = o->instances ? o->instances : 1;
    d.index_va = o->indexed && o->index_known ? o->index_va : 0;
    d.index_type = o->index_type;
    d.indirect_va = o->indirect_va;
    if (o->indirect_va) {
        // As the packet path does: the arguments read here give the renderer
        // the count and instances it sizes buffers by; the draw itself stays
        // indirect ({count, instances, first index, base vertex, first
        // instance} for an indexed draw, {count, instances, first vertex,
        // first instance} otherwise).
        if (!hle_kernel_va_mapped(o->indirect_va, o->indexed ? 20 : 16)) {
            g_draw_failed.fetch_add(1);
            return;
        }
        const auto* a = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(o->indirect_va));
        // What the CPU reads here is what memory held when the token was
        // walked, not what the GPU will read: an indirect draw's arguments
        // are often written on the GPU by a compute pass earlier in the same
        // command buffer (the game's GPU particles: each emitter's live count),
        // which has not run yet - so a 0 here is no reason to drop the draw.
        // Dropping it (as this did) lost the black particles of a lamp's
        // travel and the blood rain after a boss: their counts are 0 in
        // memory until the frame's own simulation writes them. The draw stays
        // indirect and the GPU reads the real count; the CPU's values only
        // size what the renderer binds, at least one element.
        // BBHOST_INDIRECT_ZERO=skip drops them again.
        static const bool skip_zero = [] {
            const char* e = std::getenv("BBHOST_INDIRECT_ZERO");
            return e && std::strcmp(e, "skip") == 0;
        }();
        if (!a[0]) {
            if (skip_zero) return;
            g_indirect_zero_cpu.fetch_add(1, std::memory_order_relaxed);
        }
        d.index_count = a[0] ? a[0] : 1;
        d.instance_count = a[1] ? a[1] : 1;
    }
    if (!host_gpu_draw(d)) g_draw_failed.fetch_add(1);
}

void trace_draw(const char* kind, std::uint32_t count, std::uint64_t index_va, std::uint32_t instances = 0,
                std::uint64_t indirect_va = 0) {
    const std::uint64_t n = g_draws.fetch_add(1);
    // BBHOST_TRACE_DRAWKIND=1: the draw opcode, count and index base for the
    // first draws (tells DRAW_INDEX_2 from DRAW_INDEX_AUTO for a given draw).
    static const bool trace_kind = [] {
        const char* e = std::getenv("BBHOST_TRACE_DRAWKIND");
        return e && e[0] == '1';
    }();
    if (trace_kind) {
        static std::atomic<int> klogs{0};
        // Only the indexed draws (AUTO draws are fine); dump their first indices.
        // Gate to the menu frame (flip >= 600) so we sample the title, not boot.
        if (index_va && hle_video_flip_count() >= 600 && (klogs.load(std::memory_order_relaxed) < 1500 && klogs.fetch_add(1) < 1500)) {
            const std::uint16_t* p16 = hle_kernel_va_mapped(index_va, 24) ? reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(index_va)) : nullptr;
            host_log("drawkind #%llu %s count=%u index_va=0x%llx itype=%u idx:%u %u %u %u %u %u %u %u %u", static_cast<unsigned long long>(n), kind, count,
                     static_cast<unsigned long long>(index_va), t_index_type, p16 ? p16[0] : 0, p16 ? p16[1] : 0, p16 ? p16[2] : 0,
                     p16 ? p16[3] : 0, p16 ? p16[4] : 0, p16 ? p16[5] : 0, p16 ? p16[6] : 0, p16 ? p16[7] : 0, p16 ? p16[8] : 0);
        }
    }
    bool gx_render = false;
    const GpuDrawInputs* gx = hle_gx_trace_draw(t_packet_op, t_packet_va, count, instances ? instances : t_num_instances,
                                                t_index_base, t_index_type, t_regs.uconfig[0x242], t_regs.sh, t_regs.ctx,
                                                &gx_render);
    // A Scaleform draw token in shadow mode: the packet after it is the draw
    // the token described; compare the two.
    if (hle_gx_scaleform_pending()) {
        hle_gx_scaleform_check(count, index_va, t_index_type & 1, instances ? instances : t_num_instances, t_regs.uconfig[0x242],
                               t_regs.sh, t_regs.ctx);
    }
    if (count || indirect_va) {
        GpuDraw d;
        d.gx = gx;
        d.gx_render = gx && gx_render;
        d.gx_compare = gx != nullptr && !d.gx_render;  // compare mode only
        d.gx_objects = hle_gx_trace_objects();
        d.registers = cp_registers();
        d.sh = t_regs.sh;
        d.ctx = t_regs.ctx;
        d.uconfig = t_regs.uconfig;
        d.index_count = count;
        d.instance_count = instances ? instances : (t_num_instances ? t_num_instances : 1);
        d.index_va = index_va;
        d.index_type = t_index_type & 1;
        d.indirect_va = indirect_va;
        // Rendering from GX takes the geometry from the call;
        // an index buffer the call cannot place keeps the packet's address.
        const GxDrawObjects* o = d.gx_render ? d.gx_objects : nullptr;
        if (o && o->geometry && !indirect_va && o->indexed == (index_va != 0)) {
            d.index_count = o->count;
            d.instance_count = o->instances;
            if (o->indexed && o->index_known) {
                d.index_va = o->index_va;
                d.index_type = o->index_type;
            }
        }
        if (!host_gpu_draw(d)) g_draw_failed.fetch_add(1);
    }
    if (!g_trace_draws || n >= 400) return;
    const std::uint32_t* sh = t_regs.sh;
    const std::uint32_t* cx = t_regs.ctx;
    const std::uint64_t vs = (static_cast<std::uint64_t>(sh[0x49] & 0xff) << 40) | (static_cast<std::uint64_t>(sh[0x48]) << 8);
    const std::uint64_t ps = (static_cast<std::uint64_t>(sh[0x09] & 0xff) << 40) | (static_cast<std::uint64_t>(sh[0x08]) << 8);
    host_log("draw #%llu %s count=%u idx=0x%llx type=%u inst=%u prim=%u vs=0x%llx rsrc=%08x/%08x ps=0x%llx rsrc=%08x/%08x",
             static_cast<unsigned long long>(n), kind, count, static_cast<unsigned long long>(index_va), t_index_type,
             t_num_instances, t_regs.uconfig[0x242], static_cast<unsigned long long>(vs), sh[0x4a], sh[0x4b],
             static_cast<unsigned long long>(ps), sh[0x0a], sh[0x0b]);
    // Render targets: CB_COLORn_* at 0xA318 + n*0x0F: BASE, PITCH, SLICE, VIEW, INFO, ATTRIB, ...
    for (int t = 0; t < 8; ++t) {
        const std::uint32_t* c = cx + 0x318 + t * 0x0F;
        if (!c[0]) continue;
        host_log("  mrt%d base=%08x pitch=%08x slice=%08x view=%08x info=%08x attrib=%08x", t, c[0], c[1], c[2], c[3], c[4], c[5]);
    }
    host_log("  db z_info=%08x stencil_info=%08x z_read=%08x depth_ctl=%08x depth_size=%08x depth_view=%08x render_ctl=%08x",
             cx[0x10], cx[0x11], cx[0x12], cx[0x200], cx[0x0], cx[0x2], cx[0x0]);
    host_log("  cb target_mask=%08x shader_mask=%08x blend0=%08x color_ctl=%08x; pa su_sc_mode=%08x cl_clip=%08x vport=%08x/%08x scissor=%08x/%08x; spi ps_in_ctl=%08x vs_out_cfg=%08x ps_input_ena=%08x",
             cx[0x8E], cx[0x8F], cx[0x1E0], cx[0x202], cx[0x205], cx[0x204], cx[0x10F], cx[0x110], cx[0x0C], cx[0x0D],
             cx[0x1B6], cx[0x1B1], cx[0x1B3]);
}


std::atomic<std::uint64_t> g_exec_packets{0};
std::atomic<std::uint64_t> g_exec_writes{0};
const bool g_trace = [] {
    const char* e = std::getenv("BBHOST_TRACE_GNM");
    return e && e[0] == '1';
}();
// The YEBIS draw a token said was coming, until its DRAW_INDEX_2 is walked.
thread_local bool t_yebis_pending = false;
thread_local GxYebisDraw t_yebis_want{};
std::atomic<std::uint64_t> g_op_hist[256];
// BBHOST_GX_COST: nanoseconds per opcode. exec is the largest command-processor
// bucket (3.9 ms a flip) and the ring is 61% register writes, so the question
// is whether that time is spread over the register packets or concentrated in a
// few handlers. Cost mode only: this is two clock reads a packet.
std::atomic<std::uint64_t> g_op_ns[256];
const bool g_exec_cost = [] {
    const char* e = std::getenv("BBHOST_GX_COST");
    return e && std::atoi(e) != 0;
}();
std::atomic<std::uint64_t> g_token_nops{0};  // NOPs that were host-draw tokens (BBHOST_GX_NATIVE)
std::atomic<std::uint64_t> g_copy_nops{0};   // NOPs that were copy tokens (BBHOST_GX_NATIVE_COPIES)
std::atomic<std::uint64_t> g_exec_dwords{0};  // dwords of the packets executed, headers included
thread_local int t_queue_id = -1;
void drain_other_queues(int except_id, int max_ms);

// Last writes, for the crash dump.
struct WriteRec {
    std::uint64_t va;
    std::uint64_t value;
    std::uint32_t bytes;
    std::uint32_t op;
};
constexpr std::size_t kRecs = 2048;
WriteRec g_recs[kRecs];
std::atomic<std::uint64_t> g_rec_next{0};
thread_local std::uint32_t t_cur_op = 0;

// BBHOST_WATCH_RANGE=0x<addr>:0x<len> logs every CP write touching it.
const std::uint64_t g_watch_lo = [] {
    const char* e = std::getenv("BBHOST_WATCH_RANGE");
    return e ? std::strtoull(e, nullptr, 16) : 0ull;
}();
const std::uint64_t g_watch_hi = [] {
    const char* e = std::getenv("BBHOST_WATCH_RANGE");
    if (!e) {
        return 0ull;
    }
    const char* c = std::strchr(e, ':');
    return c ? std::strtoull(e, nullptr, 16) + std::strtoull(c + 1, nullptr, 16) : 0ull;
}();

// BBHOST_ARENA_MONITOR=1:
// the game's command arena marks the last word of each chunk 0 free, 1 in use
// or 2 retired, and a retired chunk is freed by an EVENT_WRITE_EOP of 0, which
// we record on the GPU. Markers are learned from those writes (8 bytes of 0
// onto a word holding 2); any other CP write over a known marker is logged.
const bool g_arena_monitor = [] {
    const char* e = std::getenv("BBHOST_ARENA_MONITOR");
    return e && e[0] == '1';
}();
std::mutex g_arena_mu;
std::set<std::uint64_t> g_arena_markers;
std::atomic<std::uint64_t> g_arena_completions{0}, g_arena_foreign{0};
std::atomic<std::uint64_t> g_arena_acquire_waits{0}, g_arena_acquire_wait_us{0}, g_arena_acquire_failed{0};  // hle_gx_arena_acquire
std::atomic<std::uint64_t> g_arena_data_writes{0}, g_arena_data_bytes{0};  // non-EOP CP writes into arena chunks (monitor)
// BBHOST_ARENA_EARLY_FREE=1 frees a retired chunk when the CP walks its zeroing
// write (cp_write). Off by default: with the GX hooks armed, 2 of 3 runs
// crashed in the GX heap's release pass or hung, against 0 of 2 with it off.
const bool g_arena_early_free = [] {
    const char* e = std::getenv("BBHOST_ARENA_EARLY_FREE");
    return e && e[0] == '1';
}();
thread_local std::vector<std::uint64_t> t_arena_frees;  // markers to free when this CP job's walk ends
std::atomic<std::uint64_t> g_arena_early_frees{0};
// BBHOST_ARENA_MONITOR=1: the CP writes held back behind queued GPU work that
// land inside command-arena chunks (labels), per chunk base, and when early free
// last zeroed each chunk (under g_arena_mu). When a chunk is handed out again, a
// queued value not in its memory yet means the GPU had not executed that write:
// it will land in the new owner's chunk.
struct ArenaQueuedWrite {
    std::uint64_t va, value;
    std::uint32_t bytes;
    std::uint64_t serial;  // the host submission it is recorded into
};
struct ArenaChunkLife {
    std::vector<ArenaQueuedWrite> queued;
    std::chrono::steady_clock::time_point freed{};
    std::uint64_t last_free_event = 0;  // g_arena_free_event at its latest early free (kept across hand-outs)
};
std::unordered_map<std::uint64_t, ArenaChunkLife> g_arena_life;
std::atomic<std::uint64_t> g_arena_queued_writes[3] = {};  // value 1, value 4, other values
// Hand-outs with a queued value not in memory (the guest may have rewritten it),
// and with a write in a submission not yet completed (the hazard).
std::atomic<std::uint64_t> g_arena_handed_out{0}, g_arena_handed_out_unlanded{0}, g_arena_handed_out_queued{0};
// Early frees, numbered, and CP walks of a buffer in a chunk freed after the
// walked job was submitted (the chunk may hold another owner's commands).
std::atomic<std::uint64_t> g_arena_free_event{0}, g_arena_stale_walks{0};
thread_local std::uint64_t t_job_free_event = ~0ull;  // the walked job's g_arena_free_event at submit; ~0: none

std::uint64_t guest_u64(std::uint64_t va) { return *reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(va)); }

// Runs before the write reaches guest memory, so the marker's old value is visible.
void note_arena_write(std::uint64_t va, std::uint64_t value, std::size_t n) {
    std::lock_guard<std::mutex> lk(g_arena_mu);
    const bool completion = t_cur_op == kEventWriteEop && n == 8 && value == 0;
    if (completion && !g_arena_markers.count(va) && hle_kernel_va_mapped(va, 8) && guest_u64(va) == 2) {
        g_arena_markers.insert(va);
    }
    auto it = g_arena_markers.lower_bound(va >= 7 ? va - 7 : 0);
    if (it == g_arena_markers.end() || *it >= va + n) return;
    if (completion && *it == va) {
        g_arena_completions.fetch_add(1);
        return;
    }
    if (g_arena_foreign.fetch_add(1) < 32) {
        host_log("arena: op 0x%02x writes 0x%llx (%zu bytes, value 0x%llx) over marker 0x%llx holding 0x%llx", t_cur_op,
                 static_cast<unsigned long long>(va), n, static_cast<unsigned long long>(value), static_cast<unsigned long long>(*it),
                 static_cast<unsigned long long>(guest_u64(*it)));
    }
}

void record(std::uint64_t va, const void* src, std::size_t n) {
    host_gpu_shadow_cp_write(va, n);
    std::uint64_t v = 0;
    std::memcpy(&v, src, n > 8 ? 8 : n);  // bytes keeps the full extent
    if (g_arena_monitor) {
        note_arena_write(va, v, n);
        // Data the CP places inside command-arena chunks (constant-RAM dumps,
        // DMA, WRITE_DATA): what later draws may read from chunk memory.
        if (t_cur_op != kEventWriteEop && hle_gx_arena_overlap(va, n, nullptr)) {
            g_arena_data_writes.fetch_add(1);
            g_arena_data_bytes.fetch_add(n);
        }
    }
    // Direct memory is also mapped at its GPU alias, which packets may address.
    static const std::uint64_t watch_alias = g_watch_hi ? hle_kernel_gpu_alias(g_watch_lo) : 0;
    const bool watched = g_watch_hi && ((va < g_watch_hi && va + n > g_watch_lo) ||
                                        (watch_alias && va < watch_alias + (g_watch_hi - g_watch_lo) && va + n > watch_alias));
    if (watched) {
        host_log("gnm WATCH: op=0x%02x write 0x%llx <- 0x%llx (%zu bytes) at flip %llu", t_cur_op, static_cast<unsigned long long>(va),
                 static_cast<unsigned long long>(v), n, static_cast<unsigned long long>(hle_video_flip_count()));
    }
    const std::uint64_t i = g_rec_next.fetch_add(1);
    WriteRec& r = g_recs[i % kRecs];
    r.va = va;
    r.value = v;
    r.bytes = static_cast<std::uint32_t>(n);
    r.op = t_cur_op;
}

std::uint64_t gpu_va(std::uint32_t lo, std::uint32_t hi) {
    return (static_cast<std::uint64_t>(hi & 0xffffu) << 32) | lo;
}

std::uint64_t gpu_clock() { return rdtsc_now(); }

bool store(std::uint64_t va, const void* src, std::size_t n) {
    if (!hle_kernel_va_mapped(va, n)) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 8) {
            host_log("gnm exec: write to unmapped 0x%llx (%zu bytes) skipped",
                     static_cast<unsigned long long>(va), n);
        }
        return false;
    }
    if (g_trace) {
        std::uint64_t v = 0;
        std::memcpy(&v, src, n > 8 ? 8 : n);
        host_log("gnm exec: write 0x%llx <- 0x%llx (%zu bytes)", static_cast<unsigned long long>(va),
                 static_cast<unsigned long long>(v), n);
    }
    record(va, src, n);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va)), src, n);
    g_exec_writes.fetch_add(1);
    return true;
}

// A CP memory write: recorded on the GPU behind the queued shader work
// when the memory is imported, otherwise flushed and written here.
// Label values the GPU will write (recorded, possibly not yet executed).
// A wait on such a label need not block: everything recorded after it on
// the one Vulkan queue runs after the write anyway. By dword address, open
// addressing: every 4- and 8-byte CP write lands here, and the node map this
// replaces allocated for each new address and missed the cache on each
// lookup (1.2% of the command processor).
class PendingLabels {
public:
    void set(std::uint64_t va, std::uint32_t value) {
        if ((used_ + 1) * 2 > slots_.size()) grow();
        Slot& s = slots_[find(va)];
        if (!s.va) {
            s.va = va;
            ++used_;
        }
        s.value = value;
    }
    bool get(std::uint64_t va, std::uint32_t& value) const {
        if (slots_.empty()) return false;
        const Slot& s = slots_[find(va)];
        if (!s.va) return false;
        value = s.value;
        return true;
    }
    std::size_t size() const { return used_; }

private:
    struct Slot {
        std::uint64_t va = 0;  // 0: empty (no label lives at 0)
        std::uint32_t value = 0;
    };
    std::vector<Slot> slots_;
    std::size_t used_ = 0;
    unsigned bits_ = 0;
    std::size_t find(std::uint64_t va) const {
        const std::size_t mask = slots_.size() - 1;
        std::size_t i = static_cast<std::size_t>(((va >> 2) * 0x9e3779b97f4a7c15ull) >> (64 - bits_));
        while (slots_[i].va && slots_[i].va != va) i = (i + 1) & mask;
        return i;
    }
    void grow() {
        std::vector<Slot> old;
        old.swap(slots_);
        bits_ = bits_ ? bits_ + 1 : 12;
        slots_.assign(std::size_t{1} << bits_, Slot{});
        for (const Slot& s : old) {
            if (s.va) slots_[find(s.va)] = s;
        }
    }
};
std::mutex g_pending_mu;
PendingLabels g_pending_labels;

// The last CP writes, for the watchdog: when the game spins on a flush counter
// that never advances, the question is always whether we ever wrote that
// address and with what.
struct LabelWrite {
    std::uint64_t va, value;
    std::uint32_t op, bytes;
};
constexpr std::size_t kLabelRing = 32;
LabelWrite g_label_ring[kLabelRing];
std::atomic<std::uint64_t> g_label_next{0};

// BBHOST_TRACE_LABEL=<hex>: every CP write to one address. The game spins on a
// flush counter without a timeout, so "did we ever write it" is the question.
const std::uint64_t g_trace_label = [] {
    const char* e = std::getenv("BBHOST_TRACE_LABEL");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
}();
std::atomic<std::uint64_t> g_trace_label_writes{0};

// The command buffer the walker is in. The engine reserves its fences *inside*
// the command buffer - legal on hardware, where the CP has long since fetched
// past them by the time the event retires - so writing one while we are still
// walking would rewrite packets we have not read yet. Those writes are held
// back and applied when the walk finishes.
thread_local std::uint64_t t_cb_lo = 0, t_cb_hi = 0;
struct DeferredWrite {
    std::uint64_t va;
    std::uint64_t value;
    std::size_t bytes;
};
thread_local std::vector<DeferredWrite> t_deferred;
std::atomic<std::uint64_t> g_deferred_writes{0};

// Note a fence that lands in the command buffer we are walking. The write
// itself must still happen at once - a WAIT_REG_MEM later in the same stream
// polls it - which is why the walker reads a snapshot of the buffer instead.
bool defer_if_in_command_buffer(std::uint64_t va, const void* src, std::size_t n) {
    if (!t_cb_lo || n > 8 || va + n <= t_cb_lo || va >= t_cb_hi) return false;
    std::uint64_t value = 0;
    std::memcpy(&value, src, n > 8 ? 8 : n);
    g_deferred_writes.fetch_add(1);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 6) {
        host_log("gnm exec: CP write 0x%llx <- %llu (%zu bytes) is inside the command buffer being walked",
                 static_cast<unsigned long long>(va), static_cast<unsigned long long>(value), n);
    }
    return false;
}

// The command buffers recently handed to us. Anything that writes over one of
// them is erasing packets - the game would never do that on hardware, so a hit
// here means we decoded some destination address wrongly.
std::mutex g_cb_ranges_mu;
std::array<std::pair<std::uint64_t, std::uint64_t>, 64> g_cb_ranges{};
std::size_t g_cb_ranges_next = 0;

void note_command_buffer(std::uint64_t base, std::size_t bytes) {
    if (!base || !bytes) return;
    std::lock_guard<std::mutex> lk(g_cb_ranges_mu);
    g_cb_ranges[g_cb_ranges_next++ % g_cb_ranges.size()] = {base, base + bytes};
}

bool overlaps_command_buffer(std::uint64_t va, std::size_t bytes, std::uint64_t* which) {
    std::lock_guard<std::mutex> lk(g_cb_ranges_mu);
    for (const auto& r : g_cb_ranges) {
        if (r.first && va < r.second && va + bytes > r.first) {
            if (which) *which = r.first;
            return true;
        }
    }
    return false;
}

void warn_if_over_command_buffer(const char* what, std::uint64_t va, std::size_t bytes) {
    std::uint64_t which = 0;
    if (!overlaps_command_buffer(va, bytes, &which)) return;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 12) {
        host_log("gnm exec: %s writes 0x%llx..0x%llx, which is inside a command buffer at 0x%llx - that erases packets",
                 what, static_cast<unsigned long long>(va), static_cast<unsigned long long>(va + bytes),
                 static_cast<unsigned long long>(which));
    }
}

// Which addresses the CP writes, and which ones a WAIT_REG_MEM gave up on.
// A fence the game waits for and nobody writes is the shape of the world-load
// hang, and the pair of these two maps names it.
std::mutex g_addr_mu;
std::map<std::uint64_t, std::uint64_t> g_written_addrs;   // address -> writes
struct WaitInfo {
    std::uint64_t count = 0;
    std::uint32_t ref = 0, mask = 0, fn = 0;
};
std::map<std::uint64_t, WaitInfo> g_timeout_addrs;  // address -> waits given up on

void note_wait_timeout(std::uint64_t va, std::uint32_t ref, std::uint32_t mask, std::uint32_t fn) {
    std::lock_guard<std::mutex> lk(g_addr_mu);
    if (g_timeout_addrs.size() >= 4096) return;
    WaitInfo& w = g_timeout_addrs[va];
    w.count += 1;
    w.ref = ref;
    w.mask = mask;
    w.fn = fn;
}

void note_label_write(std::uint64_t va, const void* src, std::size_t n) {
    {
        std::lock_guard<std::mutex> lk(g_addr_mu);
        if (g_written_addrs.size() < 8192) g_written_addrs[va] += 1;
    }
    if (n != 4 && n != 8) return;
    std::uint64_t v = 0;
    std::memcpy(&v, src, n);
    const std::uint64_t i = g_label_next.fetch_add(1);
    g_label_ring[i % kLabelRing] = {va, v, t_cur_op, static_cast<std::uint32_t>(n)};
    // BBHOST_ARENA_MONITOR: anything but zero written onto a chunk marker. The
    // game writes 1 and 2 there itself; the CP should only ever write the 0 of
    // a retired chunk's EVENT_WRITE_EOP.
    if (g_arena_monitor && v != 0 && hle_gx_arena_marker(va)) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 64) {
            host_log("arena: CP op 0x%02x writes %llu (%zu bytes) onto marker 0x%llx, walking 0x%llx..0x%llx",
                     t_cur_op, static_cast<unsigned long long>(v), n, static_cast<unsigned long long>(va),
                     static_cast<unsigned long long>(t_cb_lo), static_cast<unsigned long long>(t_cb_hi));
        }
    }
    if (g_trace_label && va == g_trace_label) {
        const std::uint64_t c = g_trace_label_writes.fetch_add(1);
        // The first forty, and after that every write of anything but zero -
        // a marker is zeroed on every free, and the rare other value is the
        // one being looked for.
        if (c < 40 || v != 0) host_log("label trace: 0x%llx <- %llu (%zu bytes, op 0x%02x)", static_cast<unsigned long long>(va),
                             static_cast<unsigned long long>(v), n, t_cur_op);
    }
}

// BBHOST_ARENA_MONITOR=1: a write host_gpu_mem_write held back, if it lands in a
// command-arena chunk (not on its marker).
void note_arena_queued(std::uint64_t va, const void* src, std::size_t n, std::uint64_t serial) {
    std::uint64_t chunk = 0;
    if (n > 8 || !hle_gx_arena_overlap(va, n, &chunk) || hle_gx_arena_marker(va)) return;
    std::uint64_t value = 0;
    std::memcpy(&value, src, n);
    g_arena_queued_writes[value == 1 ? 0 : value == 4 ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_arena_mu);
    std::vector<ArenaQueuedWrite>& queued = g_arena_life[chunk].queued;
    for (ArenaQueuedWrite& w : queued) {
        if (w.va == va) {
            w.value = value;
            w.bytes = static_cast<std::uint32_t>(n);
            w.serial = serial;
            return;
        }
    }
    if (queued.size() < 256) queued.push_back({va, value, static_cast<std::uint32_t>(n), serial});
}

void cp_write(std::uint64_t va, const void* src, std::size_t n) {
    // BBHOST_ARENA_MONITOR: a CP write of any size that covers a chunk marker
    // with anything but the zero of a retired chunk's EVENT_WRITE_EOP.
    if (g_arena_monitor && hle_gx_arena_covers_marker(va, n) && !(n == 8 && hle_gx_arena_marker(va))) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 64) {
            std::uint64_t first = 0;
            std::memcpy(&first, src, n < 8 ? n : 8);
            host_log("arena: CP op 0x%02x writes %zu bytes at 0x%llx (first 0x%llx) across a chunk marker, walking "
                     "0x%llx..0x%llx", t_cur_op, n, static_cast<unsigned long long>(va),
                     static_cast<unsigned long long>(first), static_cast<unsigned long long>(t_cb_lo),
                     static_cast<unsigned long long>(t_cb_hi));
        }
    }
    note_label_write(va, src, n);
    if (defer_if_in_command_buffer(va, src, n)) return;
    if (!hle_kernel_va_mapped(va, n)) {
        store(va, src, n);  // logs the skip
        return;
    }
    // A command-arena chunk's retirement (EVENT_WRITE_EOP of 0 onto a marker
    // holding 2) frees the chunk when this job's walk ends, not behind the
    // queued GPU work like other fences. No GPU binding reads arena memory (0 of
    // 14.6M constant, 2.2M index and 2.5k indirect bindings),
    // and deferred, a retired chunk stayed unusable for 3-4 frames until the arena
    // ran out (0x2ab0a98). Held to the end of the job because the chunk may be
    // the one being walked.
    if (g_arena_early_free && t_cur_op == kEventWriteEop && n == 8 && hle_gx_arena_marker(va)) {
        std::uint64_t value = 0;
        std::memcpy(&value, src, sizeof(value));
        if (value == 0 && guest_u64(va) == 2) {
            record(va, src, n);
            g_exec_writes.fetch_add(1);
            t_arena_frees.push_back(va);
            return;
        }
    }
    std::uint64_t serial = 0;
    if (host_gpu_mem_write_serial(va, src, n, &serial)) {
        record(va, src, n);
        g_exec_writes.fetch_add(1);
        if (g_arena_monitor) note_arena_queued(va, src, n, serial);
        if (n == 4 || n == 8) {
            std::lock_guard<std::mutex> lk(g_pending_mu);
            std::uint32_t lo, hi = 0;
            std::memcpy(&lo, src, 4);
            if (n == 8) std::memcpy(&hi, static_cast<const std::uint8_t*>(src) + 4, 4);
            g_pending_labels.set(va, lo);
            if (n == 8) g_pending_labels.set(va + 4, hi);
        }
        return;
    }
    g_flush_by_op[t_cur_op & 0xff].fetch_add(1);
    host_gpu_flush();
    store(va, src, n);
}

// The chunk frees cp_write collected during a job, applied once its walk is done.
void apply_arena_frees() {
    if (t_arena_frees.empty()) return;
    for (const std::uint64_t va : t_arena_frees) {
        auto* marker = reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(va));
        if (__sync_bool_compare_and_swap(marker, std::uint64_t{2}, std::uint64_t{0})) {
            g_arena_early_frees.fetch_add(1);
            std::uint64_t chunk = 0;
            if (g_arena_monitor && hle_gx_arena_overlap(va, 8, &chunk)) {
                std::lock_guard<std::mutex> lk(g_arena_mu);
                ArenaChunkLife& life = g_arena_life[chunk];
                life.freed = std::chrono::steady_clock::now();
                life.last_free_event = g_arena_free_event.fetch_add(1) + 1;
            }
        }
    }
    t_arena_frees.clear();
}

void write_sel(std::uint64_t va, unsigned data_sel, std::uint32_t lo, std::uint32_t hi) {
    switch (data_sel) {
        case 1: {  // 32-bit data
            cp_write(va, &lo, 4);
            break;
        }
        case 2: {  // 64-bit data
            std::uint64_t v = (static_cast<std::uint64_t>(hi) << 32) | lo;
            cp_write(va, &v, 8);
            break;
        }
        case 3: {  // GPU clock
            std::uint64_t v = gpu_clock();
            cp_write(va, &v, 8);
            break;
        }
        case 4: {  // 64-bit perf counter / system clock
            std::uint64_t v = now_us();
            cp_write(va, &v, 8);
            break;
        }
        default:
            break;
    }
}

void execute(const std::uint32_t* cb, std::size_t dwords, int depth);
void execute_packet(std::uint32_t op, const std::uint32_t* b, std::size_t n, int depth);

// The constant engine is a second command stream (the CCB) that runs ahead of
// the draw engine: it fills a ring of constant buffers and dumps them into
// memory, and the two streams keep in step through a pair of counters -
// WaitOnCeCounter holds the draw engine back until the CE has produced the next
// buffer, WaitOnDeCounterDiff holds the CE back so it cannot overwrite a buffer
// the draws have not consumed. Walking the whole CCB first (what we did) made
// every draw read whatever the CE wrote last.
struct CeFrame {
    const std::uint32_t* cb;
    std::size_t left;  // dwords
};
thread_local std::vector<CeFrame> t_ce_stack;
thread_local std::uint32_t t_ce_count = 0, t_de_count = 0;

// BBHOST_CE_HISTORY=1 (diagnostic): the recent constant-engine packets on this
// CP thread, so a draw can ask which DumpConstRam produced the resource table
// it reads and which WriteConstRam packets filled that part of const RAM
// (hle_gnm_explain_table).
const bool g_ce_history = [] {
    const char* e = std::getenv("BBHOST_CE_HISTORY");
    return e && e[0] == '1';
}();
struct CeRecord {
    std::uint64_t seq = 0, dst = 0;
    std::uint32_t off = 0, bytes = 0, ce = 0, de = 0;
    std::uint32_t head[4]{};
    bool dump = false;
};
constexpr std::size_t kCeHistory = 16384;
thread_local std::vector<CeRecord> t_ce_history;
thread_local std::uint64_t t_ce_seq = 0;

void note_ce(bool dump, std::uint64_t dst, std::uint32_t off, std::size_t bytes, const std::uint8_t* data) {
    if (!g_ce_history) return;
    if (t_ce_history.empty()) t_ce_history.resize(kCeHistory);
    CeRecord& r = t_ce_history[t_ce_seq % kCeHistory];
    r.seq = t_ce_seq++;
    r.dst = dst;
    r.off = off;
    r.bytes = static_cast<std::uint32_t>(bytes);
    r.ce = t_ce_count;
    r.de = t_de_count;
    r.dump = dump;
    std::memset(r.head, 0, sizeof(r.head));
    std::memcpy(r.head, data, std::min<std::size_t>(bytes, sizeof(r.head)));
}

// Runs one constant-engine packet. false when the stream is blocked on the draw
// engine or exhausted.
bool ce_step() {
    while (!t_ce_stack.empty()) {
        CeFrame& f = t_ce_stack.back();
        if (f.left == 0) {
            t_ce_stack.pop_back();
            continue;
        }
        const std::uint32_t hdr = f.cb[0];
        const unsigned type = hdr >> 30;
        if (type != 3) {
            const std::size_t adv = type == 2 ? 1u : 2u + ((hdr >> 16) & 0x3fff);
            if (adv > f.left) {
                f.left = 0;
                continue;
            }
            f.cb += adv;
            f.left -= adv;
            continue;
        }
        const std::uint32_t op = (hdr >> 8) & 0xff;
        const std::size_t n = static_cast<std::size_t>((hdr >> 16) & 0x3fff) + 1;
        if (1 + n > f.left) {
            f.left = 0;
            continue;
        }
        const std::uint32_t* b = f.cb + 1;
        if (op == kWaitOnDeCounterDiff) {
            const std::uint32_t diff = b[0];
            if (diff && (t_de_count - t_ce_count) >= diff) return false;  // leave it to retry
        }
        f.cb += 1 + n;
        f.left -= 1 + n;
        if (op == kIndirectBufferConst || op == kIndirectBuffer) {
            if (n >= 3) {
                const std::uint64_t va = gpu_va(b[0], b[1]);
                const std::size_t dw = b[2] & 0xfffff;
                if (dw && t_ce_stack.size() < 8 && hle_kernel_va_mapped(va, dw * 4)) {
                    t_ce_stack.push_back({reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va)), dw});
                }
            }
            return true;
        }
        t_cur_op = op;
        execute_packet(op, b, n, 1);
        return true;
    }
    return false;
}

// BBHOST_CP_CENSUS=1: what the command processor still walks,
// by packet type and dwords, with the NOP payloads (our tokens and Gnmx's
// markers) told apart by their magic.
const bool g_cp_census = [] {
    const char* e = std::getenv("BBHOST_CP_CENSUS");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_census_packets[0x100] = {}, g_census_dwords[0x100] = {};
std::mutex g_census_mu;
std::map<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>> g_census_nops;  // magic -> {packets, dwords}

std::atomic<std::uint64_t> g_census_ctx_reg[0x1000] = {};
void census_packet(std::uint32_t op, const std::uint32_t* b, std::size_t n) {
    g_census_packets[op & 0xff].fetch_add(1, std::memory_order_relaxed);
    g_census_dwords[op & 0xff].fetch_add(n + 1, std::memory_order_relaxed);
    if ((op == kSetContextReg || op == kSetUconfigReg) && n >= 1 && b[0] < 0x1000) {
        g_census_ctx_reg[b[0]].fetch_add(n, std::memory_order_relaxed);
    }
    if (op != kNop) return;
    const std::uint32_t magic = n >= 1 ? b[0] : 0;
    std::lock_guard<std::mutex> lk(g_census_mu);
    if (g_census_nops.size() < 64 || g_census_nops.count(magic)) {
        auto& e = g_census_nops[magic];
        ++e.first;
        e.second += n + 1;
    }
}

// The next two draw tokens after this packet in the stream, a
// few packets on: the slot of the second and the records of the first are
// asked for now, so they arrive while this draw is drawn. A draw's slot and
// records are written on a GX thread and are cold here, and read one line
// at a time they were most of what a draw cost.
void lookahead_tokens(const std::uint32_t* p) {
    // BBHOST_TOKEN_PREFETCH=0: no lookahead (for pricing it on a machine).
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_TOKEN_PREFETCH");
        return !(e && e[0] == '0');
    }();
    if (!on) return;
    const auto* end = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(t_cb_hi));
    int found = 0;
    for (int packets = 0; packets < 32 && found < 2 && p + 1 < end; ++packets) {
        const std::uint32_t hdr = p[0];
        if ((hdr >> 30) != 3) break;
        const std::size_t cnt = ((hdr >> 16) & 0x3fff) + 1;
        if (p + 1 + cnt > end) break;
        if (((hdr >> 8) & 0xff) == kNop && cnt >= 3 && p[1] == kGxNativeMagic) {
            hle_gx_native_prefetch(p[2] | (static_cast<std::uint64_t>(p[3]) << 32), found == 0);
            ++found;
        }
        p += 1 + cnt;
    }
}

// ---- GDS counters ----
// GX keeps an append/consume buffer's counter in the word past its records
// (buffer + size - 8) and moves it through GDS around each dispatch that
// counts with it: DMA_DATA from that word - or an initial count, a D3D11
// UAV's - into GDS dword 8 + slot before (gx_commit_compute_stage), DMA_DATA
// from GDS back into the word after (gx_dispatch_post). Its
// CopyStructureCount is a DMA_DATA from the word into indirect arguments
// (0x2568be0). That word is the counter. A dispatch counts in it
// (GpuDispatch::gds, bind_gds_counters): its initial count is written into
// the word on the GPU's timeline before the dispatch, and the store after it
// is then the word's own. A store out of GDS with no dispatch counting in
// between puts what the dword holds into the word on the GPU's timeline - an
// initial count written there, another word's count copied; the word it was
// loaded from, which holds it already, is left as it is - and every copy out
// of a counter word (or write into one) goes on the GPU's timeline too, where
// the word's value is: done on the CPU as the command processor walks, it
// read memory the GPU had not written yet. Dropped, as these were, a new
// particle system's counter kept whatever the reused memory held, and its
// CopyStructureCount handed the next indirect dispatch 3,176,062,614
// workgroups (float bits).
constexpr std::uint32_t kGdsDwords = 64;  // GDS bytes 0..0xff are followed; GX moves its counters through 0x20..0x3f
struct GdsHeld {
    enum Kind : std::uint8_t { kUnknown, kWord, kValue };
    Kind kind = kUnknown;
    std::uint64_t word = 0;   // kWord: the counter word whose count the dword holds
    std::uint32_t value = 0;  // kValue: a count loaded into it
};
thread_local GdsHeld t_gds[kGdsDwords];
std::atomic<std::uint64_t> g_gds_loads{0}, g_gds_stores_folded{0}, g_gds_stores_moved{0}, g_counter_reads{0}, g_counter_writes{0},
    g_gds_dispatches{0}, g_gds_found_after{0};
// The counter words seen, so that a copy out of one (or a write into one) goes
// on the GPU's timeline. Few: one per append/consume buffer.
std::mutex g_counter_words_mu;
std::set<std::uint64_t> g_counter_words;
std::atomic<bool> g_any_counter_words{false};

void note_counter_word(std::uint64_t word) {
    std::lock_guard<std::mutex> lk(g_counter_words_mu);
    if (g_counter_words.size() < 65536 && g_counter_words.insert(word).second) g_any_counter_words.store(true, std::memory_order_relaxed);
}

bool covers_counter_word(std::uint64_t va, std::size_t bytes) {
    if (!g_any_counter_words.load(std::memory_order_relaxed)) return false;
    std::lock_guard<std::mutex> lk(g_counter_words_mu);
    const auto it = g_counter_words.lower_bound(va & ~3ull);
    return it != g_counter_words.end() && *it < va + bytes;
}

void gds_log_once(std::atomic<int>& n, const char* fmt, unsigned long long a, unsigned long long b = 0) {
    if (n.fetch_add(1, std::memory_order_relaxed) < 4) host_log(fmt, a, b);
}

// The stores back out of GDS that follow a dispatch (gx_dispatch_post), up to
// the next dispatch, draw or load into GDS: each GDS dword's word.
void gds_stores_after(const std::uint32_t* p, std::uint64_t (&home)[kGdsDwords]) {
    const auto* end = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(t_cb_hi));
    for (int packets = 0; p && packets < 64 && p < end; ++packets) {
        const std::uint32_t hdr = p[0];
        if ((hdr >> 30) == 2) {
            ++p;
            continue;
        }
        if ((hdr >> 30) != 3) return;
        const std::uint32_t op = (hdr >> 8) & 0xff;
        const std::size_t cnt = ((hdr >> 16) & 0x3fff) + 1;
        if (p + 1 + cnt > end) return;
        const std::uint32_t* b = p + 1;
        switch (op) {
            case kDmaData: {
                if (cnt < 6) break;
                const unsigned src_sel = (b[0] >> 29) & 3, dst_sel = (b[0] >> 20) & 3;
                if (dst_sel == 1) return;  // the next dispatch's loads
                if (src_sel != 1 || (dst_sel != 0 && dst_sel != 3)) break;
                const std::uint32_t at = b[1] / 4, n = (b[5] & 0x1fffffu) / 4;
                const std::uint64_t dst = gpu_va(b[3], b[4]);
                for (std::uint32_t k = 0; k < n && at + k < kGdsDwords; ++k) home[at + k] = dst + 4ull * k;
                break;
            }
            case kDispatchDirect: case kDispatchIndirect: case kDrawIndex2: case kDrawIndexAuto: case kDrawIndexOffset2:
            case kDrawIndirect: case kDrawIndexIndirect: case kIndirectBuffer: case kIndirectBufferConst:
                return;
            case kNop:
                if (cnt >= 3 && (b[0] == kGxNativeMagic || b[0] == kGxDispatchMagic || b[0] == kGxYebisMagic || b[0] == kGxScaleformMagic))
                    return;
                break;
            default:
                break;
        }
        p += 1 + cnt;
    }
}

// The counter each GDS dword the dispatch may count in stands for: the word
// it is stored back to after the dispatch, else the word it was loaded from;
// started from a set count, another word's, or what the word holds.
void bind_gds_counters(GpuDispatch& d) {
    if (!host_gpu_gds_counters()) return;
    std::uint64_t home[kGdsDwords] = {};
    gds_stores_after(t_next_packet, home);
    int lo = -1, hi = -1;
    for (int k = 0; k < static_cast<int>(kGdsDwords); ++k) {
        if (t_gds[k].kind != GdsHeld::kUnknown || home[k]) {
            if (lo < 0) lo = k;
            hi = k;
        }
    }
    if (lo < 0) return;
    static std::atomic<int> wide{0}, valueless{0}, unloaded{0};
    if (hi - lo >= static_cast<int>(GpuDispatch::kGdsCounters)) {
        gds_log_once(wide, "gnm exec: a dispatch moves counters through GDS dwords %llu..%llu, more than one binds: the first 8 count",
                     static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi));
        hi = lo + static_cast<int>(GpuDispatch::kGdsCounters) - 1;
    }
    d.gds_base = static_cast<std::uint32_t>(lo);
    bool any = false;
    for (int k = lo; k <= hi; ++k) {
        GdsHeld& h = t_gds[k];
        GpuDispatch::GdsCounter& c = d.gds[k - lo];
        if (home[k]) g_gds_found_after.fetch_add(1, std::memory_order_relaxed);
        c.word = home[k] ? home[k] : h.kind == GdsHeld::kWord ? h.word : 0;
        if (!c.word) {
            if (h.kind == GdsHeld::kValue) {
                gds_log_once(valueless, "gnm exec: GDS dword %llu holds a count of %llu with no word after the dispatch to keep it in: it counts nothing",
                             static_cast<unsigned long long>(k), h.value);
            }
            continue;
        }
        if (h.kind == GdsHeld::kValue) {
            c.start = GpuDispatch::kGdsSet;
            c.value = h.value;
        } else if (h.kind == GdsHeld::kWord && h.word != c.word) {
            c.start = GpuDispatch::kGdsCopy;
            c.from = h.word;
        } else if (h.kind == GdsHeld::kUnknown) {
            gds_log_once(unloaded, "gnm exec: GDS dword %llu counts with nothing loaded into it: from what its word 0x%llx holds",
                         static_cast<unsigned long long>(k), static_cast<unsigned long long>(c.word));
        }
        note_counter_word(c.word);
        h.kind = GdsHeld::kWord;  // from here the dword holds this word's count: the store after folds
        h.word = c.word;
        any = true;
    }
    if (any) g_gds_dispatches.fetch_add(1, std::memory_order_relaxed);
}

// DMA_DATA into or out of GDS.
void gds_dma(const std::uint32_t* b, unsigned src_sel, unsigned dst_sel, std::size_t bytes) {
    static std::atomic<int> off_logs{0}, other_logs{0}, unknown_logs{0}, far_logs{0}, failed_logs{0};
    if (!host_gpu_gds_counters()) {
        gds_log_once(off_logs, "gnm exec: DMA_DATA through GDS (select %llu to %llu) dropped: GDS counters are off (BBHOST_GDS_COUNTERS=0)", src_sel,
                     dst_sel);
        return;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(bytes / 4);
    if (dst_sel == 1 && (src_sel == 0 || src_sel == 2 || src_sel == 3)) {
        const std::uint32_t at = b[3] / 4;  // a GDS byte offset
        const std::uint64_t src = gpu_va(b[1], b[2]);
        for (std::uint32_t k = 0; k < n; ++k) {
            if (at + k >= kGdsDwords) {
                gds_log_once(far_logs, "gnm exec: DMA_DATA into GDS dword %llu, past the %llu followed: dropped", at + k, kGdsDwords);
                break;
            }
            GdsHeld& h = t_gds[at + k];
            if (src_sel == 2) {
                h.kind = GdsHeld::kValue;
                h.value = b[1];
            } else {
                h.kind = GdsHeld::kWord;
                h.word = src + 4ull * k;
                note_counter_word(h.word);
            }
        }
        g_gds_loads.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (src_sel == 1 && (dst_sel == 0 || dst_sel == 3)) {
        const std::uint32_t at = b[1] / 4;
        const std::uint64_t dst = gpu_va(b[3], b[4]);
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::uint64_t to = dst + 4ull * k;
            if (at + k >= kGdsDwords) {
                gds_log_once(far_logs, "gnm exec: DMA_DATA out of GDS dword %llu, past the %llu followed: dropped", at + k, kGdsDwords);
                break;
            }
            GdsHeld& h = t_gds[at + k];
            note_counter_word(to);
            if (h.kind == GdsHeld::kWord && h.word == to) {
                g_gds_stores_folded.fetch_add(1, std::memory_order_relaxed);  // back where it came from: the word holds it
                continue;
            }
            bool ok = false;
            if (h.kind == GdsHeld::kWord) {
                ok = host_gpu_counter_copy(to, h.word, 4);
            } else if (h.kind == GdsHeld::kValue) {
                ok = host_gpu_counter_fill(to, h.value, 4);
            } else {
                gds_log_once(unknown_logs, "gnm exec: DMA_DATA out of GDS dword %llu, which nothing loaded, to 0x%llx: dropped",
                             at + k, static_cast<unsigned long long>(to));
                continue;
            }
            if (!ok) {
                gds_log_once(failed_logs, "gnm exec: DMA_DATA out of GDS dword %llu to 0x%llx could not go on the GPU's timeline: dropped", at + k,
                             static_cast<unsigned long long>(to));
                continue;
            }
            h.kind = GdsHeld::kWord;  // the dword's count is that word's now
            h.word = to;
            g_gds_stores_moved.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    gds_log_once(other_logs, "gnm exec: DMA_DATA from select %llu to select %llu through GDS is not a counter's move: dropped", src_sel, dst_sel);
}

void execute_packet(std::uint32_t op, const std::uint32_t* b, std::size_t n, int depth) {
    t_packet_va = reinterpret_cast<std::uintptr_t>(b) - 4;
    t_packet_op = op;
    t_next_packet = b + n;
    if (g_cp_census) census_packet(op, b, n);
    switch (op) {
        case kEventWriteEop:
            // [0] event, [1] addr_lo, [2] addr_hi[15:0] | data_sel[31:29] | int_sel[25:24], [3] data_lo, [4] data_hi
            if (n >= 5) {
                write_sel(gpu_va(b[1], b[2]), (b[2] >> 29) & 7, b[3], b[4]);
            }
            break;
        case kReleaseMem:
            // [0] event_cntl, [1] data_sel[31:29] | int_sel, [2] addr_lo, [3] addr_hi, [4] data_lo, [5] data_hi
            if (n >= 6) {
                write_sel(gpu_va(b[2], b[3]), (b[1] >> 29) & 7, b[4], b[5]);
            }
            break;
        case kEventWrite: {
            // The pixel-pipe counters behind an occlusion query. The game
            // samples them around a proxy it draws for a light - the moon -
            // and sizes that light's glare by how many pixels passed: two
            // PixelPipeStatDump events (57, index 1) write one counter per
            // pixel pipe, bit 63 marking it valid, and the game reads the
            // difference. Left alone, the memory kept whatever it held, and
            // the moon's glare flickered and swelled over the whole screen.
            const unsigned type = b[0] & 0x3f, index = (b[0] >> 8) & 0xf;
            // BBHOST_ZPASS_TRACE=1: the draws each pair of samples brackets -
            // the game samples the counters around a proxy it draws for a
            // light, and those draws are meant to be measured, not seen.
            static const bool zpass_trace = [] {
                const char* e = std::getenv("BBHOST_ZPASS_TRACE");
                return e && e[0] == '1';
            }();
            if (zpass_trace && index == kEventIndexZpassDone) {
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1) < 40) {
                    host_log("zpass: event type %u at draw %llu", type, static_cast<unsigned long long>(g_draws.load()));
                }
            }
            if (type == kPixelPipeStatDump && index == kEventIndexZpassDone && n >= 3) {
                const std::uint64_t va = gpu_va(b[1], b[2]);
                // The real counters (host/occlusion.cpp): the draws after the
                // dump are measured by Vulkan occlusion queries and the GPU
                // writes the counts here once it has run them, before the
                // labels that follow. Nothing is written now - the game zeroed
                // the block and reads it as not ready until then.
                if (host_gpu_zpass_dump(va)) break;
                // Per pixel pipe: [this sample, the next one], 16 bytes apart.
                constexpr int kPixelPipes = 8;  // 16 on a Neo; the game reads as many as it wrote
                constexpr std::uint64_t kValid = 1ull << 63;
                if (hle_kernel_va_mapped(va, kPixelPipes * 16)) {
                    // BBHOST_ZPASS_COUNT: how many pixels each sample claims
                    // passed, for working out what the game does with them.
                    static const std::uint64_t step = [] {
                        const char* e = std::getenv("BBHOST_ZPASS_COUNT");
                        return e ? std::strtoull(e, nullptr, 0) : 0x2ffffffull;
                    }();
                    static std::atomic<std::uint64_t> counter{0};
                    // Every pipe reports the same count, and a sample is
                    // ahead of the one before it. HLE fake: the pixels a draw
                    // covers are not counted, so a light behind a wall keeps
                    // its glare and the game culls nothing by its queries
                    // (the real counts are host_gpu_zpass_dump's, above).
                    const std::uint64_t now = counter.fetch_add(step, std::memory_order_relaxed) + step;
                    // Only the dump's own words, one per pipe, 16 bytes apart:
                    // the query's end dump lands at block+8 (Gnm's writer,
                    // 0x1477660), and writing a 128-byte image of value/zero
                    // pairs from there zeroed pipes 1-7's begin words and the
                    // 8 bytes after the block. Both readers (GX GetData
                    // 0x256a2f0, YEBIS 0x15db3c0) want bit 63 in all 16 words,
                    // so no query ever completed: GX culling entries stayed
                    // pending and YEBIS's luminance analysis took an
                    // uninitialised count.
                    const std::uint64_t word = now | kValid;
                    for (int i = 0; i < kPixelPipes; ++i) store(va + 16 * static_cast<std::uint64_t>(i), &word, sizeof(word));
                    // With BBHOST_ZPASS_TRACE an end dump (at block+8) prints the
                    // block: a query the game can finish has bit 63 in all 16.
                    if (zpass_trace && (va & 15) == 8) {
                        static std::atomic<int> blocks{0};
                        if (blocks.fetch_add(1) < 12) {
                            const auto* q = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(va - 8));
                            int valid = 0;
                            for (int i = 0; i < kPixelPipes * 2; ++i) valid += (q[i] & kValid) != 0;
                            host_log("zpass: block 0x%llx: %d of 16 words valid, pipe 0 begin 0x%llx end 0x%llx",
                                     static_cast<unsigned long long>(va - 8), valid, static_cast<unsigned long long>(q[0]),
                                     static_cast<unsigned long long>(q[1]));
                        }
                    }
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) == 0) {
                        host_log("HLE fake: the pixel-pipe counters an occlusion query samples are a rising count, not the pixels drawn "
                                 "(first dump to 0x%llx; BBHOST_OCCLUSION=0, or the block is not in imported memory)",
                                 static_cast<unsigned long long>(va));
                    }
                }
            }
            break;
        }
        case kSetPredication: {
            // [0] start address lo (16-byte aligned), [1] address hi[7:0] |
            // PRED_BOOL[8] (1: draw if visible) | HINT[12] | PRED_OP[18:16]
            // (0 clear, 1 ZPASS, 2 primitive count) | CONTINUE[31]. Gnm's
            // setZPassPredicationEnable/Disable; the draws between are GX
            // tokens, not packets carrying the predicate bit, so the span
            // between a set and its clear is what is predicated
            // (host/occlusion.cpp). BBHOST_PREDICATION=0 ignores it.
            if (n < 2) break;
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 4) {
                host_log("gnm exec: SET_PREDICATION block 0x%llx op %u draw-if-visible %u hint %u continue %u",
                         static_cast<unsigned long long>(gpu_va(b[0] & ~0xfu, b[1] & 0xff)), (b[1] >> 16) & 7, (b[1] >> 8) & 1,
                         (b[1] >> 12) & 1, b[1] >> 31);
            }
            host_gpu_set_predication(gpu_va(b[0] & ~0xfu, b[1] & 0xff), (b[1] >> 16) & 7, ((b[1] >> 8) & 1) != 0, (b[1] >> 31) != 0);
            break;
        }
        case kWriteConstRam: {
            // [0] byte offset into const RAM, [1..] payload
            if (n < 2) break;
            const std::uint32_t off = b[0] & 0xffff;
            const std::size_t bytes = static_cast<std::size_t>(n - 1) * 4;
            if (off + bytes <= kConstRamBytes) {
                std::memcpy(t_const_ram + off, &b[1], bytes);
                note_ce(false, 0, off, bytes, t_const_ram + off);
            }
            break;
        }
        case kDumpConstRam: {
            // [0] byte offset, [1] dwords, [2] addr_lo, [3] addr_hi
            if (n < 4) break;
            const std::uint32_t off = b[0] & 0xffff;
            const std::size_t bytes = static_cast<std::size_t>(b[1] & 0x7fff) * 4;
            if (!bytes || off + bytes > kConstRamBytes) break;
            const std::uint64_t dst = gpu_va(b[2], b[3]);
            // Does the constant engine fill holes in the draw stream itself?
            warn_if_over_command_buffer("const-RAM dump", dst, bytes);
            note_ce(true, dst, off, bytes, t_const_ram + off);
            // The native binder replaces this round trip with a copy straight
            // out of YEBIS's resource array, which rests on constant RAM being
            // a mirror of it. BBHOST_YEBIS_BIND_VERIFY=1 checks that here.
            if (yebis_bind::verifying()) {
                yebis_bind::check_dump(off, t_const_ram + off, static_cast<std::uint32_t>(bytes));
            }
            cp_write(dst, t_const_ram + off, bytes);
            break;
        }
        case kIncrementCeCounter:
            ++t_ce_count;
            break;
        case kIncrementDeCounter:
            ++t_de_count;
            break;
        case kWaitOnCeCounter:
            // The draw engine waits for the constant engine to get ahead: pull
            // the CE stream forward until it does, or until it blocks on us.
            while (t_ce_count <= t_de_count && ce_step()) {
            }
            break;
        case kWaitOnDeCounterDiff:
            break;  // handled by ce_step() in the CE stream
        case kEventWriteEos: {
            // [0] event, [1] addr_lo, [2] addr_hi[15:0] | cmd[31:29], [3] data.
            // cmd 2 (signal fence) writes the 32-bit data when the event
            // retires; that is how the engine signals "this flush is done" for
            // the CPU-side wait in its resource-map path, so dropping it hangs
            // the game on "== Stall during rendering at flush N". cmd 1 stores
            // GDS contents, which we have no GDS for.
            if (n < 4) break;
            const unsigned cmd = (b[2] >> 29) & 7;
            if (cmd == 2) {
                cp_write(gpu_va(b[1], b[2]), &b[3], 4);
            } else if (cmd == 1) {
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1) < 4) {
                    host_log("HLE fake: EVENT_WRITE_EOS GDS store to 0x%llx ignored (no GDS)",
                             static_cast<unsigned long long>(gpu_va(b[1], b[2])));
                }
            }
            break;
        }
        case kWriteData: {
            // [0] dst_sel[11:8] (5 = memory), wr_confirm, [1] addr_lo, [2] addr_hi, [3..] payload
            if (n >= 4 && (((b[0] >> 8) & 0xf) == 5 || ((b[0] >> 8) & 0xf) == 1)) {
                // The CP writes this itself, ahead of the shader pipeline,
                // as on hardware (the game orders it against GPU work with
                // EOP labels, which are GPU-side here).
                const std::uint64_t va = gpu_va(b[1], b[2]);
                // Into a GDS counter's word: on the GPU's timeline, after the
                // dispatches before it that count in the word.
                if (host_gpu_gds_counters() && covers_counter_word(va, (n - 3) * 4)) {
                    bool ok = true;
                    for (std::size_t k = 0; k + 3 < n && ok; ++k) ok = host_gpu_counter_fill(va + 4 * k, b[3 + k], 4);
                    if (ok) {
                        g_counter_writes.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
                store(va, &b[3], (n - 3) * 4);
            }
            break;
        }
        case kDmaData: {
            // [0] cntl: src_sel[30:29] (0 mem, 1 GDS, 2 data, 3 mem through L2), dst_sel[21:20] (0 mem, 1 GDS,
            // 3 mem through L2); [1] src_lo/data, [2] src_hi, [3] dst_lo, [4] dst_hi, [5] size[20:0] bytes
            if (n >= 6) {
                const unsigned src_sel = (b[0] >> 29) & 3;
                const unsigned dst_sel = (b[0] >> 20) & 3;
                const std::size_t bytes = b[5] & 0x1fffffu;
                const std::uint64_t dst = gpu_va(b[3], b[4]);
                if (bytes == 0) break;
                if (src_sel == 1 || dst_sel == 1) {
                    gds_dma(b, src_sel, dst_sel, bytes);
                    break;
                }
                if (dst_sel != 0 && dst_sel != 3) {
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) < 4) host_log("gnm exec: DMA_DATA to select %u (reserved) dropped", dst_sel);
                    break;
                }
                // Into or out of a GDS counter's word: on the GPU's timeline,
                // where the word's value is (CopyStructureCount, a reset).
                if (host_gpu_gds_counters() && (covers_counter_word(dst, bytes) || (src_sel != 2 && covers_counter_word(gpu_va(b[1], b[2]), bytes)))) {
                    const bool ok = src_sel == 2 ? host_gpu_counter_fill(dst, b[1], bytes) : host_gpu_counter_copy(dst, gpu_va(b[1], b[2]), bytes);
                    if (ok) {
                        (src_sel == 2 ? g_counter_writes : g_counter_reads).fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) < 4) {
                        host_log("gnm exec: DMA_DATA with a GDS counter's word (0x%llx <- 0x%llx, %zu bytes) could not go on the GPU's timeline: "
                                 "done at once", static_cast<unsigned long long>(dst), static_cast<unsigned long long>(gpu_va(b[1], b[2])), bytes);
                    }
                }
                // CP DMA runs on the CP, ahead of the shader pipeline, so
                // it is done here at once: the resource tables Gnmx's
                // constant update engine copies this way must be readable
                // when the following draw is recorded. A fill that covers
                // a render target clears the image instead (the image is
                // the target's storage here).
                if (src_sel == 2) {
                    warn_if_over_command_buffer("DMA fill", dst, bytes);
                    if (hle_kernel_va_mapped(dst, bytes)) {
                        record(dst, &b[1], bytes);
                        g_exec_writes.fetch_add(1);
                        static std::atomic<int> logs{0};
                        if (bytes >= (1u << 20) && logs.fetch_add(1) < 40) {
                            host_log("gnm exec: DMA fill 0x%llx %zu bytes with 0x%08x", static_cast<unsigned long long>(dst), bytes, b[1]);
                        }
                        if (!host_gpu_clear_target(dst, b[1], bytes)) {
                            auto* d = reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(dst));
                            for (std::size_t i = 0; i < bytes / 4; ++i) {
                                d[i] = b[1];
                            }
                        }
                    }
                } else if (src_sel == 0 || src_sel == 3) {
                    const std::uint64_t src = gpu_va(b[1], b[2]);
                    warn_if_over_command_buffer("DMA copy", dst, bytes);
                    if (hle_kernel_va_mapped(src, bytes) && hle_kernel_va_mapped(dst, bytes)) {
                        record(dst, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src)), bytes);
                        g_exec_writes.fetch_add(1);
                        std::memmove(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)),
                                     reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src)), bytes);
                    }
                }
            }
            break;
        }
        case kIndirectBuffer:
        case kIndirectBufferConst:
            // [0] addr_lo, [1] addr_hi, [2] size in dwords
            if (g_trace) {
                static std::atomic<int> iblogs{0};
                if ((iblogs.load(std::memory_order_relaxed) < 12 && iblogs.fetch_add(1) < 12)) {
                    host_log("gnm exec: IB lo=0x%08x hi=0x%08x size=0x%08x", b[0], n > 1 ? b[1] : 0u, n > 2 ? b[2] : 0u);
                }
            }
            if (n >= 3 && depth < 4) {
                const std::uint64_t va = gpu_va(b[0], b[1]);
                const std::size_t dw = b[2] & 0xfffffu;
                if (hle_kernel_va_mapped(va, dw * 4)) {
                    execute(reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va)), dw, depth + 1);
                }
            }
            break;
        case kWaitRegMem: {
            // [0] function[2:0] (0 always,1 <,2 <=,3 ==,4 !=,5 >=,6 >), mem_space[4]; [1] addr_lo / reg;
            // [2] addr_hi; [3] ref; [4] mask; [5] poll interval. The CP blocks here; so do we.
            if (n >= 5 && (b[0] & 0x10)) {
                const std::uint64_t va = gpu_va(b[1], b[2]);
                if (!hle_kernel_va_mapped(va, 4)) {
                    break;
                }
                const unsigned fn = b[0] & 7;
                const std::uint32_t ref = b[3];
                const std::uint32_t mask = b[4];
                auto* p = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
                const auto start = std::chrono::steady_clock::now();
                GpuPhaseTimer wait_timer(kPhaseWait);
                bool submitted = false;
                // A fence value is small (1, a flush index, a timestamp). A
                // reference that is itself a PM4 type-3 header means we are
                // looking at a wait whose tail is not what the header claimed -
                // the stream has a hole we stepped over - and no value will
                // ever match it. Do not spend the timeout on it.
                if ((ref >> 30) == 3 && (ref & 0xffff0000u) >= 0xc0000000u) {
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) < 6) {
                        host_log("gnm exec: WAIT_REG_MEM 0x%llx has a packet header (0x%08x) as its reference; the packet is malformed, not waiting",
                                 static_cast<unsigned long long>(va), ref);
                    }
                    g_wait_timeouts.fetch_add(1);
                    note_wait_timeout(va, ref, mask, fn);
                    break;
                }
                for (;;) {
                    const std::uint32_t v = __atomic_load_n(p, __ATOMIC_ACQUIRE) & mask;
                    bool ok;
                    switch (fn) {
                        case 1: ok = v < ref; break;
                        case 2: ok = v <= ref; break;
                        case 3: ok = v == ref; break;
                        case 4: ok = v != ref; break;
                        case 5: ok = v >= ref; break;
                        case 6: ok = v > ref; break;
                        default: ok = true; break;
                    }
                    if (ok) {
                        break;
                    }
                    {
                        // If the recorded GPU write already satisfies the wait,
                        // later GPU work is ordered behind it on the queue and
                        // the CP need not stall at all - not even for a submit.
                        // This is the common case during a world load (tens of
                        // thousands of waits a frame), and submitting for each
                        // one chopped the frame into three-draw command buffers.
                        bool pending_ok = false;
                        {
                            std::lock_guard<std::mutex> lk(g_pending_mu);
                            std::uint32_t pv = 0;
                            if (g_pending_labels.get(va, pv)) {
                                pv &= mask;
                                switch (fn) {
                                    case 1: pending_ok = pv < ref; break;
                                    case 2: pending_ok = pv <= ref; break;
                                    case 3: pending_ok = pv == ref; break;
                                    case 4: pending_ok = pv != ref; break;
                                    case 5: pending_ok = pv >= ref; break;
                                    case 6: pending_ok = pv > ref; break;
                                    default: pending_ok = true; break;
                                }
                            }
                        }
                        if (pending_ok) {
                            g_wait_pending.fetch_add(1);
                            break;
                        }
                    }
                    if (!submitted) {
                        // Nothing recorded covers it: the write may be sitting
                        // in the open command buffer, so get it on its way.
                        submitted = true;
                        host_gpu_submit();
                        continue;
                    }
                    // A label the GPU is going to write arrives in microseconds;
                    // waiting seconds for one only happens when it is never
                    // coming, and during a world load there are dozens of them.
                    // BBHOST_WAIT_MS overrides.
                    static const long wait_ms = [] {
                        const char* e = std::getenv("BBHOST_WAIT_MS");
                        const long v = e ? std::strtol(e, nullptr, 10) : 200;
                        return v > 0 ? v : 200;
                    }();
                    if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(wait_ms)) {
                        // Let the other queues finish first: a compute chunk
                        // queued behind us may write this very label, and its
                        // writes must land before the game recycles objects.
                        drain_other_queues(t_queue_id, 200);
                        const std::uint32_t v2 = __atomic_load_n(p, __ATOMIC_ACQUIRE) & mask;
                        bool ok2;
                        switch (fn) {
                            case 1: ok2 = v2 < ref; break;
                            case 2: ok2 = v2 <= ref; break;
                            case 3: ok2 = v2 == ref; break;
                            case 4: ok2 = v2 != ref; break;
                            case 5: ok2 = v2 >= ref; break;
                            case 6: ok2 = v2 > ref; break;
                            default: ok2 = true; break;
                        }
                        if (ok2) {
                            break;
                        }
                        // Out of patience: something we do not
                        // execute would have written it. Give up on it so the
                        // frame loop keeps going, and say so.
                        g_wait_timeouts.fetch_add(1);
                        note_wait_timeout(va, ref, mask, fn);
                        static std::atomic<int> logs{0};
                        if (logs.fetch_add(1) < 6) {
                            host_log("HLE fake: WAIT_REG_MEM 0x%llx fn=%u ref=0x%x mask=0x%x (value 0x%x) satisfied by host",
                                     static_cast<unsigned long long>(va), fn, ref, mask, v);
                            host_log("  packet: %08x %08x %08x %08x %08x %08x depth=%d", b[0], b[1], b[2], b[3], b[4],
                                     n > 5 ? b[5] : 0u, depth);
                            const std::uint32_t* m = p - 8;
                            host_log("  mem-32: %08x %08x %08x %08x %08x %08x %08x %08x", m[0], m[1], m[2], m[3], m[4],
                                     m[5], m[6], m[7]);
                            host_log("  mem+00: %08x %08x %08x %08x %08x %08x %08x %08x", p[0], p[1], p[2], p[3], p[4],
                                     p[5], p[6], p[7]);
                        }
                        // BBHOST_FORCE_LABELS=1 also writes the expected value so CPU
                        // pollers proceed; default leaves the label alone (garbage
                        // compute output must not look "ready").
                        static const bool force_labels = [] {
                            const char* e = std::getenv("BBHOST_FORCE_LABELS");
                            return e && e[0] == '1';
                        }();
                        if (force_labels) {
                            if (fn == 3 || fn == 5 || fn == 2) {
                                __atomic_store_n(const_cast<std::uint32_t*>(p), ref, __ATOMIC_RELEASE);
                            } else if (fn == 6) {
                                __atomic_store_n(const_cast<std::uint32_t*>(p), ref + 1, __ATOMIC_RELEASE);
                            }
                        }
                        break;
                    }
                    host_sleep_us(20);
                }
            }
            break;
        }
        case kSetShReg:
            // The SH registers stay while the dispatch packets run: a dispatch
            // takes its program, thread counts and user data from them.
            if (!cp_registers() && !cp_dispatch_packets()) {
                g_register_dwords_skipped.fetch_add(n, std::memory_order_relaxed);
                break;
            }
            if (n >= 2 && b[0] < 0x400) {
                for (std::size_t k = 1; k < n && b[0] + k - 1 < 0x400; ++k) {
                    t_regs.sh[b[0] + k - 1] = b[k];
                    // Checks: which packet last wrote each compute user-data slot.
                    const std::uint32_t slot = b[0] + static_cast<std::uint32_t>(k) - 1;
                    if (slot >= 0x240 && slot < 0x250) {
                        t_cs_writers[2 * (slot - 0x240)] = b[-1];
                        t_cs_writers[2 * (slot - 0x240) + 1] = t_packet_va >= 8 && hle_kernel_va_mapped(t_packet_va - 8, 8)
                                                                  ? reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(t_packet_va - 8))[1]
                                                                  : 0;
                    }
                }
            }
            break;
        case kSetContextReg:
            if (!cp_registers()) {
                g_register_dwords_skipped.fetch_add(n, std::memory_order_relaxed);
                break;
            }
            if (n >= 2 && b[0] < 0x1000) {
                for (std::size_t k = 1; k < n && b[0] + k - 1 < 0x1000; ++k) t_regs.ctx[b[0] + k - 1] = b[k];
            }
            break;
        case kSetUconfigReg:
            if (!cp_registers()) {
                g_register_dwords_skipped.fetch_add(n, std::memory_order_relaxed);
                break;
            }
            if (n >= 2 && b[0] < 0x400) {
                for (std::size_t k = 1; k < n && b[0] + k - 1 < 0x400; ++k) t_regs.uconfig[b[0] + k - 1] = b[k];
            }
            break;
        case kIndexType:
            if (n >= 1) t_index_type = b[0];
            break;
        case kNumInstances:
            if (n >= 1) t_num_instances = b[0];
            break;
        case kIndexBase:
            if (n >= 2) t_index_base = gpu_va(b[0], b[1]);
            break;
        case kIndexBufferSize:
            if (n >= 1) t_index_size = b[0];
            break;
        case kDrawIndex2:
            // [0] max_size, [1] index_base_lo, [2] index_base_hi, [3] index_count, [4] draw_initiator
            if (n >= 5) {
                if (t_yebis_pending) {
                    t_yebis_pending = false;
                    hle_gx_yebis_drew(b[3], gpu_va(b[1], b[2]), t_yebis_want);
                    hle_gx_yebis_check(t_regs.sh);
                }
                if (drop_draw_packet(kDropIndex2, b[3])) break;
                trace_draw("DRAW_INDEX_2", b[3], gpu_va(b[1], b[2]));
            }
            break;
        case kDrawIndexAuto:
            // [0] index_count, [1] draw_initiator
            if (n >= 2 && !drop_draw_packet(kDropAuto, b[0])) trace_draw("DRAW_INDEX_AUTO", b[0], 0);
            break;
        case kDrawIndexOffset2:
            // [0] max_size, [1] index_offset, [2] index_count, [3] draw_initiator
            if (n >= 4) {
                if (g_trace_draws) {
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) < 12) {
                        const std::uint64_t base = t_index_base;
                        const auto* p16 = reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(base));
                        host_log("draw_index_offset_2: base=0x%llx size=%u offset=%u count=%u type=%u max=%u; u16 at base: %u %u %u %u %u %u %u %u %u %u %u %u",
                                 static_cast<unsigned long long>(base), t_index_size, b[1], b[2], t_index_type, b[0],
                                 p16[0], p16[1], p16[2], p16[3], p16[4], p16[5], p16[6], p16[7], p16[8], p16[9], p16[10], p16[11]);
                    }
                }
                if (drop_draw_packet(kDropOffset2, b[2])) break;
                trace_draw("DRAW_INDEX_OFFSET_2", b[2], t_index_base + static_cast<std::uint64_t>(b[1]) * ((t_index_type & 1) ? 4 : 2));
            }
            break;
        case kDispatchDirect:
            // [0] dim_x, [1] dim_y, [2] dim_z, [3] dispatch_initiator
            if (n >= 3) run_dispatch("DISPATCH_DIRECT", b[0], b[1], b[2]);
            break;
        case kSetBase:
            // [0] base_index[3:0], [1] addr_lo, [2] addr_hi[15:0]. Index 1 is
            // the indirect draw/dispatch argument table.
            if (n >= 3) t_indirect_base = gpu_va(b[1], b[2]);
            break;
        case kDispatchIndirect: {
            // [0] byte offset from the SET_BASE table, [1] dispatch_initiator.
            // The dimensions come from memory: {x, y, z}.
            if (n < 1 || !t_indirect_base) break;
            const std::uint64_t args = t_indirect_base + b[0];
            if (!hle_kernel_va_mapped(args, 12)) break;
            const auto* a = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(args));
            indirect_log("DISPATCH_INDIRECT", args, a, 3);
            run_dispatch("DISPATCH_INDIRECT", a[0], a[1], a[2], args);
            break;
        }
        case kDrawIndirect: {
            // [0] byte offset, [1] base_vtx SGPR, [2] start_inst SGPR, [3] initiator.
            // Arguments in memory: {vertex_count, instances, start_vertex, start_instance}.
            if (n < 4 || !t_indirect_base) break;
            const std::uint64_t args = t_indirect_base + b[0];
            if (!hle_kernel_va_mapped(args, 16)) break;
            const auto* a = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(args));
            indirect_log("DRAW_INDIRECT", args, a, 4);
            if (drop_draw_packet(kDropIndirect, a[0])) break;
            // The CP writes the two locations into the VS user data; a
            // location of 0 means none (GX passes 0 for both).
            if ((b[1] & 0xffff) && (b[1] & 0xffff) < 16) t_regs.sh[0x4C + (b[1] & 0xffff)] = a[2];
            if ((b[2] & 0xffff) && (b[2] & 0xffff) < 16) t_regs.sh[0x4C + (b[2] & 0xffff)] = a[3];
            trace_draw("DRAW_INDIRECT", a[0], 0, a[1], args);
            break;
        }
        case kDrawIndexIndirect: {
            // Arguments: {index_count, instances, start_index, base_vertex, start_instance}.
            if (n < 4 || !t_indirect_base) break;
            const std::uint64_t args = t_indirect_base + b[0];
            if (!hle_kernel_va_mapped(args, 20)) break;
            const auto* a = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(args));
            indirect_log("DRAW_INDEX_INDIRECT", args, a, 5);
            if (drop_draw_packet(kDropIndexIndirect, a[0])) break;
            if ((b[1] & 0xffff) && (b[1] & 0xffff) < 16) t_regs.sh[0x4C + (b[1] & 0xffff)] = a[3];
            if ((b[2] & 0xffff) && (b[2] & 0xffff) < 16) t_regs.sh[0x4C + (b[2] & 0xffff)] = a[4];
            // firstIndex lives in the GPU args; bind the whole index buffer.
            trace_draw("DRAW_INDEX_INDIRECT", a[0], t_index_base, a[1], args);
            break;
        }
        case kNop:
            // BBHOST_GX_NATIVE: a GX draw's host-draw token.
            if (n >= 3 && b[0] == kGxNativeMagic) {
                g_token_nops.fetch_add(1, std::memory_order_relaxed);
                lookahead_tokens(b + n);
                const GpuDrawInputs* in = hle_gx_native_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32), t_regs.sh, t_regs.ctx);
                if (in) native_draw(in, "native");
                hle_gx_native_release();
            }
            // A YEBIS draw's token. With the flip on the method wrote no draw
            // packet and the CP draws here; otherwise the packet comes next.
            if (n >= 3 && b[0] == kGxYebisMagic) {
                GxYebisDraw yd{};
                if (const GpuDrawInputs* yin = hle_gx_yebis_token_draw(b[1] | (static_cast<std::uint64_t>(b[2]) << 32), &yd)) {
                    g_draws.fetch_add(1);
                    GpuDraw d;
                    d.gx = yin;
                    d.gx_render = true;
                    // Targets, blend, depth and viewport: the wrapper's context state when the
                    // token is complete, else the register file's.
                    d.gx_partial = !yin->complete;
                    d.gx_token = true;
                    d.gx_token_kind = "yebis";
                    d.registers = cp_registers();
                    d.sh = t_regs.sh;
                    d.ctx = t_regs.ctx;
                    d.uconfig = t_regs.uconfig;
                    d.index_count = yd.index_count;
                    d.instance_count = 1;
                    d.index_va = yd.index_va;
                    d.index_type = yd.state_built ? yd.index_type : (t_index_type & 1);
                    if (yd.state_built && !yin->complete && cp_registers()) {
                        hle_gx_wrapper_compare(*yin, yd.index_count, yd.index_va, d.index_type, 1, t_regs.uconfig[0x242], t_regs.sh, t_regs.ctx);
                    }
                    if (!host_gpu_draw(d)) g_draw_failed.fetch_add(1);
                    break;
                }
            }
            if (n >= 3 && b[0] == kGxYebisMagic) {
                GxYebisDraw want{};
                t_yebis_pending = hle_gx_yebis_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32), &want) ? (t_yebis_want = want, true)
                                                                                                            : false;
            }
            // A GX buffer-copy pass's copy token.
            if (n >= 3 && b[0] == kGxCopyMagic) {
                g_copy_nops.fetch_add(1, std::memory_order_relaxed);
                hle_gx_copy_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32));
            }
            // A GX fill's token.
            if (n >= 3 && b[0] == kGxFillMagic) hle_gx_fill_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32), t_regs.ctx);
            if (n >= 3 && b[0] == kGxUploadMagic) hle_gx_upload_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32));
            if (n >= 3 && b[0] == kGxImageCopyMagic) hle_gx_image_copy_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32));
            // A dispatch token.
            if (n >= 3 && b[0] == kGxDispatchMagic) {
                if (const GpuDispatch* dp = hle_gx_dispatch_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32))) {
                    g_dispatches.fetch_add(1);
                    GpuDispatch dd = *dp;
                    if (dd.indirect_va) {
                        if (!hle_kernel_va_mapped(dd.indirect_va, 12)) break;
                        const auto* a = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(dd.indirect_va));
                        dd.dim[0] = a[0];
                        dd.dim[1] = a[1];
                        dd.dim[2] = a[2];
                        // What memory holds now is not what the GPU will read
                        // when the counts come from the GPU (a counter copied
                        // into them earlier in this stream): the dispatch stays
                        // indirect and the GPU decides, as the packet path does.
                        if (host_gpu_gds_counters()) {
                            for (std::uint32_t& v : dd.dim) v = v ? v : 1;
                        } else if (!a[0] || !a[1] || !a[2]) {
                            break;
                        }
                    }
                    if (dd.dim[0] && dd.dim[1] && dd.dim[2] && dd.threads[0]) {
                        bind_gds_counters(dd);
                        if (!host_gpu_dispatch(dd)) g_dispatch_failed.fetch_add(1);
                    }
                }
            }
            // A Scaleform HAL draw's token.
            if (n >= 3 && b[0] == kGxScaleformMagic) {
                if (const GpuDrawInputs* in = hle_gx_scaleform_token(b[1] | (static_cast<std::uint64_t>(b[2]) << 32))) native_draw(in, "scaleform");
            }
            break;
        default:
            break;
    }
}

// Is `op` a PM4 opcode this hardware defines? Used to re-lock the walk after a
// desync, so one unreadable dword does not cost us the rest of the buffer.
bool plausible_op(std::uint32_t op) {
    switch (op) {
        case 0x10: case 0x11: case 0x12: case 0x13: case 0x15: case 0x16: case 0x1e: case 0x20:
        case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x26: case 0x27: case 0x28:
        case 0x2a: case 0x2d: case 0x2f: case 0x33: case 0x34: case 0x35: case 0x37: case 0x3c:
        case 0x3f: case 0x40: case 0x42: case 0x43: case 0x46: case 0x47: case 0x48: case 0x49:
        case 0x50: case 0x58: case 0x60: case 0x68: case 0x69: case 0x6a: case 0x76: case 0x77:
        case 0x78: case 0x79: case 0x80: case 0x81: case 0x83: case 0x84: case 0x85: case 0x86:
        case 0x88:
            return true;
        default:
            return false;
    }
}

// The first offset at or after `from` where three type-3 packets chain cleanly.
// Bloodborne's command buffers occasionally contain a dword we cannot parse;
// dropping everything after it also drops the end-of-flush fence
// the game waits on, which deadlocks it, so re-lock instead.
std::size_t resync_from(const std::uint32_t* cb, std::size_t dwords, std::size_t from) {
    for (std::size_t i = from; i < dwords; ++i) {
        std::size_t j = i;
        int good = 0;
        while (good < 3 && j < dwords) {
            const std::uint32_t h = cb[j];
            if ((h >> 30) != 3 || !plausible_op((h >> 8) & 0xff)) break;
            const std::size_t n = ((h >> 16) & 0x3fff) + 1;
            if (j + 1 + n > dwords) break;
            j += 1 + n;
            ++good;
        }
        if (good >= 3 || (good >= 1 && j >= dwords)) return i;
    }
    return dwords;
}

// BBHOST_ARENA_MONITOR=1: a buffer the CP walks for a job, in a command-arena
// chunk that early free released after that job was submitted.
void note_stale_walk(std::uint64_t va, std::uint64_t bytes, int depth) {
    std::uint64_t chunk = 0;
    if (!bytes || !hle_gx_arena_overlap(va, bytes, &chunk)) return;
    std::uint64_t freed_event = 0;
    {
        std::lock_guard<std::mutex> lk(g_arena_mu);
        const auto it = g_arena_life.find(chunk);
        if (it == g_arena_life.end()) return;
        freed_event = it->second.last_free_event;
    }
    if (freed_event <= t_job_free_event) return;
    if (g_arena_stale_walks.fetch_add(1) < 16) {
        host_log("arena: CP walks 0x%llx (%llu bytes, depth %d) in chunk 0x%llx, early-freed (event %llu) after its job was "
                 "submitted (event %llu)",
                 static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes), depth, static_cast<unsigned long long>(chunk),
                 static_cast<unsigned long long>(freed_event), static_cast<unsigned long long>(t_job_free_event));
    }
}

// Per-packet statistics add up on the thread and reach the shared counters
// once per command buffer: three locked increments a packet (~9000 packets
// a frame) each waited for the stores before them to drain.
thread_local std::uint32_t t_op_counts[256];
thread_local std::uint64_t t_op_touched[4];

void execute(const std::uint32_t* cb, std::size_t dwords, int depth) {
    struct Counts {
        int depth;
        std::uint64_t packets = 0, dwords = 0;
        ~Counts() {
            if (packets) g_exec_packets.fetch_add(packets, std::memory_order_relaxed);
            if (dwords) g_exec_dwords.fetch_add(dwords, std::memory_order_relaxed);
            if (depth != 0) return;
            for (int w = 0; w < 4; ++w) {
                for (std::uint64_t m = t_op_touched[w]; m; m &= m - 1) {
                    const int op = w * 64 + __builtin_ctzll(m);
                    g_op_hist[op].fetch_add(t_op_counts[op], std::memory_order_relaxed);
                    t_op_counts[op] = 0;
                }
                t_op_touched[w] = 0;
            }
        }
    } counts{depth};
    const std::uint64_t saved_lo = t_cb_lo, saved_hi = t_cb_hi;
    t_cb_lo = reinterpret_cast<std::uintptr_t>(cb);
    t_cb_hi = t_cb_lo + dwords * 4;
    struct Restore {
        std::uint64_t lo, hi;
        ~Restore() {
            t_cb_lo = lo;
            t_cb_hi = hi;
        }
    } restore{saved_lo, saved_hi};
    if (t_job_free_event != ~0ull) note_stale_walk(reinterpret_cast<std::uintptr_t>(cb), static_cast<std::uint64_t>(dwords) * 4, depth);
    std::size_t i = 0;
    std::size_t packets = 0;
    std::size_t resyncs = 0, resync_from_i = 0, resync_to_i = 0;
    const char* stop = "end";
    std::uint32_t last_hdr = 0;
    std::size_t last_i = 0;
    while (i < dwords) {
        const std::uint32_t hdr = cb[i];
        const unsigned type = hdr >> 30;
        if (type == 3 && plausible_op((hdr >> 8) & 0xff) && i + 1 + (((hdr >> 16) & 0x3fff) + 1) <= dwords) {
            const std::uint32_t op = (hdr >> 8) & 0xff;
            const std::size_t n = ((hdr >> 16) & 0x3fff) + 1;
            t_cur_op = op;
            if (!g_exec_cost) {
                execute_packet(op, cb + i + 1, n, depth);
            } else {
                const auto t0 = std::chrono::steady_clock::now();
                execute_packet(op, cb + i + 1, n, depth);
                g_op_ns[op].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - t0).count()),
                                      std::memory_order_relaxed);
            }
            ++counts.packets;
            ++t_op_counts[op];
            t_op_touched[op >> 6] |= 1ull << (op & 63);
            counts.dwords += 1 + n;
            ++packets;
            last_hdr = hdr;
            last_i = i;
            i += 1 + n;
        } else if (type == 2) {
            i += 1;  // type-2 NOP
        } else {
            // Anything else - zero fill at the tail of a submission, or one of
            // the fence slots the engine reserves *inside* its command buffers
            // for the CP to write - is a single dword to step over. Guessing a
            // longer skip is worse than useless: a wrong boundary invents
            // packets, and an invented WAIT_REG_MEM costs two seconds each time
            // it is polled. Only count the non-zero ones; zero fill is normal.
            if (hdr) {
                ++resyncs;
                resync_from_i = i;
                resync_to_i = i + 1;
            }
            ++i;
        }
    }
    // A walk that ends anywhere but the end of the buffer means we misparsed a
    // packet and dropped everything after it - including the end-of-flush fence
    // the game waits on. Always say so.
    if (resyncs) {
        g_resyncs.fetch_add(resyncs);
        // The first buffers we had to re-lock, for offline analysis.
        static std::atomic<int> dumped_resync{0};
        const int which = dumped_resync.fetch_add(1);
        if (which < 3) {
            char path[64];
            std::snprintf(path, sizeof(path), "build/rs-%d.bin", which);
            if (FILE* f = std::fopen(path, "wb")) {
                std::fwrite(cb, 4, dwords, f);
                std::fclose(f);
                host_log("gnm exec: wrote %s (%zu dwords, guest 0x%llx, first re-lock %zu -> %zu)", path, dwords,
                         static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(cb)), resync_from_i, resync_to_i);
            }
        }
        static std::atomic<int> rlogs{0};
        if ((rlogs.load(std::memory_order_relaxed) < 8 && rlogs.fetch_add(1) < 8)) {
            host_log("gnm exec: stepped over %zu unparsable dword(s) in a %zu-dword buffer (last at %zu: 0x%08x)",
                     resyncs, dwords, resync_from_i, cb[resync_from_i]);
        }
    }
    if (stop[0] != 'e' || i < dwords) {
        static std::atomic<int> stops{0};
        if (stops.fetch_add(1) < 12) {
            host_log("gnm exec: walk stopped early (%s) at dword %zu of %zu, depth %d, last op 0x%02x at %zu, next 0x%08x",
                     stop, i, dwords, depth, (last_hdr >> 8) & 0xff, last_i, i < dwords ? cb[i] : 0u);
            // The whole buffer, so the desync can be found offline: the packet
            // we stop on is rarely the one we mis-sized.
            static std::atomic<int> dumped{0};
            const int which = dumped.fetch_add(1);
            if (which < 3) {
                char path[64];
                std::snprintf(path, sizeof(path), "build/cb-%d.bin", which);
                if (FILE* f = std::fopen(path, "wb")) {
                    std::fwrite(cb, 4, dwords, f);
                    std::fclose(f);
                    host_log("gnm exec: wrote %s (%zu dwords, guest 0x%llx)", path, dwords,
                             static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(cb)));
                }
            }
            const std::size_t from = last_i > 4 ? last_i - 4 : 0;
            for (std::size_t k = from; k < dwords && k < i + 12; k += 8) {
                host_log("    cb[%zu]: %08x %08x %08x %08x %08x %08x %08x %08x", k, cb[k], k + 1 < dwords ? cb[k + 1] : 0u,
                         k + 2 < dwords ? cb[k + 2] : 0u, k + 3 < dwords ? cb[k + 3] : 0u, k + 4 < dwords ? cb[k + 4] : 0u,
                         k + 5 < dwords ? cb[k + 5] : 0u, k + 6 < dwords ? cb[k + 6] : 0u, k + 7 < dwords ? cb[k + 7] : 0u);
            }
        }
    }
    if (g_trace) {
        static std::atomic<int> dumps{0};
        if (depth == 1 && dumps.fetch_add(1) < 3) {
            for (std::size_t k = 0; k < dwords && k < 160; k += 8) {
                host_log("  cb+%03zx: %08x %08x %08x %08x %08x %08x %08x %08x", k * 4, cb[k],
                         k + 1 < dwords ? cb[k + 1] : 0, k + 2 < dwords ? cb[k + 2] : 0, k + 3 < dwords ? cb[k + 3] : 0,
                         k + 4 < dwords ? cb[k + 4] : 0, k + 5 < dwords ? cb[k + 5] : 0, k + 6 < dwords ? cb[k + 6] : 0,
                         k + 7 < dwords ? cb[k + 7] : 0);
            }
        }
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 400) {
            host_log("gnm exec: cb=%p depth=%d dwords=%zu consumed=%zu packets=%zu stop=%s hdr0=0x%08x last=0x%08x@%zu at=0x%08x 0x%08x 0x%08x 0x%08x",
                     static_cast<const void*>(cb), depth, dwords, i, packets, stop, dwords ? cb[0] : 0u,
                     last_hdr, last_i, i < dwords ? cb[i] : 0u, i + 1 < dwords ? cb[i + 1] : 0u,
                     i + 2 < dwords ? cb[i + 2] : 0u, i + 3 < dwords ? cb[i + 3] : 0u);
        }
    }
}

}  // namespace

namespace {

// `trusted` buffers are our own snapshots, which hle_kernel_va_mapped() would
// reject: it only knows guest mappings.
void execute_one(const void* cb, std::size_t bytes, bool trusted = false) {
    GpuPhaseTimer timer(kPhaseExec);
    if (!cb || bytes < 4) {
        return;
    }
    if (!trusted && !hle_kernel_va_mapped(reinterpret_cast<std::uintptr_t>(cb), bytes)) {
        return;
    }
    execute(static_cast<const std::uint32_t*>(cb), bytes / 4, 0);
}

// The command processor: one host thread per hardware queue (graphics ring,
// each mapped compute ring) so a WAIT_REG_MEM on one queue never blocks the
// queue that would satisfy it, as on the real GPU.
struct Job {
    std::vector<std::pair<const void*, std::size_t>> cbs;
    std::vector<std::pair<const void*, std::size_t>> ce_cbs;  // constant engine, paired by index
    // BBHOST_CP_COPY=1: our copy of the buffers above, so the game rewriting
    // them before the command processor gets there cannot desync the walk.
    std::deque<std::vector<std::uint8_t>> owned;  // deque: push_back must not move the earlier buffers
    std::vector<std::pair<const void*, std::size_t>> guest;  // the originals, for the snapshot comparison
    std::function<void()> done;
    std::uint64_t reflex_frame_id = 0;  // immutable simulation ID captured on the submitting thread
    std::uint64_t free_event = 0;  // g_arena_free_event when submitted (BBHOST_ARENA_MONITOR)
};

std::vector<std::pair<const void*, std::size_t>> snapshot_cbs(const std::vector<std::pair<const void*, std::size_t>>& in,
                                                              std::deque<std::vector<std::uint8_t>>& owned) {
    std::vector<std::pair<const void*, std::size_t>> out;
    out.reserve(in.size());
    for (const auto& cb : in) {
        if (!cb.first || cb.second < 4 || !hle_kernel_va_mapped(reinterpret_cast<std::uintptr_t>(cb.first), cb.second)) {
            out.push_back(cb);
            continue;
        }
        owned.emplace_back(static_cast<const std::uint8_t*>(cb.first), static_cast<const std::uint8_t*>(cb.first) + cb.second);
        out.emplace_back(owned.back().data(), cb.second);
    }
    return out;
}
struct Queue {
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable drained;  // submit-side back pressure
    std::deque<Job> jobs;
    std::uint64_t submitted = 0;
    std::uint64_t retired = 0;
    bool started = false;
};
std::mutex g_queues_mu;
std::atomic<std::uint64_t> g_submitted_total{0}, g_retired_total{0};
std::unordered_map<int, Queue*> g_queues;

// BBHOST_CPUPROFILE=<file>:<first flip>:<last flip>: sample the graphics
// command-processor thread between those flips with gperftools. Run with
// LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE_PER_THREAD_TIMERS=1 and no
// CPUPROFILE, then `go tool pprof -top build/bbhost <file>`. Only this host
// thread may get a profiling timer: a process-wide SIGPROF can land in guest
// code while FS points at the guest TCB, and the handler's TLS use would
// corrupt it.
struct CpuProfile {
    std::string path;
    std::uint64_t first = 0, last = 0;
    int (*start)(const char*) = nullptr;
    void (*stop)() = nullptr;
    bool running = false, done = false;
};

CpuProfile cpu_profile_config() {
    CpuProfile p;
    const char* e = std::getenv("BBHOST_CPUPROFILE");
    if (!e) return p;
    const std::string spec(e);
    const std::size_t a = spec.find(':');
    const std::size_t b = a == std::string::npos ? a : spec.find(':', a + 1);
    if (b == std::string::npos) {
        host_log("cpuprofile: BBHOST_CPUPROFILE wants <file>:<first flip>:<last flip>");
        return p;
    }
    if (!std::getenv("CPUPROFILE_PER_THREAD_TIMERS") || std::getenv("CPUPROFILE")) {
        host_log("cpuprofile: needs CPUPROFILE_PER_THREAD_TIMERS=1 and no CPUPROFILE (guest threads must not be signalled)");
        return p;
    }
#if defined(_WIN32)
    void (*register_thread)() = nullptr;  // gperftools is a Linux affair
#else
    const auto register_thread = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "ProfilerRegisterThread"));
    p.start = reinterpret_cast<int (*)(const char*)>(dlsym(RTLD_DEFAULT, "ProfilerStart"));
    p.stop = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "ProfilerStop"));
#endif
    if (!register_thread || !p.start || !p.stop) {
        host_log("cpuprofile: libprofiler.so is not loaded (LD_PRELOAD it)");
        p.start = nullptr;
        return p;
    }
    register_thread();
    p.path = spec.substr(0, a);
    p.first = std::strtoull(spec.c_str() + a + 1, nullptr, 10);
    p.last = std::strtoull(spec.c_str() + b + 1, nullptr, 10);
    return p;
}

void cpu_profile_tick(CpuProfile& p) {
    if (!p.start || p.done) return;
    const std::uint64_t flip = hle_video_flip_count();
    if (!p.running && flip >= p.first) {
        p.running = p.start(p.path.c_str()) != 0;
        host_log("cpuprofile: %s at flip %llu -> %s", p.running ? "started" : "failed to start",
                 static_cast<unsigned long long>(flip), p.path.c_str());
        if (!p.running) p.done = true;
    } else if (p.running && flip >= p.last) {
        p.stop();
        p.running = false;
        p.done = true;
        host_log("cpuprofile: stopped at flip %llu", static_cast<unsigned long long>(flip));
    }
}

void cp_thread(Queue* q, int id) {
    t_queue_id = id;
    {
        char name[16];
        std::snprintf(name, sizeof(name), "bb-cp%d", id);
        host_thread_set_name(name);
        // Above the guest's threads on Windows: the game's frame waits on
        // this one, and a GX worker that took its core delayed every draw
        // behind it (core/host_clock.h, BBHOST_THREAD_PRIO=0).
        host_thread_set_class(HostThreadClass::GpuFeed, name);
    }
    CpuProfile profile = id == 0 ? cpu_profile_config() : CpuProfile{};
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(q->mu);
            q->cv.wait(lk, [q] { return !q->jobs.empty(); });
            job = std::move(q->jobs.front());
            q->jobs.pop_front();
        }
        host::reflex::set_cp_thread_frame_id(job.reflex_frame_id);
        cpu_profile_tick(profile);
        t_job_free_event = g_arena_monitor ? job.free_event : ~0ull;
        for (const auto& cb : job.cbs) note_command_buffer(reinterpret_cast<std::uintptr_t>(cb.first), cb.second);
        for (const auto& cb : job.ce_cbs) note_command_buffer(reinterpret_cast<std::uintptr_t>(cb.first), cb.second);
        for (std::size_t i = 0; i < job.cbs.size(); ++i) {
            // Arm the constant engine for this pair, then let the draw stream
            // pull it forward; anything the draws never waited for runs at the
            // end so its memory writes still land.
            t_ce_stack.clear();
            t_ce_count = t_de_count = 0;
            if (i < job.ce_cbs.size() && job.ce_cbs[i].first && job.ce_cbs[i].second >= 4 &&
                (!job.owned.empty() ||
                 hle_kernel_va_mapped(reinterpret_cast<std::uintptr_t>(job.ce_cbs[i].first), job.ce_cbs[i].second))) {
                t_ce_stack.push_back({static_cast<const std::uint32_t*>(job.ce_cbs[i].first), job.ce_cbs[i].second / 4});
            }
            execute_one(job.cbs[i].first, job.cbs[i].second, !job.owned.empty());
            // With a snapshot, compare it against guest memory afterwards: if
            // they differ, something rewrote the buffer while we walked it.
            if (!job.owned.empty() && i < job.guest.size() && job.guest[i].first) {
                const auto* live = static_cast<const std::uint32_t*>(job.guest[i].first);
                const auto* snap = static_cast<const std::uint32_t*>(job.cbs[i].first);
                const std::size_t dw = job.cbs[i].second / 4;
                if (hle_kernel_va_mapped(reinterpret_cast<std::uintptr_t>(live), job.cbs[i].second)) {
                    std::size_t diffs = 0, first = 0;
                    for (std::size_t k = 0; k < dw; ++k) {
                        if (live[k] != snap[k]) {
                            if (!diffs) first = k;
                            ++diffs;
                        }
                    }
                    if (diffs) {
                        static std::atomic<int> logs{0};
                        if (logs.fetch_add(1) < 8) {
                            host_log("gnm exec: %zu dword(s) of the %zu-dword buffer changed while we walked it; first at %zu (snapshot %08x, now %08x)",
                                     diffs, dw, first, snap[first], live[first]);
                        }
                    }
                }
            }
            while (ce_step()) {
            }
            t_ce_stack.clear();
        }
        apply_arena_frees();  // the job's buffers are all walked
        t_job_free_event = ~0ull;
        // Submit after every job: its labels - held back to the end of the
        // command buffer (host_gpu_mem_write) - then land when its own work
        // has run, as on the console. Recording on into one command buffer
        // while the game's submissions queue up held them back a frame or
        // more, and the GX layer's pool of blocks for draw resource tables,
        // freed by those labels, ran empty: 12,879 and 14,530 lazy walks in
        // two three-minute soaks, and a walk that found every block still
        // waiting was an allocation that failed - a draw whose tables pointed
        // at address 0 (black frames, black icons, particles pulled across the
        // screen: decomp/gx/block_reclaim.cpp). Submitting every job: none, at
        // no measurable cost (main-loop work 10.74 and 11.35 ms a frame
        // against 11.48 and 11.03, pinned). When this
        // merged (77e6dc5, 2026-09-12) each submission cost ~7 s per 300 world
        // flips; the recorder thread and the shadow's own command buffer have
        // since taken that away. BBHOST_SUBMIT_EVERY_JOB=0 merges again.
        static const bool every_job = [] {
            const char* e = std::getenv("BBHOST_SUBMIT_EVERY_JOB");
            return !(e && e[0] == '0');
        }();
        bool more_queued = false;
        if (!every_job) {
            std::lock_guard<std::mutex> lk(q->mu);
            more_queued = !q->jobs.empty();
        }
        if (every_job || !more_queued) host_gpu_submit();
        if (job.done) {
            job.done();
        }
        {
            std::lock_guard<std::mutex> lk(q->mu);
            ++q->retired;
            g_retired_total.fetch_add(1, std::memory_order_release);
        }
        host::reflex::set_cp_thread_frame_id(0);
        q->cv.notify_all();
        q->drained.notify_all();
    }
}

void drain_other_queues(int except_id, int max_ms) {
    std::vector<Queue*> qs;
    {
        std::lock_guard<std::mutex> lk(g_queues_mu);
        for (auto& kv : g_queues) {
            if (kv.first != except_id) {
                qs.push_back(kv.second);
            }
        }
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
    for (Queue* q : qs) {
        std::unique_lock<std::mutex> lk(q->mu);
        const std::uint64_t target = q->submitted;
        q->cv.wait_until(lk, deadline, [&] { return q->retired >= target; });
    }
}

Queue* queue_for(int id) {
    std::lock_guard<std::mutex> lk(g_queues_mu);
    auto it = g_queues.find(id);
    if (it != g_queues.end()) {
        return it->second;
    }
    auto* q = new Queue();
    g_queues[id] = q;
    return q;
}

}  // namespace

void hle_gnm_explain_table(std::uint64_t va, std::size_t bytes) {
    if (!g_ce_history || t_ce_history.empty()) return;
    const std::uint64_t newest = t_ce_seq;
    const std::uint64_t oldest = newest > kCeHistory ? newest - kCeHistory : 0;
    const CeRecord* dump = nullptr;
    for (std::uint64_t s = newest; s-- > oldest;) {
        const CeRecord& r = t_ce_history[s % kCeHistory];
        if (r.dump && va >= r.dst && va < r.dst + r.bytes) {
            dump = &r;
            break;
        }
    }
    if (!dump) {
        host_log("cehist: no DumpConstRam in the last %llu CE packets covers 0x%llx (now ce=%u de=%u)",
                 static_cast<unsigned long long>(newest - oldest), static_cast<unsigned long long>(va), t_ce_count, t_de_count);
        return;
    }
    const std::uint32_t lo = dump->off + static_cast<std::uint32_t>(va - dump->dst);
    const std::uint32_t hi = lo + static_cast<std::uint32_t>(bytes);
    host_log("cehist: 0x%llx came from DumpConstRam #%llu (%llu CE packets ago): const RAM +0x%x %u bytes -> 0x%llx, "
             "ce=%u de=%u at the dump, now ce=%u de=%u; this range is const RAM +0x%x..+0x%x",
             static_cast<unsigned long long>(va), static_cast<unsigned long long>(dump->seq),
             static_cast<unsigned long long>(newest - dump->seq), dump->off, dump->bytes,
             static_cast<unsigned long long>(dump->dst), dump->ce, dump->de, t_ce_count, t_de_count, lo, hi);
    int shown = 0;
    for (std::uint64_t s = newest; s-- > oldest && shown < 24;) {
        const CeRecord& r = t_ce_history[s % kCeHistory];
        if (r.off >= hi || r.off + r.bytes <= lo) continue;
        ++shown;
        host_log("cehist:   %s #%llu (%+lld vs that dump) +0x%x %u bytes ce=%u de=%u: %08x %08x %08x %08x",
                 r.dump ? "dump " : "write", static_cast<unsigned long long>(s),
                 static_cast<long long>(s) - static_cast<long long>(dump->seq), r.off, r.bytes, r.ce, r.de,
                 r.head[0], r.head[1], r.head[2], r.head[3]);
    }
}

void hle_gnm_submit_async(int queue_id, const std::vector<std::pair<const void*, std::size_t>>& cbs,
                          std::function<void()> done, const std::vector<std::pair<const void*, std::size_t>>& ce_cbs) {
    // BBHOST_SYNC_GNM=1 executes the graphics queue on the submitting thread
    // (debugging aid: no backlog between the game and the command processor).
    static const bool sync_gfx = [] {
        const char* e = std::getenv("BBHOST_SYNC_GNM");
        return e && e[0] == '1';
    }();
    const auto frame_id = host::reflex::current_sim_frame_id();
    if (sync_gfx && queue_id == 0) {
        const auto saved_frame = host::reflex::cp_thread_frame_id();
        host::reflex::set_cp_thread_frame_id(frame_id);
        const int saved = t_queue_id;
        t_queue_id = 0;
        for (std::size_t i = 0; i < cbs.size(); ++i) {
            t_ce_stack.clear();
            t_ce_count = t_de_count = 0;
            if (i < ce_cbs.size() && ce_cbs[i].first && ce_cbs[i].second >= 4 &&
                hle_kernel_va_mapped(reinterpret_cast<std::uintptr_t>(ce_cbs[i].first), ce_cbs[i].second)) {
                t_ce_stack.push_back({static_cast<const std::uint32_t*>(ce_cbs[i].first), ce_cbs[i].second / 4});
            }
            execute_one(cbs[i].first, cbs[i].second);
            while (ce_step()) {
            }
            t_ce_stack.clear();
        }
        apply_arena_frees();
        host_gpu_flush();
        t_queue_id = saved;
        if (done) {
            done();
        }
        host::reflex::set_cp_thread_frame_id(saved_frame);
        return;
    }
    Queue* q = queue_for(queue_id);
    std::unique_lock<std::mutex> lk(q->mu);
    if (!q->started) {
        q->started = true;
        std::thread(cp_thread, q, queue_id).detach();
    }
    // Walk a snapshot of the command buffers, not the guest's copy: the engine
    // puts its fences *inside* them (a slot every few dozen packets) and the
    // CP writes those as it goes; a fence recorded before a mid-walk submit
    // lands in the live buffer while we are still reading it, and one ahead
    // of the walk overwrote a packet header - an unparsable dword, a packet
    // lost (an ACQUIRE_MEM's header became the fence's 1), sometimes a 200 ms
    // wait on a label that never came, in most soaks.
    // Snapshot walks: 0 of those in two soaks, 1 each in the live pair, the
    // same frames; with GX drawing natively the stream is ~1-2 MB/s, so the
    // copy costs nothing measurable (it was off when every draw went through
    // PM4). BBHOST_CP_COPY=0 walks the guest memory directly.
    static const bool copy_cbs = [] {
        const char* e = std::getenv("BBHOST_CP_COPY");
        return !(e && e[0] == '0');
    }();
    if (copy_cbs) {
        Job job;
        job.cbs = snapshot_cbs(cbs, job.owned);
        job.ce_cbs = snapshot_cbs(ce_cbs, job.owned);
        job.guest = cbs;
        job.done = std::move(done);
        q->jobs.push_back(std::move(job));
    } else {
        q->jobs.push_back(Job{cbs, ce_cbs, {}, {}, std::move(done)});
    }
    q->jobs.back().reflex_frame_id = frame_id;
    q->jobs.back().free_event = g_arena_free_event.load(std::memory_order_relaxed);
    ++q->submitted;
    g_submitted_total.fetch_add(1, std::memory_order_release);
    q->cv.notify_all();
    // Do not let the game get far ahead of the command processor. On hardware
    // the GPU consumes a command buffer as it is submitted; here a software
    // walker can fall many submits behind, and the game - which recycles its
    // command buffers on its own schedule - then rewrites a buffer we have not
    // walked yet. That shows up as a PM4 desync mid-buffer ("walk stopped
    // early"), and everything after it is lost, including the end-of-flush
    // fence the game waits on: it then spins for a flush that can never
    // complete. That was with two submits of slack, which is also what bounded
    // the pipeline: the render thread spent 62% of its time here at a 60 fps
    // target, the game could not run a frame ahead of the command processor,
    // and neither was ever busy. Six (a frame and a half) took the world from
    // ~48 to ~52 fps at that target, with no walk stopping early in any run.
    // Ten: running through new areas, the command processor's bursts (new
    // textures and pipelines) stalled the render thread and with it the
    // game's main loop; a three-minute run/turn/attack soak went from 41-61
    // seconds under 58 fps to 12-15, and the frames presented from ~58.7 to
    // ~59.8 a second. The command processor never waits on presentation (the
    // present thread keeps the latest frame), so it drains the queue once a
    // burst is over: the extra latency is only during the burst. 16 did no
    // better.
    static const std::uint64_t max_backlog = [] {
        const char* e = std::getenv("BBHOST_CP_BACKLOG");
        const long v = e ? std::strtol(e, nullptr, 10) : 10;
        return static_cast<std::uint64_t>(v > 0 ? v : 1);
    }();
    MainThreadWait timed(0);  // back-pressure from the command processor, for frame stats and BBHOST_WAIT_LOG
    q->drained.wait(lk, [q] { return q->submitted - q->retired <= max_backlog; });
}

void hle_gnm_cp_drain() {
    std::vector<Queue*> qs;
    {
        std::lock_guard<std::mutex> lk(g_queues_mu);
        for (auto& kv : g_queues) {
            qs.push_back(kv.second);
        }
    }
    for (Queue* q : qs) {
        std::unique_lock<std::mutex> lk(q->mu);
        const std::uint64_t target = q->submitted;
        q->cv.wait_for(lk, std::chrono::seconds(5), [&] { return q->retired >= target; });
    }
}

std::string hle_gnm_flush_reasons() {
    std::string out;
    if (const std::uint64_t t = g_wait_timeouts.load()) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), " wait-timeouts:%llu", static_cast<unsigned long long>(t));
        out += buf;
    }
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " waits-by-recorded-write:%llu", static_cast<unsigned long long>(g_wait_pending.load()));
        out += buf;
    }
    for (int op = 0; op < 256; ++op) {
        const std::uint64_t n = g_flush_by_op[op].load();
        if (n) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), " %02x:%llu", op, static_cast<unsigned long long>(n));
            out += buf;
        }
    }
    return out;
}

// Lock-free totals for the texture cache's retirements, which run on the
// command processor itself (the queue locks may be held there).
std::uint64_t hle_gnm_submitted_total() { return g_submitted_total.load(std::memory_order_acquire); }
std::uint64_t hle_gnm_retired_total() { return g_retired_total.load(std::memory_order_acquire); }

std::uint64_t hle_gnm_cp_backlog() {
    std::uint64_t total = 0;
    std::lock_guard<std::mutex> lk(g_queues_mu);
    for (auto& kv : g_queues) {
        std::lock_guard<std::mutex> ql(kv.second->mu);
        total += kv.second->submitted - kv.second->retired;
    }
    return total;
}

namespace {

// The known arena markers grouped into arenas with the state each marker holds
// now. One arena's chunks share their offset within a 64 KiB alignment (the
// two seen: ...018 and ...008 for 256 KiB chunks), and markers are learned out
// of order, so the low 16 bits group them; distance does not. "Other" values
// are chunks being filled: the in-use 1 is overwritten while a chunk is used.
// Caller holds g_arena_mu.
struct ArenaCount {
    std::uint64_t first = 0;
    unsigned markers = 0, free = 0, in_use = 0, retired = 0, other = 0;
};

std::vector<ArenaCount> arena_counts_locked() {
    std::map<std::uint64_t, ArenaCount> by_offset;
    for (const std::uint64_t m : g_arena_markers) {
        if (!hle_kernel_va_mapped(m, 8)) continue;
        ArenaCount& a = by_offset[m & 0xffff];
        if (!a.markers) a.first = m;
        ++a.markers;
        const std::uint64_t v = guest_u64(m);
        if (v == 0) ++a.free;
        else if (v == 1) ++a.in_use;
        else if (v == 2) ++a.retired;
        else ++a.other;
    }
    std::vector<ArenaCount> out;
    for (auto& [offset, a] : by_offset) out.push_back(a);
    return out;
}

std::uint64_t g_arena_flips = 0, g_arena_last_low = 0;
unsigned g_arena_min_free = ~0u;
int g_arena_low_logs = 0;

}  // namespace

void hle_gnm_arena_flip() {
    if (!g_arena_monitor) return;
    const std::uint64_t backlog = hle_gnm_cp_backlog();
    std::lock_guard<std::mutex> lk(g_arena_mu);
    ++g_arena_flips;
    for (const ArenaCount& a : arena_counts_locked()) {
        if (a.markers < 16) continue;  // too little of the arena learned to judge
        if (a.free < g_arena_min_free) g_arena_min_free = a.free;
        if (a.free * 8 >= a.markers || g_arena_flips - g_arena_last_low < 30 || g_arena_low_logs >= 200) continue;
        g_arena_last_low = g_arena_flips;
        ++g_arena_low_logs;
        host_log("arena: LOW at flip %llu: arena 0x%llx, %u markers known: %u free, %u in use, %u retired, %u other; "
                 "completions %llu, foreign writes %llu, cp-backlog %llu",
                 static_cast<unsigned long long>(g_arena_flips), static_cast<unsigned long long>(a.first), a.markers, a.free, a.in_use,
                 a.retired, a.other, static_cast<unsigned long long>(g_arena_completions.load()),
                 static_cast<unsigned long long>(g_arena_foreign.load()), static_cast<unsigned long long>(backlog));
    }
}

std::string hle_gnm_arena_report() {
    std::string out;
    char buf[160];
    if (const std::uint64_t waits = g_arena_acquire_waits.load()) {
        std::snprintf(buf, sizeof(buf), " acquires waited %llu (%llu ms, failed %llu);", static_cast<unsigned long long>(waits),
                      static_cast<unsigned long long>(g_arena_acquire_wait_us.load() / 1000),
                      static_cast<unsigned long long>(g_arena_acquire_failed.load()));
        out += buf;
    }
    if (const std::uint64_t frees = g_arena_early_frees.load()) {
        std::snprintf(buf, sizeof(buf), " early frees %llu;", static_cast<unsigned long long>(frees));
        out += buf;
    }
    if (!g_arena_monitor) return out;
    std::lock_guard<std::mutex> lk(g_arena_mu);
    for (const ArenaCount& a : arena_counts_locked()) {
        std::snprintf(buf, sizeof(buf), " 0x%llx: %u known, %u free, %u in use, %u retired, %u other;",
                      static_cast<unsigned long long>(a.first), a.markers, a.free, a.in_use, a.retired, a.other);
        out += buf;
    }
    std::snprintf(buf, sizeof(buf), " min free %u, completions %llu, foreign writes %llu, data writes %llu (%llu KiB)",
                  g_arena_min_free == ~0u ? 0u : g_arena_min_free, static_cast<unsigned long long>(g_arena_completions.load()),
                  static_cast<unsigned long long>(g_arena_foreign.load()), static_cast<unsigned long long>(g_arena_data_writes.load()),
                  static_cast<unsigned long long>(g_arena_data_bytes.load() / 1024));
    out += buf;
    char life[480];
    std::snprintf(life, sizeof(life), "; queued writes into chunks: value 1 %llu, value 4 %llu, other %llu; chunks handed out %llu, "
                  "with a queued value not in memory %llu, with a write in a submission not completed %llu; "
                  "CP walks of chunks freed after their job was submitted %llu",
                  static_cast<unsigned long long>(g_arena_queued_writes[0].load()), static_cast<unsigned long long>(g_arena_queued_writes[1].load()),
                  static_cast<unsigned long long>(g_arena_queued_writes[2].load()), static_cast<unsigned long long>(g_arena_handed_out.load()),
                  static_cast<unsigned long long>(g_arena_handed_out_unlanded.load()),
                  static_cast<unsigned long long>(g_arena_handed_out_queued.load()),
                  static_cast<unsigned long long>(g_arena_stale_walks.load()));
    out += life;
    g_arena_min_free = ~0u;
    return out;
}

namespace {

// The command arenas the host acquire has seen: what the renderer and the
// monitor test addresses against. An entry is written before the count that
// publishes it, so readers take no lock.
struct ArenaExtent {
    std::uint64_t allocator = 0, base = 0, end = 0, chunk_bytes = 0;
};
ArenaExtent g_arena_extents[8];
std::atomic<int> g_arena_extent_count{0};
std::mutex g_arena_extent_mu;

void note_arena_extent(const std::uint64_t* allocator) {
    const std::uint64_t id = reinterpret_cast<std::uintptr_t>(allocator);
    for (int i = 0, n = g_arena_extent_count.load(); i < n; ++i) {
        if (g_arena_extents[i].allocator == id) return;
    }
    std::lock_guard<std::mutex> lk(g_arena_extent_mu);
    const int n = g_arena_extent_count.load();
    for (int i = 0; i < n; ++i) {
        if (g_arena_extents[i].allocator == id) return;
    }
    if (n == 8 || !allocator[1] || !allocator[2]) return;
    const std::uint64_t chunk_bytes = allocator[1] * 8;
    g_arena_extents[n] = {id, allocator[3], allocator[3] + allocator[2] * chunk_bytes, chunk_bytes};
    g_arena_extent_count.store(n + 1);
    // BBHOST_ARENA_WATCH=1: a thread that scans this arena's markers every
    // 2 ms and logs each one that becomes anything but 0, 1 or 2, with what
    // it was before and where the frame and the host GPU were. The protocol
    // only ever writes those three; whoever writes anything else is what
    // leaks the chunk.
    if (const char* e = std::getenv("BBHOST_ARENA_WATCH"); e && e[0] == '1') {
        const std::uint64_t base = allocator[3], count = allocator[2];
        std::thread([base, count, chunk_bytes] {
            std::vector<std::uint64_t> last(count, 0);
            int logs = 0;
            while (logs < 200) {
                for (std::uint64_t i = 0; i < count; ++i) {
                    const std::uint64_t v = guest_u64(base + (i + 1) * chunk_bytes - 8);
                    if (v != last[i]) {
                        if (v > 2 && logs++ < 200) {
                            host_log("arena watch: chunk %llu marker 0x%llx: 0x%llx -> 0x%llx at flip %llu, host GPU "
                                     "completed %llu",
                                     static_cast<unsigned long long>(i),
                                     static_cast<unsigned long long>(base + (i + 1) * chunk_bytes - 8),
                                     static_cast<unsigned long long>(last[i]), static_cast<unsigned long long>(v),
                                     static_cast<unsigned long long>(hle_video_flip_count()),
                                     static_cast<unsigned long long>(host_gpu_completed_submits()));
                            if (logs < 20) host_gpu_explain_value(base + (i + 1) * chunk_bytes - 8, v);
                        }
                        last[i] = v;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }).detach();
    }
    host_log("arena: allocator 0x%llx: %llu chunks of %llu KiB at 0x%llx", static_cast<unsigned long long>(id),
             static_cast<unsigned long long>(allocator[2]), static_cast<unsigned long long>(chunk_bytes / 1024),
             static_cast<unsigned long long>(allocator[3]));
}

}  // namespace

bool hle_gx_arena_overlap(std::uint64_t va, std::uint64_t bytes, std::uint64_t* chunk_base) {
    for (int i = 0, n = g_arena_extent_count.load(); i < n; ++i) {
        const ArenaExtent& a = g_arena_extents[i];
        if (va < a.end && va + (bytes ? bytes : 1) > a.base) {
            if (chunk_base) *chunk_base = va > a.base ? a.base + (va - a.base) / a.chunk_bytes * a.chunk_bytes : a.base;
            return true;
        }
    }
    return false;
}

bool hle_gx_arena_marker(std::uint64_t va) {
    for (int i = 0, n = g_arena_extent_count.load(); i < n; ++i) {
        const ArenaExtent& a = g_arena_extents[i];
        if (va >= a.base && va + 8 <= a.end) return (va + 8 - a.base) % a.chunk_bytes == 0;
    }
    return false;
}

bool hle_gnm_arena_monitor() { return g_arena_monitor; }

// Whether [va, va+bytes) covers a chunk's marker. The first few host-GPU
// writes that do are logged by the caller: a marker is only ever written by
// the game (0 -> 1 -> 2) and by the zeroing EVENT_WRITE_EOP.
bool hle_gx_arena_covers_marker(std::uint64_t va, std::uint64_t bytes) {
    for (int i = 0, n = g_arena_extent_count.load(); i < n; ++i) {
        const ArenaExtent& a = g_arena_extents[i];
        if (va >= a.end || va + bytes <= a.base) continue;
        const std::uint64_t lo = va > a.base ? va : a.base;
        const std::uint64_t hi = va + bytes < a.end ? va + bytes : a.end;
        // The first marker at or after lo: base + k*chunk - 8.
        const std::uint64_t k = (lo - a.base + 8 + a.chunk_bytes - 1) / a.chunk_bytes;
        const std::uint64_t marker = a.base + k * a.chunk_bytes - 8;
        if (k && marker >= lo && marker + 8 <= hi) return true;
    }
    return false;
}

// The game's command-arena acquire (guest 0x2ad3b80), patched to this at load
// (runtime.cpp, 1.09 only; BBHOST_ARENA_WAIT=0 keeps the original). The scan is
// the eboot's: from the chunk after the last one taken (+0x20), claim the first
// marker holding 0 with a 0 -> 1 compare-and-swap and return 0; otherwise 1 when
// some chunk is retired (2) and waiting for its completion, 2 when none is.
// A retired chunk frees only when the host GPU executes the EVENT_WRITE_EOP
// recorded behind the work that may still read it, which lags further than on
// hardware, and several game threads take chunks in parallel. The refill
// callback (sub_2aaf230) cannot handle a failure (0x2ab0a98 writes through
// null), while sub_2aaf030 already retries with a yield: so a failed scan feeds
// the GPU and scans again, for up to 10 s.
// BBHOST_ARENA_MONITOR=1: a chunk just handed out, against the writes queued
// into it during its previous life (note_arena_queued).
void note_arena_handed_out(std::uint64_t chunk) {
    // Read before g_arena_mu: every submission before this has run on the GPU.
    const std::uint64_t completed = host_gpu_completed_submits();
    std::lock_guard<std::mutex> lk(g_arena_mu);
    g_arena_handed_out.fetch_add(1, std::memory_order_relaxed);
    auto it = g_arena_life.find(chunk);
    if (it == g_arena_life.end()) return;
    ArenaChunkLife& life = it->second;
    unsigned unlanded = 0, waiting = 0;
    ArenaQueuedWrite example{};
    std::uint64_t example_now = 0;
    for (const ArenaQueuedWrite& w : life.queued) {
        std::uint64_t now = 0;
        std::memcpy(&now, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(w.va)), w.bytes);
        if (now != w.value) ++unlanded;
        if (w.serial < completed) continue;
        if (!waiting) {
            example = w;
            example_now = now;
        }
        ++waiting;
    }
    if (unlanded) g_arena_handed_out_unlanded.fetch_add(1, std::memory_order_relaxed);
    if (waiting && g_arena_handed_out_queued.fetch_add(1) < 16) {
        const bool freed = life.freed.time_since_epoch().count() != 0;
        const double ms = freed ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - life.freed).count() : -1.0;
        host_log("arena: chunk 0x%llx handed out while %u of %zu CP writes into it wait in submissions not completed yet "
                 "(0x%llx <- %llu in submission %llu, %llu completed, memory holds %llu); %s %.1f ms ago",
                 static_cast<unsigned long long>(chunk), waiting, life.queued.size(), static_cast<unsigned long long>(example.va),
                 static_cast<unsigned long long>(example.value), static_cast<unsigned long long>(example.serial),
                 static_cast<unsigned long long>(completed), static_cast<unsigned long long>(example_now),
                 freed ? "early-freed" : "not early-freed; timing unknown", ms);
    }
    life.queued.clear();
    life.freed = {};
}

GUEST_ABI int hle_gx_arena_acquire(std::uint64_t* allocator, std::uint64_t* out) {
    note_arena_extent(allocator);
    auto scan = [allocator, out]() -> int {
        const std::uint64_t chunk = allocator[1], count = allocator[2], base = allocator[3];
        if (!count) return 2;
        const std::uint64_t first = allocator[4] + 1;
        int status = 2;
        for (std::uint64_t i = 0; i < count; ++i) {
            const std::uint64_t idx = (first + i) % count;
            auto* marker = reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(base + (idx * chunk + chunk - 1) * 8));
            const std::uint64_t v = *marker;
            if (v == 0) {
                if (__sync_bool_compare_and_swap(marker, std::uint64_t{0}, std::uint64_t{1})) {
                    allocator[4] = idx;
                    *out = base + idx * chunk * 8;
                    if (g_arena_monitor) note_arena_handed_out(*out);
                    return 0;
                }
            } else if (v == 2) {
                status = 1;
            }
        }
        return status;
    };
    int status = scan();
    if (status == 0) return 0;
    MainThreadWait timed(0);  // for frame stats and BBHOST_WAIT_LOG
    const auto start = std::chrono::steady_clock::now();
    do {
        host_gpu_submit();  // the completions the CP has recorded go to the GPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        status = scan();
    } while (status != 0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(10));
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    g_arena_acquire_waits.fetch_add(1);
    g_arena_acquire_wait_us.fetch_add(static_cast<std::uint64_t>(us));
    if (status != 0) g_arena_acquire_failed.fetch_add(1);
    // Every failure logs (up to 256); successes are sampled.
    static std::atomic<int> logs{0}, failure_logs{0};
    if (status != 0 ? (failure_logs.load(std::memory_order_relaxed) < 256 && failure_logs.fetch_add(1) < 256) : logs.fetch_add(1) < 32) {
        host_log("arena: acquire waited %.1f ms for a free chunk (allocator 0x%llx, %llu chunks): %s", static_cast<double>(us) / 1000.0,
                 static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(allocator)), static_cast<unsigned long long>(allocator[2]),
                 status == 0 ? "got one" : status == 1 ? "none after 10 s, some retired" : "none after 10 s, all in use");
    }
    return status;
}

void hle_gnm_execute(const void* cb, std::size_t bytes) { execute_one(cb, bytes); }

// Async-signal-unsafe on purpose: only called from the crash handler.
void hle_gnm_dump_recent_writes(std::uint64_t near_va, unsigned count) {
    const std::uint64_t total = g_rec_next.load();
    const std::uint64_t first = total > kRecs ? total - kRecs : 0;
    unsigned shown = 0;
    for (std::uint64_t i = total; i > first && shown < count; --i) {
        const WriteRec& r = g_recs[(i - 1) % kRecs];
        const bool close = near_va && r.va + r.bytes + 0x100 > near_va && r.va < near_va + 0x100;
        if (near_va && !close && shown >= count / 2) {
            continue;
        }
        host_log("  gnm write #%llu op=0x%02x 0x%llx <- 0x%llx (%u bytes)%s", static_cast<unsigned long long>(i - 1),
                 r.op, static_cast<unsigned long long>(r.va), static_cast<unsigned long long>(r.value), r.bytes,
                 close ? "  <-- near fault" : "");
        ++shown;
    }
}

// Crash diagnostics: every write still in the ring that overlaps [va, va + bytes)
// or wrote `value` (when nonzero), oldest first. Async-signal-unsafe on purpose.
void hle_gnm_find_writes(std::uint64_t va, std::uint64_t bytes, std::uint64_t value) {
    const std::uint64_t total = g_rec_next.load();
    const std::uint64_t first = total > kRecs ? total - kRecs : 0;
    unsigned shown = 0;
    for (std::uint64_t i = first; i < total && shown < 24; ++i) {
        const WriteRec& r = g_recs[i % kRecs];
        const bool overlaps = r.va < va + bytes && r.va + r.bytes > va;
        const bool same_value = value && r.value == value;
        if (!overlaps && !same_value) continue;
        host_log("  gnm write #%llu op=0x%02x 0x%llx <- 0x%llx (%u bytes)%s%s", static_cast<unsigned long long>(i), r.op,
                 static_cast<unsigned long long>(r.va), static_cast<unsigned long long>(r.value), r.bytes,
                 overlaps ? "  <-- over the node" : "", same_value ? "  <-- the node's first word" : "");
        ++shown;
    }
    host_log("  gnm writes searched: #%llu to #%llu for 0x%llx (%llu bytes) and value 0x%llx", static_cast<unsigned long long>(first),
             static_cast<unsigned long long>(total), static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes),
             static_cast<unsigned long long>(value));
}

std::string hle_gnm_op_window() {
    // The opcodes the command processor names (the constants at the top).
    static const std::pair<std::uint32_t, const char*> kNames[] = {
        {kNop, "nop"}, {kSetBase, "set-base"}, {kIndexBufferSize, "index-buffer-size"}, {kDispatchDirect, "dispatch-direct"},
        {kDispatchIndirect, "dispatch-indirect"}, {kDrawIndirect, "draw-indirect"}, {kDrawIndexIndirect, "draw-index-indirect"},
        {kIndexBase, "index-base"}, {kDrawIndex2, "draw-index-2"}, {kIndexType, "index-type"}, {kDrawIndexAuto, "draw-index-auto"},
        {kNumInstances, "num-instances"}, {kIndirectBufferConst, "indirect-buffer-const"}, {kDrawIndexOffset2, "draw-index-offset-2"},
        {kWriteData, "write-data"}, {kWaitRegMem, "wait-reg-mem"}, {kIndirectBuffer, "indirect-buffer"},
        {kEventWriteEop, "event-write-eop"}, {kEventWriteEos, "event-write-eos"}, {kReleaseMem, "release-mem"}, {kDmaData, "dma-data"},
        {kSetContextReg, "set-context-reg"}, {kSetShReg, "set-sh-reg"}, {kSetUconfigReg, "set-uconfig-reg"},
        {kWriteConstRam, "write-const-ram"}, {kDumpConstRam, "dump-const-ram"}, {kIncrementCeCounter, "increment-ce-counter"},
        {kIncrementDeCounter, "increment-de-counter"}, {kWaitOnCeCounter, "wait-on-ce-counter"},
        {kWaitOnDeCounterDiff, "wait-on-de-counter-diff"},
    };
    static std::uint64_t last[256] = {};  // the flip report is the only caller
    static std::uint64_t last_tokens = 0, last_dwords = 0, last_copies = 0;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> ops;  // (packets, opcode), most first
    std::uint64_t total = 0;
    for (std::uint32_t op = 0; op < 256; ++op) {
        const std::uint64_t now = g_op_hist[op].load(std::memory_order_relaxed);
        if (now != last[op]) ops.emplace_back(now - last[op], op);
        total += now - last[op];
        last[op] = now;
    }
    std::sort(ops.rbegin(), ops.rend());
    const std::uint64_t tokens = g_token_nops.load(std::memory_order_relaxed);
    const std::uint64_t dwords = g_exec_dwords.load(std::memory_order_relaxed);
    const std::uint64_t copies = g_copy_nops.load(std::memory_order_relaxed);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "total=%llu dwords=%llu native-draw-tokens=%llu copy-tokens=%llu", static_cast<unsigned long long>(total),
                  static_cast<unsigned long long>(dwords - last_dwords), static_cast<unsigned long long>(tokens - last_tokens),
                  static_cast<unsigned long long>(copies - last_copies));
    last_tokens = tokens;
    last_copies = copies;
    last_dwords = dwords;
    std::string out = buf;
    for (const auto& [packets, op] : ops) {
        const char* name = nullptr;
        for (const auto& [code, n] : kNames) {
            if (code == op) name = n;
        }
        if (name) {
            std::snprintf(buf, sizeof(buf), " %s=%llu", name, static_cast<unsigned long long>(packets));
        } else {
            std::snprintf(buf, sizeof(buf), " op-%02x=%llu", op, static_cast<unsigned long long>(packets));
        }
        out += buf;
    }
    return out;
}

void hle_gnm_exec_histogram() {
    host_log("gnm dispatches=%llu skipped=%llu draws=%llu draw-failures=%llu", static_cast<unsigned long long>(g_dispatches.load()),
             static_cast<unsigned long long>(g_dispatch_failed.load()), static_cast<unsigned long long>(g_draws.load()),
             static_cast<unsigned long long>(g_draw_failed.load()));
    host_log("gnm: indirect draws whose count read 0 on the CPU, drawn for the GPU to read its own (BBHOST_INDIRECT_ZERO=skip drops them): %llu",
             static_cast<unsigned long long>(g_indirect_zero_cpu.load()));
    if (host_gpu_gds_counters()) {
        std::size_t words = 0;
        {
            std::lock_guard<std::mutex> lk(g_counter_words_mu);
            words = g_counter_words.size();
        }
        host_log("gnm: GDS counters: %llu dispatches counted in words (%llu of their words named by the stores after them), %llu loads "
                 "into GDS, %llu stores back into the word loaded or counted in (left as they are) and %llu into another word on the GPU's "
                 "timeline, %zu counter words; %llu copies out of and %llu writes into them on the GPU's timeline",
                 static_cast<unsigned long long>(g_gds_dispatches.load()), static_cast<unsigned long long>(g_gds_found_after.load()),
                 static_cast<unsigned long long>(g_gds_loads.load()), static_cast<unsigned long long>(g_gds_stores_folded.load()),
                 static_cast<unsigned long long>(g_gds_stores_moved.load()), words, static_cast<unsigned long long>(g_counter_reads.load()),
                 static_cast<unsigned long long>(g_counter_writes.load()));
    }
    host_gpu_report();
    for (int op = 0; op < 256; ++op) {
        const std::uint64_t n = g_op_hist[op].load();
        if (n) {
            const std::uint64_t ns = g_op_ns[op].load();
            if (ns) {
                host_log("gnm op 0x%02x: %llu, %llu ms total, %llu ns each", op, static_cast<unsigned long long>(n),
                         static_cast<unsigned long long>(ns / 1000000), static_cast<unsigned long long>(ns / n));
            } else {
                host_log("gnm op 0x%02x: %llu", op, static_cast<unsigned long long>(n));
            }
        }
    }
}


void hle_gnm_warn_if_command_buffer(const char* what, std::uint64_t va, std::size_t bytes) {
    warn_if_over_command_buffer(what, va, bytes);
}

void hle_gnm_wait_report() {
    std::lock_guard<std::mutex> lk(g_addr_mu);
    host_log("hang: %zu address(es) had a wait give up; the worst, with how often we wrote them:", g_timeout_addrs.size());
    std::vector<std::pair<std::uint64_t, WaitInfo>> v(g_timeout_addrs.begin(), g_timeout_addrs.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    for (std::size_t i = 0; i < v.size() && i < 12; ++i) {
        const auto it = g_written_addrs.find(v[i].first);
        std::uint32_t now = 0;
        if (hle_kernel_va_mapped(v[i].first, 4)) {
            std::memcpy(&now, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(v[i].first)), 4);
        }
        host_log("  wait 0x%llx gave up %llu time(s) (fn=%u ref=0x%08x mask=0x%08x, value now 0x%08x); CP writes to it: %llu",
                 static_cast<unsigned long long>(v[i].first), static_cast<unsigned long long>(v[i].second.count),
                 v[i].second.fn, v[i].second.ref, v[i].second.mask, now,
                 static_cast<unsigned long long>(it == g_written_addrs.end() ? 0 : it->second));
    }
}

void hle_gnm_cp_report() {
    if (g_cp_census) {
        struct Row {
            std::uint32_t op;
            std::uint64_t packets, dwords;
        };
        std::vector<Row> rows;
        std::uint64_t total_packets = 0, total_dwords = 0;
        for (std::uint32_t op = 0; op < 0x100; ++op) {
            const std::uint64_t p = g_census_packets[op].load(), d = g_census_dwords[op].load();
            if (!p) continue;
            rows.push_back({op, p, d});
            total_packets += p;
            total_dwords += d;
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& c) { return a.dwords > c.dwords; });
        std::string by_op;
        for (const Row& r : rows) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), " op %02x: %llu packets %llu dwords;", r.op, static_cast<unsigned long long>(r.packets),
                          static_cast<unsigned long long>(r.dwords));
            by_op += buf;
        }
        host_log("cp-census: %llu packets, %llu dwords walked;%s", static_cast<unsigned long long>(total_packets),
                 static_cast<unsigned long long>(total_dwords), by_op.c_str());
        std::lock_guard<std::mutex> lk(g_census_mu);
        std::vector<std::pair<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>>> nops(g_census_nops.begin(), g_census_nops.end());
        std::sort(nops.begin(), nops.end(), [](const auto& a, const auto& c) { return a.second.second > c.second.second; });
        std::string by_magic;
        for (std::size_t i = 0; i < nops.size() && i < 16; ++i) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), " %08x: %llu x%llu dwords;", nops[i].first, static_cast<unsigned long long>(nops[i].second.first),
                          static_cast<unsigned long long>(nops[i].second.second));
            by_magic += buf;
        }
        host_log("cp-census: NOP payloads by magic:%s", by_magic.c_str());
        std::vector<std::pair<std::uint32_t, std::uint64_t>> regs;
        for (std::uint32_t r = 0; r < 0x1000; ++r) {
            if (const std::uint64_t d = g_census_ctx_reg[r].load()) regs.push_back({r, d});
        }
        std::sort(regs.begin(), regs.end(), [](const auto& a, const auto& c) { return a.second > c.second; });
        std::string by_reg;
        for (std::size_t i = 0; i < regs.size() && i < 20; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), " 0x%03x: %llu;", regs[i].first, static_cast<unsigned long long>(regs[i].second));
            by_reg += buf;
        }
        host_log("cp-census: context/user-config registers still written (dwords), %zu distinct:%s", regs.size(), by_reg.c_str());
    }
    std::string dropped;
    for (int k = 0; k < kDroppedDraws; ++k) {
        if (const std::uint64_t n = g_draw_packets_dropped[k].load()) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), " %s %llu;", kDroppedDrawName[k], static_cast<unsigned long long>(n));
            dropped += buf;
        }
    }
    host_log("cp: register file %s (%llu dwords of SET_*_REG skipped); draw packets %s%s; dispatch packets %s DISPATCH_DIRECT %llu, "
             "DISPATCH_INDIRECT %llu",
             cp_registers() ? "kept" : "not kept", static_cast<unsigned long long>(g_register_dwords_skipped.load()),
             cp_draw_packets() ? "drawn" : "not drawn:", dropped.empty() ? (cp_draw_packets() ? "" : " none") : dropped.c_str(),
             cp_dispatch_packets() ? "run; dropped" : "not run:", static_cast<unsigned long long>(g_dispatch_packets_dropped[0].load()),
             static_cast<unsigned long long>(g_dispatch_packets_dropped[1].load()));
    std::size_t labels = 0;
    {
        std::lock_guard<std::mutex> lk(g_pending_mu);
        labels = g_pending_labels.size();
    }
    host_log("cp: label values recorded for the GPU to write, by address: %zu addresses; %llu waits satisfied by one", labels,
             static_cast<unsigned long long>(g_wait_pending.load()));
}

void hle_gnm_label_report() {
    host_log("hang: CP backlog %llu jobs, %llu packets executed, %llu waits pending, %llu wait timeouts",
             static_cast<unsigned long long>(hle_gnm_cp_backlog()), static_cast<unsigned long long>(g_exec_packets.load()),
             static_cast<unsigned long long>(g_wait_pending.load()), static_cast<unsigned long long>(g_wait_timeouts.load()));
    if (g_trace_label) {
        std::uint64_t now = 0;
        if (hle_kernel_va_mapped(g_trace_label, 8)) {
            std::memcpy(&now, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g_trace_label)), 8);
        }
        host_log("hang: traced label 0x%llx: %llu CP writes, value now %llu",
                 static_cast<unsigned long long>(g_trace_label), static_cast<unsigned long long>(g_trace_label_writes.load()),
                 static_cast<unsigned long long>(now));
    }
    const std::uint64_t next = g_label_next.load();
    const std::uint64_t first = next > kLabelRing ? next - kLabelRing : 0;
    host_log("hang: %llu CP label writes so far; the last %llu:", static_cast<unsigned long long>(next),
             static_cast<unsigned long long>(next - first));
    for (std::uint64_t i = first; i < next; ++i) {
        const LabelWrite& w = g_label_ring[i % kLabelRing];
        host_log("  label 0x%llx <- %llu (%u bytes, op 0x%02x)", static_cast<unsigned long long>(w.va),
                 static_cast<unsigned long long>(w.value), w.bytes, w.op);
    }
}

std::uint64_t hle_gnm_draw_count() { return g_draws.load(std::memory_order_relaxed); }

void hle_gnm_cp_write(std::uint64_t va, const void* src, std::size_t n) { cp_write(va, src, n); }

void hle_gnm_exec_stats(std::uint64_t* packets, std::uint64_t* writes) {
    if (packets) {
        *packets = g_exec_packets.load();
    }
    if (writes) {
        *writes = g_exec_writes.load();
    }
}
