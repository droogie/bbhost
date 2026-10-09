#include "hle/common.h"

#include "core/portable.h"
#include "engine/graphics_patch.h"
#include "engine/rebirth.h"
#include "engine/frame_rate.h"
#include "hle/equeue.h"
#include "hle/fs.h"
#include "hle/hle.h"
#include "core/imports.h"
#include "hle/modules.h"
#include "host/frame_stats.h"
#include "host/gpu.h"
#include "hle/platform.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr unsigned kDrawInitDwords = 0x100;

void* g_tf_ring;
std::mutex g_gnm_mu;
int g_queue_next = 1;
struct ComputeQueue {
    std::uint32_t* ring = nullptr;
    unsigned size_dw = 0;
    std::uint32_t* read_ptr = nullptr;
    unsigned queued_dw = 0;  // segment end already handed to the CP thread
};
std::unordered_map<int, ComputeQueue> g_queues;

struct GnmEq {
    HostEqueue* eq = nullptr;
    int id = 0;
    void* udata = nullptr;
};
std::vector<GnmEq> g_gnm_eq;

void* tf_ring() {
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    if (!g_tf_ring) {
        g_tf_ring = nullptr;
#if defined(_WIN32)
        g_tf_ring = VirtualAlloc(nullptr, 0x20000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        g_tf_ring = mmap(nullptr, 0x20000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_tf_ring == MAP_FAILED) {
            g_tf_ring = nullptr;
        }
#endif
        if (g_tf_ring) {
            std::memset(g_tf_ring, 0, 0x20000);
        }
        host_log("sceGnm tessellation ring %p", g_tf_ring);
    }
    return g_tf_ring;
}

unsigned draw_init(std::uint32_t* cmd, unsigned n) {
    if (!cmd || n == 0) {
        return 0;
    }
    // Gnmx advances its write pointer by whatever we return, so every dword we
    // claim has to be covered by a packet. Zeroing the block and writing a
    // two-dword NOP left the rest as a hole in the command stream: the command
    // processor then reads packet payloads as headers, loses alignment, and the
    // end-of-flush fences after it never execute. One NOP over the whole
    // reservation, the same way the shader setters pad theirs.
    const unsigned used = n > kDrawInitDwords ? kDrawInitDwords : n;
    std::memset(cmd, 0, static_cast<std::size_t>(used) * 4);
    if (used == 1) {
        cmd[0] = 0x80000000u;  // a one-dword type-2 filler
    } else {
        cmd[0] = 0xC0001000u | ((static_cast<std::uint32_t>(used - 2) & 0x3fffu) << 16);
    }
    return used;
}

void post_gnm_eop() {
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    for (auto& e : g_gnm_eq) {
        if (!e.eq) {
            continue;
        }
        HostEvent ev{};
        ev.ident = static_cast<std::uint64_t>(e.id);
        ev.filter = -1;
        ev.data = 1;
        ev.udata = e.udata;
        equeue_post(e.eq, ev);
    }
}

GUEST_ABI void* hle_gnm_tf_ring() { return tf_ring(); }
// Razor GPU capture is not loaded on a PC host.
GUEST_ABI int hle_razor_not_loaded() { return 0; }
GUEST_ABI void hle_razor_capture(const char*) {}

}  // namespace

void hle_gnm_detach_equeue(HostEqueue* eq) {
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    for (auto& e : g_gnm_eq) {
        if (e.eq == eq) {
            e.eq = nullptr;
        }
    }
}

namespace {

GUEST_ABI unsigned hle_gnm_draw_init(std::uint32_t* cmd, unsigned n) { return draw_init(cmd, n); }
GUEST_ABI unsigned hle_gnm_dispatch_init(std::uint32_t* cmd, unsigned n) { return draw_init(cmd, n); }

// The Gnm shader setters write the stage's register packets into the
// caller's command buffer; Gnmx reserves a fixed number of dwords for each
// (measured at the eboot's call sites) and advances by that much. Register
// offsets are the GCN SH (0x2C00 base) and context (0xA000 base) spaces.
struct PacketWriter {
    std::uint32_t* p;
    std::uint32_t* end;
    static std::uint32_t header(unsigned op, unsigned payload) {
        return 0xC0000000u | ((static_cast<std::uint32_t>(payload - 1) & 0x3fffu) << 16) | (op << 8);
    }
    void regs(unsigned op, std::uint32_t base, std::uint32_t reg, const std::uint32_t* vals, unsigned n) {
        if (p + 2 + n > end) {
            return;
        }
        *p++ = header(op, n + 1);
        *p++ = reg - base;
        for (unsigned i = 0; i < n; ++i) {
            *p++ = vals[i];
        }
    }
    void sh(std::uint32_t reg, const std::uint32_t* vals, unsigned n) { regs(0x76, 0x2C00, reg, vals, n); }
    void ctx(std::uint32_t reg, const std::uint32_t* vals, unsigned n) { regs(0x69, 0xA000, reg, vals, n); }
    void nop_fill() {
        // One NOP packet covering the rest of the reservation (a NOP header
        // with payload count 0 means a single dword).
        if (p >= end) {
            return;
        }
        const unsigned n = static_cast<unsigned>(end - p);
        if (n == 1) {
            *p++ = 0xC0001000u;  // NOP, count field 0 -> hmm: 1-dword NOP is 0x80000000 (type-2); use that
            p[-1] = 0x80000000u;
            return;
        }
        *p++ = header(0x10, n - 1);
        while (p < end) {
            *p++ = 0;
        }
    }
};

// Register structs as laid out by Gnm.
struct CsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2, num_thread_x, num_thread_y, num_thread_z;
};
struct PsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2, z_format, col_format, input_ena, input_addr, in_control, baryc_cntl,
        db_shader_control, cb_shader_mask;
};
struct VsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2, vs_out_config, pos_format, pa_cl_vs_out_cntl;
};

int write_cs(std::uint32_t* cmd, unsigned n, const CsStageRegs* r) {
    if (!cmd || n < 25) {
        return static_cast<int>(0x80D10001);  // SCE_GNM_ERROR_...: too small
    }
    CsStageRegs z{};
    if (!r) r = &z;
    PacketWriter w{cmd, cmd + 25};
    w.sh(0x2E0C, &r->pgm_lo, 2);       // COMPUTE_PGM_LO/HI
    w.sh(0x2E12, &r->rsrc1, 2);        // COMPUTE_PGM_RSRC1/2
    w.sh(0x2E07, &r->num_thread_x, 3); // COMPUTE_NUM_THREAD_X/Y/Z
    w.nop_fill();
    return 0;
}
int write_ps(std::uint32_t* cmd, unsigned n, const PsStageRegs* r) {
    if (!cmd || n < 40) {
        return static_cast<int>(0x80D10001);
    }
    PsStageRegs z{};
    if (!r) r = &z;
    PacketWriter w{cmd, cmd + 40};
    w.sh(0x2C08, &r->pgm_lo, 4);              // SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_PS
    w.ctx(0xA1C4, &r->z_format, 2);           // SPI_SHADER_Z_FORMAT, SPI_SHADER_COL_FORMAT
    w.ctx(0xA1B3, &r->input_ena, 2);          // SPI_PS_INPUT_ENA, SPI_PS_INPUT_ADDR
    w.ctx(0xA1B6, &r->in_control, 1);         // SPI_PS_IN_CONTROL
    w.ctx(0xA1B8, &r->baryc_cntl, 1);         // SPI_BARYC_CNTL
    w.ctx(0xA203, &r->db_shader_control, 1);  // DB_SHADER_CONTROL
    w.ctx(0xA08F, &r->cb_shader_mask, 1);     // CB_SHADER_MASK
    w.nop_fill();
    return 0;
}
int write_vs(std::uint32_t* cmd, unsigned n, const VsStageRegs* r, std::uint32_t modifier) {
    if (!cmd || n < 29) {
        return static_cast<int>(0x80D10001);
    }
    VsStageRegs z{};
    if (!r) r = &z;
    VsStageRegs m = *r;
    m.rsrc1 |= modifier;  // fetch-shader VGPR component count etc.
    PacketWriter w{cmd, cmd + 29};
    w.sh(0x2C48, &m.pgm_lo, 4);               // SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_VS
    w.ctx(0xA1B1, &m.vs_out_config, 1);       // SPI_VS_OUT_CONFIG
    w.ctx(0xA1C3, &m.pos_format, 1);          // SPI_SHADER_POS_FORMAT
    w.ctx(0xA207, &m.pa_cl_vs_out_cntl, 1);   // PA_CL_VS_OUT_CNTL
    w.nop_fill();
    return 0;
}
// The tessellation and geometry stages (register layouts as Gnm passes them:
// program address, then RSRC1/RSRC2, then the stage's state). These were NOP
// fills until 2026-09-19, so a tessellated draw reached the command processor
// with no LS/HS program and no VGT_LS_HS_CONFIG / VGT_TF_PARAM: the particle
// effects (fire, item glints, the lamp's travel light) are patches that the
// tessellator expands into quads.
struct LsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2;
};
struct HsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2, vgt_tf_param, vgt_hos_max_tess_level, vgt_hos_min_tess_level;
};
struct EsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2;
};
struct GsStageRegs {
    std::uint32_t pgm_lo, pgm_hi, rsrc1, rsrc2, vgt_strmout_config, vgt_gs_out_prim_type, vgt_gs_instance_cnt;
};
int write_ls(std::uint32_t* cmd, unsigned n, const LsStageRegs* r, std::uint32_t modifier) {
    if (!cmd || n < 23 || !r) return static_cast<int>(0x80D10001);
    // The modifier replaces RSRC1's VGPR-component and user-SGPR fields.
    const std::uint32_t mask = (modifier & 0xfffffc3fu) == 0 ? 0xfffffc3fu : 0xfcfffc3fu;
    const std::uint32_t rsrc[2] = {modifier ? ((r->rsrc1 & mask) | modifier) : r->rsrc1, r->rsrc2};
    const std::uint32_t pgm[2] = {r->pgm_lo, 0};
    PacketWriter w{cmd, cmd + 23};
    w.sh(0x2D48, pgm, 2);         // SPI_SHADER_PGM_LO/HI_LS
    w.sh(0x2D4B, &r->rsrc2, 1);   // SPI_SHADER_PGM_RSRC2_LS
    w.sh(0x2D4A, rsrc, 2);        // SPI_SHADER_PGM_RSRC1/RSRC2_LS
    w.nop_fill();
    return 0;
}
int write_hs(std::uint32_t* cmd, unsigned n, const HsStageRegs* r, std::uint32_t ls_hs_config) {
    if (!cmd || n < 30 || !r) return static_cast<int>(0x80D10001);
    const std::uint32_t pgm[2] = {r->pgm_lo, 0};
    PacketWriter w{cmd, cmd + 30};
    w.sh(0x2D08, pgm, 2);                          // SPI_SHADER_PGM_LO/HI_HS
    w.sh(0x2D0A, &r->rsrc1, 2);                    // SPI_SHADER_PGM_RSRC1/RSRC2_HS
    w.ctx(0xA286, &r->vgt_hos_max_tess_level, 2);  // VGT_HOS_MAX_TESS_LEVEL, VGT_HOS_MIN_TESS_LEVEL
    w.ctx(0xA2DB, &r->vgt_tf_param, 1);            // VGT_TF_PARAM
    w.ctx(0xA2D6, &ls_hs_config, 1);               // VGT_LS_HS_CONFIG
    w.nop_fill();
    return 0;
}
int write_es(std::uint32_t* cmd, unsigned n, const EsStageRegs* r, std::uint32_t modifier) {
    if (!cmd || n < 20 || !r) return static_cast<int>(0x80D10001);
    const std::uint32_t rsrc[2] = {modifier ? ((r->rsrc1 & 0xfcfffc3fu) | modifier) : r->rsrc1, r->rsrc2};
    const std::uint32_t pgm[2] = {r->pgm_lo, 0};
    PacketWriter w{cmd, cmd + 20};
    w.sh(0x2CC8, pgm, 2);   // SPI_SHADER_PGM_LO/HI_ES
    w.sh(0x2CCA, rsrc, 2);  // SPI_SHADER_PGM_RSRC1/RSRC2_ES
    w.nop_fill();
    return 0;
}
int write_gs(std::uint32_t* cmd, unsigned n, const GsStageRegs* r) {
    if (!cmd || n < 29 || !r) return static_cast<int>(0x80D10001);
    const std::uint32_t pgm[2] = {r->pgm_lo, 0};
    PacketWriter w{cmd, cmd + 29};
    w.sh(0x2C88, pgm, 2);                       // SPI_SHADER_PGM_LO/HI_GS
    w.sh(0x2C8A, &r->rsrc1, 2);                 // SPI_SHADER_PGM_RSRC1/RSRC2_GS
    w.ctx(0xA2E5, &r->vgt_strmout_config, 1);   // VGT_STRMOUT_CONFIG
    w.ctx(0xA29B, &r->vgt_gs_out_prim_type, 1); // VGT_GS_OUT_PRIM_TYPE
    w.ctx(0xA2E4, &r->vgt_gs_instance_cnt, 1);  // VGT_GS_INSTANCE_CNT
    w.nop_fill();
    return 0;
}

GUEST_ABI int hle_gnm_set_ps(std::uint32_t* cmd, unsigned n, const PsStageRegs* r) { return write_ps(cmd, n, r); }
GUEST_ABI int hle_gnm_upd_ps(std::uint32_t* cmd, unsigned n, const PsStageRegs* r) { return write_ps(cmd, n, r); }
GUEST_ABI int hle_gnm_set_vs(std::uint32_t* cmd, unsigned n, const VsStageRegs* r, std::uint32_t mod) { return write_vs(cmd, n, r, mod); }
GUEST_ABI int hle_gnm_upd_vs(std::uint32_t* cmd, unsigned n, const VsStageRegs* r, std::uint32_t mod) { return write_vs(cmd, n, r, mod); }
GUEST_ABI int hle_gnm_set_es(std::uint32_t* cmd, unsigned n, const EsStageRegs* r, std::uint32_t mod) { return write_es(cmd, n, r, mod); }
GUEST_ABI int hle_gnm_set_hs(std::uint32_t* cmd, unsigned n, const HsStageRegs* r, std::uint32_t cfg) { return write_hs(cmd, n, r, cfg); }
GUEST_ABI int hle_gnm_upd_hs(std::uint32_t* cmd, unsigned n, const HsStageRegs* r, std::uint32_t cfg) { return write_hs(cmd, n, r, cfg); }
GUEST_ABI int hle_gnm_set_ls(std::uint32_t* cmd, unsigned n, const LsStageRegs* r, std::uint32_t mod) { return write_ls(cmd, n, r, mod); }
GUEST_ABI int hle_gnm_set_cs(std::uint32_t* cmd, unsigned n, const CsStageRegs* r) { return write_cs(cmd, n, r); }
GUEST_ABI int hle_gnm_set_gs(std::uint32_t* cmd, unsigned n, const GsStageRegs* r) { return write_gs(cmd, n, r); }
GUEST_ABI int hle_gnm_upd_gs(std::uint32_t* cmd, unsigned n, const GsStageRegs* r) { return write_gs(cmd, n, r); }
// The debug markers are real packets: Gnmx reserves `size` dwords for each and
// advances its write pointer by that much whether we write anything or not.
// Writing nothing left those dwords holding the previous frame's contents, so
// the command processor read stale payloads as packet headers, lost alignment,
// and the end-of-flush fences after them never ran. The packet is a NOP whose
// payload is a magic word and the marker string.
constexpr std::uint32_t kMarkerPush = 0x68750001u, kMarkerPop = 0x68750002u, kMarkerSet = 0x68750003u,
                        kMarkerColorPush = 0x6875000eu;

int write_marker(std::uint32_t* cmd, unsigned size, std::uint32_t kind, const char* text, const std::uint32_t* extra,
                 unsigned extra_dwords) {
    if (!cmd || size < 2) {
        return static_cast<int>(0x80D10001);
    }
    std::memset(cmd, 0, static_cast<std::size_t>(size) * 4);
    cmd[0] = 0xC0001000u | ((static_cast<std::uint32_t>(size - 2) & 0x3fffu) << 16);
    cmd[1] = kind;
    unsigned at = 2;
    for (unsigned i = 0; i < extra_dwords && at < size; ++i) cmd[at++] = extra[i];
    if (text && at < size) {
        const std::size_t room = static_cast<std::size_t>(size - at) * 4;
        const std::size_t n = std::strlen(text) + 1;
        std::memcpy(cmd + at, text, n < room ? n : room);
    }
    return 0;
}

GUEST_ABI int hle_gnm_set_marker(std::uint32_t* cmd, unsigned size, const char* text) {
    return write_marker(cmd, size, kMarkerSet, text, nullptr, 0);
}
GUEST_ABI int hle_gnm_push_marker(std::uint32_t* cmd, unsigned size, const char* text) {
    return write_marker(cmd, size, kMarkerPush, text, nullptr, 0);
}
GUEST_ABI int hle_gnm_push_color_marker(std::uint32_t* cmd, unsigned size, const char* text, std::uint32_t color) {
    return write_marker(cmd, size, kMarkerColorPush, text, &color, 1);
}
GUEST_ABI int hle_gnm_pop_marker(std::uint32_t* cmd, unsigned size) {
    return write_marker(cmd, size, kMarkerPop, nullptr, nullptr, 0);
}
GUEST_ABI int hle_gnm_wait_flip(void*, const void*, int, int) { return 0; }

GUEST_ABI unsigned hle_gnm_cu_index(unsigned i) { return i; }
GUEST_ABI unsigned hle_gnm_cu_mask(unsigned m) { return m; }

// The draw buffers and, paired with them by index, the constant-engine ones.
// The command processor interleaves the two through the CE/DE counters.
std::vector<std::pair<const void*, std::size_t>> collect_submit(unsigned count, void** dcb, unsigned* dcb_bytes) {
    std::vector<std::pair<const void*, std::size_t>> cbs;
    for (unsigned i = 0; i < count; ++i) {
        cbs.emplace_back(dcb && dcb_bytes && dcb[i] && dcb_bytes[i] ? dcb[i] : nullptr,
                         dcb && dcb_bytes && dcb[i] ? dcb_bytes[i] : 0);
    }
    return cbs;
}
std::vector<std::pair<const void*, std::size_t>> collect_ce(unsigned count, void** ccb, unsigned* ccb_bytes) {
    std::vector<std::pair<const void*, std::size_t>> cbs;
    for (unsigned i = 0; i < count; ++i) {
        cbs.emplace_back(ccb && ccb_bytes && ccb[i] && ccb_bytes[i] ? ccb[i] : nullptr,
                         ccb && ccb_bytes && ccb[i] ? ccb_bytes[i] : 0);
    }
    return cbs;
}

GUEST_ABI int hle_gnm_submit(unsigned count, void** dcb, unsigned* dcb_bytes, void** ccb, unsigned* ccb_bytes) {
    hle_gnm_submit_async(0, collect_submit(count, dcb, dcb_bytes), [] { post_gnm_eop(); },
                         collect_ce(count, ccb, ccb_bytes));
    return 0;
}
GUEST_ABI int hle_gnm_submit_flip(unsigned count, void** dcb, unsigned* dcb_bytes, void** ccb, unsigned* ccb_bytes,
                                 int handle, int buffer, unsigned, std::int64_t arg) {
    hle_gnm_submit_async(0, collect_submit(count, dcb, dcb_bytes),
                         [handle, buffer, arg] {
                             hle_video_finish_flip(handle, buffer, arg, true);  // the job's commands are recorded
                             post_gnm_eop();
                         },
                         collect_ce(count, ccb, ccb_bytes));
    // Diagnostic: watch the SprjScaleform singleton's resident-resource
    // loader (*(singleton+8)) and the "resident load complete" flag
    // (byte at 0x5962878+0x32) across the first flips, to see whether the
    // menu-load step stops ticking or its loader goes null after ~7 menus.
    static const bool trace_menuload = [] {
        const char* e = std::getenv("BBHOST_TRACE_MENULOAD");
        return e && e[0] == '1';
    }();
    if (trace_menuload) {
        const std::uint64_t sp = *reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(0x5940368));
        const std::uint64_t loader = sp ? *reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(sp + 8)) : 0;
        const std::uint8_t done = *reinterpret_cast<const volatile std::uint8_t*>(static_cast<std::uintptr_t>(0x59628aa));
        static std::atomic<int> mlogs{0};
        static std::uint64_t last_loader = 1;  // impossible initial
        static std::uint8_t last_done = 0xff;
        const int m = mlogs.load();
        if (m < 40 || loader != last_loader || done != last_done) {
            mlogs.fetch_add(1);
            last_loader = loader; last_done = done;
            host_log("scaleform-load: singleton=0x%llx loader(+8)=0x%llx resident_done=%u",
                     static_cast<unsigned long long>(sp), static_cast<unsigned long long>(loader), done);
        }
    }
    hle_gnm_arena_flip();  // BBHOST_ARENA_MONITOR
    frame_stats_on_flip();  // BBHOST_FRAME_STATS
    static std::atomic<int> flip_logs{0};
    const int n = flip_logs.fetch_add(1);
    // A flip that came BBHOST_STALL_MS (100 by default; 0 turns this off) or
    // more after the previous one: what the command processor and the texture
    // cache spent in between, so a hitch (a new area loading) shows whether it
    // went to pipeline builds, dispatches, draws, texture uploads or waits. The
    // command processor runs behind the game, so work in the gap can belong to
    // the frames just before it.
    static const std::uint64_t stall_ms = [] {
        const char* e = std::getenv("BBHOST_STALL_MS");
        return e ? std::strtoull(e, nullptr, 10) : 100ull;
    }();
    if (stall_ms) {
        static std::mutex stall_mu;
        std::lock_guard<std::mutex> stall_lock(stall_mu);
        const auto t = std::chrono::steady_clock::now();
        const GpuStats s = host_gpu_stats();
        const std::uint64_t uploads = host_gpu_texture_uploads(), hashes = host_gpu_texture_hashes();
        const std::uint64_t hash_us = host_gpu_texture_hash_us(), repl = host_gpu_surface_replacements();
        std::uint64_t tex_us[5], pl_us[5];
        host_gpu_texture_time_us(tex_us);
        host_gpu_pipeline_time_us(pl_us);
        static std::uint64_t prev_tex_us[5] = {tex_us[0], tex_us[1], tex_us[2], tex_us[3], tex_us[4]};
        static std::uint64_t prev_pl_us[5] = {pl_us[0], pl_us[1], pl_us[2], pl_us[3], pl_us[4]};
        static auto prev_t = t;
        static GpuStats prev = s;
        static std::uint64_t prev_uploads = uploads, prev_hashes = hashes, prev_hash_us = hash_us, prev_repl = repl, stalls = 0;
        const std::uint64_t gap_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t - prev_t).count());
        if (gap_ms >= stall_ms && stalls++ < 4000) {
            std::string ph;
            for (int i = 0; i < kPhaseCount; ++i) {
                char buf[48];
                std::snprintf(buf, sizeof(buf), " %s=%llu", kPhaseNames[i],
                              static_cast<unsigned long long>((s.phase_ns[i] - prev.phase_ns[i]) / 1000000));
                ph += buf;
            }
            host_log("stall: flip #%d came %llu ms after the previous one (cp-backlog %llu); in between cp-thread ms:%s; graphics pipelines %llu, "
                     "draws %llu, dispatches %llu, texture uploads %llu, texture hashes %llu in %llu ms, surfaces replaced %llu; "
                     "texture ms: images %llu, staging %llu, untile %llu, record %llu, views %llu; pipeline ms: translate %llu, "
                     "modules %llu, create %llu; stage translations %llu, stages reused %llu",
                     n, static_cast<unsigned long long>(gap_ms), static_cast<unsigned long long>(hle_gnm_cp_backlog()), ph.c_str(),
                     static_cast<unsigned long long>(s.pipelines - prev.pipelines), static_cast<unsigned long long>(s.draws - prev.draws),
                     static_cast<unsigned long long>(s.dispatches - prev.dispatches), static_cast<unsigned long long>(uploads - prev_uploads),
                     static_cast<unsigned long long>(hashes - prev_hashes), static_cast<unsigned long long>((hash_us - prev_hash_us) / 1000),
                     static_cast<unsigned long long>(repl - prev_repl), static_cast<unsigned long long>((tex_us[0] - prev_tex_us[0]) / 1000),
                     static_cast<unsigned long long>((tex_us[1] - prev_tex_us[1]) / 1000),
                     static_cast<unsigned long long>((tex_us[2] - prev_tex_us[2]) / 1000),
                     static_cast<unsigned long long>((tex_us[3] - prev_tex_us[3]) / 1000),
                     static_cast<unsigned long long>((tex_us[4] - prev_tex_us[4]) / 1000),
                     static_cast<unsigned long long>((pl_us[0] - prev_pl_us[0]) / 1000), static_cast<unsigned long long>((pl_us[1] - prev_pl_us[1]) / 1000),
                     static_cast<unsigned long long>((pl_us[2] - prev_pl_us[2]) / 1000), static_cast<unsigned long long>(pl_us[3] - prev_pl_us[3]),
                     static_cast<unsigned long long>(pl_us[4] - prev_pl_us[4]));
        }
        for (int i = 0; i < 5; ++i) {
            prev_tex_us[i] = tex_us[i];
            prev_pl_us[i] = pl_us[i];
        }
        prev_t = t;
        prev = s;
        prev_uploads = uploads;
        prev_hashes = hashes;
        prev_hash_us = hash_us;
        prev_repl = repl;
    }
    if (n < 8 || (n % 300) == 0) {
        static GpuStats last{};
        static auto last_t = std::chrono::steady_clock::now();
        const GpuStats now = host_gpu_stats();
        // Process CPU time (user + system, every thread) since the last report:
        // what the game, the command processor and the host spent on these flips.
        static std::uint64_t last_cpu_ms = 0;
        const std::uint64_t cpu_ms = host_process_cpu_ms();
        const auto t = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t - last_t).count();
        host_log("sceGnmSubmitAndFlip #%d handle=%d buf=%d arg=%lld cp-backlog=%llu | %.1f s: cpu-ms=%llu draws=%llu draw-calls=%llu draw-failures=%llu draws-empty=%llu dispatches=%llu "
                 "submits=%llu gpu-wait=%llu ms pipelines=%llu transfers=%llu/%llu; host-side writes by op:%s",
                 n, handle, buffer, static_cast<long long>(arg), static_cast<unsigned long long>(hle_gnm_cp_backlog()),
                 secs, static_cast<unsigned long long>(cpu_ms - last_cpu_ms), static_cast<unsigned long long>(now.draws - last.draws),
                 static_cast<unsigned long long>(now.draw_calls - last.draw_calls),
                 static_cast<unsigned long long>(now.draw_failures - last.draw_failures),
                 static_cast<unsigned long long>(now.draws_empty - last.draws_empty),
                 static_cast<unsigned long long>(now.dispatches - last.dispatches),
                 static_cast<unsigned long long>(now.flushes - last.flushes),
                 static_cast<unsigned long long>((now.gpu_us - last.gpu_us) / 1000),
                 static_cast<unsigned long long>(now.pipelines - last.pipelines),
                 static_cast<unsigned long long>(now.transfers - last.transfers),
                 static_cast<unsigned long long>(now.transfer_batches - last.transfer_batches), hle_gnm_flush_reasons().c_str());
        {
            // The window's draws by how their pixel shader ran: the share the
            // lifter (BBHOST_DECOMP) carries.
            std::uint64_t ps[kDrawPsCount];
            for (int i = 0; i < kDrawPsCount; ++i) ps[i] = now.draws_ps[i] - last.draws_ps[i];
            const std::uint64_t shaded = ps[kDrawPsLifted] + ps[kDrawPsTranslated] + ps[kDrawPsFallback];
            host_log("  pixel shaders: %llu draws lifted (%.1f%%), %llu translated, %llu on the fallback; %llu draws without one",
                     static_cast<unsigned long long>(ps[kDrawPsLifted]),
                     shaded ? 100.0 * static_cast<double>(ps[kDrawPsLifted]) / static_cast<double>(shaded) : 0.0,
                     static_cast<unsigned long long>(ps[kDrawPsTranslated]), static_cast<unsigned long long>(ps[kDrawPsFallback]),
                     static_cast<unsigned long long>(ps[kDrawPsNone]));
        }
        if (const std::string arena = hle_gnm_arena_report(); !arena.empty()) host_log("arena:%s", arena.c_str());
        std::string ph;
        for (int i = 0; i < kPhaseCount; ++i) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), " %s=%llu", kPhaseNames[i],
                          static_cast<unsigned long long>((now.phase_ns[i] - last.phase_ns[i]) / 1000000));
            ph += buf;
        }
        host_log("  cp-thread ms:%s", ph.c_str());
        host_log("  gnm ops: %s", hle_gnm_op_window().c_str());
        const std::string sources = hle_gx_source_window();
        if (!sources.empty()) host_log("  gnm sources: %s", sources.c_str());
        host_gpu_save_pipeline_cache();
        if (const std::string c = host_gpu_compile_report(); !c.empty()) host_log("  compile: %s", c.c_str());
        static std::uint64_t last_hashes = 0, last_hash_us = 0;
        const std::uint64_t hashes = host_gpu_texture_hashes(), hash_us = host_gpu_texture_hash_us();
        host_log("  media:%s%s texture-uploads=%llu texture-hashes=%llu in %llu ms",
                 hle_av_stats().c_str(), hle_audio_stats().c_str(),
                 static_cast<unsigned long long>(host_gpu_texture_uploads()),
                 static_cast<unsigned long long>(hashes - last_hashes),
                 static_cast<unsigned long long>((hash_us - last_hash_us) / 1000));
        static std::uint64_t last_repl = 0;
        const std::uint64_t repl = host_gpu_surface_replacements();
        if (repl != last_repl) host_log("  surfaces replaced: %llu", static_cast<unsigned long long>(repl - last_repl));
        last_repl = repl;
        last_hashes = hashes;
        last_hash_us = hash_us;
        const std::string prof = host_gpu_profile_report();
        if (!prof.empty()) host_log("  gpu profile:%s", prof.c_str());
        if (const std::string busy = host_gpu_busy_report(); !busy.empty()) host_log("  %s", busy.c_str());
        host_log("  fill: %s", host_gpu_fill_stats().c_str());
        // The guest's clock reads and sync slow paths a second (hle/kernel.cpp),
        // and the pacing: the frame limiter's margin and spin
        // (engine/frame_rate.cpp), the vblank clock's lateness and the flips'
        // latencies (hle/video.cpp).
        {
            std::string sync;
            const std::string clocks = hle_timing_window(secs, &sync);
            host_log("  clocks: %s", clocks.c_str());
            host_log("  sync: %s", sync.c_str());
            host_log("  pacing: %s; %s", frame_rate_limiter_window().c_str(), hle_video_pacing_window().c_str());
        }
        host_log("  %s", host_gpu_shadow_report().c_str());
        if (const std::string r = host_gpu_recorder_report(); !r.empty()) host_log("  %s", r.c_str());
        if (const std::string r = host_gpu_occlusion_report(); !r.empty()) host_log("  %s", r.c_str());
        if (const std::string r = host_gpu_image_heap_report(); !r.empty()) host_log("  %s", r.c_str());
        if (const std::string r = host_gpu_ps_wave_report(); !r.empty()) host_log("  %s", r.c_str());
        if (const std::string r = host_gpu_memory_budget_report(); !r.empty()) host_log("  %s", r.c_str());
        // What the PC enhancements and Bloom cost in these flips: the file
        // lookups that passed the generated overlays by, the frames that read
        // the altar's flags, the views that drew YEBIS's glare.
        if (const std::string r = hle_fs_overlay_report(); !r.empty()) host_log("  %s", r.c_str());
        if (const std::string r = rebirth_report(); !r.empty()) host_log("  %s", r.c_str());
        if (const std::string r = graphics_glare_report(); !r.empty()) host_log("  %s", r.c_str());
        static const bool count_calls = std::getenv("BBHOST_HLE_COUNT") != nullptr;
        if (count_calls) hle_call_counts_report();
        last = now;
        last_t = t;
        last_cpu_ms = cpu_ms;
    }
    return 0;
}
GUEST_ABI int hle_gnm_validate(unsigned, void**, unsigned*, void**, unsigned*) { return 0; }

GUEST_ABI int hle_gnm_add_eq(HostEqueue* eq, int id, void* udata) {
    if (!eq) {
        return sce_err(EINVAL);
    }
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    g_gnm_eq.push_back(GnmEq{eq, id, udata});
    host_log("sceGnmAddEqEvent id=%d eq=%p", id, static_cast<void*>(eq));
    return 0;
}
GUEST_ABI int hle_gnm_del_eq(HostEqueue* eq, int id) {
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    g_gnm_eq.erase(std::remove_if(g_gnm_eq.begin(), g_gnm_eq.end(),
                                  [&](const GnmEq& e) { return e.eq == eq && e.id == id; }),
                   g_gnm_eq.end());
    return 0;
}
GUEST_ABI int hle_gnm_eq_type(const HostEvent* ev) { return ev ? static_cast<int>(ev->ident) : 0; }
GUEST_ABI int hle_gnm_eq_ts(HostEqueue*, int, std::uint64_t* ts) {
    if (ts) {
        *ts = now_us();
    }
    return 0;
}

// sceGnmMapComputeQueue(pipe, queue, ringBase, ringSizeDw, readPtrAddr)
GUEST_ABI int hle_gnm_map_queue(unsigned pipe, unsigned queue, void* ring, unsigned size_dw, void* read_ptr) {
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    int id = g_queue_next++;
    g_queues[id] = ComputeQueue{static_cast<std::uint32_t*>(ring), size_dw, static_cast<std::uint32_t*>(read_ptr)};
    if (read_ptr) {
        *static_cast<std::uint32_t*>(read_ptr) = 0;
    }
    host_log("sceGnmMapComputeQueue pipe=%u q=%u ring=%p size=%u dw -> %d", pipe, queue, ring, size_dw, id);
    return id;
}
GUEST_ABI int hle_gnm_map_queue_prio(unsigned pipe, unsigned queue, void* ring, unsigned size_dw, void* read_ptr,
                                    int) {
    return hle_gnm_map_queue(pipe, queue, ring, size_dw, read_ptr);
}
GUEST_ABI int hle_gnm_unmap_queue(int id) {
    std::lock_guard<std::mutex> lock(g_gnm_mu);
    g_queues.erase(id);
    return 0;
}
// sceGnmDingDong(queueId, nextStartOffsetInDw): execute ring packets from
// the read pointer up to the new write offset, then publish the read pointer.
GUEST_ABI int hle_gnm_dingdong(int id, unsigned next_dw) {
    ComputeQueue q;
    std::uint32_t rd;
    {
        std::lock_guard<std::mutex> lock(g_gnm_mu);
        auto it = g_queues.find(id);
        if (it == g_queues.end()) {
            return sce_err(EINVAL);
        }
        q = it->second;
        rd = q.queued_dw;
        if (next_dw > q.size_dw) {
            next_dw = q.size_dw;
        }
        it->second.queued_dw = next_dw;
    }
    if (!q.ring || !q.size_dw) {
        return 0;
    }
    if (rd >= q.size_dw) {
        rd = 0;
    }
    std::vector<std::pair<const void*, std::size_t>> cbs;
    if (next_dw >= rd) {
        cbs.emplace_back(q.ring + rd, (next_dw - rd) * 4);
    } else {
        cbs.emplace_back(q.ring + rd, (q.size_dw - rd) * 4);
        cbs.emplace_back(q.ring, next_dw * 4);
    }
    std::uint32_t* read_ptr = q.read_ptr;
    hle_gnm_submit_async(id, cbs, [read_ptr, next_dw] {
        if (read_ptr) {
            __atomic_store_n(read_ptr, next_dw, __ATOMIC_RELEASE);
        }
        post_gnm_eop();
    });
    return 0;
}
GUEST_ABI int hle_gnm_flush() { return 0; }
GUEST_ABI int hle_gnm_hw_status(void* st) {
    if (st) {
        std::memset(st, 0, 64);
    }
    return 0;
}
GUEST_ABI int hle_gnm_submit_done() {
    // The guest marks the end of a frame's submissions; nothing to wait for.
    return 0;
}
GUEST_ABI int hle_gnm_submits_ok() { return 1; }
GUEST_ABI int hle_gnm_flip_done(int handle, int buffer, unsigned, std::int64_t arg) {
    hle_video_finish_flip(handle, buffer, arg);
    post_gnm_eop();
    return 0;
}
GUEST_ABI int hle_gnm_mip_setup(void*, int) { return 0; }
GUEST_ABI int hle_gnm_mip_reset(void*) { return 0; }
GUEST_ABI int hle_gnm_mip_disable() { return 0; }
GUEST_ABI int hle_gnm_gs_rings(unsigned, unsigned) { return 0; }
GUEST_ABI int hle_gnm_capture() { return 0; }
GUEST_ABI int hle_gnm_capture_busy() { return 0; }
GUEST_ABI int hle_gnm_user_pa() { return 0; }

}  // namespace

void hle_register_gnm() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceGnmDrawInitDefaultHardwareState200", hle_gnm_draw_init);
    REG("sceGnmDispatchInitDefaultHardwareState", hle_gnm_dispatch_init);
    REG("sceGnmSetPsShader", hle_gnm_set_ps);
    REG("sceGnmUpdatePsShader", hle_gnm_upd_ps);
    REG("sceGnmSetVsShader", hle_gnm_set_vs);
    REG("sceGnmUpdateVsShader", hle_gnm_upd_vs);
    REG("sceGnmSetEsShader", hle_gnm_set_es);
    REG("sceGnmSetHsShader", hle_gnm_set_hs);
    REG("sceGnmUpdateHsShader", hle_gnm_upd_hs);
    REG("sceGnmSetLsShader", hle_gnm_set_ls);
    REG("sceGnmSetCsShader", hle_gnm_set_cs);
    REG("sceGnmSetGsShader", hle_gnm_set_gs);
    REG("sceGnmUpdateGsShader", hle_gnm_upd_gs);
    REG("sceGnmInsertSetMarker", hle_gnm_set_marker);
    REG("sceGnmInsertSetColorMarker", hle_gnm_push_color_marker);
    REG("sceGnmInsertPushMarker", hle_gnm_push_marker);
    REG("sceGnmInsertPushColorMarker", hle_gnm_push_color_marker);
    REG("sceGnmInsertPopMarker", hle_gnm_pop_marker);
    REG("sceGnmInsertWaitFlipDone", hle_gnm_wait_flip);
    REG("sceGnmLogicalCuIndexToPhysicalCuIndex", hle_gnm_cu_index);
    REG("sceGnmLogicalCuMaskToPhysicalCuMask", hle_gnm_cu_mask);
    REG("sceGnmSubmitCommandBuffers", hle_gnm_submit);
    REG("sceGnmSubmitAndFlipCommandBuffers", hle_gnm_submit_flip);
    REG("sceGnmValidateCommandBuffers", hle_gnm_validate);
    REG("sceGnmAddEqEvent", hle_gnm_add_eq);
    REG("sceRazorIsLoaded", hle_razor_not_loaded);
    REG("sceRazorCaptureImmediate", hle_razor_capture);
    REG("sceRazorCaptureSinceLastFlip", hle_razor_capture);
    REG("sceRazorCaptureCommandBuffersOnlyImmediate", hle_razor_capture);
    REG("sceRazorCaptureCommandBuffersOnlySinceLastFlip", hle_razor_capture);
    REG("sceGnmDeleteEqEvent", hle_gnm_del_eq);
    REG("sceGnmGetEqEventType", hle_gnm_eq_type);
    REG("sceGnmGetEqTimeStamp", hle_gnm_eq_ts);
    REG("sceGnmMapComputeQueue", hle_gnm_map_queue);
    REG("sceGnmMapComputeQueueWithPriority", hle_gnm_map_queue_prio);
    REG("sceGnmUnmapComputeQueue", hle_gnm_unmap_queue);
    REG("sceGnmDingDong", hle_gnm_dingdong);
    REG("sceGnmFlushGarlic", hle_gnm_flush);
    REG("sceGnmDebugHardwareStatus", hle_gnm_hw_status);
    REG("sceGnmSubmitDone", hle_gnm_submit_done);
    REG("sceGnmAreSubmitsAllowed", hle_gnm_submits_ok);
    REG("sceGnmRequestFlipAndSubmitDone", hle_gnm_flip_done);
    REG("sceGnmSetupMipStatsReport", hle_gnm_mip_setup);
    REG("sceGnmRequestMipStatsReportAndReset", hle_gnm_mip_reset);
    REG("sceGnmDisableMipStatsReport", hle_gnm_mip_disable);
    REG("sceGnmSetGsRingSizes", hle_gnm_gs_rings);
    REG("sceGnmGetTheTessellationFactorRingBufferBaseAddress", hle_gnm_tf_ring);
    REG("sceGnmDriverTriggerCapture", hle_gnm_capture);
    REG("sceGnmDriverCaptureInProgress", hle_gnm_capture_busy);
    REG("sceGnmIsUserPaEnabled", hle_gnm_user_pa);
#undef REG
}
