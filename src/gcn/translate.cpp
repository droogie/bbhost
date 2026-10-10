#include "gcn/translate.h"

#include "gcn/half.h"
#include "gcn/spirv.h"
#include "gcn/wave.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

namespace gcn {

namespace {
std::atomic<bool> g_runtime_sample_offsets{false};
}  // namespace
void set_runtime_sample_offsets(bool on) { g_runtime_sample_offsets.store(on, std::memory_order_relaxed); }
bool runtime_sample_offsets() { return g_runtime_sample_offsets.load(std::memory_order_relaxed); }

std::string translation_switches() {
    // The variables as given rather than what each decides: two spellings of
    // one setting only cost a second entry.
    const auto env = [](const char* name) {
        const char* e = std::getenv(name);
        return std::string(name) + "=" + (e ? e : "") + ";";
    };
    std::string s = std::string("offsets=") + (runtime_sample_offsets() ? "1" : "0") + ";half_rtz=" + (native_half_rtz() ? "1" : "0") +
                    ";export_rtz=" + (export_rtz_on() ? "1" : "0") + ";legacy_mul=" + (legacy_mul_min_form() ? "min" : "select") + ";";
    for (const char* name : {"BBHOST_DST_SEL_BRANCH", "BBHOST_LOOP_BLOCKS", "BBHOST_LOOP_REACH", "BBHOST_SPILL_CELLS"}) s += env(name);
    return s;
}

std::string ResourcePath::str() const {
    std::string s = "user_sgpr[" + std::to_string(user_sgpr) + "]";
    for (const ResourceStep& l : loads) {
        s += (l.vsharp ? "->vsharp(+" : "->load(+") + std::to_string(l.offset_dw * 4) + ")";
    }
    if (!immediate) {
        s += (final_vsharp ? "->vsharp(+" : "->load(+") + std::to_string(final_offset_dw * 4) + ")";
    }
    return s;
}

namespace {

using spv::Id;

constexpr std::uint32_t kFetchBias = 0x01000000;  // block offsets of the inlined fetch shader
constexpr std::uint32_t kLdsWordsMax = 16384;  // GCN's 64 KiB per workgroup

// COMPUTE_PGM_RSRC2.LDS_SIZE: bits 15..23, in units of 128 dwords. Declaring
// the full 64 KiB regardless made every pipeline that touches LDS invalid on a
// device with a 48 KiB limit, which is a device-lost waiting to happen.
std::uint32_t lds_words_for(const TranslateOptions& opt) {
    const std::uint32_t granules = (opt.rsrc2 >> 15) & 0x1ff;
    std::uint32_t words = granules * 128;
    const std::uint32_t cap = opt.max_lds_bytes / 4;
    if (!words) words = cap;  // the shader uses LDS but the register says none
    if (words > cap) words = cap;
    if (words > kLdsWordsMax) words = kLdsWordsMax;
    return words ? words : 1;
}

// Symbolic value of an SGPR for resource tracking.
struct SymVal {
    enum Kind { Unknown, UserData, Loaded } kind = Unknown;
    int user_sgpr = 0;                 // UserData: index; Loaded: root user sgpr
    std::vector<ResourceStep> loads;   // Loaded: chain of dereferences (last = this block)
    int dw = 0;                        // dword within the loaded block
};

struct Translator {
    const Program& prog;
    const TranslateOptions& opt;
    TranslateResult res;
    spv::Module m;

    // Types
    Id t_void, t_bool, t_u32, t_i32, t_u64, t_f32, t_v2u, t_v4u, t_v2f, t_v3f, t_v4f, t_v4b, t_v2i, t_v3i, t_v4i, t_v3u;
    Id p_fn_u32, p_fn_bool, p_fn_u64;
    Id p_psb_u32;  // PhysicalStorageBuffer pointer to u32
    Id t_psb_block, p_psb_block;  // Block { u32 } reached through a physical address
    Id t_psb_block64, p_psb_block64, p_psb_u64;  // Block { u64 } for page-table entries
    Id p_wg_u32, t_lds_array, p_wg_lds;
    // TranslateOptions::tess_role: LDS is the buffer at StageParams::lds_address.
    bool lds_in_buffer = false;
    Id lds_base64 = 0;
    // TranslateOptions::tess_lds_bound: how many bytes from lds_base64 the
    // stage may touch - the patch's window, or StageParams::lds_bytes - and
    // whether member 3 of the params block is the uvec4 that holds the latter.
    Id lds_limit = 0;
    Id lds_patch_ok = 0;  // bool: the window's patch lies inside lds_bytes (0: not checked)
    bool params_head_v4 = false;
    Id p_in_v4f, p_in_v4u, p_in_u32, p_in_f32, p_in_v3u, p_in_bool, p_out_v4f, p_out_f32, p_uc_sampler;
    Id t_fn_main;

    // Register storage (lazily created function variables).
    std::map<int, Id> sgpr_vars, vgpr_vars;
    Id v_vcc_lo = 0, v_vcc_hi = 0, v_exec_lo = 0, v_exec_hi = 0, v_m0 = 0, v_scc = 0;
    Id v_exec_bit = 0, v_vcc_bit = 0;  // cached lane bits
    Id v_pc = 0, v_ret = 0;
    Id v_lane = 0;                     // this invocation's lane index (u32 var)
    Id v_lds = 0;                      // Workgroup array
    // Tessellation: the host stages' own built-ins.
    Id in_tess_coord = 0, in_primitive_id = 0, in_invocation_id = 0;
    // The control point as stage attributes rather than a buffer.
    Id v_attr_out = 0, v_attr_in = 0;
    bool lds_in_attributes = false;
    Id out_tess_outer = 0, out_tess_inner = 0;
    int tf_stores_seen = 0;  // HullTcs: buffer stores mapped onto the levels
    // Inputs / outputs
    std::map<int, Id> in_params;       // PS: attr -> Input vec4
    std::map<std::uint32_t, Id> in_location_vars;  // PS: location -> the Input vec4 there (in_param)
    std::map<int, Id> out_params;      // VS: param n -> Output vec4; PS: mrt n -> Output vec4
    Id out_position = 0, out_frag_depth = 0, out_clip = 0, out_point_size = 0;
    Id v_debug_texel = 0;
    int clip_count = 0;
    Id in_frag_coord = 0, in_front_facing = 0, in_vertex_index = 0, in_instance_index = 0, in_local_id = 0,
       in_workgroup_id = 0, in_subgroup_lane = 0, in_helper = 0;
    Id ubo_var = 0;
    Id v_patch_culled = 0;  // Domain: this patch has no projection (tess_patch_cull)
    std::vector<Id> interface;
    std::map<std::uint32_t, Id> vertex_in_vars;  // TranslateOptions::vertex_input: location -> Input uvec4
    // Resources
    std::map<int, SymVal> sym;  // symbolic SGPR values
    // Scalar loads land when the shader waits on lgkmcnt, not when issued.
    // Sony's compiler relies on that: a VOPC may write s[0:1] between an
    // s_load_dwordx4 s[0:3] and its s_waitcnt, and the S# still wins. The
    // loaded values are re-stored at the next s_waitcnt.
    struct PendingSmrd {
        int sdst;
        std::vector<Id> values;
        std::vector<SymVal> syms;
    };
    std::vector<PendingSmrd> pending_smrd;
    std::map<int, Id> smrd_shadow;  // per SGPR: the value of the last scalar load into it
    Id smrd_shadow_var(int i) {
        auto it = smrd_shadow.find(i);
        if (it != smrd_shadow.end()) return it->second;
        const Id v = m.local_variable(p_fn_u32, c_zero_u);
        m.name(v, "sload" + std::to_string(i));
        smrd_shadow[i] = v;
        return v;
    }
    struct ImageVar {
        Id var, type;
        bool cube = false;  // ImageBinding::cube
        std::uint32_t kind = 0;  // ImageBinding::kind
        // TranslateOptions::bindless: `var` is the global array of this image
        // type and `slot` the binding whose index the params block holds.
        bool bindless = false;
        std::uint32_t slot = 0;
    };
    // TranslateOptions::bindless: the params block's image_index and
    // sampler_index members, and the global arrays (one alias of the image
    // array per image type, and the sampler array).
    std::uint32_t bindless_image_member = 0, bindless_sampler_member = 0;
    std::map<Id, Id> bindless_image_arrays;
    Id bindless_sampler_array = 0;
    Id bindless_array(Id t_elem, std::uint32_t binding, const char* name) {
        const Id t_arr = m.type_runtime_array(t_elem);
        const Id var = m.global_variable(m.type_pointer(spv::ScUniformConstant, t_arr), spv::ScUniformConstant);
        m.decorate(var, spv::DecDescriptorSet, {opt.bindless_set});
        m.decorate(var, spv::DecBinding, {binding});
        m.name(var, name);
        interface.push_back(var);
        m.capability(spv::CapRuntimeDescriptorArray);
        return var;
    }
    Id load_image(const ImageVar& img) {
        if (!img.bindless) return m.load(img.type, img.var);
        const Id slot = params_u32(bindless_image_member, img.slot);
        return m.load(img.type, m.access_chain(m.type_pointer(spv::ScUniformConstant, img.type), img.var, {slot}));
    }
    Id load_sampler(std::size_t index) {
        if (!opt.bindless) return m.load(m.type_sampler(), sampler_vars[index]);
        const Id slot = params_u32(bindless_sampler_member, index);
        return m.load(m.type_sampler(), m.access_chain(p_uc_sampler, bindless_sampler_array, {slot}));
    }
    // An image op's result in the image's sampled type, as the float vector
    // the texel writers take (the bits are what land in the VGPRs).
    Id texel_type(const ImageVar& img) const { return img.kind == 1 ? t_v4u : img.kind == 2 ? t_v4i : t_v4f; }
    Id texel_as_v4f(const ImageVar& img, Id texel) { return img.kind ? m.emit(spv::OpBitcast, t_v4f, {texel}) : texel; }
    std::map<std::string, std::size_t> image_index;    // path -> res.images index
    std::vector<ImageVar> image_vars;
    std::map<std::string, std::size_t> sampler_index;
    std::vector<Id> sampler_vars;
    std::map<std::string, std::size_t> buffer_index;  // ResourcePath::str() of the V# -> res.buffers index
    std::vector<Id> buffer_vars;
    Id p_ssbo_block = 0, p_ssbo_u32 = 0;
    // Every definition of each SGPR in the program and its inlined fetch
    // shader, in emission order: the symbolic value a scalar load gave it
    // (sym_key), or "?" for any other write. A V# bound once per draw is sound
    // only if the definitions of its registers that can reach the load are all
    // that V#'s words (see translate()).
    struct SgprDef {
        std::uint64_t seq;
        std::string key;
        std::uint32_t block;  // start offset of the block it is in (straight-line: 0)
    };
    std::map<int, std::vector<SgprDef>> sgpr_defs;
    std::uint64_t def_seq = 0;
    std::uint32_t cur_block = 0;  // block being emitted, in straight-line and forward-only programs
    std::map<std::uint32_t, std::set<std::uint32_t>> block_succ;  // forward-only programs: each block's successors
    bool in_smrd_write = false;
    struct BufferSite {
        int reg;
        std::size_t index;
        std::string def0, def1;  // sym_key of [reg] and [reg + 1] at the load
        std::uint64_t seq;
        std::uint32_t block;
    };
    std::vector<BufferSite> buffer_sites;
    Id p_uni_u32 = 0;
    // Control flow
    std::set<std::uint32_t> block_starts;
    Id lbl_dispatch_merge = 0, lbl_loop_merge = 0;
    // Forward-only programs skip the PC dispatcher (emit_dag): each block runs
    // under a "reached" flag that the jump leaving its predecessor sets.
    bool dag = false;
    bool ordered = false;  // instructions run in emission order: straight-line or forward-only (cb_ssbo soundness)
    std::set<std::uint16_t> lane_set;                         // pair registers (low code) with the lane bit known set
    std::set<std::pair<Id, Id>> lane_pairs;                   // SSA (lo, hi) words with the lane bit known set
    std::map<std::uint32_t, std::set<std::uint16_t>> lane_in; // DAG block entry: meet of the incoming edges
    // The resource paths (sym, lane_syms) the same way. The walk in program
    // order left a block with what the block before it wrote, which is wrong
    // for a block a branch skipped to: a pixel shader that branches over a
    // block reloading s[0:7] (b382dd8f) samples s[4:11] - still its user
    // data - on the other side, and that T# was "not traceable".
    std::map<std::uint32_t, std::map<int, SymVal>> sym_in;
    std::map<std::uint32_t, std::map<std::pair<int, int>, SymVal>> lane_syms_in;
    std::set<std::uint16_t> lane_end;                         // meet over every way out of the program
    bool lane_end_seen = false;
    std::map<std::uint32_t, Id> reach_vars;  // block start -> u32 flag
    Id lbl_block_merge = 0;                  // merge of the guarded block being emitted
    // Programs with loops but no PC transfers: the same guarded blocks, with a
    // loop construct around each loop (emit_dag). A loop spans a backward
    // branch's target to the branch; loops that overlap without nesting are
    // one. A backward jump sets its target's flag and the `again` of the
    // innermost loop holding both, which goes round while it is set.
    struct LoopRegion {
        std::uint32_t start = 0, end = 0;  // the first block; the last backward branch in it
        Id again = 0, lbl_loop = 0, lbl_continue = 0, lbl_exit = 0;
    };
    bool dag_loop = false;
    std::vector<LoopRegion> loop_regions;  // by start, outer before inner
    std::vector<LoopRegion*> open_loops;   // outermost first, while emitting
    std::set<std::uint32_t> back_targets;  // entered around a loop: nothing known of their lanes
    Id fn_main = 0;
    Id fn_xlate = 0;
    Id fn_wqm = 0;
    Id c_zero_u, c_one_u;
    std::uint32_t cur_offset = 0;       // current instruction offset (for diagnostics)
    std::uint32_t block_end_offset = 0;
    bool block_terminated = false;
    std::set<std::string> reported;

    Translator(const Program& p, const TranslateOptions& o) : prog(p), opt(o) {}

    void error(const std::string& what) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%06x: ", cur_offset);
        const std::string msg = buf + what;
        if (reported.insert(what).second) {
            res.errors.push_back(msg);
        }
    }

    // ---- types
    void setup_types() {
        t_void = m.type_void();
        t_bool = m.type_bool();
        t_u32 = m.type_int(32, false);
        t_i32 = m.type_int(32, true);
        t_u64 = m.type_int(64, false);
        t_f32 = m.type_float(32);
        t_v2u = m.type_vector(t_u32, 2);
        t_v3u = m.type_vector(t_u32, 3);
        t_v4u = m.type_vector(t_u32, 4);
        t_v2f = m.type_vector(t_f32, 2);
        t_v3f = m.type_vector(t_f32, 3);
        t_v4f = m.type_vector(t_f32, 4);
        t_v4b = m.type_vector(t_bool, 4);
        t_v2i = m.type_vector(t_i32, 2);
        t_v3i = m.type_vector(t_i32, 3);
        t_v4i = m.type_vector(t_i32, 4);
        p_fn_u32 = m.type_pointer(spv::ScFunction, t_u32);
        p_fn_bool = m.type_pointer(spv::ScFunction, t_bool);
        p_fn_u64 = m.type_pointer(spv::ScFunction, t_u64);
        p_psb_u32 = m.type_pointer(spv::ScPhysicalStorageBuffer, t_u32);
        t_psb_block = m.type_struct({t_u32});
        m.decorate(t_psb_block, spv::DecBlock);
        m.member_decorate(t_psb_block, 0, spv::DecOffset, {0});
        p_psb_block = m.type_pointer(spv::ScPhysicalStorageBuffer, t_psb_block);
        t_psb_block64 = m.type_struct({t_u64});
        m.decorate(t_psb_block64, spv::DecBlock);
        m.member_decorate(t_psb_block64, 0, spv::DecOffset, {0});
        p_psb_block64 = m.type_pointer(spv::ScPhysicalStorageBuffer, t_psb_block64);
        p_psb_u64 = m.type_pointer(spv::ScPhysicalStorageBuffer, t_u64);
        p_in_v4f = m.type_pointer(spv::ScInput, t_v4f);
        p_in_v4u = m.type_pointer(spv::ScInput, t_v4u);
        p_in_u32 = m.type_pointer(spv::ScInput, t_u32);
        p_in_f32 = m.type_pointer(spv::ScInput, t_f32);
        p_in_v3u = m.type_pointer(spv::ScInput, t_v3u);
        p_in_bool = m.type_pointer(spv::ScInput, t_bool);
        p_out_v4f = m.type_pointer(spv::ScOutput, t_v4f);
        p_out_f32 = m.type_pointer(spv::ScOutput, t_f32);
        p_uc_sampler = m.type_pointer(spv::ScUniformConstant, m.type_sampler());
        t_fn_main = m.type_function(t_void, {});
        c_zero_u = m.const_u32(0);
        c_one_u = m.const_u32(1);
    }

    // ---- register access
    Id sgpr_var(int i) {
        auto it = sgpr_vars.find(i);
        if (it != sgpr_vars.end()) return it->second;
        const Id v = m.local_variable(p_fn_u32, c_zero_u);
        m.name(v, "s" + std::to_string(i));
        sgpr_vars[i] = v;
        return v;
    }
    Id vgpr_var(int i) {
        auto it = vgpr_vars.find(i);
        if (it != vgpr_vars.end()) return it->second;
        const Id v = m.local_variable(p_fn_u32, c_zero_u);
        m.name(v, "v" + std::to_string(i));
        vgpr_vars[i] = v;
        return v;
    }
    Id ld(Id var) { return m.load(t_u32, var); }
    Id ldb(Id var) { return m.load(t_bool, var); }

    Id f(Id u) { return m.emit(spv::OpBitcast, t_f32, {u}); }
    Id u(Id fl) { return m.emit(spv::OpBitcast, t_u32, {fl}); }
    Id i(Id x) { return m.emit(spv::OpBitcast, t_i32, {x}); }
    Id ui(Id x) { return m.emit(spv::OpBitcast, t_u32, {x}); }
    Id b2u(Id b) { return m.emit(spv::OpSelect, t_u32, {b, c_one_u, c_zero_u}); }
    Id b2mask(Id b) { return m.emit(spv::OpSelect, t_u32, {b, m.const_u32(0xffffffffu), c_zero_u}); }
    Id u2b(Id x) { return m.emit(spv::OpINotEqual, t_bool, {x, c_zero_u}); }
    Id cf(float v) { return m.const_f32(v); }
    Id cu(std::uint32_t v) { return m.const_u32(v); }

    Id lane() { return ld(v_lane); }
    // Bit of a 64-bit mask (lo, hi) for this lane.
    Id mask_bit(Id lo, Id hi) {
        const Id l = lane();
        const Id lt32 = m.emit(spv::OpULessThan, t_bool, {l, cu(32)});
        const Id word = m.emit(spv::OpSelect, t_u32, {lt32, lo, hi});
        const Id sh = m.emit(spv::OpBitwiseAnd, t_u32, {l, cu(31)});
        const Id bit = m.emit(spv::OpBitwiseAnd, t_u32, {m.emit(spv::OpShiftRightLogical, t_u32, {word, sh}), c_one_u});
        return u2b(bit);
    }
    // Ballot of a per-lane bool -> (lo, hi).
    void ballot(Id cond, Id& lo, Id& hi) {
        const Id b = m.emit(spv::OpGroupNonUniformBallot, t_v4u, {cu(spv::ScopeSubgroup), cond});
        lo = m.emit(spv::OpCompositeExtract, t_u32, {b, 0});
        hi = m.emit(spv::OpCompositeExtract, t_u32, {b, 1});
    }
    Id exec_bit() { return ldb(v_exec_bit); }
    Id vcc_bit() { return ldb(v_vcc_bit); }
    void refresh_exec_bit() { m.store(v_exec_bit, mask_bit(ld(v_exec_lo), ld(v_exec_hi))); }
    void refresh_vcc_bit() { m.store(v_vcc_bit, mask_bit(ld(v_vcc_lo), ld(v_vcc_hi))); }

    // Static EXEC knowledge. lane_set holds the 64-bit registers whose running
    // lane's bit is known set; with kExecLo in it EXEC masking changes nothing,
    // so masked writes are plain stores. Every lane proves the same facts, so
    // exec_cond() may also stand in for the EXEC bit inside ballots.
    bool exec_known() const { return opt.exec_known && lane_set.count(kExecLo) != 0; }
    Id exec_cond() { return exec_known() ? m.const_bool(true) : exec_bit(); }
    template <typename F>
    void exec_guard(F body) {
        if (exec_known()) {
            body();
        } else {
            if_then(exec_bit(), body);
        }
    }
    void lane_forget(std::uint16_t code) {
        lane_set.erase(code);
        lane_set.erase(static_cast<std::uint16_t>(code - 1));
    }
    static void lane_meet(std::set<std::uint16_t>& into, const std::set<std::uint16_t>& from) {
        for (auto it = into.begin(); it != into.end();) it = from.count(*it) ? std::next(it) : into.erase(it);
    }
    template <typename K>
    static void sym_meet(std::map<K, SymVal>& into, const std::map<K, SymVal>& from) {
        for (auto it = into.begin(); it != into.end();) {
            const auto jt = from.find(it->first);
            it = jt != from.end() && sym_key(jt->second) == sym_key(it->second) ? std::next(it) : into.erase(it);
        }
    }
    void lane_edge(std::uint32_t target) {  // forward DAG edge
        auto [s, sfresh] = sym_in.try_emplace(target, sym);
        if (!sfresh) sym_meet(s->second, sym);
        auto [l, lfresh] = lane_syms_in.try_emplace(target, lane_syms);
        if (!lfresh) sym_meet(l->second, lane_syms);
        auto [it, fresh] = lane_in.try_emplace(target, lane_set);
        if (!fresh) lane_meet(it->second, lane_set);
    }
    void lane_exit() {
        if (lane_end_seen) {
            lane_meet(lane_end, lane_set);
        } else {
            lane_end = lane_set;
            lane_end_seen = true;
        }
    }

    // Read a 32-bit scalar operand code.
    Id read_s(std::uint16_t code, const Inst& in) {
        if (code < 104) return ld(sgpr_var(code));
        if (code >= 256) return ld(vgpr_var(code - 256));
        if (code == kVccLo) return ld(v_vcc_lo);
        if (code == kVccHi) return ld(v_vcc_hi);
        if (code == kExecLo) return ld(v_exec_lo);
        if (code == kExecHi) return ld(v_exec_hi);
        if (code == kM0) return ld(v_m0);
        if (code == kScc) return b2u(ldb(v_scc));
        if (code == kVccz) return b2u(m.emit(spv::OpIEqual, t_bool, {m.emit(spv::OpBitwiseOr, t_u32, {ld(v_vcc_lo), ld(v_vcc_hi)}), c_zero_u}));
        if (code == kExecz) return b2u(m.emit(spv::OpIEqual, t_bool, {m.emit(spv::OpBitwiseOr, t_u32, {ld(v_exec_lo), ld(v_exec_hi)}), c_zero_u}));
        if (code == kLiteral) return cu(in.literal);
        if (code == 128) return c_zero_u;
        if (code >= 129 && code <= 192) return cu(code - 128);
        if (code >= 193 && code <= 208) return cu(static_cast<std::uint32_t>(-static_cast<int>(code - 192)));
        switch (code) {
        case 240: return u(cf(0.5f));
        case 241: return u(cf(-0.5f));
        case 242: return u(cf(1.0f));
        case 243: return u(cf(-1.0f));
        case 244: return u(cf(2.0f));
        case 245: return u(cf(-2.0f));
        case 246: return u(cf(4.0f));
        case 247: return u(cf(-4.0f));
        case 108: case 109: case 110: case 111:  // tba/tma
            return c_zero_u;
        default:
            break;
        }
        if (code >= 112 && code < 124) return ld(sgpr_var(200 + code - 112));  // ttmp as extra sgprs
        error("unsupported operand code " + std::to_string(code));
        return c_zero_u;
    }
    // 64-bit operand: (lo, hi) words.
    void read_pair(std::uint16_t code, const Inst& in, Id& lo, Id& hi) {
        read_pair_raw(code, in, lo, hi);
        if (lane_set.count(code)) lane_pairs.insert({lo, hi});
    }
    void read_pair_raw(std::uint16_t code, const Inst& in, Id& lo, Id& hi) {
        if (code < 104 || code >= 256 || (code >= 112 && code < 124)) {
            lo = read_s(code, in);
            hi = read_s(code + 1, in);
            return;
        }
        if (code == kVccLo) { lo = ld(v_vcc_lo); hi = ld(v_vcc_hi); return; }
        if (code == kExecLo) { lo = ld(v_exec_lo); hi = ld(v_exec_hi); return; }
        if (code == kLiteral) { lo = cu(in.literal); hi = c_zero_u; return; }  // literal is sign-extended? no: zero-extended for b64 ops
        if (code == 128) { lo = c_zero_u; hi = c_zero_u; return; }
        if (code >= 129 && code <= 192) { lo = cu(code - 128); hi = c_zero_u; return; }
        if (code >= 193 && code <= 208) { lo = cu(static_cast<std::uint32_t>(-static_cast<int>(code - 192))); hi = cu(0xffffffffu); return; }
        if (code == kScc) { lo = b2u(ldb(v_scc)); hi = c_zero_u; return; }
        error("unsupported 64-bit operand code " + std::to_string(code));
        lo = hi = c_zero_u;
    }
    Id read_u64(std::uint16_t code, const Inst& in) {
        Id lo, hi;
        read_pair(code, in, lo, hi);
        return make_u64(lo, hi);
    }
    Id make_u64(Id lo, Id hi) {
        const Id lo64 = m.emit(spv::OpUConvert, t_u64, {lo});
        const Id hi64 = m.emit(spv::OpUConvert, t_u64, {hi});
        return m.emit(spv::OpBitwiseOr, t_u64, {lo64, m.emit(spv::OpShiftLeftLogical, t_u64, {hi64, m.const_u64(32)})});
    }
    void split_u64(Id v, Id& lo, Id& hi) {
        lo = m.emit(spv::OpUConvert, t_u32, {v});
        hi = m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpShiftRightLogical, t_u64, {v, m.const_u64(32)})});
    }

    void write_s(std::uint16_t code, Id value) {
        lane_forget(code);
        if (code < 104) {
            m.store(sgpr_var(code), value);
            sym.erase(code);
            if (!in_smrd_write) sgpr_defs[code].push_back({def_seq++, "?", cur_block});
            return;
        }
        if (code == kVccLo) { m.store(v_vcc_lo, value); refresh_vcc_bit(); return; }
        if (code == kVccHi) { m.store(v_vcc_hi, value); refresh_vcc_bit(); return; }
        if (code == kExecLo) { m.store(v_exec_lo, value); refresh_exec_bit(); return; }
        if (code == kExecHi) { m.store(v_exec_hi, value); refresh_exec_bit(); return; }
        if (code == kM0) { m.store(v_m0, value); return; }
        if (code >= 112 && code < 124) { m.store(sgpr_var(200 + code - 112), value); return; }
        if (code >= 108 && code <= 111) return;
        error("unsupported scalar destination " + std::to_string(code));
    }
    void write_pair(std::uint16_t code, Id lo, Id hi) {
        const bool lane = lane_pairs.count({lo, hi}) != 0;
        if (code == kVccLo) {
            m.store(v_vcc_lo, lo); m.store(v_vcc_hi, hi); refresh_vcc_bit();
        } else if (code == kExecLo) {
            m.store(v_exec_lo, lo); m.store(v_exec_hi, hi); refresh_exec_bit();
        } else {
            write_s(code, lo);
            write_s(code + 1, hi);
        }
        lane_forget(code);
        lane_forget(static_cast<std::uint16_t>(code + 1));
        if (lane) lane_set.insert(code);
    }
    // SGPR spills. Short of SGPRs, the compiler parks a scalar in a lane of a
    // VGPR (v_writelane_b32 vN, sK, lane) and takes it back later
    // (v_readlane_b32 sD, vN, lane). The resource path rides along, so a T#
    // or S# that went through a spill is still traceable to user data: a
    // pixel shader whose sampler came back from lane 10 of v3 (ab702c43) was
    // "not traceable", and its pipeline and every draw of it were refused.
    // Keyed by (VGPR, lane); any other write to the VGPR forgets its lanes.
    std::map<std::pair<int, int>, SymVal> lane_syms;
    // The value itself (gcn/wave.h, spill cells): a v_readlane_b32 that only
    // the v_writelane_b32s of its (VGPR, lane) can reach since the last other
    // write to the VGPR reads the scalar they stored, not another invocation
    // - which holds it only where lane L has a pixel. BBHOST_SPILL_CELLS=0:
    // every read is a shuffle, as before.
    SpillCells spill;
    std::map<std::pair<int, int>, Id> cell_vars;
    Id spill_cell(int vgpr, int lane) {
        auto [it, fresh] = cell_vars.try_emplace({vgpr, lane}, 0);
        if (fresh) {
            it->second = m.local_variable(p_fn_u32, c_zero_u);
            m.name(it->second, "cell_v" + std::to_string(vgpr) + "_" + std::to_string(lane));
        }
        return it->second;
    }
    static bool const_lane(std::uint16_t code, int& lane) {
        if (code < 128 || code > 192) return false;  // an inline integer 0..64
        lane = code - 128;
        return lane < 64;
    }
    void forget_lanes(int vgpr) {
        for (auto it = lane_syms.lower_bound({vgpr, 0}); it != lane_syms.end() && it->first.first == vgpr;) it = lane_syms.erase(it);
    }
    // v_cvt_pkrtz_f16_f32's two floats, by the VGPR that holds their pack, for as
    // long as it does and the block that packed them is the current one: a
    // compressed export of the VGPR rounds them (emit_rtz_half) rather than
    // decoding the pack - where the pack was written: under a varying EXEC
    // (`exec`, the bit its write selected on) only where that bit is set.
    struct PackedPair {
        Id lo, hi;
        std::uint32_t block;
        Id exec = 0;
    };
    std::map<int, PackedPair> packed_pairs;
    // The device rounds f32 to f16 toward zero itself (native_half_rtz, the
    // module under RoundingModeRTZ 16): the f16 pair type for its conversions.
    bool native_rtz = false;
    Id t_v2h = 0;
    // A float pair rounded toward zero to half precision and back, natively.
    Id native_rtz_pair(Id v2f) { return m.emit(spv::OpFConvert, t_v2f, {m.emit(spv::OpFConvert, t_v2h, {v2f})}); }
    // VGPR write predicated on EXEC (a plain store where the lane bit is known set).
    Id last_write_exec = 0;  // the EXEC bit the last write_v selected on (0: EXEC known set)
    void write_v(int idx, Id value) {
        if (!lane_syms.empty()) forget_lanes(idx);
        if (!packed_pairs.empty()) packed_pairs.erase(idx);
        const Id var = vgpr_var(idx);
        last_write_exec = exec_known() ? 0 : exec_bit();
        m.store(var, last_write_exec ? m.emit(spv::OpSelect, t_u32, {last_write_exec, value, ld(var)}) : value);
    }
    // VALU destination that may be a VGPR (VOP1/2) — same thing, kept for clarity.
    void write_vdst(const Inst& in, Id value) { write_v(in.dst, value); }
    void set_scc(Id b) { m.store(v_scc, b); }

    // ---- SPIR-V helpers
    Id fadd(Id a, Id b) { return m.emit(spv::OpFAdd, t_f32, {a, b}); }
    Id fsub(Id a, Id b) { return m.emit(spv::OpFSub, t_f32, {a, b}); }
    Id fmul(Id a, Id b) { return m.emit(spv::OpFMul, t_f32, {a, b}); }
    Id fdiv(Id a, Id b) { return m.emit(spv::OpFDiv, t_f32, {a, b}); }
    Id fneg(Id a) { return m.emit(spv::OpFNegate, t_f32, {a}); }
    Id fabs_(Id a) { return m.ext_inst(t_f32, spv::GlslFAbs, {a}); }
    Id fma_(Id a, Id b, Id c) { return m.ext_inst(t_f32, spv::GlslFma, {a, b, c}); }
    Id fmin_(Id a, Id b) { return m.ext_inst(t_f32, spv::GlslNMin, {a, b}); }
    Id fmax_(Id a, Id b) { return m.ext_inst(t_f32, spv::GlslNMax, {a, b}); }
    Id iadd(Id a, Id b) { return m.emit(spv::OpIAdd, t_u32, {a, b}); }
    Id isub(Id a, Id b) { return m.emit(spv::OpISub, t_u32, {a, b}); }
    Id imul(Id a, Id b) { return m.emit(spv::OpIMul, t_u32, {a, b}); }
    Id iand(Id a, Id b) { return m.emit(spv::OpBitwiseAnd, t_u32, {a, b}); }
    Id ior(Id a, Id b) { return m.emit(spv::OpBitwiseOr, t_u32, {a, b}); }
    Id ixor(Id a, Id b) { return m.emit(spv::OpBitwiseXor, t_u32, {a, b}); }
    Id inot(Id a) { return m.emit(spv::OpNot, t_u32, {a}); }
    Id shl(Id a, Id b) { return m.emit(spv::OpShiftLeftLogical, t_u32, {a, iand(b, cu(31))}); }
    Id shr(Id a, Id b) { return m.emit(spv::OpShiftRightLogical, t_u32, {a, iand(b, cu(31))}); }
    Id sar(Id a, Id b) { return m.emit(spv::OpShiftRightArithmetic, t_u32, {a, iand(b, cu(31))}); }
    Id sel(Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_u32, {c, a, b}); }
    Id fsel(Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_f32, {c, a, b}); }
    Id band(Id a, Id b) { return m.emit(spv::OpLogicalAnd, t_bool, {a, b}); }
    Id bor(Id a, Id b) { return m.emit(spv::OpLogicalOr, t_bool, {a, b}); }
    Id bnot(Id a) { return m.emit(spv::OpLogicalNot, t_bool, {a}); }
    Id feq(Id a, Id b) { return m.emit(spv::OpFOrdEqual, t_bool, {a, b}); }
    Id flt(Id a, Id b) { return m.emit(spv::OpFOrdLessThan, t_bool, {a, b}); }
    Id fgt(Id a, Id b) { return m.emit(spv::OpFOrdGreaterThan, t_bool, {a, b}); }
    Id fle(Id a, Id b) { return m.emit(spv::OpFOrdLessThanEqual, t_bool, {a, b}); }
    Id fge(Id a, Id b) { return m.emit(spv::OpFOrdGreaterThanEqual, t_bool, {a, b}); }
    Id ieq(Id a, Id b) { return m.emit(spv::OpIEqual, t_bool, {a, b}); }
    Id ine(Id a, Id b) { return m.emit(spv::OpINotEqual, t_bool, {a, b}); }
    Id ult(Id a, Id b) { return m.emit(spv::OpULessThan, t_bool, {a, b}); }
    Id ugt(Id a, Id b) { return m.emit(spv::OpUGreaterThan, t_bool, {a, b}); }
    Id ule(Id a, Id b) { return m.emit(spv::OpULessThanEqual, t_bool, {a, b}); }
    Id uge(Id a, Id b) { return m.emit(spv::OpUGreaterThanEqual, t_bool, {a, b}); }
    Id slt(Id a, Id b) { return m.emit(spv::OpSLessThan, t_bool, {a, b}); }
    Id sgt(Id a, Id b) { return m.emit(spv::OpSGreaterThan, t_bool, {a, b}); }
    Id sle(Id a, Id b) { return m.emit(spv::OpSLessThanEqual, t_bool, {a, b}); }
    Id sge(Id a, Id b) { return m.emit(spv::OpSGreaterThanEqual, t_bool, {a, b}); }
    Id isnan(Id a) { return m.emit(spv::OpIsNan, t_bool, {a}); }
    Id popcnt(Id a) { return m.emit(spv::OpBitCount, t_u32, {a}); }
    Id sext24(Id a) { return ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(a), c_zero_u, cu(24)})); }
    Id zext24(Id a) { return iand(a, cu(0xffffff)); }
    Id u2f(Id a) { return m.emit(spv::OpConvertUToF, t_f32, {a}); }
    Id i2f(Id a) { return m.emit(spv::OpConvertSToF, t_f32, {i(a)}); }
    Id f2u(Id a) { return m.emit(spv::OpConvertFToU, t_u32, {a}); }
    Id f2i(Id a) { return ui(m.emit(spv::OpConvertFToS, t_i32, {a})); }
    Id vec4f(Id a, Id b, Id c, Id d) { return m.emit(spv::OpCompositeConstruct, t_v4f, {a, b, c, d}); }
    SpvScalarTypes half_types() const { return {t_bool, t_u32, t_i32, t_f32}; }
    Id extract(Id type, Id v, std::uint32_t k) { return m.emit(spv::OpCompositeExtract, type, {v, k}); }
    Id ubfe(Id v, std::uint32_t off, std::uint32_t n) { return m.emit(spv::OpBitFieldUExtract, t_u32, {v, cu(off), cu(n)}); }

    // 64-bit scalar ops on (lo,hi) pairs via u64.
    Id pair64(Id lo, Id hi) { return make_u64(lo, hi); }

    // Physical storage buffer access.
    Id guest_ptr(Id guest_u64) {
        const Id host = m.emit(spv::OpFunctionCall, t_u64, {fn_xlate, guest_u64});
        const Id block = m.emit(spv::OpConvertUToPtr, p_psb_block, {host});
        return m.access_chain(p_psb_u32, block, {c_zero_u});
    }
    Id load_u32_at(Id guest_u64) {
        const Id p = guest_ptr(guest_u64);
        return m.emit(spv::OpLoad, t_u32, {p, 2u /* Aligned */, 4u});
    }
    void store_u32_at(Id guest_u64, Id value) {
        const Id p = guest_ptr(guest_u64);
        m.emit_void(spv::OpStore, {p, value, 2u, 4u});
    }
    // n consecutive dwords with one page-table walk when they sit in one page,
    // which a resource table or constant block nearly always does. A walk per
    // dword was most of a deferred light's memory traffic (136 walks per pixel
    // for 24 load instructions). Across a page boundary the next page may be a
    // different import, so that case still walks per dword.
    std::vector<Id> load_u32_run(Id guest_u64, int n) {
        if (n == 1) return {load_u32_at(guest_u64)};
        std::vector<Id> vars(static_cast<std::size_t>(n));
        for (Id& v : vars) v = m.local_variable(p_fn_u32, c_zero_u);
        const Id within = m.emit(spv::OpBitwiseAnd, t_u64, {guest_u64, m.const_u64((1ull << kPageShift) - 1)});
        const Id end = m.emit(spv::OpIAdd, t_u64, {within, m.const_u64(static_cast<std::uint64_t>(n) * 4)});
        const Id one_page = m.emit(spv::OpULessThanEqual, t_bool, {end, m.const_u64(1ull << kPageShift)});
        const Id l_one = m.fresh(), l_split = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {one_page, l_one, l_split});
        m.label(l_one);
        const Id host = m.emit(spv::OpFunctionCall, t_u64, {fn_xlate, guest_u64});
        for (int k = 0; k < n; ++k) {
            const Id at = k ? m.emit(spv::OpIAdd, t_u64, {host, m.const_u64(static_cast<std::uint64_t>(k) * 4)}) : host;
            const Id block = m.emit(spv::OpConvertUToPtr, p_psb_block, {at});
            m.store(vars[k], m.emit(spv::OpLoad, t_u32, {m.access_chain(p_psb_u32, block, {c_zero_u}), 2u /* Aligned */, 4u}));
        }
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_split);
        for (int k = 0; k < n; ++k) m.store(vars[k], load_u32_at(addr_addc(guest_u64, k * 4)));
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_merge);
        std::vector<Id> vals;
        for (Id v : vars) vals.push_back(ld(v));
        return vals;
    }
    // n consecutive dwords from a byte address of any alignment: the aligned
    // dwords that cover them, shifted together. A load the SPIR-V marks
    // Aligned 4 at an address that is not (an 8- or 16-bit element at an odd
    // place: the game's buffer-to-image copy reads one byte a thread) is
    // undefined - NVIDIA read it as asked, AMD's older GPUs force the address
    // down or fault.
    std::vector<Id> load_u32_any(Id guest_u64, int n) {
        const Id low = m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpBitwiseAnd, t_u64, {guest_u64, m.const_u64(3)})});
        const Id aligned = m.emit(spv::OpBitwiseAnd, t_u64, {guest_u64, m.const_u64(~3ull)});
        std::vector<Id> vars(static_cast<std::size_t>(n));
        for (Id& v : vars) v = m.local_variable(p_fn_u32, c_zero_u);
        const Id is_aligned = ieq(low, c_zero_u);  // before the merge: nothing may come between it and its branch
        const Id l_aligned = m.fresh(), l_shifted = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {is_aligned, l_aligned, l_shifted});
        m.label(l_aligned);
        {
            const std::vector<Id> w = load_u32_run(aligned, n);
            for (int k = 0; k < n; ++k) m.store(vars[static_cast<std::size_t>(k)], w[static_cast<std::size_t>(k)]);
        }
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_shifted);
        {
            const std::vector<Id> w = load_u32_run(aligned, n + 1);
            const Id sh = shl(low, cu(3));  // 8, 16 or 24
            const Id rsh = isub(cu(32), sh);
            for (int k = 0; k < n; ++k) {
                m.store(vars[static_cast<std::size_t>(k)], ior(shr(w[static_cast<std::size_t>(k)], sh), shl(w[static_cast<std::size_t>(k) + 1], rsh)));
            }
        }
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_merge);
        std::vector<Id> vals;
        for (Id v : vars) vals.push_back(ld(v));
        return vals;
    }
    // The bytes of (v0, v1) - a 64-bit value, low dword first - that the mask
    // (m0, m1) selects, stored at a byte address of any alignment through the
    // aligned dwords they cover (up to three). A dword covered whole takes a
    // plain store; one covered in part an atomic and/or, so the bytes of it
    // other invocations store - the neighbouring 8- or 16-bit elements, which
    // a read-modify-write would race with - survive.
    void store_masked_any(Id guest_u64, Id v0, Id v1, Id m0, Id m1) {
        const Id low = m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpBitwiseAnd, t_u64, {guest_u64, m.const_u64(3)})});
        const Id aligned = m.emit(spv::OpBitwiseAnd, t_u64, {guest_u64, m.const_u64(~3ull)});
        const Id sh = shl(low, cu(3));
        const Id rsh = isub(cu(32), sh);
        const Id whole = ieq(sh, c_zero_u);
        const Id dv[3] = {shl(v0, sh), sel(whole, v1, ior(shl(v1, sh), shr(v0, rsh))), sel(whole, c_zero_u, shr(v1, rsh))};
        const Id dm[3] = {shl(m0, sh), sel(whole, m1, ior(shl(m1, sh), shr(m0, rsh))), sel(whole, c_zero_u, shr(m1, rsh))};
        for (int k = 0; k < 3; ++k) {
            const Id at = addr_addc(aligned, static_cast<std::uint32_t>(k) * 4);
            const Id full = ieq(dm[k], cu(0xffffffffu));
            if_then(full, [&] { store_u32_at(at, dv[k]); });
            if_then(m.emit(spv::OpLogicalAnd, t_bool, {ine(dm[k], c_zero_u), m.emit(spv::OpLogicalNot, t_bool, {full})}), [&] {
                const Id p = guest_ptr(at);
                m.emit(spv::OpAtomicAnd, t_u32, {p, cu(spv::ScopeDevice), cu(spv::MsNone), inot(dm[k])});
                m.emit(spv::OpAtomicOr, t_u32, {p, cu(spv::ScopeDevice), cu(spv::MsNone), iand(dv[k], dm[k])});
            });
        }
    }
    // A page-table walk the shader keeps (TranslateResult::walks).
    void note_walk(const std::string& reason) { ++res.walks[reason]; }
    // How a buffer element load or store reaches its V# and addresses it, for
    // TranslateResult::walks.
    std::string buffer_walk_form(const Inst& in) {
        ResourcePath p;
        std::string s = !sym_path(in.srsrc, 4, p) ? " from an untraced V#" : p.immediate ? " from a V# in user data" : " from a V# in a table";
        const bool constant_offset = in.soffset >= 128 && in.soffset <= 208;  // register codes of the constants 0, 1..64, -1..-16
        return s + (in.idxen || in.offen ? " indexed" : constant_offset ? " at a constant offset" : " at a register offset");
    }

    // ---- constant buffers through storage-buffer bindings (TranslateOptions::cb_ssbo)
    std::size_t buffer_for(const ResourcePath& path, const std::string& key, bool pointer, std::uint32_t end_dw) {
        auto it = buffer_index.find(key);
        if (it != buffer_index.end()) {
            BufferBinding& b = res.buffers[it->second];
            b.max_dw = std::max(b.max_dw, end_dw);
            return it->second;
        }
        if (!p_ssbo_block) {
            const Id t_words = m.type_runtime_array(t_u32);
            m.decorate(t_words, spv::DecArrayStride, {4});
            const Id t_block = m.type_struct({t_words});
            m.decorate(t_block, spv::DecBlock);
            m.member_decorate(t_block, 0, spv::DecOffset, {0});
            p_ssbo_block = m.type_pointer(spv::ScStorageBuffer, t_block);
            p_ssbo_u32 = m.type_pointer(spv::ScStorageBuffer, t_u32);
        }
        const std::size_t index = res.buffers.size();
        BufferBinding b;
        b.path = path;
        b.pointer = pointer;
        b.binding = opt.cb_binding_base + static_cast<std::uint32_t>(index);
        b.max_dw = end_dw;
        const Id var = m.global_variable(p_ssbo_block, spv::ScStorageBuffer);
        m.decorate(var, spv::DecDescriptorSet,
                   {opt.cb_descriptor_set < 0 ? opt.descriptor_set : static_cast<std::uint32_t>(opt.cb_descriptor_set)});
        m.decorate(var, spv::DecBinding, {b.binding});
        m.name(var, "cb" + std::to_string(index));
        interface.push_back(var);
        res.buffers.push_back(b);
        buffer_vars.push_back(var);
        buffer_index[key] = index;
        return index;
    }
    // n dwords at `off_dw` of the constant buffer whose V# (or, with `pointer`,
    // whose address) sits in SGPRs [reg, reg + 1], traced to `path`: from its
    // storage-buffer binding when the host bound it, otherwise through the page
    // table like any other load.
    std::vector<Id> load_buffer_run(const ResourcePath& path, const std::string& key, bool pointer, int reg, std::uint32_t off_dw,
                                    int n, Id guest_u64) {
        const std::size_t index = buffer_for(path, key, pointer, off_dw + static_cast<std::uint32_t>(n));
        return read_bound(index, reg, cu(off_dw), n, guest_u64);
    }
    // StageParams member `member` (an array of 8 dwords in uvec4s), entry `index`.
    Id params_u32(std::uint32_t member, std::size_t index) {
        return m.load(t_u32, m.access_chain(p_uni_u32, ubo_var,
                                            {cu(member), cu(static_cast<std::uint32_t>(index / 4)), cu(static_cast<std::uint32_t>(index % 4))}));
    }
    Id buffer_bound(std::size_t index) {
        const Id valid_bits = m.load(t_u32, params_head_v4 ? m.access_chain(p_uni_u32, ubo_var, {cu(3), c_zero_u})
                                                           : m.access_chain(p_uni_u32, ubo_var, {cu(3)}));
        return ine(iand(shr(valid_bits, cu(static_cast<std::uint32_t>(index))), c_one_u), c_zero_u);
    }
    // n dwords of buffers[index] from dword `first_dw` of the V# (whose base
    // sits in SGPRs [reg, reg + 1]): from the storage-buffer binding when the
    // host bound it, otherwise the page-table walk at `guest_u64`.
    std::vector<Id> read_bound(std::size_t index, int reg, Id first_dw, int n, Id guest_u64) {
        res.buffer_at[cur_offset] = static_cast<std::uint32_t>(index);
        buffer_sites.push_back({reg, index, sym_key(sym_of(reg)), sym_key(sym_of(reg + 1)), def_seq++, cur_block});
        if (opt.cb_no_fallback) {
            const Id bias = params_u32(4, index);
            std::vector<Id> vals;
            for (int k = 0; k < n; ++k) {
                const Id at = iadd(bias, iadd(first_dw, cu(static_cast<std::uint32_t>(k))));
                vals.push_back(m.load(t_u32, m.access_chain(p_ssbo_u32, buffer_vars[index], {c_zero_u, at})));
            }
            return vals;
        }
        std::vector<Id> vars(static_cast<std::size_t>(n));
        for (Id& v : vars) v = m.local_variable(p_fn_u32, c_zero_u);
        const Id valid = buffer_bound(index);
        const Id l_bound = m.fresh(), l_walk = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {valid, l_bound, l_walk});
        m.label(l_bound);
        const Id bias = params_u32(4, index);
        for (int k = 0; k < n; ++k) {
            const Id at = iadd(bias, iadd(first_dw, cu(static_cast<std::uint32_t>(k))));
            m.store(vars[static_cast<std::size_t>(k)], m.load(t_u32, m.access_chain(p_ssbo_u32, buffer_vars[index], {c_zero_u, at})));
        }
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_walk);
        const std::vector<Id> walked = load_u32_run(guest_u64, n);
        for (int k = 0; k < n; ++k) m.store(vars[static_cast<std::size_t>(k)], walked[static_cast<std::size_t>(k)]);
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_merge);
        std::vector<Id> vals;
        for (Id v : vars) vals.push_back(ld(v));
        return vals;
    }
    Id addr_add(Id base64, Id off32) { return m.emit(spv::OpIAdd, t_u64, {base64, m.emit(spv::OpUConvert, t_u64, {off32})}); }
    Id addr_addc(Id base64, std::uint32_t off) { return off ? m.emit(spv::OpIAdd, t_u64, {base64, m.const_u64(off)}) : base64; }

    // Guarded region: emits `if (cond) { body }` as a structured selection.
    template <typename F>
    void if_then(Id cond, F body) {
        const Id l_then = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {cond, l_then, l_merge});
        m.label(l_then);
        body();
        m.emit_void(spv::OpBranch, {l_merge});
        m.label(l_merge);
    }

    // ---- resource tracking (symbolic SGPR values)
    void sym_user_data() {
        const int n = user_sgpr_count();
        for (int k = 0; k < n; ++k) {
            SymVal v;
            v.kind = SymVal::UserData;
            v.user_sgpr = k;
            sym[k] = v;
        }
    }
    int user_sgpr_count() const { return static_cast<int>((opt.rsrc2 >> 1) & 0x1f); }
    // A symbolic SGPR value as a string: equal strings are the same value
    // throughout one draw.
    static std::string sym_key(const SymVal& v) {
        if (v.kind == SymVal::Unknown) return "?";
        std::string s = (v.kind == SymVal::UserData ? "u" : "l") + std::to_string(v.user_sgpr);
        for (const ResourceStep& st : v.loads) s += "/" + std::to_string(st.offset_dw) + (st.vsharp ? "v" : "p");
        return s + "#" + std::to_string(v.dw);
    }
    // Record that s[dst..dst+n) = load(s[base:base+1] + dword offset)
    SymVal sym_of(int sgpr) const {
        auto it = sym.find(sgpr);
        return it == sym.end() ? SymVal{} : it->second;
    }
    // s[dst..dst+n) = load(*b + dword offset); `b` was captured before the
    // destination registers were written (they may overlap the base pair).
    void sym_load(int dst, int n, const SymVal& b, std::uint32_t dw_offset, bool known_offset, bool via_vsharp) {
        for (int k = 0; k < n; ++k) sym.erase(dst + k);
        if (!known_offset || b.kind == SymVal::Unknown) return;
        for (int k = 0; k < n; ++k) {
            SymVal v;
            v.kind = SymVal::Loaded;
            v.user_sgpr = b.user_sgpr;
            v.loads = b.loads;
            if (b.kind == SymVal::Loaded) {
                // The pointer/V# we dereference sits at dword b.dw of the
                // block the previous step loaded.
                v.loads.back().offset_dw += static_cast<std::uint32_t>(b.dw);
            }
            v.loads.push_back({dw_offset, via_vsharp});
            v.dw = k;
            sym[dst + k] = v;
        }
    }
    bool sym_path(int sgpr, int ndw, ResourcePath& path) {
        auto it = sym.find(sgpr);
        if (it == sym.end() || it->second.kind == SymVal::Unknown) return false;
        const SymVal& v = it->second;
        // All ndw registers must come from the same block, consecutive.
        for (int k = 1; k < ndw; ++k) {
            auto jt = sym.find(sgpr + k);
            if (jt == sym.end() || jt->second.kind != v.kind) return false;
            const SymVal& w = jt->second;
            if (v.kind == SymVal::UserData) {
                if (w.user_sgpr != v.user_sgpr + k) return false;
            } else if (w.user_sgpr != v.user_sgpr || w.loads != v.loads || w.dw != v.dw + k) {
                return false;
            }
        }
        path.user_sgpr = v.user_sgpr;
        if (v.kind == SymVal::UserData) {
            path.immediate = true;
            return true;
        }
        // loads: [steps..., final block]; the resource sits at block offset + dw
        path.loads.assign(v.loads.begin(), v.loads.end() - 1);
        path.final_offset_dw = static_cast<int>(v.loads.back().offset_dw) + v.dw;
        path.final_vsharp = v.loads.back().vsharp;
        return true;
    }

    // A store or atomic through the V# in s[srsrc..+3].
    void note_store(const Inst& in) {
        ResourcePath path;
        if (!sym_path(in.srsrc, 4, path)) {
            ++res.untraced_stores;
            return;
        }
        const std::string key = path.str();
        for (const ResourcePath& p : res.store_buffers) {
            if (p.str() == key) return;
        }
        res.store_buffers.push_back(path);
    }

    // ---- images
    ImageVar image_for(const Inst& in, bool storage, bool depth, std::uint32_t& dim_out, bool& arrayed_out) {
        ResourcePath path;
        std::string key;
        if (sym_path(in.srsrc, in.r128 ? 4 : 8, path)) {
            key = path.str();
        } else {
            error("image T# in s[" + std::to_string(in.srsrc) + "] not traceable to user data");
            key = "untraceable#" + std::to_string(in.offset);
        }
        key += storage ? "/storage" : "/sampled";
        key += depth ? "/depth" : "";
        auto it = image_index.find(key);
        if (it != image_index.end()) {
            const ImageBinding& b = res.images[it->second];
            dim_out = b.dim;
            arrayed_out = b.arrayed;
            res.image_at[cur_offset] = static_cast<std::uint32_t>(it->second);
            return image_vars[it->second];
        }
        // Dimensions come from the host's resolved T# for this binding index.
        const std::size_t index = res.images.size();
        std::uint32_t dim = spv::Dim2D;
        bool arrayed = false;
        std::uint32_t kind = 0;
        if (index < opt.image_dims.size()) {
            dim = opt.image_dims[index].first & 0xff;
            kind = (opt.image_dims[index].first >> 8) & 3;
            arrayed = opt.image_dims[index].second;
        }
        if (depth || kind == 3) kind = 0;
        // Sea Islands samples a cube T# as a 2D array of faces, never as a
        // Vulkan cube: the coordinates are a face position and a face id (mimg).
        const bool cube = dim == spv::DimCube;
        if (cube) {
            dim = spv::Dim2D;
            arrayed = true;
        }
        dim_out = dim;
        arrayed_out = arrayed;
        ImageBinding b;
        b.path = path;
        b.sgpr = in.srsrc;
        b.storage = storage;
        b.r128 = in.r128;
        b.da = in.da;
        b.dim = dim;
        b.arrayed = arrayed;
        b.depth = depth;
        b.cube = cube;
        b.kind = kind;
        b.binding = (storage ? kBindingStorageImage0 : kBindingImage0) + static_cast<std::uint32_t>(index);
        if (storage) {
            m.capability(spv::CapStorageImageReadWithoutFormat);
            m.capability(spv::CapStorageImageWriteWithoutFormat);
        }
        if (dim == spv::Dim1D) m.capability(storage ? spv::CapImage1D : spv::CapSampled1D);
        if (dim == spv::DimCube && arrayed) m.capability(spv::CapSampledCubeArray);
        const Id t_img = m.type_image(kind == 1 ? t_u32 : kind == 2 ? t_i32 : t_f32, static_cast<spv::Dim>(dim), depth, arrayed,
                                      false, storage ? 2 : 1);
        Id var;
        if (opt.bindless) {
            // Every binding of one image type reads the same alias of the
            // global array, at the slot the params block gives it.
            auto arr = bindless_image_arrays.find(t_img);
            if (arr == bindless_image_arrays.end()) {
                arr = bindless_image_arrays
                          .emplace(t_img, bindless_array(t_img, storage ? kBindlessStorageImages : kBindlessImages,
                                                         storage ? "storage_images" : "images"))
                          .first;
                m.capability(storage ? spv::CapStorageImageArrayDynamicIndexing : spv::CapSampledImageArrayDynamicIndexing);
            }
            var = arr->second;
        } else {
            const Id p_img = m.type_pointer(spv::ScUniformConstant, t_img);
            var = m.global_variable(p_img, spv::ScUniformConstant);
            m.decorate(var, spv::DecDescriptorSet, {opt.descriptor_set});
            m.decorate(var, spv::DecBinding, {b.binding});
            m.name(var, "img" + std::to_string(index));
            interface.push_back(var);
        }
        res.images.push_back(b);
        image_vars.push_back({var, t_img, cube, kind, opt.bindless, static_cast<std::uint32_t>(index)});
        image_index[key] = image_vars.size() - 1;
        res.image_at[cur_offset] = static_cast<std::uint32_t>(image_vars.size() - 1);
        return image_vars.back();
    }
    // The sampler binding the instruction's S# is (res.samplers index).
    std::size_t sampler_for(const Inst& in, bool compare = false) {
        ResourcePath path;
        std::string key;
        if (sym_path(in.ssamp, 4, path)) {
            key = path.str();
        } else {
            error("sampler S# in s[" + std::to_string(in.ssamp) + "] not traceable to user data");
            key = "untraceable#" + std::to_string(in.offset);
        }
        if (compare) key += "/cmp";
        auto it = sampler_index.find(key);
        if (it != sampler_index.end()) {
            res.sampler_at[cur_offset] = static_cast<std::uint32_t>(it->second);
            return it->second;
        }
        SamplerBinding b;
        b.path = path;
        b.sgpr = in.ssamp;
        b.compare = compare;
        b.binding = kBindingSampler0 + static_cast<std::uint32_t>(res.samplers.size());
        Id var;
        if (opt.bindless) {
            if (!bindless_sampler_array) {
                bindless_sampler_array = bindless_array(m.type_sampler(), kBindlessSamplers, "samplers");
                m.capability(spv::CapSampledImageArrayDynamicIndexing);
            }
            var = bindless_sampler_array;
        } else {
            var = m.global_variable(p_uc_sampler, spv::ScUniformConstant);
            m.decorate(var, spv::DecDescriptorSet, {opt.descriptor_set});
            m.decorate(var, spv::DecBinding, {b.binding});
            m.name(var, "smp" + std::to_string(res.samplers.size()));
            interface.push_back(var);
        }
        res.samplers.push_back(b);
        sampler_vars.push_back(var);
        sampler_index[key] = sampler_vars.size() - 1;
        res.sampler_at[cur_offset] = static_cast<std::uint32_t>(sampler_vars.size() - 1);
        return sampler_vars.size() - 1;
    }

    // ---- inputs/outputs
    Id in_param(int attr) {
        auto it = in_params.find(attr);
        if (it != in_params.end()) return it->second;
        const std::uint32_t location = static_cast<std::size_t>(attr) < opt.ps_input_map.size()
                                           ? opt.ps_input_map[attr]
                                           : static_cast<std::uint32_t>(attr);
        // Inputs the input map sends to one register read one variable: two
        // Input variables at one location are invalid SPIR-V (an input no vertex
        // output matches, 0x20, lands on register 0's location). The first input
        // at a location decides its flat decoration.
        if (auto at = in_location_vars.find(location); at != in_location_vars.end()) return in_params[attr] = at->second;
        const Id var = m.global_variable(p_in_v4f, spv::ScInput);
        m.decorate(var, spv::DecLocation, {location});
        if ((opt.ps_flat_mask >> attr) & 1) m.decorate(var, spv::DecFlat);
        m.name(var, "attr" + std::to_string(attr));
        interface.push_back(var);
        in_params[attr] = var;
        in_location_vars[location] = var;
        return var;
    }
    Id out_param(int n) {
        auto it = out_params.find(n);
        if (it != out_params.end()) return it->second;
        const Id var = m.global_variable(p_out_v4f, spv::ScOutput);
        m.decorate(var, spv::DecLocation, {static_cast<std::uint32_t>(n)});
        m.name(var, (opt.stage == Stage::Pixel ? "mrt" : "param") + std::to_string(n));
        interface.push_back(var);
        out_params[n] = var;
        return var;
    }
    // The outputs a VS export of param n writes: location n, or with
    // link_outputs every location whose pixel-shader input names param n.
    std::vector<Id> param_outputs(int n) {
        if (opt.stage != Stage::Vertex || !opt.link_outputs) return {out_param(n)};
        std::vector<Id> vars;
        for (std::size_t k = 0; k < opt.output_links.size(); ++k) {
            if (opt.output_links[k] == n) vars.push_back(out_param(static_cast<int>(k)));
        }
        return vars;
    }
    Id builtin_in(Id ptr_type, spv::BuiltIn bi, const char* nm) {
        const Id var = m.global_variable(ptr_type, spv::ScInput);
        m.decorate(var, spv::DecBuiltIn, {static_cast<std::uint32_t>(bi)});
        if (opt.stage == Stage::Pixel && bi != spv::BiFragCoord) m.decorate(var, spv::DecFlat);
        m.name(var, nm);
        interface.push_back(var);
        return var;
    }

    // ---- control flow preparation
    static bool is_branch(const Inst& in) {
        return in.enc == Enc::SOPP && (in.op == 2 || (in.op >= 4 && in.op <= 9));
    }
    // TranslateOptions::vertex_input: s_swappc_b64 loads the elements in place
    // instead of transferring to a fetch shader.
    bool vertex_input_call() const { return !opt.fetch && !opt.vertex_input.empty(); }

    void find_blocks(const Program& p, std::uint32_t bias) {
        block_starts.insert(bias);
        const std::uint32_t end = bias + (p.insts.empty() ? 0 : p.insts.back().offset + p.insts.back().size * 4);
        std::set<std::uint32_t> starts;
        for (const Inst& in : p.insts) {
            const std::uint32_t next = bias + in.offset + in.size * 4;
            if (is_branch(in)) {
                starts.insert(bias + in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4);
                starts.insert(next);
            } else if (in.enc == Enc::SOPP && in.op == 1) {
                starts.insert(next);
            } else if (in.enc == Enc::SOP1 && (in.op == 32 || (in.op == 33 && !vertex_input_call()))) {  // s_setpc / s_swappc
                starts.insert(next);
            }
        }
        for (std::uint32_t s : starts) {
            if (s >= bias && s < end) block_starts.insert(s);
        }
    }

    // ---- VALU implementations
    struct Mods {
        std::uint8_t abs = 0, neg = 0, omod = 0;
        bool clamp = false;
    };
    Id src_f(Id raw, int k, const Mods& md) {
        Id v = f(raw);
        if (md.abs & (1 << k)) v = fabs_(v);
        if (md.neg & (1 << k)) v = fneg(v);
        return v;
    }
    Id result_f(Id v, const Mods& md) {
        if (md.omod == 1) v = fmul(v, cf(2.0f));
        if (md.omod == 2) v = fmul(v, cf(4.0f));
        if (md.omod == 3) v = fmul(v, cf(0.5f));
        if (md.clamp) v = m.ext_inst(t_f32, spv::GlslFClamp, {v, cf(0.0f), cf(1.0f)});
        return u(v);
    }
    Id mul_legacy(Id a, Id b) {
        // GCN's legacy multiply: zero times anything, infinity and NaN too, is
        // zero. Written as DXVK writes D3D9's (min(|a|, |b|) == 0 ? 0 : a * b),
        // which Mesa recognises as one v_mul_legacy_f32; two compares, an or
        // and a select before. NMin, so a NaN beside a zero still gives zero
        // (FMin may answer the NaN, as NVIDIA's did). BBHOST_LEGACY_MUL=select:
        // the old form.
        if (legacy_mul_min_form()) return fsel(feq(fmin_(fabs_(a), fabs_(b)), cf(0.0f)), cf(0.0f), fmul(a, b));
        const Id zero = bor(feq(a, cf(0.0f)), feq(b, cf(0.0f)));
        return fsel(zero, cf(0.0f), fmul(a, b));
    }

    // VOP2 op with two sources (already raw u32 ids); returns result u32 or 0 if handled with side effects.
    bool vop2(std::uint32_t op, const Inst& in, Id s0, Id s1, const Mods& md, Id carry_in_pair_lo, Id carry_in_pair_hi,
              std::uint16_t carry_out) {
        const bool vop3 = in.enc == Enc::VOP3;
        auto F0 = [&] { return src_f(s0, 0, md); };
        auto F1 = [&] { return src_f(s1, 1, md); };
        auto done_f = [&](Id v) { write_vdst(in, result_f(v, md)); return true; };
        auto done_u = [&](Id v) { write_vdst(in, v); return true; };
        switch (op) {
        case 0: {  // v_cndmask_b32: src1 if mask bit else src0
            Id bit = vop3 ? mask_bit(carry_in_pair_lo, carry_in_pair_hi) : vcc_bit();
            // VOP3: the sources' abs and neg modifiers apply, abs first, as to a
            // float operation's (the compiler writes |a|, |b| and -a, -b forms):
            // sign-bit operations on the words, which the select passes on.
            const auto modified = [&](Id x, int k) {
                if (!vop3) return x;
                if (md.abs & (1 << k)) x = iand(x, cu(0x7fffffffu));
                if (md.neg & (1 << k)) x = ixor(x, cu(0x80000000u));
                return x;
            };
            return done_u(sel(bit, modified(s1, 1), modified(s0, 0)));
        }
        case 1: {  // v_readlane_b32 sdst, vsrc, lane
            int cell_lane = 0;
            Id v;
            if (spill.reads.count(cur_offset) && in.src0 >= 256 &&
                const_lane(vop3 ? in.src1 : static_cast<std::uint16_t>(in.src1 - 256), cell_lane)) {
                v = ld(spill_cell(in.src0 - 256, cell_lane));  // a spilled scalar: the cell its v_writelane_b32 filled
            } else {
                const Id lane_sel = iand(s1, cu(63));
                v = m.emit(spv::OpGroupNonUniformShuffle, t_u32, {cu(spv::ScopeSubgroup), s0, lane_sel});
            }
            write_s(in.dst, v);
            // A spilled scalar coming back (lane_syms): its resource path too.
            int ln = 0;
            if (in.dst < 104 && in.src0 >= 256 && const_lane(vop3 ? in.src1 : static_cast<std::uint16_t>(in.src1 - 256), ln)) {
                auto it = lane_syms.find({in.src0 - 256, ln});
                if (it != lane_syms.end()) sym[in.dst] = it->second;
            }
            return true;
        }
        case 2: {  // v_writelane_b32 vdst, ssrc, lane
            const Id lane_sel = iand(s1, cu(63));
            const Id var = vgpr_var(in.dst);
            packed_pairs.erase(in.dst);  // a lane of it is a scalar now, no pack's
            m.store(var, sel(ieq(lane(), lane_sel), s0, ld(var)));
            int ln = 0;
            if (cur_offset < kFetchBias && !spill.cells.empty() &&
                const_lane(vop3 ? in.src1 : static_cast<std::uint16_t>(in.src1 - 256), ln) && spill.cells.count({in.dst, ln})) {
                m.store(spill_cell(in.dst, ln), s0);  // GCN's writelane ignores EXEC: so does the cell
            }
            if (const_lane(vop3 ? in.src1 : static_cast<std::uint16_t>(in.src1 - 256), ln)) {
                const SymVal sv = in.src0 < 104 ? sym_of(in.src0) : SymVal{};
                if (sv.kind != SymVal::Unknown) {
                    lane_syms[{in.dst, ln}] = sv;
                } else {
                    lane_syms.erase({in.dst, ln});
                }
            } else {
                forget_lanes(in.dst);  // a lane not known here: any of them
            }
            return true;
        }
        case 3: return done_f(fadd(F0(), F1()));
        case 4: return done_f(fsub(F0(), F1()));
        case 5: return done_f(fsub(F1(), F0()));
        case 6: return done_f(fadd(mul_legacy(F0(), F1()), f(ld(vgpr_var(in.dst)))));  // v_mac_legacy_f32
        case 7: return done_f(mul_legacy(F0(), F1()));
        case 8: return done_f(fmul(F0(), F1()));
        case 9: return done_u(imul(sext24(s0), sext24(s1)));   // v_mul_i32_i24
        case 10: {  // v_mul_hi_i32_i24
            const Id r = m.emit(spv::OpSMulExtended, m.type_struct({t_i32, t_i32}), {i(sext24(s0)), i(sext24(s1))});
            return done_u(ui(extract(t_i32, r, 1)));
        }
        case 11: return done_u(imul(zext24(s0), zext24(s1)));  // v_mul_u32_u24
        case 12: {
            const Id r = m.emit(spv::OpUMulExtended, m.type_struct({t_u32, t_u32}), {zext24(s0), zext24(s1)});
            return done_u(extract(t_u32, r, 1));
        }
        case 13: return done_f(fmin_(F0(), F1()));  // legacy min/max: treat as IEEE
        case 14: return done_f(fmax_(F0(), F1()));
        case 15: return done_f(fmin_(F0(), F1()));
        case 16: return done_f(fmax_(F0(), F1()));
        case 17: return done_u(ui(m.ext_inst(t_i32, spv::GlslSMin, {i(s0), i(s1)})));
        case 18: return done_u(ui(m.ext_inst(t_i32, spv::GlslSMax, {i(s0), i(s1)})));
        case 19: return done_u(m.ext_inst(t_u32, spv::GlslUMin, {s0, s1}));
        case 20: return done_u(m.ext_inst(t_u32, spv::GlslUMax, {s0, s1}));
        case 21: return done_u(shr(s0, s1));   // v_lshr_b32
        case 22: return done_u(shr(s1, s0));   // v_lshrrev_b32
        case 23: return done_u(sar(s0, s1));
        case 24: return done_u(sar(s1, s0));
        case 25: return done_u(shl(s0, s1));
        case 26: return done_u(shl(s1, s0));
        case 27: return done_u(iand(s0, s1));
        case 28: return done_u(ior(s0, s1));
        case 29: return done_u(ixor(s0, s1));
        case 30: return done_u(shl(isub(shl(c_one_u, iand(s0, cu(31))), c_one_u), iand(s1, cu(31))));  // v_bfm_b32
        case 31: return done_f(fma_(F0(), F1(), f(ld(vgpr_var(in.dst)))));  // v_mac_f32
        case 32: return done_f(fma_(F0(), f(cu(in.literal)), F1()));       // v_madmk_f32: s0 * K + s1
        case 33: return done_f(fma_(F0(), F1(), f(cu(in.literal))));       // v_madak_f32: s0 * s1 + K
        case 34: return done_u(iadd(popcnt(s0), s1));                      // v_bcnt_u32_b32
        case 35: {  // v_mbcnt_lo_u32_b32: popcount(s0 & lanes below (lo half)) + s1
            const Id l = lane();
            const Id lt32 = ult(l, cu(32));
            const Id below = sel(lt32, isub(shl(c_one_u, l), c_one_u), cu(0xffffffffu));
            return done_u(iadd(popcnt(iand(s0, below)), s1));
        }
        case 36: {  // v_mbcnt_hi_u32_b32
            const Id l = lane();
            const Id lt32 = ult(l, cu(32));
            const Id below = sel(lt32, c_zero_u, isub(shl(c_one_u, isub(l, cu(32))), c_one_u));
            return done_u(iadd(popcnt(iand(s0, below)), s1));
        }
        case 37: case 38: case 39: {  // v_add_i32 / v_sub_i32 / v_subrev_i32 with carry out
            Id a = s0, b = s1;
            if (op == 39) std::swap(a, b);
            const Id st = m.type_struct({t_u32, t_u32});
            const Id r = m.emit(op == 37 ? spv::OpIAddCarry : spv::OpISubBorrow, st, {a, b});
            const Id carry = u2b(extract(t_u32, r, 1));
            write_vdst(in, extract(t_u32, r, 0));
            write_carry(carry_out, carry);
            return true;
        }
        case 40: case 41: case 42: {  // v_addc_u32 / v_subb_u32 / v_subbrev_u32
            const Id cin = vop3 ? mask_bit(carry_in_pair_lo, carry_in_pair_hi) : vcc_bit();
            const Id c = b2u(cin);
            Id a = s0, b = s1;
            if (op == 42) std::swap(a, b);
            const Id st = m.type_struct({t_u32, t_u32});
            Id r1, r2, sum, carry;
            if (op == 40) {
                r1 = m.emit(spv::OpIAddCarry, st, {a, b});
                r2 = m.emit(spv::OpIAddCarry, st, {extract(t_u32, r1, 0), c});
                sum = extract(t_u32, r2, 0);
                carry = u2b(ior(extract(t_u32, r1, 1), extract(t_u32, r2, 1)));
            } else {
                r1 = m.emit(spv::OpISubBorrow, st, {a, b});
                r2 = m.emit(spv::OpISubBorrow, st, {extract(t_u32, r1, 0), c});
                sum = extract(t_u32, r2, 0);
                carry = u2b(ior(extract(t_u32, r1, 1), extract(t_u32, r2, 1)));
            }
            write_vdst(in, sum);
            write_carry(carry_out, carry);
            return true;
        }
        case 43: return done_f(m.ext_inst(t_f32, spv::GlslLdexp, {F0(), i(s1)}));  // v_ldexp_f32
        case 47: {  // v_cvt_pkrtz_f16_f32
            const Id lo = F0(), hi = F1();
            const Id v2 = m.emit(spv::OpCompositeConstruct, t_v2f, {lo, hi});
            done_u(native_rtz ? m.emit(spv::OpBitcast, t_u32, {m.emit(spv::OpFConvert, t_v2h, {v2})})
                              : m.ext_inst(t_u32, spv::GlslPackHalf2x16, {v2}));
            if (export_rtz_on()) packed_pairs[in.dst] = {lo, hi, m.blocks(), last_write_exec};
            return true;
        }
        case 48: {  // v_cvt_pk_u16_u32
            const Id lo = m.ext_inst(t_u32, spv::GlslUMin, {s0, cu(0xffff)});
            const Id hi = m.ext_inst(t_u32, spv::GlslUMin, {s1, cu(0xffff)});
            return done_u(ior(lo, shl(hi, cu(16))));
        }
        case 49: {  // v_cvt_pk_i16_i32
            const Id lo = ui(m.ext_inst(t_i32, spv::GlslSClamp, {i(s0), m.const_i32(-32768), m.const_i32(32767)}));
            const Id hi = ui(m.ext_inst(t_i32, spv::GlslSClamp, {i(s1), m.const_i32(-32768), m.const_i32(32767)}));
            return done_u(ior(iand(lo, cu(0xffff)), shl(hi, cu(16))));
        }
        default:
            break;
        }
        return false;
    }
    void write_carry(std::uint16_t dst_pair, Id carry) {
        Id lo, hi;
        ballot(band(carry, exec_cond()), lo, hi);
        write_pair(dst_pair, lo, hi);
    }

    Id rcp(Id x) { return fdiv(cf(1.0f), x); }
    Id rsq(Id x) { return m.ext_inst(t_f32, spv::GlslInverseSqrt, {x}); }
    Id clamp_inf(Id v) {
        // v_rsq_clamp / v_rcp_clamp: +-inf -> +-FLT_MAX
        const Id inf = m.emit(spv::OpIsInf, t_bool, {v});
        const Id big = m.ext_inst(t_f32, spv::GlslFSign, {v});
        return fsel(inf, fmul(big, cf(3.4028235e38f)), v);
    }

    bool vop1(std::uint32_t op, const Inst& in, Id s0, const Mods& md) {
        auto F0 = [&] { return src_f(s0, 0, md); };
        auto done_f = [&](Id v) { write_vdst(in, result_f(v, md)); return true; };
        auto done_u = [&](Id v) { write_vdst(in, v); return true; };
        switch (op) {
        case 0: return true;  // v_nop
        case 1: return done_u(s0);
        case 2: {  // v_readfirstlane_b32
            const Id v = m.emit(spv::OpGroupNonUniformBroadcastFirst, t_u32, {cu(spv::ScopeSubgroup), s0});
            write_s(in.dst, v);
            return true;
        }
        case 5: return done_f(i2f(s0));
        case 6: return done_f(u2f(s0));
        case 7: {  // v_cvt_u32_f32 (saturating, NaN -> 0)
            const Id x = F0();
            const Id c = m.ext_inst(t_f32, spv::GlslFClamp, {x, cf(0.0f), cf(4294967040.0f)});
            return done_u(sel(isnan(x), c_zero_u, f2u(c)));
        }
        case 8: {  // v_cvt_i32_f32
            const Id x = F0();
            const Id c = m.ext_inst(t_f32, spv::GlslFClamp, {x, cf(-2147483648.0f), cf(2147483520.0f)});
            return done_u(sel(isnan(x), c_zero_u, f2i(c)));
        }
        case 10: {  // v_cvt_f16_f32
            const Id v2 = m.emit(spv::OpCompositeConstruct, t_v2f, {F0(), cf(0.0f)});
            return done_u(iand(m.ext_inst(t_u32, spv::GlslPackHalf2x16, {v2}), cu(0xffff)));
        }
        case 11:  // v_cvt_f32_f16 (exact decoding: gcn/half.h)
            return done_f(emit_unpack_half16(m, half_types(), iand(s0, cu(0xffff))));
        case 12: return done_u(f2i(m.ext_inst(t_f32, spv::GlslFloor, {fadd(F0(), cf(0.5f))})));  // v_cvt_rpi_i32_f32
        case 13: return done_u(f2i(m.ext_inst(t_f32, spv::GlslFloor, {F0()})));                   // v_cvt_flr_i32_f32
        case 14: {  // v_cvt_off_f32_i4: 4-bit signed -> (x - ?)/16 ... offset table: value/16 for signed nibble
            const Id nib = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(s0), c_zero_u, cu(4)}));
            return done_f(fmul(i2f(nib), cf(1.0f / 16.0f)));
        }
        case 17: case 18: case 19: case 20:  // v_cvt_f32_ubyteN
            return done_f(u2f(ubfe(s0, 8 * (op - 17), 8)));
        case 32: return done_f(m.ext_inst(t_f32, spv::GlslFract, {F0()}));
        case 33: return done_f(m.ext_inst(t_f32, spv::GlslTrunc, {F0()}));
        case 34: return done_f(m.ext_inst(t_f32, spv::GlslCeil, {F0()}));
        case 35: return done_f(m.ext_inst(t_f32, spv::GlslRoundEven, {F0()}));
        case 36: return done_f(m.ext_inst(t_f32, spv::GlslFloor, {F0()}));
        case 37: return done_f(m.ext_inst(t_f32, spv::GlslExp2, {F0()}));
        case 38: case 39: return done_f(m.ext_inst(t_f32, spv::GlslLog2, {F0()}));
        case 40: return done_f(clamp_inf(rcp(F0())));
        case 41: return done_f(rcp(F0()));
        case 42: return done_f(rcp(F0()));
        case 43: return done_f(rcp(F0()));
        case 44: return done_f(clamp_inf(rsq(F0())));
        case 45: return done_f(rsq(F0()));
        case 46: return done_f(rsq(F0()));
        case 51: return done_f(m.ext_inst(t_f32, spv::GlslSqrt, {F0()}));
        case 53: return done_f(m.ext_inst(t_f32, spv::GlslSin, {fmul(F0(), cf(6.2831853f))}));  // input in revolutions
        case 54: return done_f(m.ext_inst(t_f32, spv::GlslCos, {fmul(F0(), cf(6.2831853f))}));
        case 55: return done_u(inot(s0));
        case 56: return done_u(m.emit(spv::OpBitReverse, t_u32, {s0}));
        case 57: {  // v_ffbh_u32: count leading zeros, -1 if zero
            const Id msb = m.ext_inst(t_u32, spv::GlslFindUMsb, {s0});
            return done_u(sel(ieq(s0, c_zero_u), cu(0xffffffffu), isub(cu(31), msb)));
        }
        case 58: return done_u(m.ext_inst(t_u32, spv::GlslFindILsb, {s0}));  // -1 when zero
        case 59: {  // v_ffbh_i32
            const Id msb = ui(m.ext_inst(t_i32, spv::GlslFindSMsb, {i(s0)}));
            const Id none = bor(ieq(s0, c_zero_u), ieq(s0, cu(0xffffffffu)));
            return done_u(sel(none, cu(0xffffffffu), isub(cu(31), msb)));
        }
        default:
            break;
        }
        return false;
    }

    // Compare op (VOPC opcode) on two raw sources; returns a bool.
    Id vcmp_cond(std::uint32_t op, Id s0, Id s1, const Mods& md) {
        const std::uint32_t hi = (op >> 5) & 7;
        const std::uint32_t lo = op & 0xf;
        if (hi == 0 || hi == 2) {  // f32 (signalling variants treated the same)
            const Id a = src_f(s0, 0, md), b = src_f(s1, 1, md);
            const Id unord = bor(isnan(a), isnan(b));
            switch (lo) {
            case 0: return m.const_bool(false);
            case 1: return flt(a, b);
            case 2: return feq(a, b);
            case 3: return fle(a, b);
            case 4: return fgt(a, b);
            case 5: return band(bnot(unord), bnot(feq(a, b)));  // lg
            case 6: return fge(a, b);
            case 7: return bnot(unord);                         // o
            case 8: return unord;                               // u
            case 9: return bnot(fge(a, b));                     // nge
            case 10: return bor(unord, feq(a, b));              // nlg
            case 11: return bnot(fgt(a, b));                    // ngt
            case 12: return bnot(fle(a, b));                    // nle
            case 13: return bnot(feq(a, b));                    // neq
            case 14: return bnot(flt(a, b));                    // nlt
            default: return m.const_bool(true);
            }
        }
        if (hi == 1 || hi == 3) {
            error("f64 compare unsupported");
            return m.const_bool(false);
        }
        if (hi == 5 || hi == 7) {
            error("i64/u64 compare unsupported");
            return m.const_bool(false);
        }
        if (lo == 8) {  // class
            error("v_cmp_class unsupported");
            return m.const_bool(false);
        }
        const bool is_signed = hi == 4;
        switch (lo) {
        case 0: return m.const_bool(false);
        case 1: return is_signed ? slt(s0, s1) : ult(s0, s1);
        case 2: return ieq(s0, s1);
        case 3: return is_signed ? sle(s0, s1) : ule(s0, s1);
        case 4: return is_signed ? sgt(s0, s1) : ugt(s0, s1);
        case 5: return ine(s0, s1);
        case 6: return is_signed ? sge(s0, s1) : uge(s0, s1);
        default: return m.const_bool(true);
        }
    }
    void vcmp(std::uint32_t op, const Inst& in, Id s0, Id s1, const Mods& md, std::uint16_t dst_pair) {
        const Id c = band(vcmp_cond(op, s0, s1, md), exec_cond());
        Id lo, hi;
        ballot(c, lo, hi);
        write_pair(dst_pair, lo, hi);
        if (op & 0x10) {  // cmpx also writes exec
            write_pair(kExecLo, lo, hi);
        }
    }

    // Cube map helpers (GCN v_cube* semantics).
    void cube_ops(std::uint32_t op, const Inst& in, Id s0, Id s1, Id s2, const Mods& md) {
        const Id x = src_f(s0, 0, md), y = src_f(s1, 1, md), z = src_f(s2, 2, md);
        const Id ax = fabs_(x), ay = fabs_(y), az = fabs_(z);
        const Id z_major = band(fge(az, ax), fge(az, ay));
        const Id y_major = band(bnot(z_major), fge(ay, ax));
        const Id zero = cf(0.0f);
        Id r;
        switch (op) {
        case 0x144: {  // v_cubeid_f32: face id 0..5
            const Id zid = fsel(fge(z, zero), cf(4.0f), cf(5.0f));
            const Id yid = fsel(fge(y, zero), cf(2.0f), cf(3.0f));
            const Id xid = fsel(fge(x, zero), cf(0.0f), cf(1.0f));
            r = fsel(z_major, zid, fsel(y_major, yid, xid));
            break;
        }
        case 0x145: {  // v_cubesc_f32
            const Id zsc = fsel(fge(z, zero), x, fneg(x));
            const Id ysc = x;
            const Id xsc = fsel(fge(x, zero), fneg(z), z);
            r = fsel(z_major, zsc, fsel(y_major, ysc, xsc));
            break;
        }
        case 0x146: {  // v_cubetc_f32
            const Id ztc = fneg(y);
            const Id ytc = fsel(fge(y, zero), z, fneg(z));
            const Id xtc = fneg(y);
            r = fsel(z_major, ztc, fsel(y_major, ytc, xtc));
            break;
        }
        default: {  // v_cubema_f32: 2 * major axis
            r = fmul(cf(2.0f), fsel(z_major, z, fsel(y_major, y, x)));
            break;
        }
        }
        write_vdst(in, result_f(r, md));
    }

    bool vop3_only(std::uint32_t op, const Inst& in, Id s0, Id s1, Id s2, const Mods& md) {
        auto F0 = [&] { return src_f(s0, 0, md); };
        auto F1 = [&] { return src_f(s1, 1, md); };
        auto F2 = [&] { return src_f(s2, 2, md); };
        auto done_f = [&](Id v) { write_vdst(in, result_f(v, md)); return true; };
        auto done_u = [&](Id v) { write_vdst(in, v); return true; };
        switch (op) {
        case 0x140: return done_f(fadd(mul_legacy(F0(), F1()), F2()));
        case 0x141: return done_f(fma_(F0(), F1(), F2()));
        case 0x142: return done_u(iadd(imul(sext24(s0), sext24(s1)), s2));
        case 0x143: return done_u(iadd(imul(zext24(s0), zext24(s1)), s2));
        case 0x144: case 0x145: case 0x146: case 0x147:
            cube_ops(op, in, s0, s1, s2, md);
            return true;
        case 0x148: {  // v_bfe_u32
            const Id off = iand(s1, cu(31)), n = iand(s2, cu(31));
            return done_u(m.emit(spv::OpBitFieldUExtract, t_u32, {s0, off, n}));
        }
        case 0x149: {
            const Id off = iand(s1, cu(31)), n = iand(s2, cu(31));
            return done_u(ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(s0), off, n})));
        }
        case 0x14a: return done_u(ior(iand(s0, s1), iand(inot(s0), s2)));  // v_bfi_b32
        case 0x14b: return done_f(fma_(F0(), F1(), F2()));
        case 0x14e: {  // v_alignbit_b32: (s0:s1 >> s2[4:0]) low 32
            const Id v = make_u64(s1, s0);
            const Id sh = m.emit(spv::OpUConvert, t_u64, {iand(s2, cu(31))});
            return done_u(m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpShiftRightLogical, t_u64, {v, sh})}));
        }
        case 0x151: return done_f(fmin_(fmin_(F0(), F1()), F2()));
        case 0x152: return done_u(ui(m.ext_inst(t_i32, spv::GlslSMin, {i(ui(m.ext_inst(t_i32, spv::GlslSMin, {i(s0), i(s1)}))), i(s2)})));
        case 0x153: return done_u(m.ext_inst(t_u32, spv::GlslUMin, {m.ext_inst(t_u32, spv::GlslUMin, {s0, s1}), s2}));
        case 0x154: return done_f(fmax_(fmax_(F0(), F1()), F2()));
        case 0x155: return done_u(ui(m.ext_inst(t_i32, spv::GlslSMax, {i(ui(m.ext_inst(t_i32, spv::GlslSMax, {i(s0), i(s1)}))), i(s2)})));
        case 0x156: return done_u(m.ext_inst(t_u32, spv::GlslUMax, {m.ext_inst(t_u32, spv::GlslUMax, {s0, s1}), s2}));
        case 0x157: {  // v_med3_f32
            const Id a = F0(), b = F1(), c = F2();
            return done_f(fmax_(fmin_(a, b), fmin_(fmax_(a, b), c)));
        }
        case 0x158: {
            const Id a = i(s0), b = i(s1), c = i(s2);
            const Id mn = m.ext_inst(t_i32, spv::GlslSMin, {a, b});
            const Id mx = m.ext_inst(t_i32, spv::GlslSMax, {a, b});
            return done_u(ui(m.ext_inst(t_i32, spv::GlslSMax, {mn, m.ext_inst(t_i32, spv::GlslSMin, {mx, c})})));
        }
        case 0x159: {
            const Id mn = m.ext_inst(t_u32, spv::GlslUMin, {s0, s1});
            const Id mx = m.ext_inst(t_u32, spv::GlslUMax, {s0, s1});
            return done_u(m.ext_inst(t_u32, spv::GlslUMax, {mn, m.ext_inst(t_u32, spv::GlslUMin, {mx, s2})}));
        }
        case 0x15d: return done_u(iadd(ui(m.ext_inst(t_i32, spv::GlslSAbs, {i(isub(s0, s1))})), s2));  // v_sad_u32
        case 0x15e: {  // v_cvt_pk_u8_f32: byte s1 of s2 replaced by u8(s0)
            const Id byte = f2u(m.ext_inst(t_f32, spv::GlslFClamp, {F0(), cf(0.0f), cf(255.0f)}));
            const Id shift = shl(iand(s1, cu(3)), cu(3));
            const Id mask = shl(cu(0xff), shift);
            return done_u(ior(iand(s2, inot(mask)), shl(byte, shift)));
        }
        case 0x161: {  // v_lshl_b64
            const Id v = read_pair_vgpr64(in.src0, in);
            const Id r = m.emit(spv::OpShiftLeftLogical, t_u64, {v, m.emit(spv::OpUConvert, t_u64, {iand(s1, cu(63))})});
            Id lo, hi;
            split_u64(r, lo, hi);
            write_v(in.dst, lo);
            write_v(in.dst + 1, hi);
            return true;
        }
        case 0x169: return done_u(imul(s0, s1));  // v_mul_lo_u32
        case 0x16a: {
            const Id r = m.emit(spv::OpUMulExtended, m.type_struct({t_u32, t_u32}), {s0, s1});
            return done_u(extract(t_u32, r, 1));
        }
        case 0x16b: return done_u(imul(s0, s1));  // v_mul_lo_i32
        case 0x16c: {
            const Id r = m.emit(spv::OpSMulExtended, m.type_struct({t_i32, t_i32}), {i(s0), i(s1)});
            return done_u(ui(extract(t_i32, r, 1)));
        }
        default:
            break;
        }
        return false;
    }
    Id read_pair_vgpr64(std::uint16_t code, const Inst& in) {
        Id lo, hi;
        read_pair(code, in, lo, hi);
        return make_u64(lo, hi);
    }

    // ---- scalar ALU
    void sop2(const Inst& in) {
        auto S0 = [&] { return read_s(in.src0, in); };
        auto S1 = [&] { return read_s(in.src1, in); };
        auto done = [&](Id v, Id scc) { write_s(in.dst, v); if (scc) set_scc(scc); };
        auto done64 = [&](Id lo, Id hi) {
            write_pair(in.dst, lo, hi);
            set_scc(ine(ior(lo, hi), c_zero_u));
        };
        auto bits64 = [&](spv::Op op, bool not1 = false) {
            Id a0, a1, b0, b1;
            read_pair(in.src0, in, a0, a1);
            read_pair(in.src1, in, b0, b1);
            const bool la = lane_pairs.count({a0, a1}) != 0, lb = lane_pairs.count({b0, b1}) != 0;
            if (not1) { b0 = inot(b0); b1 = inot(b1); }
            const Id r0 = m.emit(op, t_u32, {a0, b0});
            const Id r1 = m.emit(op, t_u32, {a1, b1});
            // The lane bit stays set through a & b (both), a | b (either), a | ~b (a).
            const bool lane = op == spv::OpBitwiseAnd ? !not1 && la && lb : op == spv::OpBitwiseOr && (la || (!not1 && lb));
            if (lane) lane_pairs.insert({r0, r1});
            done64(r0, r1);
        };
        switch (in.op) {
        case 0: {  // s_add_u32
            const Id r = m.emit(spv::OpIAddCarry, m.type_struct({t_u32, t_u32}), {S0(), S1()});
            done(extract(t_u32, r, 0), u2b(extract(t_u32, r, 1)));
            break;
        }
        case 1: {  // s_sub_u32
            const Id r = m.emit(spv::OpISubBorrow, m.type_struct({t_u32, t_u32}), {S0(), S1()});
            done(extract(t_u32, r, 0), u2b(extract(t_u32, r, 1)));
            break;
        }
        case 2: case 3: {  // s_add_i32 / s_sub_i32: scc = signed overflow
            const Id a = S0(), b = S1();
            const Id r = in.op == 2 ? iadd(a, b) : isub(a, b);
            Id ovf;
            if (in.op == 2) {
                ovf = band(ieq(shr(ixor(a, b), cu(31)), c_zero_u), ine(shr(ixor(a, r), cu(31)), c_zero_u));
            } else {
                ovf = band(ine(shr(ixor(a, b), cu(31)), c_zero_u), ine(shr(ixor(a, r), cu(31)), c_zero_u));
            }
            done(r, ovf);
            break;
        }
        case 4: {  // s_addc_u32
            const Id st = m.type_struct({t_u32, t_u32});
            const Id r1 = m.emit(spv::OpIAddCarry, st, {S0(), S1()});
            const Id r2 = m.emit(spv::OpIAddCarry, st, {extract(t_u32, r1, 0), b2u(ldb(v_scc))});
            done(extract(t_u32, r2, 0), u2b(ior(extract(t_u32, r1, 1), extract(t_u32, r2, 1))));
            break;
        }
        case 5: {  // s_subb_u32
            const Id st = m.type_struct({t_u32, t_u32});
            const Id r1 = m.emit(spv::OpISubBorrow, st, {S0(), S1()});
            const Id r2 = m.emit(spv::OpISubBorrow, st, {extract(t_u32, r1, 0), b2u(ldb(v_scc))});
            done(extract(t_u32, r2, 0), u2b(ior(extract(t_u32, r1, 1), extract(t_u32, r2, 1))));
            break;
        }
        case 6: { const Id a = S0(), b = S1(); const Id c = slt(a, b); done(sel(c, a, b), c); break; }
        case 7: { const Id a = S0(), b = S1(); const Id c = ult(a, b); done(sel(c, a, b), c); break; }
        case 8: { const Id a = S0(), b = S1(); const Id c = sgt(a, b); done(sel(c, a, b), c); break; }
        case 9: { const Id a = S0(), b = S1(); const Id c = ugt(a, b); done(sel(c, a, b), c); break; }
        case 10: done(sel(ldb(v_scc), S0(), S1()), 0); break;  // s_cselect_b32
        case 11: {
            Id a0, a1, b0, b1;
            read_pair(in.src0, in, a0, a1);
            read_pair(in.src1, in, b0, b1);
            const Id c = ldb(v_scc);
            write_pair(in.dst, sel(c, a0, b0), sel(c, a1, b1));
            break;
        }
        case 14: { const Id r = iand(S0(), S1()); done(r, ine(r, c_zero_u)); break; }
        case 15: bits64(spv::OpBitwiseAnd); break;
        case 16: { const Id r = ior(S0(), S1()); done(r, ine(r, c_zero_u)); break; }
        case 17: bits64(spv::OpBitwiseOr); break;
        case 18: { const Id r = ixor(S0(), S1()); done(r, ine(r, c_zero_u)); break; }
        case 19: bits64(spv::OpBitwiseXor); break;
        case 20: { const Id r = iand(S0(), inot(S1())); done(r, ine(r, c_zero_u)); break; }
        case 21: bits64(spv::OpBitwiseAnd, true); break;
        case 22: { const Id r = ior(S0(), inot(S1())); done(r, ine(r, c_zero_u)); break; }
        case 23: bits64(spv::OpBitwiseOr, true); break;
        case 24: { const Id r = inot(iand(S0(), S1())); done(r, ine(r, c_zero_u)); break; }
        case 25: {
            Id a0, a1, b0, b1;
            read_pair(in.src0, in, a0, a1);
            read_pair(in.src1, in, b0, b1);
            done64(inot(iand(a0, b0)), inot(iand(a1, b1)));
            break;
        }
        case 26: { const Id r = inot(ior(S0(), S1())); done(r, ine(r, c_zero_u)); break; }
        case 27: {
            Id a0, a1, b0, b1;
            read_pair(in.src0, in, a0, a1);
            read_pair(in.src1, in, b0, b1);
            done64(inot(ior(a0, b0)), inot(ior(a1, b1)));
            break;
        }
        case 28: { const Id r = inot(ixor(S0(), S1())); done(r, ine(r, c_zero_u)); break; }
        case 29: {
            Id a0, a1, b0, b1;
            read_pair(in.src0, in, a0, a1);
            read_pair(in.src1, in, b0, b1);
            done64(inot(ixor(a0, b0)), inot(ixor(a1, b1)));
            break;
        }
        case 30: { const Id r = shl(S0(), S1()); done(r, ine(r, c_zero_u)); break; }
        case 31: {
            const Id v = read_u64(in.src0, in);
            const Id r = m.emit(spv::OpShiftLeftLogical, t_u64, {v, m.emit(spv::OpUConvert, t_u64, {iand(S1(), cu(63))})});
            Id lo, hi; split_u64(r, lo, hi); done64(lo, hi);
            break;
        }
        case 32: { const Id r = shr(S0(), S1()); done(r, ine(r, c_zero_u)); break; }
        case 33: {
            const Id v = read_u64(in.src0, in);
            const Id r = m.emit(spv::OpShiftRightLogical, t_u64, {v, m.emit(spv::OpUConvert, t_u64, {iand(S1(), cu(63))})});
            Id lo, hi; split_u64(r, lo, hi); done64(lo, hi);
            break;
        }
        case 34: { const Id r = sar(S0(), S1()); done(r, ine(r, c_zero_u)); break; }
        case 35: {
            const Id v = read_u64(in.src0, in);
            const Id r = m.emit(spv::OpShiftRightArithmetic, t_u64, {v, m.emit(spv::OpUConvert, t_u64, {iand(S1(), cu(63))})});
            Id lo, hi; split_u64(r, lo, hi); done64(lo, hi);
            break;
        }
        case 36: {  // s_bfm_b32: ((1 << s0[4:0]) - 1) << s1[4:0]
            const Id r = shl(isub(shl(c_one_u, iand(S0(), cu(31))), c_one_u), iand(S1(), cu(31)));
            done(r, 0);
            break;
        }
        case 38: done(imul(S0(), S1()), 0); break;
        case 39: {  // s_bfe_u32: offset s1[4:0], width s1[22:16]
            const Id s1 = S1();
            const Id r = m.emit(spv::OpBitFieldUExtract, t_u32, {S0(), iand(s1, cu(31)), iand(shr(s1, cu(16)), cu(0x7f))});
            done(r, ine(r, c_zero_u));
            break;
        }
        case 40: {
            const Id s1 = S1();
            const Id r = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(S0()), iand(s1, cu(31)), iand(shr(s1, cu(16)), cu(0x7f))}));
            done(r, ine(r, c_zero_u));
            break;
        }
        case 44: {  // s_absdiff_i32
            const Id r = ui(m.ext_inst(t_i32, spv::GlslSAbs, {i(isub(S0(), S1()))}));
            done(r, ine(r, c_zero_u));
            break;
        }
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    void sopk(const Inst& in) {
        const Id imm = cu(static_cast<std::uint32_t>(in.imm));
        switch (in.op) {
        case 0: write_s(in.dst, imm); break;
        case 2: write_s(in.dst, sel(ldb(v_scc), imm, read_s(in.dst, in))); break;
        case 3: set_scc(ieq(read_s(in.dst, in), imm)); break;
        case 4: set_scc(ine(read_s(in.dst, in), imm)); break;
        case 5: set_scc(sgt(read_s(in.dst, in), imm)); break;
        case 6: set_scc(sge(read_s(in.dst, in), imm)); break;
        case 7: set_scc(slt(read_s(in.dst, in), imm)); break;
        case 8: set_scc(sle(read_s(in.dst, in), imm)); break;
        case 9: set_scc(ieq(read_s(in.dst, in), cu(in.imm & 0xffff))); break;
        case 10: set_scc(ine(read_s(in.dst, in), cu(in.imm & 0xffff))); break;
        case 11: set_scc(ugt(read_s(in.dst, in), cu(in.imm & 0xffff))); break;
        case 12: set_scc(uge(read_s(in.dst, in), cu(in.imm & 0xffff))); break;
        case 13: set_scc(ult(read_s(in.dst, in), cu(in.imm & 0xffff))); break;
        case 14: set_scc(ule(read_s(in.dst, in), cu(in.imm & 0xffff))); break;
        case 15: {  // s_addk_i32
            const Id a = read_s(in.dst, in);
            const Id r = iadd(a, imm);
            const Id ovf = band(ieq(shr(ixor(a, imm), cu(31)), c_zero_u), ine(shr(ixor(a, r), cu(31)), c_zero_u));
            write_s(in.dst, r);
            set_scc(ovf);
            break;
        }
        case 16: write_s(in.dst, imul(read_s(in.dst, in), imm)); break;
        case 18: write_s(in.dst, c_zero_u); break;  // s_getreg_b32 (hw regs): 0
        case 19: case 21: break;                   // s_setreg: ignored
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    void sop1(const Inst& in) {
        auto S0 = [&] { return read_s(in.src0, in); };
        switch (in.op) {
        case 3: {
            const SymVal sv = in.src0 < 104 ? sym_of(in.src0) : SymVal{};
            write_s(in.dst, S0());
            if (in.dst < 104 && sv.kind != SymVal::Unknown) sym[in.dst] = sv;
            break;
        }
        case 4: {
            const SymVal s0 = in.src0 < 104 ? sym_of(in.src0) : SymVal{};
            const SymVal s1 = in.src0 < 104 ? sym_of(in.src0 + 1) : SymVal{};
            Id lo, hi;
            read_pair(in.src0, in, lo, hi);
            write_pair(in.dst, lo, hi);
            if (in.dst < 104) {
                if (s0.kind != SymVal::Unknown) sym[in.dst] = s0;
                if (s1.kind != SymVal::Unknown) sym[in.dst + 1] = s1;
            }
            break;
        }
        case 5: write_s(in.dst, sel(ldb(v_scc), S0(), read_s(in.dst, in))); break;
        case 6: {
            Id lo, hi, d0, d1;
            read_pair(in.src0, in, lo, hi);
            read_pair(in.dst, in, d0, d1);
            const Id c = ldb(v_scc);
            write_pair(in.dst, sel(c, lo, d0), sel(c, hi, d1));
            break;
        }
        case 7: { const Id r = inot(S0()); write_s(in.dst, r); set_scc(ine(r, c_zero_u)); break; }
        case 8: { Id lo, hi; read_pair(in.src0, in, lo, hi); lo = inot(lo); hi = inot(hi); write_pair(in.dst, lo, hi); set_scc(ine(ior(lo, hi), c_zero_u)); break; }
        case 9: { const Id r = m.emit(spv::OpFunctionCall, t_u32, {fn_wqm, S0()}); write_s(in.dst, r); set_scc(ine(r, c_zero_u)); break; }
        case 10: {
            Id lo, hi;
            read_pair(in.src0, in, lo, hi);
            const bool lane = lane_pairs.count({lo, hi}) != 0;  // whole-quad mode keeps a set bit
            lo = m.emit(spv::OpFunctionCall, t_u32, {fn_wqm, lo});
            hi = m.emit(spv::OpFunctionCall, t_u32, {fn_wqm, hi});
            if (lane) lane_pairs.insert({lo, hi});
            write_pair(in.dst, lo, hi);
            set_scc(ine(ior(lo, hi), c_zero_u));
            break;
        }
        case 11: write_s(in.dst, m.emit(spv::OpBitReverse, t_u32, {S0()})); break;
        case 13: { const Id r = popcnt(inot(S0())); write_s(in.dst, r); set_scc(ine(r, c_zero_u)); break; }
        case 14: { Id lo, hi; read_pair(in.src0, in, lo, hi); const Id r = iadd(popcnt(inot(lo)), popcnt(inot(hi))); write_s(in.dst, r); set_scc(ine(r, c_zero_u)); break; }
        case 15: { const Id r = popcnt(S0()); write_s(in.dst, r); set_scc(ine(r, c_zero_u)); break; }
        case 16: { Id lo, hi; read_pair(in.src0, in, lo, hi); const Id r = iadd(popcnt(lo), popcnt(hi)); write_s(in.dst, r); set_scc(ine(r, c_zero_u)); break; }
        case 19: write_s(in.dst, m.ext_inst(t_u32, spv::GlslFindILsb, {S0()})); break;  // s_ff1_i32_b32
        case 20: {
            Id lo, hi;
            read_pair(in.src0, in, lo, hi);
            const Id l = m.ext_inst(t_u32, spv::GlslFindILsb, {lo});
            const Id h = m.ext_inst(t_u32, spv::GlslFindILsb, {hi});
            const Id r = sel(ine(lo, c_zero_u), l, sel(ine(hi, c_zero_u), iadd(h, cu(32)), cu(0xffffffffu)));
            write_s(in.dst, r);
            break;
        }
        case 21: {  // s_flbit_i32_b32: leading zeros
            const Id s = S0();
            const Id msb = m.ext_inst(t_u32, spv::GlslFindUMsb, {s});
            write_s(in.dst, sel(ieq(s, c_zero_u), cu(0xffffffffu), isub(cu(31), msb)));
            break;
        }
        case 25: write_s(in.dst, ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(S0()), c_zero_u, cu(8)}))); break;
        case 26: write_s(in.dst, ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(S0()), c_zero_u, cu(16)}))); break;
        case 27: write_s(in.dst, iand(read_s(in.dst, in), inot(shl(c_one_u, iand(S0(), cu(31)))))); break;
        case 29: write_s(in.dst, ior(read_s(in.dst, in), shl(c_one_u, iand(S0(), cu(31))))); break;
        case 32: {  // s_setpc_b64: return from the fetch shader
            m.store(v_pc, ld(v_ret));
            end_block_jump();
            break;
        }
        case 33: {  // s_swappc_b64: call the fetch shader
            const std::uint32_t ret = cur_offset + 4;
            write_pair(in.dst, cu(ret), c_zero_u);
            if (vertex_input_call()) {
                load_vertex_input();
            } else if (opt.fetch) {
                m.store(v_ret, cu(ret));
                m.store(v_pc, cu(kFetchBias));
                end_block_jump();
            }
            break;
        }
        case 36: case 37: case 38: case 39: case 40: case 41: case 42: case 43: {  // s_*_saveexec_b64
            Id lo, hi;
            read_pair(in.src0, in, lo, hi);
            const Id e0 = ld(v_exec_lo), e1 = ld(v_exec_hi);
            const bool ls = lane_pairs.count({lo, hi}) != 0, le = lane_set.count(kExecLo) != 0;
            if (le) lane_pairs.insert({e0, e1});
            write_pair(in.dst, e0, e1);
            Id r0, r1;
            switch (in.op) {
            case 36: r0 = iand(lo, e0); r1 = iand(hi, e1); break;
            case 37: r0 = ior(lo, e0); r1 = ior(hi, e1); break;
            case 38: r0 = ixor(lo, e0); r1 = ixor(hi, e1); break;
            case 39: r0 = iand(lo, inot(e0)); r1 = iand(hi, inot(e1)); break;
            case 40: r0 = ior(lo, inot(e0)); r1 = ior(hi, inot(e1)); break;
            case 41: r0 = inot(iand(lo, e0)); r1 = inot(iand(hi, e1)); break;
            case 42: r0 = inot(ior(lo, e0)); r1 = inot(ior(hi, e1)); break;
            default: r0 = inot(ixor(lo, e0)); r1 = inot(ixor(hi, e1)); break;
            }
            if ((in.op == 36 && ls && le) || (in.op == 37 && (ls || le)) || (in.op == 40 && ls)) lane_pairs.insert({r0, r1});
            write_pair(kExecLo, r0, r1);
            set_scc(ine(ior(r0, r1), c_zero_u));
            break;
        }
        case 52: {  // s_abs_i32
            const Id r = ui(m.ext_inst(t_i32, spv::GlslSAbs, {i(S0())}));
            write_s(in.dst, r);
            set_scc(ine(r, c_zero_u));
            break;
        }
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    void sopc(const Inst& in) {
        const Id a = read_s(in.src0, in), b = read_s(in.src1, in);
        switch (in.op) {
        case 0: set_scc(ieq(a, b)); break;
        case 1: set_scc(ine(a, b)); break;
        case 2: set_scc(sgt(a, b)); break;
        case 3: set_scc(sge(a, b)); break;
        case 4: set_scc(slt(a, b)); break;
        case 5: set_scc(sle(a, b)); break;
        case 6: set_scc(ieq(a, b)); break;
        case 7: set_scc(ine(a, b)); break;
        case 8: set_scc(ugt(a, b)); break;
        case 9: set_scc(uge(a, b)); break;
        case 10: set_scc(ult(a, b)); break;
        case 11: set_scc(ule(a, b)); break;
        case 12: set_scc(ieq(iand(shr(a, iand(b, cu(31))), c_one_u), c_zero_u)); break;
        case 13: set_scc(ieq(iand(shr(a, iand(b, cu(31))), c_one_u), c_one_u)); break;
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    // Ends the current block after pc was stored.
    void end_block_jump() {
        pending_smrd.clear();  // deferred results are only re-applied within a block
        m.emit_void(spv::OpBranch, {dag ? lbl_block_merge : lbl_dispatch_merge});
        block_terminated = true;
    }
    // A target outside the program has no flag: nothing after it runs, which is
    // what the dispatcher's default case does too.
    void jump_to(std::uint32_t target) {
        if (!dag) {
            m.store(v_pc, cu(target));
        } else if (auto it = reach_vars.find(target); it != reach_vars.end()) {
            m.store(it->second, c_one_u);
            if (dag_loop && target <= cur_block) m.store(loop_again(target), c_one_u);
            lane_edge(target);
            block_succ[cur_block].insert(target);
        } else {
            lane_exit();
        }
        end_block_jump();
    }
    void cond_jump(Id cond, std::uint32_t target, std::uint32_t fallthrough) {
        if (!dag) {
            m.store(v_pc, sel(cond, cu(target), cu(fallthrough)));
        } else {
            if (auto it = reach_vars.find(target); it != reach_vars.end()) {
                m.store(it->second, sel(cond, c_one_u, ld(it->second)));
                if (dag_loop && target <= cur_block) {
                    const Id again = loop_again(target);
                    m.store(again, sel(cond, c_one_u, ld(again)));
                }
                lane_edge(target);
                block_succ[cur_block].insert(target);
            } else {
                lane_exit();
            }
            if (auto it = reach_vars.find(fallthrough); it != reach_vars.end()) {
                block_succ[cur_block].insert(fallthrough);
                m.store(it->second, sel(cond, ld(it->second), c_one_u));
                lane_edge(fallthrough);
            } else {
                lane_exit();
            }
        }
        end_block_jump();
    }

    void sopp(const Inst& in, std::uint32_t bias) {
        const std::uint32_t next = bias + in.offset + 4;
        const std::uint32_t target = bias + in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
        switch (in.op) {
        case 0: break;   // s_nop
        case 1: {        // s_endpgm (in a DAG no later block's flag gets set)
            lane_exit();
            m.emit_void(spv::OpBranch, {dag ? lbl_block_merge : lbl_loop_merge});
            block_terminated = true;
            break;
        }
        case 2: jump_to(target); break;
        case 4: cond_jump(bnot(ldb(v_scc)), target, next); break;
        case 5: cond_jump(ldb(v_scc), target, next); break;
        case 6: cond_jump(ieq(ior(ld(v_vcc_lo), ld(v_vcc_hi)), c_zero_u), target, next); break;
        case 7: cond_jump(ine(ior(ld(v_vcc_lo), ld(v_vcc_hi)), c_zero_u), target, next); break;
        case 8: cond_jump(ieq(ior(ld(v_exec_lo), ld(v_exec_hi)), c_zero_u), target, next); break;
        case 9: cond_jump(ine(ior(ld(v_exec_lo), ld(v_exec_hi)), c_zero_u), target, next); break;
        case 10:  // s_barrier
            // A hull shader's LDS is a buffer here, and the patch's control
            // points read what the others wrote to it after the barrier.
            // A tessellation control stage may not name workgroup memory
            // (GLSL450), so the buffer gets a barrier of its own first.
            if (opt.stage == Stage::TessControl) {
                if (lds_in_buffer) {
                    m.emit_void(spv::OpMemoryBarrier, {cu(spv::ScopeDevice), cu(spv::MsAcquireRelease | spv::MsUniformMemory)});
                }
                m.emit_void(spv::OpControlBarrier, {cu(spv::ScopeWorkgroup), cu(spv::ScopeInvocation), cu(spv::MsNone)});
                break;
            }
            m.emit_void(spv::OpControlBarrier, {cu(spv::ScopeWorkgroup), cu(spv::ScopeWorkgroup),
                                                cu(spv::MsAcquireRelease | (lds_in_buffer ? spv::MsUniformMemory : spv::MsWorkgroupMemory))});
            break;
        case 12:  // s_waitcnt: lgkmcnt in simm16[11:8], 0xf = not waited on
            if (((in.imm >> 8) & 0xf) != 0xf) land_pending_smrd();
            break;
        case 13: case 14: case 15: case 16: case 17: case 19: case 20: case 21: case 22:
            break;   // sethalt, sleep, setprio, sendmsg, sendmsghalt, icache_inv, perf, ttrace
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    // ---- scalar memory
    void smrd(const Inst& in) {
        const int count = in.op < 8 ? (1 << in.op) : (in.op < 13 ? (1 << (in.op - 8)) : 0);
        if (count == 0) {
            // s_dcache_inv / s_memtime: nothing to do (memtime -> 0)
            if (in.op == 30) write_pair(in.sdst, c_zero_u, c_zero_u);
            return;
        }
        Id base;
        Id offset;                  // bytes
        bool known = false;
        std::uint32_t known_off = 0;
        if (in.imm_flag) {
            offset = cu(static_cast<std::uint32_t>(in.imm) * 4);
            known = true;
            known_off = static_cast<std::uint32_t>(in.imm);  // dwords
        } else if (in.has_literal) {
            // Sony's compiler emits the CI literal form for dword offsets >= 256
            // (the corpus has literals like 271), so the unit is dwords like IMM.
            offset = cu(in.literal * 4);
            known = true;
            known_off = in.literal;
        } else {
            offset = read_s(static_cast<std::uint16_t>(in.imm), in);
        }
        const SymVal base_sym = sym_of(in.src0);
        PendingSmrd pending{in.sdst, {}, {}};
        ResourcePath cb_path;
        std::string cb_key;
        // A V# (s_buffer_load_*) traced to user data, or a pointer pair
        // (s_load_*) held directly in user data (the commit's resource tables),
        // is read through a storage buffer. Pointer pairs loaded from somewhere
        // else are often not pointers in the draws where the load does not run
        // (floats, small integers), and binding over them costs the lean variant.
        const bool pointer = in.op < 8;
        bool as_buffer = opt.cb_ssbo && known && sym_path(in.src0, 2, cb_path) && cb_path.user_sgpr >= 0 &&
                         cb_path.user_sgpr + 1 < 16 && (!pointer || cb_path.immediate);
        if (as_buffer) {
            const std::string path_key = cb_path.str();  // exclusions name the path, whichever way it is read
            cb_key = pointer ? path_key + "#ptr" : path_key;
            as_buffer = std::find(opt.cb_ssbo_exclude.begin(), opt.cb_ssbo_exclude.end(), path_key) == opt.cb_ssbo_exclude.end() &&
                        (buffer_index.count(cb_key) || res.buffers.size() < kMaxBuffers);
        }
        if (in.op < 8) {
            base = read_u64(in.src0, in);
        } else {
            // V#: base = word0 | (word1[7:0]) << 32 (40-bit addresses; Sony
            // keeps memory-type bits in the upper part of BASE_ADDRESS_HI)
            const Id w0 = read_s(in.src0, in), w1 = read_s(in.src0 + 1, in);
            base = make_u64(w0, iand(w1, cu(0xff)));
        }
        const Id addr = addr_add(base, offset);
        if (!as_buffer) {
            ResourcePath traced;
            const char* why = !opt.cb_ssbo                         ? " with constant buffers off"
                              : !known                             ? " at a register offset"
                              : !sym_path(in.src0, 2, traced)      ? " from an untraced base"
                              : traced.user_sgpr < 0 || traced.user_sgpr + 1 >= 16 ? " from outside user data"
                              : pointer && !traced.immediate       ? " through a loaded pointer"
                              : res.buffers.size() >= kMaxBuffers  ? " past the buffer limit"
                                                                   : " excluded";
            note_walk(std::string(pointer ? "s_load" : "s_buffer_load") + why);
        }
        const std::vector<Id> loaded = as_buffer ? load_buffer_run(cb_path, cb_key, pointer, in.src0, known_off, count, addr)
                                                 : load_u32_run(addr, count);
        in_smrd_write = true;
        for (int k = 0; k < count; ++k) {
            const Id v = loaded[k];
            write_s(in.sdst + k, v);
            if (in.sdst + k < 104) m.store(smrd_shadow_var(in.sdst + k), v);
            pending.values.push_back(v);
        }
        in_smrd_write = false;
        sym_load(in.sdst, count, base_sym, known_off, known, in.op >= 8);
        for (int k = 0; k < count; ++k) sgpr_defs[in.sdst + k].push_back({def_seq++, sym_key(sym_of(in.sdst + k)), cur_block});
        for (int k = 0; k < count; ++k) pending.syms.push_back(sym_of(in.sdst + k));
        pending_smrd.push_back(std::move(pending));
    }
    // Re-stores the pending loads' values (from their shadow variables, so
    // the SSA values need not dominate) and restores the resource paths.
    void land_pending_smrd() {
        for (const PendingSmrd& p : pending_smrd) {
            for (std::size_t k = 0; k < p.values.size(); ++k) {
                const int reg = p.sdst + static_cast<int>(k);
                if (reg >= 104) continue;
                m.store(sgpr_var(reg), m.load(t_u32, smrd_shadow_var(reg)));
                lane_forget(static_cast<std::uint16_t>(reg));
                if (p.syms[k].kind != SymVal::Unknown) sym[reg] = p.syms[k];
            }
        }
        pending_smrd.clear();
    }

    // ---- buffer memory
    struct BufferAddr {
        Id addr;   // u64 guest address of the element
        Id stride;
    };
    BufferAddr buffer_address(const Inst& in) {
        const Id w0 = read_s(in.srsrc, in), w1 = read_s(in.srsrc + 1, in);
        const Id base = make_u64(w0, iand(w1, cu(0xff)));
        const Id stride = iand(shr(w1, cu(16)), cu(0x3fff));
        Id off = cu(in.offset12);
        off = iadd(off, read_s(in.soffset, in));
        int va = in.vaddr;
        if (in.idxen) {
            const Id index = ld(vgpr_var(va++));
            off = iadd(off, imul(index, stride));
        }
        if (in.offen) {
            off = iadd(off, ld(vgpr_var(va)));
        }
        return {addr_add(base, off), stride};
    }
    void mubuf(const Inst& in) {
        const std::uint32_t op = in.op;
        if (op < 112) note_walk(mnemonic(in) + buffer_walk_form(in));
        if ((op >= 4 && op <= 7) || (op >= 24 && op <= 31) || (op >= 48 && op < 112)) note_store(in);
        auto load_n = [&](int n) {
            const BufferAddr ba = buffer_address(in);
            const std::vector<Id> vals = load_u32_run(ba.addr, n);
            for (int k = 0; k < n; ++k) write_v(in.vdata + k, vals[k]);
        };
        auto store_n = [&](int n) {
            const BufferAddr ba = buffer_address(in);
            std::vector<Id> vals;
            for (int k = 0; k < n; ++k) vals.push_back(ld(vgpr_var(in.vdata + k)));
            exec_guard([&] {
                for (int k = 0; k < n; ++k) store_u32_at(addr_addc(ba.addr, k * 4), vals[k]);
            });
        };
        switch (op) {
        case 0: case 1: case 2: case 3: {  // buffer_load_format_*: format from V#
            const BufferAddr ba = buffer_address(in);
            const Id w3 = read_s(in.srsrc + 3, in);
            const Id dfmt = ubfe(w3, 15, 4), nfmt = ubfe(w3, 12, 3);
            std::vector<Id> comps = format_load_dynamic(ba.addr, dfmt, nfmt, 4);
            const Id is_int = bor(ieq(nfmt, cu(4)), ieq(nfmt, cu(5)));
            write_swizzled(in.vdata, static_cast<int>(op) + 1, comps, w3, is_int);
            break;
        }
        case 4: case 5: case 6: case 7: {  // buffer_store_format_*: format from V#
            const BufferAddr ba = buffer_address(in);
            const Id w3 = read_s(in.srsrc + 3, in);
            const Id dfmt = ubfe(w3, 15, 4), nfmt = ubfe(w3, 12, 3);
            const int nv = static_cast<int>(op) - 3;
            std::vector<Id> vals;
            for (int k = 0; k < nv; ++k) vals.push_back(ld(vgpr_var(in.vdata + k)));
            exec_guard([&] { format_store_dynamic(ba.addr, dfmt, nfmt, vals); });
            break;
        }
        case 8: case 9: case 10: case 11: {  // ubyte/sbyte/ushort/sshort
            const BufferAddr ba = buffer_address(in);
            const Id aligned = m.emit(spv::OpBitwiseAnd, t_u64, {ba.addr, m.const_u64(~3ull)});
            const Id shift = shl(m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpBitwiseAnd, t_u64, {ba.addr, m.const_u64(3)})}), cu(3));
            const Id word = shr(load_u32_at(aligned), shift);
            Id v;
            if (op == 8) v = iand(word, cu(0xff));
            else if (op == 9) v = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(word), c_zero_u, cu(8)}));
            else if (op == 10) v = iand(word, cu(0xffff));
            else v = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(word), c_zero_u, cu(16)}));
            write_v(in.vdata, v);
            break;
        }
        case 12: load_n(1); break;
        case 13: load_n(2); break;
        case 14: load_n(4); break;
        case 15: load_n(3); break;
        case 24: case 26: {  // byte/short store: only its bytes, atomically (a neighbour's survive)
            const BufferAddr ba = buffer_address(in);
            const Id mask = cu(op == 24 ? 0xffu : 0xffffu);
            const Id val = iand(ld(vgpr_var(in.vdata)), mask);
            exec_guard([&] { store_masked_any(ba.addr, val, c_zero_u, mask, c_zero_u); });
            break;
        }
        case 28:
            // In a hull shader the first six single-dword buffer stores are
            // the patch's tessellation factors on their way to the factor
            // ring - four outer then two inner, which is the order the
            // hardware reads them back in. The host tessellator takes them
            // from the built-ins instead, and the ring itself is never set up
            // by this game (its size and base registers are 0).
            // A triangle patch has four: three outer, one inner.
            if (opt.tess_role == TranslateOptions::TessRole::HullTcs && tf_stores_seen < (opt.tess_quads ? 6 : 4)) {
                const int k = tf_stores_seen++;
                const int nouter = opt.tess_quads ? 4 : 3;
                const Id val = f(ld(vgpr_var(in.vdata)));
                const Id arr = k < nouter ? out_tess_outer : out_tess_inner;
                const Id idx = cu(static_cast<std::uint32_t>(k < nouter ? k : k - nouter));
                exec_guard([&] { m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_f32), arr, {idx}), val); });
                break;
            }
            store_n(1);
            break;
        case 29: store_n(2); break;
        case 30: store_n(4); break;
        case 31: store_n(3); break;
        case 50: {  // buffer_atomic_add (returns pre-op value when glc)
            const BufferAddr ba = buffer_address(in);
            const Id val = ld(vgpr_var(in.vdata));
            Id result_var = m.local_variable(p_fn_u32, c_zero_u);
            exec_guard([&] {
                const Id p = guest_ptr(ba.addr);
                const Id r = m.emit(spv::OpAtomicIAdd, t_u32, {p, cu(spv::ScopeDevice), cu(spv::MsNone), val});
                m.store(result_var, r);
            });
            if (in.glc) write_v(in.vdata, ld(result_var));
            break;
        }
        case 112: case 113: break;  // cache writeback/invalidate
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    // Component layout of a buffer data format: returns {bits per component..., count}.
    static bool format_layout(std::uint32_t dfmt, int& count, int bits[4]) {
        switch (dfmt) {
        case 1: count = 1; bits[0] = 8; return true;
        case 2: count = 1; bits[0] = 16; return true;
        case 3: count = 2; bits[0] = bits[1] = 8; return true;
        case 4: count = 1; bits[0] = 32; return true;
        case 5: count = 2; bits[0] = bits[1] = 16; return true;
        case 6: count = 3; bits[0] = 11; bits[1] = 11; bits[2] = 10; return true;  // 10_11_11 (x=11? order per spec: R11 G11 B10)
        case 7: count = 3; bits[0] = 10; bits[1] = 11; bits[2] = 11; return true;
        case 8: count = 4; bits[0] = 2; bits[1] = 10; bits[2] = 10; bits[3] = 10; return true;  // 10_10_10_2: R low? see below
        case 9: count = 4; bits[0] = 10; bits[1] = 10; bits[2] = 10; bits[3] = 2; return true;
        case 10: count = 4; bits[0] = bits[1] = bits[2] = bits[3] = 8; return true;
        case 11: count = 2; bits[0] = bits[1] = 32; return true;
        case 12: count = 4; bits[0] = bits[1] = bits[2] = bits[3] = 16; return true;
        case 13: count = 3; bits[0] = bits[1] = bits[2] = 32; return true;
        case 14: count = 4; bits[0] = bits[1] = bits[2] = bits[3] = 32; return true;
        default: return false;
        }
    }
    // Convert one raw component (bits wide, right-aligned in `raw`) per numeric format.
    Id convert_component(Id raw, int bits, std::uint32_t nfmt) {
        const float maxu = static_cast<float>((1ull << bits) - 1);
        const float maxs = static_cast<float>((1ull << (bits - 1)) - 1);
        switch (nfmt) {
        case 0:  // unorm
            if (bits == 32) return raw;
            return u(fdiv(u2f(raw), cf(maxu)));
        case 1: {  // snorm
            if (bits == 32) return raw;
            const Id s = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(raw), c_zero_u, cu(bits)}));
            return u(fmax_(fdiv(i2f(s), cf(maxs)), cf(-1.0f)));
        }
        case 2: return u(u2f(raw));  // uscaled
        case 3: return u(i2f(ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(raw), c_zero_u, cu(bits)}))));
        case 4: return raw;          // uint
        case 5: return ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(raw), c_zero_u, cu(bits)}));
        case 6: {  // snorm_ogl
            const Id s = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(raw), c_zero_u, cu(bits)}));
            return u(fdiv(i2f(s), cf(maxs)));
        }
        case 7:  // float
            if (bits == 32) return raw;
            if (bits == 16) return u(extract(t_f32, m.ext_inst(t_v2f, spv::GlslUnpackHalf2x16, {raw}), 0));
            if (bits == 11) return u(unpack_f11(raw));
            if (bits == 10) return u(unpack_f10(raw));
            return raw;
        default:
            return raw;
        }
    }
    Id unpack_f11(Id raw) {  // 5-bit exponent, 6-bit mantissa, no sign -> f32 via half trick
        // f11 = half without sign and with a 6-bit mantissa: shift into half layout (mantissa 10 bits)
        const Id half = shl(iand(raw, cu(0x7ff)), cu(4));
        return extract(t_f32, m.ext_inst(t_v2f, spv::GlslUnpackHalf2x16, {half}), 0);
    }
    Id unpack_f10(Id raw) {
        const Id half = shl(iand(raw, cu(0x3ff)), cu(5));
        return extract(t_f32, m.ext_inst(t_v2f, spv::GlslUnpackHalf2x16, {half}), 0);
    }
    // Applies the V#'s DST_SEL_X..W (word3 bits 11:0) to loaded components:
    // 0 -> 0, 1 -> 1 (float 1.0 unless the format is an integer one), 4..7 -> component.
    std::vector<Id> apply_dst_sel_rt(const std::vector<Id>& comps, Id w3, Id integer_fmt) {
        std::vector<Id> out;
        const Id one = sel(integer_fmt, c_one_u, u(cf(1.0f)));
        for (int k = 0; k < 4; ++k) {
            const Id s3 = ubfe(w3, 3 * k, 3);
            Id v = sel(ieq(s3, cu(4)), comps[0],
                       sel(ieq(s3, cu(5)), comps[1],
                           sel(ieq(s3, cu(6)), comps[2],
                               sel(ieq(s3, cu(7)), comps[3], sel(ieq(s3, c_one_u), one, c_zero_u)))));
            out.push_back(v);
        }
        return out;
    }
    // Writes `n` loaded components to VGPRs vdata.. through the V#'s DST_SEL.
    // Nearly every V# a shader loads through - arrays of constants, instance
    // data - selects X, Y, Z, W: the components go as loaded, and the selects
    // that apply another order run only when it is another (a branch on word
    // 3, which is uniform). They were most of the hot vertex shaders' VALU
    // (~120 of 348 in 9da6aacb). BBHOST_DST_SEL_BRANCH=0: the selects always.
    void write_swizzled(int vdata, int n, const std::vector<Id>& comps, Id w3, Id integer_fmt) {
        static const bool branch = [] {
            const char* e = std::getenv("BBHOST_DST_SEL_BRANCH");
            return !(e && e[0] == '0');
        }();
        if (!branch) {
            const std::vector<Id> swz = apply_dst_sel_rt(comps, w3, integer_fmt);
            for (int k = 0; k < n; ++k) write_v(vdata + k, swz[k]);
            return;
        }
        for (int k = 0; k < n; ++k) write_v(vdata + k, comps[k]);
        const std::uint32_t mask = (1u << (3 * n)) - 1;
        const Id other = m.emit(spv::OpINotEqual, t_bool, {iand(w3, cu(mask)), cu(0xfacu & mask)});
        if_then(other, [&] {
            const std::vector<Id> swz = apply_dst_sel_rt(comps, w3, integer_fmt);
            for (int k = 0; k < n; ++k) write_v(vdata + k, swz[k]);
        });
    }
    // Static-format load of `want` components from addr.
    std::vector<Id> format_load(Id addr, std::uint32_t dfmt, std::uint32_t nfmt, int want) {
        int count, bits[4];
        if (!format_layout(dfmt, count, bits)) {
            error("unsupported buffer data format " + std::to_string(dfmt));
            return std::vector<Id>(4, c_zero_u);
        }
        int total = 0;
        for (int k = 0; k < count; ++k) total += bits[k];
        // Load enough dwords (an 8- or 16-bit element need not sit on one).
        return format_decode(load_u32_any(addr, (total + 31) / 32), dfmt, nfmt, want);
    }
    // TranslateOptions::cb_ssbo: a typed load indexed into a V# traced to user
    // data or a table (tbuffer_load_format_* with idxen: arrays of constants)
    // reads the V#'s storage buffer at dword (offset12 + soffset) / 4 + index *
    // stride / 4, with the stride and word 3 the host passes. The host binds
    // the V#'s records and leaves the binding unbound when the stride is not in
    // dwords. False when the load has another shape.
    const char* indexed_reject = "";  // why indexed_buffer_load declined the last load, for TranslateResult::walks
    Id cb_w3_spec[kMaxBuffers] = {};   // the specialization constant of each buffer's word 3, once used
    Id cb_stride_spec[kMaxBuffers] = {};  // and of its stride (kCbStrideSpecId)
    bool indexed_buffer_load(const Inst& in, std::vector<Id>& comps, Id& w3) {
        int count = 0, bits[4] = {};
        const auto reject = [&](const char* why) {
            indexed_reject = why;
            return false;
        };
        if (!opt.cb_ssbo) return reject("");
        if (!in.idxen || in.offen || in.addr64) return reject(", offset register or 64-bit address");
        if (in.soffset < 128 || in.soffset > 208) return reject(", register offset");
        if (!format_layout(in.dfmt, count, bits)) return reject(", unknown format");
        const std::int32_t constant = in.soffset <= 192 ? static_cast<std::int32_t>(in.soffset) - 128 : 192 - static_cast<std::int32_t>(in.soffset);
        const std::int32_t off = static_cast<std::int32_t>(in.offset12) + constant;
        ResourcePath path;
        if (off < 0 || off % 4) return reject(", offset not in dwords");
        if (!sym_path(in.srsrc, 2, path) || path.user_sgpr < 0 || path.user_sgpr + 1 >= 16) return reject(", untraced V#");
        const std::string key = path.str();
        if (std::find(opt.cb_ssbo_exclude.begin(), opt.cb_ssbo_exclude.end(), key) != opt.cb_ssbo_exclude.end()) return reject(", excluded");
        if (!buffer_index.count(key) && res.buffers.size() >= kMaxBuffers) return reject(", past the buffer limit");
        int total = 0;
        for (int k = 0; k < count; ++k) total += bits[k];
        const int nwords = (total + 31) / 32;
        const std::size_t index = buffer_for(path, key, false, static_cast<std::uint32_t>(nwords));
        res.buffers[index].indexed = true;
        Id stride = params_u32(5, index);
        if (opt.cb_no_fallback) {
            Id& spec = cb_stride_spec[index];  // translate.h kCbStrideSpecId
            if (!spec) spec = m.spec_const_u32(0, kCbStrideSpecId + static_cast<std::uint32_t>(index));
            stride = sel(ine(spec, c_zero_u), spec, stride);
        }
        const Id stride_dw = shr(stride, cu(2));
        const Id first = iadd(cu(static_cast<std::uint32_t>(off / 4)), imul(ld(vgpr_var(in.vaddr)), stride_dw));
        // The element's guest address feeds only the fallback walk.
        const Id guest = opt.cb_no_fallback ? m.const_u64(0) : buffer_address(in).addr;
        comps = format_decode(read_bound(index, in.srsrc, first, nwords, guest), in.dfmt, in.nfmt, 4);
        if (opt.cb_no_fallback) {
            Id& spec = cb_w3_spec[index];  // translate.h kCbW3SpecId
            if (!spec) spec = m.spec_const_u32(0, kCbW3SpecId + static_cast<std::uint32_t>(index));
            w3 = sel(ine(spec, c_zero_u), spec, params_u32(6, index));
        } else {
            w3 = sel(buffer_bound(index), params_u32(6, index), read_s(in.srsrc + 3, in));
        }
        return true;
    }
    // A static format's `want` components from the dwords holding one element.
    std::vector<Id> format_decode(const std::vector<Id>& words, std::uint32_t dfmt, std::uint32_t nfmt, int want) {
        int count, bits[4];
        std::vector<Id> out(4, c_zero_u);
        if (!format_layout(dfmt, count, bits)) return out;
        int bit = 0;
        for (int k = 0; k < count && k < want; ++k) {
            Id raw;
            if (bits[k] == 32) {
                raw = words[bit / 32];
            } else {
                raw = ubfe(words[bit / 32], bit % 32, bits[k]);
            }
            out[k] = convert_component(raw, bits[k], nfmt);
            bit += bits[k];
        }
        // Missing components default: 0,0,0,1.0 (as float) for float formats.
        for (int k = count; k < 4; ++k) out[k] = k == 3 ? (nfmt == 4 || nfmt == 5 ? c_one_u : u(cf(1.0f))) : c_zero_u;
        return out;
    }
    // TranslateOptions::vertex_formats_from_params: an element converted as
    // load_vertex_input converts it for a constant word 3, with the conversion
    // read from StageParams::vertex_formats (vertex_format_descriptor) at run
    // time. The kinds are the four a cold world load's elements use, so the
    // loader stays a few ops a component: a loader with every number format and
    // swizzle made vertex shaders 58% larger and a cold load's command-processor
    // compiles 5 s slower. Each candidate is convert_component's own expression,
    // constants included: a divisor the driver only sees at run time rounds
    // differently (a skinned mesh's unorm8 weights moved its silhouette).
    Id vertex_format_spec[16] = {};
    void load_vertex_element_from_params(const VertexElement& el, Id raw) {
        if (el.location >= 16) {
            error("vertex input: formats from params need locations below 16");
            return;
        }
        Id& spec = vertex_format_spec[el.location];  // one per location, whichever call site loads it
        if (!spec) spec = m.spec_const_u32(0, kVertexFormatSpecId + el.location);  // translate.h
        const Id from_params = m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(7), cu(el.location / 4), cu(el.location % 4)}));
        const Id d = sel(ine(spec, c_zero_u), spec, from_params);
        const Id kind = iand(d, cu(7));
        const Id count = iand(shr(d, cu(4)), cu(7));
        const Id one = sel(bor(ieq(kind, cu(2)), ieq(kind, cu(4))), c_one_u, u(cf(1.0f)));
        for (std::uint32_t k = 0; k < el.count && k < 4; ++k) {
            const Id c = extract(t_u32, raw, k);
            const Id unorm8 = u(fdiv(u2f(c), cf(255.0f)));
            const Id sint16 = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(c), c_zero_u, cu(16)}));
            const Id converted = sel(ieq(kind, cu(3)), unorm8, sel(ieq(kind, cu(4)), sint16, c));  // kinds 1 and 2: the raw bits
            const Id past = sel(ine(iand(d, cu(0x100u << k)), c_zero_u), one, c_zero_u);     // DST_SEL 1 or 0 past the format's components
            write_v(static_cast<int>(el.vdata + k), sel(ult(cu(k), count), converted, past));
        }
    }
    // TranslateOptions::vertex_input: at the fetch shader's call, each element
    // from its vertex input (raw components in an unsigned format as wide as
    // the V#'s data format; packed formats as one dword), converted as the
    // number format and DST_SEL in the record's word 3 say, which is what
    // buffer_load_format_* does with the record at run time.
    void load_vertex_input() {
        for (const VertexElement& el : opt.vertex_input) {
            Id& var = vertex_in_vars[el.location];
            if (!var) {
                var = m.global_variable(p_in_v4u, spv::ScInput);
                m.decorate(var, spv::DecLocation, {el.location});
                interface.push_back(var);
            }
            const Id raw = m.load(t_v4u, var);
            if (opt.vertex_formats_from_params) {
                load_vertex_element_from_params(el, raw);
                continue;
            }
            const std::uint32_t dfmt = (el.w3 >> 15) & 0xf, nfmt = (el.w3 >> 12) & 7;
            int count = 0, bits[4] = {};
            if (!format_layout(dfmt, count, bits)) {
                error("vertex input: unsupported buffer data format " + std::to_string(dfmt));
                return;
            }
            const bool packed = dfmt >= 6 && dfmt <= 9;
            const Id one = nfmt == 4 || nfmt == 5 ? c_one_u : u(cf(1.0f));
            std::vector<Id> comps(4, c_zero_u);
            int bit = 0;
            for (int k = 0; k < count; ++k) {
                const Id c = packed ? ubfe(extract(t_u32, raw, 0), bit, bits[k]) : extract(t_u32, raw, k);
                comps[k] = convert_component(c, bits[k], nfmt);
                bit += bits[k];
            }
            if (count < 4) comps[3] = one;
            for (std::uint32_t k = 0; k < el.count && k < 4; ++k) {
                const std::uint32_t s3 = (el.w3 >> (3 * k)) & 7;
                write_v(static_cast<int>(el.vdata + k), s3 >= 4 ? comps[s3 - 4] : s3 == 1 ? one : c_zero_u);
            }
        }
    }
    // Dynamic format (from V#): supports the 32-bit and 16-bit families; others load raw dwords.
    std::vector<Id> format_load_dynamic(Id addr, Id dfmt, Id nfmt, int want) {
        std::vector<Id> out;
        // Select stride/width by dfmt at runtime: handle 32, 32_32, 32_32_32, 32_32_32_32 (raw floats),
        // 16, 16_16, 16_16_16_16 (float16 or unorm), 8_8_8_8 (unorm) — common vertex/format cases.
        const std::vector<Id> words = load_u32_any(addr, want);
        const Id w0 = words[0];
        const Id w1 = want > 1 ? words[1] : w0;
        const Id w2 = want > 2 ? words[2] : w0;
        const Id w3 = want > 3 ? words[3] : w0;
        const Id is_half = ieq(nfmt, cu(7));
        const Id is_unorm = ieq(nfmt, c_zero_u);
        // Component count of the format decides which slots exist; the rest read as 0,0,0,1.
        const Id cnt = sel(bor(bor(ieq(dfmt, cu(1)), ieq(dfmt, cu(2))), ieq(dfmt, cu(4))), c_one_u,
                       sel(bor(bor(ieq(dfmt, cu(3)), ieq(dfmt, cu(5))), ieq(dfmt, cu(11))), cu(2),
                           sel(ieq(dfmt, cu(13)), cu(3), cu(4))));
        // Number format applied to a narrow component: unorm/snorm scale,
        // u/sscaled convert, uint/sint stay integers (bone indices!), float
        // is half for 16-bit.
        auto narrow = [&](Id raw, int bits, Id flt) {
            const Id sext = ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(raw), c_zero_u, cu(bits)}));
            const float maxu = static_cast<float>((1u << bits) - 1), maxs = static_cast<float>((1u << (bits - 1)) - 1);
            const Id unorm = u(fdiv(u2f(raw), cf(maxu)));
            const Id snorm = u(fmax_(fdiv(i2f(sext), cf(maxs)), cf(-1.0f)));
            const Id uscaled = u(u2f(raw)), sscaled = u(i2f(sext));
            return sel(ieq(nfmt, c_zero_u), unorm,
                   sel(bor(ieq(nfmt, c_one_u), ieq(nfmt, cu(6))), snorm,
                   sel(ieq(nfmt, cu(2)), uscaled,
                   sel(ieq(nfmt, cu(3)), sscaled,
                   sel(ieq(nfmt, cu(4)), raw,
                   sel(ieq(nfmt, cu(5)), sext, flt))))));
        };
        (void)is_half;
        (void)is_unorm;
        for (int k = 0; k < want; ++k) {
            const Id word32 = k == 0 ? w0 : k == 1 ? w1 : k == 2 ? w2 : w3;
            // 16-bit: components packed 2 per dword
            const Id raw16 = ubfe(k < 2 ? w0 : w1, (k & 1) * 16, 16);
            const Id half = u(extract(t_f32, m.ext_inst(t_v2f, spv::GlslUnpackHalf2x16, {raw16}), 0));
            const Id v16 = narrow(raw16, 16, half);
            const Id raw8 = ubfe(w0, k * 8, 8);
            const Id v8 = narrow(raw8, 8, raw8);
            const Id is16 = bor(bor(ieq(dfmt, cu(2)), ieq(dfmt, cu(5))), ieq(dfmt, cu(12)));
            const Id is8 = bor(ieq(dfmt, cu(1)), bor(ieq(dfmt, cu(3)), ieq(dfmt, cu(10))));
            const Id loaded = sel(is16, v16, sel(is8, v8, word32));
            const Id missing = k == 3 ? sel(bor(ieq(nfmt, cu(4)), ieq(nfmt, cu(5))), c_one_u, u(cf(1.0f))) : c_zero_u;
            out.push_back(sel(ult(cu(static_cast<std::uint32_t>(k)), cnt), loaded, missing));
        }
        return out;
    }
    // Store `vals` (raw dwords holding floats or ints) in the V#'s data format.
    void format_store_dynamic(Id addr, Id dfmt, Id nfmt, const std::vector<Id>& vals) {
        const int n = static_cast<int>(vals.size());
        auto f_at = [&](int k) { return k < n ? f(vals[k]) : cf(0.0f); };
        auto u_at = [&](int k) { return k < n ? vals[k] : c_zero_u; };
        const Id is_float = ieq(nfmt, cu(7));
        const Id is_unorm = ieq(nfmt, c_zero_u);
        // 32-bit components: dfmt 4 (x), 11 (xy), 13 (xyz), 14 (xyzw): raw stores.
        const Id is32 = bor(bor(ieq(dfmt, cu(4)), ieq(dfmt, cu(11))), bor(ieq(dfmt, cu(13)), ieq(dfmt, cu(14))));
        if_then(is32, [&] {
            const Id cnt = sel(ieq(dfmt, cu(4)), c_one_u, sel(ieq(dfmt, cu(11)), cu(2), sel(ieq(dfmt, cu(13)), cu(3), cu(4))));
            for (int k = 0; k < n; ++k) {
                if_then(ult(cu(static_cast<std::uint32_t>(k)), cnt), [&] { store_u32_at(addr_addc(addr, k * 4), vals[k]); });
            }
        });
        // 16-bit components: dfmt 2 (x), 5 (xy), 12 (xyzw): half or unorm16.
        const Id is16 = bor(bor(ieq(dfmt, cu(2)), ieq(dfmt, cu(5))), ieq(dfmt, cu(12)));
        if_then(is16, [&] {
            auto pack16 = [&](int k) {
                const Id half = m.ext_inst(t_u32, spv::GlslPackHalf2x16, {m.emit(spv::OpCompositeConstruct, t_v2f, {f_at(k), f_at(k + 1)})});
                auto un = [&](Id v) { return f2u(fadd(fmul(m.ext_inst(t_f32, spv::GlslFClamp, {v, cf(0.0f), cf(1.0f)}), cf(65535.0f)), cf(0.5f))); };
                const Id unorm = ior(un(f_at(k)), shl(un(f_at(k + 1)), cu(16)));
                const Id raw = ior(iand(u_at(k), cu(0xffff)), shl(u_at(k + 1), cu(16)));
                return sel(is_float, half, sel(is_unorm, unorm, raw));
            };
            // 2, 4 or 8 bytes: a lone 16-bit component leaves its neighbour's two.
            const Id m0 = sel(ieq(dfmt, cu(2)), cu(0xffffu), cu(0xffffffffu));
            const Id m1 = sel(ieq(dfmt, cu(12)), cu(0xffffffffu), c_zero_u);
            store_masked_any(addr, pack16(0), pack16(2), m0, m1);
        });
        // 8-bit components: dfmt 1 (x), 3 (xy), 10 (xyzw): unorm8 or raw bytes.
        const Id is8 = bor(bor(ieq(dfmt, cu(1)), ieq(dfmt, cu(3))), ieq(dfmt, cu(10)));
        if_then(is8, [&] {
            Id packed = c_zero_u;
            for (int k = 0; k < 4; ++k) {
                const Id un = f2u(fadd(fmul(m.ext_inst(t_f32, spv::GlslFClamp, {f_at(k), cf(0.0f), cf(1.0f)}), cf(255.0f)), cf(0.5f)));
                const Id byte = sel(is_unorm, un, iand(u_at(k), cu(0xff)));
                packed = ior(packed, shl(byte, cu(8 * k)));
            }
            // 1, 2 or 4 bytes, wherever the element sits.
            const Id m0 = sel(ieq(dfmt, cu(1)), cu(0xffu), sel(ieq(dfmt, cu(3)), cu(0xffffu), cu(0xffffffffu)));
            store_masked_any(addr, packed, c_zero_u, m0, c_zero_u);
        });
    }
    void mtbuf(const Inst& in) {
        const int n = (in.op & 3) + 1;
        if (in.op < 4) {
            std::vector<Id> comps;
            Id w3 = 0;
            if (!indexed_buffer_load(in, comps, w3)) {
                note_walk(mnemonic(in) + buffer_walk_form(in) + indexed_reject);
                comps = format_load(buffer_address(in).addr, in.dfmt, in.nfmt, 4);
                w3 = read_s(in.srsrc + 3, in);
            }
            const Id is_int = m.const_bool(in.nfmt == 4 || in.nfmt == 5);
            write_swizzled(in.vdata, n, comps, w3, is_int);
        } else {
            note_walk(mnemonic(in) + buffer_walk_form(in));
            note_store(in);
            const BufferAddr ba = buffer_address(in);
            int count, bits[4];
            if (!format_layout(in.dfmt, count, bits)) {
                error("unsupported tbuffer store format " + std::to_string(in.dfmt));
                return;
            }
            std::vector<Id> vals;
            for (int k = 0; k < n; ++k) vals.push_back(ld(vgpr_var(in.vdata + k)));
            exec_guard([&] {
                if (bits[0] == 32) {
                    for (int k = 0; k < n && k < count; ++k) store_u32_at(addr_addc(ba.addr, k * 4), vals[k]);
                } else if (bits[0] == 16 && in.nfmt == 7) {
                    for (int k = 0; k < n && k < count; k += 2) {
                        const Id a = f(vals[k]);
                        const Id b = (k + 1 < n) ? f(vals[k + 1]) : cf(0.0f);
                        const Id packed = m.ext_inst(t_u32, spv::GlslPackHalf2x16, {m.emit(spv::OpCompositeConstruct, t_v2f, {a, b})});
                        // A lone last component (16-bit x) leaves the next two bytes.
                        store_masked_any(addr_addc(ba.addr, (k / 2) * 4), packed, c_zero_u, cu(k + 1 < count ? 0xffffffffu : 0xffffu), c_zero_u);
                    }
                } else if (bits[0] == 8 && count == 4 && in.nfmt == 0) {
                    Id packed = c_zero_u;
                    for (int k = 0; k < 4; ++k) {
                        const Id v = k < n ? f(vals[k]) : cf(0.0f);
                        const Id b = f2u(fadd(fmul(m.ext_inst(t_f32, spv::GlslFClamp, {v, cf(0.0f), cf(1.0f)}), cf(255.0f)), cf(0.5f)));
                        packed = ior(packed, shl(b, cu(8 * k)));
                    }
                    store_masked_any(ba.addr, packed, c_zero_u, cu(0xffffffffu), c_zero_u);
                } else {
                    error("unsupported tbuffer store format " + std::to_string(in.dfmt) + "/" + std::to_string(in.nfmt));
                }
            });
        }
    }

    // ---- LDS
    // An LDS byte address in the buffer standing in for LDS: the dword's
    // offset from lds_base64, and (TranslateOptions::tess_lds_bound) whether
    // it is inside the window and the window inside the buffer - 0 when
    // nothing is checked. Outside, the offset is 0, which is always there.
    std::pair<Id, Id> lds_buffer_at(Id byte_addr) {
        const Id off = iand(byte_addr, cu(~3u));
        if (!lds_limit) return {off, 0};
        Id ok = ult(off, lds_limit);
        if (lds_patch_ok) ok = band(ok, lds_patch_ok);
        return {sel(ok, off, c_zero_u), ok};
    }
    Id lds_buffer_ptr(Id off) {
        const Id at = m.emit(spv::OpIAdd, t_u64, {lds_base64, m.emit(spv::OpUConvert, t_u64, {off})});
        const Id block = m.emit(spv::OpConvertUToPtr, p_psb_block, {at});
        return m.access_chain(p_psb_u32, block, {c_zero_u});
    }
    Id lds_ptr(Id byte_addr) {
        if (lds_in_buffer) return lds_buffer_ptr(lds_buffer_at(byte_addr).first);
        const Id idx = shr(byte_addr, cu(2));
        return m.access_chain(p_wg_u32, v_lds, {idx});
    }
    // The dword a flat LDS address names, as (uvec4 slot, component). The
    // address is the offset inside one patch once the patch term is zero, and
    // it is taken modulo the slot count so a read past the control point - the
    // patch constants the hull shader writes, which these domain shaders do
    // not use - lands somewhere defined instead of out of bounds.
    std::pair<Id, Id> attr_index(Id byte_addr) {
        const Id d = shr(byte_addr, cu(2));
        const Id slot = m.emit(spv::OpUMod, t_u32, {shr(d, cu(2)), cu(std::max<std::uint32_t>(1, opt.tess_attr_vec4s))});
        return {slot, iand(d, cu(3))};
    }
    Id lds_load(Id byte_addr) {
        if (lds_in_attributes) {
            if (!v_attr_in) return c_zero_u;
            const auto [slot, comp] = attr_index(byte_addr);
            // One control point a patch, so the per-vertex array has one entry.
            return m.load(t_u32, m.access_chain(m.type_pointer(spv::ScInput, t_u32), v_attr_in, {c_zero_u, slot, comp}));
        }
        if (lds_in_buffer) {
            // Bounded, the load is of a dword that is there whatever the
            // address, and what it gives is 0 past the end - every lane runs
            // it, EXEC or not, so a lane switched off reads too.
            const auto [off, ok] = lds_buffer_at(byte_addr);
            const Id v = m.emit(spv::OpLoad, t_u32, {lds_buffer_ptr(off), 2u /* Aligned */, 4u});
            return ok ? sel(ok, v, c_zero_u) : v;
        }
        return m.load(t_u32, lds_ptr(byte_addr));
    }
    void lds_store(Id byte_addr, Id v) {
        if (lds_in_attributes) {
            if (!v_attr_out) return;
            const auto [slot, comp] = attr_index(byte_addr);
            m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_u32), v_attr_out, {slot, comp}), v);
            return;
        }
        if (lds_in_buffer) {
            // Past the end the write is dropped, as the hardware drops it.
            const auto [off, ok] = lds_buffer_at(byte_addr);
            const auto store = [&, off = off] { m.emit_void(spv::OpStore, {lds_buffer_ptr(off), v, 2u /* Aligned */, 4u}); };
            if (ok) {
                if_then(ok, store);
            } else {
                store();
            }
            return;
        }
        m.store(lds_ptr(byte_addr), v);
    }
    void ds(const Inst& in) {
        const Id addr = ld(vgpr_var(in.vaddr));
        const std::uint32_t off = in.offset0 | (in.offset1 << 8);
        auto rd = [&](Id a) { return lds_load(a); };
        auto wr = [&](Id a, Id v) { lds_store(a, v); };
        switch (in.op) {
        case 13: {  // ds_write_b32
            const Id v = read_s(in.src0, in);
            exec_guard([&] { wr(iadd(addr, cu(off)), v); });
            break;
        }
        case 14: case 15: {  // ds_write2_b32 / st64
            const std::uint32_t mul = in.op == 14 ? 4 : 256;
            const Id v0 = read_s(in.src0, in), v1 = read_s(in.src1, in);
            exec_guard([&] {
                wr(iadd(addr, cu(in.offset0 * mul)), v0);
                wr(iadd(addr, cu(in.offset1 * mul)), v1);
            });
            break;
        }
        case 77: {  // ds_write_b64
            const Id v0 = read_s(in.src0, in), v1 = read_s(in.src0 + 1, in);
            exec_guard([&] {
                wr(iadd(addr, cu(off)), v0);
                wr(iadd(addr, cu(off + 4)), v1);
            });
            break;
        }
        case 54: write_v(in.dst, rd(iadd(addr, cu(off)))); break;  // ds_read_b32
        case 55: case 56: {
            const std::uint32_t mul = in.op == 55 ? 4 : 256;
            write_v(in.dst, rd(iadd(addr, cu(in.offset0 * mul))));
            write_v(in.dst + 1, rd(iadd(addr, cu(in.offset1 * mul))));
            break;
        }
        case 118: write_v(in.dst, rd(iadd(addr, cu(off)))); write_v(in.dst + 1, rd(iadd(addr, cu(off + 4)))); break;
        case 53: {  // ds_swizzle_b32
            // Its one operand is the VGPR in the address field; DATA0 is unused
            // (the game's programs leave it 0, so reading it shuffled v0).
            const Id v = addr;
            const Id l = lane();
            Id src_lane;
            if (off & 0x8000) {  // quad permute mode
                const Id q = iand(l, cu(~3u));
                const Id within = iand(l, cu(3));
                const Id shift = shl(within, c_one_u);
                const Id pick = iand(shr(cu(off & 0xff), shift), cu(3));
                src_lane = ior(q, pick);
            } else {
                const Id and_mask = cu(off & 0x1f), or_mask = cu((off >> 5) & 0x1f), xor_mask = cu((off >> 10) & 0x1f);
                const Id in_group = iand(l, cu(31));
                const Id sw = ixor(ior(iand(in_group, and_mask), or_mask), xor_mask);
                src_lane = ior(iand(l, cu(32)), sw);
            }
            write_v(in.dst, m.emit(spv::OpGroupNonUniformShuffle, t_u32, {cu(spv::ScopeSubgroup), v, src_lane}));
            break;
        }
        case 61: case 62: {  // ds_consume / ds_append: counter at M0 base + offset
            const Id base = iadd(ld(v_m0), cu(off));
            const Id active = exec_cond();
            Id blo, bhi;
            ballot(active, blo, bhi);
            const Id total = iadd(popcnt(blo), popcnt(bhi));
            const Id l = lane();
            const Id lt32 = ult(l, cu(32));
            const Id below_lo = iand(blo, sel(lt32, isub(shl(c_one_u, l), c_one_u), cu(0xffffffffu)));
            const Id below_hi = iand(bhi, sel(lt32, c_zero_u, isub(shl(c_one_u, isub(l, cu(32))), c_one_u)));
            const Id rank = iadd(popcnt(below_lo), popcnt(below_hi));
            const Id result = m.local_variable(p_fn_u32, c_zero_u);
            Id first = m.emit(spv::OpGroupNonUniformElect, t_bool, {cu(spv::ScopeSubgroup)});
            // A counter past the buffer's end is left alone (0 comes back).
            if (lds_in_buffer) {
                if (const Id ok = lds_buffer_at(base).second) first = band(first, ok);
            }
            if_then(first, [&] {
                const Id p = lds_ptr(base);
                const Id delta = in.op == 62 ? total : isub(c_zero_u, total);
                const Id r = m.emit(spv::OpAtomicIAdd, t_u32, {p, cu(spv::ScopeWorkgroup), cu(spv::MsNone), delta});
                m.store(result, r);
            });
            const Id broadcast = m.emit(spv::OpGroupNonUniformBroadcastFirst, t_u32, {cu(spv::ScopeSubgroup), ld(result)});
            write_v(in.dst, in.op == 62 ? iadd(broadcast, rank) : isub(broadcast, iadd(rank, c_one_u)));
            break;
        }
        default:
            error(std::string("unsupported ") + mnemonic(in));
        }
    }

    // ---- images
    void mimg(const Inst& in) {
        const std::uint32_t op = in.op;
        const bool is_sample = op >= 32 && op < 64;
        const bool is_gather = op >= 64 && op < 96;
        const bool is_sample_cd = op >= 104 && op < 112;
        if (op == 8 || op == 9) {  // image_store(_mip)
            std::uint32_t dim; bool arrayed;
            const ImageVar img = image_for(in, true, false, dim, arrayed);
            const int ncoord = coord_count(dim, arrayed);
            std::vector<Id> coords;
            for (int k = 0; k < ncoord; ++k) coords.push_back(i(ld(vgpr_var(in.vaddr + k))));
            const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2i : ncoord == 3 ? t_v3i : t_v4i, coords);
            std::vector<Id> vals;
            for (int k = 0; k < 4; ++k) {
                const Id raw = ld(vgpr_var(in.vdata + std::min(k, popcount4(in.dmask) - 1)));
                vals.push_back(img.kind == 1 ? raw : img.kind == 2 ? i(raw) : f(raw));
            }
            const Id texel = img.kind ? m.emit(spv::OpCompositeConstruct, texel_type(img), vals) : vec4f(vals[0], vals[1], vals[2], vals[3]);
            exec_guard([&] {
                const Id image = load_image(img);
                m.emit_void(spv::OpImageWrite, {image, coord, texel});
            });
            return;
        }
        if (op == 0 || op == 1) {  // image_load(_mip)
            std::uint32_t dim; bool arrayed;
            const ImageVar img = image_for(in, false, false, dim, arrayed);
            const int ncoord = coord_count(dim, arrayed);
            std::vector<Id> coords;
            for (int k = 0; k < ncoord; ++k) coords.push_back(i(ld(vgpr_var(in.vaddr + k))));
            const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2i : ncoord == 3 ? t_v3i : t_v4i, coords);
            const Id image = load_image(img);
            std::vector<std::uint32_t> ops = {image, coord};
            if (op == 1) {
                ops.push_back(spv::IoLod);
                ops.push_back(i(ld(vgpr_var(in.vaddr + ncoord))));
            }
            const Id texel = texel_as_v4f(img, m.emit(spv::OpImageFetch, texel_type(img), ops));
            write_texel(in, texel);
            return;
        }
        if (op == 14) {  // image_get_resinfo: vaddr = mip; returns w,h,d,mips by dmask
            std::uint32_t dim; bool arrayed;
            const ImageVar img = image_for(in, false, false, dim, arrayed);
            m.capability(spv::CapImageQuery);
            const Id image = load_image(img);
            const int ncomp = coord_count(dim, arrayed);
            const Id size_type = ncomp == 1 ? t_i32 : ncomp == 2 ? t_v2i : t_v3i;
            const Id size = m.emit(spv::OpImageQuerySizeLod, size_type, {image, i(ld(vgpr_var(in.vaddr)))});
            const Id levels = m.emit(spv::OpImageQueryLevels, t_i32, {image});
            Id comps[4] = {c_zero_u, c_zero_u, c_zero_u, ui(levels)};
            for (int k = 0; k < ncomp; ++k) comps[k] = ui(ncomp == 1 ? size : extract(t_i32, size, k));
            if (img.cube) comps[2] = c_one_u;  // a cube reports one face deep, not its layer count
            int out = 0;
            for (int k = 0; k < 4; ++k) {
                if ((in.dmask >> k) & 1) write_v(in.vdata + out++, comps[k]);
            }
            return;
        }
        if (op == 96) {  // image_get_lod
            std::uint32_t dim; bool arrayed;
            const ImageVar img = image_for(in, false, false, dim, arrayed);
            m.capability(spv::CapImageQuery);
            const Id sampler = load_sampler(sampler_for(in));
            const Id image = load_image(img);
            const Id si = m.emit(spv::OpSampledImage, m.type_sampled_image(img.type), {image, sampler});
            const int ncoord = coord_count(dim, false);
            std::vector<Id> coords;
            for (int k = 0; k < ncoord; ++k) coords.push_back(f(ld(vgpr_var(in.vaddr + k))));
            const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2f : t_v3f, coords);
            const Id lod = m.emit(spv::OpImageQueryLod, t_v2f, {si, coord});
            int out = 0;
            for (int k = 0; k < 2; ++k) {
                if ((in.dmask >> k) & 1) write_v(in.vdata + out++, u(extract(t_f32, lod, k)));
            }
            return;
        }
        if (!(is_sample || is_gather || is_sample_cd)) {
            error(std::string("unsupported ") + mnemonic(in));
            return;
        }
        // Decode sample variant flags from the mnemonic bits.
        std::uint32_t v = is_sample ? op - 32 : is_gather ? op - 64 : op - 104;
        bool has_o = false, has_c = false, has_cl = false, has_b = false, has_l = false, has_d = false, has_lz = false, has_cd = false;
        if (is_sample || is_gather) {
            has_o = (v & 16) != 0;
            has_c = (v & 8) != 0;
            const std::uint32_t low = v & 7;  // 0 base,1 cl,2 d,3 d_cl,4 l,5 b,6 b_cl,7 lz
            has_cl = low == 1 || low == 3 || low == 6;
            has_d = low == 2 || low == 3;
            has_l = low == 4;
            has_b = low == 5 || low == 6;
            has_lz = low == 7;
        } else {
            has_cd = true;
            has_o = (v & 4) != 0;
            has_c = (v & 2) != 0;
            has_cl = (v & 1) != 0;
        }
        std::uint32_t dim; bool arrayed;
        const ImageVar img = image_for(in, false, has_c, dim, arrayed);
        const std::size_t sampler_idx = sampler_for(in, has_c);
        const bool unnormalized = in.unorm || (sampler_idx < opt.sampler_force_unnormalized.size() &&
                                               opt.sampler_force_unnormalized[sampler_idx]);
        const Id sampler = load_sampler(sampler_idx);
        const Id image = load_image(img);
        const Id si = m.emit(spv::OpSampledImage, m.type_sampled_image(img.type), {image, sampler});
        int va = in.vaddr;
        Id offset = 0;
        std::vector<Id> offs;
        if (has_o) {
            const Id packed = ld(vgpr_var(va++));
            const int nc = coord_count(dim, false);
            for (int k = 0; k < nc; ++k) offs.push_back(i(ui(m.emit(spv::OpBitFieldSExtract, t_i32, {i(packed), cu(8 * k), cu(6)}))));
            offset = nc == 1 ? offs[0] : m.emit(spv::OpCompositeConstruct, nc == 2 ? t_v2i : t_v3i, offs);
        }
        Id bias = 0;
        if (has_b) bias = f(ld(vgpr_var(va++)));
        Id dref = 0;
        if (has_c) dref = f(ld(vgpr_var(va++)));
        // GCN places explicit derivatives BEFORE the spatial coordinates.
        std::vector<Id> gx, gy;
        Id dx = 0, dy = 0;
        if (has_d || has_cd) {
            const int ng = coord_count(dim, false);
            for (int k = 0; k < ng; ++k) gx.push_back(f(ld(vgpr_var(va++))));
            for (int k = 0; k < ng; ++k) gy.push_back(f(ld(vgpr_var(va++))));
        }
        const int ncoord = coord_count(dim, arrayed);
        std::vector<Id> coords;
        for (int k = 0; k < ncoord; ++k) coords.push_back(f(ld(vgpr_var(va++))));
        if (img.cube && !is_gather) {
            // Cube samples: s and t lie in [1, 2] on the face (the shader adds
            // 1.5 to sc/cubema and tc/cubema), and a cube array adds 8 per cube
            // to the face id. The layer is face - 2 * floor(face / 8).
            coords[0] = fsub(coords[0], cf(1.0f));
            coords[1] = fsub(coords[1], cf(1.0f));
            coords[2] = fma_(m.ext_inst(t_f32, spv::GlslFloor, {fdiv(coords[2], cf(8.0f))}), cf(-2.0f), coords[2]);
        }
        // UNORM means UNnormalized: a set bit requests texel coordinates.
        // Normalized coordinates may legitimately be negative or exceed one;
        // wrapping/clamping belongs to the sampler, never a fract() heuristic.
        // The sampler can independently force texel coordinates (S# bit 15).
        if (unnormalized) ++res.unnormalized_samples;
        if (unnormalized && !img.cube) {
            m.capability(spv::CapImageQuery);
            const int ns = coord_count(dim, false);
            const int nq = coord_count(dim, arrayed);
            const Id size_type = nq == 1 ? t_i32 : nq == 2 ? t_v2i : t_v3i;
            const Id size = m.emit(spv::OpImageQuerySizeLod, size_type, {image, i(c_zero_u)});
            for (int k = 0; k < ns; ++k) {
                const Id extent = nq == 1 ? size : extract(t_i32, size, k);
                const Id scale = fdiv(cf(1.0f), i2f(ui(extent)));
                coords[k] = fmul(coords[k], scale);
                if (!gx.empty()) {
                    gx[k] = fmul(gx[k], scale);
                    gy[k] = fmul(gy[k], scale);
                }
            }
        }
        if (offset && !is_gather && !img.cube && !runtime_sample_offsets()) {
            // No run-time Offset on a sample without maintenance8
            // (VUID-RuntimeSpirv-Offset-10213; AMD's RX 580 has none): the
            // coordinates move by the offset in level 0's texels instead -
            // the texel image_sample_lz_o reads, the game's only offset sample.
            m.capability(spv::CapImageQuery);
            const int nq = coord_count(dim, arrayed);
            const Id size = m.emit(spv::OpImageQuerySizeLod, nq == 1 ? t_i32 : nq == 2 ? t_v2i : t_v3i, {image, i(c_zero_u)});
            for (std::size_t k = 0; k < offs.size(); ++k) {
                const Id extent = nq == 1 ? size : extract(t_i32, size, static_cast<std::uint32_t>(k));
                coords[k] = fma_(i2f(ui(offs[k])), fdiv(cf(1.0f), i2f(ui(extent))), coords[k]);
            }
            offset = 0;
        }
        const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct,
            ncoord == 2 ? t_v2f : ncoord == 3 ? t_v3f : t_v4f, coords);
        if (!gx.empty()) {
            const int ng = static_cast<int>(gx.size());
            dx = ng == 1 ? gx[0] : m.emit(spv::OpCompositeConstruct, ng == 2 ? t_v2f : t_v3f, gx);
            dy = ng == 1 ? gy[0] : m.emit(spv::OpCompositeConstruct, ng == 2 ? t_v2f : t_v3f, gy);
        }
        Id lod = 0;
        if (has_l) lod = f(ld(vgpr_var(va++)));
        if (has_lz) lod = cf(0.0f);
        if (has_cl) va++;  // clamp value: ignored
        std::uint32_t mask = 0;
        std::vector<std::uint32_t> extra;
        if (bias) { mask |= spv::IoBias; extra.push_back(bias); }
        if (lod) { mask |= spv::IoLod; extra.push_back(lod); }
        if (dx) { mask |= spv::IoGrad; extra.push_back(dx); extra.push_back(dy); }
        if (offset) { mask |= spv::IoOffset; extra.push_back(offset); m.capability(spv::CapImageGatherExtended); }
        if (opt.stage != Stage::Pixel && !lod && !dx) {
            lod = cf(0.0f);
            mask |= spv::IoLod;
            extra.push_back(lod);
        }
        const bool explicit_lod = lod || dx;
        Id texel;
        if (is_gather) {
            // A gather reads the base level: it takes an offset but no LOD, bias
            // or gradients (NVIDIA's compiler corrupts its heap on a gather with
            // a Lod operand, PS 6c5edb96), and Vulkan wants its component a
            // constant. LOD zero is what it samples anyway.
            if (has_l || bias || dx) {
                error("gather with an explicit LOD, bias or gradients");
                return;
            }
            std::vector<std::uint32_t> ops = {si, coord};
            if (has_c) ops.push_back(dref); else ops.push_back(m.const_i32(static_cast<std::int32_t>(gather_component(in.dmask))));
            if (offset) {
                ops.push_back(spv::IoOffset);
                ops.push_back(offset);
            }
            texel = has_c ? m.emit(spv::OpImageDrefGather, t_v4f, ops)
                          : texel_as_v4f(img, m.emit(spv::OpImageGather, texel_type(img), ops));
            write_texel_all(in, texel);
            return;
        }
        if (has_c) {
            std::vector<std::uint32_t> ops = {si, coord, dref};
            if (mask) { ops.push_back(mask); ops.insert(ops.end(), extra.begin(), extra.end()); }
            const Id r = m.emit(explicit_lod ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleDrefImplicitLod, t_f32, ops);
            texel = vec4f(r, r, r, r);
        } else {
            std::vector<std::uint32_t> ops = {si, coord};
            if (mask) { ops.push_back(mask); ops.insert(ops.end(), extra.begin(), extra.end()); }
            texel = texel_as_v4f(img, m.emit(explicit_lod ? spv::OpImageSampleExplicitLod : spv::OpImageSampleImplicitLod,
                                             texel_type(img), ops));
        }
        write_texel(in, texel);
    }
    static int popcount4(std::uint32_t m4) { return ((m4 >> 0) & 1) + ((m4 >> 1) & 1) + ((m4 >> 2) & 1) + ((m4 >> 3) & 1); }
    static std::uint32_t gather_component(std::uint32_t dmask) {
        for (std::uint32_t k = 0; k < 4; ++k) if ((dmask >> k) & 1) return k;
        return 0;
    }
    static int coord_count(std::uint32_t dim, bool arrayed) {
        int n = dim == spv::Dim1D ? 1 : dim == spv::Dim2D ? 2 : 3;
        if (arrayed) n++;
        return n;
    }
    void note_debug_texel(Id texel) {
        if (opt.debug_ps != 2) return;
        if (!v_debug_texel) {
            const Id p_fn_v4f = m.type_pointer(spv::ScFunction, t_v4f);
            v_debug_texel = m.local_variable(p_fn_v4f, m.const_composite(t_v4f, {cf(1.0f), cf(0.0f), cf(1.0f), cf(1.0f)}));
        }
        m.store(v_debug_texel, texel);
    }
    void write_texel(const Inst& in, Id texel) {
        note_debug_texel(texel);
        int out = 0;
        for (int k = 0; k < 4; ++k) {
            if ((in.dmask >> k) & 1) write_v(in.vdata + out++, u(extract(t_f32, texel, k)));
        }
    }
    void write_texel_all(const Inst& in, Id texel) {
        note_debug_texel(texel);
        for (int k = 0; k < 4; ++k) write_v(in.vdata + k, u(extract(t_f32, texel, k)));
    }

    // ---- interpolation and export
    void vintrp(const Inst& in) {
        const Id var = in_param(in.attr);
        const Id v = m.load(t_v4f, var);
        write_v(in.dst, u(extract(t_f32, v, in.attr_chan)));
    }
    void exp(const Inst& in) {
        std::vector<Id> comps(4, cf(0.0f));
        if (in.compr) {
            for (int pair = 0; pair < 2; ++pair) {
                if (!((in.dmask >> (pair * 2)) & 3)) continue;
                if (const auto pk = packed_pairs.find(in.vsrc[pair]); pk != packed_pairs.end() && pk->second.block == m.blocks() && pk->second.exec) {
                    // Packed under a varying EXEC: rounded where it was written, the old word decoded elsewhere.
                    const Id packed = ld(vgpr_var(in.vsrc[pair]));
                    Id r0, r1, d0, d1;
                    if (native_rtz) {
                        const Id r = native_rtz_pair(m.emit(spv::OpCompositeConstruct, t_v2f, {pk->second.lo, pk->second.hi}));
                        r0 = extract(t_f32, r, 0);
                        r1 = extract(t_f32, r, 1);
                        const Id d = m.emit(spv::OpFConvert, t_v2f, {m.emit(spv::OpBitcast, t_v2h, {packed})});
                        d0 = extract(t_f32, d, 0);
                        d1 = extract(t_f32, d, 1);
                    } else {
                        r0 = emit_rtz_half(m, half_types(), pk->second.lo);
                        r1 = emit_rtz_half(m, half_types(), pk->second.hi);
                        d0 = emit_unpack_half16(m, half_types(), iand(packed, cu(0xffff)));
                        d1 = emit_unpack_half16(m, half_types(), shr(packed, cu(16)));
                    }
                    comps[pair * 2] = m.emit(spv::OpSelect, t_f32, {pk->second.exec, r0, d0});
                    comps[pair * 2 + 1] = m.emit(spv::OpSelect, t_f32, {pk->second.exec, r1, d1});
                    continue;
                }
                if (const auto pk = packed_pairs.find(in.vsrc[pair]); pk != packed_pairs.end() && pk->second.block == m.blocks()) {
                    if (native_rtz) {
                        const Id r = native_rtz_pair(m.emit(spv::OpCompositeConstruct, t_v2f, {pk->second.lo, pk->second.hi}));
                        comps[pair * 2] = extract(t_f32, r, 0);
                        comps[pair * 2 + 1] = extract(t_f32, r, 1);
                    } else {
                        comps[pair * 2] = emit_rtz_half(m, half_types(), pk->second.lo);
                        comps[pair * 2 + 1] = emit_rtz_half(m, half_types(), pk->second.hi);
                    }
                    continue;
                }
                const Id packed = ld(vgpr_var(in.vsrc[pair]));
                if (native_rtz) {  // f16 to f32 is exact
                    const Id r = m.emit(spv::OpFConvert, t_v2f, {m.emit(spv::OpBitcast, t_v2h, {packed})});
                    comps[pair * 2] = extract(t_f32, r, 0);
                    comps[pair * 2 + 1] = extract(t_f32, r, 1);
                    continue;
                }
                // Exact decoding, which a driver cannot fold into the pack (gcn/half.h).
                comps[pair * 2] = emit_unpack_half16(m, half_types(), iand(packed, cu(0xffff)));
                comps[pair * 2 + 1] = emit_unpack_half16(m, half_types(), shr(packed, cu(16)));
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                if ((in.dmask >> k) & 1) comps[k] = f(ld(vgpr_var(in.vsrc[k])));
            }
        }
        if (in.tgt < 8) {  // mrt
            if (opt.stage != Stage::Pixel) { error("mrt export outside pixel shader"); return; }
            const Id var = out_param(in.tgt);
            if (opt.debug_ps == 1) {
                m.store(var, vec4f(cf(1.0f), cf(0.0f), cf(1.0f), cf(1.0f)));
                return;
            }
            if (opt.debug_ps == 2) {
                m.store(var, v_debug_texel ? m.load(t_v4f, v_debug_texel) : vec4f(cf(1.0f), cf(0.0f), cf(1.0f), cf(1.0f)));
                return;
            }
            if (opt.debug_ps == 3) {
                const Id uv = m.load(t_v4f, in_param(1));
                m.store(var, vec4f(extract(t_f32, uv, 0), extract(t_f32, uv, 1), cf(0.0f), cf(1.0f)));
                return;
            }
            if (in.dmask == 0xf) {
                m.store(var, vec4f(comps[0], comps[1], comps[2], comps[3]));
            } else {
                Id cur = m.load(t_v4f, var);
                for (int k = 0; k < 4; ++k) {
                    if ((in.dmask >> k) & 1) cur = m.emit(spv::OpCompositeInsert, t_v4f, {comps[k], cur, static_cast<std::uint32_t>(k)});
                }
                m.store(var, cur);
            }
        } else if (in.tgt == 8) {  // mrtz
            if (!out_frag_depth) {
                out_frag_depth = m.global_variable(p_out_f32, spv::ScOutput);
                m.decorate(out_frag_depth, spv::DecBuiltIn, {spv::BiFragDepth});
                interface.push_back(out_frag_depth);
                m.execution_mode(fn_main, spv::ExDepthReplacing);
            }
            if (in.dmask & 1) m.store(out_frag_depth, comps[0]);
        } else if (in.tgt == 9) {
            // null export
        } else if (in.tgt >= 12 && in.tgt < 16) {  // pos
            if (in.tgt == 12) {
                if (!out_position) {
                    out_position = m.global_variable(p_out_v4f, spv::ScOutput);
                    m.decorate(out_position, spv::DecBuiltIn, {spv::BiPosition});
                    if (opt.invariant_position) m.decorate(out_position, spv::DecInvariant);
                    interface.push_back(out_position);
                }
                Id pos = vec4f(comps[0], comps[1], comps[2], comps[3]);
                if (opt.tess_role == TranslateOptions::TessRole::Domain) {
                    // A particle the game leaves dead carries values that make
                    // no position: the console's clipper drops a primitive
                    // whose position is not a number (every comparison fails),
                    // where a PC GPU is free to draw it across the screen.
                    // Collapse those vertices onto one point instead, which is
                    // the same nothing, and keep the enormous ones out too.
                    // A particle the camera is inside reaches behind it: one
                    // corner with w at or below zero and the clipper stretches
                    // what is left across the frame. The game's own domain
                    // shader fades those out; until that is understood, drop
                    // the vertex (BBHOST_TESS_NEAR_CULL=0 keeps it).
                    Id bad = m.emit(spv::OpIsNan, t_bool, {comps[0]});
                    if (v_patch_culled) bad = m.emit(spv::OpLogicalOr, t_bool, {bad, m.load(t_bool, v_patch_culled)});
                    if (opt.tess_near_cull > 0.0f) {
                        bad = m.emit(spv::OpLogicalOr, t_bool,
                                     {bad, m.emit(spv::OpFOrdLessThan, t_bool, {comps[3], cf(opt.tess_near_cull)})});
                    }
                    for (int k = 1; k < 4; ++k) {
                        bad = m.emit(spv::OpLogicalOr, t_bool, {bad, m.emit(spv::OpIsNan, t_bool, {comps[k]})});
                    }
                    for (int k = 0; k < 4; ++k) {
                        const Id mag = fabs_(comps[k]);
                        bad = m.emit(spv::OpLogicalOr, t_bool, {bad, m.emit(spv::OpFOrdGreaterThan, t_bool, {mag, cf(1e9f)})});
                    }
                    pos = m.emit(spv::OpSelect, t_v4f, {bad, vec4f(cf(0.0f), cf(0.0f), cf(0.0f), cf(1.0f)), pos});
                }
                m.store(out_position, pos);
            } else {
                // PA_CL_VS_OUT_CNTL packs the extra vectors after pos0: the misc
                // vector (bit 20), then clip/cull distance vectors 0 (bit 22) and 1 (bit 23).
                const std::uint32_t cntl = opt.vs_out_cntl;
                int slot = 13;
                int misc_slot = -1, cc0_slot = -1, cc1_slot = -1;
                if (cntl & (1u << 20)) misc_slot = slot++;
                if (cntl & (1u << 22)) cc0_slot = slot++;
                if (cntl & (1u << 23)) cc1_slot = slot++;
                if (static_cast<int>(in.tgt) == misc_slot) {
                    if (cntl & (1u << 16)) {  // USE_VTX_POINT_SIZE
                        if (!out_point_size) {
                            out_point_size = m.global_variable(p_out_f32, spv::ScOutput);
                            m.decorate(out_point_size, spv::DecBuiltIn, {1 /* PointSize */});
                            interface.push_back(out_point_size);
                        }
                        m.store(out_point_size, comps[0]);
                    }
                    if (cntl & ((1u << 17) | (1u << 18) | (1u << 19) | (1u << 21))) {
                        error("misc vector export with edge flag / RT index / viewport / kill unsupported");
                    }
                } else if (static_cast<int>(in.tgt) == cc0_slot || static_cast<int>(in.tgt) == cc1_slot) {
                    if (cntl & TranslateOptions::kVsOutCntlClipVarying) {
                        if (static_cast<int>(in.tgt) != cc0_slot) return;  // the host sends only distances 0-3 this way
                        std::vector<Id> d(4, cf(1.0f));
                        for (int k = 0; k < 4; ++k) {
                            if (((in.dmask >> k) & 1) && ((cntl >> k) & 1)) d[k] = comps[k];
                        }
                        m.store(out_param(static_cast<int>(TranslateOptions::vs_clip_location(cntl))), vec4f(d[0], d[1], d[2], d[3]));
                        return;
                    }
                    const int base = static_cast<int>(in.tgt) == cc0_slot ? 0 : 4;
                    if (!out_clip) {
                        // Clip distances 0..7 enabled by bits 0..7 (cull bits 8..15 are folded in).
                        const std::uint32_t ena = (cntl & 0xff) | ((cntl >> 8) & 0xff);
                        clip_count = 0;
                        for (int k = 0; k < 8; ++k) if ((ena >> k) & 1) clip_count = k + 1;
                        if (clip_count == 0) clip_count = 1;
                        const Id t_arr = m.type_array(t_f32, cu(static_cast<std::uint32_t>(clip_count)));
                        const Id p_arr = m.type_pointer(spv::ScOutput, t_arr);
                        out_clip = m.global_variable(p_arr, spv::ScOutput);
                        m.decorate(out_clip, spv::DecBuiltIn, {3 /* ClipDistance */});
                        m.capability(spv::CapClipDistance);
                        interface.push_back(out_clip);
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (!((in.dmask >> k) & 1) || base + k >= clip_count) continue;
                        const Id p = m.access_chain(p_out_f32, out_clip, {cu(static_cast<std::uint32_t>(base + k))});
                        m.store(p, comps[k]);
                    }
                } else {
                    error("pos" + std::to_string(in.tgt - 12) + " export not enabled by PA_CL_VS_OUT_CNTL");
                }
            }
        } else if (in.tgt >= 32 && in.tgt < 64) {  // param
            const Id value = vec4f(comps[0], comps[1], comps[2], comps[3]);
            for (const Id var : param_outputs(static_cast<int>(in.tgt) - 32)) m.store(var, value);
        } else {
            error("export target " + std::to_string(in.tgt));
        }
    }

    // ---- VALU dispatch
    void valu(const Inst& in) {
        Mods md;
        if (in.enc == Enc::VOP2) {
            Id s0 = read_s(in.src0, in);
            Id s1;
            if (in.op == 1 || in.op == 2) {
                // readlane/writelane: the second operand is a scalar code
                s1 = read_s(in.src1 - 256, in);
                if (in.op == 2) {
                    // v_writelane: src0 is scalar too (already read as generic)
                }
            } else {
                s1 = read_s(in.src1, in);
            }
            if (!vop2(in.op, in, s0, s1, md, 0, 0, kVccLo)) error(std::string("unsupported ") + mnemonic(in));
            return;
        }
        if (in.enc == Enc::VOP1) {
            if (!vop1(in.op, in, read_s(in.src0, in), md)) error(std::string("unsupported ") + mnemonic(in));
            return;
        }
        if (in.enc == Enc::VOPC) {
            vcmp(in.op, in, read_s(in.src0, in), read_s(in.src1, in), md, kVccLo);
            return;
        }
        // VOP3
        md.abs = in.abs;
        md.neg = in.neg;
        md.omod = in.omod;
        md.clamp = in.clamp;
        if (in.op < 0x100) {
            md.abs = in.abs;
            vcmp(in.op, in, read_s(in.src0, in), read_s(in.src1, in), md, in.sdst);
            return;
        }
        if (in.op < 0x140) {
            const std::uint32_t op = in.op - 0x100;
            const Id s0 = read_s(in.src0, in);
            const Id s1 = read_s(in.src1, in);
            Id lo = 0, hi = 0;
            if (op == 0 || op == 40 || op == 41 || op == 42) read_pair(in.src2, in, lo, hi);
            if (op == 31 || op == 6) {
                // v_mac in VOP3 reads vdst as src2
            }
            if (!vop2(op, in, s0, s1, md, lo, hi, in.sdst)) error(std::string("unsupported ") + mnemonic(in));
            return;
        }
        if (in.op < 0x180) {
            if (!vop3_only(in.op, in, read_s(in.src0, in), read_s(in.src1, in), read_s(in.src2, in), md)) {
                error(std::string("unsupported ") + mnemonic(in));
            }
            return;
        }
        if (!vop1(in.op - 0x180, in, read_s(in.src0, in), md)) error(std::string("unsupported ") + mnemonic(in));
    }

    // ---- per-instruction dispatch
    void translate_inst(const Inst& in, std::uint32_t bias) {
        cur_offset = bias + in.offset;
        switch (in.enc) {
        case Enc::SOP2: sop2(in); break;
        case Enc::SOPK: sopk(in); break;
        case Enc::SOP1: sop1(in); break;
        case Enc::SOPC: sopc(in); break;
        case Enc::SOPP: sopp(in, bias); break;
        case Enc::SMRD: smrd(in); break;
        case Enc::VOP2: case Enc::VOP1: case Enc::VOPC: case Enc::VOP3: valu(in); break;
        case Enc::VINTRP: vintrp(in); break;
        case Enc::DS: ds(in); break;
        case Enc::MUBUF: mubuf(in); break;
        case Enc::MTBUF: mtbuf(in); break;
        case Enc::MIMG: mimg(in); break;
        case Enc::EXP: exp(in); break;
        default: error("unknown encoding"); break;
        }
    }

    // Emit the module.
    void run() {
        m.capability(spv::CapShader);
        m.capability(spv::CapInt64);
        m.capability(spv::CapGroupNonUniform);
        m.capability(spv::CapGroupNonUniformBallot);
        m.capability(spv::CapGroupNonUniformShuffle);
        m.capability(spv::CapPhysicalStorageBufferAddresses);
        m.extension("SPV_KHR_physical_storage_buffer");
        m.memory_model(5348 /* PhysicalStorageBuffer64 */, 1 /* GLSL450 */);
        setup_types();
        if (spill_cells_on()) spill = spill_cells(prog);
        native_rtz = native_half_rtz() && program_allows_native_half_rtz(prog);
        if (native_rtz) {
            m.capability(spv::CapFloat16);
            m.capability(spv::CapDenormPreserve);
            m.capability(spv::CapRoundingModeRTZ);
            t_v2h = m.type_vector(m.type_float(16), 2);
        }

        // Uniform block (std140), StageParams: { u64 l1_table; u64 reserved;
        // uvec4 user[4]; uint cb_valid; uvec4 cb_bias_dw[4]; uvec4 cb_stride[4];
        // uvec4 cb_w3[4]; uvec4 vertex_formats[4], declared only for vertex_formats_from_params }
        const Id t_user_arr = m.type_array(t_v4u, cu(4));
        m.decorate(t_user_arr, spv::DecArrayStride, {16});
        const Id t_bias_arr = t_user_arr;  // the same uvec4[4] type (types are deduplicated): decorated once
        const bool vertex_w3 = (opt.stage == Stage::Vertex && opt.vertex_formats_from_params && !opt.vertex_input.empty()) ||
                               opt.tess_role == TranslateOptions::TessRole::LsCompute ||
                               opt.tess_role == TranslateOptions::TessRole::LsVertex ||  // the control points (kTess*)
                               (opt.tess_role == TranslateOptions::TessRole::HullTcs && opt.tess_window);  // its user data
        // The domain role also reads what it takes to leave out a patch whose
        // centre is at or behind the camera plane (StageParams::patch_cull_w).
        const bool patch_cull = opt.tess_role == TranslateOptions::TessRole::Domain && opt.tess_patch_cull;
        std::vector<Id> members = patch_cull ? std::vector<Id>{t_u64, t_u64, t_user_arr, t_u32, t_bias_arr, t_bias_arr, t_bias_arr,
                                                               t_bias_arr, t_v4f, t_v4u}
                                  : vertex_w3 ? std::vector<Id>{t_u64, t_u64, t_user_arr, t_u32, t_bias_arr, t_bias_arr, t_bias_arr, t_bias_arr}
                                              : std::vector<Id>{t_u64, t_u64, t_user_arr, t_u32, t_bias_arr, t_bias_arr, t_bias_arr};
        // A tessellation stage's LDS buffer has an end (StageParams::lds_bytes,
        // the dword after cb_valid): member 3 becomes the uvec4 of both, the
        // same 16 bytes at the same offset.
        params_head_v4 = opt.tess_role != TranslateOptions::TessRole::None && opt.tess_lds_bound;
        if (params_head_v4) members[3] = t_v4u;
        // TranslateOptions::bindless: StageParams::image_index and
        // sampler_index follow whatever the stage declares, at their own offsets.
        if (opt.bindless) {
            bindless_image_member = static_cast<std::uint32_t>(members.size());
            bindless_sampler_member = bindless_image_member + 1;
            members.push_back(t_bias_arr);
            members.push_back(t_bias_arr);
        }
        const Id t_ubo = m.type_struct(members);
        m.decorate(t_ubo, spv::DecBlock);
        m.member_decorate(t_ubo, 0, spv::DecOffset, {0});
        m.member_decorate(t_ubo, 1, spv::DecOffset, {8});
        m.member_decorate(t_ubo, 2, spv::DecOffset, {16});
        m.member_decorate(t_ubo, 3, spv::DecOffset, {80});
        m.member_decorate(t_ubo, 4, spv::DecOffset, {96});
        m.member_decorate(t_ubo, 5, spv::DecOffset, {160});
        m.member_decorate(t_ubo, 6, spv::DecOffset, {224});
        if (vertex_w3 || patch_cull) m.member_decorate(t_ubo, 7, spv::DecOffset, {288});
        if (patch_cull) {
            m.member_decorate(t_ubo, 8, spv::DecOffset, {352});
            m.member_decorate(t_ubo, 9, spv::DecOffset, {368});
        }
        if (opt.bindless) {
            m.member_decorate(t_ubo, bindless_image_member, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, image_index))});
            m.member_decorate(t_ubo, bindless_sampler_member, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, sampler_index))});
        }
        const Id p_ubo = m.type_pointer(spv::ScUniform, t_ubo);
        ubo_var = m.global_variable(p_ubo, spv::ScUniform);
        m.decorate(ubo_var, spv::DecDescriptorSet, {opt.descriptor_set});
        m.decorate(ubo_var, spv::DecBinding, {kBindingParams});
        m.name(ubo_var, "params");
        interface.push_back(ubo_var);
        const Id p_uni_u64 = m.type_pointer(spv::ScUniform, t_u64);
        p_uni_u32 = m.type_pointer(spv::ScUniform, t_u32);

        // LDS
        const std::uint32_t lds_words = lds_words_for(opt);
        const Id c_lds = cu(lds_words);
        t_lds_array = m.type_array(t_u32, c_lds);
        p_wg_lds = m.type_pointer(spv::ScWorkgroup, t_lds_array);
        p_wg_u32 = m.type_pointer(spv::ScWorkgroup, t_u32);
        // ds_swizzle_b32 is a lane shuffle that borrows the LDS hardware but
        // no LDS storage, and pixel shaders use it for cross-lane work. Only
        // the instructions that address LDS need the Workgroup variable, and
        // only those are a reason to reject a non-compute stage.
        bool uses_lds = false;
        const Inst* lds_inst = nullptr;
        for (const Inst& in : prog.insts) {
            if (in.enc != Enc::DS || in.op == 53) continue;
            uses_lds = true;
            if (!lds_inst) lds_inst = &in;
        }
        lds_in_attributes = opt.tess_lds_attributes && (opt.tess_role == TranslateOptions::TessRole::LsVertex ||
                                                        opt.tess_role == TranslateOptions::TessRole::DomainTes);
        if (uses_lds && lds_in_attributes) {
            uses_lds = false;  // the control point travels as stage attributes
        } else if (uses_lds && opt.tess_role != TranslateOptions::TessRole::None) {
            lds_in_buffer = true;  // StageParams::lds_address, loaded at the entry
            uses_lds = false;
        }
        if (lds_in_attributes) {
            const std::uint32_t n = std::max<std::uint32_t>(1, opt.tess_attr_vec4s);
            const Id t_attr = m.type_array(t_v4u, cu(n));
            if (opt.tess_role == TranslateOptions::TessRole::LsVertex) {
                v_attr_out = m.global_variable(m.type_pointer(spv::ScOutput, t_attr), spv::ScOutput);
                m.decorate(v_attr_out, spv::DecLocation, {0});
                m.name(v_attr_out, "control_point");
                interface.push_back(v_attr_out);
            } else {
                // A tessellation stage's per-vertex input is an array over the
                // patch's control points; there is one.
                const Id t_in = m.type_array(t_attr, cu(std::max<std::uint32_t>(1, opt.tess_patch_control_points)));
                v_attr_in = m.global_variable(m.type_pointer(spv::ScInput, t_in), spv::ScInput);
                m.decorate(v_attr_in, spv::DecLocation, {0});
                m.name(v_attr_in, "control_point");
                interface.push_back(v_attr_in);
            }
        }
        if (uses_lds && opt.stage != Stage::Compute) {
            cur_offset = lds_inst ? lds_inst->offset : 0;
            error(std::string("LDS use outside a compute shader (") + mnemonic(*lds_inst) +
                  "; tessellation/geometry stages are not supported yet)");
            uses_lds = false;
        }
        if (uses_lds) {
            v_lds = m.global_variable(p_wg_lds, spv::ScWorkgroup);
            m.name(v_lds, "lds");
            interface.push_back(v_lds);
            res.lds_bytes = lds_words * 4;
        }

        // Built-in inputs
        in_subgroup_lane = builtin_in(p_in_u32, spv::BiSubgroupLocalInvocationId, "lane_id");
        if (opt.stage == Stage::Vertex) {
            in_vertex_index = builtin_in(p_in_u32, spv::BiVertexIndex, "vertex_index");
            in_instance_index = builtin_in(p_in_u32, spv::BiInstanceIndex, "instance_index");
        } else if (opt.stage == Stage::Pixel) {
            in_frag_coord = builtin_in(p_in_v4f, spv::BiFragCoord, "frag_coord");
            in_front_facing = builtin_in(p_in_bool, spv::BiFrontFacing, "front_facing");
        } else if (opt.stage == Stage::TessControl) {
            in_invocation_id = builtin_in(p_in_u32, spv::BiInvocationId, "invocation_id");
            in_primitive_id = builtin_in(p_in_u32, spv::BiPrimitiveId, "primitive_id");
            // gl_TessLevelOuter[4] and gl_TessLevelInner[2], which is where
            // the hull shader's factor stores land.
            const Id t_outer = m.type_array(t_f32, cu(4)), t_inner = m.type_array(t_f32, cu(2));
            out_tess_outer = m.global_variable(m.type_pointer(spv::ScOutput, t_outer), spv::ScOutput);
            out_tess_inner = m.global_variable(m.type_pointer(spv::ScOutput, t_inner), spv::ScOutput);
            m.decorate(out_tess_outer, spv::DecBuiltIn, {spv::BiTessLevelOuter});
            m.decorate(out_tess_inner, spv::DecBuiltIn, {spv::BiTessLevelInner});
            // The levels belong to the patch, not to a control point.
            m.decorate(out_tess_outer, spv::DecPatch);
            m.decorate(out_tess_inner, spv::DecPatch);
            m.name(out_tess_outer, "tess_level_outer");
            m.name(out_tess_inner, "tess_level_inner");
            interface.push_back(out_tess_outer);
            interface.push_back(out_tess_inner);
        } else if (opt.stage == Stage::TessEval) {
            in_tess_coord = builtin_in(m.type_pointer(spv::ScInput, t_v3f), spv::BiTessCoord, "tess_coord");
            in_primitive_id = builtin_in(p_in_u32, spv::BiPrimitiveId, "primitive_id");
        } else {
            in_local_id = builtin_in(p_in_v3u, spv::BiLocalInvocationId, "local_id");
            in_workgroup_id = builtin_in(p_in_v3u, spv::BiWorkgroupId, "workgroup_id");
        }

        // Helper: xlate(va) walks the two-level page table.
        {
            const Id fn_type = m.type_function(t_u64, {t_u64});
            fn_xlate = m.begin_function(t_u64, fn_type);
            const Id param = m.emit(spv::OpFunctionParameter, t_u64, {});
            m.label();
            const Id l1_ptr = m.access_chain(p_uni_u64, ubo_var, {c_zero_u});
            const Id l1 = m.load(t_u64, l1_ptr);
            const Id i1 = m.emit(spv::OpShiftRightLogical, t_u64, {param, m.const_u64(32)});
            const Id i1c = m.ext_inst(t_u64, spv::GlslUMin, {i1, m.const_u64(kL1Entries - 1)});
            const Id l1_addr = m.emit(spv::OpIAdd, t_u64, {l1, m.emit(spv::OpShiftLeftLogical, t_u64, {i1c, m.const_u64(3)})});
            const Id l1_block = m.emit(spv::OpConvertUToPtr, p_psb_block64, {l1_addr});
            const Id l2 = m.emit(spv::OpLoad, t_u64, {m.access_chain(p_psb_u64, l1_block, {c_zero_u}), 2u, 8u});
            const Id i2 = m.emit(spv::OpBitwiseAnd, t_u64, {m.emit(spv::OpShiftRightLogical, t_u64, {param, m.const_u64(kPageShift)}),
                                                          m.const_u64(kL2Entries - 1)});
            const Id l2_addr = m.emit(spv::OpIAdd, t_u64, {l2, m.emit(spv::OpShiftLeftLogical, t_u64, {i2, m.const_u64(3)})});
            const Id l2_block = m.emit(spv::OpConvertUToPtr, p_psb_block64, {l2_addr});
            const Id page = m.emit(spv::OpLoad, t_u64, {m.access_chain(p_psb_u64, l2_block, {c_zero_u}), 2u, 8u});
            const Id within = m.emit(spv::OpBitwiseAnd, t_u64, {param, m.const_u64((1ull << kPageShift) - 1)});
            const Id host = m.emit(spv::OpIAdd, t_u64, {page, within});
            m.emit_void(spv::OpReturnValue, {host});
            m.end_function();
            m.name(fn_xlate, "xlate");
        }
        // Helper: wqm(u32): whole quad mode on a 32-bit mask
        {
            const Id fn_type = m.type_function(t_u32, {t_u32});
            fn_wqm = m.begin_function(t_u32, fn_type);
            const Id param = m.emit(spv::OpFunctionParameter, t_u32, {});
            m.label();
            Id t = ior(param, shr(param, c_one_u));
            t = ior(t, shr(t, cu(2)));
            t = iand(t, cu(0x11111111u));
            const Id r = imul(t, cu(0xf));
            m.emit_void(spv::OpReturnValue, {r});
            m.end_function();
            m.name(fn_wqm, "wqm");
        }

        // Main
        fn_main = m.begin_function(t_void, t_fn_main);
        m.name(fn_main, "main");
        m.label();
        // Registers
        v_vcc_lo = m.local_variable(p_fn_u32, c_zero_u);
        v_vcc_hi = m.local_variable(p_fn_u32, c_zero_u);
        v_exec_lo = m.local_variable(p_fn_u32, c_zero_u);
        v_exec_hi = m.local_variable(p_fn_u32, c_zero_u);
        v_m0 = m.local_variable(p_fn_u32, c_zero_u);
        v_scc = m.local_variable(p_fn_bool, m.const_bool(false));
        v_exec_bit = m.local_variable(p_fn_bool, m.const_bool(true));
        v_vcc_bit = m.local_variable(p_fn_bool, m.const_bool(false));
        v_pc = m.local_variable(p_fn_u32, c_zero_u);
        v_ret = m.local_variable(p_fn_u32, c_zero_u);
        v_lane = m.local_variable(p_fn_u32, c_zero_u);
        m.name(v_vcc_lo, "vcc_lo"); m.name(v_vcc_hi, "vcc_hi"); m.name(v_exec_lo, "exec_lo"); m.name(v_exec_hi, "exec_hi");
        m.name(v_m0, "m0"); m.name(v_scc, "scc"); m.name(v_pc, "pc");
        if (opt.stage == Stage::Pixel && opt.ps_clip_discard) {
            // TranslateOptions::kVsOutCntlClipVarying: the clipper's work, per pixel.
            const Id var = m.global_variable(p_in_v4f, spv::ScInput);
            m.decorate(var, spv::DecLocation, {opt.ps_clip_location()});
            m.name(var, "clip_distances");
            interface.push_back(var);
            const Id d = m.load(t_v4f, var);
            Id out = m.const_bool(false);
            for (std::uint32_t k = 0; k < std::min<std::uint32_t>(opt.ps_clip_count(), 4); ++k) {
                out = m.emit(spv::OpLogicalOr, t_bool, {out, m.emit(spv::OpFOrdLessThan, t_bool, {extract(t_f32, d, k), cf(0.0f)})});
            }
            const Id l_kill = m.fresh(), l_on = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {l_on, 0u});
            m.emit_void(spv::OpBranchConditional, {out, l_kill, l_on});
            m.label(l_kill);
            m.emit_void(spv::OpKill, {});
            m.label(l_on);
        }
        m.store(v_lane, m.load(t_u32, in_subgroup_lane));
        {
            Id lo, hi;
            ballot(m.const_bool(true), lo, hi);
            m.store(v_exec_lo, lo);
            m.store(v_exec_hi, hi);
            refresh_exec_bit();
            lane_set.insert(kExecLo);  // every running lane starts with its bit set
        }
        // User SGPRs
        const int nuser = std::min(user_sgpr_count(), 16);
        // Both stages of a tessellated draw read this one set, and their user
        // data differs. The domain shader keeps the usual member - it is the
        // one whose resources the host resolves - and the LS takes the slot a
        // vertex stage would hold its vertex formats in, which a patch draw
        // has none of.
        const std::uint32_t user_member = opt.tess_role == TranslateOptions::TessRole::LsVertex ||
                                                  (opt.tess_role == TranslateOptions::TessRole::HullTcs && opt.tess_window)
                                              ? 7u
                                              : 2u;
        for (int k = 0; k < nuser; ++k) {
            const Id p = m.access_chain(p_uni_u32, ubo_var, {cu(user_member), cu(k / 4), cu(k % 4)});
            m.store(sgpr_var(k), m.load(t_u32, p));
        }
        sym_user_data();
        Id lds_bytes = 0;
        if (lds_in_buffer) {
            lds_base64 = m.load(t_u64, m.access_chain(p_uni_u64, ubo_var, {cu(1)}));  // StageParams::lds_address
            if (params_head_v4) {
                // TranslateOptions::tess_lds_bound: a window's accesses stay
                // inside the window, anything else inside the buffer.
                lds_bytes = m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(3), c_one_u}));  // StageParams::lds_bytes
                lds_limit = opt.tess_window ? cu(opt.tess_window) : lds_bytes;
            }
        }
        // TranslateOptions::tess_window: the patch's own window of the buffer.
        // Bounded, a patch whose window would end past lds_bytes reads 0 and
        // writes nothing (lds_patch_ok), and its base stays the buffer's.
        const auto window_of = [&](Id patch) {
            if (lds_bytes) {
                const Id windows = m.emit(spv::OpUDiv, t_u32, {lds_bytes, cu(opt.tess_window)});
                lds_patch_ok = ult(patch, windows);
                patch = sel(lds_patch_ok, patch, c_zero_u);
            }
            lds_base64 = m.emit(spv::OpIAdd, t_u64, {lds_base64, m.emit(spv::OpUConvert, t_u64, {imul(patch, cu(opt.tess_window))})});
        };
        const bool windowed = opt.tess_window && lds_in_buffer;
        // Stage-specific initial VGPRs/SGPRs
        if (opt.tess_role == TranslateOptions::TessRole::Domain) {
            // The domain points this patch's tessellation factors produce, in
            // the order a triangle list wants them. With every factor 1 (the
            // level here) that is the unit quad as two triangles: corners
            // (0,0) (1,0) (0,1) and (1,0) (1,1) (0,1). With outer factors L
            // and the inner factor 1 - the other shape these particles use -
            // the hardware puts no point inside the patch: it fans 4L
            // triangles from the centre to the 4L points around the edge, and
            // a grid's interior points would be domain coordinates the
            // hardware never asks for.
            const std::uint32_t level = std::max<std::uint32_t>(1, opt.domain_level);
            const Id vi = m.load(t_u32, in_vertex_index);
            const Id per_patch = cu(level == 1 ? 6 : 12 * level);
            const Id p = m.emit(spv::OpUDiv, t_u32, {vi, per_patch});
            const Id k = m.emit(spv::OpUMod, t_u32, {vi, per_patch});
            Id uu, vv;
            if (level == 1) {
                const Id ox = iand(shr(cu(0x1a), k), c_one_u);
                const Id oy = iand(shr(cu(0x34), k), c_one_u);
                uu = u2f(ox);
                vv = u2f(oy);
            } else {
                const std::uint32_t ring = 4 * level;
                const Id tri = m.emit(spv::OpUDiv, t_u32, {k, cu(3)});
                const Id c = m.emit(spv::OpUMod, t_u32, {k, cu(3)});
                // The two edge points of this triangle, and the centre for c == 0.
                const Id r = sel(ieq(c, cu(2)), m.emit(spv::OpUMod, t_u32, {iadd(tri, c_one_u), cu(ring)}), tri);
                const Id side = m.emit(spv::OpUDiv, t_u32, {r, cu(level)});
                const Id step = m.emit(spv::OpUMod, t_u32, {r, cu(level)});
                const Id t = fmul(u2f(step), cf(1.0f / static_cast<float>(level)));
                const Id one = cf(1.0f), zero = cf(0.0f);
                const Id back = fsub(one, t);
                // side 0 (t,0), 1 (1,t), 2 (1-t,1), 3 (0,1-t): counter-clockwise.
                Id eu = sel(ieq(side, c_zero_u), t, sel(ieq(side, c_one_u), one, sel(ieq(side, cu(2)), back, zero)));
                Id ev = sel(ieq(side, c_zero_u), zero, sel(ieq(side, c_one_u), t, sel(ieq(side, cu(2)), one, back)));
                const Id centre = ieq(c, c_zero_u);
                uu = sel(centre, cf(0.5f), eu);
                vv = sel(centre, cf(0.5f), ev);
            }
            m.store(vgpr_var(0), u(uu));
            m.store(vgpr_var(1), u(vv));
            m.store(vgpr_var(2), p);
            m.store(vgpr_var(3), p);
            if (opt.tess_patch_cull && lds_in_buffer) {
                // The patch's own place in the world is the fourth column of
                // its three rows (dwords 15, 19 and 23); if that is at or
                // behind the camera plane the quad has no projection.
                const Id cull = m.load(t_v4f, m.access_chain(m.type_pointer(spv::ScUniform, t_v4f), ubo_var, {cu(8)}));
                const Id layout = m.load(t_v4u, m.access_chain(m.type_pointer(spv::ScUniform, t_v4u), ubo_var, {cu(9)}));
                const Id stride = extract(t_u32, layout, 0), pbase = extract(t_u32, layout, 1);
                const Id near = m.emit(spv::OpBitcast, t_f32, {extract(t_u32, layout, 2)});
                const Id at = iadd(pbase, imul(p, stride));
                auto row_w = [&](int dword) {
                    return f(lds_load(iadd(at, cu(static_cast<std::uint32_t>(dword) * 4))));
                };
                const Id px = row_w(15), py = row_w(19), pz = row_w(23);
                Id w = fmul(extract(t_f32, cull, 0), px);
                w = fadd(w, fmul(extract(t_f32, cull, 1), py));
                w = fadd(w, fmul(extract(t_f32, cull, 2), pz));
                w = fadd(w, extract(t_f32, cull, 3));
                v_patch_culled = m.local_variable(m.type_pointer(spv::ScFunction, t_bool), m.const_bool(false));
                m.name(v_patch_culled, "patch_culled");
                // A negative near means "cull every patch", which says whether
                // the flag reaches the export at all.
                m.store(v_patch_culled, opt.tess_patch_cull_all
                                            ? m.const_bool(true)
                                            : m.emit(spv::OpFOrdLessThan, t_bool, {w, near}));
            }
        } else if (opt.tess_role == TranslateOptions::TessRole::HullTcs) {
            // The hull shader's v1 packs the patch it runs: the low byte is
            // the patch within the threadgroup, which addresses the LDS the
            // domain shader reads back, and bits 8-12 its slot in the factor
            // ring - a ring the host stages do not have, so only the low byte
            // has to be right. One patch a workgroup here, so gl_PrimitiveID
            // is that byte and the rest is 0.
            const Id patch = m.load(t_u32, in_primitive_id);
            if (windowed) {
                // The game's own hull: the control point it runs in bits
                // 8-12, its patch alone in its window.
                window_of(patch);
                m.store(vgpr_var(0), patch);
                m.store(vgpr_var(1), shl(m.load(t_u32, in_invocation_id), cu(8)));
            } else {
                m.store(vgpr_var(0), m.load(t_u32, in_invocation_id));
                m.store(vgpr_var(1), iand(patch, cu(0xff)));
            }
            m.store(vgpr_var(2), c_zero_u);
            m.store(vgpr_var(3), c_zero_u);
        } else if (opt.tess_role == TranslateOptions::TessRole::DomainTes) {
            // What the hardware's tessellator hands the domain shader, from
            // the host's: the point's place in the patch and which patch it
            // is. Unlike TessRole::Domain this invents no topology at all -
            // which points exist is the tessellator's decision again.
            const Id tc = m.load(t_v3f, in_tess_coord);
            const Id patch = m.load(t_u32, in_primitive_id);
            m.store(vgpr_var(0), u(extract(t_f32, tc, 0)));
            m.store(vgpr_var(1), u(extract(t_f32, tc, 1)));
            // With the control point arriving as an attribute the shader only
            // ever addresses its own patch, so the patch term of every address
            // it works out has to be zero. With a buffer behind it the patch is
            // what tells its slice apart.
            if (windowed) window_of(patch);
            const Id which = lds_in_attributes || windowed ? c_zero_u : patch;
            m.store(vgpr_var(2), which);
            m.store(vgpr_var(3), windowed ? patch : which);
        } else if (opt.tess_role == TranslateOptions::TessRole::LsCompute ||
                   opt.tess_role == TranslateOptions::TessRole::LsVertex) {
            // The control point this invocation runs: its index in the draw
            // (v1, which addresses its LDS slot) and its vertex id (v0). As a
            // vertex stage the draw is one vertex a control point, so the
            // index is what the workgroup arithmetic worked out.
            if (lds_in_attributes) {
                // As the pipeline's vertex stage there is nothing to work out:
                // the index buffer is bound, so gl_VertexIndex is already the
                // control point's vertex id, and the invocation writes only its
                // own patch, so its slot in that patch is zero.
                m.store(vgpr_var(0), m.load(t_u32, in_vertex_index));
                m.store(vgpr_var(1), c_zero_u);
                m.store(vgpr_var(2), c_zero_u);
                m.store(vgpr_var(3), c_zero_u);
            } else {
            Id gid;
            if (opt.tess_role == TranslateOptions::TessRole::LsVertex) {
                gid = m.load(t_u32, in_vertex_index);
            } else {
                const Id wg = m.load(t_v3u, in_workgroup_id);
                const Id lid = m.load(t_v3u, in_local_id);
                gid = iadd(imul(extract(t_u32, wg, 0), cu(std::max<std::uint32_t>(1, opt.cs_threads[0]))), extract(t_u32, lid, 0));
            }
            auto tess = [&](int k) { return m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(7), cu(k / 4), cu(k % 4)})); };
            // An instanced draw's control points are its points once per
            // instance, one after another: invocation i is point i % count of
            // instance i / count (the LS reads the instance in v2 and v3).
            const Id count = tess(kTessCount);
            const Id per = sel(ieq(count, c_zero_u), c_one_u, count);
            const Id instances = tess(kTessInstances);
            const Id total = imul(per, sel(ieq(instances, c_zero_u), c_one_u, instances));
            const Id at = sel(ult(gid, total), gid, isub(total, c_one_u));
            const Id instance = iadd(tess(kTessFirstInstance), m.emit(spv::OpUDiv, t_u32, {at, per}));
            const Id point = iadd(tess(kTessFirstPoint), m.emit(spv::OpUMod, t_u32, {at, per}));
            const Id index_va = make_u64(tess(kTessIndexLo), tess(kTessIndexHi));
            const Id wide = ine(tess(kTessIndexType), c_zero_u);
            const Id has_index = ine(ior(tess(kTessIndexLo), tess(kTessIndexHi)), c_zero_u);
            const Id vid = m.local_variable(p_fn_u32, c_zero_u);
            m.store(vid, point);
            if_then(has_index, [&] {
                const Id byte = sel(wide, shl(point, cu(2)), shl(point, c_one_u));
                const Id word_at = m.emit(spv::OpIAdd, t_u64, {index_va, m.emit(spv::OpUConvert, t_u64, {iand(byte, cu(~3u))})});
                const Id word = load_u32_at(word_at);
                m.store(vid, sel(wide, word, iand(shr(word, shl(iand(byte, cu(2)), cu(3))), cu(0xffff))));
            });
            m.store(vgpr_var(0), iadd(ld(vid), tess(kTessBaseVertex)));
            if (windowed) {
                // Control point i is slot i % n of patch i / n's window.
                const Id n = cu(std::max<std::uint32_t>(1, opt.tess_patch_control_points));
                window_of(m.emit(spv::OpUDiv, t_u32, {gid, n}));
                m.store(vgpr_var(1), m.emit(spv::OpUMod, t_u32, {gid, n}));
            } else {
                m.store(vgpr_var(1), gid);
            }
            // The instance-stepped inputs and the instance id (VGPR_COMP_CNT 2
            // and 3), at a step rate of one.
            m.store(vgpr_var(2), instance);
            m.store(vgpr_var(3), instance);
            }
        } else if (opt.stage == Stage::Vertex) {
            m.store(vgpr_var(0), m.load(t_u32, in_vertex_index));
            const int comp = (opt.rsrc1 >> 24) & 3;  // VGPR_COMP_CNT
            const Id inst = m.load(t_u32, in_instance_index);
            for (int k = 1; k <= comp; ++k) m.store(vgpr_var(k), inst);
        } else if (opt.stage == Stage::Pixel) {
            int v = 0;
            const std::uint32_t ena = opt.ps_input_ena;
            // barycentrics consumed by v_interp: their VGPRs carry nothing we need
            if (ena & 1) v += 2;
            if (ena & 2) v += 2;
            if (ena & 4) v += 2;
            if (ena & 8) v += 3;
            if (ena & 0x10) v += 2;
            if (ena & 0x20) v += 2;
            if (ena & 0x40) v += 2;
            if (ena & 0x80) v += 1;
            const Id fc = m.load(t_v4f, in_frag_coord);
            for (int k = 0; k < 4; ++k) {
                if (ena & (0x100 << k)) {
                    Id c = extract(t_f32, fc, k);
                    if (k == 3) c = fdiv(cf(1.0f), c);  // POS_W_FLOAT is 1/w; FragCoord.w is 1/w already? FragCoord.w = 1/w_clip
                    m.store(vgpr_var(v++), u(c));
                }
            }
            if (ena & 0x1000) {  // FRONT_FACE
                m.store(vgpr_var(v++), b2mask(m.load(t_bool, in_front_facing)));
            }
            if (ena & 0x2000) v++;  // ANCILLARY
            if (ena & 0x4000) v++;  // SAMPLE_COVERAGE
            if (ena & 0x8000) v++;  // POS_FIXED_PT
            // s[nuser] = PS state (prim mask): leave 0
        } else {
            const std::uint32_t r2 = opt.rsrc2;
            int s = user_sgpr_count();
            const Id wg = m.load(t_v3u, in_workgroup_id);
            if (r2 & (1 << 7)) m.store(sgpr_var(s++), extract(t_u32, wg, 0));
            if (r2 & (1 << 8)) m.store(sgpr_var(s++), extract(t_u32, wg, 1));
            if (r2 & (1 << 9)) m.store(sgpr_var(s++), extract(t_u32, wg, 2));
            if (r2 & (1 << 10)) m.store(sgpr_var(s++), c_zero_u);  // TG_SIZE
            const int tid = (r2 >> 11) & 3;
            const Id lid = m.load(t_v3u, in_local_id);
            m.store(vgpr_var(0), extract(t_u32, lid, 0));
            if (tid >= 1) m.store(vgpr_var(1), extract(t_u32, lid, 1));
            if (tid >= 2) m.store(vgpr_var(2), extract(t_u32, lid, 2));
        }

        // Blocks
        find_blocks(prog, 0);
        if (opt.fetch) find_blocks(*opt.fetch, kFetchBias);
        // A straight-line shader needs no program-counter interpreter. Keeping
        // it linear lets the driver propagate register values through the
        // whole shader instead of carrying them around a loop backedge.
        if (!opt.fetch && !opt.force_dispatcher && block_starts.size() == 1 &&
            std::none_of(prog.insts.begin(), prog.insts.end(), [&](const Inst& in) {
                return is_branch(in) || (in.enc == Enc::SOP1 && (in.op == 32 || (in.op == 33 && !vertex_input_call())));
            })) {
            lbl_loop_merge = m.fresh();
            ordered = true;
            for (const Inst& in : prog.insts) {
                if (block_terminated) break;
                translate_inst(in, 0);
            }
            if (!block_terminated) {
                lane_exit();
                m.emit_void(spv::OpBranch, {lbl_loop_merge});
            }
            m.label(lbl_loop_merge);
        } else if (!opt.fetch && !opt.force_dispatcher && forward_only(prog)) {
            // Every branch goes forward: guarded blocks in address order instead
            // of a dispatcher. A switch inside a loop keeps every register live
            // around the backedge; nested ifs optimise like ordinary code.
            dag = true;
            ordered = true;
            for (std::uint32_t start : block_starts) {
                reach_vars[start] = m.local_variable(p_fn_u32, start == 0 ? c_one_u : c_zero_u);
            }
            emit_dag(prog);
        } else if (!opt.fetch && !opt.force_dispatcher && loop_blocks_enabled() && !pc_transfers(prog)) {
            // Loops, but every branch is a direct one: the forward-only blocks
            // inside one loop instead of the dispatcher. A switch over every
            // block inside the loop made the largest compute shaders (~320
            // blocks) compile for 16 s on NVIDIA and for minutes, memory
            // climbing, on AMD's compiler; guarded blocks in address order are
            // ordinary code. Instruction order is not execution order here
            // (`ordered` stays false): a later definition reaches an earlier
            // use around the loop.
            dag = true;
            dag_loop = true;
            find_loop_regions(prog);
            for (std::uint32_t start : block_starts) {
                reach_vars[start] = m.local_variable(p_fn_u32, start == 0 ? c_one_u : c_zero_u);
            }
            emit_dag(prog);
        } else {
            m.store(v_pc, c_zero_u);
            const Id lbl_loop = m.fresh(), lbl_dispatch = m.fresh(), lbl_continue = m.fresh(), lbl_default = m.fresh();
            lbl_dispatch_merge = m.fresh();
            lbl_loop_merge = m.fresh();
            m.emit_void(spv::OpBranch, {lbl_loop});
            m.label(lbl_loop);
            m.emit_void(spv::OpLoopMerge, {lbl_loop_merge, lbl_continue, 0u});
            m.emit_void(spv::OpBranch, {lbl_dispatch});
            m.label(lbl_dispatch);
            const Id pc = ld(v_pc);
            std::map<std::uint32_t, Id> block_labels;
            for (std::uint32_t start : block_starts) block_labels[start] = m.fresh();
            std::vector<std::uint32_t> sw = {pc, lbl_default};
            for (const auto& [start, lbl] : block_labels) {
                sw.push_back(start);
                sw.push_back(lbl);
            }
            m.emit_void(spv::OpSelectionMerge, {lbl_dispatch_merge, 0u});
            m.emit_void(spv::OpSwitch, sw);
            // Block bodies
            emit_blocks(prog, 0, block_labels);
            if (opt.fetch) emit_blocks(*opt.fetch, kFetchBias, block_labels);
            m.label(lbl_default);
            m.emit_void(spv::OpBranch, {lbl_loop_merge});
            m.label(lbl_dispatch_merge);
            m.emit_void(spv::OpBranch, {lbl_continue});
            m.label(lbl_continue);
            m.emit_void(spv::OpBranch, {lbl_loop});
            m.label(lbl_loop_merge);
            lane_end.clear();  // blocks re-enter through the dispatcher
        }
        if (opt.stage == Stage::Vertex && (opt.vs_out_cntl & TranslateOptions::kVsOutCntlClipVarying) &&
            !out_params.count(static_cast<int>(TranslateOptions::vs_clip_location(opt.vs_out_cntl)))) {
            m.store(out_param(static_cast<int>(TranslateOptions::vs_clip_location(opt.vs_out_cntl))),
                    vec4f(cf(1.0f), cf(1.0f), cf(1.0f), cf(1.0f)));
        }
        if (opt.stage == Stage::Pixel && !(opt.exec_known && lane_end_seen && lane_end.count(kExecLo))) {
            // Lanes whose EXEC bit is clear at the end were killed (nothing to
            // kill where every way out keeps the lane bit set).
            const Id l_kill = m.fresh(), l_ret = m.fresh();
            const Id alive = exec_bit();
            m.emit_void(spv::OpSelectionMerge, {l_ret, 0u});
            m.emit_void(spv::OpBranchConditional, {alive, l_ret, l_kill});
            m.label(l_kill);
            m.emit_void(spv::OpKill, {});
            m.label(l_ret);
        }
        m.emit_void(spv::OpReturn, {});
        m.end_function();

        const spv::ExecutionModel model = opt.stage == Stage::Vertex          ? spv::EmVertex
                                          : opt.stage == Stage::Pixel         ? spv::EmFragment
                                          : opt.stage == Stage::TessControl   ? spv::EmTessellationControl
                                          : opt.stage == Stage::TessEval      ? spv::EmTessellationEvaluation
                                                                              : spv::EmGLCompute;
        if (opt.stage == Stage::TessControl || opt.stage == Stage::TessEval) m.capability(spv::CapTessellation);
        m.entry_point(model, fn_main, "main", interface);
        if (native_rtz) {
            m.execution_mode(fn_main, spv::ExDenormPreserve, {16});
            m.execution_mode(fn_main, spv::ExRoundingModeRTZ, {16});
        }
        if (opt.stage == Stage::Pixel) {
            m.execution_mode(fn_main, spv::ExOriginUpperLeft);
            // The pipeline writes neither depth nor stencil, so the tests can
            // run first even though the shader may discard.
            if (opt.early_fragment_tests && !out_frag_depth && !opt.ps_clip_discard) m.execution_mode(fn_main, spv::ExEarlyFragmentTests);
        } else if (opt.stage == Stage::Compute) {
            m.execution_mode(fn_main, spv::ExLocalSize, {opt.cs_threads[0], opt.cs_threads[1], opt.cs_threads[2]});
        } else if (opt.stage == Stage::TessControl) {
            m.execution_mode(fn_main, spv::ExOutputVertices, {std::max<std::uint32_t>(1, opt.tess_patch_control_points)});
        } else if (opt.stage == Stage::TessEval) {
            // VGT_TF_PARAM's domain, partitioning and topology, as the host's.
            m.execution_mode(fn_main, opt.tess_quads ? spv::ExQuads : spv::ExTriangles);
            m.execution_mode(fn_main, opt.tess_spacing == 1   ? spv::ExSpacingFractionalOdd
                                      : opt.tess_spacing == 2 ? spv::ExSpacingFractionalEven
                                                              : spv::ExSpacingEqual);
            m.execution_mode(fn_main, opt.tess_cw ? spv::ExVertexOrderCw : spv::ExVertexOrderCcw);
        }
        if (opt.stage == Stage::Vertex) {
            for (const auto& kv : out_params) res.vs_params.push_back(static_cast<std::uint32_t>(kv.first));
            res.vs_clip_count = out_clip ? clip_count : 0;
            res.vs_point_size = out_point_size != 0;
        }
        if (opt.stage == Stage::Pixel) {
            for (const auto& kv : in_params) res.ps_inputs.push_back(static_cast<std::uint32_t>(kv.first));
        }
        res.bindless = opt.bindless;
        res.spirv = m.assemble();
    }

    // A PC transfer (s_setpc/s_swappc other than the vertex-input call): only
    // the dispatcher follows one.
    bool pc_transfers(const Program& p) const {
        return std::any_of(p.insts.begin(), p.insts.end(), [&](const Inst& in) {
            return in.enc == Enc::SOP1 && (in.op == 32 || (in.op == 33 && !vertex_input_call()));
        });
    }
    // No PC transfers and every branch lands later in the program: blocks can
    // run in address order.
    bool forward_only(const Program& p) const {
        return !pc_transfers(p) && std::none_of(p.insts.begin(), p.insts.end(), [&](const Inst& in) {
            if (!is_branch(in)) return false;
            const std::uint32_t target = in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
            return target <= in.offset;
        });
    }
    // BBHOST_LOOP_BLOCKS=0: programs with loops take the dispatcher, as before.
    static bool loop_blocks_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("BBHOST_LOOP_BLOCKS");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    // The loops of a program without PC transfers: each backward branch's
    // target to the branch, and the ones that overlap without one holding the
    // other made one, so that what is left nests.
    void find_loop_regions(const Program& p) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> spans;
        for (const Inst& in : p.insts) {
            if (!is_branch(in)) continue;
            const std::uint32_t target = in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
            if (target > in.offset || !block_starts.count(target)) continue;
            back_targets.insert(target);
            spans.emplace_back(target, in.offset);
        }
        for (bool merged = true; merged;) {
            merged = false;
            for (std::size_t a = 0; a < spans.size() && !merged; ++a) {
                for (std::size_t b = 0; b < spans.size() && !merged; ++b) {
                    const auto& x = spans[a];
                    const auto& y = spans[b];
                    if (a == b || !(x.first < y.first && y.first <= x.second && x.second < y.second)) continue;
                    spans[a] = {x.first, y.second};
                    spans.erase(spans.begin() + static_cast<std::ptrdiff_t>(b));
                    merged = true;
                }
            }
        }
        std::sort(spans.begin(), spans.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first < y.first : x.second > y.second; });
        spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
        for (const auto& [start, end] : spans) {
            LoopRegion r;
            r.start = start;
            r.end = end;
            r.again = m.local_variable(p_fn_u32, c_zero_u);
            m.name(r.again, "again");
            loop_regions.push_back(r);
        }
    }
    // The `again` of the innermost open loop that holds a backward jump's
    // target (it holds the jump: the jump is being emitted inside it).
    Id loop_again(std::uint32_t target) {
        for (auto it = open_loops.rbegin(); it != open_loops.rend(); ++it) {
            if ((*it)->start <= target) return (*it)->again;
        }
        error("a backward branch outside its loop");
        return loop_regions.front().again;
    }

    // Blocks in address order, each a selection on its "reached" flag. With
    // loops (dag_loop) each loop's blocks sit inside a loop construct that
    // goes round again while a backward jump to one of them was taken.
    // Control is in one block at a time, so a pass runs forward from where it
    // enters and a backward jump is its last act.
    void emit_dag(const Program& p) {
        bool open = false;
        std::size_t next_loop = 0;
        const auto open_loop = [&](LoopRegion& r) {
            r.lbl_loop = m.fresh();
            r.lbl_continue = m.fresh();
            r.lbl_exit = m.fresh();
            const Id body = m.fresh();
            m.emit_void(spv::OpBranch, {r.lbl_loop});
            m.label(r.lbl_loop);
            m.emit_void(spv::OpLoopMerge, {r.lbl_exit, r.lbl_continue, 0u});
            m.emit_void(spv::OpBranch, {body});
            m.label(body);
            m.store(r.again, c_zero_u);
            open_loops.push_back(&r);
        };
        const auto close_loop = [&] {
            LoopRegion& r = *open_loops.back();
            m.emit_void(spv::OpBranch, {r.lbl_continue});
            m.label(r.lbl_continue);
            const Id again = ine(ld(r.again), c_zero_u);
            m.emit_void(spv::OpBranchConditional, {again, r.lbl_loop, r.lbl_exit});
            m.label(r.lbl_exit);
            open_loops.pop_back();
        };
        for (const Inst& in : p.insts) {
            if (block_starts.count(in.offset)) {
                if (open) {
                    if (!block_terminated) jump_to(in.offset);  // fall through
                    m.label(lbl_block_merge);
                }
                while (!open_loops.empty() && in.offset > open_loops.back()->end) close_loop();
                while (next_loop < loop_regions.size() && loop_regions[next_loop].start == in.offset) open_loop(loop_regions[next_loop++]);
                lbl_block_merge = m.fresh();
                cur_block = in.offset;
                const Id body = m.fresh();
                const Id reached = ine(ld(reach_vars.at(in.offset)), c_zero_u);  // before the merge instruction
                m.emit_void(spv::OpSelectionMerge, {lbl_block_merge, 0u});
                m.emit_void(spv::OpBranchConditional, {reached, body, lbl_block_merge});
                m.label(body);
                if (!open_loops.empty()) m.store(reach_vars.at(in.offset), c_zero_u);  // it may run again
                open = true;
                block_terminated = false;
                // Every edge into a block is emitted before it (forward only);
                // a loop's backward edges are not, so nothing is known there.
                if (back_targets.count(in.offset)) {
                    lane_set.clear();
                } else if (auto it = lane_in.find(in.offset); it != lane_in.end()) {
                    lane_set = it->second;
                } else if (in.offset != 0) {
                    lane_set.clear();
                }
                // The resource paths likewise, except at a loop's head, where
                // the back edges are not in yet. What every edge in agrees on
                // replaces the walk's state; a register they disagree on keeps
                // the walk's value, as before - so no binding is lost that
                // was traced before.
                if (!back_targets.count(in.offset)) {
                    if (auto it = sym_in.find(in.offset); it != sym_in.end()) {
                        for (const auto& [reg, v] : it->second) sym[reg] = v;
                    }
                    if (auto it = lane_syms_in.find(in.offset); it != lane_syms_in.end()) {
                        for (const auto& [key, v] : it->second) lane_syms[key] = v;
                    }
                }
                pending_smrd.clear();
            }
            if (block_terminated) continue;  // unreachable tail of a block (after s_endpgm)
            translate_inst(in, 0);
        }
        if (open) {
            if (!block_terminated) {  // ran off the end: treat as endpgm
                lane_exit();
                m.emit_void(spv::OpBranch, {lbl_block_merge});
                block_terminated = true;
            }
            m.label(lbl_block_merge);
        }
        while (!open_loops.empty()) close_loop();
    }

    void emit_blocks(const Program& p, std::uint32_t bias, const std::map<std::uint32_t, Id>& labels) {
        bool open = false;
        for (std::size_t k = 0; k < p.insts.size(); ++k) {
            const Inst& in = p.insts[k];
            const std::uint32_t off = bias + in.offset;
            if (block_starts.count(off)) {
                if (open && !block_terminated) {
                    // fall through into the next block
                    jump_to(off);
                }
                m.label(labels.at(off));
                open = true;
                block_terminated = false;
                lane_set.clear();  // entered from the dispatcher: nothing known
                pending_smrd.clear();
            }
            if (block_terminated) continue;  // unreachable tail of a block (after s_endpgm)
            translate_inst(in, bias);
        }
        if (open && !block_terminated) {
            // program ran off the end: treat as endpgm
            m.emit_void(spv::OpBranch, {lbl_loop_merge});
            block_terminated = true;
        }
    }
};

}  // namespace

TranslateResult translate(const Program& program, const TranslateOptions& options) {
    Translator t(program, options);
    t.run();
    // A bound V# must hold wherever its loads run. The symbolic tracking
    // follows instruction order, not the control-flow graph. In a straight-line
    // or forward-only program (no inlined fetch shader) instruction order is
    // execution order, so only the definitions emitted before a load can reach
    // it:
    // - the last of them, when it is in the load's own block (straight-line
    //   code is one block), is the only one that does, and must be the bound
    //   V#'s words;
    // - otherwise, in a program of guarded blocks (forward-only, or with loops
    //   of direct branches), the definitions reaching the block's entry over
    //   the recorded block edges must all be (a path without a definition is
    //   not counted, as before);
    // - otherwise they must all be.
    // Elsewhere (the dispatcher loop, the fetch shader's call and return)
    // every definition of those registers must be. A user-data V#'s registers
    // must not be written where it counts. Failing that, the V# goes back to
    // the page-table path.
    TranslateOptions again = options;
    const bool ordered = t.ordered;
    // Programs of guarded blocks: the definitions reaching each block's entry,
    // by register (keys of each block's last definition, merged along the
    // edges). Forward-only, every edge goes forward, so one pass in address
    // order visits predecessors first; with loops the backward edges carry a
    // loop's definitions to its head, so the passes repeat until nothing is
    // added. A block's own instructions run in emission order either way.
    // BBHOST_LOOP_REACH=0: in programs with loops every definition must be the
    // V#'s, as before.
    static const bool loop_reach = [] {
        const char* e = std::getenv("BBHOST_LOOP_REACH");
        return !(e && e[0] == '0');
    }();
    const bool reach = t.dag && (!t.dag_loop || loop_reach);
    std::map<std::uint32_t, std::map<int, std::set<std::string>>> reach_in;
    if (reach) {
        std::map<std::uint32_t, std::map<int, std::pair<std::uint64_t, std::string>>> last_def;  // block -> register -> (seq, key)
        for (const auto& [reg, defs] : t.sgpr_defs) {
            for (const auto& d : defs) {
                auto& slot = last_def[d.block][reg];
                if (slot.second.empty() || d.seq > slot.first) slot = {d.seq, d.key};
            }
        }
        for (bool added = true; added;) {
            added = false;
            for (std::uint32_t b : t.block_starts) {
                std::map<int, std::set<std::string>> out = reach_in[b];
                for (const auto& [reg, def] : last_def[b]) out[reg] = {def.second};
                for (std::uint32_t s : t.block_succ[b]) {
                    for (const auto& [reg, keys] : out) {
                        std::set<std::string>& in = reach_in[s][reg];
                        const std::size_t before = in.size();
                        in.insert(keys.begin(), keys.end());
                        added |= in.size() != before;
                    }
                }
            }
            if (!t.dag_loop) break;
        }
    }
    const auto holds = [&t, &reach_in, ordered, reach](int reg, const std::string& expect, std::uint64_t site_seq, std::uint32_t site_block) {
        const auto it = t.sgpr_defs.find(reg);
        if (it == t.sgpr_defs.end()) return true;
        if (ordered || reach) {
            // The last definition before the site in its own block, else (in
            // straight-line code) the last one before it.
            const Translator::SgprDef* latest = nullptr;
            for (const auto& d : it->second) {
                if (reach && d.block != site_block) continue;
                if (d.seq < site_seq && (!latest || d.seq > latest->seq)) latest = &d;
            }
            if (latest && latest->block == site_block) return expect[0] == 'l' && latest->key == expect;
            if (!reach) return true;
            {
                const auto bi = reach_in.find(site_block);
                if (bi == reach_in.end()) return true;
                const auto ri = bi->second.find(reg);
                if (ri == bi->second.end()) return true;
                for (const std::string& key : ri->second) {
                    if (expect[0] != 'l' || key != expect) return false;
                }
                return true;
            }
        }
        for (const auto& d : it->second) {
            if (ordered && d.seq > site_seq) continue;
            if (expect[0] != 'l' || d.key != expect) return false;
        }
        return true;
    };
    // Why sites fail, for TranslateResult::walks: whether the definition
    // just before the load (the one that reaches it in straight-line code) is
    // the bound V#'s, which a reaching-definition check would accept.
    const auto latest_matches = [&t](int reg, const std::string& expect, std::uint64_t site_seq) {
        const auto it = t.sgpr_defs.find(reg);
        if (it == t.sgpr_defs.end()) return true;
        const Translator::SgprDef* latest = nullptr;
        for (const auto& d : it->second) {
            if (d.seq < site_seq && (!latest || d.seq > latest->seq)) latest = &d;
        }
        return !latest || latest->key == expect;
    };
    std::map<std::string, std::uint32_t> exclusions;
    for (const auto& site : t.buffer_sites) {
        if (holds(site.reg, site.def0, site.seq, site.block) && holds(site.reg + 1, site.def1, site.seq, site.block)) continue;
        const char* why = site.def0[0] != 'l' ? "a user-data V# whose registers are written"
                          : reach && t.dag_loop ? "registers reused, a definition reaching it differs (loops)"
                          : !ordered          ? "registers redefined, dispatcher order"
                          : !latest_matches(site.reg, site.def0, site.seq) || !latest_matches(site.reg + 1, site.def1, site.seq)
                              ? "registers reused, the latest definition differs"
                          : t.dag ? "registers reused, the latest definition matches (forward-only)"
                                  : "registers reused, the latest definition matches (straight-line)";
        ++exclusions[std::string("exclusion: ") + why];
        const std::string key = t.res.buffers[site.index].path.str();
        if (std::find(again.cb_ssbo_exclude.begin(), again.cb_ssbo_exclude.end(), key) == again.cb_ssbo_exclude.end()) {
            again.cb_ssbo_exclude.push_back(key);
        }
    }
    // Whether the fragment stage may run in a 32-lane subgroup (gcn/wave.h):
    // from the program alone, the same for either translation below.
    const std::uint32_t wave64_needs = options.stage == Stage::Pixel ? pixel_wave64_needs(program, t.spill.reads) : kWave64NotPixel;
    if (again.cb_ssbo_exclude.size() == options.cb_ssbo_exclude.size()) {
        for (const auto& [why, n] : exclusions) t.res.walks[why] += n;
        t.res.wave64_needs = wave64_needs;
        return std::move(t.res);
    }
    Translator t2(program, again);
    t2.run();
    for (const auto& [why, n] : exclusions) t2.res.walks[why] += n;
    t2.res.wave64_needs = wave64_needs;
    return std::move(t2.res);
}

// The vertex stage of a tessellated draw: the control points are already in
// the buffer the LS compute pass wrote, so this exists only to give the
// tessellator a patch to work on.
std::vector<std::uint32_t> make_tess_passthrough_vs() {
    spv::Module m;
    m.capability(spv::CapShader);
    const Id t_void = m.type_void();
    const Id t_f32 = m.type_float(32);
    const Id t_v4f = m.type_vector(t_f32, 4);
    // A vertex stage with no output at all is a shape nothing else produces,
    // and one driver's shader compiler fell over on it. Write the position it
    // would always have: the evaluation stage computes its own.
    const Id v_pos = m.global_variable(m.type_pointer(spv::ScOutput, t_v4f), spv::ScOutput);
    m.decorate(v_pos, spv::DecBuiltIn, {spv::BiPosition});
    m.name(v_pos, "position");
    const Id fn_type = m.type_function(t_void, {});
    const Id fn = m.begin_function(t_void, fn_type);
    m.label();
    const Id zero = m.const_f32(0.0f), one = m.const_f32(1.0f);
    m.store(v_pos, m.emit(spv::OpCompositeConstruct, t_v4f, {zero, zero, zero, one}));
    m.emit_void(spv::OpReturn, {});
    m.end_function();
    m.entry_point(spv::EmVertex, fn, "main", {v_pos});
    return m.assemble();
}

// The control stage: the hull shader's factors, which are constants in it, and
// nothing else. Falling back to the emulated path is what happens when they
// are not constant, so this never has to compute them.
std::vector<std::uint32_t> make_tess_constant_tcs(const float outer[4], const float inner[2], std::uint32_t control_points,
                                                 std::uint32_t attr_vec4s) {
    spv::Module m;
    m.capability(spv::CapShader);
    m.capability(spv::CapTessellation);
    const Id t_void = m.type_void();
    const Id t_f32 = m.type_float(32);
    const Id t_u32 = m.type_int(32, false);
    const Id t_outer = m.type_array(t_f32, m.const_u32(4));
    const Id t_inner = m.type_array(t_f32, m.const_u32(2));
    const Id p_out_outer = m.type_pointer(spv::ScOutput, t_outer);
    const Id p_out_inner = m.type_pointer(spv::ScOutput, t_inner);
    const Id p_out_f32 = m.type_pointer(spv::ScOutput, t_f32);
    const Id v_outer = m.global_variable(p_out_outer, spv::ScOutput);
    const Id v_inner = m.global_variable(p_out_inner, spv::ScOutput);
    m.decorate(v_outer, spv::DecBuiltIn, {spv::BiTessLevelOuter});
    m.decorate(v_inner, spv::DecBuiltIn, {spv::BiTessLevelInner});
    m.decorate(v_outer, spv::DecPatch);
    m.decorate(v_inner, spv::DecPatch);
    m.name(v_outer, "tess_level_outer");
    m.name(v_inner, "tess_level_inner");
    // A control stage whose only output is the factors is the same unusual
    // shape: give it the per-vertex position every real one has.
    const Id cps = std::max<std::uint32_t>(1, control_points);
    const Id t_v4f = m.type_vector(t_f32, 4);
    const Id t_pos_arr = m.type_array(t_v4f, m.const_u32(cps));
    const Id v_pos = m.global_variable(m.type_pointer(spv::ScOutput, t_pos_arr), spv::ScOutput);
    m.decorate(v_pos, spv::DecBuiltIn, {spv::BiPosition});
    m.name(v_pos, "position");
    const Id p_in_u32 = m.type_pointer(spv::ScInput, t_u32);
    const Id v_invocation = m.global_variable(p_in_u32, spv::ScInput);
    m.decorate(v_invocation, spv::DecBuiltIn, {spv::BiInvocationId});
    m.name(v_invocation, "invocation_id");
    // The control point the vertex stage worked out, on its way to the
    // evaluation stage. Nothing here looks at it.
    Id v_attr_in = 0, v_attr_out = 0, t_v4u = 0;
    if (attr_vec4s) {
        t_v4u = m.type_vector(t_u32, 4);
        const Id t_attr = m.type_array(t_v4u, m.const_u32(attr_vec4s));
        const Id t_arr = m.type_array(t_attr, m.const_u32(cps));
        v_attr_in = m.global_variable(m.type_pointer(spv::ScInput, t_arr), spv::ScInput);
        v_attr_out = m.global_variable(m.type_pointer(spv::ScOutput, t_arr), spv::ScOutput);
        m.decorate(v_attr_in, spv::DecLocation, {0});
        m.decorate(v_attr_out, spv::DecLocation, {0});
        m.name(v_attr_in, "control_point_in");
        m.name(v_attr_out, "control_point");
    }
    const Id fn_type = m.type_function(t_void, {});
    const Id fn = m.begin_function(t_void, fn_type);
    m.label();
    {
        const Id zero = m.const_f32(0.0f), one = m.const_f32(1.0f);
        const Id id = m.emit(spv::OpLoad, t_u32, {v_invocation});
        m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_v4f), v_pos, {id}),
                m.emit(spv::OpCompositeConstruct, t_v4f, {zero, zero, zero, one}));
    }
    if (attr_vec4s) {
        const Id id = m.emit(spv::OpLoad, t_u32, {v_invocation});
        for (std::uint32_t k = 0; k < attr_vec4s; ++k) {
            const Id at = m.const_u32(k);
            const Id v = m.load(t_v4u, m.access_chain(m.type_pointer(spv::ScInput, t_v4u), v_attr_in, {id, at}));
            m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_v4u), v_attr_out, {id, at}), v);
        }
    }
    for (int k = 0; k < 4; ++k) {
        m.store(m.access_chain(p_out_f32, v_outer, {m.const_u32(static_cast<std::uint32_t>(k))}), m.const_f32(outer[k]));
    }
    for (int k = 0; k < 2; ++k) {
        m.store(m.access_chain(p_out_f32, v_inner, {m.const_u32(static_cast<std::uint32_t>(k))}), m.const_f32(inner[k]));
    }
    m.emit_void(spv::OpReturn, {});
    m.end_function();
    std::vector<Id> iface{v_outer, v_inner, v_pos, v_invocation};
    if (attr_vec4s) {
        iface.push_back(v_attr_in);
        iface.push_back(v_attr_out);
    }
    m.entry_point(spv::EmTessellationControl, fn, "main", iface);
    m.execution_mode(fn, spv::ExOutputVertices, {std::max<std::uint32_t>(1, control_points)});
    return m.assemble();
}

std::vector<std::uint32_t> make_rect_geometry_shader(const std::vector<std::uint32_t>& param_locations, int clip_count,
                                                     bool point_size) {
    using spv::Id;
    spv::Module m;
    m.capability(spv::CapShader);
    m.capability(2 /* Geometry */);
    if (clip_count) m.capability(spv::CapClipDistance);
    m.memory_model(0 /* Logical */, 1 /* GLSL450 */);
    const Id t_void = m.type_void();
    const Id t_f32 = m.type_float(32);
    const Id t_u32 = m.type_int(32, false);
    const Id t_i32 = m.type_int(32, true);
    const Id t_v4f = m.type_vector(t_f32, 4);
    const Id c3 = m.const_u32(3);
    // gl_PerVertex { vec4 Position; float PointSize; float ClipDistance[n]; }
    std::vector<Id> members = {t_v4f};
    if (point_size) members.push_back(t_f32);
    Id t_clip_arr = 0;
    if (clip_count) {
        t_clip_arr = m.type_array(t_f32, m.const_u32(static_cast<std::uint32_t>(clip_count)));
        members.push_back(t_clip_arr);
    }
    const Id t_pv = m.type_struct(members);
    m.decorate(t_pv, spv::DecBlock);
    m.member_decorate(t_pv, 0, spv::DecBuiltIn, {spv::BiPosition});
    std::uint32_t mi = 1;
    if (point_size) m.member_decorate(t_pv, mi++, spv::DecBuiltIn, {1 /* PointSize */});
    if (clip_count) m.member_decorate(t_pv, mi++, spv::DecBuiltIn, {3 /* ClipDistance */});
    const Id t_pv_in_arr = m.type_array(t_pv, c3);
    const Id p_in_pv = m.type_pointer(spv::ScInput, t_pv_in_arr);
    const Id p_out_pv = m.type_pointer(spv::ScOutput, t_pv);
    const Id in_pv = m.global_variable(p_in_pv, spv::ScInput);
    const Id out_pv = m.global_variable(p_out_pv, spv::ScOutput);
    m.name(in_pv, "gl_in");
    m.name(out_pv, "gl_out");
    const Id t_v4_arr = m.type_array(t_v4f, c3);
    const Id p_in_v4arr = m.type_pointer(spv::ScInput, t_v4_arr);
    const Id p_out_v4 = m.type_pointer(spv::ScOutput, t_v4f);
    const Id p_in_v4 = m.type_pointer(spv::ScInput, t_v4f);
    const Id p_in_f = m.type_pointer(spv::ScInput, t_f32);
    const Id p_out_f = m.type_pointer(spv::ScOutput, t_f32);
    std::vector<Id> interface = {in_pv, out_pv};
    std::vector<std::pair<Id, Id>> params;  // (in array var, out var)
    for (std::uint32_t loc : param_locations) {
        const Id in = m.global_variable(p_in_v4arr, spv::ScInput);
        const Id out = m.global_variable(p_out_v4, spv::ScOutput);
        m.decorate(in, spv::DecLocation, {loc});
        m.decorate(out, spv::DecLocation, {loc});
        interface.push_back(in);
        interface.push_back(out);
        params.push_back({in, out});
    }
    const Id fn_type = m.type_function(t_void, {});
    const Id fn = m.begin_function(t_void, fn_type);
    m.label();
    auto in_pos = [&](std::uint32_t v) { return m.load(t_v4f, m.access_chain(p_in_v4, in_pv, {m.const_i32(static_cast<std::int32_t>(v)), m.const_i32(0)})); };
    auto in_param = [&](Id var, std::uint32_t v) { return m.load(t_v4f, m.access_chain(p_in_v4, var, {m.const_i32(static_cast<std::int32_t>(v))})); };
    auto emit_vertex = [&](Id pos, const std::vector<Id>& pvals, const std::vector<Id>& clips, Id psize) {
        m.store(m.access_chain(p_out_v4, out_pv, {m.const_i32(0)}), pos);
        std::uint32_t member = 1;
        if (point_size) {
            m.store(m.access_chain(p_out_f, out_pv, {m.const_i32(static_cast<std::int32_t>(member))}), psize);
            ++member;
        }
        for (int k = 0; k < clip_count; ++k) {
            m.store(m.access_chain(p_out_f, out_pv, {m.const_i32(static_cast<std::int32_t>(member)), m.const_i32(k)}), clips[k]);
        }
        for (std::size_t k = 0; k < params.size(); ++k) m.store(params[k].second, pvals[k]);
        m.emit_void(static_cast<spv::Op>(218) /* OpEmitVertex */, {});
    };
    std::vector<Id> pos(3), psize(3, 0);
    std::vector<std::vector<Id>> pvals(3), clips(3);
    for (std::uint32_t v = 0; v < 3; ++v) {
        pos[v] = in_pos(v);
        std::uint32_t member = 1;
        if (point_size) {
            psize[v] = m.load(t_f32, m.access_chain(p_in_f, in_pv, {m.const_i32(static_cast<std::int32_t>(v)), m.const_i32(static_cast<std::int32_t>(member))}));
            ++member;
        }
        for (int k = 0; k < clip_count; ++k) {
            clips[v].push_back(m.load(t_f32, m.access_chain(p_in_f, in_pv, {m.const_i32(static_cast<std::int32_t>(v)), m.const_i32(static_cast<std::int32_t>(member)), m.const_i32(k)})));
        }
        for (auto& pr : params) pvals[v].push_back(in_param(pr.first, v));
    }
    for (std::uint32_t v = 0; v < 3; ++v) emit_vertex(pos[v], pvals[v], clips[v], psize[v] ? psize[v] : m.const_f32(1.0f));
    // v3 = v1 + v2 - v0
    auto combine4 = [&](Id a, Id b, Id c) { return m.emit(spv::OpFSub, t_v4f, {m.emit(spv::OpFAdd, t_v4f, {b, c}), a}); };
    auto combine1 = [&](Id a, Id b, Id c) { return m.emit(spv::OpFSub, t_f32, {m.emit(spv::OpFAdd, t_f32, {b, c}), a}); };
    std::vector<Id> p3, c3v;
    for (std::size_t k = 0; k < params.size(); ++k) p3.push_back(combine4(pvals[0][k], pvals[1][k], pvals[2][k]));
    for (int k = 0; k < clip_count; ++k) c3v.push_back(combine1(clips[0][k], clips[1][k], clips[2][k]));
    emit_vertex(combine4(pos[0], pos[1], pos[2]), p3, c3v, psize[0] ? combine1(psize[0], psize[1], psize[2]) : m.const_f32(1.0f));
    m.emit_void(static_cast<spv::Op>(219) /* OpEndPrimitive */, {});
    m.emit_void(spv::OpReturn, {});
    m.end_function();
    m.entry_point(static_cast<spv::ExecutionModel>(3 /* Geometry */), fn, "main", interface);
    m.execution_mode(fn, static_cast<spv::ExecutionMode>(22 /* Triangles */));
    m.execution_mode(fn, static_cast<spv::ExecutionMode>(0 /* Invocations */), {1});
    m.execution_mode(fn, static_cast<spv::ExecutionMode>(29 /* OutputTriangleStrip */));
    m.execution_mode(fn, static_cast<spv::ExecutionMode>(26 /* OutputVertices */), {4});
    return m.assemble();
}

}  // namespace gcn
