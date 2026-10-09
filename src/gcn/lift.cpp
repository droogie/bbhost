// Typed per-pixel lifting of GCN pixel shaders: see lift.h.
//
// Execution model. The translated shader runs one invocation per lane and
// keeps EXEC and VCC as 64-bit masks; everything it reads from another lane
// comes from a ballot. The lifter represents a 64-bit mask only by this
// pixel's bit (Kind::Lane) and rejects every use that needs another lane's bit
// or the mask as a number. What remains is per-pixel arithmetic, which it
// emits as typed values in a single block:
//
//   * A VGPR write under EXEC is select(exec, new, old); where EXEC is known
//     set it is a plain value.
//   * An s_cbranch_execz skips its region only when EXEC is clear, where the
//     masked writes keep their old values anyway. The region is emitted
//     unconditionally when it writes no memory, sets EXEC only within EXEC at
//     its start, and every scalar register it writes is redefined before it
//     is read on every path out of it, around loops too (scalar writes are
//     not masked, and GCN runs a block once for all lanes when any lane needs
//     it). A lane mask it writes may stay live where it equals, wherever GCN
//     can skip the region, its value on the skipped path.
//   * An s_cbranch_scc0 region (the alpha-test kill idiom) is emitted
//     unconditionally too when SCC was "some lane of mask M is set" and the
//     region's first instruction ANDs M into EXEC: a pixel whose M bit is
//     clear writes nothing whether or not the region runs. EXEC after it must
//     If EXEC after it differs from EXEC at the branch, or the region widens
//     EXEC to whole quads (s_wqm_b64; the lifted pixel keeps its own bit), the
//     lift can differ from GCN only in pixels whose M bit is clear: EXEC at
//     s_endpgm must then lie within M, and no implicit-LOD sample may follow.
//     Widening from a varying EXEC needs M within it. Kill regions nest, the
//     inner one's mask within the outer one's.
//   * An s_cbranch_scc0/scc1 or s_cbranch_vccz/vccnz whose bit every lane
//     holds alike becomes structured control flow: SCC from a scalar compare
//     (scalar operands never read a VGPR), or VCC computed only from
//     constants, user data and loads - or VCC a comparison of values every
//     lane holds alike where EXEC is set, ANDed with an EXEC GCN cannot reach
//     the branch with empty (EXEC where the innermost region began). It has a
//     then arm, an else arm where an s_branch closes the then arm, and a
//     merge whose phis carry the values the arms left different. In an
//     else-if chain whose inner arms jump straight to the outer join, the
//     inner if joins at the s_branch closing the outer then arm instead.
//   * A backward s_branch closes a loop, left only for the instruction after
//     it: an OpLoopMerge loop per pixel with phis for what the body writes.
//     A uniform loop exits on a bit every lane holds alike; a divergent one is
//     the per-lane exit idiom on a loop mask (see "loops" below).
//   * Samples run for every pixel in uniform control flow and their texels are
//     written under EXEC. Where EXEC is known set, Vulkan helper invocations
//     compute the samples for a quad's uncovered pixels as GCN's whole-quad
//     lanes do, and helper outputs are discarded. Under a varying EXEC an
//     explicit-LOD sample reads only the pixel's own operands; an
//     implicit-LOD sample (and image_get_lod, OpImageQueryLod) needs its
//     coordinates to hold GCN's values in every pixel of each quad holding an
//     EXEC pixel, since GCN's texture unit takes the derivatives from the
//     quad's four lanes, EXEC set or not, as Vulkan takes them from the quad's
//     four invocations. In a loop pixels leave at different iterations,
//     where control flow is not uniform, only explicit-LOD samples are
//     lifted. Exports are unconditional, as in the translator; a pixel whose
//     EXEC bit is clear at s_endpgm is discarded, and every pixel exported
//     with its bit clear must be one of them.
//   * Whole-quad mode of a varying EXEC M (s_wqm_b64 exec, exec, then VALU
//     work up to s_mov_b64 exec, <M>: the coordinates of a sample in a
//     divergent region) is lifted for every pixel. What it writes holds GCN's
//     value only in the quads of M; the lifter keeps, per register, where it
//     holds GCN's value (`partial`) and checks that at every use, around
//     loops too. Not in a loop pixels leave at different iterations.
//   * ds_swizzle_b32 in quad mode (and the bit-mask patterns that stay inside
//     a quad) becomes OpGroupNonUniformQuadBroadcast or QuadSwap: Vulkan lays
//     a fragment shader's quad out as GCN does (lane 1 right of lane 0, lane 2
//     below it) and launches helper invocations for the quad's uncovered
//     pixels. It is accepted only where EXEC is known set in whole-quad mode,
//     so every lane of the quad is active on GCN and returns its own value, and
//     only of a value the helper lanes computed as GCN's did (Val::helper_inexact,
//     carried around loops), never inside a loop pixels leave at different
//     iterations.
//   * Scalar loads read the storage buffers the reference bound for them
//     (TranslateResult::buffer_at), with the translator's landing rule: a load
//     is stored at once and stored again at the next s_waitcnt on lgkmcnt,
//     within one block. Indexed tbuffer loads read the element the
//     reference's indexed binding holds, decoded as the translator decodes it.
//     A scalar load the reference walks the page table for is lifted only
//     when its words are descriptors and nothing else: the reference binds
//     images, samplers and buffers by their resource paths, never by the
//     words, so the lift loads nothing and rejects any read of them as data.
//   * SGPR spills (v_writelane_b32 vN, s, L ... v_readlane_b32 d, vN, L with a
//     constant lane): the reference reads the scalar from a variable of its
//     own wherever only the writes to that (VGPR, lane) reach the read
//     (gcn/wave.h, spill cells), and the lift keeps that cell as a register.
//     Any other v_readlane_b32 reads another pixel's VGPR and is rejected, and
//     so is reading vN as this pixel's word before it is written whole again.
//
// Operations mirror translate.cpp exactly (GLSL.std.450 Fma for mac/mad,
// NMin/NMax, the legacy multiply's zero rule, PackHalf2x16 exports, the cube
// helpers and sample operand order), so a lifted shader can be compared with
// its reference texel for texel.
#include "gcn/lift.h"

#include "gcn/half.h"
#include "gcn/spirv.h"
#include "gcn/wave.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace gcn {
namespace {

using spv::Id;

enum class Kind : std::uint8_t { Word, Float, Lane };

struct Val {
    Kind kind = Kind::Word;
    Id id = 0;
    bool constant = false;
    std::uint32_t bits = 0;  // Word / Float constant
    bool set = false;        // Lane constant
    // A helper lane (a pixel of the quad the primitive does not cover) may hold
    // something else on GCN than the lift computes for it: a mask from the
    // coverage EXEC a pixel shader starts with, or a value written outside
    // whole-quad mode. Only cross-lane reads (ds_swizzle_b32) look at it.
    bool helper_inexact = false;
};

// Register keys: SGPRs 0..103, TTMPs 200..211, VGPRs 256.., and the scalar
// state below. EXEC and VCC exist only as lane masks.
constexpr int kKeyM0 = 124;
constexpr int kKeyScc = 1000;
constexpr int kKeyExec = 1001;
constexpr int kKeyVcc = 1002;
// A spill cell (gcn/wave.h): the scalar a v_writelane_b32 parked in lane L of
// VGPR N, kept as a register of its own.
constexpr int kKeyCell = 4096;
constexpr int cell_key(int vgpr, int lane) { return kKeyCell + vgpr * 64 + lane; }
// A lane operand of v_readlane_b32 / v_writelane_b32 that names a lane
// itself: an inline integer 0..63, as translate.cpp const_lane.
bool const_lane(std::uint16_t code, int& lane) {
    if (code < 128 || code > 192) return false;
    lane = code - 128;
    return lane < 64;
}

struct Region {
    std::uint32_t branch = 0, start = 0, end = 0;  // the branch, masked instructions [start, end)
    std::set<int> scalar_writes;
    bool closed = false;
    // s_cbranch_scc0 (kill = true): SCC was "some lane of `guard` is set", and
    // the region's first instruction ANDs `guard` into EXEC, so a pixel whose
    // guard bit is clear writes nothing whether or not the region runs.
    bool kill = false;
    bool widened = false;     // the AND is followed by s_wqm_b64 exec, exec
    bool quad_exact = false;  // widened from EXEC known set: every pixel computes the region, as GCN's widened quads do
    Val guard, exec_before;
    std::map<int, Val> lanes_at_branch;  // the lane masks where the branch tests whether to skip the region
    // EXEC where the region's writes begin (after a kill region's AND and
    // widening); every EXEC set inside the region must lie within it.
    Val exec_in;
    bool disabled = false;  // an s_cbranch_scc0 whose SCC was a scalar condition: an if, not a kill region
    std::uint64_t seq = 0;  // opening order among regions and ifs
};

bool is_branch(const Inst& in) { return in.enc == Enc::SOPP && (in.op == 2 || (in.op >= 4 && in.op <= 9)); }

std::string hex_offset(std::uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06x", v);
    return buf;
}

std::string key_name(int key) {
    if (key == kKeyScc) return "scc";
    if (key == kKeyExec) return "exec";
    if (key == kKeyVcc) return "vcc";
    if (key == kKeyM0) return "m0";
    if (key < 104) return "s" + std::to_string(key);
    if (key >= 200 && key < 212) return "ttmp" + std::to_string(key - 200);
    if (key >= kKeyCell) return "lane " + std::to_string((key - kKeyCell) % 64) + " of v" + std::to_string((key - kKeyCell) / 64);
    if (key >= 256) return "v" + std::to_string(key - 256);
    return "?";
}

class Lifter {
public:
    // `loop_phis`: lane masks (loop header, key) an earlier attempt found
    // changed around a loop; they get phis instead of being assumed unchanged.
    // `loop_inexact`: registers (loop header, key) an earlier attempt found
    // coming back to the header with helper lanes GCN may hold differently
    // (Val::helper_inexact); they are inexact at the header from the start.
    using LoopPhis = std::set<std::pair<std::uint32_t, int>>;
    Lifter(const Program& p, const TranslateOptions& o, const TranslateResult& r, const LoopPhis& loop_phis = {}, const LoopPhis& loop_inexact = {})
        : prog(p), opt(o), ref(r), lane_phi_hints(loop_phis), inexact_hints(loop_inexact) {}
    LoopPhis more_lane_phis;  // set when a lift failed only because a lane mask needs a phi
    LoopPhis more_inexact;    // ... or because helper lanes came back around a loop less exact than they left

    LiftResult run() {
        check_program();
        if (res.rejections.empty()) emit();
        if (res.rejections.empty()) {
            res.spirv = m.assemble();
            finish_proof();
        } else {
            res.spirv.clear();
        }
        return std::move(res);
    }

private:
    const Program& prog;
    const TranslateOptions& opt;
    const TranslateResult& ref;
    LiftResult res;
    spv::Module m;
    std::uint32_t cur = 0;
    std::string note;  // this instruction's writes, for the listing

    // Types and interface
    Id t_void = 0, t_bool = 0, t_u32 = 0, t_i32 = 0, t_u64 = 0, t_f32 = 0, t_v2f = 0, t_v3f = 0, t_v4f = 0, t_v2i = 0, t_v3i = 0;
    Id p_uni_u32 = 0, p_in_v4f = 0, p_in_bool = 0, p_out_v4f = 0, p_ssbo_u32 = 0, p_in_u32 = 0, p_in_v4u = 0, t_v4u = 0;
    Id ubo_var = 0, fn_main = 0;
    std::vector<Id> interface;
    std::vector<Id> image_vars, image_types, sampler_vars, buffer_vars;
    std::map<int, Id> in_params, out_params, attr_loads;
    std::map<std::uint32_t, Id> in_location_vars;  // PS: location -> the Input vec4 there (attr)
    Id in_frag_coord = 0, in_front_facing = 0, in_vertex_index = 0, in_instance_index = 0, out_pos = 0;
    std::map<std::uint32_t, Id> vertex_in_vars;  // TranslateOptions::vertex_input locations
    std::map<std::uint32_t, Id> bias_loads, stride_loads, w3_loads;  // StageParams cb_bias_dw / cb_stride / cb_w3 per buffer
    struct PackedPair {
        Id lo, hi;
        std::uint32_t block;  // Module::blocks() when packed
    };
    std::map<Id, PackedPair> packed_pairs;  // v_cvt_pkrtz_f16_f32's result -> its two floats
    bool native_rtz = false;  // as the translator (gcn/half.h native_half_rtz)
    Id t_v2h = 0;

    // Values
    std::map<int, Val> reg;
    std::map<int, Val> initials;  // each register's value at entry, loaded once
    std::map<int, Val> lanes;  // pair low key (or kKeyExec / kKeyVcc) -> this pixel's bit
    std::map<Id, Id> to_float, to_word;
    struct Pending {
        int sdst;
        std::vector<Val> values;
    };
    std::vector<Pending> pending;

    // Control flow
    std::vector<Region> regions;
    std::set<std::uint32_t> block_starts;
    std::vector<Region*> open_regions;  // innermost last
    // s_cbranch_scc0/scc1 or s_cbranch_vccz/vccnz over a then arm, with an else
    // arm where an s_branch closes the then arm. check_program finds them;
    // emit() opens those whose tested bit every lane holds alike.
    struct Construct {
        std::uint32_t branch = 0;
        std::uint16_t op = 4;          // s_cbranch_scc0/scc1 (4/5) or s_cbranch_vccz/vccnz (6/7)
        bool negate = false;           // scc1/vccnz: taken where the bit is set, so the then arm runs where it is clear
        std::uint32_t else_jump = 0;   // the s_branch closing the then arm; 0 without an else arm
        std::uint32_t else_start = 0;  // the branch target: the else arm, or the join without one
        std::uint32_t join = 0;
    };
    std::vector<Construct> constructs;
    std::vector<std::uint32_t> threaded_joins;  // ifs of an else-if chain joined at the enclosing then arm's s_branch
    struct Arm {  // an open if
        Construct* c = nullptr;
        std::uint64_t seq = 0;
        // VCC was a uniform condition ANDed with EXEC, at the start of the
        // region beginning here (0: the bit itself was uniform).
        std::uint32_t split_region = 0;
        bool then_dead = false;  // the then arm ended by leaving a loop
        Id merge = 0, then_label = 0, else_label = 0, then_end = 0;
        bool in_else = false;
        std::map<int, Val> reg0, lanes0, reg1, lanes1;  // at the branch; at the then arm's end
        std::map<Id, Id> to_float0, to_word0;
        Val scc0, scc1v;
        bool scc_uniform0 = false, scc_valid0 = true, scc_uniform1 = false, scc_valid1 = true;
        std::set<int> poisoned0, poisoned1;
    };
    std::vector<Arm> arms;  // innermost last
    Id cur_label = 0;
    bool scc_uniform = false;  // scc_mask is SCC itself, from a scalar compare, not "some lane of a mask"
    bool scc_valid = true;     // false after an if whose arms left SCC different
    std::size_t if_constructs = 0;
    // SGPRs an if left holding a lane mask on one arm only: unknown after the
    // join, so reading them is rejected until they are written again.
    std::set<int> poisoned;

    // Proof bookkeeping
    std::vector<std::uint32_t> samples, exports;
    std::vector<std::uint32_t> divergent_samples;  // inside a divergent loop
    std::size_t buffer_loads = 0, masked_writes = 0;

    // The reference's spill cells (the same analysis translate.cpp runs).
    const SpillCells spill = spill_cells_on() ? spill_cells(prog) : SpillCells{};
    std::size_t cell_reads = 0;
    // Words of scalar loads the reference walks the page table for, by the
    // OpUndef that stands for them: only descriptors may come from there.
    std::map<Id, std::uint32_t> descriptor_words;  // id -> the load's offset
    std::size_t descriptor_loads = 0;
    bool descriptor(const Val& v) const { return !v.constant && v.kind == Kind::Word && descriptor_words.count(v.id) != 0; }
    void reject_descriptor_read(const Val& v) {
        const std::uint32_t at = descriptor_words.at(v.id);
        const Inst* load = inst_at(at);
        reject("data from a scalar load the reference walks the page table for (only its descriptors are lifted): " +
               std::string(load && mnemonic(*load) ? mnemonic(*load) : "the load") + " at " + hex_offset(at));
    }

    // SCC as the translator leaves it after a 64-bit mask op: some lane of
    // scc_mask is set.
    Val scc_mask;
    bool exec_write_ok = false;             // this instruction is a kill region's EXEC mask, widening or restore
    bool widen_ok = false;                  // ... and it is the widening s_wqm_b64
    // Kill regions after which the lift may differ from GCN in pixels whose
    // guard bit is clear: per pixel (derivatives could see them), or only in
    // quads where no pixel survives.
    struct KillProof {
        Val guard;
        bool per_pixel;
    };
    std::vector<KillProof> kill_proofs;
    std::vector<Val> export_execs;          // EXEC at exports where it was not known set
    std::map<Id, std::set<Id>> conjuncts;   // the operands of each lane mask built by LogicalAnd
    std::map<Id, Id> negation_of;           // lane masks LogicalNot built, both ways: each -> one that is its negation
    // Values every lane holds alike: built only from constants, user data,
    // scalar loads and other such values (scalar operands never read a VGPR).
    // An instruction's results are uniform when everything it read was.
    std::set<Id> uniform_ids;
    bool inst_uniform = true;
    // Something this instruction read is Val::helper_inexact (the EXEC of a
    // VGPR write included): so are its word and float results.
    bool inst_helper_inexact = false;
    bool uniform(const Val& v) const { return v.constant || uniform_ids.count(v.id) != 0; }
    Val note_read(const Val& v) {
        if (!uniform(v)) inst_uniform = false;
        if (v.helper_inexact) inst_helper_inexact = true;
        return v;
    }
    void note_result(const Val& v) {
        if (inst_uniform && !v.constant) uniform_ids.insert(v.id);
    }
    bool needs_kill = false;                // EXEC not known set at s_endpgm
    Val kill_exec;
    std::size_t kill_regions = 0;

    void reject(const std::string& why) {
        const std::string msg = hex_offset(cur) + ": " + why;
        if (std::find(res.rejections.begin(), res.rejections.end(), msg) == res.rejections.end()) res.rejections.push_back(msg);
    }

    // ---- program checks ------------------------------------------------------------
    void check_program() {
        if (opt.stage != Stage::Pixel && opt.stage != Stage::Vertex) reject("only pixel and vertex shaders are lifted");
        if (!opt.cb_no_fallback) reject("the reference must be the no-fallback translation (every storage buffer bound)");
        if (opt.debug_ps) reject("debug pixel-output modes are not lifted");
        if (opt.fetch) reject("an inlined fetch shader is not lifted (vertex shaders are lifted on vertex input)");
        if (!ref.ok()) reject("the reference translation failed");
        if (!prog.errors.empty()) reject("the program has decode errors");
        if (prog.insts.empty()) reject("empty program");
        if (!res.rejections.empty()) return;
        const Inst& last = prog.insts.back();
        if (!(last.enc == Enc::SOPP && last.op == 1)) {
            cur = last.offset;
            reject("the program does not end in s_endpgm");
        }
        const std::uint32_t end = last.offset + last.size * 4;
        find_loops();  // first: a forward branch may be a loop's exit
        for (const Inst& in : prog.insts) {
            cur = in.offset;
            // s_swappc_b64 calls the fetch shader; with vertex input a vertex
            // shader loads its elements there instead (translate.cpp vertex_input_call).
            const bool vertex_input_call = in.enc == Enc::SOP1 && in.op == 33 && opt.stage == Stage::Vertex && !opt.vertex_input.empty();
            if (in.enc == Enc::SOP1 && (in.op == 32 || in.op == 33) && !vertex_input_call) reject("program-counter transfer");
            if (in.enc == Enc::SOPP && in.op == 1 && &in != &last) reject("s_endpgm before the end of the program");
            if (!is_branch(in)) continue;
            const std::uint32_t target = in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
            if (loop_branch(in.offset)) continue;  // a back edge or a loop exit (find_loops)
            if (target <= in.offset || target > end) {
                reject(std::string(mnemonic(in)) + " must branch forward within the program");
            } else if (in.op == 2) {
                continue;  // it must close an if's then arm (checked below)
            } else if (in.op >= 4 && in.op <= 7) {
                if (target == in.offset + 4) {
                    reject(std::string(mnemonic(in)) + " branches over nothing");
                    continue;
                }
                Construct c;
                c.branch = in.offset;
                c.op = in.op;
                c.negate = in.op == 5 || in.op == 7;
                c.else_start = target;
                c.join = target;
                if (const Inst* jump = inst_at(target - 4); jump && jump->enc == Enc::SOPP && jump->op == 2 && !loop_branch(jump->offset)) {
                    const std::uint32_t join = jump->offset + 4 + static_cast<std::uint32_t>(jump->imm) * 4;
                    if (join > target && join < end) {
                        c.else_jump = jump->offset;
                        c.join = join;
                    }
                }
                constructs.push_back(c);
                block_starts.insert(in.offset + 4);
                block_starts.insert(c.else_start);
                block_starts.insert(c.join);
                // Without an else arm, an s_cbranch_scc0 on a mask's SCC is a kill
                // region instead; emit() decides at the branch, by what SCC holds.
                if (in.op == 4 && !c.else_jump && opt.stage == Stage::Pixel) {
                    Region r;
                    r.kill = true;
                    r.branch = in.offset;
                    r.start = in.offset + 4;
                    r.end = target;
                    regions.push_back(r);
                }
            } else if (in.op != 8) {
                reject(std::string(mnemonic(in)) + ": only s_cbranch_execz, s_cbranch_scc0/scc1, s_cbranch_vccz/vccnz and an else arm's s_branch are lifted");
            } else {
                Region r;
                r.branch = in.offset;
                r.start = in.offset + 4;
                r.end = target;
                regions.push_back(r);
                block_starts.insert(r.start);
                block_starts.insert(r.end);
            }
        }
        thread_if_joins();
        std::set<std::uint32_t> else_jumps;
        for (const Construct& c : constructs) {
            if (c.else_jump && !else_jumps.insert(c.else_jump).second) {
                cur = c.branch;
                reject("two ifs share the s_branch at " + hex_offset(c.else_jump));
            }
        }
        for (const Inst& in : prog.insts) {
            if (in.enc == Enc::SOPP && in.op == 2 && !else_jumps.count(in.offset) && !loop_branch(in.offset)) {
                cur = in.offset;
                reject("s_branch that does not close an if's then arm, end a loop or leave one");
            }
        }
        std::sort(regions.begin(), regions.end(), [](const Region& a, const Region& b) { return a.start < b.start; });
        for (std::size_t k = 0; k < regions.size(); ++k) {
            const Region& inner = regions[k];
            for (std::size_t j = 0; j < k; ++j) {
                const Region& outer = regions[j];
                // An s_cbranch_execz region may sit wholly inside another: where
                // the outer one is skipped, EXEC is clear at the inner branch too.
                // (A kill region inside another region is rejected at its branch.)
                if (inner.start < outer.end && inner.end > outer.end) {
                    cur = inner.branch;
                    reject("overlapping regions");
                }
            }
        }
        // Ifs nest with regions and with each other: anything overlapping an if
        // lies inside one of its arms or contains it.
        const auto within = [](std::uint32_t a0, std::uint32_t a1, std::uint32_t b0, std::uint32_t b1) { return b0 <= a0 && a1 <= b1; };
        for (const Construct& c : constructs) {
            const auto nests = [&](std::uint32_t x0, std::uint32_t x1) {
                return x1 <= c.branch || x0 >= c.join || within(x0, x1, c.branch + 4, c.else_jump ? c.else_jump : c.join) ||
                       (c.else_jump && within(x0, x1, c.else_start, c.join)) || within(c.branch, c.join, x0, x1);
            };
            for (const Region& r : regions) {
                if (r.branch != c.branch && !nests(r.branch, r.end)) {
                    cur = r.branch;
                    reject("a region overlaps the if at " + hex_offset(c.branch));
                }
            }
            for (const Construct& o : constructs) {
                if (&o != &c && !nests(o.branch, o.join)) {
                    cur = o.branch;
                    reject("the ifs at " + hex_offset(c.branch) + " and " + hex_offset(o.branch) + " overlap");
                }
            }
        }
        check_loop_nesting();
    }

    // An else-if chain: an if inside another's then arm whose arms jump
    // straight to the outer join, past the s_branch that closes the outer then
    // arm. That s_branch only jumps to the same join, so jumping to it instead
    // is the same program, and the inner if then joins there, inside the outer
    // then arm. Each such if takes the nearest enclosing s_branch.
    void thread_if_joins() {
        const auto target_of = [&](std::uint32_t at) {
            const Inst* in = inst_at(at);
            return in ? in->offset + 4 + static_cast<std::uint32_t>(in->imm) * 4 : 0u;
        };
        for (Construct& c : constructs) {
            std::uint32_t best = 0;
            for (const Construct& o : constructs) {
                if (&o == &c || !o.else_jump || o.branch >= c.branch || target_of(o.else_jump) != c.join) continue;
                if (o.else_jump <= c.branch || o.else_jump >= c.join) continue;
                if (c.else_jump && o.else_jump < c.else_start) continue;  // it would end the inner then arm instead
                if (!best || o.else_jump < best) best = o.else_jump;
            }
            if (!best) continue;
            if (!c.else_jump) c.else_start = best;
            c.join = best;
            block_starts.insert(best);
            threaded_joins.push_back(c.branch);
        }
    }

    // ---- values ----------------------------------------------------------------------
    Id cf(float v) { return m.const_f32(v); }
    Id cu(std::uint32_t v) { return m.const_u32(v); }
    Val word_const(std::uint32_t b) { return {Kind::Word, cu(b), true, b, false}; }
    Val float_const(float f) { return {Kind::Float, cf(f), true, std::bit_cast<std::uint32_t>(f), false}; }
    Val lane_const(bool b) { return {Kind::Lane, m.const_bool(b), true, 0, b}; }
    static Val word(Id id) { return {Kind::Word, id, false, 0, false}; }
    static Val flt(Id id) { return {Kind::Float, id, false, 0, false}; }
    static Val lane(Id id) { return {Kind::Lane, id, false, 0, false}; }

    Id as_f(const Val& v) {
        if (v.kind == Kind::Float) return v.id;
        if (v.kind == Kind::Lane) {
            reject("a lane mask is used as a number");
            return cf(0.0f);
        }
        if (descriptor(v)) reject_descriptor_read(v);
        if (v.constant) return cf(std::bit_cast<float>(v.bits));
        if (auto it = to_float.find(v.id); it != to_float.end()) return it->second;
        const Id r = m.emit(spv::OpBitcast, t_f32, {v.id});
        to_float[v.id] = r;
        to_word[r] = v.id;
        return r;
    }
    Id as_u(const Val& v) {
        if (descriptor(v)) reject_descriptor_read(v);
        if (v.kind == Kind::Word) return v.id;
        if (v.kind == Kind::Lane) {
            reject("a lane mask is used as a number");
            return cu(0);
        }
        if (v.constant) return cu(v.bits);
        if (auto it = to_word.find(v.id); it != to_word.end()) return it->second;
        const Id r = m.emit(spv::OpBitcast, t_u32, {v.id});
        to_word[v.id] = r;
        to_float[r] = v.id;
        return r;
    }
    Id as_b(const Val& v) {
        if (v.kind != Kind::Lane) {
            reject("a number is used as a lane mask");
            return m.const_bool(false);
        }
        return v.id;
    }

    // Mask combinations, with their helper lanes' bits: exact where an exact
    // constant decides the result (false in an AND, true in an OR), otherwise
    // as exact as both operands.
    static Val helper_bits(Val r, const Val& a, const Val& b, bool deciding) {
        const auto decides = [&](const Val& x) { return x.constant && x.set == deciding && !x.helper_inexact; };
        r.helper_inexact = !decides(a) && !decides(b) && (a.helper_inexact || b.helper_inexact);
        return r;
    }
    Val band(const Val& a, const Val& b) { return helper_bits(band_value(a, b), a, b, false); }
    Val bor(const Val& a, const Val& b) { return helper_bits(bor_value(a, b), a, b, true); }
    Val bxor(const Val& a, const Val& b) {
        Val r = bxor_value(a, b);
        r.helper_inexact = a.helper_inexact || b.helper_inexact;
        return r;
    }
    Val bnot(const Val& a) {
        Val r = bnot_value(a);
        r.helper_inexact = a.helper_inexact;
        return r;
    }
    Val band_value(const Val& a, const Val& b) {
        if (a.constant) return a.set ? b : lane_const(false);
        if (b.constant) return b.set ? a : lane_const(false);
        if (a.id == b.id) return a;
        if (negated(a, b)) return lane_const(false);  // a & !a: a mask with the active lanes removed from itself
        const Id r = m.emit(spv::OpLogicalAnd, t_bool, {a.id, b.id});
        std::set<Id>& c = conjuncts[r];
        for (const Val* x : {&a, &b}) {
            c.insert(x->id);
            if (const auto it = conjuncts.find(x->id); it != conjuncts.end()) c.insert(it->second.begin(), it->second.end());
        }
        return lane(r);
    }
    static bool same_lane(const Val& a, const Val& b) {
        return a.kind == Kind::Lane && b.kind == Kind::Lane && a.constant == b.constant && (a.constant ? a.set == b.set : a.id == b.id);
    }
    // One is the other's LogicalNot.
    bool negated(const Val& a, const Val& b) const {
        if (a.constant || b.constant) return a.constant && b.constant && a.set != b.set;
        const auto na = negation_of.find(a.id), nb = negation_of.find(b.id);
        return (na != negation_of.end() && na->second == b.id) || (nb != negation_of.end() && nb->second == a.id);
    }
    // Wherever `skip` is clear, `now` equals `before`: both lie within it, or
    // `now` is `before` with only masks within `skip` removed (an AND with
    // their negations). A region GCN skips because `skip` is clear in every
    // lane leaves such a mask as the lift computes it.
    bool same_where_clear(const Val& now, const Val& before, const Val& skip) const {
        if (same_lane(now, before)) return true;
        if (implies(now, skip) && implies(before, skip)) return true;
        if (now.constant || before.constant) return false;
        const auto it = conjuncts.find(now.id);
        if (it == conjuncts.end() || !it->second.count(before.id)) return false;
        const auto bc = conjuncts.find(before.id);
        for (const Id c : it->second) {
            if (c == before.id || (bc != conjuncts.end() && bc->second.count(c)) || conjuncts.count(c)) continue;  // `before`, or an AND node
            const auto neg = negation_of.find(c);
            if (neg == negation_of.end() || !implies(lane(neg->second), skip)) return false;
        }
        return true;
    }
    // Every pixel whose `e` bit is set has its `g` bit set.
    bool implies(const Val& e, const Val& g) const {
        if (g.constant) return g.set || (e.constant && !e.set);
        if (assumed.count(g.id)) return true;  // set wherever the lift is (a divergent loop's body)
        if (e.constant) return !e.set;
        if (e.id == g.id) return true;
        const auto it = conjuncts.find(e.id);
        return it != conjuncts.end() && it->second.count(g.id) != 0;
    }
    Val bor_value(const Val& a, const Val& b) {
        if (a.constant) return a.set ? lane_const(true) : b;
        if (b.constant) return b.set ? lane_const(true) : a;
        return lane(m.emit(spv::OpLogicalOr, t_bool, {a.id, b.id}));
    }
    Val bnot_value(const Val& a) {
        if (a.constant) return lane_const(!a.set);
        const Id r = m.emit(spv::OpLogicalNot, t_bool, {a.id});
        // Each is the other's negation. (A double negation stays as emitted:
        // the lifts made before keep their exact modules.)
        negation_of[r] = a.id;
        negation_of.emplace(a.id, r);
        return lane(r);
    }
    Val bxor_value(const Val& a, const Val& b) {
        if (a.constant) return a.set ? bnot(b) : b;
        if (b.constant) return b.set ? bnot(a) : a;
        return lane(m.emit(spv::OpLogicalNotEqual, t_bool, {a.id, b.id}));
    }

    Id fadd(Id a, Id b) { return m.emit(spv::OpFAdd, t_f32, {a, b}); }
    Id fsub(Id a, Id b) { return m.emit(spv::OpFSub, t_f32, {a, b}); }
    Id fmul(Id a, Id b) { return m.emit(spv::OpFMul, t_f32, {a, b}); }
    Id fdiv(Id a, Id b) { return m.emit(spv::OpFDiv, t_f32, {a, b}); }
    Id fneg(Id a) { return m.emit(spv::OpFNegate, t_f32, {a}); }
    Id fsel(Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_f32, {c, a, b}); }
    Id ext(spv::Glsl op, const std::vector<Id>& args, Id type = 0) { return m.ext_inst(type ? type : t_f32, op, args); }
    Id feq(Id a, Id b) { return m.emit(spv::OpFOrdEqual, t_bool, {a, b}); }
    Id fge(Id a, Id b) { return m.emit(spv::OpFOrdGreaterThanEqual, t_bool, {a, b}); }
    Id extract(Id type, Id v, std::uint32_t k) { return m.emit(spv::OpCompositeExtract, type, {v, k}); }
    Id ibin(spv::Op op, Id a, Id b) { return m.emit(op, t_u32, {a, b}); }
    Id to_i(Id a) { return m.emit(spv::OpBitcast, t_i32, {a}); }
    Id from_i(Id a) { return m.emit(spv::OpBitcast, t_u32, {a}); }
    Id shift(spv::Op op, Id a, Id b) { return ibin(op, a, ibin(spv::OpBitwiseAnd, b, cu(31))); }  // the translator's shl/shr/sar
    Id sext24(Id a) { return from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(a), cu(0), cu(24)})); }
    Id zext24(Id a) { return ibin(spv::OpBitwiseAnd, a, cu(0xffffff)); }
    Id is_nan(Id a) { return m.emit(spv::OpIsNan, t_bool, {a}); }
    Id ieq(Id a, Id b) { return m.emit(spv::OpIEqual, t_bool, {a, b}); }

    // ---- registers -------------------------------------------------------------------
    int user_sgpr_count() const { return std::min(static_cast<int>((opt.rsrc2 >> 1) & 0x1f), 16); }

    Val initial(int key) {
        if (key < user_sgpr_count()) {
            const Id p = m.access_chain(p_uni_u32, ubo_var, {cu(2), cu(static_cast<std::uint32_t>(key / 4)), cu(static_cast<std::uint32_t>(key % 4))});
            return word(m.load(t_u32, p));
        }
        if (key >= 256 && opt.stage == Stage::Vertex) {
            // The vertex index in v0, the instance index in v1 up to VGPR_COMP_CNT (translate.cpp, run()).
            const int comp = static_cast<int>((opt.rsrc1 >> 24) & 3);
            if (key == 256) {
                if (!in_vertex_index) in_vertex_index = builtin(p_in_u32, spv::BiVertexIndex);
                return word(m.load(t_u32, in_vertex_index));
            }
            if (key - 256 <= comp) {
                if (!in_instance_index) in_instance_index = builtin(p_in_u32, spv::BiInstanceIndex);
                return word(m.load(t_u32, in_instance_index));
            }
            return word_const(0);
        }
        if (key >= 256) {
            // The PS inputs the translator places in VGPRs (translate.cpp, run()).
            int v = 0;
            const std::uint32_t ena = opt.ps_input_ena;
            for (const auto [bit, n] : {std::pair{1u, 2}, {2u, 2}, {4u, 2}, {8u, 3}, {0x10u, 2}, {0x20u, 2}, {0x40u, 2}, {0x80u, 1}}) {
                if (ena & bit) v += n;
            }
            for (int k = 0; k < 4; ++k) {
                if (!(ena & (0x100u << k))) continue;
                if (key - 256 == v++) {
                    if (!in_frag_coord) {
                        in_frag_coord = builtin(p_in_v4f, spv::BiFragCoord);
                    }
                    Id c = extract(t_f32, m.load(t_v4f, in_frag_coord), static_cast<std::uint32_t>(k));
                    if (k == 3) c = fdiv(cf(1.0f), c);
                    return flt(c);
                }
            }
            if (ena & 0x1000) {
                if (key - 256 == v++) {
                    if (!in_front_facing) in_front_facing = builtin(p_in_bool, spv::BiFrontFacing);
                    return word(m.emit(spv::OpSelect, t_u32, {m.load(t_bool, in_front_facing), cu(0xffffffffu), cu(0)}));
                }
            }
        }
        return word_const(0);
    }
    // Loads of inputs, uniforms and parameters go to the entry block, so they
    // dominate every block that reads them.
    struct EntryScope {
        spv::Module& mod;
        bool prev;
        explicit EntryScope(spv::Module& module) : mod(module), prev(module.emitting_to_entry()) { mod.emit_to_entry(true); }
        ~EntryScope() { mod.emit_to_entry(prev); }
    };
    Val initial_value(int key) {
        auto init = initials.find(key);
        if (init == initials.end()) {
            EntryScope entry(m);
            init = initials.emplace(key, initial(key)).first;
            if (key < 256 && !init->second.constant) uniform_ids.insert(init->second.id);  // user data: every lane alike
        }
        return init->second;
    }
    Val& value(int key) {
        if (key >= 256 && key < kKeyCell && poisoned.count(key)) {
            reject(key_name(key) + " holds a word in one lane since a v_writelane_b32 and is read as this pixel's");
        }
        auto it = reg.find(key);
        if (it == reg.end()) it = reg.emplace(key, initial_value(key)).first;
        note_read(it->second);
        note_exact_read(key);
        return it->second;
    }
    static int scalar_key(std::uint16_t code) {
        if (code < 104) return code;
        if (code >= 112 && code < 124) return 200 + code - 112;
        if (code == kM0) return kKeyM0;
        return -1;
    }
    bool holds_lane(int key) const { return lanes.count(key) || (key > 0 && key < 104 && lanes.count(key - 1)); }

    Val read(std::uint16_t code, const Inst& in) {
        if (code >= 256) return value(code);
        if (const int key = scalar_key(code); key >= 0) {
            if (holds_lane(key)) reject(key_name(key) + " holds a lane mask and is read as a word");
            if (poisoned.count(key)) reject(key_name(key) + " is read after an if that left a lane mask in it on one arm only");
            return value(key);
        }
        if (code == kLiteral) return word_const(in.literal);
        if (code == 128) return word_const(0);
        if (code >= 129 && code <= 192) return word_const(code - 128u);
        if (code >= 193 && code <= 208) return word_const(static_cast<std::uint32_t>(-static_cast<int>(code - 192)));
        if (code >= 108 && code <= 111) return word_const(0);  // tba/tma, as the translator
        static const float kHalf[8] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f};
        if (code >= 240 && code <= 247) return float_const(kHalf[code - 240]);
        reject("unsupported operand " + operand_name(code, in.literal));
        return word_const(0);
    }
    void write_scalar(std::uint16_t code, const Val& v) {
        if (v.kind == Kind::Lane) {
            reject("a lane mask is stored as a word");
            return;
        }
        if (code == kVccLo || code == kVccHi) {
            // The words of VCC are not tracked: its lane bit is unknown until
            // a comparison writes it again.
            lanes.erase(kKeyVcc);
            note_region_write(kKeyVcc);
            note += " vcc:unknown";
            return;
        }
        if (code == kExecLo || code == kExecHi) {
            reject("a word written into EXEC");
            return;
        }
        if (code >= 108 && code <= 111) return;
        const int key = scalar_key(code);
        if (key < 0) {
            reject("unsupported scalar destination " + operand_name(code));
            return;
        }
        lanes.erase(key);
        if (key > 0 && key < 104) lanes.erase(key - 1);
        poisoned.erase(key);
        reg[key] = v;
        note_result(v);
        note_region_write(key);
        note += " " + key_name(key) + (v.kind == Kind::Float ? ":f32" : ":u32");
    }
    Val exec_lane() {
        if (wqm.active) reject("EXEC is used inside a whole-quad block, where it is no per-pixel mask");
        auto it = lanes.find(kKeyExec);
        if (it == lanes.end()) {
            reject("EXEC is not a known per-pixel mask");
            return lane_const(true);
        }
        return note_read(known(it->second));
    }
    Val read_lane(std::uint16_t code) {
        int key = -1;
        if (code == kExecLo && wqm.active) reject("EXEC is read inside a whole-quad block, where it is no per-pixel mask");
        if (code == kExecLo) key = kKeyExec;
        else if (code == kVccLo) key = kKeyVcc;
        else if (code < 104) key = code;
        else if (code == 128) return lane_const(false);
        else if (code == 193) return lane_const(true);  // -1: every lane
        if (key >= 0) {
            if (poisoned.count(key)) reject(key_name(key) + " is read as a lane mask after an if that left one in it on one arm only");
            if (auto it = lanes.find(key); it != lanes.end()) return note_read(known(it->second));
            reject((key == kKeyVcc ? std::string("vcc") : operand_name(code)) + " is read as a lane mask but does not hold one");
        } else {
            reject("unsupported 64-bit mask operand " + operand_name(code));
        }
        return lane_const(false);
    }
    void write_lane(std::uint16_t code, const Val& v) {
        if (v.kind != Kind::Lane) {
            reject("a word written into a lane mask");
            return;
        }
        if (!lane_write_exact(code, v)) return;
        int key = -1;
        if (code == kExecLo) {
            // Skipped, a region leaves EXEC as it was; where a set EXEC lies within
            // EXEC at the region's start, the pixels it was clear for agree.
            for (const Region* r : open_regions) {
                if (exec_write_ok || implies(v, r->exec_in)) continue;
                reject("EXEC inside the region at " + hex_offset(r->start) + " is set beyond EXEC where the region's writes begin");
                break;
            }
            key = kKeyExec;
        } else if (code == kVccLo) {
            key = kKeyVcc;
        } else if (code < 103) {
            key = code;
            reg.erase(code);
            reg.erase(code + 1);
            poisoned.erase(code);
            poisoned.erase(code + 1);
            lanes.erase(code - 1);
            lanes.erase(code + 1);
        } else {
            reject("unsupported 64-bit mask destination " + operand_name(code));
            return;
        }
        lanes[key] = v;
        note_result(v);
        note_region_write(key);
        if (code < 104) note_region_write(code + 1);
        note += " " + (key == kKeyExec ? std::string("exec") : key == kKeyVcc ? std::string("vcc") : operand_name(code)) + ":lane";
    }
    // A VGPR write under EXEC.
    void write_v(int idx, const Val& v) {
        if (v.kind == Kind::Lane) {
            reject("a lane mask stored into a VGPR");
            return;
        }
        if (wqm.active) {
            whole_quad_write(idx, v);
            return;
        }
        const Val e = exec_lane();
        const int key = 256 + idx;
        const Where read = reads;  // where the value is exact; the old value's read below does not count
        if (e.constant) {
            if (e.set) {
                reg[key] = v;
                reg[key].helper_inexact = v.helper_inexact || inst_helper_inexact;
                poisoned.erase(key);  // written whole: this pixel's word again
                note_result(v);
                note_exact_write(key, read, e);
            } else if (e.helper_inexact) {
                // GCN may write the helper lanes the lift leaves alone.
                const auto it = reg.find(key);
                Val old = it != reg.end() ? it->second : initial_value(key);
                old.helper_inexact = true;
                reg[key] = old;
            }
            note += " v" + std::to_string(idx) + (e.set ? (v.kind == Kind::Float ? ":f32" : ":u32") : ":masked-off");
            return;
        }
        const Val old = value(key);
        reads = read;
        note_exact_write(key, read, e);
        ++masked_writes;
        if (v.kind == Kind::Float || old.kind == Kind::Float) {
            reg[key] = flt(fsel(e.id, as_f(v), as_f(old)));
        } else {
            reg[key] = word(m.emit(spv::OpSelect, t_u32, {e.id, as_u(v), as_u(old)}));
        }
        reg[key].helper_inexact = v.helper_inexact || inst_helper_inexact;  // the old value and EXEC were read: in the flag
        if (uniform(v)) masked_uniform[reg[key].id] = {e, v};
        note_result(reg[key]);
        note += " v" + std::to_string(idx) + ":select";
    }
    void note_region_write(int key) {
        for (Region* r : open_regions) r->scalar_writes.insert(key);
    }
    void set_scc_mask(const Val& v) {
        scc_mask = v;
        scc_uniform = false;
        scc_valid = true;
        note_region_write(kKeyScc);
    }
    void set_scc_bool(const Val& b) {  // a scalar compare: SCC is the condition itself
        scc_mask = b;
        scc_uniform = true;
        scc_valid = true;
        note_region_write(kKeyScc);
    }

    // ---- interface -------------------------------------------------------------------
    Id builtin(Id ptr_type, spv::BuiltIn bi) {
        const Id var = m.global_variable(ptr_type, spv::ScInput);
        m.decorate(var, spv::DecBuiltIn, {static_cast<std::uint32_t>(bi)});
        if (opt.stage == Stage::Pixel && bi != spv::BiFragCoord) m.decorate(var, spv::DecFlat);  // vertex inputs take none
        interface.push_back(var);
        return var;
    }
    Id attr(int a) {
        if (auto it = attr_loads.find(a); it != attr_loads.end()) return it->second;
        Id var;
        if (auto it = in_params.find(a); it != in_params.end()) {
            var = it->second;
        } else {
            const std::uint32_t location = static_cast<std::size_t>(a) < opt.ps_input_map.size() ? opt.ps_input_map[a] : static_cast<std::uint32_t>(a);
            if (auto at = in_location_vars.find(location); at != in_location_vars.end()) {
                var = at->second;  // as translate.cpp in_param: one Input variable a location
            } else {
                var = m.global_variable(p_in_v4f, spv::ScInput);
                m.decorate(var, spv::DecLocation, {location});
                if ((opt.ps_flat_mask >> a) & 1) m.decorate(var, spv::DecFlat);
                m.name(var, "attr" + std::to_string(a));
                interface.push_back(var);
                in_location_vars[location] = var;
            }
            in_params[a] = var;
        }
        // Inputs do not change: one load per attribute serves every read.
        EntryScope entry(m);
        return attr_loads[a] = m.load(t_v4f, var);
    }
    Id out_position() {  // translate.cpp exp, pos0
        if (!out_pos) {
            out_pos = m.global_variable(p_out_v4f, spv::ScOutput);
            m.decorate(out_pos, spv::DecBuiltIn, {static_cast<std::uint32_t>(spv::BiPosition)});
            if (opt.invariant_position) m.decorate(out_pos, spv::DecInvariant);
            interface.push_back(out_pos);
        }
        return out_pos;
    }
    Id out_param(int n) {
        if (auto it = out_params.find(n); it != out_params.end()) return it->second;
        const Id var = m.global_variable(p_out_v4f, spv::ScOutput);
        m.decorate(var, spv::DecLocation, {static_cast<std::uint32_t>(n)});
        m.name(var, (opt.stage == Stage::Pixel ? "mrt" : "param") + std::to_string(n));
        interface.push_back(var);
        return out_params[n] = var;
    }
    std::vector<Id> param_outputs(int n) {  // translate.cpp param_outputs
        if (opt.stage != Stage::Vertex || !opt.link_outputs) return {out_param(n)};
        std::vector<Id> vars;
        for (std::size_t k = 0; k < opt.output_links.size(); ++k) {
            if (opt.output_links[k] == n) vars.push_back(out_param(static_cast<int>(k)));
        }
        return vars;
    }

    // ---- instructions ----------------------------------------------------------------
    struct Mods {
        std::uint8_t abs = 0, neg = 0, omod = 0;
        bool clamp = false;
    };
    Id src_f(const Val& raw, int k, const Mods& md) {
        Id v = as_f(raw);
        if (md.abs & (1 << k)) v = ext(spv::GlslFAbs, {v});
        if (md.neg & (1 << k)) v = fneg(v);
        return v;
    }
    Val result_f(Id v, const Mods& md) {
        if (md.omod == 1) v = fmul(v, cf(2.0f));
        if (md.omod == 2) v = fmul(v, cf(4.0f));
        if (md.omod == 3) v = fmul(v, cf(0.5f));
        if (md.clamp) v = ext(spv::GlslFClamp, {v, cf(0.0f), cf(1.0f)});
        return flt(v);
    }
    // One 16-bit half, decoded exactly as the translator decodes it (gcn/half.h).
    Id unpack_half(Id h) { return emit_unpack_half16(m, {t_bool, t_u32, t_i32, t_f32}, h); }
    Id mul_legacy(Id a, Id b) {  // as the translator writes it (translate.cpp mul_legacy)
        if (legacy_mul_min_form()) return fsel(feq(ext(spv::GlslNMin, {ext(spv::GlslFAbs, {a}), ext(spv::GlslFAbs, {b})}), cf(0.0f)), cf(0.0f), fmul(a, b));
        const Id zero = m.emit(spv::OpLogicalOr, t_bool, {feq(a, cf(0.0f)), feq(b, cf(0.0f))});
        return fsel(zero, cf(0.0f), fmul(a, b));
    }
    Id clamp_inf(Id v) {
        const Id inf = m.emit(spv::OpIsInf, t_bool, {v});
        return fsel(inf, fmul(ext(spv::GlslFSign, {v}), cf(3.4028235e38f)), v);
    }

    // `mask`: v_cndmask's select or the carry in, from src2 (VOP3); VCC otherwise.
    // `carry_out`: the pair the carry-out ops write (VOP3 sdst, else VCC).
    bool vop2(std::uint32_t op, const Inst& in, const Val& s0, const Val& s1, const Mods& md, const Val* mask, std::uint16_t carry_out) {
        const auto F0 = [&] { return src_f(s0, 0, md); };
        const auto F1 = [&] { return src_f(s1, 1, md); };
        const auto out_f = [&](Id v) { write_v(in.dst, result_f(v, md)); return true; };
        const auto out_u = [&](Id v) { write_v(in.dst, word(v)); return true; };
        const auto carry = [&](Id c) {  // the translator's ballot of carry & EXEC
            Val bit = lane(c);
            bit.helper_inexact = inst_helper_inexact;
            write_lane(carry_out, band(bit, exec_lane()));
        };
        const auto vdst = [&] { return as_f(value(256 + in.dst)); };
        switch (op) {
        case 0: {  // v_cndmask_b32
            const Val bit = mask ? *mask : read_lane(kVccLo);
            if (bit.kind != Kind::Lane) return true;
            if (bit.constant) {
                write_v(in.dst, bit.set ? s1 : s0);
            } else if (s0.kind == Kind::Float || s1.kind == Kind::Float) {
                write_v(in.dst, flt(fsel(bit.id, as_f(s1), as_f(s0))));
            } else {
                write_v(in.dst, word(m.emit(spv::OpSelect, t_u32, {bit.id, as_u(s1), as_u(s0)})));
            }
            return true;
        }
        case 3: return out_f(fadd(F0(), F1()));
        case 4: return out_f(fsub(F0(), F1()));
        case 5: return out_f(fsub(F1(), F0()));
        case 6: return out_f(fadd(mul_legacy(F0(), F1()), vdst()));
        case 7: return out_f(mul_legacy(F0(), F1()));
        case 8: return out_f(fmul(F0(), F1()));
        case 9: return out_u(ibin(spv::OpIMul, sext24(as_u(s0)), sext24(as_u(s1))));  // v_mul_i32_i24
        case 10: {                                                                    // v_mul_hi_i32_i24
            const Id a = to_i(sext24(as_u(s0))), b = to_i(sext24(as_u(s1)));
            return out_u(from_i(extract(t_i32, m.emit(spv::OpSMulExtended, m.type_struct({t_i32, t_i32}), {a, b}), 1)));
        }
        case 11: return out_u(ibin(spv::OpIMul, zext24(as_u(s0)), zext24(as_u(s1))));  // v_mul_u32_u24
        case 12: {                                                                    // v_mul_hi_u32_u24
            const Id a = zext24(as_u(s0)), b = zext24(as_u(s1));
            return out_u(extract(t_u32, m.emit(spv::OpUMulExtended, m.type_struct({t_u32, t_u32}), {a, b}), 1));
        }
        case 13: case 15: return out_f(ext(spv::GlslNMin, {F0(), F1()}));
        case 14: case 16: return out_f(ext(spv::GlslNMax, {F0(), F1()}));
        case 17: return out_u(from_i(ext(spv::GlslSMin, {to_i(as_u(s0)), to_i(as_u(s1))}, t_i32)));
        case 18: return out_u(from_i(ext(spv::GlslSMax, {to_i(as_u(s0)), to_i(as_u(s1))}, t_i32)));
        case 19: return out_u(ext(spv::GlslUMin, {as_u(s0), as_u(s1)}, t_u32));
        case 20: return out_u(ext(spv::GlslUMax, {as_u(s0), as_u(s1)}, t_u32));
        case 21: return out_u(shift(spv::OpShiftRightLogical, as_u(s0), as_u(s1)));
        case 22: return out_u(shift(spv::OpShiftRightLogical, as_u(s1), as_u(s0)));
        case 23: return out_u(shift(spv::OpShiftRightArithmetic, as_u(s0), as_u(s1)));
        case 24: return out_u(shift(spv::OpShiftRightArithmetic, as_u(s1), as_u(s0)));
        case 25: return out_u(shift(spv::OpShiftLeftLogical, as_u(s0), as_u(s1)));
        case 26: return out_u(shift(spv::OpShiftLeftLogical, as_u(s1), as_u(s0)));
        case 27: write_v(in.dst, word(m.emit(spv::OpBitwiseAnd, t_u32, {as_u(s0), as_u(s1)}))); return true;
        case 28: write_v(in.dst, word(m.emit(spv::OpBitwiseOr, t_u32, {as_u(s0), as_u(s1)}))); return true;
        case 29: write_v(in.dst, word(m.emit(spv::OpBitwiseXor, t_u32, {as_u(s0), as_u(s1)}))); return true;
        case 30: {  // v_bfm_b32
            const Id ones = ibin(spv::OpISub, shift(spv::OpShiftLeftLogical, cu(1), ibin(spv::OpBitwiseAnd, as_u(s0), cu(31))), cu(1));
            return out_u(shift(spv::OpShiftLeftLogical, ones, ibin(spv::OpBitwiseAnd, as_u(s1), cu(31))));
        }
        case 31: return out_f(ext(spv::GlslFma, {F0(), F1(), vdst()}));                   // v_mac_f32
        case 32: return out_f(ext(spv::GlslFma, {F0(), as_f(word_const(in.literal)), F1()}));  // v_madmk_f32
        case 33: return out_f(ext(spv::GlslFma, {F0(), F1(), as_f(word_const(in.literal))}));  // v_madak_f32
        case 34: return out_u(ibin(spv::OpIAdd, m.emit(spv::OpBitCount, t_u32, {as_u(s0)}), as_u(s1)));  // v_bcnt_u32_b32
        case 37: case 38: case 39: {  // v_add_i32 / v_sub_i32 / v_subrev_i32 with carry out
            Id a = as_u(s0), b = as_u(s1);
            if (op == 39) std::swap(a, b);
            const Id r = m.emit(op == 37 ? spv::OpIAddCarry : spv::OpISubBorrow, m.type_struct({t_u32, t_u32}), {a, b});
            const Id c = m.emit(spv::OpINotEqual, t_bool, {extract(t_u32, r, 1), cu(0)});
            write_v(in.dst, word(extract(t_u32, r, 0)));
            carry(c);
            return true;
        }
        case 40: case 41: case 42: {  // v_addc_u32 / v_subb_u32 / v_subbrev_u32
            const Val cin = mask ? *mask : read_lane(kVccLo);
            if (cin.kind != Kind::Lane) return true;
            const Id c_in = m.emit(spv::OpSelect, t_u32, {cin.id, cu(1), cu(0)});
            Id a = as_u(s0), b = as_u(s1);
            if (op == 42) std::swap(a, b);
            const Id st = m.type_struct({t_u32, t_u32});
            const spv::Op o = op == 40 ? spv::OpIAddCarry : spv::OpISubBorrow;
            const Id r1 = m.emit(o, st, {a, b});
            const Id r2 = m.emit(o, st, {extract(t_u32, r1, 0), c_in});
            const Id c = m.emit(spv::OpINotEqual, t_bool, {ibin(spv::OpBitwiseOr, extract(t_u32, r1, 1), extract(t_u32, r2, 1)), cu(0)});
            write_v(in.dst, word(extract(t_u32, r2, 0)));
            carry(c);
            return true;
        }
        case 43: return out_f(ext(spv::GlslLdexp, {F0(), to_i(as_u(s1))}));  // v_ldexp_f32
        case 47: {  // v_cvt_pkrtz_f16_f32
            const Id lo = F0(), hi = F1();
            const Id v2 = m.emit(spv::OpCompositeConstruct, t_v2f, {lo, hi});
            const Id packed = native_rtz ? m.emit(spv::OpBitcast, t_u32, {m.emit(spv::OpFConvert, t_v2h, {v2})})
                                         : ext(spv::GlslPackHalf2x16, {v2}, t_u32);
            // A compressed export of it in this block rounds these, as the translator does (translate.cpp exp).
            if (export_rtz_on()) packed_pairs[packed] = {lo, hi, m.blocks()};
            write_v(in.dst, word(packed));
            return true;
        }
        case 48: {  // v_cvt_pk_u16_u32
            const Id lo = ext(spv::GlslUMin, {as_u(s0), cu(0xffff)}, t_u32);
            const Id hi = ext(spv::GlslUMin, {as_u(s1), cu(0xffff)}, t_u32);
            return out_u(ibin(spv::OpBitwiseOr, lo, shift(spv::OpShiftLeftLogical, hi, cu(16))));
        }
        case 49: {  // v_cvt_pk_i16_i32
            const Id lo = from_i(ext(spv::GlslSClamp, {to_i(as_u(s0)), m.const_i32(-32768), m.const_i32(32767)}, t_i32));
            const Id hi = from_i(ext(spv::GlslSClamp, {to_i(as_u(s1)), m.const_i32(-32768), m.const_i32(32767)}, t_i32));
            return out_u(ibin(spv::OpBitwiseOr, ibin(spv::OpBitwiseAnd, lo, cu(0xffff)), shift(spv::OpShiftLeftLogical, hi, cu(16))));
        }
        default: return false;
        }
    }

    bool vop1(std::uint32_t op, const Inst& in, const Val& s0, const Mods& md) {
        const auto F0 = [&] { return src_f(s0, 0, md); };
        const auto out_f = [&](Id v) { write_v(in.dst, result_f(v, md)); return true; };
        const auto out_u = [&](Id v) { write_v(in.dst, word(v)); return true; };
        const auto f_to_i = [&](Id x) { return from_i(m.emit(spv::OpConvertFToS, t_i32, {x})); };
        switch (op) {
        case 0: return true;
        case 1: write_v(in.dst, s0); return true;
        case 5: return out_f(m.emit(spv::OpConvertSToF, t_f32, {to_i(as_u(s0))}));  // v_cvt_f32_i32
        case 6: return out_f(m.emit(spv::OpConvertUToF, t_f32, {as_u(s0)}));        // v_cvt_f32_u32
        case 7: {  // v_cvt_u32_f32 (saturating, NaN -> 0)
            const Id x = F0();
            const Id c = ext(spv::GlslFClamp, {x, cf(0.0f), cf(4294967040.0f)});
            const Id nan = is_nan(x);
            return out_u(m.emit(spv::OpSelect, t_u32, {nan, cu(0), m.emit(spv::OpConvertFToU, t_u32, {c})}));
        }
        case 8: {  // v_cvt_i32_f32
            const Id x = F0();
            const Id c = ext(spv::GlslFClamp, {x, cf(-2147483648.0f), cf(2147483520.0f)});
            const Id nan = is_nan(x);
            return out_u(m.emit(spv::OpSelect, t_u32, {nan, cu(0), f_to_i(c)}));
        }
        case 10: {  // v_cvt_f16_f32
            const Id v2 = m.emit(spv::OpCompositeConstruct, t_v2f, {F0(), cf(0.0f)});
            return out_u(ibin(spv::OpBitwiseAnd, ext(spv::GlslPackHalf2x16, {v2}, t_u32), cu(0xffff)));
        }
        case 11:  // v_cvt_f32_f16
            return out_f(unpack_half(ibin(spv::OpBitwiseAnd, as_u(s0), cu(0xffff))));
        case 12: return out_u(f_to_i(ext(spv::GlslFloor, {fadd(F0(), cf(0.5f))})));  // v_cvt_rpi_i32_f32
        case 13: return out_u(f_to_i(ext(spv::GlslFloor, {F0()})));                   // v_cvt_flr_i32_f32
        case 14: {                                                                    // v_cvt_off_f32_i4
            const Id nib = m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(as_u(s0)), cu(0), cu(4)});
            return out_f(fmul(m.emit(spv::OpConvertSToF, t_f32, {nib}), cf(1.0f / 16.0f)));
        }
        case 17: case 18: case 19: case 20:  // v_cvt_f32_ubyteN
            return out_f(m.emit(spv::OpConvertUToF, t_f32, {m.emit(spv::OpBitFieldUExtract, t_u32, {as_u(s0), cu(8 * (op - 17)), cu(8)})}));
        case 53: return out_f(ext(spv::GlslSin, {fmul(F0(), cf(6.2831853f))}));  // input in revolutions
        case 54: return out_f(ext(spv::GlslCos, {fmul(F0(), cf(6.2831853f))}));
        case 55: return out_u(m.emit(spv::OpNot, t_u32, {as_u(s0)}));
        case 56: return out_u(m.emit(spv::OpBitReverse, t_u32, {as_u(s0)}));
        case 57: {  // v_ffbh_u32
            const Id x = as_u(s0);
            const Id msb = ext(spv::GlslFindUMsb, {x}, t_u32);
            return out_u(m.emit(spv::OpSelect, t_u32, {ieq(x, cu(0)), cu(0xffffffffu), ibin(spv::OpISub, cu(31), msb)}));
        }
        case 58: return out_u(ext(spv::GlslFindILsb, {as_u(s0)}, t_u32));
        case 59: {  // v_ffbh_i32
            const Id x = as_u(s0);
            const Id msb = from_i(ext(spv::GlslFindSMsb, {to_i(x)}, t_i32));
            const Id none = m.emit(spv::OpLogicalOr, t_bool, {ieq(x, cu(0)), ieq(x, cu(0xffffffffu))});
            return out_u(m.emit(spv::OpSelect, t_u32, {none, cu(0xffffffffu), ibin(spv::OpISub, cu(31), msb)}));
        }
        case 32: return out_f(ext(spv::GlslFract, {F0()}));
        case 33: return out_f(ext(spv::GlslTrunc, {F0()}));
        case 34: return out_f(ext(spv::GlslCeil, {F0()}));
        case 35: return out_f(ext(spv::GlslRoundEven, {F0()}));
        case 36: return out_f(ext(spv::GlslFloor, {F0()}));
        case 37: return out_f(ext(spv::GlslExp2, {F0()}));
        case 38: case 39: return out_f(ext(spv::GlslLog2, {F0()}));
        case 40: return out_f(clamp_inf(fdiv(cf(1.0f), F0())));
        case 41: case 42: case 43: return out_f(fdiv(cf(1.0f), F0()));
        case 44: return out_f(clamp_inf(ext(spv::GlslInverseSqrt, {F0()})));
        case 45: case 46: return out_f(ext(spv::GlslInverseSqrt, {F0()}));
        case 51: return out_f(ext(spv::GlslSqrt, {F0()}));
        default: return false;
        }
    }

    Id vcmp_cond(std::uint32_t op, const Val& s0, const Val& s1, const Mods& md) {
        const std::uint32_t hi = (op >> 5) & 7, lo = op & 0xf;
        if (hi == 0 || hi == 2) {
            const Id a = src_f(s0, 0, md), b = src_f(s1, 1, md);
            const auto isnan = [&](Id x) { return m.emit(spv::OpIsNan, t_bool, {x}); };
            const auto lor = [&](Id x, Id y) { return m.emit(spv::OpLogicalOr, t_bool, {x, y}); };
            const auto land = [&](Id x, Id y) { return m.emit(spv::OpLogicalAnd, t_bool, {x, y}); };
            const auto lnot = [&](Id x) { return m.emit(spv::OpLogicalNot, t_bool, {x}); };
            const Id unord = lor(isnan(a), isnan(b));
            switch (lo) {
            case 0: return m.const_bool(false);
            case 1: return m.emit(spv::OpFOrdLessThan, t_bool, {a, b});
            case 2: return feq(a, b);
            case 3: return m.emit(spv::OpFOrdLessThanEqual, t_bool, {a, b});
            case 4: return m.emit(spv::OpFOrdGreaterThan, t_bool, {a, b});
            case 5: return land(lnot(unord), lnot(feq(a, b)));
            case 6: return fge(a, b);
            case 7: return lnot(unord);
            case 8: return unord;
            case 9: return lnot(fge(a, b));
            case 10: return lor(unord, feq(a, b));
            case 11: return lnot(m.emit(spv::OpFOrdGreaterThan, t_bool, {a, b}));
            case 12: return lnot(m.emit(spv::OpFOrdLessThanEqual, t_bool, {a, b}));
            case 13: return lnot(feq(a, b));
            case 14: return lnot(m.emit(spv::OpFOrdLessThan, t_bool, {a, b}));
            default: return m.const_bool(true);
            }
        }
        if (hi != 4 && hi != 6) {
            reject("64-bit or class comparisons are not lifted");
            return m.const_bool(false);
        }
        const bool is_signed = hi == 4;
        const Id a = as_u(s0), b = as_u(s1);
        switch (lo) {
        case 0: return m.const_bool(false);
        case 1: return m.emit(is_signed ? spv::OpSLessThan : spv::OpULessThan, t_bool, {a, b});
        case 2: return m.emit(spv::OpIEqual, t_bool, {a, b});
        case 3: return m.emit(is_signed ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {a, b});
        case 4: return m.emit(is_signed ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {a, b});
        case 5: return m.emit(spv::OpINotEqual, t_bool, {a, b});
        case 6: return m.emit(is_signed ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {a, b});
        default: return m.const_bool(true);
        }
    }
    void vcmp(std::uint32_t op, const Val& s0, const Val& s1, const Mods& md, std::uint16_t dst_pair) {
        // The ballot the translator writes is the comparison AND this lane's EXEC.
        const Id cond = vcmp_cond(op, s0, s1, md);
        Val bit = lane(cond);
        bit.helper_inexact = inst_helper_inexact;  // from the operands
        const Val e = exec_lane();
        const Val c = band(bit, e);
        note_compare_split(c, e, op, s0, s1, md, cond);
        write_lane(dst_pair, c);
        if (op & 0x10) write_lane(kExecLo, c);
    }

    void cube_ops(std::uint32_t op, const Inst& in, const Val& s0, const Val& s1, const Val& s2, const Mods& md) {
        const Id x = src_f(s0, 0, md), y = src_f(s1, 1, md), z = src_f(s2, 2, md);
        const Id ax = ext(spv::GlslFAbs, {x}), ay = ext(spv::GlslFAbs, {y}), az = ext(spv::GlslFAbs, {z});
        const Id z_major = m.emit(spv::OpLogicalAnd, t_bool, {fge(az, ax), fge(az, ay)});
        const Id y_major = m.emit(spv::OpLogicalAnd, t_bool, {m.emit(spv::OpLogicalNot, t_bool, {z_major}), fge(ay, ax)});
        const Id zero = cf(0.0f);
        Id r;
        switch (op) {
        case 0x144: r = fsel(z_major, fsel(fge(z, zero), cf(4.0f), cf(5.0f)), fsel(y_major, fsel(fge(y, zero), cf(2.0f), cf(3.0f)), fsel(fge(x, zero), cf(0.0f), cf(1.0f)))); break;
        case 0x145: r = fsel(z_major, fsel(fge(z, zero), x, fneg(x)), fsel(y_major, x, fsel(fge(x, zero), fneg(z), z))); break;
        case 0x146: r = fsel(z_major, fneg(y), fsel(y_major, fsel(fge(y, zero), z, fneg(z)), fneg(y))); break;
        default: r = fmul(cf(2.0f), fsel(z_major, z, fsel(y_major, y, x))); break;
        }
        write_v(in.dst, result_f(r, md));
    }

    bool vop3_only(std::uint32_t op, const Inst& in, const Val& s0, const Val& s1, const Val& s2, const Mods& md) {
        const auto F0 = [&] { return src_f(s0, 0, md); };
        const auto F1 = [&] { return src_f(s1, 1, md); };
        const auto F2 = [&] { return src_f(s2, 2, md); };
        const auto out_f = [&](Id v) { write_v(in.dst, result_f(v, md)); return true; };
        const auto out_u = [&](Id v) { write_v(in.dst, word(v)); return true; };
        const auto band31 = [&](const Val& v) { return ibin(spv::OpBitwiseAnd, as_u(v), cu(31)); };
        const auto i3 = [&](spv::Glsl inner, spv::Glsl outer) {  // outer(inner(s0, s1), s2) on i32
            const Id r = from_i(ext(inner, {to_i(as_u(s0)), to_i(as_u(s1))}, t_i32));
            return from_i(ext(outer, {to_i(r), to_i(as_u(s2))}, t_i32));
        };
        switch (op) {
        case 0x140: return out_f(fadd(mul_legacy(F0(), F1()), F2()));
        case 0x141: case 0x14b: return out_f(ext(spv::GlslFma, {F0(), F1(), F2()}));
        case 0x142: return out_u(ibin(spv::OpIAdd, ibin(spv::OpIMul, sext24(as_u(s0)), sext24(as_u(s1))), as_u(s2)));  // v_mad_i32_i24
        case 0x143: return out_u(ibin(spv::OpIAdd, ibin(spv::OpIMul, zext24(as_u(s0)), zext24(as_u(s1))), as_u(s2)));  // v_mad_u32_u24
        case 0x144: case 0x145: case 0x146: case 0x147: cube_ops(op, in, s0, s1, s2, md); return true;
        case 0x148: return out_u(m.emit(spv::OpBitFieldUExtract, t_u32, {as_u(s0), band31(s1), band31(s2)}));  // v_bfe_u32
        case 0x149: return out_u(from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(as_u(s0)), band31(s1), band31(s2)})));
        case 0x14a: {  // v_bfi_b32
            const Id a = as_u(s0);
            return out_u(ibin(spv::OpBitwiseOr, ibin(spv::OpBitwiseAnd, a, as_u(s1)), ibin(spv::OpBitwiseAnd, m.emit(spv::OpNot, t_u32, {a}), as_u(s2))));
        }
        case 0x151: return out_f(ext(spv::GlslNMin, {ext(spv::GlslNMin, {F0(), F1()}), F2()}));
        case 0x152: return out_u(i3(spv::GlslSMin, spv::GlslSMin));
        case 0x153: return out_u(ext(spv::GlslUMin, {ext(spv::GlslUMin, {as_u(s0), as_u(s1)}, t_u32), as_u(s2)}, t_u32));
        case 0x154: return out_f(ext(spv::GlslNMax, {ext(spv::GlslNMax, {F0(), F1()}), F2()}));
        case 0x155: return out_u(i3(spv::GlslSMax, spv::GlslSMax));
        case 0x156: return out_u(ext(spv::GlslUMax, {ext(spv::GlslUMax, {as_u(s0), as_u(s1)}, t_u32), as_u(s2)}, t_u32));
        case 0x157: {
            const Id a = F0(), b = F1(), c = F2();
            return out_f(ext(spv::GlslNMax, {ext(spv::GlslNMin, {a, b}), ext(spv::GlslNMin, {ext(spv::GlslNMax, {a, b}), c})}));
        }
        case 0x158: {  // v_med3_i32
            const Id a = to_i(as_u(s0)), b = to_i(as_u(s1)), c = to_i(as_u(s2));
            const Id mn = ext(spv::GlslSMin, {a, b}, t_i32), mx = ext(spv::GlslSMax, {a, b}, t_i32);
            return out_u(from_i(ext(spv::GlslSMax, {mn, ext(spv::GlslSMin, {mx, c}, t_i32)}, t_i32)));
        }
        case 0x159: {  // v_med3_u32
            const Id a = as_u(s0), b = as_u(s1), c = as_u(s2);
            const Id mn = ext(spv::GlslUMin, {a, b}, t_u32), mx = ext(spv::GlslUMax, {a, b}, t_u32);
            return out_u(ext(spv::GlslUMax, {mn, ext(spv::GlslUMin, {mx, c}, t_u32)}, t_u32));
        }
        case 0x15d:  // v_sad_u32
            return out_u(ibin(spv::OpIAdd, from_i(ext(spv::GlslSAbs, {to_i(ibin(spv::OpISub, as_u(s0), as_u(s1)))}, t_i32)), as_u(s2)));
        case 0x15e: {  // v_cvt_pk_u8_f32: byte s1 of s2 replaced by u8(s0)
            const Id byte = m.emit(spv::OpConvertFToU, t_u32, {ext(spv::GlslFClamp, {F0(), cf(0.0f), cf(255.0f)})});
            const Id sh = shift(spv::OpShiftLeftLogical, ibin(spv::OpBitwiseAnd, as_u(s1), cu(3)), cu(3));
            const Id mask = shift(spv::OpShiftLeftLogical, cu(0xff), sh);
            return out_u(ibin(spv::OpBitwiseOr, ibin(spv::OpBitwiseAnd, as_u(s2), m.emit(spv::OpNot, t_u32, {mask})),
                              shift(spv::OpShiftLeftLogical, byte, sh)));
        }
        case 0x169: case 0x16b: return out_u(ibin(spv::OpIMul, as_u(s0), as_u(s1)));  // v_mul_lo_u32 / v_mul_lo_i32
        case 0x16a: return out_u(extract(t_u32, m.emit(spv::OpUMulExtended, m.type_struct({t_u32, t_u32}), {as_u(s0), as_u(s1)}), 1));
        case 0x16c: {
            const Id a = to_i(as_u(s0)), b = to_i(as_u(s1));
            return out_u(from_i(extract(t_i32, m.emit(spv::OpSMulExtended, m.type_struct({t_i32, t_i32}), {a, b}), 1)));
        }
        default: return false;
        }
    }

    // ---- lane writes (SGPR spills) ---------------------------------------------------
    // v_writelane_b32 vN, s, L parks a scalar in lane L of vN (ignoring EXEC);
    // v_readlane_b32 d, vN, L takes it back. Where the reference reads the
    // spill cell (SpillCells::reads), the lift reads the value the cell's
    // writes left on this path, from the register cell_key(N, L): its
    // updates join at ifs like any register's, and one inside a region is a
    // scalar write the region's liveness check sees. vN itself now differs
    // between lanes, so it is not this pixel's word until written whole.
    void lane_op(const Inst& in) {
        const bool vop3 = in.enc == Enc::VOP3;
        const bool write = (vop3 ? in.op - 0x100 : in.op) == 2;
        const std::uint16_t lane_code = vop3 ? in.src1 : static_cast<std::uint16_t>(in.src1 - 256);  // a scalar code either way
        int ln = -1;
        const bool known = const_lane(lane_code, ln);
        if (write) {
            if (in.src0 >= 256) {
                reject("v_writelane_b32 of a VGPR");
                return;
            }
            const Val v = read(in.src0, in);
            if (known) {
                // Moved, never converted (a descriptor word stays one). A cell
                // the reference does not read is never read here either.
                reg[cell_key(in.dst, ln)] = v;
                note_result(v);
                note_region_write(cell_key(in.dst, ln));
            }
            // Without a known lane every cell of vN is gone in the reference
            // too (gcn/wave.h): no later v_readlane_b32 of vN reads one.
            poisoned.insert(256 + in.dst);
            note += " " + (known ? key_name(cell_key(in.dst, ln)) : "v" + std::to_string(in.dst) + " lane ?") + ":cell";
            return;
        }
        if (in.src0 < 256) {
            reject("v_readlane_b32 of a scalar operand");
            return;
        }
        const int vgpr = in.src0 - 256;
        if (!known) {
            reject("v_readlane_b32 of a lane held in a register reads another pixel's VGPR");
            return;
        }
        if (!spill.reads.count(in.offset)) {
            reject("v_readlane_b32 reads " + key_name(cell_key(vgpr, ln)) +
                   ", which no v_writelane_b32 alone reaches on every path: another pixel's VGPR, not a spilled scalar");
            return;
        }
        const auto it = reg.find(cell_key(vgpr, ln));
        if (it == reg.end()) {  // the reference's analysis says the cell holds; the lift found no write on this path
            reject("v_readlane_b32 of a spill cell the lift saw no v_writelane_b32 fill");
            return;
        }
        note_read(it->second);
        write_scalar(in.dst, it->second);
        ++cell_reads;
    }

    // ---- 64-bit operands -------------------------------------------------------------
    // A 64-bit operand's words, as translate.cpp read_pair_raw reads them: a
    // register pair, or an inline integer (sign-extended) or literal
    // (zero-extended). Lane masks (VCC, EXEC, an SGPR pair holding one) and SCC
    // are not lifted as numbers.
    bool read_pair_words(std::uint16_t code, const Inst& in, Val& lo, Val& hi) {
        if (code < 104 || code >= 256 || (code >= 112 && code < 124)) {
            lo = read(code, in);
            hi = read(static_cast<std::uint16_t>(code + 1), in);
            return true;
        }
        if (code == kLiteral) {
            lo = word_const(in.literal);
            hi = word_const(0);
        } else if (code >= 128 && code <= 192) {
            lo = word_const(code - 128u);
            hi = word_const(0);
        } else if (code >= 193 && code <= 208) {
            lo = word_const(static_cast<std::uint32_t>(-static_cast<int>(code - 192)));
            hi = word_const(0xffffffffu);
        } else {
            reject("unsupported 64-bit operand " + operand_name(code, in.literal, 2));
            return false;
        }
        return true;
    }
    void lshl_b64(const Inst& in) {  // v_lshl_b64: D = S0.u64 << (S1 & 63), as translate.cpp
        Val lo, hi;
        if (!read_pair_words(in.src0, in, lo, hi)) return;
        const Val s1 = read(in.src1, in);
        if (lo.constant && hi.constant && s1.constant) {  // integer arithmetic: folding is exact
            const std::uint64_t r = ((static_cast<std::uint64_t>(hi.bits) << 32) | lo.bits) << (s1.bits & 63);
            write_v(in.dst, word_const(static_cast<std::uint32_t>(r)));
            write_v(in.dst + 1, word_const(static_cast<std::uint32_t>(r >> 32)));
            return;
        }
        const auto widen = [&](const Val& v) { return m.emit(spv::OpUConvert, t_u64, {as_u(v)}); };
        const Id wide = m.emit(spv::OpBitwiseOr, t_u64, {widen(lo), m.emit(spv::OpShiftLeftLogical, t_u64, {widen(hi), m.const_u64(32)})});
        const Id r = m.emit(spv::OpShiftLeftLogical, t_u64, {wide, m.emit(spv::OpUConvert, t_u64, {ibin(spv::OpBitwiseAnd, as_u(s1), cu(63))})});
        const Id r_lo = m.emit(spv::OpUConvert, t_u32, {r});
        const Id r_hi = m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpShiftRightLogical, t_u64, {r, m.const_u64(32)})});
        write_v(in.dst, word(r_lo));
        write_v(in.dst + 1, word(r_hi));
    }

    void valu(const Inst& in) {
        Mods md;
        const auto unsupported = [&] { reject(std::string("unsupported ") + mnemonic(in)); };
        if (in.enc == Enc::VOP2) {
            if (in.op == 1 || in.op == 2) return lane_op(in);  // v_readlane_b32 / v_writelane_b32
            if (!vop2(in.op, in, read(in.src0, in), read(in.src1, in), md, nullptr, kVccLo)) unsupported();
            return;
        }
        if (in.enc == Enc::VOP1) {
            if (!vop1(in.op, in, read(in.src0, in), md)) unsupported();
            return;
        }
        if (in.enc == Enc::VOPC) {
            vcmp(in.op, read(in.src0, in), read(in.src1, in), md, kVccLo);
            return;
        }
        md.abs = in.abs;
        md.neg = in.neg;
        md.omod = in.omod;
        md.clamp = in.clamp;
        if (in.op < 0x100) {
            vcmp(in.op, read(in.src0, in), read(in.src1, in), md, in.sdst);
            return;
        }
        if (in.op == 0x161) return lshl_b64(in);  // its first operand is a 64-bit pair
        if (in.op < 0x140) {
            const std::uint32_t op = in.op - 0x100;
            if (op == 1 || op == 2) return lane_op(in);
            const Val s0 = read(in.src0, in), s1 = read(in.src1, in);
            const bool takes_mask = op == 0 || (op >= 40 && op <= 42);  // v_cndmask's select, the carry in
            Val mask;
            if (takes_mask) mask = read_lane(in.src2);
            if (!vop2(op, in, s0, s1, md, takes_mask ? &mask : nullptr, in.sdst)) unsupported();
            return;
        }
        if (in.op < 0x180) {
            // src2 is an operand only below 0x160; the 64-bit shift and the multiplies above take two.
            const Val s0 = read(in.src0, in), s1 = read(in.src1, in);
            const Val s2 = in.op < 0x160 ? read(in.src2, in) : word_const(0);
            if (!vop3_only(in.op, in, s0, s1, s2, md)) unsupported();
            return;
        }
        if (!vop1(in.op - 0x180, in, read(in.src0, in), md)) unsupported();
    }

    void sop1(const Inst& in) {
        switch (in.op) {
        case 3: write_scalar(in.dst, read(in.src0, in)); return;
        case 4: {
            const bool is_mask = in.src0 == kExecLo || in.src0 == kVccLo || (in.src0 < 104 && lanes.count(in.src0));
            if (is_mask || in.dst == kExecLo || in.dst == kVccLo) {
                write_lane(in.dst, read_lane(in.src0));
            } else {
                const Val lo = read(in.src0, in), hi = read(static_cast<std::uint16_t>(in.src0 + 1), in);
                write_scalar(in.dst, lo);
                write_scalar(static_cast<std::uint16_t>(in.dst + 1), hi);
            }
            return;  // s_mov_b64 leaves SCC, as in the translator
        }
        case 10: {  // s_wqm_b64
            const Val a = read_lane(in.src0);
            if (!a.constant && !widen_ok) {
                if (begin_whole_quad(in, a)) return;
                reject("whole-quad mode of a mask that is not constant: the result depends on the other pixels of the quad");
                return;
            }
            Val r = a;
            // Whole quads: a constant set mask leaves every lane of a quad with a
            // pixel set, helpers included; a clear one stays as exact as it was.
            r.helper_inexact = !a.set && a.helper_inexact;
            if (widen_ok) {
                // Widening a kill region's EXEC. From EXEC known set, every pixel
                // computes the region, as GCN's widened quads do wherever a pixel
                // survives; otherwise the pixel keeps its own bit. Either way the
                // pixels that differ must be killed at s_endpgm (checked there).
                Region& k = *open_regions.back();
                r.helper_inexact = true;  // the helper lanes GCN widens to, where the pixel keeps its own bit
                if (k.exec_before.constant && k.exec_before.set) {
                    r = lane_const(true);  // exact in every quad that survives
                    k.quad_exact = true;
                } else if (!implies(k.guard, k.exec_before)) {
                    // GCN's whole quads would also run pixels whose guard bit is
                    // set but whose EXEC bit was clear, which the lift leaves out.
                    reject("a kill region widens to whole quads with its guard not within EXEC at the branch");
                    return;
                }
            }
            write_lane(in.dst, r);
            set_scc_mask(r);
            return;
        }
        case 33:  // s_swappc_b64 in a vertex shader on vertex input (check_program): the elements load at the call
            write_scalar(in.dst, word_const(in.offset + 4));
            write_scalar(static_cast<std::uint16_t>(in.dst + 1), word_const(0));
            load_vertex_input();
            return;
        case 36: case 37: case 38: case 39: case 40: case 41: case 42: case 43: {  // s_*_saveexec_b64
            const Val a = read_lane(in.src0);
            const Val e = exec_lane();
            write_lane(in.dst, e);
            Val r;
            switch (in.op) {
            case 36: r = band(a, e); break;
            case 37: r = bor(a, e); break;
            case 38: r = bxor(a, e); break;
            case 39: r = band(a, bnot(e)); break;
            case 40: r = bor(a, bnot(e)); break;
            case 41: r = bnot(band(a, e)); break;
            case 42: r = bnot(bor(a, e)); break;
            default: r = bnot(bxor(a, e)); break;
            }
            write_lane(kExecLo, r);
            set_scc_mask(r);
            return;
        }
        default: reject(std::string("unsupported ") + mnemonic(in));
        }
    }

    // s_cselect_b32: S0 where SCC is set, else S1; SCC is left as it was. As
    // s_cmovk_i32, only on an SCC that is a scalar condition (from a scalar
    // compare: every lane alike, so the result is uniform where S0 and S1 are).
    void s_cselect(const Inst& in) {
        if (!scc_valid || !scc_uniform) {
            reject("s_cselect_b32 on an SCC that is not a scalar condition");
            return;
        }
        const Val a = read(in.src0, in), b = read(in.src1, in);
        if (scc_mask.constant) {
            write_scalar(in.dst, scc_mask.set ? a : b);
        } else if (a.kind == Kind::Float || b.kind == Kind::Float) {
            write_scalar(in.dst, flt(fsel(scc_mask.id, as_f(a), as_f(b))));
        } else {
            write_scalar(in.dst, word(m.emit(spv::OpSelect, t_u32, {scc_mask.id, as_u(a), as_u(b)})));
        }
    }

    void sop2(const Inst& in) {
        switch (in.op) {
        case 15: case 17: case 19: case 21: case 23: {
            const Val a = read_lane(in.src0);
            Val b = read_lane(in.src1);
            if (in.op == 21 || in.op == 23) b = bnot(b);
            const Val r = in.op == 15 || in.op == 21 ? band(a, b) : in.op == 19 ? bxor(a, b) : bor(a, b);
            write_lane(in.dst, r);
            set_scc_mask(r);
            return;
        }
        case 10: s_cselect(in); return;
        default:
            if (sop2_word_kind(in.op)) return sop2_word(in);
            reject(std::string("unsupported ") + mnemonic(in));
        }
    }

    // The 32-bit SOP2 integer operations, as translate.cpp sop2 computes them
    // (loop counters and indices): 1 when SCC is written, 2 when it is not.
    static int sop2_word_kind(std::uint16_t op) {
        switch (op) {
        case 0: case 1: case 2: case 3: case 6: case 7: case 8: case 9: case 14: case 16: case 18: case 20: case 22: case 24: case 26:
        case 28: case 30: case 32: case 34: case 39: case 40:
            return 1;
        case 36: case 38: return 2;
        default: return 0;
        }
    }
    void sop2_word(const Inst& in) {
        const Val va = read(in.src0, in), vb = read(in.src1, in);
        if (va.kind == Kind::Lane || vb.kind == Kind::Lane) {
            reject("a lane mask is used as a word");
            return;
        }
        const Id a = as_u(va), b = as_u(vb);
        const auto nonzero = [&](Id r) { return m.emit(spv::OpINotEqual, t_bool, {r, cu(0)}); };
        Id r = 0, scc = 0;
        switch (in.op) {
        case 0: case 1: {  // s_add_u32 / s_sub_u32: SCC is the carry or the borrow
            const Id c = m.emit(in.op == 0 ? spv::OpIAddCarry : spv::OpISubBorrow, m.type_struct({t_u32, t_u32}), {a, b});
            r = extract(t_u32, c, 0);
            scc = nonzero(extract(t_u32, c, 1));
            break;
        }
        case 2: case 3: {  // s_add_i32 / s_sub_i32: SCC is the signed overflow
            r = ibin(in.op == 2 ? spv::OpIAdd : spv::OpISub, a, b);
            const Id sign_ab = shift(spv::OpShiftRightLogical, ibin(spv::OpBitwiseXor, a, b), cu(31));
            const Id flipped = nonzero(shift(spv::OpShiftRightLogical, ibin(spv::OpBitwiseXor, a, r), cu(31)));
            scc = m.emit(spv::OpLogicalAnd, t_bool, {in.op == 2 ? ieq(sign_ab, cu(0)) : nonzero(sign_ab), flipped});
            break;
        }
        case 6: case 7: case 8: case 9: {  // s_min_i32 / s_min_u32 / s_max_i32 / s_max_u32: SCC is whether S0 was taken
            static const spv::Op kCmp[4] = {spv::OpSLessThan, spv::OpULessThan, spv::OpSGreaterThan, spv::OpUGreaterThan};
            scc = m.emit(kCmp[in.op - 6], t_bool, {a, b});
            r = m.emit(spv::OpSelect, t_u32, {scc, a, b});
            break;
        }
        case 14: r = ibin(spv::OpBitwiseAnd, a, b); break;
        case 16: r = ibin(spv::OpBitwiseOr, a, b); break;
        case 18: r = ibin(spv::OpBitwiseXor, a, b); break;
        case 20: r = ibin(spv::OpBitwiseAnd, a, m.emit(spv::OpNot, t_u32, {b})); break;
        case 22: r = ibin(spv::OpBitwiseOr, a, m.emit(spv::OpNot, t_u32, {b})); break;
        case 24: r = m.emit(spv::OpNot, t_u32, {ibin(spv::OpBitwiseAnd, a, b)}); break;
        case 26: r = m.emit(spv::OpNot, t_u32, {ibin(spv::OpBitwiseOr, a, b)}); break;
        case 28: r = m.emit(spv::OpNot, t_u32, {ibin(spv::OpBitwiseXor, a, b)}); break;
        case 30: r = shift(spv::OpShiftLeftLogical, a, b); break;
        case 32: r = shift(spv::OpShiftRightLogical, a, b); break;
        case 34: r = shift(spv::OpShiftRightArithmetic, a, b); break;
        case 36: r = shift(spv::OpShiftLeftLogical, ibin(spv::OpISub, shift(spv::OpShiftLeftLogical, cu(1), a), cu(1)), b); break;  // s_bfm_b32
        case 38: r = ibin(spv::OpIMul, a, b); break;
        case 39: case 40: {  // s_bfe_u32 / s_bfe_i32: offset S1[4:0], width S1[22:16]
            const Id off = ibin(spv::OpBitwiseAnd, b, cu(31));
            const Id width = ibin(spv::OpBitwiseAnd, shift(spv::OpShiftRightLogical, b, cu(16)), cu(0x7f));
            r = in.op == 39 ? m.emit(spv::OpBitFieldUExtract, t_u32, {a, off, width})
                            : from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(a), off, width}));
            break;
        }
        default: reject(std::string("unsupported ") + mnemonic(in)); return;
        }
        if (!scc && sop2_word_kind(in.op) == 1) scc = nonzero(r);  // the rest: SCC is whether the result is not zero
        write_scalar(in.dst, word(r));
        if (scc) set_scc_bool(lane(scc));
    }

    // translate.cpp sopc / sopk. Kind: 0 eq, 1 ne, 2 sgt, 3 sge, 4 slt, 5 sle,
    // 6 ugt, 7 uge, 8 ult, 9 ule, 10 bit clear, 11 bit set.
    Val scalar_compare(int kind, const Val& a, const Val& b) {
        if (a.kind == Kind::Lane || b.kind == Kind::Lane) {
            reject("a lane mask is compared as a word");
            return lane_const(false);
        }
        if (a.constant && b.constant) {
            const std::uint32_t x = a.bits, y = b.bits;
            const std::int32_t sx = static_cast<std::int32_t>(x), sy = static_cast<std::int32_t>(y);
            const bool bit = (x >> (y & 31)) & 1;
            const bool r[12] = {x == y, x != y, sx > sy, sx >= sy, sx < sy, sx <= sy, x > y, x >= y, x < y, x <= y, !bit, bit};
            return lane_const(r[kind]);
        }
        const Id x = as_u(a), y = as_u(b);
        static const spv::Op kOps[10] = {spv::OpIEqual, spv::OpINotEqual, spv::OpSGreaterThan, spv::OpSGreaterThanEqual, spv::OpSLessThan,
                                         spv::OpSLessThanEqual, spv::OpUGreaterThan, spv::OpUGreaterThanEqual, spv::OpULessThan,
                                         spv::OpULessThanEqual};
        if (kind < 10) return lane(m.emit(kOps[kind], t_bool, {x, y}));
        const Id bit = ibin(spv::OpBitwiseAnd, shift(spv::OpShiftRightLogical, x, y), cu(1));
        return lane(ieq(bit, cu(kind == 10 ? 0 : 1)));
    }

    void sopc(const Inst& in) {
        static const int kKind[14] = {0, 1, 2, 3, 4, 5, 0, 1, 6, 7, 8, 9, 10, 11};
        if (in.op > 13) {
            reject(std::string("unsupported ") + mnemonic(in));
            return;
        }
        set_scc_bool(scalar_compare(kKind[in.op], read(in.src0, in), read(in.src1, in)));
    }

    void sopk(const Inst& in) {
        const std::uint32_t imm = static_cast<std::uint32_t>(in.imm);
        switch (in.op) {
        case 0: write_scalar(in.dst, word_const(imm)); return;  // s_movk_i32
        case 2: {  // s_cmovk_i32: the immediate where SCC is set
            if (!scc_valid || !scc_uniform) {
                reject("s_cmovk_i32 on an SCC that is not a scalar condition");
                return;
            }
            const Val d = read(in.dst, in);
            if (scc_mask.constant) {
                write_scalar(in.dst, scc_mask.set ? word_const(imm) : d);
            } else {
                write_scalar(in.dst, word(m.emit(spv::OpSelect, t_u32, {scc_mask.id, cu(imm), as_u(d)})));
            }
            return;
        }
        case 3: case 4: case 5: case 6: case 7: case 8:  // s_cmpk_*_i32 against the sign-extended immediate
            set_scc_bool(scalar_compare(static_cast<int>(in.op) - 3, read(in.dst, in), word_const(imm)));
            return;
        case 9: case 10: case 11: case 12: case 13: case 14: {  // s_cmpk_*_u32 against its low 16 bits
            static const int kKind[6] = {0, 1, 6, 7, 8, 9};
            set_scc_bool(scalar_compare(kKind[in.op - 9], read(in.dst, in), word_const(imm & 0xffff)));
            return;
        }
        case 15: {  // s_addk_i32: SCC is the signed overflow
            const Id x = as_u(read(in.dst, in)), k = cu(imm);
            const Id r = ibin(spv::OpIAdd, x, k);
            const Id same_sign = ieq(shift(spv::OpShiftRightLogical, ibin(spv::OpBitwiseXor, x, k), cu(31)), cu(0));
            const Id flipped = m.emit(spv::OpINotEqual, t_bool, {shift(spv::OpShiftRightLogical, ibin(spv::OpBitwiseXor, x, r), cu(31)), cu(0)});
            write_scalar(in.dst, word(r));
            set_scc_bool(lane(m.emit(spv::OpLogicalAnd, t_bool, {same_sign, flipped})));
            return;
        }
        case 16: write_scalar(in.dst, word(ibin(spv::OpIMul, as_u(read(in.dst, in)), cu(imm)))); return;  // s_mulk_i32
        case 18: write_scalar(in.dst, word_const(0)); return;  // s_getreg_b32: hardware registers read 0, as the translator
        case 19: case 21: return;                               // s_setreg: ignored, as the translator
        default: reject(std::string("unsupported ") + mnemonic(in));
        }
    }

    void sopp(const Inst& in) {
        switch (in.op) {
        case 0: return;
        case 1: {
            const Val e = exec_lane();
            if (opt.stage == Stage::Vertex) {
                if (!(e.constant && e.set)) reject("EXEC is not known set at s_endpgm in a vertex shader, which has no discard");
                return;
            }
            // A guard is defined before its branch, so it is the same on either
            // path: with EXEC within it here, the pixels where the lift may
            // differ from GCN are killed in both, and the others match exactly
            // (an outer kill region's guard contains the inner ones').
            for (const KillProof& k : kill_proofs) {
                if (!implies(e, k.guard)) reject("EXEC at s_endpgm is not within the guard of a kill region, so a pixel where the lift differs from GCN may survive");
            }
            for (const Val& x : export_execs) {
                if (!implies(e, x)) reject("a pixel exported where EXEC was clear is not killed at s_endpgm");
            }
            if (!(e.constant && e.set)) {  // as the translator: a pixel whose EXEC bit is clear is discarded
                needs_kill = true;
                kill_exec = e;
            }
            return;
        }
        case 2: case 4: case 5: case 6: case 7: case 8: exec_lane(); return;  // region and if bookkeeping happens in emit()
        case 12:
            if (((in.imm >> 8) & 0xf) != 0xf) {
                for (const Pending& p : pending) {
                    for (std::size_t k = 0; k < p.values.size(); ++k) write_scalar(static_cast<std::uint16_t>(p.sdst + static_cast<int>(k)), p.values[k]);
                }
                pending.clear();
            }
            return;
        case 13: case 14: case 15: case 16: case 17: case 19: case 20: case 21: case 22: return;
        default: reject(std::string("unsupported ") + mnemonic(in));
        }
    }

    // A scalar load the reference walks the page table for: its words reach
    // the reference's output only if something reads them as data, since the
    // reference binds images, samplers and buffers by their resource paths
    // (translate.cpp image_for, sampler_for, indexed_buffer_load) and never by
    // the words. Each word is an OpUndef the lift never computes with: any
    // read of it as a number rejects (as_u, as_f), copies and joins keep it a
    // descriptor, and image and buffer instructions do not read it.
    void descriptor_load(const Inst& in, int count) {
        Id undef;
        {
            EntryScope entry(m);
            undef = m.emit(spv::OpUndef, t_u32, {});
        }
        descriptor_words[undef] = in.offset;
        Pending p{in.sdst, {}};
        const std::size_t mark = note.size();
        for (int k = 0; k < count; ++k) {
            write_scalar(static_cast<std::uint16_t>(in.sdst + k), word(undef));
            p.values.push_back(word(undef));
        }
        note.resize(mark);
        note += " " + operand_name(in.sdst, 0, count) + ":descriptor";
        pending.push_back(std::move(p));
        ++descriptor_loads;
    }

    void smrd(const Inst& in) {
        const int count = in.op < 8 ? (1 << in.op) : (in.op < 13 ? (1 << (in.op - 8)) : 0);
        const auto site = ref.buffer_at.find(in.offset);
        if (count && site == ref.buffer_at.end()) return descriptor_load(in, count);
        if (!count || site == ref.buffer_at.end() || site->second >= buffer_vars.size()) {
            reject(std::string(mnemonic(in)) + " is not read through a bound storage buffer in the reference");
            return;
        }
        if (!in.imm_flag && !in.has_literal) {
            reject("scalar load at a register offset");
            return;
        }
        const std::uint32_t index = site->second;
        const std::uint32_t off_dw = in.imm_flag ? static_cast<std::uint32_t>(in.imm) : in.literal;
        const Id bias = params_load(bias_loads, 4, index);
        Pending p{in.sdst, {}};
        for (int k = 0; k < count; ++k) {
            const Id at = m.emit(spv::OpIAdd, t_u32, {bias, cu(off_dw + static_cast<std::uint32_t>(k))});
            const Val v = word(m.load(t_u32, m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), at})));
            write_scalar(static_cast<std::uint16_t>(in.sdst + k), v);
            p.values.push_back(v);
        }
        pending.push_back(std::move(p));
        ++buffer_loads;
    }

    // A typed load's V# word 3 (StageParams::cb_w3), as translate.cpp
    // indexed_buffer_load reads it: also specialization constant
    // kCbW3SpecId + index (0, the default, reads the params), which an
    // optimized relink gives the draw's (render.cpp queue_library_relink) so
    // the driver folds the DST_SEL selects away.
    std::map<std::uint32_t, Id> w3_values, stride_values;
    Id w3_value(std::uint32_t index) { return spec_or_params(w3_values, w3_loads, 6, kCbW3SpecId, index); }
    // Its stride likewise (kCbStrideSpecId).
    Id stride_value(std::uint32_t index) { return spec_or_params(stride_values, stride_loads, 5, kCbStrideSpecId, index); }
    Id spec_or_params(std::map<std::uint32_t, Id>& values, std::map<std::uint32_t, Id>& loads, std::uint32_t member, std::uint32_t spec_id,
                      std::uint32_t index) {
        auto it = values.find(index);
        if (it != values.end()) return it->second;
        const Id loaded = params_load(loads, member, index);
        EntryScope entry(m);
        const Id spec = m.spec_const_u32(0, spec_id + index);
        const Id v = m.emit(spv::OpSelect, t_u32, {m.emit(spv::OpINotEqual, t_bool, {spec, cu(0)}), spec, loaded});
        return values.emplace(index, v).first->second;
    }
    Id params_load(std::map<std::uint32_t, Id>& cache, std::uint32_t member, std::uint32_t index) {
        auto it = cache.find(index);
        if (it == cache.end()) {
            EntryScope entry(m);
            const Id p = m.access_chain(p_uni_u32, ubo_var, {cu(member), cu(index / 4), cu(index % 4)});
            it = cache.emplace(index, m.load(t_u32, p)).first;
        }
        return it->second;
    }

    static bool format_layout(std::uint32_t dfmt, int& count, int bits[4]) {  // translate.cpp format_layout
        switch (dfmt) {
        case 1: count = 1; bits[0] = 8; return true;
        case 2: count = 1; bits[0] = 16; return true;
        case 3: count = 2; bits[0] = bits[1] = 8; return true;
        case 4: count = 1; bits[0] = 32; return true;
        case 5: count = 2; bits[0] = bits[1] = 16; return true;
        case 6: count = 3; bits[0] = 11; bits[1] = 11; bits[2] = 10; return true;
        case 7: count = 3; bits[0] = 10; bits[1] = 11; bits[2] = 11; return true;
        case 8: count = 4; bits[0] = 2; bits[1] = 10; bits[2] = 10; bits[3] = 10; return true;
        case 9: count = 4; bits[0] = 10; bits[1] = 10; bits[2] = 10; bits[3] = 2; return true;
        case 10: count = 4; bits[0] = bits[1] = bits[2] = bits[3] = 8; return true;
        case 11: count = 2; bits[0] = bits[1] = 32; return true;
        case 12: count = 4; bits[0] = bits[1] = bits[2] = bits[3] = 16; return true;
        case 13: count = 3; bits[0] = bits[1] = bits[2] = 32; return true;
        case 14: count = 4; bits[0] = bits[1] = bits[2] = bits[3] = 32; return true;
        default: return false;
        }
    }
    // One raw component as a word, per number format (translate.cpp convert_component).
    Id convert_component(Id raw, int bits, std::uint32_t nfmt) {
        const float maxu = static_cast<float>((1ull << bits) - 1);
        const float maxs = static_cast<float>((1ull << (bits - 1)) - 1);
        const auto sext = [&] { return m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(raw), cu(0), cu(static_cast<std::uint32_t>(bits))}); };
        const auto s2f = [&](Id i) { return m.emit(spv::OpConvertSToF, t_f32, {i}); };
        switch (nfmt) {
        case 0: return bits == 32 ? raw : as_u(flt(fdiv(m.emit(spv::OpConvertUToF, t_f32, {raw}), cf(maxu))));  // unorm
        case 1: return bits == 32 ? raw : as_u(flt(ext(spv::GlslNMax, {fdiv(s2f(sext()), cf(maxs)), cf(-1.0f)})));  // snorm
        case 2: return as_u(flt(m.emit(spv::OpConvertUToF, t_f32, {raw})));  // uscaled
        case 3: return as_u(flt(s2f(sext())));                                 // sscaled
        case 4: return raw;                                                    // uint
        case 5: return from_i(sext());                                         // sint
        case 6: return as_u(flt(fdiv(s2f(sext()), cf(maxs))));                 // snorm_ogl
        case 7:                                                                // float
            if (bits == 16) return as_u(flt(extract(t_f32, ext(spv::GlslUnpackHalf2x16, {raw}, t_v2f), 0)));
            if (bits == 10 || bits == 11) reject("packed 10- and 11-bit float components are not lifted");
            return raw;
        default: return raw;
        }
    }

    // tbuffer_load_format_* read by index from a storage buffer the reference
    // binds (translate.cpp indexed_buffer_load with cb_no_fallback): the
    // element at dword (offset12 + soffset) / 4 + index * stride / 4, decoded
    // by its static format and selected per component by the V#'s word 3.
    void mtbuf(const Inst& in) {
        if (in.op >= 4) {
            reject(std::string(mnemonic(in)) + " is not lifted");
            return;
        }
        const auto site = ref.buffer_at.find(in.offset);
        if (site == ref.buffer_at.end() || site->second >= buffer_vars.size() || !ref.buffers[site->second].indexed) {
            reject(std::string(mnemonic(in)) + " is not an indexed load from a storage buffer the reference binds");
            return;
        }
        int count = 0, bits[4] = {};
        const std::int32_t constant = in.soffset <= 192 ? static_cast<std::int32_t>(in.soffset) - 128 : 192 - static_cast<std::int32_t>(in.soffset);
        const std::int32_t off = static_cast<std::int32_t>(in.offset12) + constant;
        if (!in.idxen || in.offen || in.addr64 || in.soffset < 128 || in.soffset > 208 || off < 0 || off % 4 != 0 ||
            !format_layout(in.dfmt, count, bits)) {
            reject(std::string(mnemonic(in)) + ": the reference's indexed load has another shape");
            return;
        }
        const std::uint32_t index = site->second;
        const Id stride_dw = shift(spv::OpShiftRightLogical, stride_value(index), cu(2));
        const Id first = ibin(spv::OpIAdd, cu(static_cast<std::uint32_t>(off / 4)), ibin(spv::OpIMul, as_u(value(256 + in.vaddr)), stride_dw));
        const Id bias = params_load(bias_loads, 4, index);
        int total = 0;
        for (int k = 0; k < count; ++k) total += bits[k];
        std::vector<Id> words;
        for (int k = 0; k < (total + 31) / 32; ++k) {
            const Id at = ibin(spv::OpIAdd, bias, ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k))));
            words.push_back(m.load(t_u32, m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), at})));
        }
        const bool integer = in.nfmt == 4 || in.nfmt == 5;
        const Id one = integer ? cu(1) : as_u(float_const(1.0f));
        std::vector<Id> comps(4, cu(0));
        for (int k = 0, bit = 0; k < count; bit += bits[k], ++k) {
            const Id w = words[static_cast<std::size_t>(bit / 32)];
            const Id raw = bits[k] == 32 ? w : m.emit(spv::OpBitFieldUExtract, t_u32, {w, cu(static_cast<std::uint32_t>(bit % 32)), cu(static_cast<std::uint32_t>(bits[k]))});
            comps[static_cast<std::size_t>(k)] = convert_component(raw, bits[k], in.nfmt);
        }
        if (count < 4) comps[3] = one;  // missing components are 0, 0, 0, 1
        const Id w3 = w3_value(index);
        std::vector<Val> out;
        for (int k = 0; k < (in.op & 3) + 1; ++k) {  // DST_SEL: 4-7 a component, 1 one, else zero
            const Id s3 = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(3u * static_cast<std::uint32_t>(k)), cu(3)});
            const auto pick = [&](std::uint32_t code, Id then_v, Id else_v) { return m.emit(spv::OpSelect, t_u32, {ieq(s3, cu(code)), then_v, else_v}); };
            out.push_back(word(pick(4, comps[0], pick(5, comps[1], pick(6, comps[2], pick(7, comps[3], pick(1, one, cu(0))))))));
        }
        for (std::size_t k = 0; k < out.size(); ++k) write_v(in.vdata + static_cast<int>(k), out[k]);
        ++buffer_loads;
    }

    // TranslateOptions::vertex_input, as translate.cpp load_vertex_input: each
    // element from its input location (raw unsigned components; a packed
    // format as one dword), converted by its number format and DST_SEL.
    void load_vertex_input() {
        inst_uniform = false;  // per vertex
        for (const VertexElement& el : opt.vertex_input) {
            const std::uint32_t dfmt = (el.w3 >> 15) & 0xf, nfmt = (el.w3 >> 12) & 7;
            int count = 0, bits[4] = {};
            if (!format_layout(dfmt, count, bits)) {
                reject("vertex input: unsupported buffer data format " + std::to_string(dfmt));
                return;
            }
            Id& var = vertex_in_vars[el.location];
            if (!var) {
                var = m.global_variable(p_in_v4u, spv::ScInput);
                m.decorate(var, spv::DecLocation, {el.location});
                m.name(var, "vertex_in" + std::to_string(el.location));
                interface.push_back(var);
            }
            const bool packed = dfmt >= 6 && dfmt <= 9;
            const Id one = nfmt == 4 || nfmt == 5 ? cu(1) : as_u(float_const(1.0f));
            std::vector<Id> comps(4, cu(0));
            {
                EntryScope entry(m);  // inputs and their conversions dominate every block
                const Id raw = m.load(t_v4u, var);
                for (int k = 0, bit = 0; k < count; bit += bits[k], ++k) {
                    const Id c = packed ? m.emit(spv::OpBitFieldUExtract, t_u32, {extract(t_u32, raw, 0), cu(static_cast<std::uint32_t>(bit)),
                                                                                  cu(static_cast<std::uint32_t>(bits[k]))})
                                        : extract(t_u32, raw, static_cast<std::uint32_t>(k));
                    comps[static_cast<std::size_t>(k)] = convert_component(c, bits[k], nfmt);
                }
            }
            if (count < 4) comps[3] = one;
            for (std::uint32_t k = 0; k < el.count && k < 4; ++k) {
                const std::uint32_t s3 = (el.w3 >> (3 * k)) & 7;
                write_v(static_cast<int>(el.vdata + k), word(s3 >= 4 ? comps[s3 - 4] : s3 == 1 ? one : cu(0)));
            }
        }
    }

    static int coord_count(std::uint32_t dim, bool arrayed) {
        return (dim == spv::Dim1D ? 1 : dim == spv::Dim2D ? 2 : 3) + (arrayed ? 1 : 0);
    }

    // ---- whole-quad mode and implicit derivatives ------------------------------------
    // Every register holds GCN's value in every pixel - is exact - except after
    // whole-quad mode over a varying EXEC: s_wqm_b64 exec, exec where EXEC is a
    // mask M (a divergent region's) runs what follows, up to the restore of M,
    // for every pixel of each quad holding an M pixel (GCN's quads are four
    // lanes), and the lift runs it for every pixel. A register written there is
    // exact only in those quads. `partial` keeps where each register written so
    // is exact, as a lane mask A:
    //
    //   * quad: in every pixel of a quad holding a pixel of A;
    //   * not quad: in the pixels of A (A constant false: none the lifter names).
    //
    // A value is exact where every register it is computed from is; a write
    // under EXEC E is exact in the E pixels where the value is and elsewhere
    // where the old value was. A lane mask, an export and every operand of a
    // sample must be exact in the pixels of EXEC. The coordinates of an
    // implicit-LOD sample or of image_get_lod must also be exact in every pixel
    // of each quad holding an EXEC pixel: GCN's texture unit takes the
    // derivatives from the coordinates in the quad's four lanes, EXEC set or
    // not, and the lift samples in uniform control flow, where Vulkan takes them
    // from the quad's four invocations, helpers included.
    struct Exact {
        bool quad = false;
        Val mask;
    };
    struct Where {             // where a value is exact
        bool partial = false;  // false: in every pixel
        Exact at;
    };
    std::map<int, Exact> partial;  // register key -> where it is exact; absent: in every pixel
    Where reads;                   // where every register this instruction read so far is exact
    struct WholeQuad {             // an open whole-quad block (begin_whole_quad)
        bool active = false;
        Val mask;                  // EXEC where it was widened
        std::uint32_t start = 0, end = 0;
        std::size_t writes = 0;
    } wqm;
    std::vector<std::uint32_t> varying_samples, lod_queries;  // samples under a varying EXEC; image_get_lod

    // `partial` across an if (open_arm, then_to_else, merge_arms): each arm
    // starts from the branch's, and after the join a register is exact where
    // it is in both arms - every lane took the same one, the lift knows not which.
    struct ArmExact {
        std::map<int, Exact> at_branch, then_end;
    };
    std::vector<ArmExact> arm_exact;  // innermost last, as `arms`
    void exact_open_arm() { arm_exact.push_back({partial, {}}); }
    void exact_then_to_else() {
        arm_exact.back().then_end = std::move(partial);
        partial = arm_exact.back().at_branch;
    }
    void exact_merge_arms() {
        const std::map<int, Exact> then_end = std::move(arm_exact.back().then_end);
        arm_exact.pop_back();
        std::set<int> keys;
        for (const auto& kv : then_end) keys.insert(kv.first);
        for (const auto& kv : partial) keys.insert(kv.first);
        std::map<int, Exact> joined;
        for (int k : keys) {
            const auto t = then_end.find(k);
            const Where w = meet(t == then_end.end() ? Where{} : Where{true, t->second}, where_reg(k));
            if (w.partial) joined[k] = w.at;
        }
        partial = std::move(joined);
    }

    // Inside a kill region widened to whole quads from a varying EXEC, GCN
    // writes whole quads where the lift keeps each pixel's bit; `partial` does
    // not follow that (the pixels that differ are killed at s_endpgm), so
    // nothing there may rely on a register's value in a quad's other pixels.
    bool in_widened_kill() const {
        return std::any_of(open_regions.begin(), open_regions.end(), [](const Region* r) { return r->kill && r->widened && !r->quad_exact; });
    }

    // The pixels of `a` lie within those of `b`.
    bool within(const Exact& a, const Exact& b) const { return (b.quad || !a.quad) && implies(a.mask, b.mask); }
    Where where_reg(int key) const {
        const auto it = partial.find(key);
        return it == partial.end() ? Where{} : Where{true, it->second};
    }
    // Where both are exact: the smaller where one lies within the other.
    Where meet(const Where& a, const Where& b) {
        if (!a.partial) return b;
        if (!b.partial) return a;
        if (within(a.at, b.at)) return a;
        if (within(b.at, a.at)) return b;
        return Where{true, Exact{false, lane_const(false)}};
    }
    void note_exact_read(int key) {
        if (partial.count(key)) reads = meet(reads, where_reg(key));
    }
    // Exact in every pixel whose bit of `e` is set ...
    bool exact_in(const Where& w, const Val& e) const { return !w.partial || implies(e, w.at.mask); }
    // ... and in every pixel of each quad holding one.
    bool quad_exact_in(const Where& w, const Val& e) const { return !w.partial || (w.at.quad && implies(e, w.at.mask)); }
    // Exact in every pixel whose bit of `e` is clear.
    bool exact_outside(const Where& w, const Val& e) const {
        if (!w.partial || (e.constant && e.set)) return true;
        if (e.constant) return false;
        const auto n = negation_of.find(e.id);
        return n != negation_of.end() && implies(lane(n->second), w.at.mask);
    }
    // Register `key` after a write under EXEC `e` of a value exact at `v`.
    void note_exact_write(int key, const Where& v, const Val& e) {
        const Where old = where_reg(key);
        const bool v_in = exact_in(v, e), old_out = exact_outside(old, e);
        const Where r = v_in && old_out ? Where{} : v_in ? old : old_out ? v : meet(v, old);
        if (r.partial) {
            partial[key] = r.at;
        } else {
            partial.erase(key);
        }
    }
    // A lane mask computed from VGPRs (a comparison, a carry) is GCN's where
    // EXEC is set, clear elsewhere in both; and the write of EXEC that ends a
    // whole-quad block must restore the mask it widened.
    bool lane_write_exact(std::uint16_t code, const Val& v) {
        if (wqm.active && code == kExecLo) {
            if (!same_lane(v, wqm.mask)) {
                reject("a whole-quad block does not end by restoring EXEC to the mask it widened");
                return false;
            }
            end_whole_quad();
        }
        if (reads.partial && !exact_in(reads, exec_lane())) {
            reject("a lane mask is computed from a value the lift may hold differently from GCN in pixels of EXEC (written in whole-quad mode)");
            return false;
        }
        return true;
    }

    // s_wqm_b64 exec, exec over a varying EXEC M (not a kill region's
    // widening): a block that GCN runs for every pixel of each quad holding an M
    // pixel, ended by s_mov_b64 exec, <M> before any branch or block boundary.
    // The lift runs its VGPR writes for every pixel (whole_quad_write) and keeps
    // EXEC at M; whatever needs EXEC inside the block - a comparison, a sample,
    // an export, an EXEC read - is rejected. SCC is "some lane of M", as for
    // M's whole quads. False: not this form (the caller rejects).
    bool begin_whole_quad(const Inst& in, const Val& m_exec) {
        if (in.dst != kExecLo || in.src0 != kExecLo || opt.stage != Stage::Pixel) return false;
        if (in_widened_kill()) {
            reject("whole-quad block inside a widened kill region, where the lift keeps the pixel's own EXEC bit");
            return true;
        }
        const std::size_t at = static_cast<std::size_t>(&in - prog.insts.data());
        for (std::size_t i = at + 1; i < prog.insts.size(); ++i) {
            const Inst& p = prog.insts[i];
            if (block_starts.count(p.offset) || is_branch(p) || (p.enc == Enc::SOPP && p.op == 1)) break;
            if (p.enc == Enc::SOP1 && p.op == 4 && p.dst == kExecLo) {
                wqm = WholeQuad{true, m_exec, in.offset, p.offset, 0};
                set_scc_mask(m_exec);
                note_region_write(kKeyExec);
                note += " exec:whole-quad";
                return true;
            }
        }
        reject("whole-quad mode of a mask that is not constant, not ended by s_mov_b64 exec before a branch or block boundary");
        return true;
    }
    void end_whole_quad() {
        res.proof.push_back("whole-quad block " + hex_offset(wqm.start) + "-" + hex_offset(wqm.end) + ": EXEC widened from a varying mask " +
                            "to its quads and restored to it; its " + std::to_string(wqm.writes) +
                            " VGPR writes run for every pixel and hold GCN's value in those quads, which every later use checks");
        wqm.active = false;
    }
    // A VGPR write inside a whole-quad block: GCN writes every pixel of each
    // quad holding a pixel of the block's mask, the lift every pixel.
    void whole_quad_write(int idx, const Val& v) {
        if (!quad_exact_in(reads, wqm.mask)) {
            reject("a whole-quad block computes v" + std::to_string(idx) + " from a value the lift may hold differently from GCN in its quads");
            return;
        }
        const int key = 256 + idx;
        reg[key] = v;
        note_result(v);
        partial[key] = Exact{true, wqm.mask};
        ++wqm.writes;
        note += " v" + std::to_string(idx) + ":quad";
    }

    // The operands of a sample or image_get_lod under EXEC `e`, as the
    // instruction read them: GCN's in every pixel of EXEC; for implicit
    // derivatives, the coordinates (VGPRs coord_va..) also in every pixel of
    // each quad holding one. The sample itself runs for every pixel in uniform
    // control flow and its result is written under EXEC.
    bool sample_exact(const Val& e, bool implicit, int coord_va, int ncoord, const char* what) {
        if (!exact_in(reads, e)) {
            reject(std::string(what) + " reads an operand the lift may hold differently from GCN in pixels of EXEC (written in whole-quad mode)");
            return false;
        }
        if (implicit && !(e.constant && e.set) && in_widened_kill()) {
            reject(std::string(what) + " with implicit derivatives under a varying EXEC inside a widened kill region, where GCN computes "
                   "for whole quads and the lift keeps the pixel's own EXEC bit");
            return false;
        }
        for (int k = 0; implicit && k < ncoord; ++k) {
            if (!quad_exact_in(where_reg(256 + coord_va + k), e)) {
                reject(std::string(what) + " with implicit derivatives reads coordinates the lift may hold differently from GCN in pixels of "
                       "the quads of EXEC");
                return false;
            }
        }
        return true;
    }

    // The sampled image for the reference's image and sampler bindings `ii`,
    // `si` (the image itself in `image`): the bindings, or bindless their slots
    // from the params block (StageParams::image_index / sampler_index, members
    // 7 and 8 here).
    Id sampled_image(std::uint32_t ii, std::uint32_t si, Id& image) {
        Id sampler;
        if (ref.bindless) {
            const auto slot = [&](std::uint32_t member, std::uint32_t k) {
                return m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(member), cu(k / 4), cu(k % 4)}));
            };
            image = m.load(image_types[ii], m.access_chain(m.type_pointer(spv::ScUniformConstant, image_types[ii]), image_vars[ii], {slot(7, ii)}));
            sampler = m.load(m.type_sampler(), m.access_chain(m.type_pointer(spv::ScUniformConstant, m.type_sampler()), sampler_vars[si], {slot(8, si)}));
        } else {
            image = m.load(image_types[ii], image_vars[ii]);
            sampler = m.load(m.type_sampler(), sampler_vars[si]);
        }
        return m.emit(spv::OpSampledImage, m.type_sampled_image(image_types[ii]), {image, sampler});
    }

    // image_get_lod, as the translator reads it (translate.cpp mimg):
    // OpImageQueryLod's (d_l - level_base, lambda') - the level a sample would
    // read after the sampler's and the view's clamps, and the LOD before them
    // with the sampler's bias - for GCN's clamped and unclamped LOD, written as
    // the dmask selects. Its derivatives are implicit, as an implicit-LOD
    // sample's.
    void image_get_lod(const Inst& in) {
        if (opt.stage != Stage::Pixel) {
            reject("image_get_lod outside a pixel shader, where there are no derivatives");
            return;
        }
        const Val e = exec_lane();
        if (in.dmask & 0xc) {
            reject("image_get_lod returns two components, and its dmask selects a third or fourth");
            return;
        }
        if (std::any_of(kill_proofs.begin(), kill_proofs.end(), [](const KillProof& k) { return k.per_pixel; })) {
            reject("image_get_lod after a kill region where the lift may differ from GCN in pixels it kills: its derivatives would read them");
            return;
        }
        const auto ii = ref.image_at.find(in.offset);
        const auto si = ref.sampler_at.find(in.offset);
        if (ii == ref.image_at.end() || si == ref.sampler_at.end() || ii->second >= image_vars.size() || si->second >= sampler_vars.size()) {
            reject("the reference has no image or sampler binding for this image_get_lod");
            return;
        }
        const ImageBinding& b = ref.images[ii->second];
        if (b.storage || b.kind || b.depth || b.cube) {
            reject(b.storage ? "image_get_lod of a storage image binding"
                   : b.kind  ? "image_get_lod of an integer image binding"
                   : b.depth ? "image_get_lod of a depth binding"
                             : "image_get_lod of a cube map, whose face coordinates the translator reads as a 2D array's");
            return;
        }
        if (in.unorm || (si->second < opt.sampler_force_unnormalized.size() && opt.sampler_force_unnormalized[si->second])) {
            reject("image_get_lod with unnormalized coordinates");
            return;
        }
        const int ncoord = coord_count(b.dim, false);  // no layer: OpImageQueryLod takes the coordinates alone
        std::vector<Id> coords;
        for (int k = 0; k < ncoord; ++k) coords.push_back(as_f(value(256 + in.vaddr + k)));
        if (!sample_exact(e, true, in.vaddr, ncoord, "image_get_lod")) return;
        m.capability(spv::CapImageQuery);
        Id image;
        const Id sampled = sampled_image(ii->second, si->second, image);
        const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2f : t_v3f, coords);
        const Id lod = m.emit(spv::OpImageQueryLod, t_v2f, {sampled, coord});
        int out = 0;
        for (std::uint32_t k = 0; k < 2; ++k) {
            if ((in.dmask >> k) & 1) write_v(in.vdata + out++, flt(extract(t_f32, lod, k)));
        }
        lod_queries.push_back(in.offset);
    }

    void mimg(const Inst& in) {
        inst_uniform = false;  // texels are not tracked as uniform
        const std::uint32_t op = in.op;
        if (op == 96) {
            image_get_lod(in);
            return;
        }
        if (!(op >= 32 && op < 64)) {
            reject(std::string(mnemonic(in)) + " is not lifted");
            return;
        }
        // Inside a region too, and under a varying EXEC: the sample runs for
        // every pixel in uniform control flow and its texel is written under
        // EXEC; sample_exact decides whether that is GCN's sample.
        const Val e = exec_lane();
        const std::uint32_t v = op - 32;
        const bool has_o = (v & 16) != 0, has_c = (v & 8) != 0;
        const std::uint32_t low = v & 7;
        const bool has_cl = low == 1 || low == 3 || low == 6, has_d = low == 2 || low == 3, has_l = low == 4;
        const bool has_b = low == 5 || low == 6, has_lz = low == 7;
        if (std::any_of(kill_proofs.begin(), kill_proofs.end(), [](const KillProof& k) { return k.per_pixel; }) && !has_l && !has_lz && !has_d) {
            reject("implicit-LOD sample after a kill region where the lift may differ from GCN in pixels it kills: its derivatives would read them");
            return;
        }
        if (in_divergent_loop() && !has_l && !has_lz && !has_d) {
            reject("implicit-LOD sample inside a loop pixels leave at different iterations: its derivatives would read pixels that left");
            return;
        }
        const auto ii = ref.image_at.find(in.offset);
        const auto si = ref.sampler_at.find(in.offset);
        if (ii == ref.image_at.end() || si == ref.sampler_at.end() || ii->second >= image_vars.size() || si->second >= sampler_vars.size()) {
            reject("the reference has no image or sampler binding for this sample");
            return;
        }
        const ImageBinding& b = ref.images[ii->second];
        if (b.storage || b.depth != has_c || b.kind) {
            reject(b.storage ? "storage image binding"
                   : b.kind  ? "integer image binding"
                             : "the reference's depth binding does not match the sample's comparison");
            return;
        }
        const bool unnormalized = in.unorm || (si->second < opt.sampler_force_unnormalized.size() && opt.sampler_force_unnormalized[si->second]);
        Id image;
        const Id sampled = sampled_image(ii->second, si->second, image);
        int va = in.vaddr;
        const auto vgpr = [&](int idx) { return value(256 + idx); };
        Id offset = 0, bias = 0, dx = 0, dy = 0, lod = 0;
        std::vector<Id> offs;
        if (has_o) {
            const Id packed = m.emit(spv::OpBitcast, t_i32, {as_u(vgpr(va++))});
            const int nc = coord_count(b.dim, false);
            for (int k = 0; k < nc; ++k) offs.push_back(m.emit(spv::OpBitFieldSExtract, t_i32, {packed, cu(8u * static_cast<std::uint32_t>(k)), cu(6)}));
            offset = nc == 1 ? offs[0] : m.emit(spv::OpCompositeConstruct, nc == 2 ? t_v2i : t_v3i, offs);
            if (runtime_sample_offsets() || b.cube) m.capability(spv::CapImageGatherExtended);
        }
        if (has_b) bias = as_f(vgpr(va++));
        const Id dref = has_c ? as_f(vgpr(va++)) : 0;  // after the bias, before derivatives and coordinates
        std::vector<Id> gx, gy;
        if (has_d) {
            const int ng = coord_count(b.dim, false);
            for (int k = 0; k < ng; ++k) gx.push_back(as_f(vgpr(va++)));
            for (int k = 0; k < ng; ++k) gy.push_back(as_f(vgpr(va++)));
        }
        const int ncoord = coord_count(b.dim, b.arrayed);
        const int coord_va = va;
        std::vector<Id> coords;
        for (int k = 0; k < ncoord; ++k) coords.push_back(as_f(vgpr(va++)));
        if (b.cube) {  // the translator's cube conversion (translate.cpp mimg)
            coords[0] = fsub(coords[0], cf(1.0f));
            coords[1] = fsub(coords[1], cf(1.0f));
            coords[2] = ext(spv::GlslFma, {ext(spv::GlslFloor, {fdiv(coords[2], cf(8.0f))}), cf(-2.0f), coords[2]});
        }
        if (unnormalized && !b.cube) {
            m.capability(spv::CapImageQuery);
            const int ns = coord_count(b.dim, false), nq = coord_count(b.dim, b.arrayed);
            const Id size = m.emit(spv::OpImageQuerySizeLod, nq == 1 ? t_i32 : nq == 2 ? t_v2i : t_v3i, {image, m.emit(spv::OpBitcast, t_i32, {cu(0)})});
            for (int k = 0; k < ns; ++k) {
                const Id extent = nq == 1 ? size : extract(t_i32, size, static_cast<std::uint32_t>(k));
                const Id scale = fdiv(cf(1.0f), m.emit(spv::OpConvertSToF, t_f32, {extent}));
                coords[static_cast<std::size_t>(k)] = fmul(coords[static_cast<std::size_t>(k)], scale);
                if (!gx.empty()) {
                    gx[static_cast<std::size_t>(k)] = fmul(gx[static_cast<std::size_t>(k)], scale);
                    gy[static_cast<std::size_t>(k)] = fmul(gy[static_cast<std::size_t>(k)], scale);
                }
            }
        }
        if (offset && !b.cube && !runtime_sample_offsets()) {
            // As the translator (translate.cpp mimg): without maintenance8 the
            // coordinates move by the offset in level 0's texels.
            m.capability(spv::CapImageQuery);
            const int nq = coord_count(b.dim, b.arrayed);
            const Id size = m.emit(spv::OpImageQuerySizeLod, nq == 1 ? t_i32 : nq == 2 ? t_v2i : t_v3i, {image, m.emit(spv::OpBitcast, t_i32, {cu(0)})});
            for (std::size_t k = 0; k < offs.size(); ++k) {
                const Id extent = nq == 1 ? size : extract(t_i32, size, static_cast<std::uint32_t>(k));
                coords[k] = ext(spv::GlslFma, {m.emit(spv::OpConvertSToF, t_f32, {offs[k]}),
                                               fdiv(cf(1.0f), m.emit(spv::OpConvertSToF, t_f32, {extent})), coords[k]});
            }
            offset = 0;
        }
        const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2f : ncoord == 3 ? t_v3f : t_v4f, coords);
        if (!gx.empty()) {
            const int ng = static_cast<int>(gx.size());
            dx = ng == 1 ? gx[0] : m.emit(spv::OpCompositeConstruct, ng == 2 ? t_v2f : t_v3f, gx);
            dy = ng == 1 ? gy[0] : m.emit(spv::OpCompositeConstruct, ng == 2 ? t_v2f : t_v3f, gy);
        }
        if (has_l) lod = as_f(vgpr(va++));
        if (has_lz) lod = cf(0.0f);
        if (opt.stage != Stage::Pixel) {  // no derivatives outside a pixel shader: the translator samples level 0
            if (has_b) {
                reject("biased sample outside a pixel shader");
                return;
            }
            if (!lod && gx.empty()) lod = cf(0.0f);
        }
        if (!sample_exact(e, !lod && !dx, coord_va, ncoord, "an image sample")) return;
        std::uint32_t mask = 0;
        std::vector<std::uint32_t> ops = {sampled, coord};
        if (has_c) ops.push_back(dref);
        std::vector<std::uint32_t> extra;
        if (bias) { mask |= spv::IoBias; extra.push_back(bias); }
        if (lod) { mask |= spv::IoLod; extra.push_back(lod); }
        if (dx) { mask |= spv::IoGrad; extra.push_back(dx); extra.push_back(dy); }
        if (offset) { mask |= spv::IoOffset; extra.push_back(offset); }
        (void)has_cl;  // the clamp operand is ignored, as by the translator
        if (mask) {
            ops.push_back(mask);
            ops.insert(ops.end(), extra.begin(), extra.end());
        }
        Id texel;
        if (has_c) {  // the comparison result in all four components, as the translator
            const Id r = m.emit(lod || dx ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleDrefImplicitLod, t_f32, ops);
            texel = m.emit(spv::OpCompositeConstruct, t_v4f, {r, r, r, r});
        } else {
            texel = m.emit(lod || dx ? spv::OpImageSampleExplicitLod : spv::OpImageSampleImplicitLod, t_v4f, ops);
        }
        int out = 0;
        for (std::uint32_t k = 0; k < 4; ++k) {
            if ((in.dmask >> k) & 1) write_v(in.vdata + out++, flt(extract(t_f32, texel, k)));
        }
        (e.constant && e.set ? samples : varying_samples).push_back(in.offset);
        if (in_divergent_loop()) divergent_samples.push_back(in.offset);  // explicit LOD or gradients (checked above)
    }

    void vintrp(const Inst& in) {
        inst_uniform = false;  // per pixel
        if (opt.stage != Stage::Pixel) {
            reject("interpolation outside a pixel shader");
            return;
        }
        write_v(in.dst, flt(extract(t_f32, attr(in.attr), in.attr_chan)));
    }

    void exp(const Inst& in) {
        // The translator exports whatever EXEC holds; a pixel whose bit is clear
        // here must be killed at s_endpgm (checked there).
        const bool vertex = opt.stage == Stage::Vertex;
        if (!open_loops.empty()) {
            reject("export inside a loop");
            return;
        }
        if (const Val e = exec_lane(); !(e.constant && e.set)) {
            if (vertex) {
                reject("export where EXEC is not known set in a vertex shader, which has no discard");
                return;
            }
            export_execs.push_back(e);
        }
        if (in.tgt == 9) return;  // null export
        // Pixel shaders export colour targets 0-7; vertex shaders the position (12) and params (32-63).
        if (vertex ? !(in.tgt == 12 || (in.tgt >= 32 && in.tgt < 64)) : in.tgt >= 8) {
            reject(vertex ? "only position and param exports are lifted in a vertex shader, not target " + std::to_string(in.tgt)
                          : std::string("only colour exports are lifted"));
            return;
        }
        std::vector<Id> comps(4, cf(0.0f));
        if (in.compr) {
            for (std::uint32_t pair = 0; pair < 2; ++pair) {
                if (!((in.dmask >> (pair * 2)) & 3)) continue;
                if (const auto pk = packed_pairs.find(value(256 + in.vsrc[pair]).id); pk != packed_pairs.end() && pk->second.block == m.blocks()) {
                    if (native_rtz) {
                        const Id v2 = m.emit(spv::OpCompositeConstruct, t_v2f, {pk->second.lo, pk->second.hi});
                        const Id r = m.emit(spv::OpFConvert, t_v2f, {m.emit(spv::OpFConvert, t_v2h, {v2})});
                        comps[pair * 2] = m.emit(spv::OpCompositeExtract, t_f32, {r, 0u});
                        comps[pair * 2 + 1] = m.emit(spv::OpCompositeExtract, t_f32, {r, 1u});
                    } else {
                        comps[pair * 2] = emit_rtz_half(m, {t_bool, t_u32, t_i32, t_f32}, pk->second.lo);
                        comps[pair * 2 + 1] = emit_rtz_half(m, {t_bool, t_u32, t_i32, t_f32}, pk->second.hi);
                    }
                    continue;
                }
                const Id w = as_u(value(256 + in.vsrc[pair]));
                if (native_rtz) {  // f16 to f32 is exact
                    const Id r = m.emit(spv::OpFConvert, t_v2f, {m.emit(spv::OpBitcast, t_v2h, {w})});
                    comps[pair * 2] = m.emit(spv::OpCompositeExtract, t_f32, {r, 0u});
                    comps[pair * 2 + 1] = m.emit(spv::OpCompositeExtract, t_f32, {r, 1u});
                    continue;
                }
                comps[pair * 2] = unpack_half(ibin(spv::OpBitwiseAnd, w, cu(0xffff)));
                comps[pair * 2 + 1] = unpack_half(ibin(spv::OpShiftRightLogical, w, cu(16)));
            }
        } else {
            for (std::uint32_t k = 0; k < 4; ++k) {
                if ((in.dmask >> k) & 1) comps[k] = as_f(value(256 + in.vsrc[k]));
            }
        }
        if (!exact_in(reads, exec_lane())) {  // the pixels that survive have their EXEC bit set here (checked at s_endpgm)
            reject("an export reads a value the lift may hold differently from GCN in pixels of EXEC (written in whole-quad mode)");
            return;
        }
        if (vertex) {  // the translator stores all four components of the position and of a param
            const Id value = m.emit(spv::OpCompositeConstruct, t_v4f, comps);
            if (in.tgt == 12) {
                m.store(out_position(), value);
            } else {
                for (const Id var : param_outputs(static_cast<int>(in.tgt) - 32)) m.store(var, value);
            }
            exports.push_back(in.offset);
            return;
        }
        const Id var = out_param(in.tgt);
        if (in.dmask == 0xf) {
            m.store(var, m.emit(spv::OpCompositeConstruct, t_v4f, comps));
        } else {
            Id cur_v = m.load(t_v4f, var);
            for (std::uint32_t k = 0; k < 4; ++k) {
                if ((in.dmask >> k) & 1) cur_v = m.emit(spv::OpCompositeInsert, t_v4f, {comps[k], cur_v, k});
            }
            m.store(var, cur_v);
        }
        exports.push_back(in.offset);
    }

    // ---- cross-lane: ds_swizzle_b32 ---------------------------------------------------
    // The swizzle reads the VGPR in its address field and writes no LDS. Quad
    // mode (offset[15]): lane k of each quad reads lane offset[2k+1:2k]. Bit-mask
    // mode: lane i of each 32 reads ((i & and) | or) ^ xor, and_mask in
    // offset[4:0], or_mask [9:5], xor_mask [14:10]. An inactive lane reads as 0
    // on GCN; here every lane of the quad is active (whole-quad mode, EXEC set),
    // so the swizzle is a permutation within the quad, which Vulkan's quad
    // operations name by the same index GCN's lanes have.
    std::size_t swizzles = 0;
    Id quad_right = 0, quad_bottom = 0;  // this invocation's column and row in its quad
    void quad_caps() {
        m.capability(spv::CapGroupNonUniform);
        m.capability(spv::CapGroupNonUniformQuad);
    }
    // Lane 1 of a quad is right of lane 0 and lane 2 below it (Vulkan's quad
    // scope instance), so a pixel is in the right column where its x exceeds
    // its horizontal neighbour's, and in the bottom row likewise in y. Made once,
    // in the entry block, where every invocation runs.
    void quad_place() {
        if (quad_right) return;
        EntryScope entry(m);
        if (!in_frag_coord) in_frag_coord = builtin(p_in_v4f, spv::BiFragCoord);
        const Id fc = m.load(t_v4f, in_frag_coord);
        const Id x = extract(t_f32, fc, 0), y = extract(t_f32, fc, 1);
        quad_right = m.emit(spv::OpFOrdGreaterThan, t_bool, {x, quad_swap(t_f32, x, 0)});
        quad_bottom = m.emit(spv::OpFOrdGreaterThan, t_bool, {y, quad_swap(t_f32, y, 1)});
    }
    // QuadSwap direction 0 swaps lanes 0-1 and 2-3, 1 swaps 0-2 and 1-3, 2 swaps 0-3 and 1-2.
    Id quad_swap(Id type, Id v, std::uint32_t direction) {
        quad_caps();
        return m.emit(spv::OpGroupNonUniformQuadSwap, type, {cu(spv::ScopeSubgroup), v, cu(direction)});
    }
    // The lane of its quad each lane reads, or false where a lane reads outside
    // its quad (`why` says how).
    static bool swizzle_quad_lanes(std::uint32_t off, std::uint32_t sel[4], std::string& why) {
        if (off & 0x8000) {
            if (off & 0x7f00) {
                why = "ds_swizzle_b32 offset " + hex_offset(off) + ": quad mode with bits 14:8 set (the later rotate and FFT modes) is not lifted";
                return false;
            }
            for (std::uint32_t k = 0; k < 4; ++k) sel[k] = (off >> (2 * k)) & 3;
            return true;
        }
        const std::uint32_t and_mask = off & 0x1f, or_mask = (off >> 5) & 0x1f, xor_mask = (off >> 10) & 0x1f;
        // Lane i keeps its quad (bits 2-4 of i) only when the masks keep those bits.
        if ((and_mask & 0x1c) != 0x1c || (or_mask & 0x1c) || (xor_mask & 0x1c)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "ds_swizzle_b32 bit-mask mode (and 0x%02x, or 0x%02x, xor 0x%02x) reads lanes outside the quad, which Vulkan does not lay out",
                          and_mask, or_mask, xor_mask);
            why = buf;
            return false;
        }
        for (std::uint32_t k = 0; k < 4; ++k) sel[k] = (((k & and_mask) | or_mask) ^ xor_mask) & 3;
        return true;
    }
    void ds(const Inst& in) {
        if (in.op != 53) {
            reject(std::string(mnemonic(in) ? mnemonic(in) : "unknown") + " is not lifted");
            return;
        }
        if (opt.stage != Stage::Pixel) {
            reject("ds_swizzle_b32 outside a pixel shader");
            return;
        }
        const std::uint32_t off = in.offset0 | (static_cast<std::uint32_t>(in.offset1) << 8);
        std::uint32_t sel[4];
        std::string why;
        if (!swizzle_quad_lanes(off, sel, why)) {
            reject(why);
            return;
        }
        if (in.gds) {
            reject("ds_swizzle_b32 with GDS set");
            return;
        }
        // Every lane of the quad active on GCN: EXEC known set, and in whole-quad
        // mode (not the coverage a pixel shader starts with, where the helper
        // lanes are inactive and read as 0).
        const Val e = exec_lane();
        if (!(e.constant && e.set)) {
            reject("ds_swizzle_b32 where EXEC is not known set: a lane outside EXEC reads as 0 on GCN");
            return;
        }
        if (e.helper_inexact) {
            reject("ds_swizzle_b32 outside whole-quad mode: the quad's helper lanes are inactive on GCN and read as 0");
            return;
        }
        if (std::any_of(kill_proofs.begin(), kill_proofs.end(), [](const KillProof& k) { return k.per_pixel; })) {
            reject("ds_swizzle_b32 after a kill region where the lift may differ from GCN in pixels it kills: their quad reads them");
            return;
        }
        // In a loop pixels leave at different iterations, the rest of the quad
        // may have left: the lift's quad operation would not see their values.
        if (in_divergent_loop()) {
            reject("ds_swizzle_b32 inside a loop pixels leave at different iterations");
            return;
        }
        const Val src = value(256 + in.vaddr);
        if (const auto init = initials.find(256 + in.vaddr);
            init != initials.end() && init->second.constant && src.constant && src.kind == init->second.kind && src.bits == init->second.bits) {
            reject("ds_swizzle_b32 of v" + std::to_string(in.vaddr) + ", a pixel shader input the lift does not model (a barycentric or other SPI input)");
            return;
        }
        if (src.helper_inexact) {
            reject("ds_swizzle_b32 of v" + std::to_string(in.vaddr) +
                   ", which the quad's helper lanes may hold differently on GCN (written outside whole-quad mode)");
            return;
        }
        ++swizzles;
        // A value every invocation holds alike is the same in every lane of the quad.
        if (uniform(src) || (sel[0] == 0 && sel[1] == 1 && sel[2] == 2 && sel[3] == 3)) {
            write_v(in.dst, src);
            return;
        }
        const Id type = src.kind == Kind::Float ? t_f32 : t_u32;
        const Id v = src.kind == Kind::Float ? as_f(src) : as_u(src);
        Id r;
        if (sel[0] == sel[1] && sel[1] == sel[2] && sel[2] == sel[3]) {
            quad_caps();
            r = m.emit(spv::OpGroupNonUniformQuadBroadcast, type, {cu(spv::ScopeSubgroup), v, cu(sel[0])});
        } else {
            // Lane k reads lane k ^ d_k: itself (0) or a swap (1 across, 2 down, 3 diagonal).
            Id swapped[4] = {v, 0, 0, 0};
            for (std::uint32_t k = 0; k < 4; ++k) {
                const std::uint32_t d = sel[k] ^ k;
                if (d && !swapped[d]) swapped[d] = quad_swap(type, v, d - 1);
            }
            const std::uint32_t d0 = sel[0] ^ 0, d1 = sel[1] ^ 1, d2 = sel[2] ^ 2, d3 = sel[3] ^ 3;
            if (d0 == d1 && d1 == d2 && d2 == d3) {
                r = swapped[d0];
            } else {
                quad_place();
                const auto pick = [&](Id cond, std::uint32_t a, std::uint32_t b) {  // a where cond, b elsewhere
                    return a == b ? swapped[a] : m.emit(spv::OpSelect, type, {cond, swapped[a], swapped[b]});
                };
                const Id top = pick(quad_right, d1, d0);
                const Id bottom = d2 == d0 && d3 == d1 ? top : pick(quad_right, d3, d2);
                r = top == bottom ? top : m.emit(spv::OpSelect, type, {quad_bottom, bottom, top});
            }
        }
        write_v(in.dst, src.kind == Kind::Float ? flt(r) : word(r));
    }

    void lift_inst(const Inst& in) {
        switch (in.enc) {
        case Enc::SOP1: sop1(in); break;
        case Enc::SOP2: sop2(in); break;
        case Enc::SOPC: sopc(in); break;
        case Enc::SOPK: sopk(in); break;
        case Enc::SOPP: sopp(in); break;
        case Enc::SMRD: smrd(in); break;
        case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3: valu(in); break;
        case Enc::VINTRP: vintrp(in); break;
        case Enc::MIMG: mimg(in); break;
        case Enc::MTBUF: mtbuf(in); break;
        case Enc::EXP: exp(in); break;
        case Enc::DS: ds(in); break;
        default: reject(std::string(mnemonic(in) ? mnemonic(in) : "unknown") + " is not lifted"); break;
        }
    }

    // ---- scalar liveness after a region ---------------------------------------------
    // Scalar state an instruction reads and redefines, for the instructions the
    // lifter accepts. False for anything else (treated as reading everything).
    static bool scalar_rw(const Inst& in, std::set<int>& r, std::set<int>& w) {
        const auto key = [](std::uint16_t c) -> int {
            if (c < 104) return c;
            if (c >= 112 && c < 124) return 200 + c - 112;
            if (c == kM0) return kKeyM0;
            if (c == kExecLo || c == kExecHi) return kKeyExec;
            if (c == kVccLo || c == kVccHi) return kKeyVcc;
            if (c == kScc) return kKeyScc;
            return -1;
        };
        const auto rd = [&](std::uint16_t c) {
            if (const int k = key(c); k >= 0) r.insert(k);
            if (c == kVccz) r.insert(kKeyVcc);
            if (c == kExecz) r.insert(kKeyExec);
        };
        const auto rd_pair = [&](std::uint16_t c) {
            rd(c);
            if (c < 103) rd(static_cast<std::uint16_t>(c + 1));
        };
        const auto wr = [&](std::uint16_t c) {
            if (const int k = key(c); k >= 0) w.insert(k);
        };
        const auto wr_pair = [&](std::uint16_t c) {
            wr(c);
            if (c < 103) wr(static_cast<std::uint16_t>(c + 1));
        };
        switch (in.enc) {
        case Enc::SOP1:
            if (in.op == 3) { rd(in.src0); wr(in.dst); return true; }
            if (in.op == 4) { rd_pair(in.src0); wr_pair(in.dst); return true; }
            if (in.op == 10) { rd_pair(in.src0); wr_pair(in.dst); w.insert(kKeyScc); return true; }
            if (in.op >= 36 && in.op <= 43) { rd_pair(in.src0); r.insert(kKeyExec); wr_pair(in.dst); w.insert(kKeyExec); w.insert(kKeyScc); return true; }
            return false;
        case Enc::SOP2:
            if (in.op == 15 || in.op == 17 || in.op == 19 || in.op == 21 || in.op == 23) {
                rd_pair(in.src0); rd_pair(in.src1); wr_pair(in.dst); w.insert(kKeyScc); return true;
            }
            if (in.op == 10) { rd(in.src0); rd(in.src1); r.insert(kKeyScc); wr(in.dst); return true; }  // s_cselect_b32
            if (const int kind = sop2_word_kind(in.op)) {
                rd(in.src0); rd(in.src1); wr(in.dst);
                if (kind == 1) w.insert(kKeyScc);
                return true;
            }
            return false;
        case Enc::SOPC:
            rd(in.src0);
            rd(in.src1);
            w.insert(kKeyScc);
            return in.op <= 13;
        case Enc::SOPK:
            if (in.op == 0 || in.op == 18) { wr(in.dst); return true; }
            if (in.op == 2) { rd(in.dst); r.insert(kKeyScc); wr(in.dst); return true; }
            if (in.op >= 3 && in.op <= 14) { rd(in.dst); w.insert(kKeyScc); return true; }
            if (in.op == 15 || in.op == 16) { rd(in.dst); wr(in.dst); if (in.op == 15) w.insert(kKeyScc); return true; }
            return in.op == 19 || in.op == 21;
        case Enc::SOPP:
            if (in.op == 1 || in.op == 8) r.insert(kKeyExec);
            if (in.op == 4 || in.op == 5) r.insert(kKeyScc);
            if (in.op == 6 || in.op == 7) r.insert(kKeyVcc);
            return in.op == 0 || in.op == 1 || in.op == 2 || (in.op >= 4 && in.op <= 8) || in.op == 12 ||
                   (in.op >= 13 && in.op <= 22 && in.op != 18);
        case Enc::SMRD: {
            const int count = in.op < 8 ? (1 << in.op) : (in.op < 13 ? (1 << (in.op - 8)) : 0);
            if (!count) return false;
            rd_pair(in.src0);
            if (!in.imm_flag && !in.has_literal) rd(static_cast<std::uint16_t>(in.imm));
            for (int k = 0; k < count; ++k) wr(static_cast<std::uint16_t>(in.sdst + k));
            return true;
        }
        case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3:
            if ((in.enc == Enc::VOP2 && (in.op == 1 || in.op == 2)) || (in.enc == Enc::VOP3 && (in.op == 0x101 || in.op == 0x102))) {
                // v_readlane_b32 / v_writelane_b32: the lane operand is a scalar code, and the cell is state of its own
                const std::uint16_t lane_code = in.enc == Enc::VOP3 ? in.src1 : static_cast<std::uint16_t>(in.src1 - 256);
                int ln = 0;
                if (!const_lane(lane_code, ln)) return false;  // any lane: not modelled
                if (in.op == 2 || in.op == 0x102) {
                    rd(in.src0);
                    w.insert(cell_key(in.dst, ln));
                } else {
                    if (in.src0 >= 256) r.insert(cell_key(in.src0 - 256, ln));
                    wr(in.dst);
                }
                return true;
            }
            if (in.enc == Enc::VOP3 && in.op == 0x161) {  // v_lshl_b64: a 64-bit first operand
                rd_pair(in.src0);
                rd(in.src1);
                return true;
            }
            rd(in.src0);
            if (in.enc != Enc::VOP1) rd(in.src1);
            if (in.enc == Enc::VOP3) {
                if (in.op >= 0x140 && in.op < 0x160) rd(in.src2);
                if (in.op == 0x100 || (in.op >= 0x128 && in.op <= 0x12a)) rd_pair(in.src2);  // select mask, carry in
                if (in.op < 0x100 || (in.op >= 0x125 && in.op <= 0x12a)) wr_pair(in.sdst);   // comparison, carry out
            }
            if (in.enc == Enc::VOPC || (in.enc == Enc::VOP2 && in.op >= 37 && in.op <= 42)) w.insert(kKeyVcc);
            if (in.enc == Enc::VOP2 && (in.op == 0 || (in.op >= 40 && in.op <= 42))) r.insert(kKeyVcc);
            return true;
        case Enc::MIMG:
            for (int k = 0; k < (in.r128 ? 4 : 8); ++k) rd(static_cast<std::uint16_t>(in.srsrc + k));
            for (int k = 0; k < 4; ++k) rd(static_cast<std::uint16_t>(in.ssamp + k));
            return true;
        case Enc::MTBUF:
            if (in.op >= 4) return false;
            rd_pair(in.srsrc);  // the V# the reference traced to its binding
            return true;
        case Enc::VINTRP: case Enc::EXP:
            return true;
        case Enc::DS:
            return in.op == 53;  // ds_swizzle_b32: no scalar operand
        default:
            return false;
        }
    }

    // The EXEC writes a kill region may make: its first instruction masks EXEC
    // with the guard, the second may widen it, and the last may restore it.
    bool kill_exec_write(Region& r, const Inst& in) {
        if (in.offset == r.start) {
            std::uint16_t other = 0xffff;
            if (in.enc == Enc::SOP2 && in.op == 15 && in.dst == kExecLo) {
                if (in.src0 == kExecLo) other = in.src1;
                else if (in.src1 == kExecLo) other = in.src0;
            }
            if (other == 0xffff || !same_lane(read_lane(other), r.guard)) {
                reject("the s_cbranch_scc0 region does not begin by masking EXEC with the mask SCC tested");
                return false;
            }
            return true;
        }
        if (in.offset == r.start + 4 && in.enc == Enc::SOP1 && in.op == 10 && in.dst == kExecLo && in.src0 == kExecLo) {
            r.widened = true;
            widen_ok = true;
            return true;
        }
        return false;  // later EXEC writes follow the region rule in write_lane
    }

    const Inst* inst_at(std::uint32_t offset) const {
        const auto it = std::lower_bound(prog.insts.begin(), prog.insts.end(), offset,
                                         [](const Inst& in, std::uint32_t off) { return in.offset < off; });
        return it != prog.insts.end() && it->offset == offset ? &*it : nullptr;
    }
    Construct* construct_at(std::uint32_t offset) {
        for (Construct& c : constructs) {
            if (c.branch == offset) return &c;
        }
        return nullptr;
    }

    // ---- a uniform condition under EXEC -----------------------------------------------
    // A VGPR written under a varying EXEC with a value every lane holds alike
    // holds that value wherever that EXEC is set.
    struct MaskedUniform {
        Val exec, value;
    };
    std::map<Id, MaskedUniform> masked_uniform;
    // A comparison's ballot that is a condition every lane holds alike ANDed
    // with EXEC: the comparison of operands that are uniform where `exec` is set.
    struct CompareSplit {
        Val exec;
        std::uint32_t op = 0;
        Val u0, u1;
        Mods md;
        Id cond = 0;  // the uniform condition, once emitted
    };
    std::map<Id, CompareSplit> compare_splits;
    Val vcc_bit;              // what the s_cbranch_vccz/vccnz being lifted tests (vcc_branch_bit)
    std::uint32_t vcc_split = 0;
    static bool empty(const Val& v) { return !v.constant && !v.id; }
    // What `v` holds wherever `e` is set, if every lane holds it alike there.
    Val uniform_where(const Val& v, const Val& e) const {
        if (uniform(v)) return v;
        if (v.constant || v.kind == Kind::Lane) return Val{};
        if (const auto it = masked_uniform.find(v.id); it != masked_uniform.end() && implies(e, it->second.exec)) return it->second.value;
        return Val{};
    }
    void note_compare_split(const Val& ballot, const Val& e, std::uint32_t op, const Val& s0, const Val& s1, const Mods& md, Id cond) {
        if (ballot.constant || e.constant) return;
        const Val u0 = uniform_where(s0, e), u1 = uniform_where(s1, e);
        if (empty(u0) || empty(u1)) return;
        CompareSplit s;
        s.exec = e;
        s.op = op;
        s.u0 = u0;
        s.u1 = u1;
        s.md = md;
        if (uniform(s0) && uniform(s1)) s.cond = cond;  // the comparison itself is uniform
        compare_splits[ballot.id] = s;
    }
    // GCN cannot reach this point with `e` empty in the whole wave: `e` is EXEC
    // where the innermost open region began, which its s_cbranch_execz tested
    // (or which a kill region narrowed to a guard that its s_cbranch_scc0 tested).
    const Region* region_tested(const Val& e) const {
        if (open_regions.empty()) return nullptr;
        const Region& r = *open_regions.back();
        if (!same_lane(e, r.exec_in)) return nullptr;
        if (r.kill && !implies(r.guard, r.exec_before)) return nullptr;
        return &r;
    }
    // The bit an s_cbranch_vccz/vccnz at this point tests, when every lane
    // holds it alike: VCC itself, or the uniform condition of a ballot ANDed
    // with an EXEC GCN cannot reach the branch with empty (its wave-wide test
    // is then the condition). `split_region`: the region that EXEC began.
    bool vcc_branch_bit(Val& bit, std::uint32_t& split_region) {
        split_region = 0;
        const auto v = lanes.find(kKeyVcc);
        if (v == lanes.end() || poisoned.count(kKeyVcc)) return false;
        if (uniform(v->second)) {
            bit = v->second;
            return true;
        }
        const auto s = v->second.constant ? compare_splits.end() : compare_splits.find(v->second.id);
        const auto e = lanes.find(kKeyExec);
        if (s == compare_splits.end() || e == lanes.end() || !same_lane(e->second, s->second.exec)) return false;
        const Region* r = region_tested(e->second);
        if (!r) return false;
        if (!s->second.cond) s->second.cond = vcmp_cond(s->second.op, s->second.u0, s->second.u1, s->second.md);
        bit = lane(s->second.cond);
        split_region = r->start;
        return true;
    }
    std::string if_condition_proof(const Arm& a) const {
        if (a.c->op <= 5) return "SCC from a scalar compare";
        if (!a.split_region) return "VCC computed only from constants, user data and loads";
        return "VCC is a comparison of values every lane holds alike where EXEC is set, ANDed with EXEC as it was when the region at " +
               hex_offset(a.split_region) + " began, which GCN cannot reach empty; the wave-wide test is the comparison";
    }

    // An if on a scalar condition: a selection whose then arm runs where the
    // branch falls through. Every lane takes the same arm, so the arms are
    // plain structured control flow.
    void open_arm(Construct& c, std::uint64_t seq) {
        Arm a;
        a.c = &c;
        a.seq = seq;
        a.merge = m.fresh();
        a.then_label = m.fresh();
        a.else_label = m.fresh();
        a.reg0 = reg;
        a.lanes0 = lanes;
        a.to_float0 = to_float;
        a.to_word0 = to_word;
        a.scc0 = scc_mask;
        a.scc_uniform0 = scc_uniform;
        a.scc_valid0 = scc_valid;
        a.poisoned0 = poisoned;
        exact_open_arm();
        // s_cbranch_scc0 and s_cbranch_vccz jump where their bit is clear, so the then arm runs where it is set;
        // s_cbranch_scc1 and s_cbranch_vccnz the other way round.
        const Val bit = c.op <= 5 ? scc_mask : vcc_bit;
        a.split_region = c.op <= 5 ? 0 : vcc_split;
        const Id cond = bit.constant ? m.const_bool(bit.set != c.negate) : c.negate ? m.emit(spv::OpLogicalNot, t_bool, {bit.id}) : bit.id;
        m.emit_void(spv::OpSelectionMerge, {a.merge, 0u});
        m.emit_void(spv::OpBranchConditional, {cond, a.then_label, a.else_label});
        cur_label = m.label(a.then_label);
        arms.push_back(std::move(a));
        ++if_constructs;
    }
    // The then arm ends: its values are kept for the merge, and the else arm
    // (or an empty else block) starts from the state at the branch.
    void then_to_else(Arm& a) {
        a.reg1 = reg;
        a.lanes1 = lanes;
        a.scc1v = scc_mask;
        a.scc_uniform1 = scc_uniform;
        a.scc_valid1 = scc_valid;
        a.poisoned1 = poisoned;
        if (block_dead) {  // it left the loop: no edge to the merge
            a.then_dead = true;
            block_dead = false;
        } else {
            m.emit_void(spv::OpBranch, {a.merge});
        }
        a.then_end = cur_label;
        reg = a.reg0;
        lanes = a.lanes0;
        to_float = a.to_float0;
        to_word = a.to_word0;
        scc_mask = a.scc0;
        scc_uniform = a.scc_uniform0;
        scc_valid = a.scc_valid0;
        poisoned = a.poisoned0;
        exact_then_to_else();
        cur_label = m.label(a.else_label);
        a.in_else = true;
    }
    // The join: values the arms left different become phis, typed as the then
    // arm's value (the else arm, still open, converts); conversions made inside
    // the arms are forgotten.
    void merge_arms(Arm& a) {
        if (a.then_dead || block_dead) {
            merge_live_arm(a);
            return;
        }
        const auto same = [](const Val& x, const Val& y) {
            return x.kind == y.kind && x.constant == y.constant && (x.constant ? (x.kind == Kind::Lane ? x.set == y.set : x.bits == y.bits) : x.id == y.id);
        };
        const auto at = [&](const std::map<int, Val>& arm, int key) {
            if (const auto it = arm.find(key); it != arm.end()) return it->second;
            if (const auto it = a.reg0.find(key); it != a.reg0.end()) return it->second;
            return initial_value(key);
        };
        struct Phi {
            int key;
            Kind kind;
            Id then_value, else_value;
            bool uni;  // both arms' values are uniform, and so is the condition
        };
        std::vector<Phi> phis;
        std::set<int> lane_keys;
        for (const auto& [k, v] : a.lanes1) lane_keys.insert(k);
        for (const auto& [k, v] : lanes) lane_keys.insert(k);
        std::map<int, Val> merged_lanes;
        std::set<int> lost = a.poisoned1;
        lost.insert(poisoned.begin(), poisoned.end());
        for (int k : lane_keys) {
            const auto t = a.lanes1.find(k), e = lanes.find(k);
            if (t == a.lanes1.end() || e == lanes.end()) {
                // A lane mask on one arm only: unknown after the join. EXEC and VCC
                // have no word form, so a missing mask already rejects their reads;
                // an SGPR pair is poisoned, words and mask alike.
                if (k != kKeyVcc && k != kKeyExec) {
                    lost.insert(k);
                    lost.insert(k + 1);
                }
                continue;
            }
            if (same(t->second, e->second)) {
                merged_lanes[k] = t->second;
            } else {
                phis.push_back({k, Kind::Lane, t->second.id, e->second.id, uniform(t->second) && uniform(e->second)});
            }
        }
        std::set<int> keys;
        for (const auto& [k, v] : a.reg1) keys.insert(k);
        for (const auto& [k, v] : reg) keys.insert(k);
        std::map<int, Val> merged;
        for (int k : keys) {
            if (lane_keys.count(k) || (k > 0 && k < 104 && lane_keys.count(k - 1)) || lost.count(k)) continue;  // a lane mask's words, or poisoned
            const Val t = at(a.reg1, k), e = at(reg, k);
            if (same(t, e)) {
                merged[k] = t;
                continue;
            }
            if (descriptor(t) || descriptor(e)) {  // a descriptor word on either arm: still never data
                merged[k] = descriptor(t) ? t : e;
                continue;
            }
            phis.push_back({k, t.kind, t.id, t.kind == Kind::Float ? as_f(e) : as_u(e), uniform(t) && uniform(e)});
        }
        const bool scc_same = a.scc_valid1 && scc_valid && a.scc_uniform1 == scc_uniform && same(a.scc1v, scc_mask);
        const bool scc_phi = !scc_same && a.scc_valid1 && scc_valid && a.scc_uniform1 && scc_uniform;
        const Id else_end = cur_label;
        m.emit_void(spv::OpBranch, {a.merge});
        cur_label = m.label(a.merge);
        for (const Phi& p : phis) {
            const Id type = p.kind == Kind::Lane ? t_bool : p.kind == Kind::Float ? t_f32 : t_u32;
            const Id id = m.emit(spv::OpPhi, type, {p.then_value, a.then_end, p.else_value, else_end});
            if (p.uni) uniform_ids.insert(id);
            if (p.kind == Kind::Lane) {
                merged_lanes[p.key] = lane(id);
            } else {
                merged[p.key] = p.kind == Kind::Float ? flt(id) : word(id);
            }
        }
        if (scc_phi) {
            scc_mask = lane(m.emit(spv::OpPhi, t_bool, {a.scc1v.id, a.then_end, scc_mask.id, else_end}));
        } else if (!scc_same) {
            scc_valid = false;
        }
        reg = std::move(merged);
        poisoned = std::move(lost);
        exact_merge_arms();
        lanes = std::move(merged_lanes);
        to_float = a.to_float0;
        to_word = a.to_word0;
        static const char* const kBranch[4] = {"s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz"};
        res.proof.push_back("if at " + hex_offset(a.c->branch) + " (" + kBranch[a.c->op - 4] + "): " + if_condition_proof(a) +
                            ", the same in every lane; " + (a.c->else_jump ? "then and else arms" : "a then arm") +
                            " joined at " + hex_offset(a.c->join) + " with " + std::to_string(phis.size()) + " phis" +
                            (std::count(threaded_joins.begin(), threaded_joins.end(), a.c->branch)
                                 ? " (its arms jump past this s_branch to the join it jumps to: the same program)"
                                 : ""));
    }
    // One arm of the if left the loop it is in: the join has the other arm's
    // values, from its one edge.
    void merge_live_arm(Arm& a) {
        const bool else_dead = block_dead;
        block_dead = false;
        if (else_dead) partial = std::move(arm_exact.back().then_end);  // where the live arm's registers are exact
        arm_exact.pop_back();
        if (a.then_dead && else_dead) {
            reject("both arms of the if at " + hex_offset(a.c->branch) + " leave the loop");
            return;
        }
        if (!else_dead) m.emit_void(spv::OpBranch, {a.merge});
        cur_label = m.label(a.merge);
        if (else_dead) {
            reg = a.reg1;
            lanes = a.lanes1;
            scc_mask = a.scc1v;
            scc_uniform = a.scc_uniform1;
            scc_valid = a.scc_valid1;
            poisoned = a.poisoned1;
        }
        to_float = a.to_float0;
        to_word = a.to_word0;
        static const char* const kBranch[4] = {"s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz"};
        res.proof.push_back("if at " + hex_offset(a.c->branch) + " (" + kBranch[a.c->op - 4] + "): " + if_condition_proof(a) +
                            ", the same in every lane; its " + (else_dead ? "else" : "then") + " arm leaves the loop, so the join at " +
                            hex_offset(a.c->join) + " has the other arm's values");
    }
    void close_arm() {
        Arm& a = arms.back();
        if (!a.in_else) then_to_else(a);  // an if without an else arm: an empty else block
        // Helper lanes after the join: inexact where either arm left them so
        // (merge_arms makes its phis without the flag). An arm that left the
        // loop it is in does not reach the join: merge_live_arm keeps the
        // other arm's values, flags and all.
        std::set<int> inexact_regs, inexact_lanes;
        const auto collect = [&](const std::map<int, Val>& then_end, const std::map<int, Val>& else_end, const std::map<int, Val>& at_branch,
                                 std::set<int>& out) {
            const auto flag = [&](const std::map<int, Val>& arm, int k) {
                if (const auto it = arm.find(k); it != arm.end()) return it->second.helper_inexact;
                const auto it = at_branch.find(k);
                return it != at_branch.end() && it->second.helper_inexact;  // else the value at entry: exact
            };
            for (const auto* arm : {&then_end, &else_end}) {
                for (const auto& kv : *arm) {
                    if (flag(then_end, kv.first) || flag(else_end, kv.first)) out.insert(kv.first);
                }
            }
        };
        if (!a.then_dead && !block_dead) {
            collect(a.reg1, reg, a.reg0, inexact_regs);
            collect(a.lanes1, lanes, a.lanes0, inexact_lanes);
        }
        merge_arms(a);
        for (auto& [k, v] : reg) v.helper_inexact = v.helper_inexact || inexact_regs.count(k) != 0;
        for (auto& [k, v] : lanes) v.helper_inexact = v.helper_inexact || inexact_lanes.count(k) != 0;
        arms.pop_back();
    }

    // ---- liveness over GCN's control flow -------------------------------------------
    // The instructions control can reach next from instruction `i` on GCN: the
    // next one and every branch target, backward ones included. Every path the
    // lift takes is one of these (it runs regions unconditionally, which is
    // the fall-through edge), so a value no path reads is dead in both.
    std::size_t index_of(std::uint32_t offset) const {
        const auto it = std::lower_bound(prog.insts.begin(), prog.insts.end(), offset,
                                         [](const Inst& in, std::uint32_t off) { return in.offset < off; });
        return static_cast<std::size_t>(it - prog.insts.begin());
    }
    void successors(std::size_t i, std::vector<std::size_t>& out) const {
        out.clear();
        const Inst& in = prog.insts[i];
        if (in.enc == Enc::SOPP && in.op == 1) return;  // s_endpgm
        if (!(in.enc == Enc::SOPP && in.op == 2) && i + 1 < prog.insts.size()) out.push_back(i + 1);
        if (is_branch(in)) {
            const std::uint32_t target = in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
            if (const std::size_t t = index_of(target); t < prog.insts.size() && prog.insts[t].offset == target) out.push_back(t);
        }
    }
    // Whether some path from instruction `from` reads one of `keys` before
    // writing it. On a read, `at` is the reading instruction and `key` the key;
    // `unmodelled` is set when the path meets an instruction scalar_rw cannot
    // describe. With `exec_reads`, every vector and memory instruction reads
    // EXEC as well (it runs under it).
    bool read_before_written(std::size_t from, const std::set<int>& keys, bool exec_reads, std::uint32_t& at, int& key,
                             bool& unmodelled) const {
        unmodelled = false;
        std::map<std::size_t, std::set<int>> seen;
        std::vector<std::pair<std::size_t, std::set<int>>> work;
        if (from < prog.insts.size() && !keys.empty()) work.push_back({from, keys});
        std::vector<std::size_t> next;
        while (!work.empty()) {
            auto [i, live] = std::move(work.back());
            work.pop_back();
            std::set<int>& done = seen[i];
            for (auto it = live.begin(); it != live.end();) it = done.count(*it) ? live.erase(it) : std::next(it);
            if (live.empty()) continue;
            done.insert(live.begin(), live.end());
            const Inst& in = prog.insts[i];
            std::set<int> rd, wr;
            if (!scalar_rw(in, rd, wr)) {
                at = in.offset;
                unmodelled = true;
                return true;
            }
            if (exec_reads) {
                switch (in.enc) {
                case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3: case Enc::VINTRP: case Enc::MIMG: case Enc::MTBUF:
                case Enc::MUBUF: case Enc::EXP: case Enc::DS:
                    rd.insert(kKeyExec);
                    break;
                default: break;
                }
            }
            for (int k : live) {
                if (rd.count(k)) {
                    at = in.offset;
                    key = k;
                    return true;
                }
            }
            for (int k : wr) live.erase(k);
            if (live.empty()) continue;
            successors(i, next);
            for (std::size_t j : next) work.push_back({j, live});
        }
        return false;
    }

    void close_region(Region& r, std::size_t next_index) {
        r.closed = true;
        std::set<int> live = r.scalar_writes;
        live.erase(kKeyExec);  // EXEC changes inside regions are rejected where they happen, or handled below
        // GCN skips the region only where the mask it tests is clear in every
        // lane; a lane mask that is the same there either way may stay live.
        const Val skip = r.kill ? r.guard : r.exec_before;
        std::size_t masks_kept = 0;
        for (auto it = live.begin(); it != live.end();) {
            const int k = *it;
            const int pair = k == kKeyVcc || (k < 104 && lanes.count(k)) ? k : k > 0 && k < 104 && lanes.count(k - 1) ? k - 1 : -1;
            const auto now = pair >= 0 ? lanes.find(pair) : lanes.end();
            const auto before = pair >= 0 ? r.lanes_at_branch.find(pair) : r.lanes_at_branch.end();
            if (now != lanes.end() && before != r.lanes_at_branch.end() && same_where_clear(now->second, before->second, skip)) {
                it = live.erase(it);
                ++masks_kept;
            } else {
                ++it;
            }
        }
        std::string exec_after;
        bool divergent = false;
        if (r.kill) {
            // Skipped, the region leaves EXEC as it was at the branch. Where the
            // lifted EXEC differs, or the region widened it, the lift can differ
            // from GCN only in pixels whose guard bit is clear.
            const auto it = lanes.find(kKeyExec);
            const bool same = it != lanes.end() && same_lane(it->second, r.exec_before);
            if (r.quad_exact) {
                exec_after = "every pixel computed the region, so the lift differs from GCN only in quads where no pixel survives";
                kill_proofs.push_back({r.guard, false});
                divergent = true;
            } else {
                exec_after = same ? "EXEC is as it was at the branch" : "EXEC differs from the skipped path only where the guard bit is clear";
                divergent = r.widened || !same;
                if (divergent) kill_proofs.push_back({r.guard, true});
            }
            ++kill_regions;
        }
        // Every path out of the region, through the rest of the program and
        // around loops, must write each of them before reading it.
        std::uint32_t read_at = 0;
        int read_key = 0;
        bool unmodelled = false;
        if (read_before_written(next_index, live, false, read_at, read_key, unmodelled)) {
            cur = read_at;
            const Inst* in = inst_at(read_at);
            if (unmodelled) {
                reject("cannot show the scalar writes of region " + hex_offset(r.start) + "-" + hex_offset(r.end) + " are dead: " +
                       (in && mnemonic(*in) ? mnemonic(*in) : "unknown") + " is not modelled");
            } else {
                reject(key_name(read_key) + " is written inside region " + hex_offset(r.start) + "-" + hex_offset(r.end) +
                       " (where GCN runs the block for every lane if any lane needs it) and read here");
            }
            return;
        }
        std::string names;
        for (int k : r.scalar_writes) names += (names.empty() ? "" : ", ") + key_name(k);
        std::string how;
        if (!r.kill) {
            how = ": writes masked by EXEC at " + hex_offset(r.branch) + ", and EXEC set inside only within it";
        } else if (r.quad_exact) {
            how = ": s_cbranch_scc0 at " + hex_offset(r.branch) + " tests the mask the region ANDs into EXEC and widens to whole quads; " +
                  exec_after + "; s_endpgm kills exactly the guard's clear pixels";
        } else {
            how = ": s_cbranch_scc0 at " + hex_offset(r.branch) + " tests the mask the region ANDs into EXEC first" +
                  (r.widened ? " and widens (each pixel keeps its own bit)" : ", so a pixel whose bit is clear writes nothing either way") + "; " +
                  exec_after + (divergent ? "; s_endpgm kills exactly the guard's clear pixels and no implicit-LOD sample follows" : "");
        }
        res.proof.push_back("region " + hex_offset(r.start) + "-" + hex_offset(r.end) + how +
                            "; samples with the operands they read GCN's (see the samples above), no memory writes" +
                            (names.empty() ? "" : "; scalar writes (" + names + ") are redefined before any later read on every path") +
                            (masks_kept ? "; " + std::to_string(masks_kept) +
                                              " lane masks written inside equal, wherever GCN can skip the region, their value on the skipped path"
                                        : ""));
    }

    // ---- loops -----------------------------------------------------------------------
    // A loop is a backward s_branch (its back edge, at `latch`) to its header.
    // Control leaves it only for the instruction after the back edge (`exit`),
    // from branches inside it. The lift is one OpLoopMerge loop per pixel: the
    // registers the body writes become phis at the header, and the one exit
    // carries its values to the merge.
    //
    //   * A uniform loop exits on a bit every lane holds alike (SCC from a
    //     scalar compare, a uniform VCC, a constant): every pixel runs the
    //     iterations GCN's wave runs. EXEC, and every lane mask an iteration
    //     reads before writing, must come back to the header as they left it
    //     (or, on a second attempt, a lane mask gets a phi).
    //   * A divergent loop is the compiler's per-lane exit idiom,
    //         s_andn2_b64 L, L, exec   ; L loses the lanes leaving (EXEC holds them)
    //         s_cbranch_scc0 exit      ; no lane left in L: the wave is done
    //     with L equal to EXEC where the loop begins. A pixel leaves when its
    //     own L bit clears. On GCN it then runs the remaining iterations with
    //     its EXEC bit clear (L only loses lanes, and EXEC in the body stays
    //     within L), so its VGPRs keep what they held when it left. After the
    //     exit test the body knows the pixel's L bit is set, and so is EXEC as
    //     it was at entry (L started as it); a pixel back at the header has
    //     EXEC and L set, which is EXEC at entry again, so both keep that value
    //     there and must come back set. Everything scalar the loop writes
    //     (SGPRs, SCC, VCC, EXEC) must be dead after it, as GCN may run more
    //     iterations than the pixel did, and no implicit-LOD sample or other
    //     cross-lane read may run inside (the rest of its quad may have left).
    struct Loop {
        std::uint32_t header = 0, latch = 0, exit = 0;
        std::size_t header_index = 0, latch_index = 0;
        int mask = -1;                // divergent: the loop mask's SGPR pair
        std::uint32_t mask_exit = 0;  // divergent: the s_cbranch_scc0 that tests it
        // While lifting
        Id preheader = 0, header_label = 0, continue_label = 0, merge_label = 0;
        struct Phi {
            int key;
            Kind kind;
            Id id;
            bool uniform;         // a scalar from values every lane holds alike: checked at the back edge
            bool helper_inexact;  // Val::helper_inexact at the header: the back edge's value may not be less exact
        };
        std::vector<Phi> phis;
        std::map<int, Val> unchanged;  // lane masks assumed to come back to the header as they left it
        std::set<int> writes;          // scalar keys the body writes (scalar_rw)
        std::size_t vgpr_phis = 0, lane_phis = 0;
        struct Exit {
            Id from = 0;
            std::map<int, Val> reg, lanes;
            Val scc;
            bool scc_uniform = false, scc_valid = true;
            std::set<int> poisoned;
            bool per_pixel = false;
        };
        std::vector<Exit> exits;
        std::map<Id, Id> to_float0, to_word0;
        std::vector<Id> assumed;  // divergent: what the body knows is set
        std::size_t arms0 = 0, regions0 = 0;
    };
    std::vector<Loop> loops;  // by header
    std::vector<Loop*> open_loops;
    std::set<std::uint32_t> loop_exits;  // branches that leave their innermost loop for its exit
    std::set<Id> assumed;                // lane masks set wherever the lift now is (a divergent loop's body)
    bool block_dead = false;             // the current block ended by leaving a loop
    const LoopPhis lane_phi_hints, inexact_hints;
    std::size_t loops_lifted = 0, divergent_loops = 0;

    bool loop_branch(std::uint32_t offset) const {
        if (loop_exits.count(offset)) return true;
        return std::any_of(loops.begin(), loops.end(), [&](const Loop& l) { return l.latch == offset; });
    }
    const Loop* innermost_loop(std::uint32_t offset) const {
        const Loop* best = nullptr;
        for (const Loop& l : loops) {
            if (offset >= l.header && offset <= l.latch && (!best || l.header > best->header)) best = &l;
        }
        return best;
    }
    bool in_divergent_loop() const {
        return std::any_of(open_loops.begin(), open_loops.end(), [](const Loop* l) { return l->mask >= 0; });
    }
    // `v` as the lift knows it here: set where a divergent loop's body knows so.
    // The helper lanes' bits stay as exact as `v`'s were.
    Val known(const Val& v) {
        if (v.constant || v.kind != Kind::Lane || assumed.empty()) return v;
        Val r;
        if (assumed.count(v.id)) {
            r = lane_const(true);
        } else if (const auto it = negation_of.find(v.id); it != negation_of.end() && assumed.count(it->second)) {
            r = lane_const(false);
        } else {
            return v;
        }
        r.helper_inexact = v.helper_inexact;
        return r;
    }

    void find_loops() {
        for (std::size_t i = 0; i < prog.insts.size(); ++i) {
            const Inst& in = prog.insts[i];
            if (!is_branch(in)) continue;
            const std::uint32_t target = in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
            if (target > in.offset) continue;
            cur = in.offset;
            const std::size_t h = index_of(target);
            if (in.op != 2) {
                reject(std::string(mnemonic(in)) + " branches backward: only an s_branch is lifted as a loop's back edge");
            } else if (h >= prog.insts.size() || prog.insts[h].offset != target) {
                reject("s_branch branches backward to no instruction");
            } else {
                Loop l;
                l.header = target;
                l.latch = in.offset;
                l.exit = in.offset + 4;
                l.header_index = h;
                l.latch_index = i;
                loops.push_back(l);
            }
        }
        std::sort(loops.begin(), loops.end(), [](const Loop& a, const Loop& b) { return a.header < b.header; });
        bool nested = true;
        for (std::size_t k = 1; k < loops.size(); ++k) {
            for (std::size_t j = 0; j < k; ++j) {
                const Loop& o = loops[j];
                const Loop& l = loops[k];
                cur = l.latch;
                if (l.header == o.header) {
                    reject("two back edges to the loop header at " + hex_offset(o.header));
                    nested = false;
                } else if (l.header <= o.latch && l.latch > o.latch) {
                    reject("the loops at " + hex_offset(o.header) + " and " + hex_offset(l.header) + " overlap");
                    nested = false;
                }
            }
        }
        if (!nested) return;
        // Every other branch stays inside its innermost loop, or leaves it for its exit.
        for (const Inst& in : prog.insts) {
            if (!is_branch(in)) continue;
            const std::uint32_t target = in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4;
            const Loop* from = innermost_loop(in.offset);
            if ((from && in.offset == from->latch) || target <= in.offset) continue;  // a back edge, or rejected above
            cur = in.offset;
            if (from && target == from->exit) {
                loop_exits.insert(in.offset);
                continue;
            }
            if (from && target > from->latch) reject("a branch leaves the loop at " + hex_offset(from->header) + " for somewhere other than its exit");
            for (const Loop& l : loops) {
                if (target > l.header && target <= l.latch && !(in.offset >= l.header && in.offset <= l.latch)) {
                    reject("a branch enters the loop at " + hex_offset(l.header) + " past its header");
                }
            }
        }
    }
    // Ifs and regions nest with loops: inside the body, around the whole loop, or apart.
    void check_loop_nesting() {
        for (Loop& l : loops) {
            // The divergent exit idiom, at the loop's top level: s_andn2_b64 L,
            // L, exec right before an s_cbranch_scc0 to the exit. (Inside an if
            // arm the same pair is the uniform loop's way out: L is EXEC there,
            // so no lane is left and the branch is always taken.)
            for (std::size_t i = l.header_index + 1; i < l.latch_index; ++i) {
                const Inst& br = prog.insts[i];
                const Inst& andn2 = prog.insts[i - 1];
                if (!(br.enc == Enc::SOPP && br.op == 4 && loop_exits.count(br.offset) && innermost_loop(br.offset) == &l)) continue;
                if (!(andn2.enc == Enc::SOP2 && andn2.op == 21 && andn2.dst < 103 && andn2.src0 == andn2.dst && andn2.src1 == kExecLo)) continue;
                const auto inside = [&](std::uint32_t x0, std::uint32_t x1) { return x0 >= l.header && x0 < br.offset && br.offset < x1; };
                if (std::any_of(constructs.begin(), constructs.end(), [&](const Construct& c) { return inside(c.branch, c.join); }) ||
                    std::any_of(regions.begin(), regions.end(), [&](const Region& r) { return inside(r.branch, r.end); })) {
                    continue;
                }
                if (l.mask >= 0) {
                    cur = br.offset;
                    reject("two divergent exits from the loop at " + hex_offset(l.header));
                }
                l.mask = andn2.dst;
                l.mask_exit = br.offset;
            }
            const auto fits = [&](std::uint32_t x0, std::uint32_t x1) {
                return x1 <= l.header || x0 >= l.exit || (x0 >= l.header && x1 <= l.latch) || (x0 < l.header && x1 >= l.exit);
            };
            for (const Region& r : regions) {
                if (!fits(r.branch, r.end)) {
                    cur = r.branch;
                    reject("a region overlaps the loop at " + hex_offset(l.header));
                }
            }
            for (const Construct& c : constructs) {
                if (!fits(c.branch, c.join)) {
                    cur = c.branch;
                    reject("the if at " + hex_offset(c.branch) + " overlaps the loop at " + hex_offset(l.header));
                }
            }
        }
    }

    // VGPRs an instruction may write (more is harmless: a phi that comes back unchanged).
    static void vector_writes(const Inst& in, std::set<int>& w) {
        const auto range = [&](int first, int n) {
            for (int k = 0; k < n; ++k) w.insert(256 + first + k);
        };
        switch (in.enc) {
        case Enc::VOP1: if (in.op != 0 && in.op != 2) range(in.dst, 1); break;  // not v_nop, v_readfirstlane_b32
        case Enc::VOP2: if (in.op != 1) range(in.dst, 1); break;                // not v_readlane_b32
        case Enc::VOP3: if (in.op >= 0x100 && in.op != 0x101 && in.op != 0x182) range(in.dst, 2); break;  // not a compare; 64-bit results
        case Enc::VINTRP: range(in.dst, 1); break;
        case Enc::MIMG: range(in.vdata, 4); break;
        case Enc::MTBUF: case Enc::MUBUF: range(in.vdata, 4); break;
        case Enc::DS: range(in.dst, 4); break;
        default: break;
        }
    }
    static void exec_reader(const Inst& in, std::set<int>& rd) {
        switch (in.enc) {
        case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3: case Enc::VINTRP: case Enc::MIMG: case Enc::MTBUF:
        case Enc::MUBUF: case Enc::EXP: case Enc::DS:
            rd.insert(kKeyExec);
            break;
        default: break;
        }
    }
    // The keys of `keys` some path from instruction `from` reads before writing
    // (vector and memory instructions read EXEC).
    std::set<int> read_first(std::size_t from, const std::set<int>& keys, bool& unmodelled) const {
        std::set<int> found;
        unmodelled = false;
        std::map<std::size_t, std::set<int>> seen;
        std::vector<std::pair<std::size_t, std::set<int>>> work;
        if (from < prog.insts.size() && !keys.empty()) work.push_back({from, keys});
        std::vector<std::size_t> next;
        while (!work.empty()) {
            auto [i, live] = std::move(work.back());
            work.pop_back();
            std::set<int>& done = seen[i];
            for (auto it = live.begin(); it != live.end();) it = done.count(*it) || found.count(*it) ? live.erase(it) : std::next(it);
            if (live.empty()) continue;
            done.insert(live.begin(), live.end());
            std::set<int> rd, wr;
            if (!scalar_rw(prog.insts[i], rd, wr)) {
                unmodelled = true;
                return found;
            }
            exec_reader(prog.insts[i], rd);
            for (auto it = live.begin(); it != live.end();) {
                if (rd.count(*it)) {
                    found.insert(*it);
                    it = live.erase(it);
                } else {
                    it = wr.count(*it) ? live.erase(it) : std::next(it);
                }
            }
            if (live.empty()) continue;
            successors(i, next);
            for (std::size_t j : next) work.push_back({j, live});
        }
        return found;
    }

    void open_loop(Loop& l) {
        cur = l.header;
        if (block_dead) {
            reject("a loop begins after an unconditional loop exit");
            return;
        }
        std::set<int> vgprs;
        for (std::size_t i = l.header_index; i <= l.latch_index; ++i) {
            std::set<int> rd, wr;
            if (!scalar_rw(prog.insts[i], rd, wr)) {
                cur = prog.insts[i].offset;
                reject(std::string(mnemonic(prog.insts[i]) ? mnemonic(prog.insts[i]) : "unknown") + " inside the loop at " + hex_offset(l.header) +
                       " is not modelled");
                return;
            }
            l.writes.insert(wr.begin(), wr.end());
            vector_writes(prog.insts[i], vgprs);
        }
        bool unmodelled = false;
        const std::set<int> exposed = read_first(l.header_index, l.writes, unmodelled);
        if (unmodelled) {
            reject("cannot show what the loop at " + hex_offset(l.header) + " reads before writing");
            return;
        }
        if (l.writes.count(kKeyScc)) {
            if (exposed.count(kKeyScc)) {
                reject("SCC is read before it is written in an iteration of the loop at " + hex_offset(l.header));
                return;
            }
            scc_valid = false;
        }
        struct Entry {
            int key;
            Kind kind;
            Id id;
            bool uni, inexact;
        };
        // Helper lanes at the header: as exact as on entry, unless an earlier
        // attempt saw them come back less exact.
        const auto inexact_at_header = [&](int k, const Val& v) { return v.helper_inexact || inexact_hints.count({l.header, k}) != 0; };
        std::vector<Entry> words;
        std::vector<Entry> masks;  // lane masks that get phis
        const auto add_word = [&](int k) {
            const Val v = value(k);
            const bool f = v.kind == Kind::Float;
            words.push_back({k, f ? Kind::Float : Kind::Word, f ? v.id : as_u(v), k < 256 && uniform(v), inexact_at_header(k, v)});
        };
        for (int k : vgprs) add_word(k);
        const auto e = lanes.find(kKeyExec);
        Val exec0 = e != lanes.end() ? e->second : lane_const(true);
        for (int k : l.writes) {
            if (k == kKeyScc || k == kKeyExec || k == l.mask || k == l.mask + 1) continue;
            const bool lane_key = k == kKeyVcc || lanes.count(k);
            const bool lane_high = k > 0 && k < 104 && lanes.count(k - 1);
            if (lane_high) continue;
            const bool read = exposed.count(k) || (k < 103 && lane_key && exposed.count(k + 1));
            if (!lane_key) {
                if (read) {
                    add_word(k);
                } else {
                    poisoned.insert(k);  // written before it is read: its value here is never used
                }
                continue;
            }
            const auto it = lanes.find(k);
            if (!read) {
                lanes.erase(k);
                if (k < 104) {
                    poisoned.insert(k);
                    poisoned.insert(k + 1);
                }
            } else if (it == lanes.end()) {
                continue;  // no mask here (VCC's words were written): reading it before writing it rejects
            } else if (lane_phi_hints.count({l.header, k})) {
                masks.push_back({k, Kind::Lane, it->second.constant ? m.const_bool(it->second.set) : it->second.id, false,
                                 inexact_at_header(k, it->second)});
            } else {
                l.unchanged[k] = it->second;
            }
        }
        if (l.writes.count(kKeyExec) && l.mask < 0) l.unchanged[kKeyExec] = exec0;
        if (l.mask >= 0) {
            const auto lm = lanes.find(l.mask);
            if (lm == lanes.end() || !same_lane(lm->second, exec0)) {
                reject("the loop at " + hex_offset(l.header) + " tests " + key_name(l.mask) + " as its loop mask, which does not start as EXEC");
                return;
            }
            l.unchanged[kKeyExec] = exec0;
            l.unchanged[l.mask] = lm->second;
        }
        for (auto& [k, v] : l.unchanged) {  // assumed unchanged, helper lanes included
            if (!inexact_at_header(k, v)) continue;
            v.helper_inexact = true;
            if (const auto it = lanes.find(k); it != lanes.end()) it->second.helper_inexact = true;
        }
        l.preheader = cur_label;
        l.header_label = m.fresh();
        l.continue_label = m.fresh();
        l.merge_label = m.fresh();
        l.to_float0 = to_float;
        l.to_word0 = to_word;
        l.arms0 = arms.size();
        l.regions0 = open_regions.size();
        m.emit_void(spv::OpBranch, {l.header_label});
        cur_label = m.label(l.header_label);
        for (const Entry& w : words) {
            const Id phi = m.emit(spv::OpPhi, w.kind == Kind::Float ? t_f32 : t_u32, {w.id, l.preheader, w.id, l.continue_label});
            reg[w.key] = w.kind == Kind::Float ? flt(phi) : word(phi);
            reg[w.key].helper_inexact = w.inexact;
            if (w.uni) uniform_ids.insert(phi);
            l.phis.push_back({w.key, w.kind, phi, w.uni, w.inexact});
            l.vgpr_phis += w.key >= 256;
        }
        for (const Entry& k : masks) {
            const Id phi = m.emit(spv::OpPhi, t_bool, {k.id, l.preheader, k.id, l.continue_label});
            lanes[k.key] = lane(phi);
            lanes[k.key].helper_inexact = k.inexact;
            l.phis.push_back({k.key, Kind::Lane, phi, false, k.inexact});
            ++l.lane_phis;
        }
        const Id body = m.fresh();
        m.emit_void(spv::OpLoopMerge, {l.merge_label, l.continue_label, 0u});
        m.emit_void(spv::OpBranch, {body});
        cur_label = m.label(body);
        pending.clear();
        open_loops.push_back(&l);
        if (l.mask >= 0) {
            // A pixel in the body has EXEC at entry set: it took the exit test
            // with its mask set in the first iteration, and the mask started as EXEC.
            if (!exec0.constant) l.assumed.push_back(exec0.id);
        }
    }

    // A branch to the innermost loop's exit, after its instruction was lifted.
    void loop_exit(const Inst& in) {
        Loop& l = *open_loops.back();
        if (innermost_loop(in.offset) != &l) {
            reject("a branch leaves a loop that is not the innermost open one");
            return;
        }
        Id leave = 0;  // where the branch is taken; 0 with `always`
        bool always = false, per_pixel = false;
        if (in.op == 2) {
            always = true;
        } else if (in.op == 4 || in.op == 5) {
            const bool on_set = in.op == 5;  // s_cbranch_scc1 is taken where SCC is set
            const Val scc = known(scc_mask);
            if (!scc_valid) {
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) + " on an SCC an if left different on its arms");
                return;
            } else if (scc.constant) {
                if (scc.set != on_set) return;  // never taken
                always = true;
            } else if (scc_uniform) {
                leave = on_set ? scc.id : bnot(scc).id;
            } else if (in.offset == l.mask_exit) {
                per_pixel = true;
                if (!divergent_exit(l, scc)) return;
                leave = bnot(scc).id;
            } else {
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                       " on an SCC that is neither a scalar condition nor the divergent loop's exit test");
                return;
            }
        } else if (in.op == 6 || in.op == 7) {
            Val bit;
            std::uint32_t split = 0;
            if (!vcc_branch_bit(bit, split)) {
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) + " on a VCC that may differ between lanes");
                return;
            }
            bit = known(bit);
            const bool on_set = in.op == 7;  // s_cbranch_vccnz is taken where VCC is set
            if (bit.constant) {
                if (bit.set != on_set) return;
                always = true;
            } else {
                leave = on_set ? bit.id : bnot(bit).id;
            }
        } else {
            reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                   ": only s_branch, s_cbranch_scc0/scc1 and s_cbranch_vccz/vccnz are lifted as loop exits");
            return;
        }
        if (!l.exits.empty()) {
            reject("a second exit from the loop at " + hex_offset(l.header) + " (one is lifted)");
            return;
        }
        Loop::Exit x;
        x.reg = reg;
        x.lanes = lanes;
        x.scc = scc_mask;
        x.scc_uniform = scc_uniform;
        x.scc_valid = scc_valid;
        x.poisoned = poisoned;
        x.per_pixel = per_pixel;
        if (always) {
            // It ends the arm it is in: an if's then arm, or its else arm.
            const std::uint32_t next = in.offset + 4;
            const bool ends_arm = arms.size() > l.arms0 && (next == arms.back().c->else_jump || next == arms.back().c->join);
            if (!ends_arm) {
                reject("an unconditional exit from the loop at " + hex_offset(l.header) + " that does not end an if's arm");
                return;
            }
            x.from = cur_label;
            m.emit_void(spv::OpBranch, {l.merge_label});
            block_dead = true;
        } else {
            const Id brk = m.fresh(), cont = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {cont, 0u});
            m.emit_void(spv::OpBranchConditional, {leave, brk, cont});
            x.from = m.label(brk);
            m.emit_void(spv::OpBranch, {l.merge_label});
            cur_label = m.label(cont);
        }
        l.exits.push_back(std::move(x));
        if (per_pixel) {
            const Val mk = known(scc_mask);
            l.assumed.push_back(mk.id);
            if (const auto it = conjuncts.find(mk.id); it != conjuncts.end()) l.assumed.insert(l.assumed.end(), it->second.begin(), it->second.end());
            for (const Id a : l.assumed) assumed.insert(a);
        }
    }
    // The divergent exit test: at the loop's top level, on the mask's current
    // value, which lies within EXEC at the header.
    bool divergent_exit(const Loop& l, const Val& m_val) {
        if (arms.size() != l.arms0 || open_regions.size() != l.regions0) {
            reject("the divergent exit of the loop at " + hex_offset(l.header) + " is inside an if or a region of its body");
            return false;
        }
        const auto lm = lanes.find(l.mask);
        if (lm == lanes.end() || !same_lane(known(lm->second), m_val)) {
            reject("the divergent exit of the loop at " + hex_offset(l.header) + " does not test " + key_name(l.mask) + " as it is");
            return false;
        }
        if (!implies(m_val, l.unchanged.at(kKeyExec))) {
            reject("the mask the divergent exit of the loop at " + hex_offset(l.header) + " tests is not within EXEC at the header");
            return false;
        }
        return true;
    }

    // The back edge, after its s_branch was lifted.
    void loop_back_edge(Loop& l) {
        cur = l.latch;
        if (block_dead) {
            reject("the back edge of the loop at " + hex_offset(l.header) + " follows an unconditional exit");
            return;
        }
        if (arms.size() != l.arms0 || open_regions.size() != l.regions0) {
            reject("an if or a region is still open at the back edge of the loop at " + hex_offset(l.header));
            return;
        }
        if (l.mask >= 0 && (l.exits.empty() || !l.exits[0].per_pixel)) {
            reject("the divergent loop at " + hex_offset(l.header) + " was not left at its exit test");
            return;
        }
        std::vector<std::pair<Id, Id>> back;  // phi, value
        // What the header assumed of the helper lanes must hold for the value
        // that comes back; if not, a second attempt assumes less.
        const auto helper_lanes_kept = [&](int key, bool at_header, const Val& v) {
            if (!v.helper_inexact || at_header) return true;
            more_inexact.insert({l.header, key});
            reject(key_name(key) + " comes back to the header of the loop at " + hex_offset(l.header) +
                   " with helper lanes GCN may hold differently, which the header did not assume");
            return false;
        };
        for (const Loop::Phi& p : l.phis) {
            if (p.kind == Kind::Lane) {
                const auto it = lanes.find(p.key);
                if (it == lanes.end() || poisoned.count(p.key)) {
                    reject(key_name(p.key) + " leaves the header of the loop at " + hex_offset(l.header) + " as a lane mask and comes back as something else");
                    return;
                }
                const Val v = known(it->second);
                if (!helper_lanes_kept(p.key, p.helper_inexact, v)) return;
                back.push_back({p.id, v.constant ? m.const_bool(v.set) : v.id});
                continue;
            }
            if (poisoned.count(p.key) || lanes.count(p.key) || (p.key > 0 && p.key < 104 && lanes.count(p.key - 1))) {
                reject(key_name(p.key) + " leaves the header of the loop at " + hex_offset(l.header) + " as a word and comes back as a lane mask or unknown");
                return;
            }
            const Val v = value(p.key);
            if (p.uniform && !uniform(v)) {
                reject(key_name(p.key) + " comes back to the header of the loop at " + hex_offset(l.header) + " with a value lanes may not hold alike");
                return;
            }
            if (!helper_lanes_kept(p.key, p.helper_inexact, v)) return;
            back.push_back({p.id, p.kind == Kind::Float ? as_f(v) : as_u(v)});
        }
        // (In a divergent loop's body EXEC at entry is known set, so EXEC and
        // the mask must come back known set.)
        for (const auto& [k, entry] : l.unchanged) {
            const auto it = lanes.find(k);
            if (it == lanes.end() || !same_lane(known(it->second), known(entry))) {
                if (k != kKeyExec && k != l.mask && it != lanes.end()) more_lane_phis.insert({l.header, k});
                reject(key_name(k) + " comes back to the header of the loop at " + hex_offset(l.header) + " changed");
                return;
            }
            if (!helper_lanes_kept(k, entry.helper_inexact, it->second)) return;
        }
        m.emit_void(spv::OpBranch, {l.continue_label});
        m.label(l.continue_label);
        m.emit_void(spv::OpBranch, {l.header_label});
        for (const auto& [phi, v] : back) m.set_operand(phi, 2, v);
        block_dead = true;  // the merge block starts at the exit, next
    }

    // The exit's merge block, at the loop's exit.
    void close_loop() {
        Loop& l = *open_loops.back();
        open_loops.pop_back();
        cur = l.exit;
        for (const Id a : l.assumed) assumed.erase(a);
        if (l.exits.empty()) {
            reject("the loop at " + hex_offset(l.header) + " has no exit");
            return;
        }
        const Loop::Exit& x = l.exits[0];
        cur_label = m.label(l.merge_label);
        block_dead = false;
        reg = x.reg;
        lanes = x.lanes;
        scc_mask = x.scc;
        scc_uniform = x.scc_uniform;
        scc_valid = x.scc_valid;
        poisoned = x.poisoned;
        to_float = l.to_float0;
        to_word = l.to_word0;
        pending.clear();
        if (l.mask >= 0) {
            // GCN may run more iterations than this pixel: nothing scalar the
            // loop writes may be read after it before it is written again.
            std::uint32_t at = 0;
            int key = 0;
            bool unmodelled = false;
            if (read_before_written(index_of(l.exit), l.writes, true, at, key, unmodelled)) {
                cur = at;
                reject(unmodelled ? "cannot show what the divergent loop at " + hex_offset(l.header) + " writes is dead after it"
                                  : key_name(key) + " is written inside the divergent loop at " + hex_offset(l.header) +
                                        " and read after it (GCN may run more iterations than the pixel)");
                return;
            }
            ++divergent_loops;
        }
        ++loops_lifted;
        std::string kept;
        for (const auto& [k, v] : l.unchanged) kept += (kept.empty() ? "" : ", ") + key_name(k);
        res.proof.push_back(
            "loop " + hex_offset(l.header) + "-" + hex_offset(l.latch) + ": " +
            (l.mask >= 0 ? "divergent, the per-lane exit idiom on " + key_name(l.mask) + " at " + hex_offset(l.mask_exit) +
                               " (EXEC and the mask keep EXEC at entry at the header and come back set; what the loop writes is dead after "
                               "it; no implicit-LOD sample inside)"
                         : std::string("uniform, left on a bit every lane holds alike")) +
            "; " + std::to_string(l.phis.size()) + " phis (" + std::to_string(l.vgpr_phis) + " VGPRs, " + std::to_string(l.lane_phis) + " lane masks)" +
            (kept.empty() ? "" : "; back at the header unchanged: " + kept));
    }

    // ---- module ----------------------------------------------------------------------
    void emit() {
        m.capability(spv::CapShader);
        m.capability(spv::CapInt64);
        m.memory_model(0 /* Logical */, 1 /* GLSL450 */);
        t_void = m.type_void();
        t_bool = m.type_bool();
        t_u32 = m.type_int(32, false);
        t_i32 = m.type_int(32, true);
        t_u64 = m.type_int(64, false);
        t_f32 = m.type_float(32);
        t_v2f = m.type_vector(t_f32, 2);
        native_rtz = native_half_rtz() && program_allows_native_half_rtz(prog);
        if (native_rtz) {
            m.capability(spv::CapFloat16);
            m.capability(spv::CapDenormPreserve);
            m.capability(spv::CapRoundingModeRTZ);
            t_v2h = m.type_vector(m.type_float(16), 2);
        }
        t_v3f = m.type_vector(t_f32, 3);
        t_v4f = m.type_vector(t_f32, 4);
        t_v2i = m.type_vector(t_i32, 2);
        t_v3i = m.type_vector(t_i32, 3);
        t_v4u = m.type_vector(t_u32, 4);
        p_in_v4f = m.type_pointer(spv::ScInput, t_v4f);
        p_in_u32 = m.type_pointer(spv::ScInput, t_u32);
        p_in_v4u = m.type_pointer(spv::ScInput, t_v4u);
        p_in_bool = m.type_pointer(spv::ScInput, t_bool);
        p_out_v4f = m.type_pointer(spv::ScOutput, t_v4f);
        p_uni_u32 = m.type_pointer(spv::ScUniform, t_u32);

        // StageParams, as the translator declares it: { u64 l1_table; u64
        // reserved; uvec4 user[4]; uint cb_valid; uvec4 cb_bias_dw[4];
        // uvec4 cb_stride[4]; uvec4 cb_w3[4] }.
        const Id t_user_arr = m.type_array(t_v4u, cu(4));
        m.decorate(t_user_arr, spv::DecArrayStride, {16});
        const Id t_bias_arr = t_user_arr;  // the same uvec4[4] type (types are deduplicated): decorated once
        // With the reference bindless (TranslateOptions::bindless), the image
        // and sampler slots follow, at StageParams's own offsets.
        std::vector<Id> members = {t_u64, t_u64, t_user_arr, t_u32, t_bias_arr, t_bias_arr, t_bias_arr};
        if (ref.bindless) {
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
        if (ref.bindless) {
            m.member_decorate(t_ubo, 7, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, image_index))});
            m.member_decorate(t_ubo, 8, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, sampler_index))});
        }
        ubo_var = m.global_variable(m.type_pointer(spv::ScUniform, t_ubo), spv::ScUniform);
        m.decorate(ubo_var, spv::DecDescriptorSet, {opt.descriptor_set});
        m.decorate(ubo_var, spv::DecBinding, {kBindingParams});
        m.name(ubo_var, "params");
        interface.push_back(ubo_var);

        // The reference's bindings, same numbers and types - or, bindless, the
        // same global arrays it reads (one alias per image type).
        std::map<Id, Id> image_arrays;
        const auto global_array = [&](Id t_elem, std::uint32_t binding, const char* name) {
            const Id var = m.global_variable(m.type_pointer(spv::ScUniformConstant, m.type_runtime_array(t_elem)), spv::ScUniformConstant);
            m.decorate(var, spv::DecDescriptorSet, {opt.bindless_set});
            m.decorate(var, spv::DecBinding, {binding});
            m.name(var, name);
            interface.push_back(var);
            m.capability(spv::CapRuntimeDescriptorArray);
            m.capability(spv::CapSampledImageArrayDynamicIndexing);
            return var;
        };
        for (std::size_t k = 0; k < ref.images.size(); ++k) {
            const ImageBinding& b = ref.images[k];
            if (b.dim == spv::Dim1D) m.capability(b.storage ? spv::CapImage1D : spv::CapSampled1D);
            if (b.dim == spv::DimCube && b.arrayed) m.capability(spv::CapSampledCubeArray);
            const Id t_img = m.type_image(b.kind == 1 ? t_u32 : b.kind == 2 ? t_i32 : t_f32, static_cast<spv::Dim>(b.dim), b.depth,
                                          b.arrayed, false, b.storage ? 2 : 1);
            Id var;
            if (ref.bindless) {
                auto arr = image_arrays.find(t_img);
                if (arr == image_arrays.end()) {
                    arr = image_arrays.emplace(t_img, global_array(t_img, b.storage ? kBindlessStorageImages : kBindlessImages,
                                                                  b.storage ? "storage_images" : "images")).first;
                    if (b.storage) m.capability(spv::CapStorageImageArrayDynamicIndexing);
                }
                var = arr->second;
            } else {
                var = m.global_variable(m.type_pointer(spv::ScUniformConstant, t_img), spv::ScUniformConstant);
                m.decorate(var, spv::DecDescriptorSet, {opt.descriptor_set});
                m.decorate(var, spv::DecBinding, {b.binding});
                m.name(var, "img" + std::to_string(k));
                interface.push_back(var);
            }
            image_vars.push_back(var);
            image_types.push_back(t_img);
        }
        Id sampler_array = 0;
        for (std::size_t k = 0; k < ref.samplers.size(); ++k) {
            Id var;
            if (ref.bindless) {
                if (!sampler_array) sampler_array = global_array(m.type_sampler(), kBindlessSamplers, "samplers");
                var = sampler_array;
            } else {
                var = m.global_variable(m.type_pointer(spv::ScUniformConstant, m.type_sampler()), spv::ScUniformConstant);
                m.decorate(var, spv::DecDescriptorSet, {opt.descriptor_set});
                m.decorate(var, spv::DecBinding, {ref.samplers[k].binding});
                m.name(var, "smp" + std::to_string(k));
                interface.push_back(var);
            }
            sampler_vars.push_back(var);
        }
        if (!ref.buffers.empty()) {
            const Id t_words = m.type_runtime_array(t_u32);
            m.decorate(t_words, spv::DecArrayStride, {4});
            const Id t_block = m.type_struct({t_words});
            m.decorate(t_block, spv::DecBlock);
            m.member_decorate(t_block, 0, spv::DecOffset, {0});
            const Id p_block = m.type_pointer(spv::ScStorageBuffer, t_block);
            p_ssbo_u32 = m.type_pointer(spv::ScStorageBuffer, t_u32);
            for (std::size_t k = 0; k < ref.buffers.size(); ++k) {
                const Id var = m.global_variable(p_block, spv::ScStorageBuffer);
                m.decorate(var, spv::DecDescriptorSet,
                           {opt.cb_descriptor_set < 0 ? opt.descriptor_set : static_cast<std::uint32_t>(opt.cb_descriptor_set)});
                m.decorate(var, spv::DecBinding, {ref.buffers[k].binding});
                m.name(var, "cb" + std::to_string(k));
                interface.push_back(var);
                buffer_vars.push_back(var);
            }
        }

        fn_main = m.begin_function(t_void, m.type_function(t_void, {}));
        m.name(fn_main, "main");
        cur_label = m.label();
        lanes[kKeyExec] = lane_const(true);   // every running pixel starts with its bit set
        lanes[kKeyExec].helper_inexact = true;  // and GCN's helper lanes with theirs clear, until whole-quad mode
        lanes[kKeyVcc] = lane_const(false);   // VCC starts clear
        scc_mask = lane_const(false);         // and so does SCC

        std::size_t next_region = 0;
        std::uint64_t seq = 0;
        for (std::size_t i = 0; i < prog.insts.size(); ++i) {
            const Inst& in = prog.insts[i];
            cur = in.offset;
            // A loop's exit begins its merge block, before what encloses the loop closes.
            if (!open_loops.empty() && in.offset == open_loops.back()->exit) {
                close_loop();
                if (!res.rejections.empty()) return;
            }
            // Close the regions and ifs that end here, innermost first.
            for (;;) {
                const bool region_due = !open_regions.empty() && in.offset >= open_regions.back()->end;
                const bool arm_due = !arms.empty() && in.offset == arms.back().c->join && (arms.back().in_else || !arms.back().c->else_jump);
                if (!region_due && !arm_due) break;
                if (region_due && (!arm_due || open_regions.back()->seq > arms.back().seq)) {
                    Region* done = open_regions.back();
                    open_regions.pop_back();
                    close_region(*done, i);
                } else {
                    close_arm();
                }
                if (!res.rejections.empty()) return;
            }
            // After an unconditional loop exit only the s_branch closing that then arm may follow.
            if (block_dead && !(!arms.empty() && !arms.back().in_else && in.offset == arms.back().c->else_jump)) {
                reject("an instruction follows an unconditional loop exit");
                return;
            }
            if (block_starts.count(in.offset)) pending.clear();  // the translator's landing rule is per block
            Construct* const branch_if = construct_at(in.offset);
            const bool uniform_if = branch_if && (branch_if->op <= 5 ? scc_valid && scc_uniform : vcc_branch_bit(vcc_bit, vcc_split));
            if (next_region < regions.size() && in.offset == regions[next_region].branch) {
                Region& r = regions[next_region];
                if (uniform_if) {
                    r.disabled = true;  // an if, not a kill region
                } else if (r.kill && !scc_valid) {
                    reject("s_cbranch_scc0 tests an SCC an earlier if left different on its arms");
                    return;
                } else if (r.kill && !open_loops.empty()) {
                    reject("s_cbranch_scc0 kill region inside a loop");
                    return;
                } else {
                    // A kill region inside another is the same idiom one level
                    // down: GCN evaluates its branch only where the outer one
                    // runs, the lift always, and a pixel whose guard bit is
                    // clear writes nothing either way.
                    r.guard = scc_mask;
                    r.exec_before = exec_lane();
                    r.lanes_at_branch = lanes;
                }
            } else if (branch_if && !uniform_if) {
                reject(std::string(mnemonic(in)) +
                       (branch_if->op <= 5 ? " tests an SCC that is not from a scalar compare (only an s_cbranch_scc0 without an else arm can be a kill region)"
                                           : " tests a VCC that may differ between lanes"));
                return;
            }
            if (next_region < regions.size() && in.offset >= regions[next_region].start) {
                Region& r = regions[next_region++];
                if (!r.disabled) {
                    r.exec_in = r.exec_before;  // a kill region narrows it with its first instructions
                    r.seq = ++seq;
                    open_regions.push_back(&r);
                }
            }
            // A loop begins inside what is open here.
            for (Loop& l : loops) {
                if (l.header != in.offset) continue;
                open_loop(l);
                if (!res.rejections.empty()) return;
            }
            note.clear();
            inst_uniform = true;
            inst_helper_inexact = false;
            reads = {};
            Region* const starting = !open_regions.empty() && open_regions.back()->kill ? open_regions.back() : nullptr;
            exec_write_ok = starting && kill_exec_write(*starting, in);
            lift_inst(in);
            if (exec_write_ok) starting->exec_in = exec_lane();
            exec_write_ok = widen_ok = false;
            char line[160];
            std::snprintf(line, sizeof(line), "%06x  %-56s", in.offset, format(in).c_str());
            res.listing += line;
            res.listing += (open_regions.empty() ? "" : " [region]") + std::string(arms.empty() ? "" : arms.back().in_else ? " [else]" : " [then]") + note + "\n";
            if (is_branch(in)) pending.clear();
            if (!res.rejections.empty()) return;
            if (!open_loops.empty() && in.offset == open_loops.back()->latch) {
                loop_back_edge(*open_loops.back());
            } else if (loop_exits.count(in.offset)) {
                loop_exit(in);
            }
            if (!res.rejections.empty()) return;
            if (uniform_if) open_arm(*branch_if, ++seq);
            if (!arms.empty() && !arms.back().in_else && in.offset == arms.back().c->else_jump) then_to_else(arms.back());
        }
        for (;;) {  // what the end of the program closes
            if (!open_regions.empty() && (arms.empty() || open_regions.back()->seq > arms.back().seq)) {
                Region* done = open_regions.back();
                open_regions.pop_back();
                close_region(*done, prog.insts.size());
            } else if (!arms.empty()) {
                reject("an if is still open at the end of the program");
            } else if (!open_loops.empty()) {
                reject("a loop is still open at the end of the program");
            } else {
                break;
            }
            if (!res.rejections.empty()) return;
        }
        if (needs_kill) {  // translate.cpp's end of main
            const Id l_kill = m.fresh(), l_ret = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {l_ret, 0u});
            m.emit_void(spv::OpBranchConditional, {kill_exec.constant ? m.const_bool(kill_exec.set) : kill_exec.id, l_ret, l_kill});
            m.label(l_kill);
            m.emit_void(spv::OpKill, {});
            m.label(l_ret);
        }
        m.emit_void(spv::OpReturn, {});
        m.end_function();
        if (opt.stage == Stage::Vertex) {
            // The pipeline, the pixel shader's inputs and a rect list's geometry
            // shader are laid out from the reference's outputs: the lift must
            // declare exactly those.
            std::vector<std::uint32_t> params;
            for (const auto& kv : out_params) params.push_back(static_cast<std::uint32_t>(kv.first));
            if (params != ref.vs_params || ref.vs_clip_count || ref.vs_point_size) {
                cur = prog.insts.back().offset;
                reject("the lifted outputs differ from the reference's (params, clip distances or point size)");
                return;
            }
            m.entry_point(spv::EmVertex, fn_main, "main", interface);
            if (native_rtz) {
                m.execution_mode(fn_main, spv::ExDenormPreserve, {16});
                m.execution_mode(fn_main, spv::ExRoundingModeRTZ, {16});
            }
            return;
        }
        m.entry_point(spv::EmFragment, fn_main, "main", interface);
        m.execution_mode(fn_main, spv::ExOriginUpperLeft);
        if (native_rtz) {
            m.execution_mode(fn_main, spv::ExDenormPreserve, {16});
            m.execution_mode(fn_main, spv::ExRoundingModeRTZ, {16});
        }
        if (opt.early_fragment_tests) m.execution_mode(fn_main, spv::ExEarlyFragmentTests);
    }

    void finish_proof() {
        std::vector<std::string> head;
        const std::size_t execz_regions = static_cast<std::size_t>(std::count_if(regions.begin(), regions.end(), [](const Region& r) { return !r.kill; }));
        head.push_back(std::to_string(prog.insts.size()) + " instructions, " +
                       (loops_lifted ? std::to_string(loops_lifted) + " loops (" + std::to_string(divergent_loops) + " with per-lane exits) and otherwise forward: "
                                     : std::string("forward only: ")) +
                       std::to_string(execz_regions) + " s_cbranch_execz regions, " +
                       std::to_string(kill_regions) + " s_cbranch_scc0 kill regions and " + std::to_string(if_constructs) +
                       " ifs on a bit every lane holds alike; no other branch or program-counter transfer");
        head.push_back(std::string("EXEC per pixel or vertex: set at entry; whole-quad mode only of a constant mask, in a kill region or in a "
                                   "block that restores the mask it widened; comparison masks are the pixel's own bit; ") +
                       (varying_samples.empty() ? "set again before every sample; " : "") +
                       (needs_kill ? "a pixel whose bit is clear at s_endpgm is discarded, as by the translator, and every pixel exported with "
                                     "its bit clear is among them"
                                   : "set at every export and at s_endpgm (no discard)"));
        std::string at;
        for (std::uint32_t s : samples) at += " " + hex_offset(s);
        std::string divergent_at;
        for (std::uint32_t s : divergent_samples) divergent_at += " " + hex_offset(s);
        head.push_back(std::to_string(samples.size()) + " image samples outside regions with EXEC set, in uniform control flow (helper "
                       "invocations stand in for whole-quad lanes):" + at +
                       (divergent_samples.empty() ? ""
                                                  : "; except, with explicit LOD or gradients, inside a loop pixels leave at different "
                                                    "iterations, where EXEC is set in its body:" + divergent_at));
        if (!varying_samples.empty() || !lod_queries.empty()) {
            std::string varying, queries;
            for (std::uint32_t s : varying_samples) varying += " " + hex_offset(s);
            for (std::uint32_t s : lod_queries) queries += " " + hex_offset(s);
            head.push_back(std::to_string(varying_samples.size()) + " image samples under a varying EXEC" + (varying.empty() ? "" : ":" + varying) +
                           "; " + std::to_string(lod_queries.size()) + " image_get_lod" + (queries.empty() ? "" : ":" + queries) +
                           "; in uniform control flow, results written under EXEC, operands GCN's in every pixel of EXEC and the "
                           "coordinates of implicit derivatives in every pixel of the quads of EXEC");
        }
        head.push_back(std::to_string(exports.size()) + " colour exports with EXEC set");
        head.push_back(std::to_string(buffer_loads) + " scalar loads, every one from a storage buffer the reference binds; " +
                       std::to_string(masked_writes) + " VGPR writes under a varying EXEC become selects");
        if (swizzles) {
            head.push_back(std::to_string(swizzles) + " ds_swizzle_b32 within a quad as quad operations in uniform control flow: each under EXEC "
                           "known set in whole-quad mode, of a value the quad's helper lanes computed as on GCN, and no kill region where "
                           "pixels may differ before it");
        }
        if (descriptor_loads) {
            head.push_back(std::to_string(descriptor_loads) + " scalar loads the reference walks the page table for hold descriptors only: "
                           "no word of them is read as data, and the reference binds by resource path, not by the words");
        }
        if (cell_reads) {
            head.push_back(std::to_string(cell_reads) + " v_readlane_b32 of spilled scalars, each from the spill cell the reference reads "
                           "(only its v_writelane_b32s reach it, on every path); a VGPR a scalar is parked in is not read as this "
                           "pixel's word until written whole");
        }
        res.proof.insert(res.proof.begin(), head.begin(), head.end());
    }
};

}  // namespace

namespace {
LiftResult lift_stage(Stage stage, const Program& program, const TranslateOptions& options, const TranslateResult& reference) {
    if (options.stage != stage) {
        LiftResult wrong;
        wrong.rejections.push_back(stage == Stage::Pixel ? "the options are not a pixel shader's" : "the options are not a vertex shader's");
        return wrong;
    }
    // A lane mask a loop changes before it is read again gets a phi on a second
    // attempt: the first assumes every one comes back unchanged, which the
    // usual loop-exit idioms need to fold their tests. Likewise a register
    // whose helper lanes come back around a loop less exact than they left
    // is inexact at the header on the next attempt.
    Lifter::LoopPhis phis, inexact;
    for (int attempt = 0;; ++attempt) {
        Lifter l(program, options, reference, phis, inexact);
        LiftResult r = l.run();
        const std::size_t before = phis.size() + inexact.size();
        phis.insert(l.more_lane_phis.begin(), l.more_lane_phis.end());
        inexact.insert(l.more_inexact.begin(), l.more_inexact.end());
        if (phis.size() + inexact.size() == before || attempt == 8) return r;
    }
}
}  // namespace

LiftResult lift_pixel_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference) {
    if (options.ps_clip_discard) {  // TranslateOptions::kVsOutCntlClipVarying has no lifted counterpart
        LiftResult r;
        r.rejections.push_back("clip distances discarded per pixel");
        return r;
    }
    return lift_stage(Stage::Pixel, program, options, reference);
}

LiftResult lift_vertex_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference) {
    if (options.vertex_formats_from_params) {  // translate.cpp load_vertex_element_from_params has no lifted counterpart yet
        LiftResult r;
        r.rejections.push_back("vertex formats from the params block");
        return r;
    }
    if (options.vs_out_cntl & TranslateOptions::kVsOutCntlClipVarying) {
        LiftResult r;
        r.rejections.push_back("clip distances as a varying");
        return r;
    }
    return lift_stage(Stage::Vertex, program, options, reference);
}

}  // namespace gcn
