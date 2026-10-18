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
//     clear writes nothing whether or not the region runs. If EXEC after it
//     differs from EXEC at the branch, or the region widens EXEC to whole
//     quads (s_wqm_b64; the lifted pixel keeps its own bit), the lift can
//     differ from GCN only in pixels whose M bit is clear: EXEC at s_endpgm
//     must then lie within M, and no implicit-LOD sample may follow. GCN skips
//     the region in a wave where M is clear in every lane and keeps EXEC as it
//     was at the branch, so EXEC must also be set again, from what does not
//     read it, before it decides anything (an export, a mask made of it,
//     s_endpgm). Widening from a varying EXEC needs M within it. Kill regions
//     nest, the inner one's mask within the outer one's.
//   * An s_cbranch_scc0/scc1 or s_cbranch_vccz/vccnz whose bit every lane
//     holds alike becomes structured control flow: SCC from a scalar compare
//     (scalar operands never read a VGPR), or VCC computed only from
//     constants, user data and loads - or VCC a comparison of values every
//     lane holds alike where EXEC is set, ANDed with an EXEC GCN cannot reach
//     the branch with empty (EXEC where the innermost region began). It has a
//     then arm, an else arm where an s_branch closes the then arm, and a
//     merge whose phis carry the values the arms left different. In an
//     else-if chain whose inner arms jump straight to the outer join, the
//     inner if joins at the s_branch closing the outer then arm instead. GCN
//     tests all 64 bits, so the helper lanes must hold the lift's bits or be
//     clear (Val::helpers_clear), and a mask clear for the lift's pixels
//     decides only when it is clear in every lane of the wave (Val::wave_exact).
//   * A backward s_branch closes a loop, left only for the instruction after
//     it: an OpLoopMerge loop per pixel with phis for what the body writes.
//     A uniform loop exits on a bit every lane holds alike; a divergent one is
//     the per-lane exit idiom on a loop mask (see "loops" below). No exit sits
//     inside a region of the body, which GCN skips in some iterations.
//   * Samples run for every pixel in uniform control flow and their texels are
//     written under EXEC. Where EXEC is known set, Vulkan helper invocations
//     compute the samples for a quad's uncovered pixels as GCN's whole-quad
//     lanes do, and helper outputs are discarded. Under a varying EXEC an
//     explicit-LOD sample reads only the pixel's own operands; an
//     implicit-LOD sample (and image_get_lod, OpImageQueryLod) needs its
//     coordinates to hold GCN's values in every pixel of each quad holding an
//     EXEC pixel, since GCN's texture unit takes the derivatives from the
//     quad's four lanes, EXEC set or not, as Vulkan takes them from the quad's
//     four invocations; the quad's helper lanes compute them on GCN only in
//     whole-quad mode (Val::helper_inexact). In a loop pixels leave at
//     different iterations, where control flow is not uniform, only
//     explicit-LOD samples are lifted. Exports are unconditional, as in the
//     translator; a pixel whose EXEC bit is clear at s_endpgm is discarded,
//     and every pixel exported with its bit clear must be one of them.
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
//     only of a value the helper lanes computed as GCN's did
//     (Val::helper_inexact, carried around loops), never inside a loop pixels
//     leave at different iterations.
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
// NMin/NMax, the legacy multiply's zero rule, PackHalf2x16 exports and their
// rounding toward zero, the cube helpers, v_cndmask_b32's source modifiers on
// the sign bit, and sample operand order), so a lifted shader can be compared
// with its reference texel for texel.
#include "gcn/lift.h"

#include "gcn/half.h"
#include "gcn/spirv.h"
#include "gcn/wave.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
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
    // A Lane constant that holds in every lane of GCN's wave, helper lanes and
    // lanes no pixel occupies included: an inline 0 or -1, 0 & x, or x & ~x
    // where both are one mask (Val::content). Other constants say what the
    // lift's pixels hold (EXEC set "for every running pixel", a mask a loop's
    // body knows for its pixel). A clear mask decides a wave-wide test (SCC of
    // a mask operation, VCCZ) only when it is clear in every lane.
    bool wave_exact = false;
    // A lane mask's 64 bits on GCN, as a tag (0: unknown): copies keep it, a
    // new mask gets its own. A mask that takes another's id by meeting a
    // constant that holds only for the lift's pixels (x & EXEC, where EXEC is
    // "set" for the running pixels only) gets its own tag too: in the other
    // lanes it need not be x.
    std::uint32_t content = 0;
    // A lane mask whose bits GCN's helper lanes hold clear: the coverage EXEC a
    // pixel shader starts with, an AND with one, a comparison or carry made
    // under one (inactive lanes write 0). A wave-wide test of it (VCCZ) then
    // sees only the covered pixels, whatever the helper lanes computed.
    bool helpers_clear = false;
    // A word the lift knows as sym_base + sym_coef * the invocation's index -
    // a constant (coef 0), or in a hull shader an address of the invocation's
    // own control point - beside its SPIR-V value, so an LDS address can name
    // its slot (TessLift). Kept through the integer operations address
    // arithmetic uses; anything else loses it.
    bool sym = false;
    std::uint32_t sym_base = 0, sym_coef = 0;
    // Where the form holds: everywhere (0), or where the lane mask
    // Lifter::sym_masks[sym_where - 1] is set - a VGPR written under a varying
    // EXEC is a select, and holds the written word only there.
    std::uint32_t sym_where = 0;
    // compute: a part of GCN's append idiom (Lifter::gds_counter, mbcnt),
    // which the lift keeps apart until the two parts are added together - 1
    // the base a ds_append step gives the wave, 2 a ds_consume's, 3 a lane's
    // rank among a mask's lanes (v_mbcnt over the mask), 4 a rank's high half
    // (v_mbcnt_hi alone). Its word is the invocation's own slot (a base) or 0
    // (a rank), so base + rank is the slot; any other use of a part rejects.
    // `part_mask`: the lane mask the step or rank was taken over (constant:
    // part_mask_set), which the parts must share.
    std::uint8_t wave_part = 0;
    Id part_mask = 0;
    bool part_mask_constant = false, part_mask_set = false;
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
    std::set<int> in_mask0;  // Lifter::in_mask where the region begins
    // Run as a branch on the pixel's EXEC bit (Lifter::region_branch_open):
    // its merge and skip blocks, and the lift's state at the branch.
    bool branched = false;
    spv::Id rb_merge = 0, rb_skip = 0;
    std::map<int, Val> rb_reg0, rb_lanes0;
    std::map<spv::Id, spv::Id> rb_to_float0, rb_to_word0;
    Val rb_scc0;
    bool rb_scc_uniform0 = false, rb_scc_valid0 = true;
    std::set<int> rb_poisoned0, rb_in_mask0;
    // A compute lift's region as a selection on the invocation's EXEC bit
    // (Lifter::branch_region): an invocation whose bit is clear skips it, as
    // GCN's wave skips it where no lane's is set. (The pixel lift's own branch
    // on the pixel's bit is `branched` above, Lifter::region_branch_open.)
    bool selected = false;
};

bool is_branch(const Inst& in) { return in.enc == Enc::SOPP && (in.op == 2 || (in.op >= 4 && in.op <= 9)); }

// A module's outputs as TranslateResult reports a vertex stage's: the
// locations of its Output variables, its ClipDistance array's length (0:
// none) and whether it writes PointSize. For an evaluation stage, whose
// translation does not report them.
void module_outputs(const std::vector<std::uint32_t>& w, std::vector<std::uint32_t>& locations, int& clip, bool& point_size) {
    locations.clear();
    clip = 0;
    point_size = false;
    std::map<std::uint32_t, std::uint32_t> constants, array_length, pointee, location, builtin;
    std::set<std::uint32_t> outputs;
    std::map<std::uint32_t, std::uint32_t> var_type;
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (n == 0 || i + n > w.size()) break;
        const std::uint32_t* a = &w[i + 1];
        if (op == 43 && n >= 4) constants[a[1]] = a[2];                      // OpConstant
        if (op == 28 && n >= 4) array_length[a[0]] = a[2];                   // OpTypeArray (length id)
        if (op == 32 && n >= 4) pointee[a[0]] = a[2];                        // OpTypePointer
        if (op == 59 && n >= 4 && a[2] == 3) outputs.insert(a[1]), var_type[a[1]] = a[0];  // OpVariable Output
        if (op == 71 && n >= 4 && a[1] == 30) location[a[0]] = a[2];        // OpDecorate Location
        if (op == 71 && n >= 4 && a[1] == 11) builtin[a[0]] = a[2];         // OpDecorate BuiltIn
        i += n;
    }
    for (const std::uint32_t v : outputs) {
        if (const auto l = location.find(v); l != location.end()) locations.push_back(l->second);
        if (const auto b = builtin.find(v); b != builtin.end()) {
            if (b->second == 1) point_size = true;
            if (b->second == 3) {
                const auto t = pointee.find(var_type[v]);
                const auto len = t == pointee.end() ? array_length.end() : array_length.find(t->second);
                const auto c = len == array_length.end() ? constants.end() : constants.find(len->second);
                clip = c == constants.end() ? 1 : static_cast<int>(c->second);
            }
        }
    }
    std::sort(locations.begin(), locations.end());
}

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
    // `loop_nonuniform` (compute): VGPRs (loop header, key) an earlier attempt
    // found coming back to the header with a value lanes may hold differently;
    // the others that enter a loop alike are taken to stay alike, as scalars are.
    Lifter(const Program& p, const TranslateOptions& o, const TranslateResult& r, const LoopPhis& loop_phis = {}, const LoopPhis& loop_inexact = {},
           const TessLift* tess_lift = nullptr, const LoopPhis& loop_nonuniform = {})
        : prog(p), opt(o), ref(r), tess(tess_lift), lane_phi_hints(loop_phis), inexact_hints(loop_inexact), nonuniform_hints(loop_nonuniform) {}
    LoopPhis more_lane_phis;  // set when a lift failed only because a lane mask needs a phi
    LoopPhis more_inexact;    // ... or because helper lanes came back around a loop less exact than they left
    LoopPhis more_nonuniform;  // ... or because a VGPR came back to a loop header holding values lanes may not hold alike

    LiftResult run() {
        check_program();
        if (res.rejections.empty()) emit();
        if (res.rejections.empty() && !more_nonuniform.empty()) {
            cur = prog.insts.empty() ? 0 : prog.insts.back().offset;
            reject("a VGPR comes back to a loop header holding values lanes may not hold alike (assumed alike there: another attempt)");
        }
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
    // A tessellation stage on stage I/O (lift.h TessLift), or null.
    const TessLift* tess = nullptr;
    bool tess_ls() const { return tess && opt.tess_role == TranslateOptions::TessRole::LsVertex; }
    bool tess_ds() const { return tess && opt.tess_role == TranslateOptions::TessRole::DomainTes; }
    bool tess_hs() const { return tess && opt.tess_role == TranslateOptions::TessRole::HullTcs; }
    // TessLift::window: the game's own hull's patch window as stage I/O.
    bool window() const { return tess && tess->window; }
    // A stage that exports a position and params: a vertex shader, or the
    // domain shader as the evaluation stage.
    bool vertex_like() const { return opt.stage == Stage::Vertex || opt.stage == Stage::TessEval; }
    // A compute shader of a dispatch (no tessellation role): one invocation a
    // lane of GCN's wave, every buffer reached through the page table as the
    // reference reaches it (lift_compute_shader).
    bool compute() const { return opt.stage == Stage::Compute && opt.tess_role == TranslateOptions::TessRole::None; }
    // Threads in the dispatch's workgroup, and whether GCN's waves of it are
    // all whole (every lane of each 64 runs an invocation).
    std::uint32_t workgroup_threads() const {
        return std::max<std::uint32_t>(1, opt.cs_threads[0]) * std::max<std::uint32_t>(1, opt.cs_threads[1]) *
               std::max<std::uint32_t>(1, opt.cs_threads[2]);
    }
    LiftResult res;
    spv::Module m;
    std::uint32_t cur = 0;
    std::string note;  // this instruction's writes, for the listing

    // Types and interface
    Id t_void = 0, t_bool = 0, t_u32 = 0, t_i32 = 0, t_u64 = 0, t_f32 = 0, t_v2f = 0, t_v3f = 0, t_v4f = 0, t_v2i = 0, t_v3i = 0;
    Id p_uni_u32 = 0, p_in_v4f = 0, p_in_bool = 0, p_out_v4f = 0, p_ssbo_u32 = 0, p_in_u32 = 0, p_in_v4u = 0, t_v4u = 0;
    Id ubo_var = 0, fn_main = 0;
    // compute(): guest memory through the page table at StageParams::l1_table,
    // as the reference's xlate() walks it - physical storage buffer pointers to
    // a dword, and the helper function.
    Id p_psb_u32 = 0, p_psb_block = 0, p_psb_block64 = 0, p_psb_u64 = 0, p_uni_u64 = 0, fn_xlate = 0;
    Id in_local_id = 0, in_workgroup_id = 0, in_subgroup_lane = 0;
    // StageParams::gds_counter's member (0: the program counts nothing in GDS).
    std::uint32_t gds_member = 0;
    std::vector<Id> interface;
    std::vector<Id> image_vars, image_types, sampler_vars, buffer_vars;
    std::map<int, Id> in_params, out_params, attr_loads;
    std::map<std::uint32_t, Id> in_location_vars;  // PS: location -> the Input vec4 there (attr)
    Id in_frag_coord = 0, in_front_facing = 0, in_vertex_index = 0, in_instance_index = 0, out_pos = 0, in_helper = 0;
    Id in_tess_coord = 0, in_invocation_id = 0;
    // TessLift::window: the hull's outputs and the domain shader's inputs -
    // the output control points (uvec4[control points][cp_stride / 16]) and
    // the patch constants (uvec4[patch_constants], Patch) - and the
    // tessellation levels.
    Id cp_out = 0, patch_io = 0, out_tess_outer = 0, out_tess_inner = 0;
    std::uint32_t tf_stores = 0;  // factor stores seen (translate.cpp tf_stores_seen)
    std::size_t lds_dynamic = 0, barriers = 0;
    // TessLift: the control point the LS writes (uvec4[attr_vec4s]) and the
    // domain shader reads (uvec4[control points][attr_vec4s]), at location 0.
    Id control_point = 0;
    std::size_t lds_reads = 0, lds_writes = 0, lds_static = 0;
    std::map<std::uint32_t, Id> vertex_in_vars;  // TranslateOptions::vertex_input locations
    std::map<std::uint32_t, Id> bias_loads, stride_loads;  // StageParams cb_bias_dw / cb_stride per buffer
    struct PackedPair {
        Id lo, hi;
        std::uint32_t block;  // Module::blocks() when packed
    };
    std::map<Id, PackedPair> packed_pairs;  // v_cvt_pkrtz_f16_f32's result -> its two floats
    // ... and that result written under a varying EXEC: the select -> the pair
    // and the EXEC bit it selected on (translate.cpp PackedPair::exec).
    struct MaskedPack {
        PackedPair pair;
        Id exec;
    };
    std::map<Id, MaskedPack> masked_packs;
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
    std::deque<Construct> region_ifs;  // the branched regions' selections (open_region_arm), stable for Arm::c
    std::vector<std::uint32_t> threaded_joins;  // ifs of an else-if chain joined at the enclosing then arm's s_branch
    struct Arm {  // an open if
        Construct* c = nullptr;
        std::uint64_t seq = 0;
        bool region = false;  // a branched region's selection (open_region_arm): its condition differs between lanes
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
        std::set<int> in_mask0, in_mask1;  // Lifter::in_mask at the branch; at the then arm's end
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
    std::size_t spill_reuses = 0;  // write_v under a varying EXEC over a VGPR whose lanes held spilled words

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
    // compute(): values computed from the workgroup's id - the same in every
    // lane of a wave, but not in every wave of the dispatch, so no loop GCN
    // may skip with a region takes its trip count from one (draw_wide_exit).
    std::set<Id> wg_ids;
    bool inst_wg = false;
    Val note_read(const Val& v) {
        if (!uniform(v)) inst_uniform = false;
        if (v.helper_inexact) inst_helper_inexact = true;
        if (!v.constant && wg_ids.count(v.id)) inst_wg = true;
        return v;
    }
    void note_result(const Val& v) {
        if (inst_uniform && !v.constant) uniform_ids.insert(v.id);
        if (inst_wg && !v.constant) wg_ids.insert(v.id);
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
        if (opt.stage != Stage::Pixel && opt.stage != Stage::Vertex && !tess_ds() && !tess_hs() && !compute()) {
            reject(opt.stage == Stage::Compute ? "a compute stage with a tessellation role (the LS pass) is not lifted"
                                               : "only pixel and vertex shaders are lifted");
        }
        if (compute()) {
            // The lift reads memory as its reference does, so it binds what the
            // reference binds: a dispatch's translation walks the page table for
            // every buffer; its lean one (ComputeLean: cb_ssbo, cb_no_fallback)
            // reads the accesses it bound (TranslateResult::buffer_at) from
            // their storage buffers and walks the rest, and so does the lift.
            if (opt.cb_ssbo && !opt.cb_no_fallback) {
                reject("a compute reference with constant buffers bound and a page-table fallback (cb_ssbo without cb_no_fallback) is not lifted");
            }
            if (opt.fetch || !opt.vertex_input.empty()) reject("a compute stage with vertex inputs");
            if (!ref.ok()) reject("the reference translation failed");
            if (!prog.errors.empty()) reject("the program has decode errors");
            if (prog.insts.empty()) reject("empty program");
            if ((opt.rsrc2 >> 10) & 1) reject("TG_SIZE in an SGPR: the reference gives 0 for it, GCN the workgroup's wave count");
            if (((opt.rsrc2 >> 15) & 0x1ff) || ref.lds_bytes) {
                for (const Inst& in : prog.insts) {
                    if (in.enc == Enc::DS && !(in.op == 53) && !((in.op == 61 || in.op == 62) && in.gds)) {
                        cur = in.offset;
                        reject(std::string(mnemonic(in) ? mnemonic(in) : "an LDS access") + ": LDS (groupshared memory) is not lifted");
                        break;
                    }
                }
            }
            if (!res.rejections.empty()) return;
        }
        if (opt.tess_role != TranslateOptions::TessRole::None && !tess_ls() && !tess_ds() && !tess_hs()) {
            reject("a tessellation stage is lifted only as the LS on vertex input, the hull as the control stage or the domain shader as the evaluation stage");
        }
        if (tess && !window() && !opt.tess_lds_attributes) reject("a tessellation stage is lifted with its control point as stage attributes or in a hull's window");
        if (tess_hs() && !window()) reject("only the game's own hull (a window a patch) is lifted as a control stage");
        if (window() && (tess_hs() || tess_ds()) && !opt.tess_window) reject("the reference does not lay the patch out in a window");
        if (window() && (!tess->ls_stride || tess->ls_stride % 16 || !tess->cp_stride || tess->cp_stride % 16 || !tess->control_points ||
                         tess->control_points > 32 || tess->patch_constants > 32)) {
            reject("a window layout with strides that are not whole vec4s, or no control points");
        }
        if (tess_ls() && opt.vertex_formats_from_params) reject("the LS takes its user data where vertex formats from the params would be");
        if (!opt.cb_no_fallback && !compute()) reject("the reference must be the no-fallback translation (every storage buffer bound)");
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
    Val word_const(std::uint32_t b) {
        Val v{Kind::Word, cu(b), true, b, false};
        v.sym = true;
        v.sym_base = b;
        return v;
    }
    Val float_const(float f) { return {Kind::Float, cf(f), true, std::bit_cast<std::uint32_t>(f), false}; }
    Val lane_const(bool b) { return {Kind::Lane, m.const_bool(b), true, 0, b}; }
    static Val word(Id id) { return {Kind::Word, id, false, 0, false}; }
    static Val flt(Id id) { return {Kind::Float, id, false, 0, false}; }
    static Val lane(Id id) { return {Kind::Lane, id, false, 0, false}; }

    // A part of the append idiom read as a number other than by adding the
    // two parts (Val::wave_part).
    void reject_wave_part(const Val& v) {
        static const char* const kPart[5] = {"", "a ds_append step's wave base", "a ds_consume step's wave base", "a lane's rank (v_mbcnt)",
                                             "a rank's high half (v_mbcnt_hi)"};
        reject(std::string(kPart[v.wave_part < 5 ? v.wave_part : 0]) +
               " is used other than added to its lanes' ranks: GCN's base and rank are the wave's, the lift's the invocation's own slot");
    }
    Val part_mask_val(const Val& p) { return p.part_mask_constant ? lane_const(p.part_mask_set) : lane(p.part_mask); }
    Id as_f(const Val& v) {
        if (v.wave_part) reject_wave_part(v);
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
        if (v.wave_part) reject_wave_part(v);
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
    Val band(const Val& a, const Val& b) {
        Val r = helper_bits(band_value(a, b), a, b, false);
        r.helpers_clear = a.helpers_clear || b.helpers_clear;
        return r;
    }
    Val bor(const Val& a, const Val& b) {
        Val r = helper_bits(bor_value(a, b), a, b, true);
        r.helpers_clear = a.helpers_clear && b.helpers_clear;
        return r;
    }
    Val bxor(const Val& a, const Val& b) {
        Val r = bxor_value(a, b);
        r.helper_inexact = a.helper_inexact || b.helper_inexact;
        r.helpers_clear = a.helpers_clear && b.helpers_clear;
        return r;
    }
    Val bnot(const Val& a) {
        Val r = bnot_value(a);
        r.helper_inexact = a.helper_inexact;
        r.helpers_clear = false;
        return r;
    }
    // Constants in every lane of the wave (Val::wave_exact): 0 & x, 1 | x and a
    // mask ANDed with its own complement are; a constant met with one that is
    // not is only as exact as both.
    static Val wave_exact_as(Val r, bool exact) {
        r.wave_exact = exact;
        return r;
    }
    std::uint32_t contents = 0;                          // Val::content tags given out
    std::map<std::uint32_t, std::uint32_t> complement;  // content -> its complement's, both ways
    Val new_content(Val v) {
        v.content = ++contents;
        return v;
    }
    // Two masks whose 64 bits are each other's complement.
    bool complements(const Val& a, const Val& b) const {
        if (!a.content || !b.content) return false;
        const auto ia = complement.find(a.content), ib = complement.find(b.content);
        return (ia != complement.end() && ia->second == b.content) || (ib != complement.end() && ib->second == a.content);
    }
    // x & c, x | c, x ^ c for a constant c: x itself where c is that in every
    // lane of the wave, otherwise a mask of its own there.
    Val folded(const Val& x, const Val& c) { return c.wave_exact ? x : new_content(x); }
    Val band_value(const Val& a, const Val& b) {
        if (a.constant && b.constant) return wave_exact_as(lane_const(a.set && b.set), a.set && b.set ? a.wave_exact && b.wave_exact
                                                                                                    : (!a.set && a.wave_exact) || (!b.set && b.wave_exact));
        if (a.constant) return a.set ? folded(b, a) : wave_exact_as(lane_const(false), a.wave_exact);
        if (b.constant) return b.set ? folded(a, b) : wave_exact_as(lane_const(false), b.wave_exact);
        if (a.id == b.id) return a.content && a.content == b.content ? a : new_content(a);
        if (negated(a, b)) return wave_exact_as(lane_const(false), complements(a, b));  // a & !a: a mask with the active lanes removed from itself
        const Id r = m.emit(spv::OpLogicalAnd, t_bool, {a.id, b.id});
        std::set<Id>& c = conjuncts[r];
        for (const Val* x : {&a, &b}) {
            c.insert(x->id);
            if (const auto it = conjuncts.find(x->id); it != conjuncts.end()) c.insert(it->second.begin(), it->second.end());
        }
        return new_content(lane(r));
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
        // A mask the lift knows set where it runs (a divergent loop's mask in
        // its body) that lost the lanes of `skip` (the exit idiom in a region
        // on the lanes leaving): set, as it was, wherever `skip` is clear.
        if (!now.constant && !skip.constant && ((before.constant && before.set) || (!before.constant && assumed.count(before.id)))) {
            const auto n = negation_of.find(now.id);
            if (n != negation_of.end() && n->second == skip.id) return true;
        }
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
        if (a.constant && b.constant) return wave_exact_as(lane_const(a.set || b.set), a.set || b.set ? (a.set && a.wave_exact) || (b.set && b.wave_exact)
                                                                                                   : a.wave_exact && b.wave_exact);
        if (a.constant) return a.set ? wave_exact_as(lane_const(true), a.wave_exact) : folded(b, a);
        if (b.constant) return b.set ? wave_exact_as(lane_const(true), b.wave_exact) : folded(a, b);
        return new_content(lane(m.emit(spv::OpLogicalOr, t_bool, {a.id, b.id})));
    }
    Val bnot_value(const Val& a) {
        if (a.constant) return wave_exact_as(lane_const(!a.set), a.wave_exact);
        const Id r = m.emit(spv::OpLogicalNot, t_bool, {a.id});
        // Each is the other's negation. (A double negation stays as emitted:
        // the lifts made before keep their exact modules.)
        negation_of[r] = a.id;
        negation_of.emplace(a.id, r);
        const Val n = new_content(lane(r));
        if (a.content) {
            complement[n.content] = a.content;
            complement.emplace(a.content, n.content);
        }
        return n;
    }
    Val bxor_value(const Val& a, const Val& b) {
        if (a.constant && b.constant) return wave_exact_as(lane_const(a.set != b.set), a.wave_exact && b.wave_exact);
        if (a.constant) return folded(a.set ? bnot(b) : b, a);
        if (b.constant) return folded(b.set ? bnot(a) : a, b);
        return new_content(lane(m.emit(spv::OpLogicalNotEqual, t_bool, {a.id, b.id})));
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
            // The LS shares its set with the evaluation stage, and takes its user
            // data where a vertex stage keeps its vertex formats (translate.cpp),
            // as does the game's own hull - so in a window's pipeline, where the
            // hull has that slot, the LS may read none.
            if (tess_ls() && window() && !tess->own_params) {
                reject("the LS of a window reads user data as a value, which its pipeline's params do not hold");
            }
            const std::uint32_t member = (tess_ls() || tess_hs()) && !(tess && tess->own_params) ? 7 : 2;
            const Id p = m.access_chain(p_uni_u32, ubo_var, {cu(member), cu(static_cast<std::uint32_t>(key / 4)), cu(static_cast<std::uint32_t>(key % 4))});
            return word(m.load(t_u32, p));
        }
        if (compute() && key < 104) {
            // The SGPRs after the user data, as the reference fills them
            // (translate.cpp run()): the workgroup id's enabled components, in
            // order. The same in every lane of a wave (a wave is one
            // workgroup's), but not in every wave of the dispatch (wg_ids).
            int s = static_cast<int>((opt.rsrc2 >> 1) & 0x1f);
            for (std::uint32_t k = 0; k < 3; ++k) {
                if (!((opt.rsrc2 >> (7 + k)) & 1)) continue;
                if (key == s++) {
                    if (!in_workgroup_id) in_workgroup_id = builtin(m.type_pointer(spv::ScInput, m.type_vector(t_u32, 3)), spv::BiWorkgroupId);
                    const Val v = word(extract(t_u32, m.load(m.type_vector(t_u32, 3), in_workgroup_id), k));
                    wg_ids.insert(v.id);
                    return v;
                }
            }
            return word_const(0);  // as the reference: an SGPR nothing filled reads 0
        }
        if (key >= 256 && compute()) {
            // v0 the invocation's x in the workgroup, v1 and v2 its y and z up
            // to TIDIG_COMP_CNT (translate.cpp run()).
            const int comp = static_cast<int>((opt.rsrc2 >> 11) & 3);
            if (key - 256 > comp || key - 256 > 2) return word_const(0);
            if (!in_local_id) in_local_id = builtin(m.type_pointer(spv::ScInput, m.type_vector(t_u32, 3)), spv::BiLocalInvocationId);
            return word(extract(t_u32, m.load(m.type_vector(t_u32, 3), in_local_id), static_cast<std::uint32_t>(key - 256)));
        }
        if (key >= 256 && tess_ls()) {
            // As the vertex stage of a patch draw: v0 the control point's vertex
            // id, its slot in the patch (v1) 0 - the invocation writes only its
            // own control point - and the instance VGPRs 0 for one control
            // point as attributes (translate.cpp LsVertex) or, in a window, the
            // instance (as LsCompute gives it: the game's draw is instanced).
            if (key == 256) {
                if (!in_vertex_index) in_vertex_index = builtin(p_in_u32, spv::BiVertexIndex);
                return word(m.load(t_u32, in_vertex_index));
            }
            if ((key == 258 || key == 259) && window()) {
                if (!in_instance_index) in_instance_index = builtin(p_in_u32, spv::BiInstanceIndex);
                return word(m.load(t_u32, in_instance_index));
            }
            return word_const(0);
        }
        if (key >= 256 && tess_hs()) {
            // translate.cpp HullTcs with a window: v0 the patch, v1 the control
            // point it runs << 8 (the patch alone in its window: relative patch
            // 0), v2 and v3 0. The patch's absolute index differs between the
            // host's pipelines (a chunked or instanced draw), and no hull of
            // this game reads it.
            if (key == 256) {
                reject("the hull reads v0, the patch's absolute index");
                return word_const(0);
            }
            if (key != 257) return word_const(0);
            if (!in_invocation_id) in_invocation_id = builtin(p_in_u32, spv::BiInvocationId);
            Val v = word(shift(spv::OpShiftLeftLogical, m.load(t_u32, in_invocation_id), cu(8)));
            v.sym = true;
            v.sym_coef = 256;
            return v;
        }
        if (key >= 256 && tess_ds()) {
            // The point's place in the patch from the host's tessellator, and the
            // patch 0: the stage reads only its own patch (translate.cpp). In a
            // window v3 is the patch's absolute index, which, as for the hull, no
            // domain shader of this game reads.
            if (key == 259 && window()) {
                reject("the domain shader reads v3, the patch's absolute index");
                return word_const(0);
            }
            if (key > 257) return word_const(0);
            if (!in_tess_coord) in_tess_coord = builtin(m.type_pointer(spv::ScInput, t_v3f), spv::BiTessCoord);
            return flt(extract(t_f32, m.load(t_v3f, in_tess_coord), static_cast<std::uint32_t>(key - 256)));
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
            in_mask.erase(kKeyVcc);
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
        in_mask.erase(key);
        if (key > 0 && key < 104) {
            lanes.erase(key - 1);
            in_mask.erase(key - 1);
        }
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
        else if (code == 128) {
            Val zero = wave_exact_as(lane_const(false), true);
            zero.helpers_clear = true;
            return zero;
        }
        else if (code == 193) return wave_exact_as(lane_const(true), true);  // -1: every lane
        if (key >= 0) {
            if (poisoned.count(key)) reject(key_name(key) + " is read as a lane mask after an if that left one in it on one arm only");
            if (auto it = lanes.find(key); it != lanes.end()) return note_read(known(it->second));
            reject((key == kKeyVcc ? std::string("vcc") : operand_name(code)) + " is read as a lane mask but does not hold one");
        } else {
            reject("unsupported 64-bit mask operand " + operand_name(code));
        }
        return lane_const(false);
    }
    // `within`: the mask lies within the loop mask of the innermost open loop
    // pixels leave at different iterations (in_mask), as the writer computed it.
    void write_lane(std::uint16_t code, const Val& v, bool within = false) {
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
            in_mask.erase(code - 1);
            in_mask.erase(code + 1);
        } else {
            reject("unsupported 64-bit mask destination " + operand_name(code));
            return;
        }
        if (!note_in_mask(key, within)) return;
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
        if (const Loop* d = innermost_divergent_loop(); d && !in_mask.count(kKeyExec)) {
            reject("v" + std::to_string(idx) + " is written where EXEC may hold lanes beyond the loop mask of the loop at " + hex_offset(d->header) +
                   ": on GCN the lanes that already left would write it");
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
        if (poisoned.count(key)) {
            // The VGPR's lanes hold words a v_writelane_b32 put there (a spill
            // the program has read back), or what each lane held before: where
            // EXEC is clear GCN keeps them, and the lift has no per-pixel value
            // for that. The value written is GCN's where EXEC is set, and the
            // register is exact only there (Lifter::partial): a read that needs
            // it in another pixel - an export, a sample, a mask - rejects.
            const Where at = meet(read, Where{true, Exact{false, e}});
            reg[key] = v;
            reg[key].helper_inexact = v.helper_inexact || inst_helper_inexact;
            poisoned.erase(key);
            partial[key] = at.at;
            note_result(reg[key]);
            note += " v" + std::to_string(idx) + ":exact-under-exec";
            ++spill_reuses;
            return;
        }
        if (v.wave_part || (reg.count(key) && reg.at(key).wave_part)) {
            // A part of the append idiom (Val::wave_part) under a varying
            // EXEC: written where EXEC is the mask it was taken over, it holds
            // in that mask's lanes; a write over one must cover every lane it
            // held in.
            const Val old = value(key);
            if (v.wave_part && !same_lane(e, part_mask_val(v))) {
                reject("v" + std::to_string(idx) + " takes a part of the append idiom under another EXEC than its mask");
                return;
            }
            if (old.wave_part && !implies(part_mask_val(old), e)) {
                reject("v" + std::to_string(idx) + " holds a part of the append idiom and is overwritten only in some of its lanes");
                return;
            }
            reads = read;
            note_exact_write(key, read, e);
            ++masked_writes;
            const Id vn = v.wave_part ? v.id : as_u(v), on = old.wave_part ? old.id : as_u(old);
            Val r = word(m.emit(spv::OpSelect, t_u32, {e.id, vn, on}));
            if (v.wave_part) {
                r.wave_part = v.wave_part;
                r.part_mask = v.part_mask;
                r.part_mask_constant = v.part_mask_constant;
                r.part_mask_set = v.part_mask_set;
            }
            reg[key] = r;
            note += " v" + std::to_string(idx) + (v.wave_part ? ":part" : ":select");
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
        if (tess && sym_holds(v, e)) {  // TessLift: the written word's form, where EXEC is set
            reg[key].sym = true;
            reg[key].sym_base = v.sym_base;
            reg[key].sym_coef = v.sym_coef;
            reg[key].sym_where = sym_where_of(e);
        }
        reg[key].helper_inexact = v.helper_inexact || inst_helper_inexact;  // the old value and EXEC were read: in the flag
        if (uniform(v)) masked_uniform[reg[key].id] = {e, v};
        if (const auto pk = packed_pairs.find(v.id); pk != packed_pairs.end() && !v.constant) masked_packs[reg[key].id] = {pk->second, e.id};
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
        if (inst_uniform && !b.constant) uniform_ids.insert(b.id);  // of values every lane holds alike, as scalar operands are
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
        if (opt.stage != Stage::Vertex || !opt.link_outputs) return {out_param(n)};  // as translate.cpp: an evaluation stage writes param n at n
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
        // x * x: one operand is zero only when both are, so the zero rule
        // never decides (a dot product's squares, a third of the light pass's).
        if (legacy_mul_plain() || a == b) return fmul(a, b);
        if (legacy_mul_min_form()) return fsel(feq(ext(spv::GlslNMin, {ext(spv::GlslFAbs, {a}), ext(spv::GlslFAbs, {b})}), cf(0.0f)), cf(0.0f), fmul(a, b));
        const Id zero = m.emit(spv::OpLogicalOr, t_bool, {feq(a, cf(0.0f)), feq(b, cf(0.0f))});
        return fsel(zero, cf(0.0f), fmul(a, b));
    }
    Id clamp_inf(Id v) {  // as the translator writes it (translate.cpp clamp_inf): the sign onto FLT_MAX's bits
        const Id inf = m.emit(spv::OpIsInf, t_bool, {v});
        const Id sign = ibin(spv::OpBitwiseAnd, m.emit(spv::OpBitcast, t_u32, {v}), cu(0x80000000u));
        return fsel(inf, m.emit(spv::OpBitcast, t_f32, {ibin(spv::OpBitwiseOr, sign, cu(0x7f7fffffu))}), v);
    }

    // `mask`: v_cndmask's select or the carry in, from src2 (VOP3); VCC otherwise.
    // `carry_out`: the pair the carry-out ops write (VOP3 sdst, else VCC).
    bool vop2(std::uint32_t op, const Inst& in, const Val& s0, const Val& s1, const Mods& md, const Val* mask, std::uint16_t carry_out) {
        const auto F0 = [&] { return src_f(s0, 0, md); };
        const auto F1 = [&] { return src_f(s1, 1, md); };
        const auto out_f = [&](Id v) { write_v(in.dst, result_f(v, md)); return true; };
        const auto out_u = [&](Id v) { write_v(in.dst, word(v)); return true; };
        const auto carry = [&](Id c) {  // the translator's ballot of carry & EXEC
            Val bit = new_content(lane(c));  // the carry, ANDed with EXEC below
            bit.helper_inexact = inst_helper_inexact;
            write_lane(carry_out, band(bit, exec_lane()), mask_within(kExecLo));  // a VALU mask: clear where EXEC is
        };
        const auto vdst = [&] { return as_f(value(256 + in.dst)); };
        switch (op) {
        case 0: {  // v_cndmask_b32
            const Val bit = mask ? *mask : read_lane(kVccLo);
            if (bit.kind != Kind::Lane) return true;
            if (md.omod || md.clamp) {
                reject("v_cndmask_b32 with an output modifier or clamp");
                return true;
            }
            // VOP3: the sources' abs and neg modifiers apply, abs first, as to a
            // float operation's - on the sign bit, as the translator applies them
            // (translate.cpp vop2), so the selected word keeps every other bit.
            const auto modified = [&](const Val& x, int k) {
                const bool abs = (md.abs >> k) & 1, neg = (md.neg >> k) & 1;
                if (!abs && !neg) return x;
                if (x.constant) return word_const((abs ? x.bits & 0x7fffffffu : x.bits) ^ (neg ? 0x80000000u : 0u));
                Id w = as_u(x);
                if (abs) w = ibin(spv::OpBitwiseAnd, w, cu(0x7fffffffu));
                if (neg) w = ibin(spv::OpBitwiseXor, w, cu(0x80000000u));
                return word(w);
            };
            const Val a = modified(s0, 0), b = modified(s1, 1);
            if (bit.constant) {
                write_v(in.dst, bit.set ? b : a);
            } else if (a.kind == Kind::Float || b.kind == Kind::Float) {
                write_v(in.dst, flt(fsel(bit.id, as_f(b), as_f(a))));
            } else {
                write_v(in.dst, word(m.emit(spv::OpSelect, t_u32, {bit.id, as_u(b), as_u(a)})));
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
            Val x = s0, y = s1;
            if ((x.wave_part || y.wave_part) && !combine_wave_parts(op, x, y)) return true;
            Id a = as_u(x), b = as_u(y);
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
        Val bit = new_content(lane(cond));         // the comparison, ANDed with EXEC below
        bit.helper_inexact = inst_helper_inexact;  // from the operands
        const Val e = exec_lane();
        const Val c = band(bit, e);
        note_compare_split(c, e, op, s0, s1, md, cond);
        const bool within = mask_within(kExecLo);  // clear where EXEC is
        write_lane(dst_pair, c, within);
        if (op & 0x10) write_lane(kExecLo, c, within);
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
                ++res.wave.spill_lanes;
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
        ++res.wave.spill_lanes;
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

    // ---- affine words (Val::sym, TessLift) -------------------------------------------
    // Invocations a word may differ between: a hull's control points, else 1.
    std::uint32_t sym_invocations() const { return tess_hs() ? std::max<std::uint32_t>(1, tess->control_points) : 1u; }
    std::vector<Val> sym_masks;  // Val::sym_where
    std::uint32_t sym_where_of(const Val& e) {
        if (e.constant && e.set) return 0;
        for (std::size_t k = 0; k < sym_masks.size(); ++k) {
            if (same_lane(sym_masks[k], e)) return static_cast<std::uint32_t>(k + 1);
        }
        sym_masks.push_back(e);
        return static_cast<std::uint32_t>(sym_masks.size());
    }
    // The affine form holds for this invocation wherever EXEC is `e`.
    bool sym_holds(const Val& v, const Val& e) const {
        return v.sym && v.kind != Kind::Lane && (v.sym_where == 0 || implies(e, sym_masks[v.sym_where - 1]));
    }
    Val current_exec() {
        const auto it = lanes.find(kKeyExec);
        return it == lanes.end() ? lane_const(false) : it->second;
    }
    // The operand's value without the side effects of a read (value() runs
    // the instruction's own).
    Val peek(std::uint16_t code, const Inst& in) {
        if (code >= 256) {
            const auto it = reg.find(code);
            return it != reg.end() ? it->second : initial_value(code);
        }
        if (const int key = scalar_key(code); key >= 0) {
            const auto it = reg.find(key);
            return it != reg.end() ? it->second : initial_value(key);
        }
        if (code == kLiteral) return word_const(in.literal);
        if (code >= 128 && code <= 192) return word_const(code - 128u);
        if (code >= 193 && code <= 208) return word_const(static_cast<std::uint32_t>(-static_cast<int>(code - 192)));
        return Val{};
    }
    // GCN's result of an integer operation the address arithmetic uses, as
    // the translator and the lift compute it; false for any other.
    static bool int_op(const Inst& in, std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t& r) {
        const auto s24 = [](std::uint32_t x) { return static_cast<std::uint32_t>(static_cast<std::int32_t>(x << 8) >> 8); };
        const auto u24 = [](std::uint32_t x) { return x & 0xffffffu; };
        std::uint32_t op = 0xffff;
        if (in.enc == Enc::VOP2) op = in.op;
        if (in.enc == Enc::VOP3 && in.op >= 0x100 && in.op < 0x140) op = in.op - 0x100;
        if (in.enc == Enc::VOP3 && in.op >= 0x140) op = in.op;  // VOP3-only opcodes keep their numbers
        switch (op) {
        case 9: r = s24(a) * s24(b); return true;      // v_mul_i32_i24
        case 11: r = u24(a) * u24(b); return true;     // v_mul_u32_u24
        case 21: r = a >> (b & 31); return true;       // v_lshr_b32
        case 22: r = b >> (a & 31); return true;       // v_lshrrev_b32
        case 25: r = a << (b & 31); return true;       // v_lshl_b32
        case 26: r = b << (a & 31); return true;       // v_lshlrev_b32
        case 27: r = a & b; return true;               // v_and_b32
        case 28: r = a | b; return true;               // v_or_b32
        case 37: r = a + b; return true;               // v_add_i32
        case 38: r = a - b; return true;               // v_sub_i32
        case 39: r = b - a; return true;               // v_subrev_i32
        case 0x142: r = s24(a) * s24(b) + c; return true;  // v_mad_i32_i24
        case 0x143: r = u24(a) * u24(b) + c; return true;  // v_mad_u32_u24
        case 0x148: {                                       // v_bfe_u32
            const std::uint32_t off = b & 31, width = c & 31;
            r = width ? (a >> off) & ((1u << width) - 1u) : 0u;
            return true;
        }
        case 0x169: case 0x16b: r = a * b; return true;     // v_mul_lo_u32 / v_mul_lo_i32
        default: return false;
        }
    }
    static bool scalar_int_op(const Inst& in, std::uint32_t a, std::uint32_t b, std::uint32_t& r) {
        if (in.enc == Enc::SOPK) {
            if (in.op == 0) { r = static_cast<std::uint32_t>(in.imm); return true; }      // s_movk_i32
            if (in.op == 15) { r = a + static_cast<std::uint32_t>(in.imm); return true; } // s_addk_i32
            if (in.op == 16) { r = a * static_cast<std::uint32_t>(in.imm); return true; } // s_mulk_i32
            return false;
        }
        if (in.enc != Enc::SOP2) return false;
        switch (in.op) {
        case 0: case 2: r = a + b; return true;
        case 1: case 3: r = a - b; return true;
        case 14: r = a & b; return true;
        case 16: r = a | b; return true;
        case 30: r = a << (b & 31); return true;
        case 32: r = a >> (b & 31); return true;
        case 38: r = a * b; return true;
        case 39: case 40: {  // s_bfe_u32 / s_bfe_i32: offset [4:0], width [22:16] of S1
            const std::uint32_t off = b & 31, width = (b >> 16) & 0x7f;
            if (!width) { r = 0; return true; }
            if (width >= 32) { r = a >> off; return true; }
            r = (a >> off) & ((1u << width) - 1u);
            if (in.op == 40 && ((r >> (width - 1)) & 1)) r |= ~((1u << width) - 1u);
            return true;
        }
        default: return false;
        }
    }
    // The result's affine form, from its operands' (`ops`, each base + coef *
    // the invocation): computed for every invocation and kept when the results
    // are affine again.
    template <typename F>
    bool sym_eval(const std::vector<Val>& ops, F&& f, std::uint32_t& base, std::uint32_t& coef) {
        const Val e = current_exec();
        for (const Val& v : ops) {
            if (!sym_holds(v, e)) return false;
        }
        const std::uint32_t n = sym_invocations();
        std::uint32_t r0 = 0, r1 = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::vector<std::uint32_t> x;
            for (const Val& v : ops) x.push_back(v.sym_base + v.sym_coef * i);
            std::uint32_t r = 0;
            if (!f(x, r)) return false;
            if (i == 0) r0 = r;
            if (i == 1) r1 = r;
            if (i >= 1 && r != r0 + (r1 - r0) * i) return false;
        }
        base = r0;
        coef = n > 1 ? r1 - r0 : 0;
        return true;
    }
    // After an integer VALU write with EXEC known set: its affine form.
    void sym_vector(const Inst& in, const std::vector<Val>& ops) {
        if (in.enc == Enc::VOP3 && (in.abs || in.neg || in.omod || in.clamp)) return;
        const auto it = reg.find(256 + in.dst);
        if (it == reg.end() || it->second.kind == Kind::Lane) return;
        const Val e = current_exec();
        if (e.constant && !e.set) return;  // nothing written
        std::uint32_t base = 0, coef = 0;
        if (sym_eval(ops, [&](const std::vector<std::uint32_t>& x, std::uint32_t& r) {
                return int_op(in, x[0], x.size() > 1 ? x[1] : 0, x.size() > 2 ? x[2] : 0, r);
            }, base, coef)) {
            // Under a varying EXEC the register is a select: the form holds where EXEC is set.
            it->second.sym = true;
            it->second.sym_base = base;
            it->second.sym_coef = coef;
            it->second.sym_where = sym_where_of(e);
        }
    }
    void sym_scalar(const Inst& in, const std::vector<Val>& ops) {
        const int key = scalar_key(in.dst);
        if (key < 0) return;
        const auto it = reg.find(key);
        if (it == reg.end() || it->second.kind == Kind::Lane) return;
        std::uint32_t base = 0, coef = 0;
        if (sym_eval(ops, [&](const std::vector<std::uint32_t>& x, std::uint32_t& r) {
                return scalar_int_op(in, x.empty() ? 0 : x[0], x.size() > 1 ? x[1] : 0, r);
            }, base, coef) && coef == 0) {
            it->second.sym = true;
            it->second.sym_base = base;
            it->second.sym_coef = 0;
        }
    }

    void valu(const Inst& in) {
        // TessLift: the operands as they were, for the result's affine form.
        std::vector<Val> sym_ops;
        if (tess && (in.enc == Enc::VOP2 || in.enc == Enc::VOP3)) {
            std::uint32_t probe = 0;
            if (int_op(in, 0, 0, 0, probe)) {
                sym_ops.push_back(peek(in.src0, in));
                sym_ops.push_back(peek(in.src1, in));
                if (in.enc == Enc::VOP3 && in.op >= 0x140 && in.op < 0x160) sym_ops.push_back(peek(in.src2, in));
            }
        }
        valu_inst(in);
        if (!sym_ops.empty() && res.rejections.empty()) sym_vector(in, sym_ops);
    }

    void valu_inst(const Inst& in) {
        Mods md;
        const auto unsupported = [&] { reject(std::string("unsupported ") + mnemonic(in)); };
        if (in.enc == Enc::VOP2) {
            if (in.op == 1 || in.op == 2) return lane_op(in);  // v_readlane_b32 / v_writelane_b32
            if (compute() && (in.op == 35 || in.op == 36)) return mbcnt(in, in.op == 36, in.src0, in.src1);
            if (!vop2(in.op, in, read(in.src0, in), read(in.src1, in), md, nullptr, kVccLo)) unsupported();
            return;
        }
        if (in.enc == Enc::VOP1) {
            if (in.op == 2 && compute()) return readfirstlane(in);
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
            if (compute() && (op == 35 || op == 36)) {
                if (in.abs || in.neg || in.omod || in.clamp) {
                    unsupported();
                    return;
                }
                return mbcnt(in, op == 36, in.src0, in.src1);
            }
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
        if (in.op == 0x182 && compute()) return readfirstlane(in);
        if (!vop1(in.op - 0x180, in, read(in.src0, in), md)) unsupported();
    }
    // v_readfirstlane_b32 (compute): the first EXEC lane's word, which is
    // this invocation's where every lane holds it alike.
    void readfirstlane(const Inst& in) {
        const Val v = read(in.src0, in);
        if (v.kind == Kind::Lane) {
            reject("v_readfirstlane_b32 of a lane mask");
            return;
        }
        if (!uniform(v)) {
            reject("v_readfirstlane_b32 of a value lanes may hold differently: the first EXEC lane's is another invocation's");
            return;
        }
        write_scalar(in.dst, v);
        ++first_lane_reads;
    }

    void sop1(const Inst& in) {
        switch (in.op) {
        case 3: write_scalar(in.dst, read(in.src0, in)); return;
        case 4: {
            const bool is_mask = in.src0 == kExecLo || in.src0 == kVccLo || (in.src0 < 104 && lanes.count(in.src0));
            if (is_mask || in.dst == kExecLo || in.dst == kVccLo) {
                write_lane(in.dst, read_lane(in.src0), mask_within(in.src0));
            } else {
                const Val lo = read(in.src0, in), hi = read(static_cast<std::uint16_t>(in.src0 + 1), in);
                write_scalar(in.dst, lo);
                write_scalar(static_cast<std::uint16_t>(in.dst + 1), hi);
            }
            return;  // s_mov_b64 leaves SCC, as in the translator
        }
        case 10: {  // s_wqm_b64
            if (opt.stage != Stage::Pixel) {
                reject("whole-quad mode outside a pixel shader: a quad's other lanes are other vertices");
                return;
            }
            ++res.wave.whole_quad;
            const Val a = read_lane(in.src0);
            if (!a.constant && !widen_ok) {
                if (begin_whole_quad(in, a)) return;
                reject("whole-quad mode of a mask that is not constant: the result depends on the other pixels of the quad");
                return;
            }
            // A constant the body of such a loop knows holds for its pixel
            // only: GCN's quads also hold lanes that already left.
            if (in_divergent_loop()) {
                reject("whole-quad mode inside a loop pixels leave at different iterations, where GCN would set the lanes of a quad that left "
                       "while a quad-mate stays");
                return;
            }
            // Clear for every pixel the lift runs, but maybe not in GCN's helper
            // lanes: their quads' covered pixels would come out set.
            if (!a.set && a.helper_inexact && !a.helpers_clear && !widen_ok) {
                reject("whole-quad mode of a mask clear for every covered pixel that GCN's helper lanes may have set: GCN sets their quads' pixels");
                return;
            }
            Val r = a;
            // Whole quads: a constant set mask leaves every lane of a quad with a
            // pixel set, helpers included; a clear one stays as exact as it was.
            r.helper_inexact = !a.set && a.helper_inexact;
            r.helpers_clear = !a.set && a.helpers_clear;
            if (widen_ok) {
                // Widening a kill region's EXEC. From EXEC known set, every pixel
                // computes the region, as GCN's widened quads do wherever a pixel
                // survives; otherwise the pixel keeps its own bit. Either way the
                // pixels that differ must be killed at s_endpgm (checked there).
                Region& k = *open_regions.back();
                r.helper_inexact = true;  // the helper lanes GCN widens to, where the pixel keeps its own bit
                r.helpers_clear = false;
                r = new_content(r);       // GCN's is its whole quads
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
            const bool a_in = mask_within(in.src0), e_in = mask_within(kExecLo);
            write_lane(in.dst, e, e_in);
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
            // and: within if either is; or, xor: if both are; andn2: if the mask
            // is; the complements of EXEC (orn2, nand, nor, xnor) hold lanes outside it.
            const bool r_in = in.op == 36 ? a_in || e_in : in.op == 37 || in.op == 38 ? a_in && e_in : in.op == 39 ? a_in : false;
            write_lane(kExecLo, r, r_in);
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
            // and: within the loop mask if either operand is; andn2: if the
            // first is; or, xor: if both are; orn2 holds lanes outside it.
            const bool a_in = mask_within(in.src0), b_in = mask_within(in.src1);
            const bool r_in = in.op == 15 ? a_in || b_in : in.op == 21 ? a_in : in.op == 23 ? false : a_in && b_in;
            write_lane(in.dst, r, r_in);
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
            if (compute()) return;  // a dispatch exports nothing: its stores were made under EXEC already
            const Val e = exec_lane();
            if (tess_hs() || tess_ls()) return;  // outputs written under EXEC already; nothing to discard
            if (vertex_like()) {
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
        case 10:  // s_barrier: the patch's control points meet (translate.cpp: a control barrier in a control stage)
            if (compute()) {
                // The workgroup's invocations meet, their memory made visible to
                // each other (the reference names workgroup memory only; the
                // buffers and images a dispatch shares are uniform and image
                // memory). Every invocation must reach it: not inside a loop
                // invocations leave at different iterations.
                if (!require_converged("s_barrier")) return;
                m.emit_void(spv::OpControlBarrier, {cu(spv::ScopeWorkgroup), cu(spv::ScopeWorkgroup),
                                                    cu(spv::MsAcquireRelease | spv::MsUniformMemory | spv::MsWorkgroupMemory | spv::MsImageMemory)});
                ++barriers;
                return;
            }
            if (!tess_hs()) {
                reject("s_barrier outside a hull shader");
                return;
            }
            if (!open_regions.empty() || !arms.empty() || !open_loops.empty()) {
                reject("s_barrier inside a region, an if or a loop, where the patch's invocations may not all reach it");
                return;
            }
            m.emit_void(spv::OpControlBarrier, {cu(spv::ScopeWorkgroup), cu(spv::ScopeInvocation), cu(spv::MsNone)});
            ++barriers;
            return;
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

    // ---- guest memory (compute) ------------------------------------------------------
    // A dispatch's translation binds no buffer: every load, store and atomic
    // walks the page table at StageParams::l1_table (translate.cpp xlate,
    // guest_ptr), V#s and pointers read as data. The lift reads and writes the
    // same guest memory the same way, each access as the reference makes it:
    // the address from the same words, the same dwords loaded (a run of them
    // in one page with one walk), the same format conversions, and stores,
    // atomics and partial-dword writes made only where the invocation's EXEC
    // bit is set, as GCN performs them only for the lanes EXEC holds.
    std::size_t guest_loads = 0, guest_stores = 0, guest_atomics = 0, image_stores = 0;
    void declare_guest_memory() {
        p_uni_u64 = m.type_pointer(spv::ScUniform, t_u64);
        const Id t_block = m.type_struct({t_u32});
        m.decorate(t_block, spv::DecBlock);
        m.member_decorate(t_block, 0, spv::DecOffset, {0});
        p_psb_block = m.type_pointer(spv::ScPhysicalStorageBuffer, t_block);
        p_psb_u32 = m.type_pointer(spv::ScPhysicalStorageBuffer, t_u32);
        const Id t_block64 = m.type_struct({t_u64});
        m.decorate(t_block64, spv::DecBlock);
        m.member_decorate(t_block64, 0, spv::DecOffset, {0});
        p_psb_block64 = m.type_pointer(spv::ScPhysicalStorageBuffer, t_block64);
        p_psb_u64 = m.type_pointer(spv::ScPhysicalStorageBuffer, t_u64);
        // xlate(va): l1[va >> 32] -> the L2 table; l2[(va >> 16) & 0xffff] ->
        // the 64 KiB page's device address (translate.h StageParams).
        fn_xlate = m.begin_function(t_u64, m.type_function(t_u64, {t_u64}));
        const Id va = m.emit(spv::OpFunctionParameter, t_u64, {});
        m.label();
        const Id l1 = m.load(t_u64, m.access_chain(p_uni_u64, ubo_var, {cu(0)}));
        const Id i1 = ext(spv::GlslUMin, {m.emit(spv::OpShiftRightLogical, t_u64, {va, m.const_u64(32)}), m.const_u64(kL1Entries - 1)}, t_u64);
        const Id l1_at = m.emit(spv::OpIAdd, t_u64, {l1, m.emit(spv::OpShiftLeftLogical, t_u64, {i1, m.const_u64(3)})});
        const Id l2 = m.emit(spv::OpLoad, t_u64, {m.access_chain(p_psb_u64, m.emit(spv::OpConvertUToPtr, p_psb_block64, {l1_at}), {cu(0)}), 2u, 8u});
        const Id i2 = m.emit(spv::OpBitwiseAnd, t_u64, {m.emit(spv::OpShiftRightLogical, t_u64, {va, m.const_u64(kPageShift)}), m.const_u64(kL2Entries - 1)});
        const Id l2_at = m.emit(spv::OpIAdd, t_u64, {l2, m.emit(spv::OpShiftLeftLogical, t_u64, {i2, m.const_u64(3)})});
        const Id page = m.emit(spv::OpLoad, t_u64, {m.access_chain(p_psb_u64, m.emit(spv::OpConvertUToPtr, p_psb_block64, {l2_at}), {cu(0)}), 2u, 8u});
        const Id within = m.emit(spv::OpBitwiseAnd, t_u64, {va, m.const_u64((1ull << kPageShift) - 1)});
        m.emit_void(spv::OpReturnValue, {m.emit(spv::OpIAdd, t_u64, {page, within})});
        m.end_function();
        m.name(fn_xlate, "xlate");
    }
    Id make_u64(Id lo, Id hi) {
        return m.emit(spv::OpBitwiseOr, t_u64, {m.emit(spv::OpUConvert, t_u64, {lo}),
                                                m.emit(spv::OpShiftLeftLogical, t_u64, {m.emit(spv::OpUConvert, t_u64, {hi}), m.const_u64(32)})});
    }
    Id addr_add(Id base64, Id off32) { return m.emit(spv::OpIAdd, t_u64, {base64, m.emit(spv::OpUConvert, t_u64, {off32})}); }
    Id addr_addc(Id base64, std::uint32_t off) { return off ? m.emit(spv::OpIAdd, t_u64, {base64, m.const_u64(off)}) : base64; }
    Id host_dword(Id host) { return m.access_chain(p_psb_u32, m.emit(spv::OpConvertUToPtr, p_psb_block, {host}), {cu(0)}); }
    Id guest_ptr(Id va) { return host_dword(m.emit(spv::OpFunctionCall, t_u64, {fn_xlate, va})); }
    Id load_u32_at(Id va) { return m.emit(spv::OpLoad, t_u32, {guest_ptr(va), 2u /* Aligned */, 4u}); }
    void store_u32_at(Id va, Id v) { m.emit_void(spv::OpStore, {guest_ptr(va), v, 2u, 4u}); }
    // A selection on `cond` whose arms compute `n` words each; their values
    // joined at the merge (cur_label follows the blocks emitted).
    std::vector<Id> select_words(Id cond, std::size_t n, const std::function<std::vector<Id>()>& then_arm,
                                 const std::function<std::vector<Id>()>& else_arm) {
        const Id l_then = m.fresh(), l_else = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {cond, l_then, l_else});
        cur_label = m.label(l_then);
        const std::vector<Id> a = then_arm();
        const Id end_a = cur_label;
        m.emit_void(spv::OpBranch, {l_merge});
        cur_label = m.label(l_else);
        const std::vector<Id> b = else_arm();
        const Id end_b = cur_label;
        m.emit_void(spv::OpBranch, {l_merge});
        cur_label = m.label(l_merge);
        std::vector<Id> out;
        for (std::size_t k = 0; k < n; ++k) out.push_back(m.emit(spv::OpPhi, t_u32, {a[k], end_a, b[k], end_b}));
        return out;
    }
    // translate.cpp load_u32_run: n dwords with one walk when they sit in one page.
    std::vector<Id> load_u32_run(Id va, int n) {
        ++guest_loads;
        if (n == 1) return {load_u32_at(va)};
        const Id within = m.emit(spv::OpBitwiseAnd, t_u64, {va, m.const_u64((1ull << kPageShift) - 1)});
        const Id end = m.emit(spv::OpIAdd, t_u64, {within, m.const_u64(static_cast<std::uint64_t>(n) * 4)});
        const Id one_page = m.emit(spv::OpULessThanEqual, t_bool, {end, m.const_u64(1ull << kPageShift)});
        return select_words(
            one_page, static_cast<std::size_t>(n),
            [&] {
                const Id host = m.emit(spv::OpFunctionCall, t_u64, {fn_xlate, va});
                std::vector<Id> w;
                for (int k = 0; k < n; ++k) {
                    const Id at = k ? m.emit(spv::OpIAdd, t_u64, {host, m.const_u64(static_cast<std::uint64_t>(k) * 4)}) : host;
                    w.push_back(m.emit(spv::OpLoad, t_u32, {host_dword(at), 2u, 4u}));
                }
                return w;
            },
            [&] {
                std::vector<Id> w;
                for (int k = 0; k < n; ++k) w.push_back(load_u32_at(addr_addc(va, static_cast<std::uint32_t>(k) * 4)));
                return w;
            });
    }
    // translate.cpp load_u32_any: n dwords from a byte address of any
    // alignment, the aligned dwords that cover them shifted together.
    std::vector<Id> load_u32_any(Id va, int n) {
        const Id low = m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpBitwiseAnd, t_u64, {va, m.const_u64(3)})});
        const Id aligned = m.emit(spv::OpBitwiseAnd, t_u64, {va, m.const_u64(~3ull)});
        return select_words(
            ieq(low, cu(0)), static_cast<std::size_t>(n), [&] { return load_u32_run(aligned, n); },
            [&] {
                const std::vector<Id> w = load_u32_run(aligned, n + 1);
                const Id sh = shift(spv::OpShiftLeftLogical, low, cu(3));
                const Id rsh = ibin(spv::OpISub, cu(32), sh);
                std::vector<Id> out;
                for (int k = 0; k < n; ++k) {
                    out.push_back(ibin(spv::OpBitwiseOr, shift(spv::OpShiftRightLogical, w[static_cast<std::size_t>(k)], sh),
                                       shift(spv::OpShiftLeftLogical, w[static_cast<std::size_t>(k) + 1], rsh)));
                }
                return out;
            });
    }
    // `body` where cond holds, as a selection; nothing joins.
    void if_then(Id cond, const std::function<void()>& body) {
        const Id l_then = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {cond, l_then, l_merge});
        cur_label = m.label(l_then);
        body();
        m.emit_void(spv::OpBranch, {l_merge});
        cur_label = m.label(l_merge);
    }
    // translate.cpp store_masked_any: the bytes of (v0, v1) the mask selects,
    // at a byte address of any alignment - a dword covered whole a plain
    // store, one covered in part an atomic and/or, so the neighbouring bytes
    // other invocations store survive. Inside the caller's EXEC guard.
    void store_masked_any(Id va, Id v0, Id v1, Id m0, Id m1) {
        const Id low = m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpBitwiseAnd, t_u64, {va, m.const_u64(3)})});
        const Id aligned = m.emit(spv::OpBitwiseAnd, t_u64, {va, m.const_u64(~3ull)});
        const Id sh = shift(spv::OpShiftLeftLogical, low, cu(3));
        const Id rsh = ibin(spv::OpISub, cu(32), sh);
        const Id whole = ieq(sh, cu(0));
        const auto sel = [&](Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_u32, {c, a, b}); };
        const auto shl = [&](Id a, Id b) { return shift(spv::OpShiftLeftLogical, a, b); };
        const auto shr = [&](Id a, Id b) { return shift(spv::OpShiftRightLogical, a, b); };
        const Id dv[3] = {shl(v0, sh), sel(whole, v1, ibin(spv::OpBitwiseOr, shl(v1, sh), shr(v0, rsh))), sel(whole, cu(0), shr(v1, rsh))};
        const Id dm[3] = {shl(m0, sh), sel(whole, m1, ibin(spv::OpBitwiseOr, shl(m1, sh), shr(m0, rsh))), sel(whole, cu(0), shr(m1, rsh))};
        for (int k = 0; k < 3; ++k) {
            const Id at = addr_addc(aligned, static_cast<std::uint32_t>(k) * 4);
            const Id full = ieq(dm[k], cu(0xffffffffu));
            if_then(full, [&] { store_u32_at(at, dv[k]); });
            if_then(m.emit(spv::OpLogicalAnd, t_bool, {m.emit(spv::OpINotEqual, t_bool, {dm[k], cu(0)}), m.emit(spv::OpLogicalNot, t_bool, {full})}), [&] {
                const Id p = guest_ptr(at);
                m.emit(spv::OpAtomicAnd, t_u32, {p, cu(spv::ScopeDevice), cu(spv::MsNone), m.emit(spv::OpNot, t_u32, {dm[k]})});
                m.emit(spv::OpAtomicOr, t_u32, {p, cu(spv::ScopeDevice), cu(spv::MsNone), ibin(spv::OpBitwiseAnd, dv[k], dm[k])});
            });
        }
    }
    // A value computed where the invocation's EXEC bit is set (an atomic's
    // result), 0 elsewhere - where the VGPR write that takes it is masked off.
    Id where_exec_value(const Val& e, const std::function<Id()>& body) {
        if (e.constant) return e.set ? body() : cu(0);
        const Id from = cur_label;
        const Id l_then = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {e.id, l_then, l_merge});
        cur_label = m.label(l_then);
        const Id v = body();
        const Id end = cur_label;
        m.emit_void(spv::OpBranch, {l_merge});
        cur_label = m.label(l_merge);
        return m.emit(spv::OpPhi, t_u32, {v, end, cu(0), from});
    }
    // Scalar loads (s_load_dword*, s_buffer_load_dword*): translate.cpp smrd
    // without a binding - the pointer, or the V#'s 40-bit base, plus the
    // offset (dwords as an immediate or literal, bytes in an SGPR), its dwords
    // read with one walk where they share a page. Stored at once and again at
    // the next s_waitcnt on lgkmcnt, as the reference lands them; every lane
    // reads the same address, so the words are what every lane holds alike.
    void guest_scalar_load(const Inst& in, int count) {
        const Val lo = read(in.src0, in), hi = read(static_cast<std::uint16_t>(in.src0 + 1), in);
        const Id base = in.op < 8 ? make_u64(as_u(lo), as_u(hi)) : make_u64(as_u(lo), ibin(spv::OpBitwiseAnd, as_u(hi), cu(0xff)));
        Id off;
        if (in.imm_flag) {
            off = cu(static_cast<std::uint32_t>(in.imm) * 4);
        } else if (in.has_literal) {
            off = cu(in.literal * 4);  // the CI literal form: dwords, like the immediate (translate.cpp smrd)
        } else {
            off = as_u(read(static_cast<std::uint16_t>(in.imm), in));
        }
        if (!res.rejections.empty()) return;
        const std::vector<Id> w = load_u32_run(addr_add(base, off), count);
        Pending p{in.sdst, {}};
        for (int k = 0; k < count; ++k) {
            const Val v = word(w[static_cast<std::size_t>(k)]);
            write_scalar(static_cast<std::uint16_t>(in.sdst + k), v);
            p.values.push_back(v);
        }
        pending.push_back(std::move(p));
        ++buffer_loads;
    }
    // translate.cpp buffer_address: the V#'s 40-bit base, plus the
    // instruction's offset, soffset, the index times the V#'s stride (idxen)
    // and the offset VGPR (offen), as a 32-bit sum.
    struct GuestAddr {
        Id addr = 0, w3 = 0;
    };
    GuestAddr guest_buffer_address(const Inst& in) {
        GuestAddr g;
        if (in.addr64 || in.lds || in.tfe) {
            reject(std::string(mnemonic(in) ? mnemonic(in) : "a buffer access") + (in.addr64 ? " with addr64" : in.lds ? " into LDS" : " with TFE") +
                   " is not lifted");
            return g;
        }
        const Id w0 = as_u(read(in.srsrc, in)), w1 = as_u(read(static_cast<std::uint16_t>(in.srsrc + 1), in));
        const Id base = make_u64(w0, ibin(spv::OpBitwiseAnd, w1, cu(0xff)));
        const Id stride = ibin(spv::OpBitwiseAnd, shift(spv::OpShiftRightLogical, w1, cu(16)), cu(0x3fff));
        Id off = ibin(spv::OpIAdd, cu(in.offset12), as_u(read(in.soffset, in)));
        int va = in.vaddr;
        if (in.idxen) off = ibin(spv::OpIAdd, off, ibin(spv::OpIMul, as_u(value(256 + va++)), stride));
        if (in.offen) off = ibin(spv::OpIAdd, off, as_u(value(256 + va)));
        g.addr = addr_add(base, off);
        return g;
    }
    // translate.cpp format_decode: a static format's components from the
    // dwords of one element, 0, 0, 0, 1 past them.
    std::vector<Id> format_decode(const std::vector<Id>& words, std::uint32_t dfmt, std::uint32_t nfmt) {
        int count = 0, bits[4] = {};
        std::vector<Id> out(4, cu(0));
        if (!format_layout(dfmt, count, bits)) return out;
        for (int k = 0, bit = 0; k < count; bit += bits[k], ++k) {
            const Id w = words[static_cast<std::size_t>(bit / 32)];
            const Id raw = bits[k] == 32 ? w : m.emit(spv::OpBitFieldUExtract, t_u32, {w, cu(static_cast<std::uint32_t>(bit % 32)), cu(static_cast<std::uint32_t>(bits[k]))});
            out[static_cast<std::size_t>(k)] = convert_component(raw, bits[k], nfmt);
        }
        for (int k = count; k < 4; ++k) out[static_cast<std::size_t>(k)] = k == 3 ? (nfmt == 4 || nfmt == 5 ? cu(1) : as_u(float_const(1.0f))) : cu(0);
        return out;
    }
    // translate.cpp format_load_dynamic: buffer_load_format_* with the format
    // from the V#'s word 3, the 32-, 16- and 8-bit families, 0, 0, 0, 1 past
    // the format's components.
    std::vector<Id> format_load_dynamic(Id va, Id dfmt, Id nfmt, int want) { return format_decode_dynamic(load_u32_any(va, want), dfmt, nfmt, want); }
    // translate.cpp format_decode_dynamic: the components from an element's first dwords.
    std::vector<Id> format_decode_dynamic(const std::vector<Id>& words, Id dfmt, Id nfmt, int want) {
        const Id w0 = words[0], w1 = want > 1 ? words[1] : w0, w2 = want > 2 ? words[2] : w0, w3 = want > 3 ? words[3] : w0;
        const auto sel = [&](Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_u32, {c, a, b}); };
        const auto lor = [&](Id a, Id b) { return m.emit(spv::OpLogicalOr, t_bool, {a, b}); };
        const Id cnt = sel(lor(lor(ieq(dfmt, cu(1)), ieq(dfmt, cu(2))), ieq(dfmt, cu(4))), cu(1),
                           sel(lor(lor(ieq(dfmt, cu(3)), ieq(dfmt, cu(5))), ieq(dfmt, cu(11))), cu(2), sel(ieq(dfmt, cu(13)), cu(3), cu(4))));
        const auto narrow = [&](Id raw, int bits, Id flt_bits) {
            const Id sext = from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(raw), cu(0), cu(static_cast<std::uint32_t>(bits))}));
            const float maxu = static_cast<float>((1u << bits) - 1), maxs = static_cast<float>((1u << (bits - 1)) - 1);
            const Id unorm = as_u(flt(fdiv(m.emit(spv::OpConvertUToF, t_f32, {raw}), cf(maxu))));
            const Id snorm = as_u(flt(ext(spv::GlslNMax, {fdiv(m.emit(spv::OpConvertSToF, t_f32, {to_i(sext)}), cf(maxs)), cf(-1.0f)})));
            const Id uscaled = as_u(flt(m.emit(spv::OpConvertUToF, t_f32, {raw}))), sscaled = as_u(flt(m.emit(spv::OpConvertSToF, t_f32, {to_i(sext)})));
            return sel(ieq(nfmt, cu(0)), unorm,
                       sel(lor(ieq(nfmt, cu(1)), ieq(nfmt, cu(6))), snorm,
                           sel(ieq(nfmt, cu(2)), uscaled, sel(ieq(nfmt, cu(3)), sscaled, sel(ieq(nfmt, cu(4)), raw, sel(ieq(nfmt, cu(5)), sext, flt_bits))))));
        };
        std::vector<Id> out;
        for (int k = 0; k < want; ++k) {
            const Id word32 = k == 0 ? w0 : k == 1 ? w1 : k == 2 ? w2 : w3;
            const Id raw16 = m.emit(spv::OpBitFieldUExtract, t_u32, {k < 2 ? w0 : w1, cu(static_cast<std::uint32_t>((k & 1) * 16)), cu(16)});
            const Id half = as_u(flt(extract(t_f32, ext(spv::GlslUnpackHalf2x16, {raw16}, t_v2f), 0)));
            const Id v16 = narrow(raw16, 16, half);
            const Id raw8 = m.emit(spv::OpBitFieldUExtract, t_u32, {w0, cu(static_cast<std::uint32_t>(k * 8)), cu(8)});
            const Id v8 = narrow(raw8, 8, raw8);
            const Id is16 = lor(lor(ieq(dfmt, cu(2)), ieq(dfmt, cu(5))), ieq(dfmt, cu(12)));
            const Id is8 = lor(ieq(dfmt, cu(1)), lor(ieq(dfmt, cu(3)), ieq(dfmt, cu(10))));
            const Id loaded = sel(is16, v16, sel(is8, v8, word32));
            const Id missing = k == 3 ? sel(lor(ieq(nfmt, cu(4)), ieq(nfmt, cu(5))), cu(1), as_u(float_const(1.0f))) : cu(0);
            out.push_back(sel(m.emit(spv::OpULessThan, t_bool, {cu(static_cast<std::uint32_t>(k)), cnt}), loaded, missing));
        }
        return out;
    }
    // translate.cpp apply_dst_sel_rt: a V#'s DST_SEL over loaded components -
    // 4..7 a component, 1 one (an integer 1 for an integer format), else 0.
    // The reference writes the components as loaded and applies the select
    // only where it is not the identity: the same values.
    std::vector<Id> apply_dst_sel(const std::vector<Id>& comps, Id w3, Id integer_fmt) {
        const auto sel = [&](Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_u32, {c, a, b}); };
        const Id one = sel(integer_fmt, cu(1), as_u(float_const(1.0f)));
        std::vector<Id> out;
        for (std::uint32_t k = 0; k < 4; ++k) {
            const Id s3 = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(3 * k), cu(3)});
            out.push_back(sel(ieq(s3, cu(4)), comps[0],
                              sel(ieq(s3, cu(5)), comps[1], sel(ieq(s3, cu(6)), comps[2], sel(ieq(s3, cu(7)), comps[3], sel(ieq(s3, cu(1)), one, cu(0)))))));
        }
        return out;
    }
    // translate.cpp store_dst_sel_rt: what a format store writes through a
    // descriptor's DST_SEL - memory component c takes the first data
    // component whose select names it, 0 where none does. The reference
    // selects only where the select is not the identity: the same values.
    std::vector<Id> store_dst_sel(const std::vector<Id>& data, Id w3) {
        const int n = static_cast<int>(data.size());
        std::vector<Id> out;
        for (int c = 0; c < n; ++c) {
            Id v = cu(0);
            for (int k = n - 1; k >= 0; --k) {
                const Id s3 = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(3u * static_cast<std::uint32_t>(k)), cu(3)});
                v = m.emit(spv::OpSelect, t_u32, {ieq(s3, cu(4u + static_cast<std::uint32_t>(c))), data[static_cast<std::size_t>(k)], v});
            }
            out.push_back(v);
        }
        return out;
    }
    // translate.cpp format_store_dynamic: data in the V#'s data format, the
    // 32-, 16- (half or unorm16) and 8-bit (unorm8 or raw) families. Inside
    // the caller's EXEC guard.
    void format_store_dynamic(Id va, Id dfmt, Id nfmt, const std::vector<Id>& vals) {
        const int n = static_cast<int>(vals.size());
        const auto f_at = [&](int k) { return k < n ? as_f(word(vals[static_cast<std::size_t>(k)])) : cf(0.0f); };
        const auto u_at = [&](int k) { return k < n ? vals[static_cast<std::size_t>(k)] : cu(0); };
        const auto sel = [&](Id c, Id a, Id b) { return m.emit(spv::OpSelect, t_u32, {c, a, b}); };
        const auto lor = [&](Id a, Id b) { return m.emit(spv::OpLogicalOr, t_bool, {a, b}); };
        const auto f2u = [&](Id x) { return m.emit(spv::OpConvertFToU, t_u32, {x}); };
        const Id is_float = ieq(nfmt, cu(7)), is_unorm = ieq(nfmt, cu(0));
        const Id is32 = lor(lor(ieq(dfmt, cu(4)), ieq(dfmt, cu(11))), lor(ieq(dfmt, cu(13)), ieq(dfmt, cu(14))));
        if_then(is32, [&] {
            const Id cnt = sel(ieq(dfmt, cu(4)), cu(1), sel(ieq(dfmt, cu(11)), cu(2), sel(ieq(dfmt, cu(13)), cu(3), cu(4))));
            for (int k = 0; k < n; ++k) {
                if_then(m.emit(spv::OpULessThan, t_bool, {cu(static_cast<std::uint32_t>(k)), cnt}),
                        [&] { store_u32_at(addr_addc(va, static_cast<std::uint32_t>(k) * 4), vals[static_cast<std::size_t>(k)]); });
            }
        });
        const Id is16 = lor(lor(ieq(dfmt, cu(2)), ieq(dfmt, cu(5))), ieq(dfmt, cu(12)));
        if_then(is16, [&] {
            const auto pack16 = [&](int k) {
                const Id half = ext(spv::GlslPackHalf2x16, {m.emit(spv::OpCompositeConstruct, t_v2f, {f_at(k), f_at(k + 1)})}, t_u32);
                const auto un = [&](Id v) { return f2u(fadd(fmul(ext(spv::GlslFClamp, {v, cf(0.0f), cf(1.0f)}), cf(65535.0f)), cf(0.5f))); };
                const Id unorm = ibin(spv::OpBitwiseOr, un(f_at(k)), shift(spv::OpShiftLeftLogical, un(f_at(k + 1)), cu(16)));
                const Id raw = ibin(spv::OpBitwiseOr, ibin(spv::OpBitwiseAnd, u_at(k), cu(0xffff)), shift(spv::OpShiftLeftLogical, u_at(k + 1), cu(16)));
                return sel(is_float, half, sel(is_unorm, unorm, raw));
            };
            const Id m0 = sel(ieq(dfmt, cu(2)), cu(0xffffu), cu(0xffffffffu));
            const Id m1 = sel(ieq(dfmt, cu(12)), cu(0xffffffffu), cu(0));
            store_masked_any(va, pack16(0), pack16(2), m0, m1);
        });
        const Id is8 = lor(lor(ieq(dfmt, cu(1)), ieq(dfmt, cu(3))), ieq(dfmt, cu(10)));
        if_then(is8, [&] {
            Id packed = cu(0);
            for (int k = 0; k < 4; ++k) {
                const Id un = f2u(fadd(fmul(ext(spv::GlslFClamp, {f_at(k), cf(0.0f), cf(1.0f)}), cf(255.0f)), cf(0.5f)));
                const Id byte = sel(is_unorm, un, ibin(spv::OpBitwiseAnd, u_at(k), cu(0xff)));
                packed = ibin(spv::OpBitwiseOr, packed, shift(spv::OpShiftLeftLogical, byte, cu(8u * static_cast<std::uint32_t>(k))));
            }
            const Id m0 = sel(ieq(dfmt, cu(1)), cu(0xffu), sel(ieq(dfmt, cu(3)), cu(0xffffu), cu(0xffffffffu)));
            store_masked_any(va, packed, cu(0), m0, cu(0));
        });
    }
    // MUBUF in a compute shader (translate.cpp mubuf, every access through the
    // page table): format loads and stores through the V#'s word 3, dword,
    // byte and short loads and stores, buffer_atomic_add.
    void guest_mubuf(const Inst& in) {
        const std::uint32_t op = in.op;
        const char* name = mnemonic(in) ? mnemonic(in) : "unknown";
        if (op == 112 || op == 113) return;  // cache writeback / invalidate
        const bool load_format = op <= 3, store_format = op >= 4 && op <= 7, load_small = op >= 8 && op <= 11, load_dw = op >= 12 && op <= 15;
        const bool store_small = op == 24 || op == 26, store_dw = op >= 28 && op <= 31, atomic_add = op == 50;
        if (!load_format && !store_format && !load_small && !load_dw && !store_small && !store_dw && !atomic_add) {
            reject(std::string(name) + " is not lifted");
            return;
        }
        const GuestAddr ga = guest_buffer_address(in);
        if (!res.rejections.empty()) return;
        if (load_format) {
            const Id w3 = as_u(read(static_cast<std::uint16_t>(in.srsrc + 3), in));
            const Id dfmt = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(15), cu(4)}), nfmt = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(12), cu(3)});
            const std::vector<Id> comps = format_load_dynamic(ga.addr, dfmt, nfmt, 4);
            const Id is_int = m.emit(spv::OpLogicalOr, t_bool, {ieq(nfmt, cu(4)), ieq(nfmt, cu(5))});
            const std::vector<Id> swz = apply_dst_sel(comps, w3, is_int);
            for (std::uint32_t k = 0; k <= op; ++k) write_v(in.vdata + static_cast<int>(k), word(swz[k]));
            ++guest_loads;
            return;
        }
        if (load_small) {
            const Id aligned = m.emit(spv::OpBitwiseAnd, t_u64, {ga.addr, m.const_u64(~3ull)});
            const Id sh = shift(spv::OpShiftLeftLogical, m.emit(spv::OpUConvert, t_u32, {m.emit(spv::OpBitwiseAnd, t_u64, {ga.addr, m.const_u64(3)})}), cu(3));
            const Id w = shift(spv::OpShiftRightLogical, load_u32_at(aligned), sh);
            Id v;
            if (op == 8) v = ibin(spv::OpBitwiseAnd, w, cu(0xff));
            else if (op == 9) v = from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(w), cu(0), cu(8)}));
            else if (op == 10) v = ibin(spv::OpBitwiseAnd, w, cu(0xffff));
            else v = from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(w), cu(0), cu(16)}));
            write_v(in.vdata, word(v));
            ++guest_loads;
            return;
        }
        if (load_dw) {
            const int n = op == 12 ? 1 : op == 13 ? 2 : op == 14 ? 4 : 3;
            const std::vector<Id> w = load_u32_run(ga.addr, n);
            for (int k = 0; k < n; ++k) write_v(in.vdata + k, word(w[static_cast<std::size_t>(k)]));
            return;
        }
        const Val e = exec_lane();
        if (store_format) {
            const Id w3 = as_u(read(static_cast<std::uint16_t>(in.srsrc + 3), in));
            const Id dfmt = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(15), cu(4)}), nfmt = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(12), cu(3)});
            std::vector<Id> vals;
            for (std::uint32_t k = 0; k < op - 3; ++k) vals.push_back(as_u(value(256 + in.vdata + static_cast<int>(k))));
            vals = store_dst_sel(vals, w3);
            where_exec(e, [&] { format_store_dynamic(ga.addr, dfmt, nfmt, vals); });
        } else if (store_small) {
            const Id mask = cu(op == 24 ? 0xffu : 0xffffu);
            const Id v = ibin(spv::OpBitwiseAnd, as_u(value(256 + in.vdata)), mask);
            where_exec(e, [&] { store_masked_any(ga.addr, v, cu(0), mask, cu(0)); });
        } else if (store_dw) {
            const int n = op == 28 ? 1 : op == 29 ? 2 : op == 30 ? 4 : 3;
            std::vector<Id> vals;
            for (int k = 0; k < n; ++k) vals.push_back(as_u(value(256 + in.vdata + k)));
            where_exec(e, [&] {
                for (int k = 0; k < n; ++k) store_u32_at(addr_addc(ga.addr, static_cast<std::uint32_t>(k) * 4), vals[static_cast<std::size_t>(k)]);
            });
        } else {  // buffer_atomic_add: the value before the add where GLC is set
            const Id v = as_u(value(256 + in.vdata));
            const Id r = where_exec_value(e, [&] {
                return m.emit(spv::OpAtomicIAdd, t_u32, {guest_ptr(ga.addr), cu(spv::ScopeDevice), cu(spv::MsNone), v});
            });
            if (in.glc) {
                inst_uniform = false;
                write_v(in.vdata, word(r));
            }
            ++guest_atomics;
            note += " atomic";
            return;
        }
        ++guest_stores;
        note += " stored";
    }
    // MTBUF in a compute shader (translate.cpp mtbuf without a binding):
    // typed loads by the instruction's format with the identity select (or
    // the V#'s, BBHOST_TBUFFER_DST_SEL=1), typed stores of 32-bit components,
    // halves and unorm8x4 - the forms the reference stores.
    void guest_mtbuf(const Inst& in) {
        const int n = (in.op & 3) + 1;
        int count = 0, bits[4] = {};
        if (!format_layout(in.dfmt, count, bits)) {
            reject(std::string(mnemonic(in) ? mnemonic(in) : "a typed buffer access") + " with data format " + std::to_string(in.dfmt));
            return;
        }
        const GuestAddr ga = guest_buffer_address(in);
        if (!res.rejections.empty()) return;
        if (in.op < 4) {
            int total = 0;
            for (int k = 0; k < count; ++k) total += bits[k];
            const std::vector<Id> comps = format_decode(load_u32_any(ga.addr, (total + 31) / 32), in.dfmt, in.nfmt);
            if (tbuffer_dst_sel()) {
                const Id w3 = as_u(read(static_cast<std::uint16_t>(in.srsrc + 3), in));
                const std::vector<Id> swz = apply_dst_sel(comps, w3, m.const_bool(in.nfmt == 4 || in.nfmt == 5));
                for (int k = 0; k < n; ++k) write_v(in.vdata + k, word(swz[static_cast<std::size_t>(k)]));
            } else {
                for (int k = 0; k < n; ++k) write_v(in.vdata + k, word(k < count ? comps[static_cast<std::size_t>(k)] : cu(0)));
            }
            ++guest_loads;
            return;
        }
        std::vector<Id> vals;
        for (int k = 0; k < n; ++k) vals.push_back(as_u(value(256 + in.vdata + k)));
        if (!(bits[0] == 32 || (bits[0] == 16 && in.nfmt == 7) || (bits[0] == 8 && count == 4 && in.nfmt == 0))) {
            reject("tbuffer store of format " + std::to_string(in.dfmt) + "/" + std::to_string(in.nfmt) + ", which the reference does not store");
            return;
        }
        const Val e = exec_lane();
        where_exec(e, [&] {
            if (bits[0] == 32) {
                for (int k = 0; k < n && k < count; ++k) store_u32_at(addr_addc(ga.addr, static_cast<std::uint32_t>(k) * 4), vals[static_cast<std::size_t>(k)]);
            } else if (bits[0] == 16) {
                for (int k = 0; k < n && k < count; k += 2) {
                    const Id a = as_f(word(vals[static_cast<std::size_t>(k)]));
                    const Id b = k + 1 < n ? as_f(word(vals[static_cast<std::size_t>(k) + 1])) : cf(0.0f);
                    const Id packed = ext(spv::GlslPackHalf2x16, {m.emit(spv::OpCompositeConstruct, t_v2f, {a, b})}, t_u32);
                    store_masked_any(addr_addc(ga.addr, static_cast<std::uint32_t>(k / 2) * 4), packed, cu(0), cu(k + 1 < count ? 0xffffffffu : 0xffffu), cu(0));
                }
            } else {
                Id packed = cu(0);
                for (int k = 0; k < 4; ++k) {
                    const Id v = k < n ? as_f(word(vals[static_cast<std::size_t>(k)])) : cf(0.0f);
                    const Id b = m.emit(spv::OpConvertFToU, t_u32, {fadd(fmul(ext(spv::GlslFClamp, {v, cf(0.0f), cf(1.0f)}), cf(255.0f)), cf(0.5f))});
                    packed = ibin(spv::OpBitwiseOr, packed, shift(spv::OpShiftLeftLogical, b, cu(8u * static_cast<std::uint32_t>(k))));
                }
                store_masked_any(ga.addr, packed, cu(0), cu(0xffffffffu), cu(0));
            }
        });
        ++guest_stores;
        note += " stored";
    }
    // image_store(_mip) in a compute shader (translate.cpp mimg): the
    // texel at the integer coordinates, its components the VGPRs DMASK names
    // (0 for the others) put through the inverse of the T#'s select, written
    // where the invocation's EXEC bit is set. The reference writes level 0 of
    // the view the host binds for the T# for image_store_mip too.
    void guest_image_store(const Inst& in) {
        const auto ii = ref.image_at.find(in.offset);
        if (ii == ref.image_at.end() || ii->second >= image_vars.size() || !ref.images[ii->second].storage) {
            reject(std::string("the reference has no storage image binding for this ") + (mnemonic(in) ? mnemonic(in) : "image store"));
            return;
        }
        if (in.tfe || in.lwe) {
            reject(std::string(mnemonic(in) ? mnemonic(in) : "an image store") + " with TFE or LWE");
            return;
        }
        const ImageBinding& b = ref.images[ii->second];
        const int ncoord = coord_count(b.dim, b.arrayed);
        std::vector<Id> coords;
        for (int k = 0; k < ncoord; ++k) coords.push_back(to_i(as_u(value(256 + in.vaddr + k))));
        const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2i : ncoord == 3 ? t_v3i : m.type_vector(t_i32, 4), coords);
        std::vector<Id> raws;
        for (int k = 0, next = 0; k < 4; ++k) raws.push_back((in.dmask >> k) & 1 ? as_u(value(256 + in.vdata + next++)) : cu(0));
        raws = store_dst_sel(raws, as_u(read(static_cast<std::uint16_t>(in.srsrc + 3), in)));
        std::vector<Id> vals;
        for (const Id r : raws) vals.push_back(b.kind == 1 ? r : b.kind == 2 ? to_i(r) : as_f(word(r)));
        const Id texel = m.emit(spv::OpCompositeConstruct, b.kind == 1 ? t_v4u : b.kind == 2 ? m.type_vector(t_i32, 4) : t_v4f, vals);
        if (!res.rejections.empty()) return;
        where_exec(exec_lane(), [&] { m.emit_void(spv::OpImageWrite, {image_only(ii->second), coord, texel}); });
        ++image_stores;
        note += " stored";
    }

    void smrd(const Inst& in) {
        const int count = in.op < 8 ? (1 << in.op) : (in.op < 13 ? (1 << (in.op - 8)) : 0);
        if (compute()) {
            if (in.op == 30) {  // s_memtime: 0, as the reference
                write_scalar(in.sdst, word_const(0));
                write_scalar(static_cast<std::uint16_t>(in.sdst + 1), word_const(0));
                return;
            }
            if (in.op == 31) return;  // s_dcache_inv
            if (!count || count > 16) {
                reject(std::string(mnemonic(in) ? mnemonic(in) : "a scalar memory instruction") + " is not lifted");
                return;
            }
            // A load the lean reference bound reads its storage buffer (below);
            // one it walks, the page table, as it does.
            if (!opt.cb_ssbo || !ref.buffer_at.count(in.offset)) {
                guest_scalar_load(in, count);
                return;
            }
            ++cs_bound_loads;
        }
        const auto site = ref.buffer_at.find(in.offset);
        if (count && site == ref.buffer_at.end()) return descriptor_load(in, count);
        if (!count || site == ref.buffer_at.end() || site->second >= buffer_vars.size()) {
            reject(std::string(mnemonic(in)) + " is not read through a bound storage buffer in the reference");
            return;
        }
        const std::uint32_t index = site->second;
        if (!in.imm_flag && !in.has_literal) {
            if (!ref.buffers[index].ranged || in.op < 8) {
                reject("scalar load at a register offset from a buffer the reference does not bind whole");
                return;
            }
            ranged_load(in, count, index);
            return;
        }
        const std::uint32_t off_dw = in.imm_flag ? static_cast<std::uint32_t>(in.imm) : in.literal;
        // TessLift: the tessellation constants, as the words the host gives
        // the reference there (12 of them; past those the binding holds none).
        if (tess && !tess->constants_path.empty() && ref.buffers[index].path.str() == tess->constants_path) {
            Pending p{in.sdst, {}};
            for (int k = 0; k < count; ++k) {
                const std::uint32_t dw = off_dw + static_cast<std::uint32_t>(k);
                if (dw >= 12) {
                    reject("a load past the tessellation constants' 12 words");
                    return;
                }
                const Val v = word_const(tess->constants[dw]);
                write_scalar(static_cast<std::uint16_t>(in.sdst + k), v);
                p.values.push_back(v);
            }
            pending.push_back(std::move(p));
            ++constant_loads;
            return;
        }
        const Id bias = cb_bias(index);
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
    std::size_t constant_loads = 0;  // TessLift::constants folded

    // s_buffer_load_dword* at a register offset from a buffer the reference
    // binds whole (BufferBinding::ranged), as translate.cpp load_buffer_ranged
    // reads it: dword (offset >> 2) + k of the V#'s memory, 0 at or past its
    // range in dwords (StageParams::cb_stride), as GCN's range check returns -
    // an index past it reads dword 0 of the binding instead, in bounds.
    std::size_t ranged_loads = 0;
    void ranged_load(const Inst& in, int count, std::uint32_t index) {
        const Id first = shift(spv::OpShiftRightLogical, as_u(read(static_cast<std::uint16_t>(in.imm), in)), cu(2));
        const Id bias = cb_bias(index);
        const Id range_dw = params_load(stride_loads, 5, index);
        Pending p{in.sdst, {}};
        for (int k = 0; k < count; ++k) {
            const Id at = ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k)));
            const Id in_range = m.emit(spv::OpULessThan, t_bool, {at, range_dw});
            const Id slot = ibin(spv::OpIAdd, bias, m.emit(spv::OpSelect, t_u32, {in_range, at, cu(0)}));
            const Id word_v = m.load(t_u32, m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), slot}));
            const Val v = word(m.emit(spv::OpSelect, t_u32, {in_range, word_v, cu(0)}));
            write_scalar(static_cast<std::uint16_t>(in.sdst + k), v);
            p.values.push_back(v);
        }
        pending.push_back(std::move(p));
        ++buffer_loads;
        ++ranged_loads;
    }

    // A typed load's stride (StageParams::cb_stride), as translate.cpp
    // indexed_buffer_load reads it: also specialization constant
    // kCbStrideSpecId + index (0, the default, reads the params), which an
    // optimized relink gives the draw's (render.cpp queue_library_relink).
    std::map<std::uint32_t, Id> stride_values;
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
    // buffers[index]'s dword bias inside its binding, as translate.cpp cb_bias reads it.
    Id cb_bias(std::uint32_t index) { return opt.cb_bias_zero ? cu(0) : params_load(bias_loads, 4, index); }
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
    // by its static format, with the identity select (gcn::tbuffer_dst_sel).
    void mtbuf(const Inst& in) {
        if (compute()) {
            // What the lean reference bound reads or writes the binding; the rest walks.
            const auto site = ref.buffer_at.find(in.offset);
            if (!opt.cb_ssbo || site == ref.buffer_at.end()) return guest_mtbuf(in);
            if (bound_element_shape(in)) return bound_element_cs(in, site->second);
            if (in.op >= 4) {
                reject(std::string(mnemonic(in)) + ": a store the reference binds in a shape the lift does not take");
                return;
            }
            ++cs_bound_loads;
        }
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
        const Id bias = cb_bias(index);
        int total = 0;
        for (int k = 0; k < count; ++k) total += bits[k];
        std::vector<Id> words;
        for (int k = 0; k < (total + 31) / 32; ++k) {
            const Id at = ibin(spv::OpIAdd, bias, ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k))));
            words.push_back(m.load(t_u32, m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), at})));
        }
        typed_load_result(in, words, count, bits);
        ++buffer_loads;
    }
    // A typed load's VGPRs from the dwords of its element, as the translator
    // decodes them (format_decode) and writes them: the identity select, or
    // the V#'s with BBHOST_TBUFFER_DST_SEL=1.
    void typed_load_result(const Inst& in, const std::vector<Id>& words, int count, const int bits[4]) {
        const bool integer = in.nfmt == 4 || in.nfmt == 5;
        const Id one = integer ? cu(1) : as_u(float_const(1.0f));
        std::vector<Id> comps(4, cu(0));
        for (int k = 0, bit = 0; k < count; bit += bits[k], ++k) {
            const Id w = words[static_cast<std::size_t>(bit / 32)];
            const Id raw = bits[k] == 32 ? w : m.emit(spv::OpBitFieldUExtract, t_u32, {w, cu(static_cast<std::uint32_t>(bit % 32)), cu(static_cast<std::uint32_t>(bits[k]))});
            comps[static_cast<std::size_t>(k)] = convert_component(raw, bits[k], in.nfmt);
        }
        std::vector<Val> out;
        if (tbuffer_dst_sel()) {
            // BBHOST_TBUFFER_DST_SEL=1, as the translator then reads it: the
            // V#'s select over the components with 0, 0, 0, 1 past them.
            if (count < 4) comps[3] = one;
            const Id w3 = as_u(read(static_cast<std::uint16_t>(in.srsrc + 3), in));
            for (int k = 0; k < (in.op & 3) + 1; ++k) {  // DST_SEL: 4-7 a component, 1 one, else zero
                const Id s3 = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(3u * static_cast<std::uint32_t>(k)), cu(3)});
                const auto pick = [&](std::uint32_t code, Id then_v, Id else_v) { return m.emit(spv::OpSelect, t_u32, {ieq(s3, cu(code)), then_v, else_v}); };
                out.push_back(word(pick(4, comps[0], pick(5, comps[1], pick(6, comps[2], pick(7, comps[3], pick(1, one, cu(0))))))));
            }
        } else {
            // The identity select: the format's components in order, 0 past them.
            for (int k = 0; k < (in.op & 3) + 1; ++k) out.push_back(word(k < count ? comps[static_cast<std::size_t>(k)] : cu(0)));
        }
        for (std::size_t k = 0; k < out.size(); ++k) write_v(in.vdata + static_cast<int>(k), out[k]);
    }

    // translate.cpp bound_compute_access, which a lean reference binds: an
    // element at a VGPR index or offset through a V#'s storage buffer - typed
    // loads at a VGPR offset, typed stores of 32-bit components,
    // buffer_load_format_* with the V#'s format, buffer_atomic_add. The
    // element's first dword (offset12 + soffset) / 4, plus index * stride / 4
    // with idxen, plus the VGPR offset / 4 with offen; an index at or past
    // num_records - or, with no index, a dword at or past the range in
    // cb_stride - reads 0 and is not written; stores and the atomic only where
    // the invocation's EXEC bit is set. The same expressions as the reference,
    // in its order.
    static bool bound_element_shape(const Inst& in) {
        const bool mt = in.enc == Enc::MTBUF;
        const int kind = mt ? (in.op < 4 ? 1 : in.op < 8 ? 2 : 0) : (in.op <= 3 ? 3 : in.op == 50 ? 4 : 0);
        return kind && (in.idxen || in.offen) && !(kind == 1 && !in.offen);
    }
    std::size_t cs_bound_elements = 0;
    void bound_element_cs(const Inst& in, std::uint32_t index) {
        const std::string name = mnemonic(in) ? mnemonic(in) : "a buffer access";
        const bool mt = in.enc == Enc::MTBUF;
        const bool typed_load = mt && in.op < 4, typed_store = mt && in.op >= 4, format_load = !mt && in.op <= 3, atomic_add = !mt && in.op == 50;
        if (index >= buffer_vars.size()) {
            reject(name + " is not read through a storage buffer the reference binds");
            return;
        }
        const BufferBinding& b = ref.buffers[index];
        const std::int32_t constant = in.soffset <= 192 ? static_cast<std::int32_t>(in.soffset) - 128 : 192 - static_cast<std::int32_t>(in.soffset);
        const std::int32_t off = static_cast<std::int32_t>(in.offset12) + constant;
        int count = 0, bits[4] = {};
        if ((in.idxen ? !b.indexed : !b.ranged) || ((typed_store || atomic_add) && !b.written) || in.addr64 || in.lds || in.tfe || in.soffset < 128 ||
            in.soffset > 208 || off < 0 || off % 4 != 0 || (mt && (!format_layout(in.dfmt, count, bits) || bits[0] != 32))) {
            reject(name + ": the reference's bound element has another shape");
            return;
        }
        int va = in.vaddr;
        Id first = cu(static_cast<std::uint32_t>(off / 4));
        Id index_ok = 0, vindex = 0, voffset = 0, records = 0;
        bool index_zero = false;
        if (in.idxen) {
            const Val vi = value(256 + va++);
            vindex = as_u(vi);
            // An index the lift knows is 0 adds nothing: no stride is read
            // (DXC would drop the stride's specialization constant, which
            // the HLSL path's interface check then misses).
            index_zero = vi.constant && vi.bits == 0;
            if (!index_zero) {
                const Id stride_dw = shift(spv::OpShiftRightLogical, stride_value(index), cu(2));
                first = ibin(spv::OpIAdd, first, ibin(spv::OpIMul, vindex, stride_dw));
            }
            records = as_u(read(static_cast<std::uint16_t>(in.srsrc + 2), in));
            index_ok = m.emit(spv::OpULessThan, t_bool, {vindex, records});
        }
        if (in.offen) {
            voffset = as_u(value(256 + va));
            first = ibin(spv::OpIAdd, first, shift(spv::OpShiftRightLogical, voffset, cu(2)));
        }
        if (!res.rejections.empty()) return;
        const Id bias = cb_bias(index);
        const Id range_dw = in.idxen ? 0 : params_load(stride_loads, 5, index);
        const auto ok_at = [&](Id at) { return in.idxen ? index_ok : m.emit(spv::OpULessThan, t_bool, {at, range_dw}); };
        const auto word_ptr = [&](Id at) { return m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), ibin(spv::OpIAdd, bias, at)}); };
        const auto load_at = [&](Id at) {
            const Id ok = ok_at(at);
            const Id w = m.load(t_u32, word_ptr(m.emit(spv::OpSelect, t_u32, {ok, at, cu(0)})));
            return m.emit(spv::OpSelect, t_u32, {ok, w, cu(0)});
        };
        // `body` where `cond` holds, as the reference's if_then.
        const auto where = [&](Id cond, const std::function<void()>& body) {
            const Id l_then = m.fresh(), l_merge = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
            m.emit_void(spv::OpBranchConditional, {cond, l_then, l_merge});
            cur_label = m.label(l_then);
            body();
            m.emit_void(spv::OpBranch, {l_merge});
            cur_label = m.label(l_merge);
        };
        ++cs_bound_elements;
        if (typed_load) {
            std::vector<Id> words;
            for (int k = 0; k < count; ++k) words.push_back(load_at(ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k)))));
            typed_load_result(in, words, count, bits);
            return;
        }
        if (format_load) {
            const Id w3 = as_u(read(static_cast<std::uint16_t>(in.srsrc + 3), in));
            const Id dfmt = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(15), cu(4)}), nfmt = m.emit(spv::OpBitFieldUExtract, t_u32, {w3, cu(12), cu(3)});
            const auto sel = [&](Id c, Id a, Id x) { return m.emit(spv::OpSelect, t_u32, {c, a, x}); };
            std::vector<Id> words;
            if (in.idxen) {
                // translate.cpp: at a byte offset, the element's dwords funnel-shifted
                // out of the binding's (any stride; past the records 0).
                const Id stride = index_zero ? cu(0) : stride_value(index);
                Id at_byte = ibin(spv::OpIAdd, cu(static_cast<std::uint32_t>(off)), ibin(spv::OpIMul, vindex, stride));
                if (in.offen) at_byte = ibin(spv::OpIAdd, at_byte, voffset);
                const Id d0 = shift(spv::OpShiftRightLogical, at_byte, cu(2));
                const Id sh = shift(spv::OpShiftLeftLogical, ibin(spv::OpBitwiseAnd, at_byte, cu(3)), cu(3));
                const Id end_dw = shift(spv::OpShiftRightLogical, ibin(spv::OpIAdd, ibin(spv::OpIMul, records, index_zero ? stride_value(index) : stride), cu(3)), cu(2));
                std::vector<Id> raw;
                for (std::uint32_t k = 0; k < 5; ++k) {
                    const Id at = ibin(spv::OpIAdd, d0, cu(k));
                    const Id ok = m.emit(spv::OpLogicalAnd, t_bool, {index_ok, m.emit(spv::OpULessThan, t_bool, {at, end_dw})});
                    const Id w = m.load(t_u32, word_ptr(sel(ok, at, cu(0))));
                    raw.push_back(sel(ok, w, cu(0)));
                }
                const Id aligned = ieq(sh, cu(0));
                const Id back = ibin(spv::OpISub, cu(32), sh);
                for (std::uint32_t k = 0; k < 4; ++k) {
                    words.push_back(sel(aligned, raw[k],
                                        ibin(spv::OpBitwiseOr, shift(spv::OpShiftRightLogical, raw[k], sh), shift(spv::OpShiftLeftLogical, raw[k + 1], back))));
                }
            } else {
                const Id last = sel(m.emit(spv::OpLogicalOr, t_bool, {ieq(dfmt, cu(11)), ieq(dfmt, cu(12))}), cu(1),
                                    sel(ieq(dfmt, cu(13)), cu(2), sel(ieq(dfmt, cu(14)), cu(3), cu(0))));
                for (std::uint32_t k = 0; k < 4; ++k) {
                    const Id kk = k ? ext(spv::GlslUMin, {cu(k), last}, t_u32) : cu(0);
                    words.push_back(load_at(ibin(spv::OpIAdd, first, kk)));
                }
            }
            const std::vector<Id> comps = format_decode_dynamic(words, dfmt, nfmt, 4);
            const Id is_int = m.emit(spv::OpLogicalOr, t_bool, {ieq(nfmt, cu(4)), ieq(nfmt, cu(5))});
            const std::vector<Id> swz = apply_dst_sel(comps, w3, is_int);
            for (std::uint32_t k = 0; k <= in.op; ++k) write_v(in.vdata + static_cast<int>(k), word(swz[k]));
            return;
        }
        const Val e = exec_lane();
        if (typed_store) {
            const int n = (in.op & 3) + 1;
            std::vector<Id> vals;
            for (int k = 0; k < n; ++k) vals.push_back(as_u(value(256 + in.vdata + k)));
            where_exec(e, [&] {
                for (int k = 0; k < n && k < count; ++k) {
                    const Id at = ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k)));
                    where(ok_at(at), [&] { m.store(word_ptr(at), vals[static_cast<std::size_t>(k)]); });
                }
            });
            note += " stored";
            return;
        }
        // buffer_atomic_add: the value before the add where GLC is set, 0 where it did not run.
        const Id v = as_u(value(256 + in.vdata));
        const Id r = where_exec_value(e, [&] {
            const Id ok = ok_at(first);
            const Id from = cur_label;
            const Id l_then = m.fresh(), l_merge = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
            m.emit_void(spv::OpBranchConditional, {ok, l_then, l_merge});
            cur_label = m.label(l_then);
            const Id added = m.emit(spv::OpAtomicIAdd, t_u32, {word_ptr(first), cu(spv::ScopeDevice), cu(spv::MsNone), v});
            const Id end = cur_label;
            m.emit_void(spv::OpBranch, {l_merge});
            cur_label = m.label(l_merge);
            return m.emit(spv::OpPhi, t_u32, {added, end, cu(0), from});
        });
        if (in.glc) {
            inst_uniform = false;
            write_v(in.vdata, word(r));
        }
        note += " atomic";
    }

    // buffer_store_dword* a compute shader's lean reference makes through a
    // binding (translate.cpp store_bound): dword (soffset + offset12) >> 2 + k
    // of the V#'s binding, each written where it is inside the V#'s range in
    // dwords (StageParams::cb_stride) and the invocation's EXEC bit is set, as
    // the reference's exec_guard and range check have it. A dispatch has no
    // helper invocations, and a store in a loop runs in each iteration the
    // invocation runs, as the guest stores do.
    std::size_t cs_bound_loads = 0, cs_bound_stores = 0;
    void bound_store_cs(const Inst& in, std::uint32_t index) {
        if (index >= buffer_vars.size() || !ref.buffers[index].written) {
            reject(std::string(mnemonic(in)) + " is not a store the reference makes through a binding");
            return;
        }
        if (in.idxen || in.offen || in.addr64) {
            reject(std::string(mnemonic(in)) + ": the reference's bound store has another shape");
            return;
        }
        const int n = in.op == 28 ? 1 : in.op == 29 ? 2 : in.op == 30 ? 4 : 3;
        const Id first = shift(spv::OpShiftRightLogical, ibin(spv::OpIAdd, cu(in.offset12), as_u(read(in.soffset, in))), cu(2));
        const Id bias = cb_bias(index);
        const Id range_dw = params_load(stride_loads, 5, index);
        std::vector<Id> vals;
        for (int k = 0; k < n; ++k) vals.push_back(as_u(value(256 + in.vdata + k)));
        if (!res.rejections.empty()) return;
        where_exec(exec_lane(), [&] {
            for (int k = 0; k < n; ++k) {
                const Id at = ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k)));
                const Id in_range = m.emit(spv::OpULessThan, t_bool, {at, range_dw});
                const Id l_store = m.fresh(), l_merge = m.fresh();
                m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
                m.emit_void(spv::OpBranchConditional, {in_range, l_store, l_merge});
                m.label(l_store);
                m.store(m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), ibin(spv::OpIAdd, bias, at)}), vals[static_cast<std::size_t>(k)]);
                m.emit_void(spv::OpBranch, {l_merge});
                cur_label = m.label(l_merge);
            }
        });
        note += " stored";
        ++cs_bound_stores;
    }

    // buffer_store_dword* the reference makes through a binding (translate.cpp
    // store_bound): dword (soffset + offset12) >> 2 + k of the V#'s binding,
    // written where it is inside the V#'s range in dwords (StageParams::
    // cb_stride) and this invocation's EXEC bit is set - as GCN performs a
    // store only in range and in the lanes EXEC holds. A vertex shader only,
    // whose invocations have no helpers, and outside loops, so each store
    // runs once per invocation where the translation's runs.
    std::size_t buffer_stores = 0;
    void mubuf(const Inst& in) {
        if (compute()) {
            const auto site = ref.buffer_at.find(in.offset);
            if (!opt.cb_ssbo || site == ref.buffer_at.end()) return guest_mubuf(in);
            if (bound_element_shape(in)) return bound_element_cs(in, site->second);
            if (in.op >= 28 && in.op <= 31) return bound_store_cs(in, site->second);
            reject(std::string(mnemonic(in)) + ": the reference binds it in a shape the lift does not take");
            return;
        }
        if (in.op < 28 || in.op > 31) {
            reject(std::string(mnemonic(in)) + " is not lifted");
            return;
        }
        if (opt.stage != Stage::Vertex) {
            reject(std::string(mnemonic(in)) + " outside a vertex shader is not lifted");
            return;
        }
        const auto site = ref.buffer_at.find(in.offset);
        if (site == ref.buffer_at.end() || site->second >= buffer_vars.size() || !ref.buffers[site->second].written) {
            reject(std::string(mnemonic(in)) + " is not a store the reference makes through a binding");
            return;
        }
        if (in.idxen || in.offen || in.addr64) {
            reject(std::string(mnemonic(in)) + ": the reference's bound store has another shape");
            return;
        }
        if (!open_loops.empty()) {
            reject(std::string(mnemonic(in)) + " inside a loop is not lifted");
            return;
        }
        const Val e = exec_lane();
        if (e.constant && !e.set) return;  // no invocation stores
        const int n = in.op == 28 ? 1 : in.op == 29 ? 2 : in.op == 30 ? 4 : 3;
        const std::uint32_t index = site->second;
        const Id first = shift(spv::OpShiftRightLogical, ibin(spv::OpIAdd, cu(in.offset12), as_u(read(in.soffset, in))), cu(2));
        const Id bias = cb_bias(index);
        const Id range_dw = params_load(stride_loads, 5, index);
        for (int k = 0; k < n; ++k) {
            const Id v = as_u(value(256 + in.vdata + k));
            const Id at = ibin(spv::OpIAdd, first, cu(static_cast<std::uint32_t>(k)));
            Id cond = m.emit(spv::OpULessThan, t_bool, {at, range_dw});
            if (!e.constant) cond = m.emit(spv::OpLogicalAnd, t_bool, {cond, e.id});
            const Id l_store = m.fresh(), l_merge = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
            m.emit_void(spv::OpBranchConditional, {cond, l_store, l_merge});
            m.label(l_store);
            m.store(m.access_chain(p_ssbo_u32, buffer_vars[index], {cu(0), ibin(spv::OpIAdd, bias, at)}), v);
            m.emit_void(spv::OpBranch, {l_merge});
            cur_label = m.label(l_merge);
        }
        note += " stored";
        ++buffer_stores;
    }

    // TranslateOptions::vertex_formats_from_params, as translate.cpp
    // load_vertex_element_from_params: the element converted by its descriptor
    // (vertex_format_descriptor) - specialization constant kVertexFormatSpecId
    // + location, which an optimized relink gives the pipeline's own, or
    // StageParams::vertex_formats where that is 0 - each candidate the
    // translator's own expression, constants included (a divisor the driver
    // sees only at run time rounds differently from a constant one).
    std::map<std::uint32_t, Id> vertex_format_spec;  // one per location, whichever call site loads it
    std::size_t param_format_elements = 0;
    void load_vertex_element_from_params(const VertexElement& el, Id var) {
        if (el.location >= 16) {
            reject("vertex input: formats from params need locations below 16");
            return;
        }
        std::vector<Id> out;
        {
            EntryScope entry(m);  // inputs and their conversions dominate every block
            const Id raw = m.load(t_v4u, var);
            Id& spec = vertex_format_spec[el.location];
            if (!spec) spec = m.spec_const_u32(0, kVertexFormatSpecId + el.location);
            const Id from_params = m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(7), cu(el.location / 4), cu(el.location % 4)}));
            const Id d = m.emit(spv::OpSelect, t_u32, {m.emit(spv::OpINotEqual, t_bool, {spec, cu(0)}), spec, from_params});
            const Id kind = ibin(spv::OpBitwiseAnd, d, cu(7));
            const Id count = ibin(spv::OpBitwiseAnd, ibin(spv::OpShiftRightLogical, d, cu(4)), cu(7));
            const Id integer = m.emit(spv::OpLogicalOr, t_bool, {ieq(kind, cu(2)), ieq(kind, cu(4))});
            const Id one = m.emit(spv::OpSelect, t_u32, {integer, cu(1), cu(std::bit_cast<std::uint32_t>(1.0f))});
            const Id packing = ibin(spv::OpBitwiseAnd, ibin(spv::OpShiftRightLogical, d, cu(12)), cu(3));  // VertexElement::packed
            for (std::uint32_t k = 0; k < el.count && k < 4; ++k) {
                const Id byte = m.emit(spv::OpBitFieldUExtract, t_u32, {extract(t_u32, raw, 0), cu(8 * k), cu(8)});
                const Id half = m.emit(spv::OpBitFieldUExtract, t_u32, {extract(t_u32, raw, k / 2), cu(16 * (k % 2)), cu(16)});
                const Id c = m.emit(spv::OpSelect, t_u32, {ieq(packing, cu(1)), byte, m.emit(spv::OpSelect, t_u32, {ieq(packing, cu(2)), half, extract(t_u32, raw, k)})});
                const Id unorm8 = m.emit(spv::OpBitcast, t_u32, {fdiv(m.emit(spv::OpConvertUToF, t_f32, {c}), cf(255.0f))});
                const Id sint16 = from_i(m.emit(spv::OpBitFieldSExtract, t_i32, {to_i(c), cu(0), cu(16)}));
                // kinds 1 and 2: the raw bits
                const Id converted = m.emit(spv::OpSelect, t_u32, {ieq(kind, cu(3)), unorm8, m.emit(spv::OpSelect, t_u32, {ieq(kind, cu(4)), sint16, c})});
                // DST_SEL 1 or 0 past the format's components
                const Id past = m.emit(spv::OpSelect, t_u32, {m.emit(spv::OpINotEqual, t_bool, {ibin(spv::OpBitwiseAnd, d, cu(0x100u << k)), cu(0)}), one, cu(0)});
                out.push_back(m.emit(spv::OpSelect, t_u32, {m.emit(spv::OpULessThan, t_bool, {cu(k), count}), converted, past}));
            }
        }
        for (std::size_t k = 0; k < out.size(); ++k) write_v(static_cast<int>(el.vdata + k), word(out[k]));
        ++param_format_elements;
    }

    // TranslateOptions::vertex_input, as translate.cpp load_vertex_input: each
    // element from its input location (raw unsigned components; a packed
    // format as one dword), converted by its number format and DST_SEL - or,
    // with vertex_formats_from_params, by its descriptor in the params.
    void load_vertex_input() {
        inst_uniform = false;  // per vertex
        for (const VertexElement& el : opt.vertex_input) {
            Id& var = vertex_in_vars[el.location];
            if (!var) {
                var = m.global_variable(p_in_v4u, spv::ScInput);
                m.decorate(var, spv::DecLocation, {el.location});
                m.name(var, "vertex_in" + std::to_string(el.location));
                interface.push_back(var);
            }
            if (opt.vertex_formats_from_params) {
                load_vertex_element_from_params(el, var);
                if (!res.rejections.empty()) return;
                continue;
            }
            const std::uint32_t dfmt = (el.w3 >> 15) & 0xf, nfmt = (el.w3 >> 12) & 7;
            int count = 0, bits[4] = {};
            if (!format_layout(dfmt, count, bits)) {
                reject("vertex input: unsupported buffer data format " + std::to_string(dfmt));
                return;
            }
            // A packed format, or components the host packed (VertexElement::packed): taken from the input's dwords.
            const bool packed = (dfmt >= 6 && dfmt <= 9) || (el.packed && el.packed == vertex_packed_bits(dfmt));
            const Id one = nfmt == 4 || nfmt == 5 ? cu(1) : as_u(float_const(1.0f));
            std::vector<Id> comps(4, cu(0));
            {
                EntryScope entry(m);  // inputs and their conversions dominate every block
                const Id raw = m.load(t_v4u, var);
                for (int k = 0, bit = 0; k < count; bit += bits[k], ++k) {
                    const Id c = packed ? m.emit(spv::OpBitFieldUExtract, t_u32, {extract(t_u32, raw, static_cast<std::uint32_t>(bit / 32)),
                                                                                  cu(static_cast<std::uint32_t>(bit % 32)), cu(static_cast<std::uint32_t>(bits[k]))})
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

    // A write under EXEC E leaves a register exact in two places at once: the
    // E pixels where the value is, and the others where the old value was.
    // Neither need lie within the other (a value written in whole-quad mode
    // for one arm's quads, then the other arm's pixels written over it), so
    // `Exact` may name a union: a mask id made for it alone, never in the
    // module, whose parts are listed here. Every test of an Exact goes
    // through parts(); a mask test that does not (implies) finds no
    // relation for such an id, which only rejects.
    std::map<Id, std::vector<Exact>> unions;
    Id next_union = 0x7f000000u;  // above any id a module reaches
    std::vector<Exact> parts(const Exact& x) const {
        if (!x.mask.constant) {
            if (const auto it = unions.find(x.mask.id); it != unions.end()) return it->second;
        }
        return {x};
    }
    static bool within_part(const Lifter& l, const Exact& a, const Exact& b) { return (b.quad || !a.quad) && l.implies(a.mask, b.mask); }
    // Exact in the pixels of `a` and in those of `b` (a part that lies within
    // another is left out); `fallback` when that takes more than eight parts.
    Exact union_of(const Exact& a, const Exact& b, const Exact& fallback) {
        std::vector<Exact> out;
        const auto add = [&](const Exact& x) {
            for (const Exact& q : out) {
                if (within_part(*this, x, q)) return;
            }
            out.erase(std::remove_if(out.begin(), out.end(), [&](const Exact& q) { return within_part(*this, q, x); }), out.end());
            out.push_back(x);
        };
        for (const Exact& x : parts(a)) add(x);
        for (const Exact& x : parts(b)) add(x);
        if (out.size() == 1) return out[0];
        if (out.size() > 8) return fallback;
        const Exact u{false, lane(next_union++)};
        unions[u.mask.id] = std::move(out);
        return u;
    }

    // The pixels of `a` lie within those of `b`: each part of `a` within a part of `b`.
    bool within(const Exact& a, const Exact& b) const {
        for (const Exact& x : parts(a)) {
            bool in = false;
            for (const Exact& y : parts(b)) {
                if (within_part(*this, x, y)) {
                    in = true;
                    break;
                }
            }
            if (!in) return false;
        }
        return true;
    }
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
    // Exact in every pixel whose bit of `e` is set (within one part) ...
    bool exact_in(const Where& w, const Val& e) const {
        if (!w.partial) return true;
        for (const Exact& x : parts(w.at)) {
            if (implies(e, x.mask)) return true;
        }
        return false;
    }
    // ... and in every pixel of each quad holding one.
    bool quad_exact_in(const Where& w, const Val& e) const {
        if (!w.partial) return true;
        for (const Exact& x : parts(w.at)) {
            if (x.quad && implies(e, x.mask)) return true;
        }
        return false;
    }
    // Exact in every pixel whose bit of `e` is clear.
    bool exact_outside(const Where& w, const Val& e) const {
        if (!w.partial || (e.constant && e.set)) return true;
        if (e.constant) return false;
        const auto n = negation_of.find(e.id);
        if (n == negation_of.end()) return false;
        for (const Exact& x : parts(w.at)) {
            if (implies(lane(n->second), x.mask)) return true;
        }
        return false;
    }
    // Register `key` after a write under EXEC `e` of a value exact at `v`:
    // exact in the E pixels where the value is and elsewhere where the old
    // value was. With the value exact in all of E and the old one not outside
    // it, that is the union of E and the old value's pixels.
    void note_exact_write(int key, const Where& v, const Val& e) {
        const Where old = where_reg(key);
        const bool v_in = exact_in(v, e), old_out = exact_outside(old, e);
        const Where r = v_in && old_out ? Where{}
                        : v_in          ? Where{true, union_of(Exact{false, e}, old.at, old.at)}
                        : old_out       ? v
                                        : meet(v, old);
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
        if (in_divergent_loop()) {  // the quads' other pixels may have left the loop: the lift would not run them
            reject("whole-quad block inside a loop pixels leave at different iterations");
            return true;
        }
        const std::size_t at = static_cast<std::size_t>(&in - prog.insts.data());
        const auto loop_header = [&](std::uint32_t offset) {  // a block of the lift's own, and each iteration enters it with another EXEC
            return std::any_of(loops.begin(), loops.end(), [&](const Loop& l) { return l.header == offset; });
        };
        for (std::size_t i = at + 1; i < prog.insts.size(); ++i) {
            const Inst& p = prog.insts[i];
            if (block_starts.count(p.offset) || loop_header(p.offset) || is_branch(p) || (p.enc == Enc::SOPP && p.op == 1)) break;
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
        reg[key].helper_inexact = v.helper_inexact || inst_helper_inexact;  // GCN's helper lanes compute it in the block's quads
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
            // The quad's helper lanes too: GCN's texture unit reads their
            // VGPRs, which they computed only in whole-quad mode.
            if (const auto it = reg.find(256 + coord_va + k); it != reg.end() && it->second.helper_inexact) {
                reject(std::string(what) + " with implicit derivatives reads coordinates the quad's helper lanes may hold differently on GCN "
                       "(written outside whole-quad mode)");
                return false;
            }
        }
        return true;
    }

    // The sampled image for the reference's image and sampler bindings `ii`,
    // `si` (the image itself in `image`): the bindings, or bindless their slots
    // from the params block (StageParams::image_index / sampler_index, the
    // members after what the stage declares).
    std::uint32_t bindless_image_member = 7, bindless_sampler_member = 8;
    Id sampled_image(std::uint32_t ii, std::uint32_t si, Id& image) {
        Id sampler;
        if (ref.bindless) {
            const auto slot = [&](std::uint32_t member, std::uint32_t k) {
                return m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(member), cu(k / 4), cu(k % 4)}));
            };
            image = m.load(image_types[ii], m.access_chain(m.type_pointer(spv::ScUniformConstant, image_types[ii]), image_vars[ii],
                                                           {slot(bindless_image_member, ii)}));
            sampler = m.load(m.type_sampler(), m.access_chain(m.type_pointer(spv::ScUniformConstant, m.type_sampler()), sampler_vars[si],
                                                             {slot(bindless_sampler_member, si)}));
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
        if (in_divergent_loop()) {
            reject("image_get_lod inside a loop pixels leave at different iterations: its derivatives would read pixels that left");
            return;
        }
        const Val e = exec_lane();
        if (in.dmask & 0xc) {
            reject("image_get_lod returns two components, and its dmask selects a third or fourth");
            return;
        }
        if (in.dmask & 1) {
            reject("image_get_lod's clamped LOD (component 0): Vulkan's level after the view's and the sampler's clamps is not shown to be "
                   "GCN's");
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
        ++res.wave.lod_queries;
    }

    // The reference's image `ii` itself, for the instructions that take no
    // sampler: its binding, or bindless its slot.
    Id image_only(std::uint32_t ii) {
        if (!ref.bindless) return m.load(image_types[ii], image_vars[ii]);
        const Id slot = m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(bindless_image_member), cu(ii / 4), cu(ii % 4)}));
        return m.load(image_types[ii], m.access_chain(m.type_pointer(spv::ScUniformConstant, image_types[ii]), image_vars[ii], {slot}));
    }

    // image_load(_mip) and image_get_resinfo, as translate.cpp mimg: a texel
    // fetched at the integer coordinates in the VGPRs (and the level), its
    // bits as they land in the VGPRs for every texel kind; and the image's size
    // at the level in a VGPR with its level count, a cube one face deep. Neither
    // takes derivatives or a sampler.
    void image_load_or_size(const Inst& in) {
        const auto ii = ref.image_at.find(in.offset);
        if (ii == ref.image_at.end() || ii->second >= image_vars.size()) {
            reject(std::string("the reference has no image binding for this ") + mnemonic(in));
            return;
        }
        const ImageBinding& b = ref.images[ii->second];
        if (b.storage) {
            reject(std::string(mnemonic(in)) + " of a storage image binding");
            return;
        }
        const auto vgpr = [&](int idx) { return value(256 + idx); };
        if (in.op == 14) {  // image_get_resinfo: vaddr = the level; w, h, d, levels by dmask
            m.capability(spv::CapImageQuery);
            const Id image = image_only(ii->second);
            const int ncomp = coord_count(b.dim, b.arrayed);
            const Id size_type = ncomp == 1 ? t_i32 : ncomp == 2 ? t_v2i : t_v3i;
            const Id size = m.emit(spv::OpImageQuerySizeLod, size_type, {image, to_i(as_u(vgpr(in.vaddr)))});
            const Id levels = m.emit(spv::OpImageQueryLevels, t_i32, {image});
            Id comps[4] = {cu(0), cu(0), cu(0), from_i(levels)};
            for (int k = 0; k < ncomp; ++k) comps[k] = from_i(ncomp == 1 ? size : extract(t_i32, size, static_cast<std::uint32_t>(k)));
            if (b.cube) comps[2] = cu(1);
            int out = 0;
            for (std::uint32_t k = 0; k < 4; ++k) {
                if ((in.dmask >> k) & 1) write_v(in.vdata + out++, word(comps[k]));
            }
            return;
        }
        const int ncoord = coord_count(b.dim, b.arrayed);
        std::vector<Id> coords;
        for (int k = 0; k < ncoord; ++k) coords.push_back(to_i(as_u(vgpr(in.vaddr + k))));
        const Id coord = ncoord == 1 ? coords[0] : m.emit(spv::OpCompositeConstruct, ncoord == 2 ? t_v2i : ncoord == 3 ? t_v3i : m.type_vector(t_i32, 4), coords);
        const Id image = image_only(ii->second);
        std::vector<std::uint32_t> ops = {image, coord};
        if (in.op == 1) {  // image_load_mip: the level after the coordinates
            ops.push_back(spv::IoLod);
            ops.push_back(to_i(as_u(vgpr(in.vaddr + ncoord))));
        }
        const Id texel_type = b.kind == 1 ? t_v4u : b.kind == 2 ? m.type_vector(t_i32, 4) : t_v4f;
        Id texel = m.emit(spv::OpImageFetch, texel_type, ops);
        if (b.kind) texel = m.emit(spv::OpBitcast, t_v4f, {texel});
        int out = 0;
        for (std::uint32_t k = 0; k < 4; ++k) {
            if ((in.dmask >> k) & 1) write_v(in.vdata + out++, flt(extract(t_f32, texel, k)));
        }
        ++image_loads;
    }
    std::size_t image_loads = 0;

    void mimg(const Inst& in) {
        // A texel, a size or a LOD is a function of the image and sampler,
        // which are the draw's, and of the operands: every lane holds it
        // alike where every lane holds those alike (note_read sees each of
        // them read, EXEC included where it selects the write). The same
        // coordinates in every pixel of a quad have no derivative, so even an
        // implicit LOD is the same everywhere.
        const std::uint32_t op = in.op;
        if (in.tfe || in.lwe) {  // the translator does not write it either
            reject(std::string(mnemonic(in) ? mnemonic(in) : "an image instruction") +
                   " with TFE or LWE: GCN writes a status VGPR after the data, which is not lifted");
            return;
        }
        if (op == 96) {
            image_get_lod(in);
            return;
        }
        if (op == 0 || op == 1 || op == 14) {
            image_load_or_size(in);
            return;
        }
        if ((op == 8 || op == 9) && compute()) {
            guest_image_store(in);
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
        if (opt.stage == Stage::Pixel && !lod && !dx) ++res.wave.implicit_lod;
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

    // A vertex shader's pos1-3, as translate.cpp exp writes them:
    // PA_CL_VS_OUT_CNTL packs the extra vectors after the position - the misc
    // vector (bit 20), then the clip/cull distance vectors 0 (bit 22) and 1
    // (bit 23). The misc vector carries the point size (USE_VTX_POINT_SIZE); a
    // distance vector writes ClipDistance, distances 0-7 enabled by bits 0-7
    // with the cull bits 8-15 folded in - or, with kVsOutCntlClipVarying,
    // distances 0-3 to the clip location (a disabled one reads 1), which the
    // pixel stage discards by.
    Id out_clip = 0, out_point_size = 0, out_frag_depth = 0;
    int clip_count = 0;
    bool extra_position_vector(const Inst& in, const std::vector<Id>& comps) {
        const std::uint32_t cntl = opt.vs_out_cntl;
        int slot = 13, misc_slot = -1, cc0_slot = -1, cc1_slot = -1;
        if (cntl & (1u << 20)) misc_slot = slot++;
        if (cntl & (1u << 22)) cc0_slot = slot++;
        if (cntl & (1u << 23)) cc1_slot = slot++;
        const int tgt = static_cast<int>(in.tgt);
        if (tgt == misc_slot) {
            if (cntl & ((1u << 17) | (1u << 18) | (1u << 19) | (1u << 21))) {
                reject("a misc vector export with an edge flag, render target or viewport index or kill flag");
                return false;
            }
            if (cntl & (1u << 16)) {  // USE_VTX_POINT_SIZE
                if (!out_point_size) {
                    out_point_size = m.global_variable(m.type_pointer(spv::ScOutput, t_f32), spv::ScOutput);
                    m.decorate(out_point_size, spv::DecBuiltIn, {1 /* PointSize */});
                    interface.push_back(out_point_size);
                }
                m.store(out_point_size, comps[0]);
            }
            return true;
        }
        if (tgt != cc0_slot && tgt != cc1_slot) {
            reject("pos" + std::to_string(tgt - 12) + " export not enabled by PA_CL_VS_OUT_CNTL");
            return false;
        }
        if (cntl & TranslateOptions::kVsOutCntlClipVarying) {
            if (tgt != cc0_slot) return true;  // the host sends only distances 0-3 this way
            std::vector<Id> d(4, cf(1.0f));
            for (std::uint32_t k = 0; k < 4; ++k) {
                if (((in.dmask >> k) & 1) && ((cntl >> k) & 1)) d[k] = comps[k];
            }
            m.store(out_param(static_cast<int>(TranslateOptions::vs_clip_location(cntl))), m.emit(spv::OpCompositeConstruct, t_v4f, d));
            return true;
        }
        const int base = tgt == cc0_slot ? 0 : 4;
        if (!out_clip) {
            const std::uint32_t ena = (cntl & 0xff) | ((cntl >> 8) & 0xff);
            clip_count = 0;
            for (int k = 0; k < 8; ++k) {
                if ((ena >> k) & 1) clip_count = k + 1;
            }
            if (clip_count == 0) clip_count = 1;
            const Id t_arr = m.type_array(t_f32, cu(static_cast<std::uint32_t>(clip_count)));
            out_clip = m.global_variable(m.type_pointer(spv::ScOutput, t_arr), spv::ScOutput);
            m.decorate(out_clip, spv::DecBuiltIn, {3 /* ClipDistance */});
            m.capability(spv::CapClipDistance);
            interface.push_back(out_clip);
        }
        for (int k = 0; k < 4; ++k) {
            if (!((in.dmask >> k) & 1) || base + k >= clip_count) continue;
            m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_f32), out_clip, {cu(static_cast<std::uint32_t>(base + k))}), comps[k]);
        }
        return true;
    }

    void exp(const Inst& in) {
        // The translator exports whatever EXEC holds; a pixel whose bit is clear
        // here must be killed at s_endpgm (checked there).
        const bool vertex = vertex_like();
        if (tess_ls()) {
            reject("an export in the LS, whose output is its LDS writes");
            return;
        }
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
        // Pixel shaders export colour targets 0-7; vertex shaders the position
        // (12), the vectors PA_CL_VS_OUT_CNTL packs after it (13-15) and params (32-63).
        if (vertex ? !(in.tgt >= 12 && in.tgt < 16) && !(in.tgt >= 32 && in.tgt < 64) : in.tgt > 8) {
            reject(vertex ? "only position, clip and param exports are lifted in a vertex shader, not target " + std::to_string(in.tgt)
                          : std::string("only colour and depth exports are lifted"));
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
                const Val src = value(256 + in.vsrc[pair]);
                if (const auto mp = masked_packs.find(src.id); !src.constant && mp != masked_packs.end() && mp->second.pair.block == m.blocks()) {
                    // Packed under a varying EXEC, as the translator rounds it:
                    // toward zero where the pack was written, the old word decoded elsewhere.
                    const Id w = as_u(src);
                    Id r0, r1, d0, d1;
                    if (native_rtz) {
                        const Id r = m.emit(spv::OpFConvert, t_v2f,
                                            {m.emit(spv::OpFConvert, t_v2h, {m.emit(spv::OpCompositeConstruct, t_v2f, {mp->second.pair.lo, mp->second.pair.hi})})});
                        r0 = m.emit(spv::OpCompositeExtract, t_f32, {r, 0u});
                        r1 = m.emit(spv::OpCompositeExtract, t_f32, {r, 1u});
                        const Id d = m.emit(spv::OpFConvert, t_v2f, {m.emit(spv::OpBitcast, t_v2h, {w})});
                        d0 = m.emit(spv::OpCompositeExtract, t_f32, {d, 0u});
                        d1 = m.emit(spv::OpCompositeExtract, t_f32, {d, 1u});
                    } else {
                        r0 = emit_rtz_half(m, {t_bool, t_u32, t_i32, t_f32}, mp->second.pair.lo);
                        r1 = emit_rtz_half(m, {t_bool, t_u32, t_i32, t_f32}, mp->second.pair.hi);
                        d0 = unpack_half(ibin(spv::OpBitwiseAnd, w, cu(0xffff)));
                        d1 = unpack_half(ibin(spv::OpShiftRightLogical, w, cu(16)));
                    }
                    comps[pair * 2] = fsel(mp->second.exec, r0, d0);
                    comps[pair * 2 + 1] = fsel(mp->second.exec, r1, d1);
                    continue;
                }
                const Id w = as_u(src);
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
            if (in.tgt > 12 && in.tgt < 16) {
                if (!extra_position_vector(in, comps)) return;
            } else {
                const Id value = m.emit(spv::OpCompositeConstruct, t_v4f, comps);
                if (in.tgt == 12) {
                    m.store(out_position(), value);
                } else {
                    for (const Id var : param_outputs(static_cast<int>(in.tgt) - 32)) m.store(var, value);
                }
            }
            exports.push_back(in.offset);
            return;
        }
        if (in.tgt == 8) {  // mrtz, as translate.cpp exp: the depth from x, DepthReplacing
            if (!out_frag_depth) {
                out_frag_depth = m.global_variable(m.type_pointer(spv::ScOutput, t_f32), spv::ScOutput);
                m.decorate(out_frag_depth, spv::DecBuiltIn, {static_cast<std::uint32_t>(spv::BiFragDepth)});
                interface.push_back(out_frag_depth);
                m.execution_mode(fn_main, spv::ExDepthReplacing);
            }
            if (in.dmask & 1) m.store(out_frag_depth, comps[0]);
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
    std::size_t swizzles = 0, block_swizzles = 0;  // block_swizzles: inside a whole-quad block over a varying EXEC
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
    // ---- tessellation: LDS as the control point's slots (TessLift) ------------------
    // An LDS byte address as the translator maps it onto the control point
    // (translate.cpp attr_index): slot (a / 16) % attr_vec4s, component
    // (a / 4) % 4. Known addresses are constant slots; the others are indexed
    // by the same arithmetic at run time.
    struct LdsSlot {
        Id slot = 0, comp = 0;
        bool known = false;
    };
    LdsSlot lds_slot(const Val& base, std::uint32_t off) {
        const std::uint32_t n = std::max<std::uint32_t>(1, tess->attr_vec4s);
        if (base.constant) {
            const std::uint32_t d = (base.bits + off) >> 2;
            ++lds_static;
            return {cu((d >> 2) % n), cu(d & 3), true};
        }
        const Id d = ibin(spv::OpShiftRightLogical, ibin(spv::OpIAdd, as_u(base), cu(off)), cu(2));
        return {m.emit(spv::OpUMod, t_u32, {ibin(spv::OpShiftRightLogical, d, cu(2)), cu(n)}), ibin(spv::OpBitwiseAnd, d, cu(3)), false};
    }
    // `body` where the invocation's EXEC bit is set: GCN writes LDS only from
    // the lanes EXEC holds (translate.cpp exec_guard).
    void where_exec(const Val& e, const std::function<void()>& body) {
        if (e.constant) {
            if (e.set) body();
            return;
        }
        const Id l_then = m.fresh(), l_merge = m.fresh();
        m.emit_void(spv::OpSelectionMerge, {l_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {e.id, l_then, l_merge});
        m.label(l_then);
        body();
        m.emit_void(spv::OpBranch, {l_merge});
        cur_label = m.label(l_merge);
    }
    // ---- TessLift::window: the patch's window as stage I/O ------------------------------
    // Inputs [0, in_end): control point a / ls_stride, as the LS wrote it;
    // outputs [in_end, out_end): the hull's control point (a - in_end) /
    // cp_stride; patch constants [out_end, pc_end).
    struct Window {
        std::uint32_t n, ls, cp, npc, in_end, out_end, pc_end;
    };
    Window window_layout() const {
        Window w{std::max<std::uint32_t>(1, tess->control_points), tess->ls_stride, tess->cp_stride, tess->patch_constants, 0, 0, 0};
        w.in_end = w.n * w.ls;
        w.out_end = w.in_end + w.n * w.cp;
        w.pc_end = w.out_end + w.npc * 16;
        return w;
    }
    // The stage I/O dword at a known LDS address (`sym`: base + coef * the
    // hull's invocation), for a read (`write` false) or a write by this
    // invocation; 0 with `why` when it has none.
    Id window_slot(std::uint32_t base, std::uint32_t coef, bool write, std::string& why) {
        const Window w = window_layout();
        const Id p_in_u32_ = m.type_pointer(spv::ScInput, t_u32), p_out_u32 = m.type_pointer(spv::ScOutput, t_u32);
        const Id comp = cu((base >> 2) & 3);
        const auto invocation = [&] { return m.load(t_u32, in_invocation_id); };
        if (coef == 0) {
            const std::uint32_t a = base;
            if (a < w.in_end) {
                if (write || !tess_hs()) {
                    why = write ? "a write to an input control point" : "a read of an input control point, which the evaluation stage does not see";
                    return 0;
                }
                return m.access_chain(p_in_u32_, control_point, {cu(a / w.ls), cu((a % w.ls) / 16), comp});
            }
            if (a < w.out_end) {
                const std::uint32_t k = (a - w.in_end) / w.cp, slot = ((a - w.in_end) % w.cp) / 16;
                if (tess_ds()) return m.access_chain(p_in_u32_, control_point, {cu(k), cu(slot), comp});
                if (write && w.n != 1) {
                    why = "every invocation writes output control point " + std::to_string(k);
                    return 0;
                }
                return m.access_chain(write ? p_out_u32 : p_out_u32, cp_out, {write ? invocation() : cu(k), cu(slot), comp});
            }
            if (a < w.pc_end) {
                return m.access_chain(tess_ds() ? p_in_u32_ : p_out_u32, patch_io, {cu((a - w.out_end) / 16), comp});
            }
            why = "an address past the patch constants (" + std::to_string(a) + ")";
            return 0;
        }
        // The invocation's own control point: the same offset in each.
        if (tess_hs() && coef == w.ls && base < w.ls && !write) {
            return m.access_chain(p_in_u32_, control_point, {invocation(), cu(base / 16), comp});
        }
        if (tess_hs() && coef == w.cp && base >= w.in_end && base - w.in_end < w.cp) {
            return m.access_chain(p_out_u32, cp_out, {invocation(), cu((base - w.in_end) / 16), comp});
        }
        why = "an address that is not the invocation's own control point";
        return 0;
    }
    // A hull's read at an address the lift does not know, decoded at run time
    // region by region as the reference's window lays it out (0 past it).
    Id window_read_dynamic(Id a) {
        const Window w = window_layout();
        const Id p_in_u32_ = m.type_pointer(spv::ScInput, t_u32), p_out_u32 = m.type_pointer(spv::ScOutput, t_u32);
        const auto umin = [&](Id x, std::uint32_t hi) { return ext(spv::GlslUMin, {x, cu(hi)}, t_u32); };
        const auto udiv = [&](Id x, std::uint32_t d) { return m.emit(spv::OpUDiv, t_u32, {x, cu(d)}); };
        const auto umod = [&](Id x, std::uint32_t d) { return m.emit(spv::OpUMod, t_u32, {x, cu(d)}); };
        const auto ult = [&](Id x, std::uint32_t k) { return m.emit(spv::OpULessThan, t_bool, {x, cu(k)}); };
        const Id comp = ibin(spv::OpBitwiseAnd, ibin(spv::OpShiftRightLogical, a, cu(2)), cu(3));
        const Id in_k = umin(udiv(a, w.ls), w.n - 1);
        const Id in_v = m.load(t_u32, m.access_chain(p_in_u32_, control_point, {in_k, udiv(umod(a, w.ls), 16), comp}));
        const Id b = ibin(spv::OpISub, a, cu(w.in_end));
        const Id out_k = umin(udiv(b, w.cp), w.n - 1);
        const Id out_v = m.load(t_u32, m.access_chain(p_out_u32, cp_out, {out_k, udiv(umod(b, w.cp), 16), comp}));
        Id r = m.emit(spv::OpSelect, t_u32, {ult(a, w.out_end), out_v, cu(0)});
        if (w.npc) {
            const Id pc_slot = umin(udiv(ibin(spv::OpISub, a, cu(w.out_end)), 16), w.npc - 1);
            const Id pc_v = m.load(t_u32, m.access_chain(p_out_u32, patch_io, {pc_slot, comp}));
            r = m.emit(spv::OpSelect, t_u32, {ult(a, w.out_end), out_v, m.emit(spv::OpSelect, t_u32, {ult(a, w.pc_end), pc_v, cu(0)})});
        }
        return m.emit(spv::OpSelect, t_u32, {ult(a, w.in_end), in_v, r});
    }
    void tess_lds_window(const Inst& in, bool read, const Val& base, const std::vector<std::pair<std::uint32_t, int>>& dwords) {
        const char* name = mnemonic(in) ? mnemonic(in) : "unknown";
        const bool known = sym_holds(base, current_exec());
        if (read) {
            std::vector<Val> got;
            for (const auto& [o, reg_index] : dwords) {
                std::string why;
                Id ptr = 0;
                if (known) ptr = window_slot(base.sym_base + o, base.sym_coef, false, why);
                if (ptr) {
                    got.push_back(word(m.load(t_u32, ptr)));
                    ++lds_static;
                } else if (tess_hs()) {
                    got.push_back(word(window_read_dynamic(ibin(spv::OpIAdd, as_u(base), cu(o)))));
                    ++lds_dynamic;
                } else {
                    reject(std::string(name) + ": " + (why.empty() ? std::string("an address the lift does not know") : why));
                    return;
                }
                ++lds_reads;
            }
            for (std::size_t k = 0; k < dwords.size(); ++k) write_v(dwords[k].second, got[k]);
            return;
        }
        std::vector<Id> data, ptrs;
        for (const auto& [o, reg_index] : dwords) data.push_back(as_u(value(256 + reg_index)));
        for (const auto& [o, reg_index] : dwords) {
            std::string why = "an address the lift does not know";
            const Id ptr = known ? window_slot(base.sym_base + o, base.sym_coef, true, why) : 0;
            if (!ptr) {
                reject(std::string(name) + ": " + why);
                return;
            }
            ptrs.push_back(ptr);
        }
        where_exec(exec_lane(), [&] {
            for (std::size_t k = 0; k < ptrs.size(); ++k) {
                m.store(ptrs[k], data[k]);
                ++lds_writes;
                ++lds_static;
            }
        });
    }
    // The hull's factor stores (translate.cpp case 28): the first four (a
    // triangle) or six (a quad) single-dword buffer stores, in program order,
    // are the tessellation levels, outer first - where EXEC is set.
    void tess_factor_store(const Inst& in) {
        const std::uint32_t total = opt.tess_quads ? 6 : 4, nouter = opt.tess_quads ? 4 : 3;
        if (!tess_hs() || in.enc != Enc::MUBUF || in.op != 28 || tf_stores >= total) {
            reject(std::string(mnemonic(in) ? mnemonic(in) : "unknown") + ": the lift takes only a hull's factor stores");
            return;
        }
        if (!open_loops.empty()) {
            reject("a factor store inside a loop");
            return;
        }
        const std::uint32_t k = tf_stores++;
        const Id f = as_f(value(256 + in.vdata));
        const Id arr = k < nouter ? out_tess_outer : out_tess_inner;
        const Id at = cu(k < nouter ? k : k - nouter);
        where_exec(exec_lane(), [&] { m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_f32), arr, {at}), f); });
    }

    void tess_lds(const Inst& in) {
        const char* name = mnemonic(in) ? mnemonic(in) : "unknown";
        const bool read = in.op == 54 || in.op == 55;    // ds_read_b32, ds_read2_b32
        const bool write = in.op == 13 || in.op == 14;   // ds_write_b32, ds_write2_b32
        if (in.gds) {
            reject(std::string(name) + " with GDS set");
            return;
        }
        if (!read && !write) {
            reject(std::string(name) + " is not lifted in a tessellation stage");
            return;
        }
        if (tess_hs() ? false : read != tess_ds()) {
            reject(std::string(name) + (read ? ": the LS only writes its control point" : ": the domain shader only reads its control points"));
            return;
        }
        if (!open_loops.empty()) {
            reject(std::string(name) + " inside a loop");
            return;
        }
        const Val base = value(256 + in.vaddr);
        const std::uint32_t off = in.offset0 | (static_cast<std::uint32_t>(in.offset1) << 8);
        // (byte offset from the address, VGPR) of each dword.
        std::vector<std::pair<std::uint32_t, int>> dwords;
        if (in.op == 54) dwords = {{off, in.dst}};
        if (in.op == 55) dwords = {{in.offset0 * 4u, in.dst}, {in.offset1 * 4u, in.dst + 1}};
        if (in.op == 13) dwords = {{off, in.src0 - 256}};
        if (in.op == 14) dwords = {{in.offset0 * 4u, in.src0 - 256}, {in.offset1 * 4u, in.src1 - 256}};
        if (window() && !tess_ls()) {
            tess_lds_window(in, read, base, dwords);
            return;
        }
        if (read) {
            // Every invocation reads, as every lane of the translation does; the
            // results land under EXEC.
            std::vector<Val> got;
            for (const auto& [o, reg_index] : dwords) {
                const LdsSlot sl = lds_slot(base, o);
                got.push_back(word(m.load(t_u32, m.access_chain(m.type_pointer(spv::ScInput, t_u32), control_point, {cu(0), sl.slot, sl.comp}))));
                ++lds_reads;
            }
            for (std::size_t k = 0; k < dwords.size(); ++k) write_v(dwords[k].second, got[k]);
            return;
        }
        std::vector<Id> data;
        for (const auto& [o, reg_index] : dwords) data.push_back(as_u(value(256 + reg_index)));
        std::vector<LdsSlot> slots;
        for (const auto& [o, reg_index] : dwords) slots.push_back(lds_slot(base, o));
        where_exec(exec_lane(), [&] {
            for (std::size_t k = 0; k < dwords.size(); ++k) {
                m.store(m.access_chain(m.type_pointer(spv::ScOutput, t_u32), control_point, {slots[k].slot, slots[k].comp}), data[k]);
                ++lds_writes;
            }
        });
    }

    // ---- GDS counters and wave ranks (compute) ----------------------------------------
    // The lanes of GCN's wave that the lift runs together: a subgroup
    // operation reads every invocation of it, which holds only where the
    // lift's control flow is uniform - everywhere but inside a loop
    // invocations leave at different iterations (regions run unconditionally
    // and ifs take one arm in every lane).
    std::size_t gds_lane_ops = 0, wave_ranks = 0, ranks_folded = 0, first_lane_reads = 0;
    void subgroup_caps() {
        m.capability(spv::CapGroupNonUniform);
        m.capability(spv::CapGroupNonUniformBallot);
    }
    // An operation that needs every invocation together (a subgroup
    // operation, s_barrier): inside loops invocations may leave at different
    // iterations, each such loop must keep them together - every exit taken
    // on a bit every lane holds alike (close_loop checks the exits after it).
    bool require_converged(const std::string& what) {
        for (Loop* l : open_loops) {
            if (l->mask < 0) continue;  // a uniform loop: its invocations stay together
            if (l->divergent_exit) {
                reject(what + " inside the loop at " + hex_offset(l->header) + ", which invocations leave at different iterations");
                return false;
            }
            if (l->converged_op.empty()) l->converged_op = what + " at " + hex_offset(cur);
        }
        return true;
    }
    // Whether no instruction reachable after instruction `index` reads VGPR
    // `key` (any operand that may name it; a write does not end it - under a
    // varying EXEC the old value survives in the lanes it is clear for).
    bool vgpr_unread_after(std::size_t index, int key) const {
        std::vector<bool> seen(prog.insts.size(), false);
        std::vector<std::size_t> work, next;
        successors(index, next);
        work.assign(next.begin(), next.end());
        while (!work.empty()) {
            const std::size_t i = work.back();
            work.pop_back();
            if (i >= prog.insts.size() || seen[i]) continue;
            seen[i] = true;
            std::set<int> r;
            vector_reads(prog.insts[i], r);
            if (r.count(key)) return false;
            successors(i, next);
            work.insert(work.end(), next.begin(), next.end());
        }
        return true;
    }
    // The VGPRs an instruction may read (a superset).
    static void vector_reads(const Inst& in, std::set<int>& r) {
        const auto code = [&](std::uint16_t c, int n) {
            if (c < 256) return;
            for (int k = 0; k < n; ++k) r.insert(c + k);
        };
        const auto vg = [&](int first, int n) {
            for (int k = 0; k < n; ++k) r.insert(256 + first + k);
        };
        switch (in.enc) {
        case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3: {
            // A 64-bit operation (its mnemonic says so) reads register pairs.
            const char* mn = mnemonic(in);
            const int n = mn && std::strstr(mn, "64") ? 2 : 1;
            code(in.src0, n);
            if (in.enc != Enc::VOP1) code(in.src1, n);
            if (in.enc == Enc::VOP3) code(in.src2, n);
            if ((in.enc == Enc::VOP2 && (in.op == 6 || in.op == 31)) || (in.enc == Enc::VOP3 && (in.op == 0x106 || in.op == 0x11f))) vg(in.dst, 1);
            break;
        }
        case Enc::VINTRP: vg(in.src0 >= 256 ? in.src0 - 256 : in.src0, 1); break;
        case Enc::MIMG: vg(in.vaddr, 16); vg(in.vdata, 4); break;
        case Enc::MUBUF: case Enc::MTBUF: vg(in.vaddr, 2); vg(in.vdata, 4); break;
        case Enc::DS: vg(in.vaddr, 1); code(in.src0, 2); code(in.src1, 2); break;
        case Enc::EXP:
            for (int k = 0; k < 4; ++k) vg(in.vsrc[k], 1);
            break;
        default: break;
        }
    }
    // ds_append / ds_consume in GDS (translate.cpp gds_counter, the
    // reference's TranslateOptions::gds_counters): the counter is the word
    // the dispatch binds for the GDS dword at M0[15:0] + offset
    // (StageParams::gds_counter, from gds_base), by its device address, not
    // walked through the page table; none bound, nothing counts.
    // GCN adds or takes the wave's EXEC lanes in one step and hands every
    // lane the counter from before (its base); the shader adds the lane's
    // rank among EXEC's lanes (v_mbcnt) for its slot - HLSL's
    // IncrementCounter / DecrementCounter, lowered to a wave. The lift makes
    // each invocation whose EXEC bit is set step the counter itself (as a
    // D3D11 append buffer's counter does): the counter ends the same, and an
    // invocation's own step is its slot, so base + rank comes out as it -
    // the same slots, given out in another order, as GCN's waves themselves
    // take no fixed order. The base is kept apart (Val::wave_part) until it
    // meets its rank (combine_wave_parts), any other read of it rejects; a
    // value nothing reads leaves the VGPR alone. No subgroup operation: it
    // holds inside loops invocations leave at different iterations too.
    void gds_counter(const Inst& in) {
        const char* name = mnemonic(in) ? mnemonic(in) : "a GDS counter";
        if (!opt.gds_counters || !gds_member) {
            reject(std::string(name) + " in GDS where the reference counts in LDS (its GDS counters off)");
            return;
        }
        const bool append = in.op == 62;
        const std::uint32_t off = in.offset0 | (static_cast<std::uint32_t>(in.offset1) << 8);
        const Id m0 = as_u(value(kKeyM0));
        if (!res.rejections.empty()) return;
        const Id addr = ibin(spv::OpIAdd, ibin(spv::OpBitwiseAnd, m0, cu(0xffff)), cu(off));
        const Id base = params_head(2);  // StageParams::gds_base
        const Id slot = ibin(spv::OpISub, shift(spv::OpShiftRightLogical, addr, cu(2)), base);
        const Id inside = m.emit(spv::OpULessThan, t_bool, {slot, cu(kGdsCounters)});
        const Id at = shift(spv::OpShiftLeftLogical, m.emit(spv::OpSelect, t_u32, {inside, slot, cu(0)}), cu(1));
        const auto counter_word = [&](Id k) {
            return m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(gds_member), shift(spv::OpShiftRightLogical, k, cu(2)), ibin(spv::OpBitwiseAnd, k, cu(3))}));
        };
        const Id lo = counter_word(at), hi = counter_word(ibin(spv::OpIAdd, at, cu(1)));
        const Id bound = m.emit(spv::OpLogicalAnd, t_bool, {inside, m.emit(spv::OpINotEqual, t_bool, {ibin(spv::OpBitwiseOr, lo, hi), cu(0)})});
        const Val e = exec_lane();
        const Val run = e.constant ? (e.set ? lane(bound) : lane_const(false)) : lane(m.emit(spv::OpLogicalAnd, t_bool, {e.id, bound}));
        const Id step = cu(append ? 1u : 0xffffffffu);
        ++res.wave.gds_steps;
        const std::size_t index = static_cast<std::size_t>(&in - prog.insts.data());
        if (vgpr_unread_after(index, 256 + in.dst)) {
            where_exec_value(run, [&] { return m.emit(spv::OpAtomicIAdd, t_u32, {host_dword(make_u64(lo, hi)), cu(spv::ScopeDevice), cu(spv::MsNone), step}); });
            ++gds_lane_ops;
            note += std::string(" ") + (append ? "+1" : "-1") + " per invocation (value unread)";
            return;
        }
        const Id own = where_exec_value(run, [&] {
            return m.emit(spv::OpAtomicIAdd, t_u32, {host_dword(make_u64(lo, hi)), cu(spv::ScopeDevice), cu(spv::MsNone), step});
        });
        Val b = word(own);
        b.wave_part = append ? 1 : 2;
        b.part_mask = e.constant ? 0 : e.id;
        b.part_mask_constant = e.constant;
        b.part_mask_set = e.set;
        inst_uniform = false;
        write_v(in.dst, b);
        ++gds_lane_ops;
        note += std::string(" ") + (append ? "+1" : "-1") + " per invocation (its own slot)";
    }
    // The append idiom's two parts added together (v_add_i32 base, rank;
    // v_sub_i32 / v_subrev_i32 base - rank for a consume): the invocation's
    // own slot. Both over the same mask, used where EXEC lies within it; the
    // operands come back as plain words (the slot and 0). False with a
    // rejection for any other combination.
    std::size_t wave_slots = 0;
    bool combine_wave_parts(std::uint32_t op, Val& x, Val& y) {
        const Val* base = x.wave_part == 1 || x.wave_part == 2 ? &x : y.wave_part == 1 || y.wave_part == 2 ? &y : nullptr;
        const Val* rank = x.wave_part == 3 ? &x : y.wave_part == 3 ? &y : nullptr;
        const bool consume = base && base->wave_part == 2;
        const bool shape = base && rank && base != rank &&
                           (consume ? (op == 38 && base == &x) || (op == 39 && base == &y) : op == 37);
        if (!shape) {
            reject_wave_part(x.wave_part ? x : y);
            return false;
        }
        if (!same_lane(part_mask_val(*base), part_mask_val(*rank)) || !implies(exec_lane(), part_mask_val(*base))) {
            reject("a ds_append/ds_consume base and a rank over another mask, or added where EXEC lies outside it");
            return false;
        }
        // base + rank + S1 -> slot + S1; base - (rank + S1) -> slot - S1.
        const Val slot = word(base->id), rest = word(rank->id);
        if (base == &x) {
            x = slot;
            y = rest;
        } else {
            y = slot;
            x = rest;
        }
        inst_uniform = false;
        ++wave_slots;
        note += " slot";
        return true;
    }
    // A dword of the params block's member 3 (compute: { cb_valid, lds_bytes, gds_base, pad }).
    Id params_head(std::uint32_t k) {
        EntryScope entry(m);
        return m.load(t_u32, m.access_chain(p_uni_u32, ubo_var, {cu(3), cu(k)}));
    }
    // v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32 of a lane mask (translate.cpp
    // vop2): the mask's bits of the lanes below this one, counted, plus S1 -
    // the invocation's rank among the mask's lanes. In a workgroup of 32
    // invocations or fewer no lane is 32 or above: the high half counts
    // nothing and the result is S1. The rank of the append idiom is kept as
    // a part of it (Val::wave_part, combine_wave_parts), its word S1's,
    // beside the invocation's own slot. Any other rank is the reference's:
    // the ballot of each
    // invocation's bit (its bits of lanes no invocation runs are clear: masks
    // of EXEC and comparisons), the lane SubgroupLocalInvocationId (HLSL's
    // WavePrefixCountBits), with every invocation together.
    void mbcnt(const Inst& in, bool hi, std::uint16_t mask_code, std::uint16_t s1_code) {
        const Val s1 = read(s1_code, in);
        if (hi && workgroup_threads() <= 32 && !s1.wave_part) {
            write_v(in.dst, s1);
            ++ranks_folded;
            note += " (no lane 32+)";
            return;
        }
        int key = -1;
        if (mask_code == kExecLo || mask_code == kExecHi) key = kKeyExec;
        else if (mask_code == kVccLo || mask_code == kVccHi) key = kKeyVcc;
        else if (mask_code < 104) key = lanes.count(mask_code) ? mask_code : lanes.count(mask_code - 1) ? mask_code - 1 : -1;
        if (key < 0) {
            reject(std::string(mnemonic(in)) + " of a word, not a lane mask: the lane's index in the wave, which a subgroup of another width numbers differently");
            return;
        }
        const bool high_word = mask_code == kExecHi || mask_code == kVccHi || (key >= 0 && key < 104 && mask_code == key + 1);
        if (high_word != hi) {
            reject(std::string(mnemonic(in)) + " of a mask's other half");
            return;
        }
        const Val mask = key == kKeyExec ? exec_lane() : read_lane(key == kKeyVcc ? kVccLo : static_cast<std::uint16_t>(key));
        if (mask.kind != Kind::Lane) return;
        if (!mask.helpers_clear) {
            reject(std::string(mnemonic(in)) + " of a mask whose bits of lanes no invocation runs may be set");
            return;
        }
        // The idiom's rank: the high half alone (a workgroup past 32
        // invocations), then the low half over the same mask on it - or, in
        // a workgroup of 32 or fewer, the low half on anything (the high half
        // counts nothing there). Its word is S1's: the rank itself is folded
        // into the invocation's slot, and S1 rides along (base + rank + S1 is
        // the slot + S1).
        const bool idiom = hi ? !s1.wave_part
                              : (!s1.wave_part && workgroup_threads() <= 32) || (s1.wave_part == 4 && same_lane(part_mask_val(s1), mask));
        if (idiom) {
            Val r = word(s1.wave_part ? s1.id : as_u(s1));
            r.wave_part = hi ? 4 : 3;
            r.part_mask = mask.constant ? 0 : mask.id;
            r.part_mask_constant = mask.constant;
            r.part_mask_set = mask.set;
            inst_uniform = false;
            Val w = r;
            write_v(in.dst, w);
            ++wave_ranks;
            note += hi ? " (rank's high half)" : " (rank)";
            return;
        }
        if (s1.wave_part) {
            reject_wave_part(s1);
            return;
        }
        if (!require_converged(std::string(mnemonic(in)) + " (the other lanes' bits)")) return;
        subgroup_caps();
        if (!in_subgroup_lane) {
            EntryScope entry(m);
            in_subgroup_lane = builtin(p_in_u32, spv::BiSubgroupLocalInvocationId);
        }
        Id lane_id;
        {
            EntryScope entry(m);
            lane_id = m.load(t_u32, in_subgroup_lane);
        }
        const Id ballot = m.emit(spv::OpGroupNonUniformBallot, t_v4u, {cu(spv::ScopeSubgroup), mask.constant ? m.const_bool(mask.set) : mask.id});
        const Id lt32 = m.emit(spv::OpULessThan, t_bool, {lane_id, cu(32)});
        const Id below = hi ? m.emit(spv::OpSelect, t_u32, {lt32, cu(0), ibin(spv::OpISub, shift(spv::OpShiftLeftLogical, cu(1), ibin(spv::OpISub, lane_id, cu(32))), cu(1))})
                            : m.emit(spv::OpSelect, t_u32, {lt32, ibin(spv::OpISub, shift(spv::OpShiftLeftLogical, cu(1), lane_id), cu(1)), cu(0xffffffffu)});
        const Id count = m.emit(spv::OpBitCount, t_u32, {ibin(spv::OpBitwiseAnd, extract(t_u32, ballot, hi ? 1 : 0), below)});
        inst_uniform = false;
        write_v(in.dst, word(ibin(spv::OpIAdd, count, as_u(s1))));
        ++subgroup_ranks;
        ++res.wave.lane_ranks;
    }
    std::size_t subgroup_ranks = 0;

    void ds(const Inst& in) {
        if (compute()) {
            if ((in.op == 61 || in.op == 62) && in.gds) {
                gds_counter(in);
                return;
            }
            reject(std::string(mnemonic(in) ? mnemonic(in) : "unknown") + (in.op == 53 ? " (a lane shuffle) in a compute shader is not lifted" : " is not lifted"));
            return;
        }
        if (tess && in.op != 53) {
            tess_lds(in);
            return;
        }
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
        // lanes are inactive and read as 0). Or inside a whole-quad block over
        // a varying EXEC M (begin_whole_quad): GCN runs every lane of each quad
        // holding an M pixel, so there each lane reads its own quad's, as the
        // lift's quad operation does - the block runs in uniform control flow -
        // and the result is GCN's in M's quads where the source is in all of
        // them (whole_quad_write checks; GCN writes no other quad).
        if (!wqm.active) {
            const Val e = exec_lane();
            if (!(e.constant && e.set)) {
                reject("ds_swizzle_b32 where EXEC is not known set: a lane outside EXEC reads as 0 on GCN");
                return;
            }
            if (e.helper_inexact) {
                reject("ds_swizzle_b32 outside whole-quad mode: the quad's helper lanes are inactive on GCN and read as 0");
                return;
            }
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
        if (wqm.active) ++block_swizzles;
        const bool identity = sel[0] == 0 && sel[1] == 1 && sel[2] == 2 && sel[3] == 3;
        // Each lane reads one of its quad: the result holds GCN's value in a quad
        // where the source does in the whole quad (`partial`), and nowhere the
        // lifter can name where the source does only in some of its pixels.
        if (!identity && reads.partial && !reads.at.quad) reads = Where{true, Exact{false, lane_const(false)}};
        // A value every invocation holds alike is the same in every lane of the quad.
        if (uniform(src) || identity) {
            write_v(in.dst, src);
            return;
        }
        ++res.wave.quad_swizzles;  // a quad operation (folded ones above need none)
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
        case Enc::SOP2: case Enc::SOPK: {
            // TessLift: a scalar integer operation of constants is a constant (Val::sym).
            std::vector<Val> sym_ops;
            std::uint32_t probe = 0;
            if (tess && scalar_int_op(in, 0, 0, probe)) {
                if (in.enc == Enc::SOP2) {
                    sym_ops.push_back(peek(in.src0, in));
                    sym_ops.push_back(peek(in.src1, in));
                } else if (in.op == 15 || in.op == 16) {
                    sym_ops.push_back(peek(in.dst, in));
                }
            }
            if (in.enc == Enc::SOP2) sop2(in);
            else sopk(in);
            if (tess && res.rejections.empty() && (in.enc == Enc::SOP2 ? !sym_ops.empty() : in.op == 15 || in.op == 16)) sym_scalar(in, sym_ops);
            break;
        }
        case Enc::SOPC: sopc(in); break;
        case Enc::SOPP: sopp(in); break;
        case Enc::SMRD: smrd(in); break;
        case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3: valu(in); break;
        case Enc::VINTRP: vintrp(in); break;
        case Enc::MIMG: mimg(in); break;
        case Enc::MTBUF: mtbuf(in); break;
        case Enc::MUBUF:
            if (tess_hs()) {
                tess_factor_store(in);  // a hull's tessellation factors
                break;
            }
            mubuf(in);
            break;
        case Enc::EXP: exp(in); break;
        case Enc::DS: ds(in); break;
        default: reject(std::string(mnemonic(in) ? mnemonic(in) : "unknown") + " is not lifted"); break;
        }
    }

    // ---- scalar liveness after a region ---------------------------------------------
    // Scalar state an instruction reads and redefines, for the instructions the
    // lifter accepts. False for anything else (treated as reading everything).
    bool scalar_rw(const Inst& in, std::set<int>& r, std::set<int>& w) const {
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
            if ((in.op == 1 && !compute()) || in.op == 8) r.insert(kKeyExec);  // a dispatch's s_endpgm exports and discards nothing
            if (in.op == 4 || in.op == 5) r.insert(kKeyScc);
            if (in.op == 6 || in.op == 7) r.insert(kKeyVcc);
            return in.op == 0 || in.op == 1 || in.op == 2 || (in.op >= 4 && in.op <= 8) || in.op == 10 || in.op == 12 ||
                   (in.op >= 13 && in.op <= 22 && in.op != 18);
        case Enc::SMRD: {
            const int count = in.op < 8 ? (1 << in.op) : (in.op < 13 ? (1 << (in.op - 8)) : 0);
            if (compute() && in.op == 30) {  // s_memtime
                wr_pair(in.sdst);
                return true;
            }
            if (compute() && in.op == 31) return true;  // s_dcache_inv
            if (!count) return false;
            rd_pair(in.src0);
            if (!in.imm_flag && !in.has_literal) rd(static_cast<std::uint16_t>(in.imm));
            for (int k = 0; k < count; ++k) wr(static_cast<std::uint16_t>(in.sdst + k));
            return true;
        }
        case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3:
            if (compute() && ((in.enc == Enc::VOP1 && in.op == 2) || (in.enc == Enc::VOP3 && in.op == 0x182))) {  // v_readfirstlane_b32: an SGPR
                rd(in.src0);
                wr(in.dst);
                return true;
            }
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
            if ((in.enc == Enc::VOPC || (in.enc == Enc::VOP3 && in.op < 0x100)) && (in.op & 0x10)) {  // v_cmpx_*: EXEC &= the comparison
                r.insert(kKeyExec);
                w.insert(kKeyExec);
            }
            return true;
        case Enc::MIMG:
            for (int k = 0; k < (in.r128 ? 4 : 8); ++k) rd(static_cast<std::uint16_t>(in.srsrc + k));
            for (int k = 0; k < 4; ++k) rd(static_cast<std::uint16_t>(in.ssamp + k));
            return true;
        case Enc::MTBUF:
            if (compute()) {  // every typed access reads its V# as data, and soffset
                rd_pair(in.srsrc);
                rd_pair(static_cast<std::uint16_t>(in.srsrc + 2));
                rd(in.soffset);
                return true;
            }
            if (in.op >= 4) return false;
            rd_pair(in.srsrc);  // the V# the reference traced to its binding
            return true;
        case Enc::MUBUF:  // buffer stores (a vertex shader's, a hull's factor stores): the V# and offset are read
            if (compute() ? !(in.op <= 15 || (in.op >= 24 && in.op <= 31) || in.op == 50 || in.op == 112 || in.op == 113) : (in.op < 24 || in.op > 31)) {
                return false;
            }
            rd_pair(in.srsrc);
            rd_pair(static_cast<std::uint16_t>(in.srsrc + 2));
            rd(in.soffset);
            return true;
        case Enc::VINTRP: case Enc::EXP:
            return true;
        case Enc::DS:
            if (in.op == 13 || in.op == 14 || in.op == 54 || in.op == 55) {  // the LDS accesses a tessellation lift takes: M0 bounds them
                r.insert(kKeyM0);
                return true;
            }
            if (compute() && (in.op == 61 || in.op == 62) && in.gds) {  // a GDS counter: M0 names its range
                r.insert(kKeyM0);
                return true;
            }
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
        // GCN tests all 64 bits of VCC, its helper lanes' too: those must hold
        // the lift's bits, or be clear (the comparison ran where they were
        // inactive). A VCC clear for every pixel must be clear in every lane.
        if (v->second.helper_inexact && !v->second.helpers_clear) return false;
        if (uniform(v->second)) {
            if (v->second.constant && !v->second.set && !wave_clear(v->second)) return false;
            bit = v->second;
            return true;
        }
        const auto s = v->second.constant ? compare_splits.end() : compare_splits.find(v->second.id);
        const auto e = lanes.find(kKeyExec);
        if (s == compare_splits.end() || e == lanes.end() || !same_lane(e->second, s->second.exec)) return false;
        const Region* r = region_tested(e->second);
        if (!r) return false;
        if (!s->second.cond) {
            s->second.cond = vcmp_cond(s->second.op, s->second.u0, s->second.u1, s->second.md);
            uniform_ids.insert(s->second.cond);  // of values every lane holds alike
        }
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
    // A compute region as a selection (Region::branched): taken where the
    // invocation's EXEC bit is set as the region begins, its join the
    // region's end, where the registers it wrote become phis with their
    // values from before it (close_arm, merge_arms). It is the same program:
    // the region's writes are masked by that bit (so a skipping invocation
    // keeps its old values either way), its scalar state is dead after it
    // (close_region), and nothing in it needs the other invocations
    // (branch_region). What it saves is the work GCN's waves save by skipping
    // a region no lane needs.
    std::size_t selected_regions = 0, region_merge_phis = 0;
    bool branch_region(const Region& r) const {
        if (!compute() || r.kill || r.disabled) return false;
        for (std::size_t i = index_of(r.start); i < prog.insts.size() && prog.insts[i].offset < r.end; ++i) {
            const Inst& in = prog.insts[i];
            // Wave-wide operations, which GCN runs for the wave whatever EXEC
            // holds: a barrier, a lane's rank, lane reads and writes.
            if (in.enc == Enc::SOPP && in.op == 10) return false;
            if (in.enc == Enc::VOP2 && (in.op == 1 || in.op == 2 || in.op == 35 || in.op == 36)) return false;
            if (in.enc == Enc::VOP1 && in.op == 2) return false;
            if (in.enc == Enc::VOP3 && (in.op == 0x101 || in.op == 0x102 || in.op == 0x123 || in.op == 0x124 || in.op == 0x182)) return false;
            // A way out of a loop the region sits in (its back edge or an exit).
            for (const Loop* l : open_loops) {
                if (in.offset == l->latch || (loop_exits.count(in.offset) && innermost_loop(in.offset) == l)) return false;
            }
        }
        return true;
    }
    void open_region_arm(Region& r, std::uint64_t seq) {
        region_ifs.push_back({});
        Construct& c = region_ifs.back();
        c.branch = r.branch;
        c.op = 8;  // s_cbranch_execz
        c.else_start = c.join = r.end;
        Arm a;
        a.c = &c;
        a.seq = seq;
        a.region = true;
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
        a.in_mask0 = in_mask;
        exact_open_arm();
        const Val e = exec_lane();
        const Id cond = e.constant ? m.const_bool(e.set) : e.id;
        m.emit_void(spv::OpSelectionMerge, {a.merge, 0u});
        m.emit_void(spv::OpBranchConditional, {cond, a.then_label, a.else_label});
        cur_label = m.label(a.then_label);
        arms.push_back(std::move(a));
        r.selected = true;
        ++selected_regions;
    }
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
        a.in_mask0 = in_mask;
        exact_open_arm();
        // s_cbranch_scc0 and s_cbranch_vccz jump where their bit is clear, so the then arm runs where it is set;
        // s_cbranch_scc1 and s_cbranch_vccnz the other way round.
        const Val bit = c.op <= 5 ? scc_mask : vcc_bit;
        a.split_region = c.op <= 5 ? 0 : vcc_split;
        const Id cond = bit.constant ? m.const_bool(bit.set != c.negate) : c.negate ? m.emit(spv::OpLogicalNot, t_bool, {bit.id}) : bit.id;
        m.emit_void(spv::OpSelectionMerge, {a.merge, 0u});
        m.emit_void(spv::OpBranchConditional, {cond, a.then_label, a.else_label});
        cur_label = m.label(a.then_label);
        if (c.op >= 6) ++res.wave.wave_tests;  // VCCZ tests all 64 bits; shown the same in every lane
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
        a.in_mask1 = in_mask;
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
        in_mask = a.in_mask0;
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
                merged_lanes[k].wave_exact = t->second.wave_exact && e->second.wave_exact;
                merged_lanes[k].helpers_clear = t->second.helpers_clear && e->second.helpers_clear;
                if (t->second.content != e->second.content) merged_lanes[k] = new_content(merged_lanes[k]);
            } else {
                phis.push_back({k, Kind::Lane, t->second.id, e->second.id, !a.region && uniform(t->second) && uniform(e->second)});
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
            if (t.wave_part || e.wave_part) {
                reject(key_name(k) + " holds a part of the append idiom on one arm of the if at " + hex_offset(a.c->branch) + " and joins");
                return;
            }
            phis.push_back({k, t.kind, t.id, t.kind == Kind::Float ? as_f(e) : as_u(e), !a.region && uniform(t) && uniform(e)});
        }
        const bool scc_same = a.scc_valid1 && scc_valid && a.scc_uniform1 == scc_uniform && same(a.scc1v, scc_mask);
        const bool scc_phi = !scc_same && !a.region && a.scc_valid1 && scc_valid && a.scc_uniform1 && scc_uniform;
        const Id else_end = cur_label;
        m.emit_void(spv::OpBranch, {a.merge});
        cur_label = m.label(a.merge);
        for (const Phi& p : phis) {
            const Id type = p.kind == Kind::Lane ? t_bool : p.kind == Kind::Float ? t_f32 : t_u32;
            const Id id = m.emit(spv::OpPhi, type, {p.then_value, a.then_end, p.else_value, else_end});
            if (p.uni) uniform_ids.insert(id);
            if (p.kind == Kind::Lane) {
                merged_lanes[p.key] = new_content(lane(id));
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
        for (auto it = in_mask.begin(); it != in_mask.end();) it = a.in_mask1.count(*it) ? std::next(it) : in_mask.erase(it);  // within in both arms
        exact_merge_arms();
        lanes = std::move(merged_lanes);
        to_float = a.to_float0;
        to_word = a.to_word0;
        if (a.region) {
            region_merge_phis += phis.size();
            return;  // close_region has said what the region is
        }
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
            in_mask = a.in_mask1;
        }
        to_float = a.to_float0;
        to_word = a.to_word0;
        if (a.region) {
            reject("the branched region at " + hex_offset(a.c->branch) + " leaves a loop");  // branch_region keeps loop exits out
            return;
        }
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
    // describe. With kVectorReadsExec, every vector and memory instruction reads
    // EXEC as well (it runs under it); with kMaskReadsExec, those that make a
    // mask of it do: comparisons and carries (clear where EXEC is) and exports
    // (which pixels are written and survive).
    enum ExecReads { kNoExecReads, kVectorReadsExec, kMaskReadsExec };
    bool read_before_written(std::size_t from, const std::set<int>& keys, ExecReads exec_reads, std::uint32_t& at, int& key,
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
            if (exec_reads == kVectorReadsExec) {
                switch (in.enc) {
                case Enc::VOP1: case Enc::VOP2: case Enc::VOPC: case Enc::VOP3: case Enc::VINTRP: case Enc::MIMG: case Enc::MTBUF:
                case Enc::MUBUF: case Enc::EXP: case Enc::DS:
                    rd.insert(kKeyExec);
                    break;
                default: break;
                }
            } else if (exec_reads == kMaskReadsExec) {
                const bool carry = (in.enc == Enc::VOP2 && in.op >= 37 && in.op <= 42) || (in.enc == Enc::VOP3 && in.op >= 0x125 && in.op <= 0x12a);
                if (in.enc == Enc::EXP || in.enc == Enc::VOPC || (in.enc == Enc::VOP3 && in.op < 0x100) || carry) rd.insert(kKeyExec);
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

    // ---- regions as branches --------------------------------------------------------
    // GCN skips an s_cbranch_execz region in a wave none of whose lanes has
    // its EXEC bit set; the lift ran it for every pixel and kept the old
    // values where the bit is clear (select), so a light out of range of a
    // whole wave, or YEBIS's taps where nothing is out of focus, cost every
    // pixel all of their instructions. On the Radeon 8060S in Kyo's room
    // (2026-10-15) the main light pass's lift ran 741 vector instructions to
    // the console's 445, and the reference translation, which branches on an
    // empty EXEC, ran that pass in 0.66 ms to the lift's 0.85 and YEBIS's
    // 7d668276+157a299c in 0.12 to 0.33. A region whose every instruction
    // gives a pixel the same values taken or skipped where its bit is clear
    // is now a branch on the pixel's EXEC bit: the then block holds the region
    // as before, the skip block the values at the branch, and the merge a phi
    // for each value they leave different - for a pixel whose bit is clear,
    // what the select kept. The driver then skips the then block in a wave
    // none of whose pixels takes it, as GCN does. Not for a region that reads
    // other pixels or the quad (an implicit-LOD sample, a LOD query, LDS and
    // its swizzles, v_readlane / v_writelane), writes memory or exports, holds
    // an if, a loop or a kill region, or widens EXEC to whole quads; nor one
    // inside a loop or an if. A region inside a branched one may branch too.
    // BBHOST_LIFT_BRANCH=0: every region runs for every pixel, as before.
    static bool lift_branch_on() {
        static const bool on = [] {
            const char* e = std::getenv("BBHOST_LIFT_BRANCH");
            return !(e && e[0] == '0');
        }();
        return on;
    }
    bool region_branchable(const Region& r) const {
        if (r.kill || r.disabled || r.exec_before.constant || r.exec_before.kind != Kind::Lane) return false;
        for (const Inst& in : prog.insts) {
            if (in.offset < r.start || in.offset >= r.end) continue;
            const char* n = mnemonic(in);
            if (!n) return false;
            if (in.enc == Enc::EXP || in.enc == Enc::DS) return false;
            if (in.enc == Enc::SOPP) {
                if (is_branch(in) && in.op != 8) return false;  // an if, a kill region, a loop (s_cbranch_execz: a region within)
                if (in.op == 1 || in.op == 10 || (in.op >= 16 && in.op <= 18)) return false;  // s_endpgm, s_barrier, s_sendmsg(halt), s_trap
            }
            const std::string s(n);
            if (s.find("store") != std::string::npos || s.find("atomic") != std::string::npos) return false;
            if (s.rfind("image_get_lod", 0) == 0) return false;
            if (s.rfind("image_sample", 0) == 0 || s.rfind("image_gather", 0) == 0) {
                // Its LOD given (_l, _lz) or its gradients (_d, _cd): nothing
                // from the rest of the quad.
                bool explicit_lod = false;
                for (std::size_t at = s.find('_', 6); at != std::string::npos;) {
                    const std::size_t next = s.find('_', at + 1);
                    const std::string part = s.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
                    explicit_lod |= part == "l" || part == "lz" || part == "d" || part == "cd";
                    at = next;
                }
                if (!explicit_lod) return false;
            }
            if (s == "v_readlane_b32" || s == "v_writelane_b32" || s == "v_readfirstlane_b32" || s.rfind("s_wqm", 0) == 0) return false;
        }
        return true;
    }
    void region_branch_open(Region& r) {
        r.branched = true;
        r.rb_merge = m.fresh();
        r.rb_skip = m.fresh();
        const Id then_label = m.fresh();
        r.rb_reg0 = reg;
        r.rb_lanes0 = lanes;
        r.rb_to_float0 = to_float;
        r.rb_to_word0 = to_word;
        r.rb_scc0 = scc_mask;
        r.rb_scc_uniform0 = scc_uniform;
        r.rb_scc_valid0 = scc_valid;
        r.rb_poisoned0 = poisoned;
        r.rb_in_mask0 = in_mask;
        m.emit_void(spv::OpSelectionMerge, {r.rb_merge, 0u});
        m.emit_void(spv::OpBranchConditional, {r.exec_before.id, then_label, r.rb_skip});
        cur_label = m.label(then_label);
    }
    // After close_region's checks, which see the region's end as the then
    // block leaves it.
    void region_branch_close(Region& r) {
        const auto same = [](const Val& x, const Val& y) {
            return x.kind == y.kind && x.constant == y.constant && (x.constant ? (x.kind == Kind::Lane ? x.set == y.set : x.bits == y.bits) : x.id == y.id);
        };
        const std::map<int, Val> reg1 = reg, lanes1 = lanes;
        const std::set<int> poisoned1 = poisoned, in_mask1 = in_mask;
        const Val scc1 = scc_mask;
        const bool scc_uniform1 = scc_uniform, scc_valid1 = scc_valid;
        m.emit_void(spv::OpBranch, {r.rb_merge});
        const Id then_end = cur_label;
        // The skip block: the values at the branch (any conversion a phi
        // needs is made here).
        cur_label = m.label(r.rb_skip);
        reg = r.rb_reg0;
        lanes = r.rb_lanes0;
        to_float = r.rb_to_float0;
        to_word = r.rb_to_word0;
        const auto at = [&](const std::map<int, Val>& arm, int key) {
            if (const auto it = arm.find(key); it != arm.end()) return it->second;
            if (const auto it = r.rb_reg0.find(key); it != r.rb_reg0.end()) return it->second;
            return initial_value(key);
        };
        struct Phi {
            int key;
            Kind kind;
            Id then_value, skip_value;
        };
        std::vector<Phi> phis;
        std::set<int> lane_keys;
        for (const auto& [k, v] : lanes1) lane_keys.insert(k);
        for (const auto& [k, v] : lanes) lane_keys.insert(k);
        std::map<int, Val> merged_lanes;
        std::set<int> lost = poisoned1;
        lost.insert(poisoned.begin(), poisoned.end());
        for (int k : lane_keys) {
            const auto t = lanes1.find(k);
            const auto e = lanes.find(k);
            if (t == lanes1.end() || e == lanes.end()) {
                // A mask made inside the region only: unknown after it.
                if (k != kKeyVcc && k != kKeyExec) {
                    lost.insert(k);
                    lost.insert(k + 1);
                }
                continue;
            }
            if (same(t->second, e->second)) {
                merged_lanes[k] = t->second;
                merged_lanes[k].wave_exact = t->second.wave_exact && e->second.wave_exact;
                merged_lanes[k].helpers_clear = t->second.helpers_clear && e->second.helpers_clear;
                merged_lanes[k].helper_inexact = t->second.helper_inexact || e->second.helper_inexact;
                if (t->second.content != e->second.content) merged_lanes[k] = new_content(merged_lanes[k]);
            } else {
                phis.push_back({k, Kind::Lane, t->second.id, e->second.id});
            }
        }
        std::set<int> keys;
        for (const auto& [k, v] : reg1) keys.insert(k);
        for (const auto& [k, v] : reg) keys.insert(k);
        std::map<int, Val> merged;
        std::map<int, bool> inexact;
        for (int k : keys) {
            if (lane_keys.count(k) || (k > 0 && k < 104 && lane_keys.count(k - 1)) || lost.count(k)) continue;  // a lane mask's words, or lost
            const Val t = at(reg1, k), e = at(reg, k);
            if (same(t, e)) {
                merged[k] = t;
                continue;
            }
            if (descriptor(t) || descriptor(e)) {  // a descriptor word on either path: still never data
                merged[k] = descriptor(t) ? t : e;
                continue;
            }
            inexact[k] = t.helper_inexact || e.helper_inexact;
            phis.push_back({k, t.kind, t.id, t.kind == Kind::Float ? as_f(e) : as_u(e)});
        }
        m.emit_void(spv::OpBranch, {r.rb_merge});
        const Id skip_end = cur_label;
        cur_label = m.label(r.rb_merge);
        for (const Phi& p : phis) {
            // Never uniform: the branch is the pixel's own bit.
            const Id type = p.kind == Kind::Lane ? t_bool : p.kind == Kind::Float ? t_f32 : t_u32;
            const Id id = m.emit(spv::OpPhi, type, {p.then_value, then_end, p.skip_value, skip_end});
            if (p.kind == Kind::Lane) {
                Val v = new_content(lane(id));
                v.helper_inexact = lanes1.at(p.key).helper_inexact || r.rb_lanes0.at(p.key).helper_inexact;
                merged_lanes[p.key] = v;
            } else {
                Val v = p.kind == Kind::Float ? flt(id) : word(id);
                v.helper_inexact = inexact[p.key];
                merged[p.key] = v;
            }
        }
        reg = std::move(merged);
        lanes = std::move(merged_lanes);
        to_float = r.rb_to_float0;
        to_word = r.rb_to_word0;
        poisoned = std::move(lost);
        in_mask = in_mask1;
        for (auto it = in_mask.begin(); it != in_mask.end();) it = r.rb_in_mask0.count(*it) ? std::next(it) : in_mask.erase(it);
        // SCC the region wrote is dead after it (close_region showed it): the
        // value at the branch dominates the merge.
        if (scc_valid1 && r.rb_scc_valid0 && scc_uniform1 == r.rb_scc_uniform0 && same(scc1, r.rb_scc0)) {
            scc_mask = r.rb_scc0;
            scc_uniform = r.rb_scc_uniform0;
            scc_valid = true;
        } else {
            scc_mask = r.rb_scc0;
            scc_uniform = r.rb_scc_uniform0;
            scc_valid = false;
        }
        ++branched_regions;
        res.proof.push_back("region " + hex_offset(r.start) + "-" + hex_offset(r.end) + " branches on the pixel's EXEC bit at " +
                            hex_offset(r.branch) + ": nothing in it reads another pixel or writes memory, so a pixel whose bit is clear "
                            "gets what the select kept; " + std::to_string(phis.size()) + " phis");
    }
    std::size_t branched_regions = 0;

    void close_region(Region& r, std::size_t next_index) {
        r.closed = true;
        // Within a loop's mask after the region only where so on both paths: a
        // register it writes keeps its old value where GCN skips it. (EXEC is
        // clear in every lane there.)
        for (const int k : r.scalar_writes) {
            if (k != kKeyExec && !r.in_mask0.count(k)) in_mask.erase(k);
        }
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
            // GCN skips the region only where the guard is clear in every lane
            // of the wave, all of whose pixels the lift kills - and keeps EXEC as
            // it was at the branch, which s_endpgm would keep. So on every path
            // EXEC must be set again (from what does not read it) before it
            // decides anything: an export, a mask made of it, s_endpgm.
            if (!same) {
                std::uint32_t at = 0;
                int read = 0;
                bool unknown = false;
                if (read_before_written(next_index, {kKeyExec}, kMaskReadsExec, at, read, unknown)) {
                    cur = at;
                    const Inst* in = inst_at(at);
                    reject(unknown ? "cannot show EXEC is set again after the kill region at " + hex_offset(r.start) + ": " +
                                         (in && mnemonic(*in) ? mnemonic(*in) : "unknown") + " is not modelled"
                                   : "EXEC as the kill region at " + hex_offset(r.start) + " left it decides " +
                                         (in && mnemonic(*in) ? mnemonic(*in) : "an instruction") +
                                         " here, where GCN, skipping the region in a wave no pixel of which passes, keeps EXEC as at the branch");
                    return;
                }
            }
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
            ++res.wave.kill_skips;
        }
        // Every path out of the region, through the rest of the program and
        // around loops, must write each of them before reading it.
        std::uint32_t read_at = 0;
        int read_key = 0;
        bool unmodelled = false;
        if (read_before_written(next_index, live, kNoExecReads, read_at, read_key, unmodelled)) {
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
            how = ": writes masked by EXEC at " + hex_offset(r.branch) + ", and EXEC set inside only within it" +
                  (r.selected ? std::string("; a selection on the invocation's EXEC bit there, skipped where it is clear as GCN's wave skips "
                                            "the region where no lane's is")
                              : std::string());
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
    //     (or, on a second attempt, a lane mask gets a phi). Inside a region
    //     the lift runs it also in the waves where GCN skips the region, for a
    //     trip count computed only from values every wave of the draw holds
    //     alike: the count GCN runs in any wave that enters the region.
    //   * A divergent loop is the compiler's per-lane exit idiom,
    //         s_andn2_b64 L, L, exec   ; L loses the lanes leaving (EXEC holds them)
    //         s_cbranch_scc0 exit      ; no lane left in L: the wave is done
    //     with L a copy of EXEC where the loop begins. A pixel leaves when its
    //     own L bit clears. On GCN it then runs the remaining iterations with
    //     its EXEC bit clear (L only loses lanes, and EXEC in the body stays
    //     within L), so its VGPRs keep what they held when it left. After the
    //     exit test the body knows the pixel's L bit is set, and so is EXEC as
    //     it was at entry (L started as it); a pixel back at the header has
    //     EXEC and L set, which is EXEC at entry again, so both keep that value
    //     there and must come back set. That folding holds for the pixel only:
    //     that EXEC stays within L is kept by register (in_mask), L is written
    //     only to lose lanes, no whole-quad mode runs inside, and no wave-wide
    //     test of a mask other than the exit test decides anything. Everything
    //     scalar the loop writes (SGPRs, SCC, VCC, EXEC) must be dead after it,
    //     as GCN may run more iterations than the pixel did, and no
    //     implicit-LOD sample or other cross-lane read may run inside (the rest
    //     of its quad may have left).
    struct Loop {
        std::uint32_t header = 0, latch = 0, exit = 0;
        std::size_t header_index = 0, latch_index = 0;
        int mask = -1;                // divergent: the loop mask's SGPR pair
        std::uint32_t mask_exit = 0;  // divergent: the (first) s_cbranch_scc0 that tests it
        std::set<std::uint32_t> mask_exits;  // divergent: every s_cbranch_scc0 of the exit idiom on the mask
        std::uint32_t dead = 0;       // the start of unreachable instructions between the back edge and `exit` (0: none)
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
            std::map<int, Exact> partial;
            bool per_pixel = false;
            std::set<int> in_mask;
            std::map<int, Id> words;  // a loop of several exits: each float register's bits, made on this exit's edge
        };
        std::vector<Exit> exits;
        std::size_t exit_branches = 0;  // the branches that leave it for its exit (find_loops)
        std::map<int, Exact> partial0;  // where the phis' VGPRs hold GCN's value at the header: where they did on entry
        std::map<Id, Id> to_float0, to_word0;
        std::vector<Id> assumed;  // divergent: what the body knows is set
        std::size_t arms0 = 0, regions0 = 0;
        std::set<int> in_mask0;  // Lifter::in_mask where the loop begins
        std::uint32_t region = 0;  // the innermost region (its start) the loop sits in, which GCN may skip
        // Divergent, inside a kill region widened from EXEC known set: whether
        // the pixel's quad holds a covered pixel the region's guard keeps (the
        // quads GCN runs it in), and that region's start.
        Id quad_live = 0;
        std::uint32_t quad_region = 0;
        // compute: an s_barrier or a subgroup operation inside, which needs
        // every invocation that entered the loop in each iteration (what and
        // where), and whether an exit lets invocations leave at different
        // iterations - the two cannot both hold (close_loop).
        std::string converged_op;
        bool divergent_exit = false;
    };
    std::vector<Loop> loops;  // by header
    std::vector<Loop*> open_loops;
    std::set<std::uint32_t> loop_exits;  // branches that leave their innermost loop for its exit
    std::set<Id> assumed;                // lane masks set wherever the lift now is (a divergent loop's body)
    bool block_dead = false;             // the current block ended by leaving a loop
    const LoopPhis lane_phi_hints, inexact_hints, nonuniform_hints;
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
    const Loop* innermost_divergent_loop() const {
        for (auto it = open_loops.rbegin(); it != open_loops.rend(); ++it) {
            if ((*it)->mask >= 0) return *it;
        }
        return nullptr;
    }
    // Inside a loop pixels leave at different iterations: the lane registers
    // whose 64 bits lie within its loop mask as the mask is now (keys of
    // `lanes`). GCN keeps a lane that left out of the rest of the loop only by
    // EXEC staying within the mask, which the lift's per-pixel values cannot
    // show: the body folds masks to what they hold for its own pixel
    // (known()), and EXEC as the loop found it - which still has the lanes
    // that left - holds the same value there as the mask at the header. So it
    // is kept by register, from how each mask was made: EXEC and the mask at
    // the header; a copy of one; an AND with one; an operand ANDed with the
    // complement of anything; a comparison or carry under such an EXEC (the
    // lanes outside EXEC write 0). Every VGPR write needs EXEC in it, and so
    // does the back edge.
    std::set<int> in_mask;
    bool mask_within(std::uint16_t code) const {
        if (code == kExecLo) return in_mask.count(kKeyExec) != 0;
        if (code == kVccLo) return in_mask.count(kKeyVcc) != 0;
        if (code < 104) return in_mask.count(code) != 0;
        return code == 128;  // the inline 0: no lane
    }
    // A lane mask written at `key`. The loop mask itself only loses lanes, and
    // only at the top level of its own body; after it does, nothing else is
    // known to lie within it.
    bool note_in_mask(int key, bool within) {
        if (!innermost_divergent_loop()) return true;
        for (const Loop* l : open_loops) {
            if (l->mask < 0 || key != l->mask) continue;
            if (!within) {
                reject("the loop mask " + key_name(key) + " of the loop at " + hex_offset(l->header) +
                       " is written with lanes beyond it: the lanes that left would join the loop again");
                return false;
            }
            if (l->mask == kKeyExec) {
                // A loop on EXEC: its body sets EXEC anywhere (its regions
                // save and restore it), always within the lanes still in the
                // loop; the exit test is where the loop mask narrows
                // (loop_exit), not each write.
                in_mask.insert(key);
                return true;
            }
            // Inside a region only by the exit idiom (s_andn2_b64 L, L, exec right
            // before one of the loop's divergent exits): the lanes leaving there.
            const bool exit_idiom = open_regions.size() > l->regions0 && l->mask_exits.count(cur + 4) != 0;
            if (l != open_loops.back() || arms.size() != l->arms0 || (open_regions.size() != l->regions0 && !exit_idiom)) {
                reject("the loop mask " + key_name(key) + " of the loop at " + hex_offset(l->header) +
                       " is written inside an if, a region or an inner loop of its body");
                return false;
            }
            in_mask = {key};
            return true;
        }
        if (within) {
            in_mask.insert(key);
        } else {
            in_mask.erase(key);
        }
        return true;
    }
    // The loop mask copied from EXEC by s_mov_b64 on the straight way into the
    // header, neither written since: the same 64 bits, where an equal constant
    // need not be (-1 also has the lanes no pixel runs in, which never leave).
    bool mask_copied_from_exec(const Loop& l) const {
        if (l.mask == kKeyExec) return true;  // a loop on EXEC: the mask is EXEC
        for (std::size_t i = l.header_index; i-- > 0;) {
            const Inst& p = prog.insts[i];
            if (block_starts.count(prog.insts[i + 1].offset) || is_branch(p)) return false;
            if (p.enc == Enc::SOP1 && p.op == 4 && p.dst == l.mask && p.src0 == kExecLo) return true;
            std::set<int> rd, wr;
            if (!scalar_rw(p, rd, wr) || wr.count(l.mask) || wr.count(l.mask + 1) || wr.count(kKeyExec)) return false;
        }
        return false;
    }
    // A mask clear in every lane of GCN's wave: a wave-wide test of it (SCC of
    // the operation that made it, VCCZ) says no lane is set.
    static bool wave_clear(const Val& v) { return v.constant && !v.set && v.wave_exact; }
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
        r.helpers_clear = v.helpers_clear;  // GCN's helper lanes hold what they held
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
        // A loop's exit may follow instructions nothing reaches: the back edge
        // is an unconditional s_branch, and no branch lands between it and the
        // exit (the compiler leaves the copies of a dead path there). Every exit
        // of the loop must then land past them, at the same place.
        std::set<std::uint32_t> targets;
        for (const Inst& in : prog.insts) {
            if (is_branch(in)) targets.insert(in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4);
        }
        const auto unreached = [&](std::uint32_t from, std::uint32_t to) {
            const auto it = targets.lower_bound(from);
            return it == targets.end() || *it >= to;
        };
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
            if (from && target > from->exit && !from->dead && unreached(from->exit, target) &&
                std::none_of(loop_exits.begin(), loop_exits.end(), [&](std::uint32_t e) { return innermost_loop(e) == from; }) &&
                std::none_of(loops.begin(), loops.end(), [&](const Loop& o) { return &o != from && o.header >= from->exit && o.header < target; })) {
                Loop& l = loops[static_cast<std::size_t>(from - loops.data())];
                l.dead = l.exit;
                l.exit = target;
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
            l.exit_branches = static_cast<std::size_t>(
                std::count_if(loop_exits.begin(), loop_exits.end(), [&](std::uint32_t e) { return innermost_loop(e) == &l; }));
            // The divergent exit idiom, at the loop's top level: s_andn2_b64 L,
            // L, exec right before an s_cbranch_scc0 to the exit. (Inside an if
            // arm the same pair is the uniform loop's way out: L is EXEC there,
            // so no lane is left and the branch is always taken.)
            // The same pair inside s_cbranch_execz regions of the loop's top level
            // (no if around it in the body) is a divergent exit too: EXEC there
            // is the lanes that leave, which the pair takes out of L. Every such
            // exit must take lanes out of the same mask.
            for (std::size_t i = l.header_index + 1; i < l.latch_index; ++i) {
                const Inst& br = prog.insts[i];
                const Inst& andn2 = prog.insts[i - 1];
                if (!(br.enc == Enc::SOPP && br.op == 4 && loop_exits.count(br.offset) && innermost_loop(br.offset) == &l)) continue;
                if (!(andn2.enc == Enc::SOP2 && andn2.op == 21 && andn2.dst < 103 && andn2.src0 == andn2.dst && andn2.src1 == kExecLo)) continue;
                const auto inside = [&](std::uint32_t x0, std::uint32_t x1) { return x0 >= l.header && x0 < br.offset && br.offset < x1; };
                if (std::any_of(constructs.begin(), constructs.end(), [&](const Construct& c) { return inside(c.branch, c.join); })) continue;
                if (l.mask >= 0 && andn2.dst != l.mask) {
                    cur = br.offset;
                    reject("two divergent exits from the loop at " + hex_offset(l.header) + " on different masks");
                }
                if (l.mask < 0) {
                    l.mask = andn2.dst;
                    l.mask_exit = br.offset;
                }
                l.mask_exits.insert(br.offset);
            }
            // compute(): a loop on EXEC - the compiler's form of a loop whose
            // lanes leave one by one, s_and_b64 exec, exec, <condition> (or
            // another narrowing) and an s_cbranch_execz to the exit once no lane
            // is left - at the loop's top level. Its loop mask is EXEC itself
            // (Loop::mask kKeyExec): a lane leaves when its EXEC bit clears.
            for (std::size_t i = l.header_index; compute() && i < l.latch_index; ++i) {
                const Inst& br = prog.insts[i];
                if (!(br.enc == Enc::SOPP && br.op == 8 && loop_exits.count(br.offset) && innermost_loop(br.offset) == &l)) continue;
                const auto inside = [&](std::uint32_t x0, std::uint32_t x1) { return x0 >= l.header && x0 < br.offset && br.offset < x1; };
                if (std::any_of(constructs.begin(), constructs.end(), [&](const Construct& c) { return inside(c.branch, c.join); })) continue;
                if (l.mask >= 0 && l.mask != kKeyExec) {
                    cur = br.offset;
                    reject("the loop at " + hex_offset(l.header) + " leaves on EXEC and on the mask " + key_name(l.mask));
                }
                if (l.mask < 0) {
                    l.mask = kKeyExec;
                    l.mask_exit = br.offset;
                }
                l.mask_exits.insert(br.offset);
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
    mutable std::uint32_t unmodelled_at = 0;  // read_first: the instruction it could not model
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
                unmodelled_at = prog.insts[i].offset;
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
            const Inst* u = inst_at(unmodelled_at);
            reject("cannot show what the loop at " + hex_offset(l.header) + " reads before writing: " +
                   (u && mnemonic(*u) ? mnemonic(*u) : "an instruction") + " at " + hex_offset(unmodelled_at) + " is not modelled");
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
            // compute: a VGPR every lane enters the loop holding alike is
            // taken to stay alike (checked at the back edge, another attempt
            // if not) - a loop counter from a constant, a bound from user data.
            const bool uni = (k < 256 || (compute() && !nonuniform_hints.count({l.header, k}))) && uniform(v);
            words.push_back({k, f ? Kind::Float : Kind::Word, f ? v.id : as_u(v), uni, inexact_at_header(k, v)});
        };
        for (int k : vgprs) add_word(k);
        const auto e = lanes.find(kKeyExec);
        Val exec0 = e != lanes.end() ? e->second : lane_const(true);
        for (int k : l.writes) {
            // EXEC, SCC and a divergent loop's mask are handled below. (A uniform
            // loop has no mask: s0 and s1 are registers like any other.)
            if (k == kKeyScc || k == kKeyExec || (l.mask >= 0 && l.mask < 104 && (k == l.mask || k == l.mask + 1))) continue;
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
            if (lm == lanes.end() || !same_lane(lm->second, exec0) || !mask_copied_from_exec(l)) {
                reject("the loop at " + hex_offset(l.header) + " tests " + key_name(l.mask) + " as its loop mask, which does not start as EXEC");
                return;
            }
            l.unchanged[kKeyExec] = exec0;
            l.unchanged[l.mask] = lm->second;
        }
        // In a loop pixels leave at different iterations (or inside one) the
        // header's masks are what the pixel knows, not GCN's 64 bits, which
        // lose the lanes that left: masks of their own there. Elsewhere a mask
        // comes back to the header as the one it left with (checked at the back edge).
        const bool per_pixel_header = l.mask >= 0 || in_divergent_loop();
        for (auto& [k, v] : l.unchanged) {
            if (!per_pixel_header) continue;
            v = new_content(v);
            if (const auto it = lanes.find(k); it != lanes.end()) it->second.content = v.content;
        }
        for (auto& [k, v] : l.unchanged) {  // assumed unchanged, helper lanes included
            if (!inexact_at_header(k, v)) continue;
            v.helper_inexact = true;
            if (const auto it = lanes.find(k); it != lanes.end()) it->second.helper_inexact = true;
        }
        // In a kill region widened from EXEC known set every pixel computes the
        // region, where GCN runs only the quads holding a pixel the guard keeps
        // (EXEC is the guard's whole quads): the other quads' pixels are killed,
        // but would run the loop's iterations on data GCN never looked at, so
        // they leave it at the first exit test, as on GCN, where their EXEC and
        // so their loop mask are clear. Made here, in uniform control flow (the
        // loop is not inside another that pixels leave at different
        // iterations), from the guard of each covered pixel of the quad
        // (Vulkan's helper invocations stand for the lanes GCN's coverage
        // leaves out).
        if (l.mask >= 0 && !in_divergent_loop()) {
            for (auto it = open_regions.rbegin(); it != open_regions.rend(); ++it) {
                if (!(*it)->kill || !(*it)->quad_exact) continue;
                const Val& g = (*it)->guard;
                if (!in_helper) in_helper = builtin(p_in_bool, spv::BiHelperInvocation);
                Id helper;
                {
                    EntryScope entry(m);
                    helper = m.load(t_bool, in_helper);
                }
                const Id covered = m.emit(spv::OpLogicalNot, t_bool, {helper});
                const Id kept = g.constant ? (g.set ? covered : m.const_bool(false)) : m.emit(spv::OpLogicalAnd, t_bool, {g.id, covered});
                Id any = kept;
                for (std::uint32_t d = 0; d < 3; ++d) any = m.emit(spv::OpLogicalOr, t_bool, {any, quad_swap(t_bool, kept, d)});
                l.quad_live = any;
                ++res.wave.helper_queries;
                l.quad_region = (*it)->start;
                break;
            }
        }
        l.preheader = cur_label;
        l.header_label = m.fresh();
        l.continue_label = m.fresh();
        l.merge_label = m.fresh();
        l.to_float0 = to_float;
        l.to_word0 = to_word;
        l.arms0 = arms.size();
        l.regions0 = open_regions.size();
        l.region = open_regions.empty() ? 0 : open_regions.back()->start;
        m.emit_void(spv::OpBranch, {l.header_label});
        cur_label = m.label(l.header_label);
        for (const Entry& w : words) {
            const Id phi = m.emit(spv::OpPhi, w.kind == Kind::Float ? t_f32 : t_u32, {w.id, l.preheader, w.id, l.continue_label});
            reg[w.key] = w.kind == Kind::Float ? flt(phi) : word(phi);
            reg[w.key].helper_inexact = w.inexact;
            if (w.uni) uniform_ids.insert(phi);
            l.phis.push_back({w.key, w.kind, phi, w.uni, w.inexact});
            l.vgpr_phis += w.key >= 256;
            if (const auto it = partial.find(w.key); it != partial.end()) l.partial0[w.key] = it->second;
        }
        for (const Entry& k : masks) {
            const Id phi = m.emit(spv::OpPhi, t_bool, {k.id, l.preheader, k.id, l.continue_label});
            lanes[k.key] = new_content(lane(phi));
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
        // At the header EXEC lies within the loop mask: it started as EXEC,
        // and the back edge brings EXEC back within it.
        l.in_mask0 = in_mask;
        if (l.mask >= 0) in_mask = {kKeyExec, l.mask};
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
        // GCN evaluates a branch inside an s_cbranch_execz region only in the
        // iterations where some lane enters the region; the lift runs regions
        // unconditionally, so it would take the exit in the others too. Not a
        // divergent exit: a pixel takes it only where its own EXEC bit is set
        // in the region (its mask bit taken out), and in an iteration GCN skips
        // the region no pixel's is, so the pixel stays as GCN keeps it.
        if (open_regions.size() > l.regions0 && !l.mask_exits.count(in.offset)) {
            reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) + " from inside a region of its body (" +
                   hex_offset(open_regions.back()->start) + "), which GCN skips in iterations where no lane enters it");
            return;
        }
        Id leave = 0;  // where the branch is taken; 0 with `always`
        bool always = false, per_pixel = false;
        Val tested;  // per_pixel: the loop mask the exit tests, set wherever the lift stays in the loop
        if (in.op == 2) {
            always = true;
        } else if (in.op == 4 || in.op == 5) {
            const bool on_set = in.op == 5;  // s_cbranch_scc1 is taken where SCC is set
            const Val scc = known(scc_mask);
            if (!scc_valid) {
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) + " on an SCC an if left different on its arms");
                return;
            } else if (scc_uniform) {  // a scalar compare: the same bit in every lane
                if (scc.constant) {
                    if (scc.set != on_set) return;  // never taken
                    always = true;
                } else {
                    if (!draw_wide_exit(l, scc, in)) return;
                    leave = on_set ? scc.id : bnot(scc).id;
                }
            } else if (l.mask_exits.count(in.offset) && !scc.constant) {
                per_pixel = true;
                tested = scc;
                if (!divergent_exit(l, scc)) return;
                leave = bnot(scc).id;
                if (l.quad_live) leave = m.emit(spv::OpLogicalOr, t_bool, {leave, m.emit(spv::OpLogicalNot, t_bool, {l.quad_live})});
            } else if (!l.mask_exits.count(in.offset) && in_divergent_loop()) {
                // SCC is "some lane of the mask is set", over the whole wave;
                // the body knows the masks only for the pixel it runs.
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                       " on the SCC of a lane mask inside a loop pixels leave at different iterations, where the lift knows masks only for "
                       "the pixel it runs and GCN tests the whole wave");
                return;
            } else if (scc.constant) {
                // Set for every pixel: some lane of the wave is. Clear: only a
                // mask clear in every lane of the wave (Val::wave_exact) says no lane is.
                if (!scc.set && !wave_clear(scc)) {
                    reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                           " on the SCC of a lane mask clear for the lift's pixels, which GCN's helper or empty lanes may still have set");
                    return;
                }
                if (scc.set != on_set) return;  // never taken
                always = true;
            } else {
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                       " on an SCC that is neither a scalar condition nor the divergent loop's exit test");
                return;
            }
        } else if (in.op == 6 || in.op == 7) {
            ++res.wave.wave_tests;  // VCCZ tests all 64 bits; shown the same in every lane
            Val bit;
            std::uint32_t split = 0;
            if (!vcc_branch_bit(bit, split)) {
                reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) + " on a VCC that may differ between lanes");
                return;
            }
            bit = known(bit);
            const bool on_set = in.op == 7;  // s_cbranch_vccnz is taken where VCC is set
            if (bit.constant) {
                if (!bit.set && !wave_clear(bit)) {  // GCN tests all 64 bits of VCC
                    reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                           " on a VCC clear for the pixel the lift runs, which other lanes of GCN's wave may still have set");
                    return;
                }
                if (bit.set != on_set) return;
                always = true;
            } else {
                if (!draw_wide_exit(l, bit, in)) return;
                leave = on_set ? bit.id : bnot(bit).id;
            }
        } else if (in.op == 8 && l.mask == kKeyExec && l.mask_exits.count(in.offset)) {
            // A loop on EXEC's exit test: GCN leaves once EXEC is clear in
            // every lane; an invocation whose own EXEC bit is clear has left -
            // everything it would write from here on is masked off, and EXEC
            // stays within the lanes still in the loop (note_in_mask).
            tested = exec_lane();
            if (tested.constant) {
                if (tested.set) return;  // never taken by an invocation still in the loop
                reject("the loop on EXEC at " + hex_offset(l.header) + " tests an EXEC clear for every invocation");
                return;
            }
            per_pixel = true;
            if (!divergent_exit(l, tested)) return;
            leave = bnot(tested).id;
        } else {
            reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) +
                   ": only s_branch, s_cbranch_scc0/scc1 and s_cbranch_vccz/vccnz are lifted as loop exits");
            return;
        }
        Loop::Exit x;
        x.reg = reg;
        x.lanes = lanes;
        x.scc = scc_mask;
        x.scc_uniform = scc_uniform;
        x.scc_valid = scc_valid;
        x.poisoned = poisoned;
        x.partial = partial;
        x.per_pixel = per_pixel;
        x.in_mask = in_mask;
        // With several exits a register may leave as a float on one and a word
        // on another: its bits are made here, on the edge (a conversion made in
        // a block that dominates it is reused; one made here is not kept).
        const auto edge_words = [&] {
            if (l.exit_branches < 2) return;
            for (const auto& [k, v] : reg) {
                if (v.kind != Kind::Float || v.constant) continue;
                const auto it = to_word.find(v.id);
                x.words[k] = it != to_word.end() ? it->second : m.emit(spv::OpBitcast, t_u32, {v.id});
            }
        };
        if (always) {
            // It ends the arm it is in: an if's then arm, or its else arm.
            const std::uint32_t next = in.offset + 4;
            const bool ends_arm = arms.size() > l.arms0 && (next == arms.back().c->else_jump || next == arms.back().c->join);
            if (!ends_arm) {
                reject("an unconditional exit from the loop at " + hex_offset(l.header) + " that does not end an if's arm");
                return;
            }
            x.from = cur_label;
            edge_words();
            m.emit_void(spv::OpBranch, {l.merge_label});
            block_dead = true;
        } else {
            const Id brk = m.fresh(), cont = m.fresh();
            m.emit_void(spv::OpSelectionMerge, {cont, 0u});
            m.emit_void(spv::OpBranchConditional, {leave, brk, cont});
            x.from = m.label(brk);
            edge_words();
            m.emit_void(spv::OpBranch, {l.merge_label});
            cur_label = m.label(cont);
        }
        l.exits.push_back(std::move(x));
        if (per_pixel) {
            const Val mk = tested;
            l.assumed.push_back(mk.id);
            if (const auto it = conjuncts.find(mk.id); it != conjuncts.end()) l.assumed.insert(l.assumed.end(), it->second.begin(), it->second.end());
            for (const Id a : l.assumed) assumed.insert(a);
            // A loop on EXEC: past the test the loop's lanes are EXEC's, and
            // no other mask is known to lie within them.
            if (l.mask == kKeyExec) in_mask = {kKeyExec};
            // Taken on a bit every lane holds alike, every invocation leaves at
            // the same iteration: the loop keeps its invocations together.
            if (!uniform(tested)) l.divergent_exit = true;
        }
    }
    // A uniform loop inside a region runs for every pixel, also in the waves
    // where GCN skips the region: its trip count must then be one GCN would run
    // too, from values every wave of the draw holds alike (user data, scalar
    // loads, constants) - the count of any wave that enters the region.
    bool draw_wide_exit(const Loop& l, const Val& bit, const Inst& in) {
        if (!l.region || (uniform(bit) && (bit.constant || !wg_ids.count(bit.id)))) return true;
        reject(std::string(mnemonic(in)) + " leaves the loop at " + hex_offset(l.header) + ", inside the region at " + hex_offset(l.region) +
               ", on a bit not computed from values every wave of the draw holds alike: the lift runs it also where GCN skips the region");
        return false;
    }
    // The divergent exit test: at the loop's top level, on the mask's current
    // value, which lies within EXEC at the header.
    bool divergent_exit(const Loop& l, const Val& m_val) {
        if (arms.size() != l.arms0) {
            reject("the divergent exit of the loop at " + hex_offset(l.header) + " is inside an if of its body");
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
        if (l.mask >= 0 && std::none_of(l.exits.begin(), l.exits.end(), [](const Loop::Exit& x) { return x.per_pixel; })) {
            reject("the divergent loop at " + hex_offset(l.header) + " was not left at its exit test");
            return;
        }
        // What the header took to lie within the loop mask (in_mask) must come back so.
        if (l.mask >= 0) {
            for (const int k : {kKeyExec, l.mask}) {
                if (in_mask.count(k)) continue;
                reject(key_name(k) + " comes back to the header of the loop at " + hex_offset(l.header) + " with lanes beyond its loop mask");
                return;
            }
        } else if (in_divergent_loop()) {
            for (const int k : l.in_mask0) {
                if (!l.writes.count(k) || in_mask.count(k)) continue;
                reject(key_name(k) + " comes back to the header of the loop at " + hex_offset(l.header) +
                       " with lanes beyond the mask of the loop pixels leave at different iterations around it");
                return;
            }
        }
        std::vector<std::pair<Id, Id>> back;  // phi, value
        // What a second attempt can assume instead is recorded for every
        // register here before the first of them rejects, so one more attempt
        // takes them all (a loop of 25 VGPRs ran out of attempts one at a time).
        std::string retry_rejection;
        // What the header assumed of the helper lanes must hold for the value
        // that comes back; if not, a second attempt assumes less.
        const auto helper_lanes_kept = [&](int key, bool at_header, const Val& v) {
            if (!v.helper_inexact || at_header) return;
            more_inexact.insert({l.header, key});
            if (retry_rejection.empty()) {
                retry_rejection = key_name(key) + " comes back to the header of the loop at " + hex_offset(l.header) +
                                  " with helper lanes GCN may hold differently, which the header did not assume";
            }
        };
        for (const Loop::Phi& p : l.phis) {
            if (p.kind == Kind::Lane) {
                const auto it = lanes.find(p.key);
                if (it == lanes.end() || poisoned.count(p.key)) {
                    reject(key_name(p.key) + " leaves the header of the loop at " + hex_offset(l.header) + " as a lane mask and comes back as something else");
                    return;
                }
                const Val v = known(it->second);
                helper_lanes_kept(p.key, p.helper_inexact, v);
                back.push_back({p.id, v.constant ? m.const_bool(v.set) : v.id});
                continue;
            }
            if (poisoned.count(p.key) || lanes.count(p.key) || (p.key > 0 && p.key < 104 && lanes.count(p.key - 1))) {
                reject(key_name(p.key) + " leaves the header of the loop at " + hex_offset(l.header) + " as a word and comes back as a lane mask or unknown");
                return;
            }
            const Val v = value(p.key);
            if (p.uniform && !uniform(v)) {
                if (compute() && p.key >= 256) {
                    more_nonuniform.insert({l.header, p.key});  // another attempt, with this one not alike at the header
                } else {
                    reject(key_name(p.key) + " comes back to the header of the loop at " + hex_offset(l.header) + " with a value lanes may not hold alike");
                    return;
                }
            }
            helper_lanes_kept(p.key, p.helper_inexact, v);
            // Where it holds GCN's value: at least where the header assumed.
            const auto at_header = l.partial0.find(p.key);
            const Where header = at_header == l.partial0.end() ? Where{} : Where{true, at_header->second}, now = where_reg(p.key);
            if (now.partial && (!header.partial || !within(header.at, now.at))) {
                reject(key_name(p.key) + " comes back to the header of the loop at " + hex_offset(l.header) +
                       " holding GCN's value in fewer pixels than it left with (written in whole-quad mode)");
                return;
            }
            back.push_back({p.id, p.kind == Kind::Float ? as_f(v) : as_u(v)});
        }
        // (In a divergent loop's body EXEC at entry is known set, so EXEC and
        // the mask must come back known set.)
        for (const auto& [k, entry] : l.unchanged) {
            const auto it = lanes.find(k);
            if (it == lanes.end() || !same_lane(known(it->second), known(entry))) {
                const std::string changed = key_name(k) + " comes back to the header of the loop at " + hex_offset(l.header) + " changed";
                if (k != kKeyExec && k != l.mask && it != lanes.end()) {
                    more_lane_phis.insert({l.header, k});  // a phi on the next attempt
                    if (retry_rejection.empty()) retry_rejection = changed;
                    continue;
                }
                reject(changed);
                return;
            }
            helper_lanes_kept(k, entry.helper_inexact, it->second);
            // ... and so must what the header took of the lanes the lift does not run.
            const bool per_pixel_header = l.mask >= 0 || in_divergent_loop();
            if ((entry.helpers_clear && !it->second.helpers_clear) || (entry.wave_exact && !it->second.wave_exact) ||
                (!per_pixel_header && entry.content && entry.content != it->second.content)) {
                reject(key_name(k) + " comes back to the header of the loop at " + hex_offset(l.header) +
                       " without what the header took of the lanes no pixel of the lift runs (clear in GCN's helper lanes, or the same in "
                       "every lane of the wave)");
                return;
            }
        }
        if (!retry_rejection.empty()) {
            reject(retry_rejection);
            return;
        }
        m.emit_void(spv::OpBranch, {l.continue_label});
        m.label(l.continue_label);
        m.emit_void(spv::OpBranch, {l.header_label});
        for (const auto& [phi, v] : back) m.set_operand(phi, 2, v);
        block_dead = true;  // the merge block starts at the exit, next
    }

    // Several exits join at the merge block: a value the exits leave different
    // becomes a phi over their edges (each pixel takes one); a lane mask
    // present on some exits only is lost, as at an if's join; the registers
    // are exact where they are on every exit, SCC is kept only if every exit
    // left the same, and what lies within a mask is what does on every exit.
    // A register the exits leave as different kinds (word and float) is not
    // joined: the conversion would have to be on each edge.
    bool merge_exits(const Loop& l) {
        const auto same = [](const Val& x, const Val& y) {
            return x.kind == y.kind && x.constant == y.constant && (x.constant ? (x.kind == Kind::Lane ? x.set == y.set : x.bits == y.bits) : x.id == y.id);
        };
        std::set<int> lost;
        for (const Loop::Exit& x : l.exits) lost.insert(x.poisoned.begin(), x.poisoned.end());
        std::set<int> lane_keys;
        for (const Loop::Exit& x : l.exits) {
            for (const auto& [k, v] : x.lanes) lane_keys.insert(k);
        }
        std::map<int, Val> merged_lanes;
        for (const int k : lane_keys) {
            bool everywhere = true;
            for (const Loop::Exit& x : l.exits) everywhere = everywhere && x.lanes.count(k);
            if (!everywhere) {
                if (k != kKeyVcc && k != kKeyExec) {
                    lost.insert(k);
                    lost.insert(k + 1);
                }
                continue;
            }
            const Val& first = l.exits[0].lanes.at(k);
            bool all_same = true, inexact = false;
            for (const Loop::Exit& x : l.exits) {
                all_same = all_same && same(x.lanes.at(k), first);
                inexact = inexact || x.lanes.at(k).helper_inexact;
            }
            if (all_same) {
                Val v = first;
                for (const Loop::Exit& x : l.exits) {
                    v.wave_exact = v.wave_exact && x.lanes.at(k).wave_exact;
                    v.helpers_clear = v.helpers_clear && x.lanes.at(k).helpers_clear;
                    if (x.lanes.at(k).content != first.content) v = new_content(v);
                }
                v.helper_inexact = inexact;
                merged_lanes[k] = v;
                continue;
            }
            std::vector<std::uint32_t> ops;
            for (const Loop::Exit& x : l.exits) {
                const Val& v = x.lanes.at(k);
                ops.push_back(v.constant ? m.const_bool(v.set) : v.id);
                ops.push_back(x.from);
            }
            Val v = new_content(lane(m.emit(spv::OpPhi, t_bool, ops)));
            v.helper_inexact = inexact;
            merged_lanes[k] = v;
        }
        std::set<int> keys;
        for (const Loop::Exit& x : l.exits) {
            for (const auto& [k, v] : x.reg) keys.insert(k);
        }
        std::map<int, Val> merged;
        for (const int k : keys) {
            if (lane_keys.count(k) || (k > 0 && k < 104 && lane_keys.count(k - 1)) || lost.count(k)) continue;
            std::vector<Val> vals;
            for (const Loop::Exit& x : l.exits) {
                const auto it = x.reg.find(k);
                vals.push_back(it != x.reg.end() ? it->second : initial_value(k));
            }
            bool all_same = true, inexact = false, any_descriptor = false;
            for (const Val& v : vals) {
                all_same = all_same && same(v, vals[0]);
                inexact = inexact || v.helper_inexact;
                any_descriptor = any_descriptor || descriptor(v);
            }
            if (all_same) {
                merged[k] = vals[0];
                merged[k].helper_inexact = inexact;
                continue;
            }
            if (std::any_of(vals.begin(), vals.end(), [](const Val& v) { return v.wave_part != 0; })) {
                reject(key_name(k) + " holds a part of the append idiom on an exit of the loop at " + hex_offset(l.header));
                return false;
            }
            if (any_descriptor) {  // a descriptor word on any exit: still never data
                for (const Val& v : vals) {
                    if (descriptor(v)) merged[k] = v;
                }
                continue;
            }
            bool all_float = true;
            for (const Val& v : vals) all_float = all_float && (v.kind == Kind::Float || v.constant);
            const Kind kind = all_float && vals[0].kind == Kind::Float ? Kind::Float : Kind::Word;
            std::vector<std::uint32_t> ops;
            for (std::size_t e = 0; e < vals.size(); ++e) {
                const Val& v = vals[e];
                Id op = 0;
                if (v.constant) {
                    op = kind == Kind::Float ? as_f(v) : as_u(v);  // a constant: no instruction on the edge
                } else if (kind == Kind::Float || v.kind == Kind::Word) {
                    op = v.id;
                } else if (const auto w = l.exits[e].words.find(k); w != l.exits[e].words.end()) {
                    op = w->second;
                } else {
                    reject(key_name(k) + " leaves the loop at " + hex_offset(l.header) + " as a word on one exit and a float on another");
                    return false;
                }
                ops.push_back(op);
                ops.push_back(l.exits[e].from);
            }
            const Id id = m.emit(spv::OpPhi, kind == Kind::Float ? t_f32 : t_u32, ops);
            merged[k] = kind == Kind::Float ? flt(id) : word(id);
            merged[k].helper_inexact = inexact;
        }
        // SCC: as every exit left it, or unknown.
        bool scc_same = true;
        for (const Loop::Exit& x : l.exits) {
            scc_same = scc_same && x.scc_valid && x.scc_uniform == l.exits[0].scc_uniform && same(x.scc, l.exits[0].scc);
        }
        scc_mask = l.exits[0].scc;
        scc_uniform = l.exits[0].scc_uniform;
        scc_valid = scc_same;
        // Exact where it is on every exit.
        std::map<int, Exact> joined = l.exits[0].partial;
        for (std::size_t e = 1; e < l.exits.size(); ++e) {
            const std::map<int, Exact>& other = l.exits[e].partial;
            std::set<int> pk;
            for (const auto& kv : joined) pk.insert(kv.first);
            for (const auto& kv : other) pk.insert(kv.first);
            std::map<int, Exact> next;
            for (const int k : pk) {
                const auto a = joined.find(k);
                const auto b = other.find(k);
                const Where w = meet(a == joined.end() ? Where{} : Where{true, a->second}, b == other.end() ? Where{} : Where{true, b->second});
                if (w.partial) next[k] = w.at;
            }
            joined = std::move(next);
        }
        std::set<int> within = l.exits[0].in_mask;
        for (const Loop::Exit& x : l.exits) {
            for (auto it = within.begin(); it != within.end();) it = x.in_mask.count(*it) ? std::next(it) : within.erase(it);
        }
        reg = std::move(merged);
        lanes = std::move(merged_lanes);
        poisoned = std::move(lost);
        partial = std::move(joined);
        in_mask = std::move(within);
        return true;
    }

    std::string mask_exits_text(const Loop& l) const {
        std::string t;
        for (const std::uint32_t e : l.mask_exits) {
            t += (t.empty() ? "" : ", ") + hex_offset(e);
            const auto r = std::find_if(regions.begin(), regions.end(), [&](const Region& g) { return g.branch < e && e < g.end && g.branch > l.header; });
            if (r != regions.end()) t += " (in the region at " + hex_offset(r->branch) + ")";
        }
        return t;
    }

    // The exit's merge block, at the loop's exit.
    void close_loop() {
        Loop& l = *open_loops.back();
        open_loops.pop_back();
        cur = l.exit;
        if (!l.converged_op.empty() && l.divergent_exit) {
            reject(l.converged_op + ", inside the loop at " + hex_offset(l.header) +
                   ", which invocations leave at different iterations: it needs every invocation that entered the loop");
            return;
        }
        for (const Id a : l.assumed) assumed.erase(a);
        if (l.exits.empty()) {
            reject("the loop at " + hex_offset(l.header) + " has no exit");
            return;
        }
        cur_label = m.label(l.merge_label);
        block_dead = false;
        if (l.exits.size() == 1) {
            const Loop::Exit& x = l.exits[0];
            reg = x.reg;
            lanes = x.lanes;
            scc_mask = x.scc;
            scc_uniform = x.scc_uniform;
            scc_valid = x.scc_valid;
            poisoned = x.poisoned;
            partial = x.partial;
            in_mask = x.in_mask;
        } else if (!merge_exits(l)) {
            return;
        }
        to_float = l.to_float0;
        to_word = l.to_word0;
        pending.clear();
        if (l.mask >= 0) {  // what it writes is dead after it (checked below); the rest is as it was
            in_mask = l.in_mask0;
            for (const int k : l.writes) in_mask.erase(k);
        }
        if (l.mask >= 0) {
            // GCN may run more iterations than this pixel: nothing scalar the
            // loop writes may be read after it before it is written again.
            std::uint32_t at = 0;
            int key = 0;
            bool unmodelled = false;
            if (read_before_written(index_of(l.exit), l.writes, kVectorReadsExec, at, key, unmodelled)) {
                cur = at;
                reject(unmodelled ? "cannot show what the divergent loop at " + hex_offset(l.header) + " writes is dead after it"
                                  : key_name(key) + " is written inside the divergent loop at " + hex_offset(l.header) +
                                        " and read after it (GCN may run more iterations than the pixel)");
                return;
            }
            ++divergent_loops;
            ++res.wave.lane_exits;
        }
        ++loops_lifted;
        std::string kept;
        for (const auto& [k, v] : l.unchanged) kept += (kept.empty() ? "" : ", ") + key_name(k);
        res.proof.push_back(
            "loop " + hex_offset(l.header) + "-" + hex_offset(l.latch) + ": " +
            (l.mask == kKeyExec ? "on EXEC, left at " + mask_exits_text(l) +
                                      " (s_cbranch_execz: GCN leaves once no lane is left, an invocation once its own EXEC bit is clear - its "
                                      "writes are masked from there and EXEC stays within the loop's lanes; what the loop writes of scalar "
                                      "state is dead after it; no subgroup operation inside)"
             : l.mask >= 0 ? "divergent, the per-lane exit idiom on " + key_name(l.mask) + " at " + mask_exits_text(l) +
                               " (EXEC and the mask keep EXEC at entry at the header and come back set; EXEC lies within the mask wherever "
                               "it writes a VGPR and at the back edge; what the loop writes is dead after it; no implicit-LOD sample inside)" +
                               (l.quad_live ? "; inside the kill region at " + hex_offset(l.quad_region) +
                                                  ", widened from EXEC known set, the pixels of quads without a covered pixel the guard keeps "
                                                  "leave at the first exit test, as on GCN, where their EXEC is clear"
                                            : std::string())
                         : std::string("uniform, left on a bit every lane holds alike") +
                               (l.region ? " (inside the region at " + hex_offset(l.region) +
                                               ", which GCN skips in a wave no pixel enters: the lift runs it in every wave, its trip count "
                                               "computed only from values every wave of the draw holds alike, the count GCN runs in any wave "
                                               "that enters it)"
                                         : std::string())) +
            "; " + std::to_string(l.phis.size()) + " phis (" + std::to_string(l.vgpr_phis) + " VGPRs, " + std::to_string(l.lane_phis) + " lane masks)" +
            (kept.empty() ? "" : "; back at the header unchanged: " + kept) +
            (l.exits.size() > 1 ? "; " + std::to_string(l.exits.size()) + " exits joined at " + hex_offset(l.exit) +
                                      ", where a value they leave different is a phi over their edges"
                                : std::string()) +
            (l.dead ? "; " + hex_offset(l.dead) + "-" + hex_offset(l.exit) +
                          " after the back edge is reached by no branch and is not lifted"
                    : std::string()));
    }

    // TranslateOptions::kVsOutCntlClipVarying, as translate.cpp's prologue: the
    // clipper's work, per pixel. A pixel where any of the distances the vertex
    // stage wrote at the clip location is negative is discarded before anything
    // runs, where the clipper would have cut it out of the triangle; the others
    // run the program as they would have.
    bool clip_discard = false;
    void clip_discard_prologue() {
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
        cur_label = m.label(l_on);
        clip_discard = true;
        ++res.wave.discards;
    }

    // ---- module ----------------------------------------------------------------------
    void emit() {
        m.capability(spv::CapShader);
        m.capability(spv::CapInt64);
        if (compute()) {
            // Guest memory through physical storage buffer pointers, as the
            // reference reaches it.
            m.capability(spv::CapPhysicalStorageBufferAddresses);
            m.extension("SPV_KHR_physical_storage_buffer");
            m.memory_model(5348 /* PhysicalStorageBuffer64 */, 1 /* GLSL450 */);
        } else {
            m.memory_model(0 /* Logical */, 1 /* GLSL450 */);
        }
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
        // uvec4 cb_stride[4]; uvec4 cb_w3[4] }, then a vertex stage's
        // uvec4 vertex_formats[4] when it reads its formats from them.
        const Id t_user_arr = m.type_array(t_v4u, cu(4));
        m.decorate(t_user_arr, spv::DecArrayStride, {16});
        const Id t_bias_arr = t_user_arr;  // the same uvec4[4] type (types are deduplicated): decorated once
        // With the reference bindless (TranslateOptions::bindless), the image
        // and sampler slots follow, at StageParams's own offsets.
        std::vector<Id> members = {t_u64, t_u64, t_user_arr, t_u32, t_bias_arr, t_bias_arr, t_bias_arr};
        // Member 7 (StageParams::vertex_formats): the elements' descriptors, or
        // the LS's user data.
        const bool vertex_formats = (opt.stage == Stage::Vertex && opt.vertex_formats_from_params && !opt.vertex_input.empty()) ||
                                    ((tess_ls() || tess_hs()) && !tess->own_params);
        if (vertex_formats) members.push_back(t_bias_arr);
        if (compute()) {
            // Member 3 as the uvec4 { cb_valid, lds_bytes, gds_base, pad }, and
            // member 7 StageParams::gds_counter (eight u64 guest addresses) when
            // the program counts in GDS, as the reference declares them.
            members[3] = t_v4u;
            if (opt.gds_counters && std::any_of(prog.insts.begin(), prog.insts.end(), [](const Inst& in) {
                    return in.enc == Enc::DS && in.gds && (in.op == 61 || in.op == 62);
                })) {
                gds_member = static_cast<std::uint32_t>(members.size());
                members.push_back(t_bias_arr);
            }
        }
        if (ref.bindless) {
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
        if (vertex_formats) m.member_decorate(t_ubo, 7, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, vertex_formats))});
        if (gds_member) m.member_decorate(t_ubo, gds_member, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, gds_counter))});
        if (ref.bindless) {
            m.member_decorate(t_ubo, bindless_image_member, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, image_index))});
            m.member_decorate(t_ubo, bindless_sampler_member, spv::DecOffset, {static_cast<std::uint32_t>(offsetof(StageParams, sampler_index))});
        }
        ubo_var = m.global_variable(m.type_pointer(spv::ScUniform, t_ubo), spv::ScUniform);
        m.decorate(ubo_var, spv::DecDescriptorSet, {opt.descriptor_set});
        m.decorate(ubo_var, spv::DecBinding, {tess && tess->own_params ? tess->params_binding : kBindingParams});
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
            if (b.storage && compute()) {  // as the reference declares a storage image (translate.cpp image_for)
                m.capability(spv::CapStorageImageReadWithoutFormat);
                m.capability(spv::CapStorageImageWriteWithoutFormat);
            }
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

        if ((tess_ls() || tess_ds()) && !(window() && tess_ds())) {
            // As the reference declares it (translate.cpp lds_in_attributes):
            // the LS's output, the evaluation stage's per-vertex input.
            const Id t_slots = m.type_array(t_v4u, cu(std::max<std::uint32_t>(1, tess->attr_vec4s)));
            if (tess_ls()) {
                control_point = m.global_variable(m.type_pointer(spv::ScOutput, t_slots), spv::ScOutput);
            } else {
                const Id t_in = m.type_array(t_slots, cu(std::max<std::uint32_t>(1, opt.tess_patch_control_points)));
                control_point = m.global_variable(m.type_pointer(spv::ScInput, t_in), spv::ScInput);
            }
            m.decorate(control_point, spv::DecLocation, {0});
            m.name(control_point, "control_point");
            interface.push_back(control_point);
        }
        if (window() && (tess_hs() || tess_ds())) {
            // TessLift::window: the hull reads the LS's control points and
            // writes its own and the patch constants; the domain shader reads
            // those two.
            const Window w = window_layout();
            const Id t_cp = m.type_array(m.type_array(t_v4u, cu(w.cp / 16)), cu(w.n));
            const spv::StorageClass sc = tess_hs() ? spv::ScOutput : spv::ScInput;
            if (tess_hs()) {
                control_point = m.global_variable(m.type_pointer(spv::ScInput, m.type_array(m.type_array(t_v4u, cu(w.ls / 16)), cu(w.n))), spv::ScInput);
                m.decorate(control_point, spv::DecLocation, {0});
                m.name(control_point, "control_point");
                interface.push_back(control_point);
                cp_out = m.global_variable(m.type_pointer(spv::ScOutput, t_cp), spv::ScOutput);
                m.decorate(cp_out, spv::DecLocation, {0});
                m.name(cp_out, "control_point_out");
                interface.push_back(cp_out);
                const Id t_outer = m.type_array(t_f32, cu(4)), t_inner = m.type_array(t_f32, cu(2));
                out_tess_outer = m.global_variable(m.type_pointer(spv::ScOutput, t_outer), spv::ScOutput);
                out_tess_inner = m.global_variable(m.type_pointer(spv::ScOutput, t_inner), spv::ScOutput);
                m.decorate(out_tess_outer, spv::DecBuiltIn, {spv::BiTessLevelOuter});
                m.decorate(out_tess_inner, spv::DecBuiltIn, {spv::BiTessLevelInner});
                m.decorate(out_tess_outer, spv::DecPatch);
                m.decorate(out_tess_inner, spv::DecPatch);
                m.name(out_tess_outer, "tess_level_outer");
                m.name(out_tess_inner, "tess_level_inner");
                interface.push_back(out_tess_outer);
                interface.push_back(out_tess_inner);
                in_invocation_id = builtin(p_in_u32, spv::BiInvocationId);
            } else {
                control_point = m.global_variable(m.type_pointer(spv::ScInput, t_cp), spv::ScInput);
                m.decorate(control_point, spv::DecLocation, {0});
                m.name(control_point, "control_point");
                interface.push_back(control_point);
            }
            if (w.npc) {
                patch_io = m.global_variable(m.type_pointer(sc, m.type_array(t_v4u, cu(w.npc))), sc);
                m.decorate(patch_io, spv::DecLocation, {w.cp / 16});
                m.decorate(patch_io, spv::DecPatch);
                m.name(patch_io, "patch_constant");
                interface.push_back(patch_io);
            }
        }

        if (compute()) declare_guest_memory();
        fn_main = m.begin_function(t_void, m.type_function(t_void, {}));
        m.name(fn_main, "main");
        cur_label = m.label();
        if (opt.stage == Stage::Pixel && opt.ps_clip_discard) clip_discard_prologue();
        lanes[kKeyExec] = lane_const(true);   // every running pixel starts with its bit set
        // ... and GCN's helper lanes with theirs clear, until whole-quad mode.
        // A vertex shader has none (it takes no whole-quad mode, sop1).
        lanes[kKeyExec].helper_inexact = opt.stage == Stage::Pixel;
        lanes[kKeyExec].helpers_clear = true;
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
                    if (res.rejections.empty() && done->branched) region_branch_close(*done);
                } else {
                    close_arm();
                }
                if (!res.rejections.empty()) return;
            }
            // Between a back edge and an exit past it: instructions nothing reaches (find_loops).
            if (!open_loops.empty() && open_loops.back()->dead && in.offset >= open_loops.back()->dead && in.offset < open_loops.back()->exit) {
                char line[160];
                std::snprintf(line, sizeof(line), "%06x  %-56s", in.offset, format(in).c_str());
                res.listing += line;
                res.listing += " [unreached]\n";
                continue;
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
                    // A branched region's selection closes just after it
                    // (close_region sees the body's end, then the merge).
                    const bool branch = branch_region(r);
                    const std::uint64_t arm_seq = branch ? ++seq : 0;
                    r.seq = ++seq;
                    r.in_mask0 = in_mask;
                    open_regions.push_back(&r);
                    if (lift_branch_on() && opt.stage == Stage::Pixel && open_loops.empty() && arms.empty() && region_branchable(r))
                        region_branch_open(r);
                    if (branch) open_region_arm(r, arm_seq);
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
            inst_wg = false;
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
                if (res.rejections.empty() && done->branched) region_branch_close(*done);
            } else if (!arms.empty()) {
                reject("an if is still open at the end of the program");
            } else if (!open_loops.empty()) {
                reject("a loop is still open at the end of the program");
            } else {
                break;
            }
            if (!res.rejections.empty()) return;
        }
        if (vertex_like() && (opt.vs_out_cntl & TranslateOptions::kVsOutCntlClipVarying) &&
            !out_params.count(static_cast<int>(TranslateOptions::vs_clip_location(opt.vs_out_cntl)))) {
            // translate.cpp's end of main: no distance vector exported, none discards.
            m.store(out_param(static_cast<int>(TranslateOptions::vs_clip_location(opt.vs_out_cntl))),
                    m.emit(spv::OpCompositeConstruct, t_v4f, {cf(1.0f), cf(1.0f), cf(1.0f), cf(1.0f)}));
        }
        if (needs_kill) ++res.wave.discards;
        res.wave.execz_skips = static_cast<std::uint32_t>(std::count_if(regions.begin(), regions.end(), [](const Region& r) { return !r.kill && !r.disabled; }));
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
        if (tess_hs()) {
            if (tf_stores != (opt.tess_quads ? 6u : 4u)) {
                cur = prog.insts.back().offset;
                reject(std::to_string(tf_stores) + " factor stores, not the levels a " + (opt.tess_quads ? "quad" : "triangle") + " patch has");
                return;
            }
            m.capability(spv::CapTessellation);
            m.entry_point(spv::EmTessellationControl, fn_main, "main", interface);
            m.execution_mode(fn_main, spv::ExOutputVertices, {window_layout().n});
            if (native_rtz) {
                m.execution_mode(fn_main, spv::ExDenormPreserve, {16});
                m.execution_mode(fn_main, spv::ExRoundingModeRTZ, {16});
            }
            return;
        }
        if (vertex_like()) {
            // The pipeline, the pixel shader's inputs and a rect list's geometry
            // shader are laid out from the reference's outputs: the lift must
            // declare exactly those. The reference reports a vertex stage's;
            // an evaluation stage's are read from its module.
            std::vector<std::uint32_t> params;
            for (const auto& kv : out_params) params.push_back(static_cast<std::uint32_t>(kv.first));
            std::vector<std::uint32_t> ref_params = ref.vs_params;
            int ref_clip = ref.vs_clip_count;
            bool ref_point_size = ref.vs_point_size;
            if (opt.stage == Stage::TessEval) module_outputs(ref.spirv, ref_params, ref_clip, ref_point_size);
            if (params != ref_params || (out_clip ? clip_count : 0) != ref_clip || (out_point_size != 0) != ref_point_size) {
                cur = prog.insts.back().offset;
                reject("the lifted outputs differ from the reference's (params, clip distances or point size)");
                return;
            }
            if (opt.stage == Stage::TessEval) {
                // VGT_TF_PARAM's domain, partitioning and topology, as the reference declares them.
                m.capability(spv::CapTessellation);
                m.entry_point(spv::EmTessellationEvaluation, fn_main, "main", interface);
                m.execution_mode(fn_main, opt.tess_quads ? spv::ExQuads : spv::ExTriangles);
                m.execution_mode(fn_main, opt.tess_spacing == 1   ? spv::ExSpacingFractionalOdd
                                          : opt.tess_spacing == 2 ? spv::ExSpacingFractionalEven
                                                                  : spv::ExSpacingEqual);
                m.execution_mode(fn_main, opt.tess_cw ? spv::ExVertexOrderCw : spv::ExVertexOrderCcw);
                if (native_rtz) {
                    m.execution_mode(fn_main, spv::ExDenormPreserve, {16});
                    m.execution_mode(fn_main, spv::ExRoundingModeRTZ, {16});
                }
                return;
            }
            m.entry_point(spv::EmVertex, fn_main, "main", interface);
            if (native_rtz) {
                m.execution_mode(fn_main, spv::ExDenormPreserve, {16});
                m.execution_mode(fn_main, spv::ExRoundingModeRTZ, {16});
            }
            return;
        }
        if (compute()) {
            m.entry_point(spv::EmGLCompute, fn_main, "main", interface);
            m.execution_mode(fn_main, spv::ExLocalSize, {std::max<std::uint32_t>(1, opt.cs_threads[0]), std::max<std::uint32_t>(1, opt.cs_threads[1]),
                                                         std::max<std::uint32_t>(1, opt.cs_threads[2])});
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
        if (opt.early_fragment_tests && !opt.ps_clip_discard && !out_frag_depth) m.execution_mode(fn_main, spv::ExEarlyFragmentTests);
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
                       "invocations stand in for whole-quad lanes):" + at);
        if (!varying_samples.empty() || !lod_queries.empty()) {
            std::string varying, queries;
            for (std::uint32_t s : varying_samples) varying += " " + hex_offset(s);
            for (std::uint32_t s : lod_queries) queries += " " + hex_offset(s);
            head.push_back(std::to_string(varying_samples.size()) + " image samples under a varying EXEC" + (varying.empty() ? "" : ":" + varying) +
                           "; " + std::to_string(lod_queries.size()) + " image_get_lod" + (queries.empty() ? "" : ":" + queries) +
                           "; in uniform control flow, results written under EXEC, operands GCN's in every pixel of EXEC and the "
                           "coordinates of implicit derivatives in every pixel of the quads of EXEC");
            if (!lod_queries.empty()) {
                head.push_back("image_get_lod reads only the unclamped LOD: the host's OpImageQueryLod lambda', as the translator's does, which "
                               "follows the host's own LOD computation (its precision and anisotropy, a footprint of no size) and is not shown "
                               "equal to GCN's");
            }
        }
        if (!divergent_samples.empty()) {
            head.push_back(std::to_string(divergent_samples.size()) + " of these samples inside a loop pixels leave at different iterations, "
                           "where control flow is not uniform: with explicit LOD or gradients only:" + divergent_at);
        }
        head.push_back(std::to_string(exports.size()) + " exports with EXEC set");
        if (image_loads) head.push_back(std::to_string(image_loads) + " image loads at integer coordinates, which take neither a sampler nor derivatives");
        head.push_back(std::to_string(buffer_loads) + " scalar loads, every one from a storage buffer the reference binds" +
                       (ranged_loads ? " (" + std::to_string(ranged_loads) + " at a run-time offset, each dword checked against the V#'s range "
                                       "as GCN checks it)"
                                     : std::string()) +
                       "; " + std::to_string(masked_writes) + " VGPR writes under a varying EXEC become selects" +
                       (spill_reuses ? "; " + std::to_string(spill_reuses) + " over a VGPR whose lanes held spilled words, exact only where EXEC "
                                                                              "was set (no read relies on it elsewhere)"
                                     : std::string()));
        if (buffer_stores) {
            head.push_back(std::to_string(buffer_stores) + " buffer stores, every one through a binding the reference writes, each dword where "
                           "it is inside the V#'s range and this invocation's EXEC bit is set, outside loops");
        }
        if (clip_discard) {
            head.push_back("a pixel whose clip distances (the vertex stage's, at location " + std::to_string(opt.ps_clip_location()) +
                           ") are negative is discarded at entry, by the translator's own prologue");
        }
        if (param_format_elements) {
            head.push_back(std::to_string(param_format_elements) + " vertex elements converted by their descriptor - the relink's "
                           "specialization constant, else the params block's - with the translator's expression for each kind");
        }
        if (swizzles) {
            const std::string where = block_swizzles ? "known set in whole-quad mode or (" + std::to_string(block_swizzles) +
                                                           " of them) in a whole-quad block over a varying EXEC, of a source GCN's in the block's quads"
                                                     : std::string("known set in whole-quad mode");
            head.push_back(std::to_string(swizzles) + " ds_swizzle_b32 within a quad as quad operations in uniform control flow: each under EXEC " +
                           where + ", of a value the quad's helper lanes computed as on GCN, and no kill region where pixels may differ before it");
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
        if (compute()) {
            char wg[64];
            std::snprintf(wg, sizeof(wg), "%ux%ux%u", opt.cs_threads[0], opt.cs_threads[1], opt.cs_threads[2]);
            head.push_back(std::string("a dispatch's compute shader: one invocation a lane of GCN's wave (workgroup ") + wg +
                           "), the workgroup id and local id where the reference places them; no workgroup memory (LDS)");
            if (cs_bound_loads || cs_bound_stores || cs_bound_elements) {
                head.push_back("the lean reference's bindings (ComputeLean): " + std::to_string(cs_bound_loads) +
                               " scalar and typed loads read from the storage buffer the reference binds at that instruction, " +
                               std::to_string(cs_bound_stores) +
                               " buffer stores through its written binding, each dword inside the V#'s range in dwords and where the invocation's "
                               "EXEC bit is set, " + std::to_string(cs_bound_elements) +
                               " elements at a VGPR index or offset (typed loads and stores, format loads, atomic adds) in their binding, an index "
                               "past num_records or a dword past the range read as 0 and not written");
            }
            head.push_back("guest memory through the page table at StageParams::l1_table, as the reference reaches it: " +
                           std::to_string(buffer_loads - cs_bound_loads) + " scalar loads, " + std::to_string(guest_loads) +
                           " buffer loads (dwords in one page with one walk), " +
                           std::to_string(guest_stores) + " buffer stores and " + std::to_string(guest_atomics) + " atomics and " +
                           std::to_string(image_stores) +
                           " image stores where the invocation's EXEC bit is set; a load every lane makes at one address is one value for "
                           "them all (no store of the dispatch races with it)");
            if (gds_lane_ops || wave_ranks || subgroup_ranks || ranks_folded || first_lane_reads) {
                head.push_back(std::to_string(gds_lane_ops) + " ds_append/ds_consume steps in GDS each invocation's own (the counter ends the "
                               "same), " + std::to_string(wave_slots) + " of their wave bases added to the lane's rank as the invocation's slot "
                               "(the same slots, in another order, as GCN's waves land in none); " + std::to_string(wave_ranks) +
                               " ranks folded so, " + std::to_string(subgroup_ranks) + " kept as subgroup operations with every invocation "
                               "together, " + std::to_string(ranks_folded) + " high halves 0 (no lane 32 or above); " +
                               std::to_string(first_lane_reads) + " v_readfirstlane_b32 of values every lane holds alike");
            }
            if (barriers) {
                head.push_back(std::to_string(barriers) + " s_barrier as workgroup control barriers over uniform, workgroup and image memory, "
                               "with every invocation that entered the loops around them (their exits taken on bits every lane holds alike)");
            }
        }
        if (tess) {
            const std::string model = window() ? "the patch window's stage I/O (inputs, outputs, patch constants)" : "the control point's slots";
            head.push_back("LDS as " + model + ": " + std::to_string(lds_reads) + " dword reads, " + std::to_string(lds_writes) +
                           " dword writes (each where the invocation's EXEC bit is set), " + std::to_string(lds_static) +
                           " at a slot the lift knows, " + std::to_string(lds_dynamic) +
                           " reads decoded at run time region by region as the reference addresses its window");
            if (tess_hs()) {
                head.push_back(std::to_string(tf_stores) + " factor stores as the tessellation levels, in program order (translate.cpp); " +
                               std::to_string(barriers) + " s_barrier as control barriers in uniform control flow");
            }
            if (constant_loads) {
                head.push_back(std::to_string(constant_loads) + " loads of the tessellation constants (" + tess->constants_path +
                               ") folded to the words the host gives the reference");
            }
        }
        const WaveOps& w = res.wave;
        head.push_back("wave and quad operations: speed-only, dropped - " + std::to_string(w.execz_skips) + " s_cbranch_execz skips, " +
                       std::to_string(w.kill_skips) + " kill-region skips, " + std::to_string(w.wave_tests) + " VCC tests made per pixel, " +
                       std::to_string(w.lane_exits) + " per-lane loop exits, " + std::to_string(w.whole_quad) + " s_wqm_b64, " +
                       std::to_string(w.spill_lanes) + " spill-lane moves; semantic, kept - " + std::to_string(w.quad_swizzles) +
                       " quad swizzles, " + std::to_string(w.implicit_lod) + " implicit-LOD samples, " + std::to_string(w.lod_queries) +
                       " LOD queries, " + std::to_string(w.helper_queries) + " live-quad queries, " + std::to_string(w.discards) + " discards");
        res.proof.insert(res.proof.begin(), head.begin(), head.end());
    }
};

}  // namespace

namespace {
LiftResult lift_stage(Stage stage, const Program& program, const TranslateOptions& options, const TranslateResult& reference,
                      const TessLift* tess = nullptr) {
    if (options.stage != stage) {
        LiftResult wrong;
        wrong.rejections.push_back(stage == Stage::Pixel         ? "the options are not a pixel shader's"
                                   : stage == Stage::TessEval    ? "the options are not an evaluation stage's"
                                   : stage == Stage::TessControl ? "the options are not a control stage's"
                                   : stage == Stage::Compute     ? "the options are not a compute shader's"
                                                                 : "the options are not a vertex shader's");
        return wrong;
    }
    // A lane mask a loop changes before it is read again gets a phi on a second
    // attempt: the first assumes every one comes back unchanged, which the
    // usual loop-exit idioms need to fold their tests. Likewise a register
    // whose helper lanes come back around a loop less exact than they left
    // is inexact at the header on the next attempt.
    Lifter::LoopPhis phis, inexact, nonuniform;
    for (int attempt = 0;; ++attempt) {
        Lifter l(program, options, reference, phis, inexact, tess, nonuniform);
        LiftResult r = l.run();
        const std::size_t before = phis.size() + inexact.size() + nonuniform.size();
        phis.insert(l.more_lane_phis.begin(), l.more_lane_phis.end());
        inexact.insert(l.more_inexact.begin(), l.more_inexact.end());
        nonuniform.insert(l.more_nonuniform.begin(), l.more_nonuniform.end());
        if (phis.size() + inexact.size() + nonuniform.size() == before || attempt == (stage == Stage::Compute ? 24 : 8)) return r;
    }
}
}  // namespace

LiftResult lift_pixel_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference) {
    return lift_stage(Stage::Pixel, program, options, reference);
}

LiftResult lift_vertex_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference) {
    return lift_stage(Stage::Vertex, program, options, reference);
}

namespace {
std::uint32_t branch_target(const Inst& in) { return in.offset + 4 + static_cast<std::uint32_t>(in.imm) * 4; }
// A compute program as the lift reads it, the same program (`notes` says how):
//   * the words after its first s_endpgm, when no branch before it lands
//     past it (the eboot's own programs carry data there), are left out;
//   * s_cbranch_execnz T; s_branch E; s_nop ... T: (an early-out over the
//     rest of the program) is s_cbranch_execz E; s_nop ...: EXEC set jumps to
//     T past the s_branch and the nops, clear takes the s_branch to E - the
//     same as skipping [T, E) when EXEC is clear.
Program compute_program(const Program& in, std::vector<std::string>& notes) {
    Program p = in;
    for (std::size_t i = 0; i < p.insts.size(); ++i) {
        const Inst& end = p.insts[i];
        if (!(end.enc == Enc::SOPP && end.op == 1)) continue;
        bool past = false;
        for (std::size_t k = 0; k < i; ++k) {
            if (is_branch(p.insts[k]) && branch_target(p.insts[k]) > end.offset) past = true;
        }
        if (past || i + 1 == p.insts.size()) break;
        notes.push_back(std::to_string(p.insts.size() - i - 1) + " words after s_endpgm at " + hex_offset(end.offset) +
                        " that no branch reaches are not part of the program");
        p.insts.resize(i + 1);
        p.end = end.offset + 4;
        p.errors.erase(std::remove_if(p.errors.begin(), p.errors.end(), [&](const DecodeError& e) { return e.offset > end.offset; }), p.errors.end());
        break;
    }
    std::set<std::uint32_t> targets;
    for (const Inst& in : p.insts) {
        if (is_branch(in) || (in.enc == Enc::SOPP && in.op == 9)) targets.insert(branch_target(in));
    }
    for (std::size_t i = 0; i + 1 < p.insts.size(); ++i) {
        Inst& nz = p.insts[i];
        Inst& jump = p.insts[i + 1];
        if (!(nz.enc == Enc::SOPP && nz.op == 9 && jump.enc == Enc::SOPP && jump.op == 2)) continue;
        const std::uint32_t t = branch_target(nz), e = branch_target(jump);
        if (t <= jump.offset || e < t) continue;
        bool nops = true;
        for (std::size_t k = i + 2; k < p.insts.size() && p.insts[k].offset < t; ++k) nops = nops && p.insts[k].enc == Enc::SOPP && p.insts[k].op == 0;
        const auto landing = targets.lower_bound(nz.offset + 4);
        if (!nops || (landing != targets.end() && *landing < t)) continue;
        notes.push_back("s_cbranch_execnz at " + hex_offset(nz.offset) + " over the s_branch to " + hex_offset(e) +
                        " is an s_cbranch_execz there (the region " + hex_offset(t) + "-" + hex_offset(e) + " skipped where EXEC is clear)");
        nz.op = 8;
        nz.imm = static_cast<std::int32_t>((e - nz.offset - 4) / 4);
        jump.op = 0;
        jump.imm = 0;
    }
    return p;
}
}  // namespace

LiftResult lift_compute_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference) {
    std::vector<std::string> notes;
    const Program p = compute_program(program, notes);
    LiftResult r = lift_stage(Stage::Compute, p, options, reference);
    if (r.ok()) r.proof.insert(r.proof.end(), notes.begin(), notes.end());
    return r;
}

LiftResult lift_ls_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const TessLift& tess) {
    if (options.tess_role != TranslateOptions::TessRole::LsVertex) {
        LiftResult wrong;
        wrong.rejections.push_back("the options are not the LS's as a vertex stage");
        return wrong;
    }
    return lift_stage(Stage::Vertex, program, options, reference, &tess);
}

LiftResult lift_hull_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const TessLift& tess) {
    if (options.tess_role != TranslateOptions::TessRole::HullTcs) {
        LiftResult wrong;
        wrong.rejections.push_back("the options are not the hull's as a control stage");
        return wrong;
    }
    return lift_stage(Stage::TessControl, program, options, reference, &tess);
}

LiftResult lift_domain_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference, const TessLift& tess) {
    if (options.tess_role != TranslateOptions::TessRole::DomainTes) {
        LiftResult wrong;
        wrong.rejections.push_back("the options are not the domain shader's as an evaluation stage");
        return wrong;
    }
    return lift_stage(Stage::TessEval, program, options, reference, &tess);
}

HullLift lift_constant_hull(const Program& program) {
    HullLift res;
    const auto reject = [&](std::uint32_t at, const std::string& why) { res.rejections.push_back(hex_offset(at) + ": " + why); };
    if (!program.errors.empty()) reject(0, "the program has decode errors");
    // VGPRs last written with a constant, in program order: with no branch and
    // EXEC never written, every lane runs every instruction.
    std::map<int, std::uint32_t> known;
    std::vector<std::uint32_t> stores;
    std::size_t lds_writes = 0, scalar_loads = 0;
    static const float kInline[8] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f};
    for (const Inst& in : program.insts) {
        const char* name = mnemonic(in) ? mnemonic(in) : "unknown";
        const auto writes_exec = [](std::uint16_t d) { return d == kExecLo || d == kExecHi; };
        switch (in.enc) {
        case Enc::SOPP:
            if (in.op != 0 && in.op != 1 && in.op != 12) reject(in.offset, std::string(name) + ": only straight-line code");
            break;
        case Enc::SOP1:
            if (in.op == 32 || in.op == 33) reject(in.offset, "program-counter transfer");
            if ((in.op >= 36 && in.op <= 43) || writes_exec(in.dst)) reject(in.offset, std::string(name) + " writes EXEC");
            break;
        case Enc::SOP2: case Enc::SOPK:
            if (writes_exec(in.dst)) reject(in.offset, std::string(name) + " writes EXEC");
            break;
        case Enc::SOPC:
            break;
        case Enc::SMRD:
            ++scalar_loads;
            break;
        case Enc::VOP1: case Enc::VOP2: case Enc::VOP3: {
            if (in.enc == Enc::VOP3 && in.op < 0x100) break;    // a comparison: an SGPR pair
            if (in.enc == Enc::VOP1 && in.op == 2) break;        // v_readfirstlane_b32: an SGPR
            if ((in.enc == Enc::VOP2 && (in.op == 1)) || (in.enc == Enc::VOP3 && in.op == 0x101)) break;  // v_readlane_b32
            const int d = in.dst;
            std::uint32_t bits = 0;
            bool constant = false;
            if (in.enc == Enc::VOP1 && in.op == 1) {  // v_mov_b32
                if (in.src0 >= kConstHalf && in.src0 <= kConstHalf + 7) {
                    bits = std::bit_cast<std::uint32_t>(kInline[in.src0 - kConstHalf]);
                    constant = true;
                } else if (in.src0 == kLiteral && in.has_literal) {
                    bits = in.literal;
                    constant = true;
                }
            }
            if (constant) {
                known[d] = bits;
            } else {
                known.erase(d);
                if (in.enc == Enc::VOP3 && in.op >= 0x160 && in.op < 0x180) known.erase(d + 1);  // 64-bit results
            }
            break;
        }
        case Enc::VOPC:
            break;
        case Enc::DS:
            if (in.op == 13 || in.op == 14) {
                ++lds_writes;
            } else {
                reject(in.offset, std::string(name) + ": the hull of this model reads nothing back");
            }
            break;
        case Enc::MUBUF:
            if (in.op != 28) {
                reject(in.offset, std::string(name) + ": only single-dword stores, the factors");
                break;
            }
            if (const auto it = known.find(in.vdata); it != known.end()) {
                stores.push_back(it->second);
            } else {
                reject(in.offset, "a factor store of v" + std::to_string(in.vdata) + ", which holds no constant");
            }
            break;
        default:
            reject(in.offset, std::string(name) + " is not part of a constant-factor hull");
            break;
        }
    }
    if (stores.size() != 6) reject(program.insts.empty() ? 0 : program.insts.back().offset, std::to_string(stores.size()) + " factor stores, not the six of a quad patch");
    if (!res.rejections.empty()) return res;
    for (const std::uint32_t b : stores) res.factors.push_back(std::bit_cast<float>(b));
    char line[200];
    std::snprintf(line, sizeof(line), "%zu instructions, straight line, EXEC never written: six buffer_store_dword of constants - outer %g %g %g %g, inner %g %g",
                  program.insts.size(), res.factors[0], res.factors[1], res.factors[2], res.factors[3], res.factors[4], res.factors[5]);
    res.proof.push_back(line);
    res.proof.push_back(std::to_string(lds_writes) + " LDS writes (its patch constants), which the evaluation stage of one control point "
                        "as stage attributes never reads; " + std::to_string(scalar_loads) + " scalar loads feeding only them; no other "
                        "memory, image or export instruction");
    return res;
}

void name_lift(LiftResult& lift, const TranslateResult& reference, const ShaderReflection& reflection) {
    if (!lift.ok()) return;
    lift.names = name_bindings(reflection, reference);
    lift.spirv = apply_names(lift.spirv, lift.names);
}

}  // namespace gcn
