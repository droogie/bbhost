// The pilot pixel shader (a22c7f71) lifts from the shipped bundle, and the lifter
// refuses the constructs it cannot prove equal. Skips when the dump is absent.
#include "test_app0.h"
#include "gcn/container.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

gcn::LiftResult lift(const gcn::Program& p, gcn::TranslateOptions o) {
    // The renderer translates with dimensions from the resolved T#s; the pilot's
    // last two images are cube shadow maps. Find their count with a first pass.
    const gcn::TranslateResult paths = gcn::translate(p, o);
    o.image_dims.assign(paths.images.size(), {1 /* 2D */, false});
    if (o.image_dims.size() >= 2) {
        o.image_dims[o.image_dims.size() - 1].first = 3;
        o.image_dims[o.image_dims.size() - 2].first = 3;
    }
    const gcn::TranslateResult ref = gcn::translate(p, o);
    return gcn::lift_pixel_shader(p, o, ref);
}

bool rejected_with(const gcn::LiftResult& r, const char* text) {
    for (const std::string& why : r.rejections) {
        if (why.find(text) != std::string::npos) return true;
    }
    std::fprintf(stderr, "expected a rejection containing \"%s\"; got %zu rejection(s)%s%s\n", text, r.rejections.size(),
                 r.rejections.empty() ? "" : ": ", r.rejections.empty() ? "" : r.rejections[0].c_str());
    return false;
}

// A copy of the program with the instruction at `offset` (which must be `expect`) changed.
gcn::Program mutate(const gcn::Program& p, std::uint32_t offset, const char* expect, const std::function<void(gcn::Inst&)>& change) {
    gcn::Program q = p;
    for (gcn::Inst& in : q.insts) {
        if (in.offset != offset) continue;
        const char* name = gcn::mnemonic(in);
        CHECK(name && std::strcmp(name, expect) == 0);
        change(in);
        return q;
    }
    CHECK(!"instruction offset not found");
    return q;
}

void make_nop(gcn::Inst& in) {
    in.enc = gcn::Enc::SOPP;
    in.op = 0;
}

}  // namespace

// Hand-encoded pixel shaders for the lifter's control flow: loops (uniform and
// the per-lane exit idiom), scalar writes in regions read on some paths only,
// nested kill regions, ifs on a uniform comparison under a region's EXEC, and
// else-if chains. Each lift is checked against the facts it must state or the
// reason it must give, and each lifted module is validated. No game files.
namespace control_flow {

struct Asm {
    std::vector<std::uint32_t> w;
    std::uint32_t at() const { return static_cast<std::uint32_t>(w.size() * 4); }
    void sopp(std::uint32_t op, std::int16_t simm = 0) { w.push_back(0xbf800000u | op << 16 | static_cast<std::uint16_t>(simm)); }
    // A branch from the next instruction slot to the byte offset `target`.
    void branch(std::uint32_t op, std::uint32_t target) { sopp(op, static_cast<std::int16_t>((static_cast<int>(target) - static_cast<int>(at()) - 4) / 4)); }
    // A forward branch whose target is set later by `land`.
    std::size_t forward(std::uint32_t op) {
        sopp(op);
        return w.size() - 1;
    }
    void land(std::size_t slot) { w[slot] = (w[slot] & 0xffff0000u) | static_cast<std::uint16_t>(w.size() - slot - 1); }
    void sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc) { w.push_back(0xbe800000u | sdst << 16 | op << 8 | ssrc); }
    void sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t s0, std::uint32_t s1) { w.push_back(0x80000000u | op << 23 | sdst << 16 | s1 << 8 | s0); }
    void sopc(std::uint32_t op, std::uint32_t s0, std::uint32_t s1) { w.push_back(0xbf000000u | op << 16 | s1 << 8 | s0); }
    void vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | op << 9 | src0); }
    void vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(op << 25 | vdst << 17 | vsrc1 << 9 | src0); }
    void vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(0x7c000000u | op << 17 | vsrc1 << 9 | src0); }
    void interp(std::uint32_t vdst, std::uint32_t attr, std::uint32_t chan) {  // v_interp_p1_f32 / p2_f32 over v0, v1
        w.push_back(0xc8000000u | vdst << 18 | 0u << 16 | attr << 10 | chan << 8 | 0u);
        w.push_back(0xc8000000u | vdst << 18 | 1u << 16 | attr << 10 | chan << 8 | 1u);
    }
    void exp_mrt0(std::uint32_t v) {  // exp mrt0 v, v, v, v done vm
        w.push_back(0xf8000000u | 1u << 12 | 1u << 11 | 0xfu);
        w.push_back(v | v << 8 | v << 16 | v << 24);
    }
};

// Operand codes and opcodes.
constexpr std::uint32_t kV = 256, kC0 = 128, kOne = 242, kTwo = 244, kVcc = 106, kExec = 126;
constexpr std::uint32_t kEndpgm = 1, kBranch = 2, kScc0 = 4, kScc1 = 5, kVccz = 6, kExecz = 8;  // SOPP
constexpr std::uint32_t kSMov = 3, kSMov64 = 4, kSWqm = 10, kSAndSaveexec = 36;                 // SOP1
constexpr std::uint32_t kSAddU32 = 0, kSAnd64 = 15, kSXor64 = 19, kSAndn2_64 = 21, kSLshl = 30;  // SOP2
constexpr std::uint32_t kSCmpEqI32 = 0, kSCmpGeI32 = 3;                                         // SOPC
constexpr std::uint32_t kVMov = 1, kVCvtI32F32 = 8;                                             // VOP1
constexpr std::uint32_t kVAddF32 = 3, kVMulF32 = 8, kVAddI32 = 37;                             // VOP2
constexpr std::uint32_t kCmpLtF32 = 0x01, kCmpGtF32 = 0x04, kCmpGtI32 = 0x84;                  // VOPC

// Translates as the renderer's no-fallback variant (user SGPRs s0..s3,
// interpolation at the centre in v0, v1) and lifts against that reference.
gcn::LiftResult lift(const Asm& a) {
    const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
    CHECK(p.errors.empty());
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc2 = 4u << 1;
    o.ps_input_ena = 2;
    o.descriptor_set = 1;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    const gcn::TranslateResult ref = gcn::translate(p, o);
    if (!ref.ok()) std::fprintf(stderr, "reference failed: %s\n", ref.errors.empty() ? "?" : ref.errors[0].c_str());
    CHECK(ref.ok());
    return gcn::lift_pixel_shader(p, o, ref);
}

bool lifted_valid(const gcn::LiftResult& r, const char* what) {
    for (const std::string& why : r.rejections) std::fprintf(stderr, "%s rejected: %s\n", what, why.c_str());
    if (!r.ok()) return false;
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    if (!tools.Validate(r.spirv.data(), r.spirv.size(), vo)) {
        std::fprintf(stderr, "%s: lifted SPIR-V invalid: %s\n", what, msg.c_str());
        return false;
    }
    return true;
}

bool proof_has(const gcn::LiftResult& r, const char* text) {
    for (const std::string& line : r.proof) {
        if (line.find(text) != std::string::npos) return true;
    }
    std::fprintf(stderr, "expected a proof line containing \"%s\"\n", text);
    return false;
}

std::size_t count_op(const std::vector<std::uint32_t>& spv, std::uint32_t opcode) {
    std::size_t n = 0;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t len = spv[i] >> 16;
        if (!len) break;
        n += (spv[i] & 0xffff) == opcode;
        i += len;
    }
    return n;
}
constexpr std::uint32_t kOpPhi = 245, kOpLoopMerge = 246, kOpSelectionMerge = 247;

// A uniform loop: acc += x, s0 times (s0 is user data).
Asm uniform_loop(bool counter_kept = false) {
    Asm a;
    a.interp(2, 0, 0);                    // v2 = x
    a.vop1(kVMov, 3, kC0);                // v3 = 0
    a.sop1(kSMov, 5, kC0);                // s5 = 0
    const std::uint32_t header = a.at();
    a.sopc(kSCmpGeI32, 5, 0);             // s5 >= s0
    const std::size_t exit = a.forward(kScc1);
    a.vop2(kVAddF32, 3, kV + 2, 3);       // v3 += v2
    a.sop2(kSAddU32, 5, 5, kC0 + 1);      // s5 += 1
    a.branch(kBranch, header);
    a.land(exit);
    if (counter_kept) a.vop1(kVMov, 4, 5);  // reads s5 after the loop
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

// The per-lane exit idiom: each pixel adds 1.0 to v3 while v5 < its own count
// v4, inside an s_cbranch_execz region on a per-pixel condition. `variant`
// breaks one rule each (see the cases below).
Asm divergent_loop(int variant = 0) {
    Asm a;
    a.interp(2, 0, 0);                         // v2 = the count, per pixel
    a.interp(6, 0, 1);                         // v6 = the region condition, per pixel
    a.vop1(kVCvtI32F32, 4, kV + 2);            // v4 = int(v2)
    a.vop1(kVMov, 3, kC0);                     // v3 = 0
    a.vopc(kCmpLtF32, kC0, 6);                 // vcc = 0 < v6
    a.sop1(kSAndSaveexec, 16, kVcc);
    const std::size_t region = a.forward(kExecz);
    a.vop1(kVMov, 5, kC0);                     // v5 = 0
    a.sop1(kSMov64, 10, kExec);                // EXEC kept for after the loop
    a.sop1(kSMov64, 12, variant == 1 ? 16 : kExec);  // the loop mask L (variant 1: EXEC before the region, not EXEC)
    const std::uint32_t header = a.at();
    a.vopc(kCmpGtI32, kV + 4, 5);              // vcc = v4 > v5: stay
    a.sop1(kSMov64, 14, kExec);
    a.sop2(kSAndn2_64, kExec, 14, kVcc);       // EXEC = the lanes leaving
    a.sop2(kSAndn2_64, 12, 12, kExec);         // L loses them
    const std::size_t exit = a.forward(kScc0); // no lane left: done
    a.sop2(kSAnd64, kExec, 14, 12);            // EXEC = the lanes staying
    a.vop2(kVAddF32, 3, kOne, 3);              // v3 += 1
    if (variant == 2) a.sop1(kSMov, 20, kC0 + 3);  // a scalar written inside, read after the loop
    a.vop2(kVAddI32, 5, kC0 + 1, 5);           // v5 += 1
    a.branch(kBranch, header);
    a.land(exit);
    a.sop1(kSMov64, kExec, 10);                // EXEC back
    if (variant == 2) a.vop1(kVMov, 7, 20);
    a.land(region);
    a.sop1(kSMov64, kExec, 16);
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

void loops() {
    {
        const gcn::LiftResult r = lift(uniform_loop());
        CHECK(lifted_valid(r, "uniform loop"));
        CHECK(proof_has(r, "uniform, left on a bit every lane holds alike"));
        CHECK(count_op(r.spirv, kOpLoopMerge) == 1);
        CHECK(count_op(r.spirv, kOpPhi) >= 2);  // the accumulator and the counter
    }
    // The counter is read after the loop: it leaves with the value it had at the exit.
    CHECK(lifted_valid(lift(uniform_loop(true)), "uniform loop, counter read after it"));
    {
        const gcn::LiftResult r = lift(divergent_loop());
        CHECK(lifted_valid(r, "divergent loop"));
        CHECK(proof_has(r, "divergent, the per-lane exit idiom on s12"));
        CHECK(count_op(r.spirv, kOpLoopMerge) == 1);
    }
    // The loop mask must start as EXEC.
    CHECK(rejected_with(lift(divergent_loop(1)), "does not start as EXEC"));
    // GCN may run more iterations than the pixel: a scalar the loop writes is dead after it.
    CHECK(rejected_with(lift(divergent_loop(2)), "written inside the divergent loop"));
    // A second way out of the loop.
    {
        Asm b;
        b.interp(2, 0, 0);
        b.vop1(kVMov, 3, kC0);
        b.sop1(kSMov, 5, kC0);
        const std::uint32_t header = b.at();
        b.sopc(kSCmpGeI32, 5, 0);
        const std::size_t exit1 = b.forward(kScc1);
        b.sopc(kSCmpEqI32, 5, kC0 + 7);
        const std::size_t exit2 = b.forward(kScc1);
        b.vop2(kVAddF32, 3, kV + 2, 3);
        b.sop2(kSAddU32, 5, 5, kC0 + 1);
        b.branch(kBranch, header);
        b.land(exit1);
        b.land(exit2);
        b.exp_mrt0(3);
        b.sopp(kEndpgm);
        CHECK(rejected_with(lift(b), "a second exit from the loop"));
    }
    // A branch from inside the loop past its exit.
    {
        Asm b;
        b.interp(2, 0, 0);
        b.vop1(kVMov, 3, kC0);
        b.sop1(kSMov, 5, kC0);
        const std::uint32_t header = b.at();
        b.sopc(kSCmpGeI32, 5, 0);
        const std::size_t far = b.forward(kScc1);
        b.vop2(kVAddF32, 3, kV + 2, 3);
        b.sop2(kSAddU32, 5, 5, kC0 + 1);
        b.branch(kBranch, header);
        b.vop1(kVMov, 3, kC0);  // skipped by the exit
        b.land(far);
        b.exp_mrt0(3);
        b.sopp(kEndpgm);
        CHECK(rejected_with(lift(b), "for somewhere other than its exit"));
    }
    // SCC carried around the loop: read at the header before the body writes it.
    {
        Asm b;
        b.interp(2, 0, 0);
        b.vop1(kVMov, 3, kC0);
        b.sop1(kSMov, 5, kC0);
        b.sopc(kSCmpGeI32, 5, 0);
        const std::uint32_t header = b.at();
        const std::size_t exit = b.forward(kScc1);
        b.vop2(kVAddF32, 3, kV + 2, 3);
        b.sop2(kSAddU32, 5, 5, kC0 + 1);
        b.sopc(kSCmpGeI32, 5, 0);
        b.branch(kBranch, header);
        b.land(exit);
        b.exp_mrt0(3);
        b.sopp(kEndpgm);
        CHECK(rejected_with(lift(b), "SCC is read before it is written"));
    }
    // An export inside a loop.
    {
        Asm b;
        b.interp(2, 0, 0);
        b.sop1(kSMov, 5, kC0);
        const std::uint32_t header = b.at();
        b.sopc(kSCmpGeI32, 5, 0);
        const std::size_t exit = b.forward(kScc1);
        b.exp_mrt0(2);
        b.sop2(kSAddU32, 5, 5, kC0 + 1);
        b.branch(kBranch, header);
        b.land(exit);
        b.sopp(kEndpgm);
        CHECK(rejected_with(lift(b), "export inside a loop"));
    }
    // A lane mask the loop changes and reads again at the header: a second
    // attempt gives it a phi (s[20:21] flips each iteration under a uniform count).
    {
        Asm b;
        b.interp(2, 0, 0);
        b.vop1(kVMov, 3, kC0);
        b.sop1(kSMov, 5, kC0);
        b.vopc(kCmpLtF32, kC0, 2);                 // vcc = 0 < x
        b.sop1(kSMov64, 20, kVcc);
        const std::uint32_t header = b.at();
        b.sopc(kSCmpGeI32, 5, 0);
        const std::size_t exit = b.forward(kScc1);
        b.sop2(kSXor64, 20, 20, kExec);            // flip
        b.w.push_back(0xd2000000u | 4u);           // v_cndmask_b32 v4, 0, v3, s[20:21] (VOP3 0x100)
        b.w.push_back(kC0 | (kV + 3) << 9 | 20u << 18);
        b.vop2(kVAddF32, 3, kV + 4, 3);
        b.sop2(kSAddU32, 5, 5, kC0 + 1);
        b.branch(kBranch, header);
        b.land(exit);
        b.exp_mrt0(3);
        b.sopp(kEndpgm);
        const gcn::LiftResult r = lift(b);
        CHECK(lifted_valid(r, "a lane mask changed around a uniform loop"));
        CHECK(proof_has(r, "1 lane masks)"));
    }
}

void regions_and_ifs() {
    // s5 is written inside an s_cbranch_execz region and read after it only
    // in if arms that write it first: dead on every path, though not on every
    // instruction after the region.
    const auto scalar_after_region = [](bool else_writes) {
        Asm a;
        a.interp(2, 0, 0);
        a.vopc(kCmpLtF32, kC0, 2);
        a.sop1(kSAndSaveexec, 10, kVcc);
        const std::size_t region = a.forward(kExecz);
        a.sop1(kSMov, 5, kC0 + 7);
        a.vop1(kVMov, 3, 5);
        a.land(region);
        a.sop1(kSMov64, kExec, 10);
        a.sopc(kSCmpEqI32, 0, kC0);
        const std::size_t to_else = a.forward(kScc1);
        a.sop1(kSMov, 5, kC0 + 1);
        a.vop1(kVMov, 4, 5);
        const std::size_t to_join = a.forward(kBranch);
        a.land(to_else);
        if (else_writes) a.sop1(kSMov, 5, kC0 + 2);
        a.vop1(kVMov, 4, 5);
        a.land(to_join);
        a.vop2(kVAddF32, 4, kV + 3, 4);
        a.exp_mrt0(4);
        a.sopp(kEndpgm);
        return a;
    };
    {
        const gcn::LiftResult r = lift(scalar_after_region(true));
        CHECK(lifted_valid(r, "a region's scalar redefined in both if arms"));
        CHECK(proof_has(r, "redefined before any later read on every path"));
    }
    CHECK(rejected_with(lift(scalar_after_region(false)), "s5 is written inside region"));

    // Two kill regions, one inside the other, both widened from EXEC known
    // set, ending together; EXEC at s_endpgm is the inner guard.
    const auto nested_kills = [](bool final_is_outer_mask) {
        Asm a;
        a.sop1(kSMov64, 20, kExec);                // the live mask
        a.sop1(kSMov64, 22, kExec);
        a.sop1(kSWqm, kExec, kExec);
        a.interp(2, 0, 0);
        a.vopc(kCmpGtF32, kC0, 2);                 // vcc = 0 > x: kill
        a.sop2(kSAndn2_64, 20, 20, kVcc);
        const std::size_t end1 = a.forward(kScc0);
        a.sop2(kSAnd64, kExec, kExec, 20);
        a.sop1(kSWqm, kExec, kExec);
        a.vopc(kCmpLtF32, kOne, 2);                // vcc = 1 < x: kill
        a.sop2(kSAndn2_64, 20, 20, kVcc);
        const std::size_t end2 = a.forward(kScc0);
        a.sop2(kSAnd64, kExec, kExec, 20);
        a.sop1(kSWqm, kExec, kExec);
        a.vop2(kVMulF32, 3, kV + 2, 2);
        a.land(end1);
        a.land(end2);
        a.sop1(kSMov64, kExec, final_is_outer_mask ? 22 : 20);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        return a;
    };
    {
        const gcn::LiftResult r = lift(nested_kills(false));
        CHECK(lifted_valid(r, "nested kill regions"));
        CHECK(proof_has(r, "0 s_cbranch_execz regions, 2 s_cbranch_scc0 kill regions"));
    }
    CHECK(rejected_with(lift(nested_kills(true)), "not within the guard of a kill region"));

    // An if on VCC = (s0 > v5) & EXEC inside a region, where v5 holds 0 in
    // every lane the region's EXEC has: the wave-wide test is s0 > 0.
    const auto vcc_under_exec = [](bool per_pixel) {
        Asm a;
        a.interp(2, 0, 0);
        a.vopc(kCmpLtF32, kC0, 2);
        a.sop1(kSAndSaveexec, 10, kVcc);
        const std::size_t region = a.forward(kExecz);
        a.vop1(kVMov, 5, per_pixel ? kV + 2 : kC0);
        a.vopc(kCmpGtF32, 0, 5);                   // vcc = s0 > v5
        const std::size_t to_else = a.forward(kVccz);
        a.vop1(kVMov, 3, kOne);
        const std::size_t to_join = a.forward(kBranch);
        a.land(to_else);
        a.vop1(kVMov, 3, kTwo);
        a.land(to_join);
        a.vop2(kVMulF32, 3, kV + 3, 2);
        a.land(region);
        a.sop1(kSMov64, kExec, 10);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        return a;
    };
    {
        const gcn::LiftResult r = lift(vcc_under_exec(false));
        CHECK(lifted_valid(r, "an if on a uniform comparison under a region's EXEC"));
        CHECK(proof_has(r, "the wave-wide test is the comparison"));
    }
    CHECK(rejected_with(lift(vcc_under_exec(true)), "VCC that may differ between lanes"));

    // An else-if chain: the inner if's arms jump straight to the outer join.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.sopc(kSCmpEqI32, 0, kC0);
        const std::size_t case0 = a.forward(kScc1);
        a.sopc(kSCmpEqI32, 0, kC0 + 1);
        const std::size_t other = a.forward(kScc0);
        a.vop1(kVMov, 3, kOne);
        const std::size_t join1 = a.forward(kBranch);
        a.land(other);
        a.vop1(kVMov, 3, kTwo);
        const std::size_t join2 = a.forward(kBranch);
        a.land(case0);
        a.vop1(kVMov, 3, kC0);
        a.land(join1);
        a.land(join2);
        a.vop2(kVMulF32, 3, kV + 3, 2);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        // join1 jumps from the inner then arm straight to the outer join: retarget it.
        const gcn::LiftResult r = lift(a);
        CHECK(lifted_valid(r, "an else-if chain"));
        CHECK(proof_has(r, "its arms jump past this s_branch to the join it jumps to"));
        CHECK(count_op(r.spirv, kOpSelectionMerge) == 2);
    }
    // 32-bit scalar arithmetic: a shifted sum decides a uniform if, and its carry another.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.sop2(kSAddU32, 5, 0, 1);
        a.sop2(kSLshl, 6, 5, kC0 + 2);
        a.sopc(kSCmpGeI32, 6, kC0 + 16);
        const std::size_t skip = a.forward(kScc1);
        a.vop2(kVMulF32, 2, kTwo, 2);
        a.land(skip);
        a.exp_mrt0(2);
        a.sopp(kEndpgm);
        CHECK(lifted_valid(lift(a), "scalar arithmetic"));
    }
}

void run() {
    loops();
    regions_and_ifs();
}

}  // namespace control_flow

int main(int argc, char** argv) {
    control_flow::run();  // hand-encoded programs, before (and without) the game's
    const std::string path = argc > 1 ? argv[1] : test_app0_file("dvdroot_ps4/shader/gxrenderershader.shaderbnd.dcx");
    std::vector<std::uint8_t> raw;
    if (!gcn::read_file(path, raw)) {
        if (g_failures) {
            std::fprintf(stderr, "gcn_lift_test: %d check(s) failed\n", g_failures);
            return 1;
        }
        std::printf("gcn_lift_test: hand-encoded programs ok; the pilot skipped: %s not readable\n", path.c_str());
        return 0;
    }
    std::string err;
    const std::vector<std::uint8_t> b = gcn::dcx_decompress(raw, &err);
    std::vector<gcn::BundleEntry> entries;
    if (b.empty() || !gcn::bnd4_entries(b, entries, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const gcn::BundleEntry* pilot = nullptr;
    for (const gcn::BundleEntry& e : entries) {
        if (e.name.find("GXLightAcc_LegacyDPointLightShadowed2.ppo") != std::string::npos) pilot = &e;
    }
    gcn::ShaderCode code;
    if (!pilot || !gcn::shader_code(pilot->data, code)) {
        std::fprintf(stderr, "pilot shader not found in %s\n", path.c_str());
        return 1;
    }
    std::size_t shdr = 0;
    for (std::size_t i = 0; i + 4 <= pilot->data.size(); ++i) {
        if (std::memcmp(pilot->data.data() + i, "Shdr", 4) == 0) {
            shdr = i;
            break;
        }
    }
    std::uint32_t regs[16];
    std::memcpy(regs, pilot->data.data() + shdr + 16, 64);
    const gcn::Program p = gcn::decode(code.words.data(), code.words.size());
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc1 = regs[4];
    o.rsrc2 = regs[5];
    o.ps_input_ena = regs[8];
    o.descriptor_set = 1;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    o.early_fragment_tests = true;

    const gcn::LiftResult ok = lift(p, o);
    for (const std::string& why : ok.rejections) std::fprintf(stderr, "pilot rejected: %s\n", why.c_str());
    CHECK(ok.ok());
    CHECK(ok.proof.size() >= 9);  // five facts plus one per region
    if (ok.ok()) {
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
        std::string msg;
        tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
            if (msg.empty()) msg = m;
        });
        spvtools::ValidatorOptions vo;
        vo.SetAllowOffsetTextureOperand(true);
        const bool valid = tools.Validate(ok.spirv.data(), ok.spirv.size(), vo);
        if (!valid) std::fprintf(stderr, "lifted SPIR-V invalid: %s\n", msg.c_str());
        CHECK(valid);
    }

    // Image dimensions predicted from the program alone (per-shader compiles,
    // step 2): the pilot's last two images are its cube shadow maps, whose face
    // coordinate is a v_cubeid_f32 result; the others are 2D.
    {
        const gcn::TranslateResult paths = gcn::translate(p, o);
        const std::vector<gcn::PredictedImage> predicted = gcn::predict_image_dims(p, paths);
        CHECK(predicted.size() == paths.images.size() && predicted.size() >= 2);
        for (std::size_t k = 0; k < predicted.size(); ++k) {
            const bool cube = k + 2 >= predicted.size();
            CHECK(predicted[k].dim == (cube ? 3u : 1u) && !predicted[k].arrayed && !predicted[k].conflict);
        }
    }

    // The storage-buffer fallback reads the page table, which the lift does not model.
    gcn::TranslateOptions fallback = o;
    fallback.cb_no_fallback = false;
    CHECK(rejected_with(lift(p, fallback), "no-fallback"));

    // Without the restore after the first masked region, the sample at 0x4cc runs
    // where EXEC is only the else-mask.
    CHECK(rejected_with(lift(mutate(p, 0x45c, "s_mov_b64", make_nop), o), "EXEC is not known set"));

    // Without the load at 0x460, s4 written in the second region (0x3ec-0x45c)
    // is read at 0x46c: GCN runs that block for every lane when any lane needs it.
    CHECK(rejected_with(lift(mutate(p, 0x460, "s_buffer_load_dwordx4", make_nop), o), "s4 is written inside region"));

    // An s_branch that closes no if's then arm (here a jump over a region).
    CHECK(rejected_with(lift(mutate(p, 0x3ac, "s_cbranch_execz", [](gcn::Inst& in) { in.op = 2; }), o), "does not close an if"));

    // A VCC branch becomes an if only where every lane holds the same VCC bit;
    // the pilot's VCC comes from per-pixel values.
    CHECK(rejected_with(lift(mutate(p, 0x3ac, "s_cbranch_execz", [](gcn::Inst& in) { in.op = 6; }), o), "VCC that may differ between lanes"));

    // A sample inside a masked region, where EXEC is the region's mask (the
    // instruction at 0x3b0 becomes the sample from 0x068).
    gcn::Inst sample;
    for (const gcn::Inst& in : p.insts) {
        if (in.offset == 0x68) sample = in;
    }
    CHECK(rejected_with(lift(mutate(p, 0x3b0, "v_mul_f32",
                                    [&](gcn::Inst& in) {
                                        const std::uint32_t at = in.offset;
                                        const std::uint8_t size = in.size;
                                        in = sample;
                                        in.offset = at;
                                        in.size = size;
                                    }),
                             o),
                        "image sample where EXEC is not known set"));

    if (g_failures) {
        std::fprintf(stderr, "gcn_lift_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("gcn_lift_test ok: pilot lifted (%zu words), six constructs rejected\n", ok.spirv.size());
    return 0;
}
