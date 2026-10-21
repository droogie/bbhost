// The HLSL printer (gcn/hlsl.h) on hand-encoded programs the lifter takes:
// quad swizzles in every quad pattern, v_cndmask_b32 with source modifiers, a
// sample and a LOD query in a divergent region, a sample's run-time offset
// (both ways the lift takes it), loops with per-lane exits (one inside a kill
// region, with the helper-lane query), a vertex shader's position with and
// without Invariant, and lifts for a device that rounds halves toward zero
// itself (packs and compressed exports, a word decoded, a pack under a partial
// EXEC, a pack read as an integer, a vertex shader with no half math, a
// constant converted, a compute shader) with what the printer refuses of such
// a lift. Each lift is printed, the source checked for the spellings that
// keep its instructions, and - where DXC is installed (host/dxc.h) -
// compiled: valid SPIR-V with the lift's interface and operations.
// Synthetic programs only: no game shader is read.
#include "gcn/half.h"
#include "gcn/hlsl.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/spirv.h"
#include "gcn/translate.h"
#include "host/dxc.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

gpu::Dxc g_dxc;
bool g_have_dxc = false;
int g_compiled = 0, g_printed = 0;

bool spirv_valid(const std::vector<std::uint32_t>& words, std::string& msg) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    return tools.Validate(words.data(), words.size(), vo);
}

// Prints `lift`, checks the source holds each of `spellings`, and compiles it.
// `runtime_offset`: whether the module samples with a run-time offset, which
// DXC compiles without its own validator (HlslResult::validate_offsets).
// `printed` and `compiled` (when given) receive the HLSL and DXC's module.
void through_hlsl(const gcn::LiftResult& lift, const std::string& what, const std::vector<std::string>& spellings = {}, bool runtime_offset = false,
                  gcn::HlslResult* printed = nullptr, std::vector<std::uint32_t>* compiled_module = nullptr) {
    CHECK(lift.ok());
    if (!lift.ok()) {
        std::fprintf(stderr, "%s: not lifted: %s\n", what.c_str(), lift.rejections.empty() ? "?" : lift.rejections[0].c_str());
        return;
    }
    gcn::HlslOptions o;
    o.title = what;
    const gcn::HlslResult h = gcn::emit_hlsl(lift.spirv, o);
    if (printed) *printed = h;
    CHECK(h.ok());
    if (!h.ok()) {
        std::fprintf(stderr, "%s: not printed: %s\n", what.c_str(), h.errors.empty() ? "?" : h.errors[0].c_str());
        return;
    }
    ++g_printed;
    CHECK(h.validate_offsets == runtime_offset);
    for (const std::string& s : spellings) {
        const bool found = h.source.find(s) != std::string::npos;
        CHECK(found);
        if (!found) std::fprintf(stderr, "%s: the HLSL has no \"%s\"\n", what.c_str(), s.c_str());
    }
    if (!g_have_dxc) return;
    gpu::DxcOptions dopt;
    std::vector<std::uint32_t> spirv;
    std::string error;
    const bool compiled = g_dxc.compile(h, dopt, spirv, error);
    CHECK(compiled);
    if (!compiled) {
        std::fprintf(stderr, "%s: DXC: %s\n%s\n", what.c_str(), error.c_str(), h.source.c_str());
        return;
    }
    std::string msg;
    const bool valid = spirv_valid(spirv, msg);
    CHECK(valid);
    if (!valid) std::fprintf(stderr, "%s: DXC's module is invalid: %s\n", what.c_str(), msg.c_str());
    const std::string iface = gcn::hlsl_interface_diff(lift.spirv, spirv), ops = gcn::hlsl_operation_diff(lift.spirv, spirv);
    CHECK(iface.empty());
    CHECK(ops.empty());
    if (!iface.empty()) std::fprintf(stderr, "%s: interface: %s\n", what.c_str(), iface.c_str());
    if (!ops.empty()) std::fprintf(stderr, "%s: operations: %s\n", what.c_str(), ops.c_str());
    if (compiled_module) *compiled_module = spirv;
    ++g_compiled;
}

// ---- Sea Islands encodings ---------------------------------------------------------
constexpr std::uint32_t kVcc = 106, kExec = 126, kZero = 128, kHalf = 240, kOne = 242, kV = 256, kEndpgm = 0xbf810000u;
constexpr std::uint32_t kGetLod = 96, kSampleLz = 39, kSampleLzO = 55;

struct Asm {
    std::vector<std::uint32_t> w;
    void sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0) { w.push_back(0xbe800000u | sdst << 16 | op << 8 | ssrc0); }
    void sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t ssrc1) {
        w.push_back(0x80000000u | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0);
    }
    void sopc(std::uint32_t op, std::uint32_t ssrc0, std::uint32_t ssrc1) { w.push_back(0xbf000000u | op << 16 | ssrc1 << 8 | ssrc0); }
    void vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | op << 9 | src0); }
    void vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(op << 25 | vdst << 17 | vsrc1 << 9 | src0); }
    void vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(0x7c000000u | op << 17 | vsrc1 << 9 | src0); }
    void interp(std::uint32_t vdst, std::uint32_t attr, std::uint32_t chan) {
        w.push_back(0xc8000000u | vdst << 18 | attr << 10 | chan << 8 | 0u);
        w.push_back(0xc8000000u | vdst << 18 | 1u << 16 | attr << 10 | chan << 8 | 1u);
    }
    void mimg(std::uint32_t op, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t dmask) {
        w.push_back(0xf0000000u | op << 18 | dmask << 8);
        w.push_back(vaddr | vdata << 8 | (4u / 4) << 16 | (12u / 4) << 21);  // T# in s[4:11], S# in s[12:15]
    }
    void swizzle(std::uint32_t vdst, std::uint32_t vaddr, std::uint32_t off) {  // ds_swizzle_b32
        w.push_back(0xd8000000u | 53u << 18 | off);
        w.push_back(vdst << 24 | vaddr);
    }
    void wqm() { sop1(10, kExec, kExec); }
    void exp_mrt0(std::uint32_t v) {  // exp mrt0, v..v+3 done vm
        w.push_back(0xf800180fu);
        w.push_back(v | (v + 1) << 8 | (v + 2) << 16 | (v + 3) << 24);
    }
    std::size_t branch(std::uint32_t op) {
        w.push_back(0xbf800000u | op << 16);
        return w.size() - 1;
    }
    void land(std::size_t at) { w[at] |= static_cast<std::uint16_t>(w.size() - at - 1); }
    void back(std::uint32_t op, std::size_t target) {
        w.push_back(0xbf800000u | op << 16 | static_cast<std::uint16_t>(static_cast<int>(target) - static_cast<int>(w.size()) - 1));
    }
};

gcn::LiftResult lift_ps(const std::vector<std::uint32_t>& words, std::uint32_t dim = 1, bool early_tests = false) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    CHECK(p.errors.empty());
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc2 = 16u << 1;  // s0-s15 user data
    o.ps_input_ena = 2;  // v0, v1: the perspective centre
    o.descriptor_set = 1;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    o.image_dims = {{dim, false}};
    o.early_fragment_tests = early_tests;
    const gcn::TranslateResult ref = gcn::translate(p, o);
    CHECK(ref.ok());
    return gcn::lift_pixel_shader(p, o, ref);
}

// The clip prologue (TranslateOptions::ps_clip_discard): with
// gcn::clip_demote its pixel is demoted to a helper - printed as the inline
// instruction with its capability and extension, which DXC keeps - and
// without, ended by discard (OpKill), as every other discard is.
void clip_prologue() {
    Asm a;
    a.sop1(4, 2, kExec);  // s[2:3] = the coverage
    a.wqm();
    a.interp(2, 0, 0);    // v2 = attr0.x
    a.swizzle(3, 2, 0x8000u | 1u | 0u << 2 | 3u << 4 | 2u << 6);  // the horizontal neighbour's
    a.w.push_back(0xbf8c007fu);  // s_waitcnt lgkmcnt(0)
    a.sop1(4, kExec, 2);
    a.exp_mrt0(3);
    a.w.push_back(kEndpgm);
    for (const bool demote : {true, false}) {
        gcn::set_clip_demote(demote);
        const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
        CHECK(p.errors.empty());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Pixel;
        o.rsrc2 = 16u << 1;
        o.ps_input_ena = 2;
        o.descriptor_set = 1;
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.ps_clip_discard = 1u | 1u << 8;  // clip distance 0, at location 1
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok());
        through_hlsl(gcn::lift_pixel_shader(p, o, ref), demote ? "clip prologue, demoted" : "clip prologue, discarded",
                     demote ? std::vector<std::string>{"spv_demote_to_helper();", "SPV_EXT_demote_to_helper_invocation"}
                            : std::vector<std::string>{"discard;"});
    }
    gcn::set_clip_demote(false);
}

// ---- the programs ------------------------------------------------------------------
void swizzles() {
    // Whole-quad mode, v2 = attr0.x, v3 = a quad-mode swizzle of it, exported.
    int lifted = 0;
    for (std::uint32_t pattern = 0; pattern < 256; ++pattern) {
        Asm a;
        a.sop1(4, 2, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.swizzle(3, 2, 0x8000u | pattern);
        a.w.push_back(0xbf8c007fu);  // s_waitcnt lgkmcnt(0)
        a.sop1(4, kExec, 2);
        a.exp_mrt0(3);
        a.w.push_back(kEndpgm);
        const gcn::LiftResult l = lift_ps(a.w);
        if (!l.ok()) continue;
        ++lifted;
        std::vector<std::string> spell;
        if (pattern == 0x55) spell = {"QuadReadLaneAt("};
        if (pattern == 0xb1) spell = {"QuadReadAcrossX("};
        if (pattern == 0x4e) spell = {"QuadReadAcrossY("};
        if (pattern == 0x1b) spell = {"QuadReadAcrossDiagonal("};
        through_hlsl(l, "quad pattern " + std::to_string(pattern), spell);
    }
    CHECK(lifted == 256);
}

void cndmask() {
    // x = attr0.x; vcc = 0 < x; v3 = vcc ? src1 : src0, with abs and neg on the sources.
    for (std::uint32_t abs = 0; abs < 4; ++abs) {
        for (std::uint32_t neg = 0; neg < 4; ++neg) {
            Asm a;
            a.interp(2, 0, 0);
            a.vopc(1 /* v_cmp_lt_f32 */, kZero, 2);
            a.w.push_back(0xd2000000u | abs << 8 | 3u);
            a.w.push_back(neg << 29 | kVcc << 18 | (kV + 2) << 9 | (kV + 2));
            a.exp_mrt0(3);
            a.w.push_back(kEndpgm);
            const gcn::LiftResult l = lift_ps(a.w);
            if (l.ok()) through_hlsl(l, "v_cndmask_b32 abs " + std::to_string(abs) + " neg " + std::to_string(neg), {"select("});
        }
    }
}

void samples_and_loops() {
    // A LOD query on coordinates computed in whole-quad mode inside a divergent region.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(1, kZero, 2);   // v_cmp_lt_f32 vcc, 0, v2
        a.sop1(36, 22, kVcc);  // s_and_saveexec_b64
        const std::size_t skip = a.branch(8);
        a.sop1(4, kVcc, kExec);
        a.wqm();
        a.vop2(8, 8, kHalf, 2);  // v_mul_f32 v8, 0.5, v2
        a.vop2(8, 9, kHalf, 3);
        a.sop1(4, kExec, kVcc);
        a.mimg(kGetLod, 4, 8, 0x2);
        a.land(skip);
        a.sop1(4, kExec, 22);
        a.exp_mrt0(4);
        a.w.push_back(kEndpgm);
        through_hlsl(lift_ps(a.w), "image_get_lod in a divergent region", {"CalculateLevelOfDetailUnclamped(", "precise float"});
    }
    // A sample at level 0 with whole-quad coordinates.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.mimg(kSampleLz, 4, 2, 0xf);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(4);
        a.w.push_back(kEndpgm);
        through_hlsl(lift_ps(a.w), "image_sample_lz", {"SampleLevel("});
    }
    // A sample whose offset is a run-time value: the Offset operand where the
    // device has maintenance8 (DXC's validator refuses it: -Vd, and the caller
    // validates), the coordinates moved by it elsewhere.
    for (const bool runtime : {false, true}) {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(8, 8, kV + 3);  // v8 = int(attr0.y): the packed offsets
        a.vop1(1, 9, kV + 2);
        a.vop1(1, 10, kV + 3);
        a.mimg(kSampleLzO, 4, 8, 0xf);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(4);
        a.w.push_back(kEndpgm);
        gcn::set_runtime_sample_offsets(runtime);
        through_hlsl(lift_ps(a.w), std::string("image_sample_lz_o, ") + (runtime ? "run-time offset" : "moved coordinates"),
                     runtime ? std::vector<std::string>{"SampleLevel("} : std::vector<std::string>{"SampleLevel(", "GetDimensions("}, runtime);
        gcn::set_runtime_sample_offsets(false);
    }
    // A uniform loop with a kill region before it: the loop's header phis
    // (declared at the top), its exit, and the values used past it.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(1, 8, kV + 2);
        a.sop1(3, 32, kZero);  // s_movk... s32 = 0
        const std::size_t header = a.w.size();
        a.vopc(1, kZero, 2);
        a.sop1(36, 22, kVcc);
        const std::size_t skip = a.branch(8);
        a.sop1(4, kVcc, kExec);
        a.wqm();
        a.vop2(8, 8, kHalf, 2);
        a.sop1(4, kExec, kVcc);
        a.land(skip);
        a.sop1(4, kExec, 22);
        a.sopc(3, 32, 0);  // s_cmp_ge_i32 s32, s0
        const std::size_t exit = a.branch(5);
        a.vop1(1, 8, kV + 2);
        a.sop2(0, 32, 32, kZero + 1);
        a.back(2, header);
        a.land(exit);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(12);
        a.w.push_back(kEndpgm);
        through_hlsl(lift_ps(a.w), "a uniform loop", {"[loop] for (;;)", "break;"});
    }
    // A loop pixels leave at different iterations, inside a kill region,
    // with the helper-lane query the lift makes of it and the discard at the end.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(4, kZero, 2);           // v_cmp_gt_f32 vcc, 0, v2: kill
        a.sop2(21, 20, 20, kVcc);      // s_andn2_b64 s[20:21], s[20:21], vcc
        const std::size_t end = a.branch(4);
        a.sop2(15, kExec, kExec, 20);  // s_and_b64 exec, exec, s[20:21]
        a.wqm();
        a.vop1(8, 4, kV + 3);          // v4 = int(v3)
        a.vop1(1, 5, kZero);
        a.vop1(1, 8, kZero);
        a.sop1(4, 26, kExec);
        a.sop1(4, 24, kExec);          // the loop mask
        const std::size_t header = a.w.size();
        a.vopc(0x84, kV + 4, 5);       // v_cmp_gt_i32 vcc, v4, v5
        a.sop1(4, 28, kExec);
        a.sop2(21, kExec, 28, kVcc);
        a.sop2(21, 24, 24, kExec);
        const std::size_t exit = a.branch(4);
        a.sop2(15, kExec, 28, 24);
        a.vop2(3, 8, kOne, 8);         // v8 += 1.0
        a.vop2(37, 5, kZero + 1, 5);   // v_add_i32 v5, 1, v5
        a.back(2, header);
        a.land(exit);
        a.sop1(4, kExec, 26);
        a.land(end);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(8);
        a.w.push_back(kEndpgm);
        through_hlsl(lift_ps(a.w), "a divergent loop in a kill region",
                     {"[[vk::ext_builtin_input(23 /* HelperInvocation */)]]", "QuadReadAcross", "discard;"});
    }
}

// A vertex shader (s_swappc to the fetch, the position from v4-v7, param 0
// from v8-v11, two float4 elements) with its position Invariant
// (BBHOST_VS_INVARIANT) and without: the member keeps the decoration.
void vertex_invariant() {
    std::vector<std::uint32_t> w = {0xbe800000u | 0u << 16 | 33u << 8 | 2u};  // s_swappc_b64 s[0:1], s[2:3]
    const auto exp = [&](std::uint32_t tgt, std::uint32_t v, bool done) {
        w.push_back(0xf8000000u | (done ? 1u << 11 : 0u) | tgt << 4 | 0xfu);
        w.push_back((v + 3) << 24 | (v + 2) << 16 | (v + 1) << 8 | v);
    };
    exp(12, 4, true);
    exp(32, 8, false);
    w.push_back(kEndpgm);
    for (const bool invariant : {false, true}) {
        const gcn::Program p = gcn::decode(w.data(), w.size());
        CHECK(p.errors.empty());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Vertex;
        o.rsrc1 = 3u << 24;  // VGPR_COMP_CNT 3
        o.descriptor_set = 0;
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.link_outputs = true;
        for (std::uint8_t r = 0; r < 32; ++r) o.output_links.push_back(r);
        o.vertex_input = {{0, 4, 4, 0u}, {1, 8, 4, 0u}};
        o.vertex_formats_from_params = true;
        o.invariant_position = invariant;
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok());
        through_hlsl(gcn::lift_vertex_shader(p, o, ref), std::string("a vertex shader, position ") + (invariant ? "invariant" : "not invariant"),
                     invariant ? std::vector<std::string>{"[[vk::ext_decorate(18 /* Invariant */)]]"} : std::vector<std::string>{"SV_Position"});
    }
}

// ---- half precision rounded by the device ------------------------------------------
// v_cvt_pkrtz_f16_f32 and the compressed exports that read it, lifted as for a
// device that rounds f32 to f16 toward zero itself (gcn/half.h
// native_half_rtz: RADV, the Steam Deck): the lift declares Float16 and the
// execution modes DenormPreserve 16 and RoundingModeRTZ 16, and converts with
// OpFConvert. The HLSL declares the modes on its entry point (inline SPIR-V,
// with their capabilities) and writes each conversion and bitcast as the
// lift's own instruction: DXC's module keeps the modes with their width (the
// interface check), and a conversion to half wherever the lift's output needs one.

// Conversions to a 16-bit float in a module, and whether it declares `mode` for 16-bit floats.
std::size_t half_conversions(const std::vector<std::uint32_t>& w) {
    std::vector<std::uint32_t> half;
    std::size_t n = 0;
    for (std::size_t i = 5; i < w.size() && (w[i] >> 16);) {
        const std::uint32_t len = w[i] >> 16, op = w[i] & 0xffff;
        const auto is_half = [&](std::uint32_t t) { return std::find(half.begin(), half.end(), t) != half.end(); };
        if (op == 22 /* OpTypeFloat */ && len >= 3 && w[i + 2] == 16) half.push_back(w[i + 1]);
        if (op == 23 /* OpTypeVector */ && len >= 4 && is_half(w[i + 2])) half.push_back(w[i + 1]);
        if (op == 115 /* OpFConvert */ && len >= 4 && is_half(w[i + 1])) ++n;
        i += len;
    }
    return n;
}
bool has_mode16(const std::vector<std::uint32_t>& w, std::uint32_t mode) {
    for (std::size_t i = 5; i < w.size() && (w[i] >> 16);) {
        const std::uint32_t len = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 16 /* OpExecutionMode */ && len == 4 && w[i + 2] == mode && w[i + 3] == 16) return true;
        i += len;
    }
    return false;
}

void exp_mrt0_compr(Asm& a, std::uint32_t v0, std::uint32_t v1) {  // exp mrt0, v0, v1 compr done vm
    a.w.push_back(0xf8001c0fu);
    a.w.push_back(v0 | v1 << 8);
}

// One RTZ lift through the printer and DXC: the modes and their capabilities
// in the source, the lift's conversions in it, DXC's module with the modes and
// at least `conversions` conversions to half.
void rtz_through_hlsl(const gcn::LiftResult& lift, const std::string& what, std::vector<std::string> spellings, std::size_t conversions) {
    CHECK(lift.ok());
    if (!lift.ok()) {
        std::fprintf(stderr, "%s: not lifted: %s\n", what.c_str(), lift.rejections.empty() ? "?" : lift.rejections[0].c_str());
        return;
    }
    CHECK(has_mode16(lift.spirv, spv::ExRoundingModeRTZ) && has_mode16(lift.spirv, spv::ExDenormPreserve));
    spellings.push_back("[[vk::ext_capability(4464 /* DenormPreserve */)]]");
    spellings.push_back("[[vk::ext_capability(4468 /* RoundingModeRTZ */)]]");
    spellings.push_back("vk::ext_execution_mode(4459, 16);");
    spellings.push_back("vk::ext_execution_mode(4463, 16);");
    gcn::HlslResult h;
    std::vector<std::uint32_t> dxc;
    through_hlsl(lift, what, spellings, false, &h, &dxc);
    const bool halves = h.source.find("float16_t") != std::string::npos;
    const bool sixteen = std::find(h.arguments.begin(), h.arguments.end(), "-enable-16bit-types") != h.arguments.end();
    CHECK(halves == (conversions > 0));
    CHECK(sixteen == halves);
    if (!dxc.empty()) {
        CHECK(has_mode16(dxc, spv::ExRoundingModeRTZ) && has_mode16(dxc, spv::ExDenormPreserve));
        const std::size_t n = half_conversions(dxc);
        CHECK(n >= conversions);
        if (n < conversions) std::fprintf(stderr, "%s: DXC's module converts to half %zu times, the lift's output needs %zu\n", what.c_str(), n, conversions);
    }
}

void native_half_rtz() {
    gcn::set_native_half_rtz(true);
    // Two packs exported compressed in their block: f32 -> f16 -> f32 for each
    // pair - and the same with early fragment tests, whose HLSL attribute
    // follows the modes' capabilities.
    for (const bool early : {false, true}) {
        Asm a;
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop2(47, 4, kV + 2, 3);  // v_cvt_pkrtz_f16_f32 v4, v2, v3
        a.vop2(47, 5, kV + 3, 2);  // v_cvt_pkrtz_f16_f32 v5, v3, v2
        exp_mrt0_compr(a, 4, 5);
        a.w.push_back(kEndpgm);
        std::vector<std::string> spell = {"spv_to_half(", "spv_from_half("};
        if (early) spell.push_back("[[vk::ext_capability(4468 /* RoundingModeRTZ */)]]\n[earlydepthstencil]\n");
        rtz_through_hlsl(lift_ps(a.w, 1, early), std::string("RTZ: two packs exported compressed") + (early ? ", early fragment tests" : ""), spell, 2);
    }
    // A word the program did not pack (user data) exported compressed beside a
    // pack: the word's halves decoded through a bitcast.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(1, 4, 0);           // v_mov_b32 v4, s0
        a.vop2(47, 5, kV + 2, 3);  // v_cvt_pkrtz_f16_f32 v5, v2, v3
        exp_mrt0_compr(a, 4, 5);
        a.w.push_back(kEndpgm);
        rtz_through_hlsl(lift_ps(a.w), "RTZ: a word and a pack exported compressed", {"spv_bitcast(", "spv_from_half(", "spv_to_half("}, 1);
    }
    // A pack under a partial EXEC: rounded where it was written, the old word decoded elsewhere.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(1, 4, 0);           // v4 = s0: the old word
        a.vopc(1, kZero, 2);       // v_cmp_lt_f32 vcc, 0, v2
        a.sop1(36, 22, kVcc);      // s_and_saveexec_b64 s[22:23], vcc
        a.vop2(47, 4, kV + 2, 3);  // v4 = pack(v2, v3) where 0 < v2
        a.sop1(4, kExec, 22);      // s_mov_b64 exec, s[22:23]
        exp_mrt0_compr(a, 4, 4);
        a.w.push_back(kEndpgm);
        rtz_through_hlsl(lift_ps(a.w), "RTZ: a pack under a partial EXEC", {"spv_bitcast(", "spv_from_half(", "spv_to_half("}, 1);
    }
    // The packed word as an integer: the halves' bits (v_cvt_f32_u32 of it, exported).
    {
        Asm a;
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop2(47, 4, kV + 2, 3);  // v_cvt_pkrtz_f16_f32 v4, v2, v3
        a.vop1(6, 5, kV + 4);      // v_cvt_f32_u32 v5, v4
        a.vop1(1, 6, kV + 5);
        a.vop1(1, 7, kV + 5);
        a.vop1(1, 8, kV + 5);
        a.exp_mrt0(5);
        a.w.push_back(kEndpgm);
        rtz_through_hlsl(lift_ps(a.w), "RTZ: a packed word read as an integer", {"spv_to_half(", "spv_bitcast("}, 1);
    }
    // A vertex shader with no half math (vertex_invariant's program): the
    // modes all the same, no half type in the HLSL.
    {
        std::vector<std::uint32_t> w = {0xbe800000u | 0u << 16 | 33u << 8 | 2u};  // s_swappc_b64 s[0:1], s[2:3]
        const auto exp = [&](std::uint32_t tgt, std::uint32_t v, bool done) {
            w.push_back(0xf8000000u | (done ? 1u << 11 : 0u) | tgt << 4 | 0xfu);
            w.push_back((v + 3) << 24 | (v + 2) << 16 | (v + 1) << 8 | v);
        };
        exp(12, 4, true);
        exp(32, 8, false);
        w.push_back(kEndpgm);
        const gcn::Program p = gcn::decode(w.data(), w.size());
        CHECK(p.errors.empty());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Vertex;
        o.rsrc1 = 3u << 24;
        o.descriptor_set = 0;
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.link_outputs = true;
        for (std::uint8_t r = 0; r < 32; ++r) o.output_links.push_back(r);
        o.vertex_input = {{0, 4, 4, 0u}, {1, 8, 4, 0u}};
        o.vertex_formats_from_params = true;
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok());
        rtz_through_hlsl(gcn::lift_vertex_shader(p, o, ref), "RTZ: a vertex shader with no half math", {"SV_Position"}, 0);
    }
    // What the printer does not take from such a lift: the modes for 32-bit
    // floats, a conversion decorated NoContraction (DXC cannot decorate one),
    // arithmetic on halves (DXC could evaluate it itself, rounding to nearest).
    {
        Asm a;
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop2(47, 4, kV + 2, 3);
        exp_mrt0_compr(a, 4, 4);
        a.w.push_back(kEndpgm);
        const gcn::LiftResult l = lift_ps(a.w);
        CHECK(l.ok());
        const auto refused = [&](const std::vector<std::uint32_t>& m, const char* why, const char* what) {
            const gcn::HlslResult h = gcn::emit_hlsl(m);
            const bool ok = !h.ok() && !h.errors.empty() && h.errors[0].find(why) != std::string::npos;
            CHECK(ok);
            if (!ok) std::fprintf(stderr, "RTZ: %s: printed, or refused for another reason: %s\n", what, h.errors.empty() ? "printed" : h.errors[0].c_str());
        };
        std::vector<std::uint32_t> wide = l.spirv, exact = l.spirv;
        for (std::size_t i = 5; i < wide.size() && (wide[i] >> 16);) {
            const std::uint32_t len = wide[i] >> 16;
            if ((wide[i] & 0xffff) == 16 && len == 4 && wide[i + 2] == spv::ExRoundingModeRTZ) wide[i + 3] = 32;
            i += len;
        }
        refused(wide, "float-controls execution mode", "RoundingModeRTZ 32");
        // OpDecorate <the first conversion> NoContraction, among the module's decorations.
        std::uint32_t conv = 0;
        std::size_t first_decoration = 0;
        for (std::size_t i = 5; i < exact.size() && (exact[i] >> 16);) {
            const std::uint32_t len = exact[i] >> 16, op = exact[i] & 0xffff;
            if (op == 71 && !first_decoration) first_decoration = i;
            if (op == 115 && !conv) conv = exact[i + 2];
            i += len;
        }
        CHECK(conv && first_decoration);
        if (conv && first_decoration) {
            const std::uint32_t d[] = {3u << 16 | 71u, conv, spv::DecNoContraction};
            exact.insert(exact.begin() + static_cast<std::ptrdiff_t>(first_decoration), std::begin(d), std::end(d));
            refused(exact, "NoContraction", "a conversion decorated NoContraction");
        }
        // A module of its own: 0.7 to half and back, as a pixel's colour, under the device's rounding
        // (0.7 is one where rounding to nearest and toward zero differ) - and the same with an add of halves.
        for (const bool add : {false, true}) {
            spv::Module m;
            m.capability(spv::CapShader);
            m.capability(spv::CapFloat16);
            m.capability(spv::CapDenormPreserve);
            m.capability(spv::CapRoundingModeRTZ);
            m.memory_model(0, 1);
            const spv::Id t_void = m.type_void(), t_f32 = m.type_float(32), t_v4f = m.type_vector(t_f32, 4), t_h = m.type_float(16);
            const spv::Id out = m.global_variable(m.type_pointer(spv::ScOutput, t_v4f), spv::ScOutput);
            m.decorate(out, spv::DecLocation, {0});
            const spv::Id fn = m.begin_function(t_void, m.type_function(t_void, {}));
            m.label();
            spv::Id h = m.emit(spv::OpFConvert, t_h, {m.const_f32(0.7f)});
            if (add) h = m.emit(spv::OpFAdd, t_h, {h, h});
            const spv::Id f = m.emit(spv::OpFConvert, t_f32, {h});
            m.store(out, m.emit(spv::OpCompositeConstruct, t_v4f, {f, f, f, f}));
            m.emit_void(spv::OpReturn, {});
            m.end_function();
            m.entry_point(spv::EmFragment, fn, "main", {out});
            m.execution_mode(fn, spv::ExOriginUpperLeft);
            m.execution_mode(fn, spv::ExDenormPreserve, {16});
            m.execution_mode(fn, spv::ExRoundingModeRTZ, {16});
            gcn::LiftResult c;
            c.spirv = m.assemble();
            if (add) {
                refused(c.spirv, "an operation on halves", "an add of halves");
            } else {
                // The constant's conversion stays the device's (no half constant in DXC's module: hlsl_operation_diff).
                rtz_through_hlsl(c, "RTZ: a constant to half and back", {"spv_to_half(0.7f)"}, 1);
            }
        }
        // A compute shader under the modes (the game's compute lifts never are:
        // their stores keep them off native rounding): a pack of 0.7 and 0.3
        // stored to a buffer word, as the lift's buffers are (a block of uint[]).
        {
            spv::Module m;
            m.capability(spv::CapShader);
            m.capability(spv::CapFloat16);
            m.capability(spv::CapDenormPreserve);
            m.capability(spv::CapRoundingModeRTZ);
            m.memory_model(0, 1);
            const spv::Id t_void = m.type_void(), t_u32 = m.type_int(32, false), t_f32 = m.type_float(32);
            const spv::Id t_v2f = m.type_vector(t_f32, 2), t_v2h = m.type_vector(m.type_float(16), 2);
            const spv::Id t_rt = m.type_runtime_array(t_u32);
            m.decorate(t_rt, spv::DecArrayStride, {4});
            const spv::Id t_block = m.type_struct({t_rt});
            m.decorate(t_block, spv::DecBlock);
            m.member_decorate(t_block, 0, spv::DecOffset, {0});
            const spv::Id buf = m.global_variable(m.type_pointer(spv::ScStorageBuffer, t_block), spv::ScStorageBuffer);
            m.decorate(buf, spv::DecDescriptorSet, {0});
            m.decorate(buf, spv::DecBinding, {0});
            const spv::Id fn = m.begin_function(t_void, m.type_function(t_void, {}));
            m.label();
            const spv::Id pair = m.emit(spv::OpCompositeConstruct, t_v2f, {m.const_f32(0.7f), m.const_f32(0.3f)});
            const spv::Id word = m.emit(spv::OpBitcast, t_u32, {m.emit(spv::OpFConvert, t_v2h, {pair})});
            m.store(m.access_chain(m.type_pointer(spv::ScStorageBuffer, t_u32), buf, {m.const_u32(0), m.const_u32(0)}), word);
            m.emit_void(spv::OpReturn, {});
            m.end_function();
            m.entry_point(spv::EmGLCompute, fn, "main", {buf});
            m.execution_mode(fn, spv::ExLocalSize, {64, 1, 1});
            m.execution_mode(fn, spv::ExDenormPreserve, {16});
            m.execution_mode(fn, spv::ExRoundingModeRTZ, {16});
            gcn::LiftResult c;
            c.spirv = m.assemble();
            rtz_through_hlsl(c, "RTZ: a compute shader storing a pack", {"[numthreads(64, 1, 1)]", "spv_to_half(", "spv_bitcast("}, 1);
        }
    }
    gcn::set_native_half_rtz(false);
}

}  // namespace

// A typed load by index (tbuffer_load_format_xyzw idxen) from a V# in a
// table, with and without its check against the V#'s num_records
// (gcn::set_index_bounds): printed, compiled by DXC, the interface and
// operations the lift's.
void typed_load_bounds() {
    std::vector<std::uint32_t> w;
    w.push_back(0xc0000000u | 2u << 22 | 8u << 15 | (2u / 2) << 9 | 1u << 8 | 4u);  // s_load_dwordx4 s[8:11], s[2:3], 0x4: the V#
    w.push_back(0xbf8c007fu);                                                       // s_waitcnt lgkmcnt(0)
    w.push_back(0x7e000000u | 4u << 17 | 1u << 9 | 1u);                             // v_mov_b32 v4, s1: the element's index
    w.push_back(0xe8000000u | 7u << 23 | 14u << 19 | 3u << 16 | 1u << 13);           // tbuffer_load_format_xyzw v[0:3], v4, s[8:11] idxen
    w.push_back(4u | 0u << 8 | (8u / 4) << 16 | 128u << 24);
    w.push_back(0xbf8c0f70u);                                                       // s_waitcnt vmcnt(0)
    w.push_back(0xf8000000u | 1u << 12 | 1u << 11 | 0xfu);                          // exp mrt0 v0, v1, v2, v3 done vm
    w.push_back(0u | 1u << 8 | 2u << 16 | 3u << 24);
    w.push_back(kEndpgm);
    for (const bool on : {true, false}) {
        gcn::set_index_bounds(on);
        const gcn::Program p = gcn::decode(w.data(), w.size());
        CHECK(p.errors.empty());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Pixel;
        o.rsrc2 = 4u << 1;
        o.ps_input_ena = 2;
        o.descriptor_set = 1;
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok() && ref.buffers.size() == 2 && ref.buffers[1].indexed);
        through_hlsl(gcn::lift_pixel_shader(p, o, ref), on ? "typed load by index, checked against its records" : "typed load by index");
    }
    gcn::set_index_bounds(false);
}

// gcn::set_image_store_bounds: a compute shader's image_store with its
// lanes outside the view's size skipped - the storage image's size printed as
// GetDimensions, the write under the compare - and without the skip, through
// DXC alike.
void image_store_bounds() {
    std::vector<std::uint32_t> w;
    w.push_back(0x7e000000u | 1u << 17 | 1u << 9 | 5u);                 // v_mov_b32 v1, s5: y
    w.push_back(0x7e000000u | 4u << 17 | 1u << 9 | (kV + 0));           // v_mov_b32 v4, v0
    for (std::uint32_t k = 5; k < 8; ++k) w.push_back(0x7e000000u | k << 17 | 1u << 9 | 128u);  // v_mov_b32 vk, 0
    w.push_back(0xf0000000u | 8u << 18 | 0xfu << 8);                    // image_store v[4:7], v[0:1], s[8:15] dmask 0xf
    w.push_back(0u | 4u << 8 | (8u / 4) << 16);
    w.push_back(kEndpgm);
    for (const bool on : {true, false}) {
        gcn::set_image_store_bounds(on);
        const gcn::Program p = gcn::decode(w.data(), w.size());
        CHECK(p.errors.empty());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Compute;
        o.rsrc2 = 16u << 1;
        o.cs_threads[0] = 64;
        o.image_dims = {{1 /* 2D */, false}};
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok());
        through_hlsl(gcn::lift_compute_shader(p, o, ref), on ? "image store, outside the view skipped" : "image store",
                     on ? std::vector<std::string>{"GetDimensions("} : std::vector<std::string>{});
    }
    gcn::set_image_store_bounds(true);
}

int main() {
    g_have_dxc = g_dxc.load();
    // A library named by BBHOST_HLSL_DXC must load: the continuous-integration
    // build names the release DXC the packages carry, which this test then
    // compiles every lift with.
    const char* wanted = std::getenv("BBHOST_HLSL_DXC");
    if (!g_have_dxc && wanted && *wanted) {
        std::fprintf(stderr, "gcn_hlsl_test: BBHOST_HLSL_DXC=%s did not load: %s\n", wanted, g_dxc.error().c_str());
        ++g_failures;
    }
    if (!g_have_dxc) std::printf("gcn_hlsl_test: %s: printing only\n", g_dxc.error().c_str());
    else std::printf("gcn_hlsl_test: DXC %s from %s\n", g_dxc.version().c_str(), g_dxc.path().c_str());
    swizzles();
    cndmask();
    samples_and_loops();
    vertex_invariant();
    clip_prologue();
    typed_load_bounds();
    image_store_bounds();
    native_half_rtz();
    std::printf("gcn_hlsl_test: %d lifts printed, %d compiled by DXC%s\n", g_printed, g_compiled, g_have_dxc ? "" : " (no DXC)");
    if (g_failures) {
        std::fprintf(stderr, "gcn_hlsl_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("gcn_hlsl_test ok\n");
    return 0;
}
