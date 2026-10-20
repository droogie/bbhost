// The pilot pixel shader (a22c7f71) lifts from the shipped bundle, and the lifter
// refuses the constructs it cannot prove equal. Skips when the dump is absent.
// Before that, hand-encoded programs check ds_swizzle_b32 (every quad pattern
// run through a small interpreter of the lifted SPIR-V over one quad, and the
// cases the lifter must refuse), v_cndmask_b32's source modifiers (through the
// same interpreter) and packed exports, texture LOD and whole-quad mode,
// loops, regions and ifs and the wave-wide tests of masks they rest on, and
// lane writes; they run without the dump.
#include "test_app0.h"
#include "gcn/container.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/hlsl.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
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

bool spirv_valid(const std::vector<std::uint32_t>& words) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    const bool valid = tools.Validate(words.data(), words.size(), vo);
    if (!valid) std::fprintf(stderr, "SPIR-V invalid: %s\n", msg.c_str());
    return valid;
}

// ---- hand-encoded pixel shaders (Sea Islands encodings) -------------------------------
std::uint32_t sopp(std::uint32_t op, std::int16_t simm) { return 0xbf800000u | op << 16 | static_cast<std::uint16_t>(simm); }
std::uint32_t sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc) { return 0xbe800000u | sdst << 16 | op << 8 | ssrc; }
std::uint32_t sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t ssrc1) {
    return 0x80000000u | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0;
}
std::uint32_t sopc(std::uint32_t op, std::uint32_t ssrc0, std::uint32_t ssrc1) { return 0xbf000000u | op << 16 | ssrc1 << 8 | ssrc0; }
std::uint32_t vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { return 0x7e000000u | vdst << 17 | op << 9 | src0; }
std::uint32_t vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t vsrc1, std::uint32_t src0) { return op << 25 | vdst << 17 | vsrc1 << 9 | src0; }
std::uint32_t vopc(std::uint32_t op, std::uint32_t vsrc1, std::uint32_t src0) { return 0x7c000000u | op << 17 | vsrc1 << 9 | src0; }
std::uint32_t vintrp(std::uint32_t op, std::uint32_t vdst, std::uint32_t attr, std::uint32_t chan, std::uint32_t vsrc) {
    return 0xc8000000u | vdst << 18 | op << 16 | attr << 10 | chan << 8 | vsrc;
}
constexpr std::uint32_t kEndpgm = 0xbf810000u;
constexpr std::uint32_t kExec = 126, kVcc = 106, kZero = 128, kOne = 242, kV = 256;  // operand codes; 242 is 1.0
constexpr std::uint32_t kMovB64 = 4, kWqmB64 = 10, kAndSaveexecB64 = 36, kAndB64 = 15;
constexpr std::uint32_t kVMov = 1, kVSub = 4, kVCndmask = 0, kVCmpGtF32 = 4;

void push(std::vector<std::uint32_t>& w, std::initializer_list<std::uint32_t> words) { w.insert(w.end(), words); }
// ds_swizzle_b32 vdst, vaddr offset:off, with DATA0 (unused by the instruction) and GDS as given.
void swizzle(std::vector<std::uint32_t>& w, std::uint32_t vdst, std::uint32_t vaddr, std::uint32_t off, std::uint32_t data0 = 0, bool gds = false) {
    push(w, {0xd8000000u | 53u << 18 | (gds ? 1u << 17 : 0u) | off, vdst << 24 | data0 << 8 | vaddr});
}
// exp mrt0 with the four VGPRs, done, vm, not compressed.
void export_mrt0(std::vector<std::uint32_t>& w, std::uint32_t v0, std::uint32_t v1, std::uint32_t v2, std::uint32_t v3) {
    push(w, {0xf8000000u | 1u << 12 | 1u << 11 | 0xfu, v3 << 24 | v2 << 16 | v1 << 8 | v0});
}
// The prologue of the game's swizzling programs: save the coverage, whole-quad
// mode, v2 = attr0.x (from the barycentrics in v0 and v1).
void wqm_prologue(std::vector<std::uint32_t>& w) {
    push(w, {sop1(kMovB64, 2, kExec), sop1(kWqmB64, kExec, kExec), vintrp(0, 2, 0, 0, 0), vintrp(1, 2, 0, 0, 1)});
}
std::vector<std::uint32_t> one_swizzle(std::uint32_t off) {
    std::vector<std::uint32_t> w;
    wqm_prologue(w);
    swizzle(w, 3, 2, off);
    push(w, {sopp(12, 0xc07f) /* s_waitcnt lgkmcnt(0) */, sop1(kMovB64, kExec, 2)});
    export_mrt0(w, 3, 3, 3, 3);
    w.push_back(kEndpgm);
    return w;
}

gcn::TranslateOptions synthetic_options(int user_sgprs = 0) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc2 = static_cast<std::uint32_t>(user_sgprs) << 1;
    o.ps_input_ena = 2;  // PERSP_CENTER: the barycentrics in v0 and v1
    o.descriptor_set = 1;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    return o;
}
struct Lifted {
    gcn::TranslateResult ref;
    gcn::LiftResult lift;
};
Lifted lift_words(const std::vector<std::uint32_t>& words, const gcn::TranslateOptions& o = synthetic_options()) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    Lifted r;
    r.ref = gcn::translate(p, o);
    r.lift = gcn::lift_pixel_shader(p, o, r.ref);
    return r;
}

// ---- a straight-line SPIR-V interpreter over one quad ---------------------------------
// Runs the lifted fragment shader for the four invocations of a quad whose
// lane 0 sits at framebuffer (x0, y0) - lane 1 right of it, lane 2 below - with
// attr0.x = the given values, and returns each invocation's mrt0.x. It knows
// only the instructions a swizzle program lifts to; anything else fails.
struct QuadRun {
    bool ok = false;
    std::string error;
    std::array<float, 4> out{};
};
QuadRun run_quad(const std::vector<std::uint32_t>& spv, float x0, float y0, const std::array<float, 4>& attr0) {
    QuadRun run;
    using Vec = std::array<std::uint32_t, 4>;
    std::map<std::uint32_t, std::array<Vec, 4>> val;  // id -> per-invocation value
    std::map<std::uint32_t, std::uint32_t> constants, builtin, location, storage;
    std::map<std::uint32_t, std::array<float, 4>> outputs;
    const auto fail = [&](const std::string& why) {
        run.error = why;
        return run;
    };
    bool in_body = false;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t n = spv[i] >> 16, op = spv[i] & 0xffff;
        if (n == 0 || i + n > spv.size()) return fail("malformed module");
        const std::uint32_t* a = &spv[i + 1];
        i += n;
        switch (op) {
        case 71:  // OpDecorate
            if (a[1] == 11) builtin[a[0]] = a[2];
            if (a[1] == 30) location[a[0]] = a[2];
            continue;
        case 59: storage[a[1]] = a[2]; continue;  // OpVariable
        case 43: constants[a[1]] = a[2]; continue;  // OpConstant (low word)
        case 41: constants[a[1]] = 1; continue;     // OpConstantTrue
        case 42: constants[a[1]] = 0; continue;     // OpConstantFalse
        case 248: in_body = true; continue;         // OpLabel
        default: break;
        }
        if (!in_body) continue;  // types, names, capabilities, the entry point
        const auto get = [&](std::uint32_t id, int lane) -> Vec {
            if (const auto c = constants.find(id); c != constants.end()) return {c->second, 0, 0, 0};
            return val[id][static_cast<std::size_t>(lane)];
        };
        const auto f = [](std::uint32_t b) { return std::bit_cast<float>(b); };
        switch (op) {
        case 61:  // OpLoad
            for (int l = 0; l < 4; ++l) {
                if (builtin.count(a[2]) && builtin[a[2]] == 15) {  // FragCoord at the pixel's centre
                    const float x = x0 + static_cast<float>(l & 1) + 0.5f, y = y0 + static_cast<float>(l >> 1) + 0.5f;
                    val[a[1]][static_cast<std::size_t>(l)] = {std::bit_cast<std::uint32_t>(x), std::bit_cast<std::uint32_t>(y), 0, 0};
                } else if (storage[a[2]] == 1 && location.count(a[2]) && location[a[2]] == 0) {
                    val[a[1]][static_cast<std::size_t>(l)] = {std::bit_cast<std::uint32_t>(attr0[static_cast<std::size_t>(l)]), 0, 0, 0};
                } else {
                    return fail("a load of something other than FragCoord or attr0");
                }
            }
            break;
        case 62:  // OpStore
            if (storage[a[0]] != 3 || !location.count(a[0]) || location[a[0]] != 0) return fail("a store to something other than mrt0");
            for (int l = 0; l < 4; ++l) outputs[0][static_cast<std::size_t>(l)] = f(get(a[1], l)[0]);
            break;
        case 81:  // OpCompositeExtract
            for (int l = 0; l < 4; ++l) val[a[1]][static_cast<std::size_t>(l)] = {get(a[2], l)[a[3]], 0, 0, 0};
            break;
        case 80:  // OpCompositeConstruct
            for (int l = 0; l < 4; ++l) {
                Vec v{};
                for (std::uint32_t k = 2; k < n - 1 && k < 6; ++k) v[k - 2] = get(a[k], l)[0];
                val[a[1]][static_cast<std::size_t>(l)] = v;
            }
            break;
        case 124:  // OpBitcast
            for (int l = 0; l < 4; ++l) val[a[1]][static_cast<std::size_t>(l)] = get(a[2], l);
            break;
        case 131: case 129: case 133:  // OpFSub, OpFAdd, OpFMul
            for (int l = 0; l < 4; ++l) {
                const float x = f(get(a[2], l)[0]), y = f(get(a[3], l)[0]);
                const float r = op == 131 ? x - y : op == 129 ? x + y : x * y;
                val[a[1]][static_cast<std::size_t>(l)] = {std::bit_cast<std::uint32_t>(r), 0, 0, 0};
            }
            break;
        case 186: case 184:  // OpFOrdGreaterThan, OpFOrdLessThan
            for (int l = 0; l < 4; ++l) {
                const float x = f(get(a[2], l)[0]), y = f(get(a[3], l)[0]);
                val[a[1]][static_cast<std::size_t>(l)] = {(op == 186 ? x > y : x < y) ? 1u : 0u, 0, 0, 0};
            }
            break;
        case 156:  // OpIsNan
            for (int l = 0; l < 4; ++l) val[a[1]][static_cast<std::size_t>(l)] = {f(get(a[2], l)[0]) != f(get(a[2], l)[0]) ? 1u : 0u, 0, 0, 0};
            break;
        case 166: case 167: case 197: case 198: case 199:  // OpLogicalOr, OpLogicalAnd, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd
            for (int l = 0; l < 4; ++l) {
                const std::uint32_t x = get(a[2], l)[0], y = get(a[3], l)[0];
                const std::uint32_t r = op == 166 || op == 197 ? x | y : op == 167 || op == 199 ? x & y : x ^ y;
                val[a[1]][static_cast<std::size_t>(l)] = {r, 0, 0, 0};
            }
            break;
        case 168:  // OpLogicalNot
            for (int l = 0; l < 4; ++l) val[a[1]][static_cast<std::size_t>(l)] = {get(a[2], l)[0] ? 0u : 1u, 0, 0, 0};
            break;
        case 169:  // OpSelect
            for (int l = 0; l < 4; ++l) val[a[1]][static_cast<std::size_t>(l)] = get(a[2], l)[0] ? get(a[3], l) : get(a[4], l);
            break;
        case 365: case 366: {  // OpGroupNonUniformQuadBroadcast / QuadSwap: quad index = lane
            if (constants[a[2]] != 3) return fail("a quad operation outside subgroup scope");
            const std::uint32_t k = constants.at(a[4]);
            for (int l = 0; l < 4; ++l) {
                const int from = op == 365 ? static_cast<int>(k) : l ^ static_cast<int>(k + 1);
                val[a[1]][static_cast<std::size_t>(l)] = get(a[3], from);
            }
            break;
        }
        case 253: case 56: break;  // OpReturn, OpFunctionEnd
        default: return fail("unsupported opcode " + std::to_string(op));
        }
    }
    if (!outputs.count(0)) return fail("mrt0 is not written");
    for (int l = 0; l < 4; ++l) run.out[static_cast<std::size_t>(l)] = outputs[0][static_cast<std::size_t>(l)];
    run.ok = true;
    return run;
}

std::size_t count_op(const std::vector<std::uint32_t>& spv, std::uint32_t opcode) {
    std::size_t n = 0;
    for (std::size_t i = 5; i < spv.size() && (spv[i] >> 16);) {
        n += (spv[i] & 0xffff) == opcode;
        i += spv[i] >> 16;
    }
    return n;
}

// Each lane of a quad reads lane sel[k]; the lifted shader, run over a quad at an
// odd origin (Vulkan does not promise even ones), must return attr0 of that lane.
void check_quad_pattern(std::uint32_t off, const std::array<std::uint32_t, 4>& sel, const char* what) {
    const Lifted l = lift_words(one_swizzle(off));
    for (const std::string& why : l.lift.rejections) std::fprintf(stderr, "%s (offset %04x) rejected: %s\n", what, off, why.c_str());
    CHECK(l.ref.ok() && l.lift.ok());
    if (!l.lift.ok()) return;
    CHECK(spirv_valid(l.lift.spirv));
    const std::array<float, 4> attr = {10.0f, 21.0f, 32.0f, 43.0f};
    for (const auto [x0, y0] : {std::pair{0.0f, 0.0f}, std::pair{7.0f, 3.0f}}) {
        const QuadRun r = run_quad(l.lift.spirv, x0, y0, attr);
        if (!r.ok) std::fprintf(stderr, "%s (offset %04x): %s\n", what, off, r.error.c_str());
        CHECK(r.ok);
        for (std::size_t k = 0; k < 4 && r.ok; ++k) {
            if (r.out[k] != attr[sel[k]]) {
                std::fprintf(stderr, "%s (offset %04x): lane %zu read %g, expected lane %u's %g\n", what, off, k, r.out[k], sel[k], attr[sel[k]]);
                ++g_failures;
            }
        }
    }
}

void synthetic_swizzles() {
    // Every quad-mode pattern: lane k reads lane offset[2k+1:2k].
    for (std::uint32_t pattern = 0; pattern < 256; ++pattern) {
        check_quad_pattern(0x8000u | pattern, {pattern & 3, (pattern >> 2) & 3, (pattern >> 4) & 3, (pattern >> 6) & 3}, "quad mode");
    }
    // Bit-mask mode where the masks keep each lane in its quad: lane i reads ((i & and) | or) ^ xor.
    for (std::uint32_t and_lo = 0; and_lo < 4; ++and_lo) {
        for (std::uint32_t or_mask = 0; or_mask < 4; ++or_mask) {
            for (std::uint32_t xor_mask = 0; xor_mask < 4; ++xor_mask) {
                const std::uint32_t and_mask = 0x1c | and_lo;
                std::array<std::uint32_t, 4> sel{};
                for (std::uint32_t k = 0; k < 4; ++k) sel[k] = (((k & and_mask) | or_mask) ^ xor_mask) & 3;
                check_quad_pattern(and_mask | or_mask << 5 | xor_mask << 10, sel, "bit-mask mode within a quad");
            }
        }
    }

    // The game's derivative idioms: the coarse ones are broadcasts, the fine ones swaps.
    {
        const Lifted coarse = lift_words(one_swizzle(0x8055));  // lane 1 everywhere
        CHECK(coarse.lift.ok() && count_op(coarse.lift.spirv, 365) == 1 && count_op(coarse.lift.spirv, 366) == 0);
        const Lifted across = lift_words(one_swizzle(0x80b1));  // [1,0,3,2]: a horizontal swap, no place in the quad needed
        CHECK(across.lift.ok() && count_op(across.lift.spirv, 366) == 1 && count_op(across.lift.spirv, 169) == 0);
        const Lifted same = lift_words(one_swizzle(0x80e4));  // [0,1,2,3]: every lane reads itself
        CHECK(same.lift.ok() && count_op(same.lift.spirv, 365) + count_op(same.lift.spirv, 366) == 0);
        bool proof = false;
        for (const std::string& p : coarse.lift.proof) proof |= p.find("1 ds_swizzle_b32") != std::string::npos;
        CHECK(proof);
    }
    // A value every lane holds alike needs no quad operation.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        w.push_back(vop1(kVMov, 4, kOne));
        swizzle(w, 3, 4, 0x80f5);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        const Lifted l = lift_words(w);
        CHECK(l.lift.ok() && count_op(l.lift.spirv, 365) + count_op(l.lift.spirv, 366) == 0);
    }
    // The operand is the VGPR in the address field; DATA0 is not read, by the
    // translator either.
    {
        std::vector<std::uint32_t> a, b, c;
        for (auto* w : {&a, &b, &c}) wqm_prologue(*w);
        a.push_back(vop1(kVMov, 5, kV + 2));
        b.push_back(vop1(kVMov, 5, kV + 2));
        c.push_back(vop1(kVMov, 5, kV + 2));
        swizzle(a, 3, 2, 0x8000, 0);
        swizzle(b, 3, 2, 0x8000, 5);  // another DATA0
        swizzle(c, 3, 5, 0x8000, 0);  // another address VGPR (the same value)
        for (auto* w : {&a, &b, &c}) {
            push(*w, {sop1(kMovB64, kExec, 2)});
            export_mrt0(*w, 3, 3, 3, 3);
            w->push_back(kEndpgm);
        }
        const Lifted la = lift_words(a), lb = lift_words(b), lc = lift_words(c);
        CHECK(la.ref.ok() && lb.ref.ok() && la.ref.spirv == lb.ref.spirv);
        CHECK(la.lift.ok() && lb.lift.ok() && la.lift.spirv == lb.lift.spirv);
        // The translation of the swizzle of v0 (the barycentric DATA0 named) is another shader.
        std::vector<std::uint32_t> d = a;
        d[d.size() - 5] = 3u << 24 | 0u << 8 | 0u;  // ds_swizzle_b32 v3, v0
        CHECK(gcn::translate(gcn::decode(d.data(), d.size()), synthetic_options()).spirv != la.ref.spirv);
        CHECK(lc.lift.ok());
    }

    // ---- refused ---------------------------------------------------------------------
    const auto refused = [](const std::vector<std::uint32_t>& w, const char* text, const gcn::TranslateOptions& o = synthetic_options()) {
        const Lifted l = lift_words(w, o);
        CHECK(l.ref.ok());
        CHECK(rejected_with(l.lift, text));
    };
    // Lanes outside the quad: bit-mask mode reaching across quads, and the
    // rotate/FFT encodings of later chips.
    refused(one_swizzle(0x1f | 4u << 10), "reads lanes outside the quad");
    refused(one_swizzle(0x0f), "reads lanes outside the quad");
    refused(one_swizzle(0xc0e4), "bits 14:8");
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        swizzle(w, 3, 2, 0x8000, 0, true);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "GDS");
    }
    // Without whole-quad mode the helper lanes are inactive on GCN and read as 0.
    {
        std::vector<std::uint32_t> w = {vintrp(0, 2, 0, 0, 0), vintrp(1, 2, 0, 0, 1)};
        swizzle(w, 3, 2, 0x8000);
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "outside whole-quad mode");
    }
    // ... nor after the coverage is restored.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        w.push_back(sop1(kMovB64, kExec, 2));
        swizzle(w, 3, 2, 0x8000);
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "outside whole-quad mode");
    }
    // A value written before whole-quad mode: GCN's helper lanes never computed it.
    {
        std::vector<std::uint32_t> w = {sop1(kMovB64, 2, kExec), vintrp(0, 2, 0, 0, 0), vintrp(1, 2, 0, 0, 1), sop1(kWqmB64, kExec, kExec)};
        swizzle(w, 3, 2, 0x8000);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "may hold differently");
    }
    // A value selected by the saved coverage: 0 in GCN's helper lanes, 1.0 here.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        push(w, {vop1(kVMov, 4, kOne), sop1(kMovB64, kVcc, 2), vop2(kVCndmask, 5, 4, kZero)});
        swizzle(w, 3, 5, 0x8000);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "may hold differently");
    }
    // The same through an if: the then arm writes v2 under the coverage.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        push(w, {sopc(6, 0, kZero) /* s_cmp_eq_u32 s0, 0 */, sopp(4, 3) /* s_cbranch_scc0 +3 */, sop1(kMovB64, kExec, 2),
                 vop1(kVMov, 2, kOne), sop1(kWqmB64, kExec, kExec)});
        swizzle(w, 3, 2, 0x8000);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "may hold differently", synthetic_options(1));
    }
    // Inside a region where EXEC is a per-pixel mask: an inactive lane reads as 0.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        push(w, {vopc(kVCmpGtF32, 2, kZero) /* vcc = 0 > v2 */, sop1(kAndSaveexecB64, 4, kVcc), sopp(8, 2) /* s_cbranch_execz +2 */});
        swizzle(w, 3, 2, 0x8000);
        push(w, {sop1(kMovB64, kExec, 4), sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "EXEC is not known set");
    }
    // After a kill region that leaves the lift different from GCN in the pixels
    // it kills, a neighbour would read them.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        push(w, {vopc(kVCmpGtF32, 2, kZero), sop2(kAndB64, 4, kVcc, kExec), sopp(4, 2) /* s_cbranch_scc0 +2 */,
                 sop2(kAndB64, kExec, kExec, 4), vop1(kVMov, 2, kOne), sop1(kMovB64, kExec, 193) /* -1 */});
        swizzle(w, 3, 2, 0x8000);
        push(w, {sop1(kMovB64, kExec, 4)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "after a kill region");
    }
    // A barycentric the lift does not model (v0 is never written).
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        swizzle(w, 3, 0, 0x8000);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        refused(w, "does not model");
    }
    // Derivative arithmetic on the swizzles stays valid SPIR-V with the subtraction kept.
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        swizzle(w, 3, 2, 0x8000);
        swizzle(w, 4, 2, 0x8055);
        swizzle(w, 5, 2, 0x80aa);
        push(w, {sopp(12, 0xc07f), vop2(kVSub, 4, 3, kV + 4), vop2(kVSub, 5, 3, kV + 5), sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 4, 5, 2, 2);
        w.push_back(kEndpgm);
        const Lifted l = lift_words(w);
        CHECK(l.lift.ok() && spirv_valid(l.lift.spirv) && count_op(l.lift.spirv, 365) == 3);
        if (l.lift.ok()) {
            const QuadRun r = run_quad(l.lift.spirv, 2.0f, 4.0f, {1.0f, 3.0f, 7.0f, 15.0f});
            CHECK(r.ok && r.out[0] == 2.0f && r.out[3] == 2.0f);  // ddx: lane 1 - lane 0, the same in every lane
        }
    }
}

// VOP3 v_cndmask_b32 applies its sources' abs and neg modifiers, abs first,
// as a float operation's (the game's own compiler writes |v7|, |v8| and -v10,
// -v9 forms, and LLVM folds negations into the instruction for this chip).
// x = attr0.x; vcc = 0 < x; v3 = vcc ? src1 : src0 with both sources v2.
std::vector<std::uint32_t> cndmask_program(std::uint32_t abs, std::uint32_t neg) {
    std::vector<std::uint32_t> w = {vintrp(0, 2, 0, 0, 0), vintrp(1, 2, 0, 0, 1)};
    w.push_back(vopc(1 /* v_cmp_lt_f32 */, 2, kZero));
    push(w, {0xd2000000u | abs << 8 | 3u, neg << 29 | kVcc << 18 | (kV + 2) << 9 | (kV + 2)});  // v_cndmask_b32 v3, v2, v2, vcc
    export_mrt0(w, 3, 3, 3, 3);
    w.push_back(kEndpgm);
    return w;
}
void cndmask_modifiers() {
    const std::array<float, 4> x = {-2.0f, 3.0f, -0.5f, 1.25f};
    const Lifted plain = lift_words(cndmask_program(0, 0));
    struct Case {
        std::uint32_t abs, neg;
        std::array<float, 4> expect;
        const char* what;
    };
    const Case cases[] = {
        {0, 0, {-2.0f, 3.0f, -0.5f, 1.25f}, "v2, v2"},
        {0, 1, {2.0f, 3.0f, 0.5f, 1.25f}, "-v2, v2"},
        {1, 0, {2.0f, 3.0f, 0.5f, 1.25f}, "|v2|, v2"},
        {0, 2, {-2.0f, -3.0f, -0.5f, -1.25f}, "v2, -v2 (the conditional negation)"},
        {3, 0, {2.0f, 3.0f, 0.5f, 1.25f}, "|v2|, |v2|"},
        {0, 3, {2.0f, -3.0f, 0.5f, -1.25f}, "-v2, -v2"},
        {1, 1, {-2.0f, 3.0f, -0.5f, 1.25f}, "-|v2|, v2 (abs, then neg)"},
    };
    for (const Case& c : cases) {
        const Lifted l = lift_words(cndmask_program(c.abs, c.neg));
        for (const std::string& why : l.lift.rejections) std::fprintf(stderr, "v_cndmask_b32 %s rejected: %s\n", c.what, why.c_str());
        CHECK(l.ref.ok() && l.lift.ok() && spirv_valid(l.lift.spirv));
        if (!l.lift.ok()) continue;
        const QuadRun r = run_quad(l.lift.spirv, 0.0f, 0.0f, x);
        if (!r.ok) std::fprintf(stderr, "v_cndmask_b32 %s: %s\n", c.what, r.error.c_str());
        CHECK(r.ok);
        for (std::size_t k = 0; k < 4 && r.ok; ++k) {
            if (std::bit_cast<std::uint32_t>(r.out[k]) != std::bit_cast<std::uint32_t>(c.expect[k])) {
                std::fprintf(stderr, "v_cndmask_b32 %s, x = %g: got %g, expected %g\n", c.what, x[k], r.out[k], c.expect[k]);
                ++g_failures;
            }
        }
        // The reference applies them too: a modified form is another translation.
        if (c.abs || c.neg) CHECK(l.ref.spirv != plain.ref.spirv);
    }
    // An output modifier or clamp on it is refused.
    {
        std::vector<std::uint32_t> w = cndmask_program(0, 1);
        w[4] |= 1u << 27;  // the second word's omod: mul:2
        CHECK(rejected_with(lift_words(w).lift, "v_cndmask_b32 with an output modifier or clamp"));
    }
}

// v_cvt_pkrtz_f16_f32 and a compressed export of its result: the export
// rounds the pair toward zero itself (gcn/half.h emit_rtz_half, whose clamp
// constant is 0x477fe000), as GCN's pack does, also where the pack was
// written under a varying EXEC - in the lift and the reference alike.
bool has_u32_constant(const std::vector<std::uint32_t>& spv, std::uint32_t value) {
    for (std::size_t i = 5; i < spv.size() && (spv[i] >> 16);) {
        if ((spv[i] & 0xffff) == 43 && (spv[i] >> 16) == 4 && spv[i + 3] == value) return true;
        i += spv[i] >> 16;
    }
    return false;
}
void packed_exports() {
    for (const bool varying : {false, true}) {
        std::vector<std::uint32_t> w = {vintrp(0, 2, 0, 0, 0), vintrp(1, 2, 0, 0, 1), vop1(kVMov, 3, kZero)};
        if (varying) push(w, {vopc(kVCmpGtF32, 2, kZero) /* vcc = 0 > x */, sop1(kAndSaveexecB64, 4, kVcc)});
        w.push_back(vop2(47, 3, 2, kV + 2));  // v_cvt_pkrtz_f16_f32 v3, v2, v2
        push(w, {0xf8000000u | 1u << 12 | 1u << 11 | 1u << 10 | 0xfu, 3u | 3u << 8});  // exp mrt0 v3, v3 done compr vm
        w.push_back(kEndpgm);
        const Lifted l = lift_words(w);
        for (const std::string& why : l.lift.rejections) std::fprintf(stderr, "packed export rejected: %s\n", why.c_str());
        CHECK(l.ref.ok() && l.lift.ok() && spirv_valid(l.lift.spirv));
        if (!has_u32_constant(l.lift.spirv, 0x477fe000u) || !has_u32_constant(l.ref.spirv, 0x477fe000u)) {
            std::fprintf(stderr, "a compressed export of a pack %s is not rounded toward zero (lift %d, reference %d)\n",
                         varying ? "under a varying EXEC" : "with EXEC set", has_u32_constant(l.lift.spirv, 0x477fe000u),
                         has_u32_constant(l.ref.spirv, 0x477fe000u));
            ++g_failures;
        }
    }
}

// ds_swizzle_b32 in a whole-quad block over a region's EXEC M (GXSsr's
// derivatives): GCN runs every lane of each quad holding an M pixel, so each
// lane reads its own quad's lane - lane 1 here, in M or not - as the lift's
// quad operation in uniform control flow does. Run over a quad whose lanes 2
// and 3 are in M and lane 1 is not: they take lane 1's attr0 minus their own.
// A source written in a block over another mask holds GCN's value only in
// that mask's quads, which the lifter cannot show cover M's: refused.
void swizzles_in_blocks() {
    const auto program = [](bool from_other_block) {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);                                                   // v2 = attr0.x in every lane
        push(w, {vop1(kVMov, 4, 240) /* v4 = 0.5 */, vop1(kVMov, 5, kV + 2) /* v5 = v2 */});
        if (from_other_block) {
            push(w, {vop1(kVMov, 6, 255), 0x41700000u /* v6 = 15.0 */, vopc(kVCmpGtF32, 2, kV + 6) /* vcc = 15 > v2 */,
                     sop1(kAndSaveexecB64, 8, kVcc), sopp(8, 4) /* s_cbranch_execz +4 */, sop1(kMovB64, kVcc, kExec),
                     sop1(kWqmB64, kExec, kExec), vop2(3, 5, 2, kOne) /* v_add_f32 v5, 1.0, v2 */, sop1(kMovB64, kExec, kVcc),
                     sop1(kMovB64, kExec, 8)});
        }
        push(w, {vop1(kVMov, 6, 255), 0x41c80000u /* v6 = 25.0 */, vopc(kVCmpGtF32, 6, kV + 2) /* vcc = v2 > 25 */,
                 sop1(kAndSaveexecB64, 4, kVcc), sopp(8, 8) /* s_cbranch_execz +8 */, sop1(kMovB64, kVcc, kExec),
                 sop1(kWqmB64, kExec, kExec)});
        swizzle(w, 3, 5, 0x8055);                                          // lane 1's v5
        push(w, {sopp(12, 0xc07f) /* s_waitcnt lgkmcnt(0) */, vop2(kVSub, 3, 2, kV + 3) /* v3 = v3 - v2 */, sop1(kMovB64, kExec, kVcc),
                 vop1(kVMov, 4, kV + 3), sop1(kMovB64, kExec, 4), sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 4, 4, 4, 4);
        w.push_back(kEndpgm);
        return w;
    };
    const Lifted l = lift_words(program(false));
    for (const std::string& why : l.lift.rejections) std::fprintf(stderr, "swizzle in a whole-quad block rejected: %s\n", why.c_str());
    CHECK(l.ref.ok() && l.lift.ok());
    if (l.lift.ok()) {
        CHECK(spirv_valid(l.lift.spirv));
        bool proof = false;
        for (const std::string& p : l.lift.proof) proof |= p.find("(1 of them) in a whole-quad block over a varying EXEC") != std::string::npos;
        CHECK(proof);
        const std::array<float, 4> attr = {10.0f, 21.0f, 32.0f, 43.0f};
        const std::array<float, 4> want = {0.5f, 0.5f, 21.0f - 32.0f, 21.0f - 43.0f};
        for (const auto [x0, y0] : {std::pair{0.0f, 0.0f}, std::pair{7.0f, 3.0f}}) {
            const QuadRun r = run_quad(l.lift.spirv, x0, y0, attr);
            if (!r.ok) std::fprintf(stderr, "swizzle in a whole-quad block: %s\n", r.error.c_str());
            CHECK(r.ok);
            for (std::size_t k = 0; k < 4 && r.ok; ++k) {
                if (r.out[k] != want[k]) {
                    std::fprintf(stderr, "swizzle in a whole-quad block: lane %zu wrote %g, expected %g\n", k, r.out[k], want[k]);
                    ++g_failures;
                }
            }
        }
    }
    CHECK(rejected_with(lift_words(program(true)).lift, "a whole-quad block computes v3 from a value the lift may hold differently"));
}

// ds_swizzle_b32 and loops. Every pixel runs a uniform loop's iterations, so a
// swizzle there sees its quad; in a loop pixels leave at different iterations
// it would not. A value that comes back to a loop's header with helper lanes
// GCN may hold differently is inexact there (found on a second attempt).
std::uint32_t branch_to(const std::vector<std::uint32_t>& w, std::uint32_t op, std::size_t target) {  // from the next slot
    return sopp(op, static_cast<std::int16_t>(static_cast<int>(target) - static_cast<int>(w.size()) - 1));
}
void swizzles_and_loops() {
    constexpr std::uint32_t kSMov = 3, kSAddU32 = 0, kSAndn2B64 = 21, kSCmpGeI32 = 3, kVAddF32 = 3, kVAddI32 = 37, kVCvtI32F32 = 8;
    constexpr std::uint32_t kCmpGtI32 = 0x84, kBranch = 2, kScc0 = 4, kScc1 = 5;
    // acc += swizzle(v6) s0 times; `inexact_write` rewrites v6 under the
    // coverage EXEC at the end of each iteration.
    const auto uniform_loop = [&](bool inexact_write) {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);  // v2 = x, in whole-quad mode
        push(w, {vop1(kVMov, 3, kZero), vop1(kVMov, 6, kV + 2), sop1(kSMov, 5, kZero)});
        const std::size_t header = w.size();
        w.push_back(sopc(kSCmpGeI32, 5, 0));  // s5 >= s0
        const std::size_t exit = w.size();
        w.push_back(0);
        swizzle(w, 4, 6, 0x8055);
        w.push_back(vop2(kVAddF32, 3, 3, kV + 4));
        if (inexact_write) {
            push(w, {sop1(kMovB64, kExec, 2), vop1(kVMov, 6, kV + 2), sop1(kWqmB64, kExec, kExec)});
        } else {
            w.push_back(vop1(kVMov, 6, kV + 3));
        }
        w.push_back(sop2(kSAddU32, 5, 5, kZero + 1));
        w.push_back(branch_to(w, kBranch, header));
        w[exit] = sopp(kScc1, static_cast<std::int16_t>(w.size() - exit - 1));
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        return w;
    };
    {
        const Lifted l = lift_words(uniform_loop(false), synthetic_options(1));
        for (const std::string& why : l.lift.rejections) std::fprintf(stderr, "swizzle in a uniform loop rejected: %s\n", why.c_str());
        CHECK(l.lift.ok() && spirv_valid(l.lift.spirv) && count_op(l.lift.spirv, 365) == 1 && count_op(l.lift.spirv, 246) == 1);
    }
    {
        const Lifted l = lift_words(uniform_loop(true), synthetic_options(1));
        CHECK(l.ref.ok());
        CHECK(rejected_with(l.lift, "ds_swizzle_b32 of v6, which the quad's helper lanes may hold differently"));
    }
    // The per-lane exit idiom in whole-quad mode: each pixel adds 1.0 while v5 < int(x).
    {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        push(w, {vop1(kVCvtI32F32, 4, kV + 2), vop1(kVMov, 3, kZero), vop1(kVMov, 5, kZero), sop1(kMovB64, 10, kExec),
                 sop1(kMovB64, 12, kExec)});
        const std::size_t header = w.size();
        push(w, {vopc(kCmpGtI32, 5, kV + 4), sop1(kMovB64, 14, kExec), sop2(kSAndn2B64, kExec, 14, kVcc), sop2(kSAndn2B64, 12, 12, kExec)});
        const std::size_t exit = w.size();
        w.push_back(0);
        w.push_back(sop2(kAndB64, kExec, 14, 12));
        swizzle(w, 6, 2, 0x8055);
        push(w, {vop2(kVAddF32, 3, 3, kV + 6), vop2(kVAddI32, 5, 5, kZero + 1)});
        w.push_back(branch_to(w, kBranch, header));
        w[exit] = sopp(kScc0, static_cast<std::int16_t>(w.size() - exit - 1));
        push(w, {sop1(kMovB64, kExec, 10), sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 3, 3, 3, 3);
        w.push_back(kEndpgm);
        const Lifted l = lift_words(w);
        CHECK(l.ref.ok());
        CHECK(rejected_with(l.lift, "ds_swizzle_b32 inside a loop pixels leave at different iterations"));
    }
    // That loop inside a region whose EXEC (`coverage`: ANDed with the
    // coverage the program started with) GCN's helper lanes do not share: in
    // the body EXEC is known set, but not in GCN's helper lanes, so what the
    // loop computes is not theirs, and a swizzle after it is refused.
    const auto loop_in_region = [&](bool coverage) {
        std::vector<std::uint32_t> w;
        wqm_prologue(w);
        push(w, {vop1(kVCvtI32F32, 4, kV + 2), vop1(kVMov, 3, kZero), vop1(kVMov, 5, kZero), vopc(kVCmpGtF32, 2, kZero) /* vcc = 0 > x */,
                 sop2(kAndB64, 16, kVcc, coverage ? 2 : kExec), sop1(kAndSaveexecB64, 18, 16)});
        const std::size_t region = w.size();
        w.push_back(0);
        push(w, {sop1(kMovB64, 10, kExec), sop1(kMovB64, 12, kExec)});
        const std::size_t header = w.size();
        push(w, {vopc(kCmpGtI32, 5, kV + 4), sop1(kMovB64, 14, kExec), sop2(kSAndn2B64, kExec, 14, kVcc), sop2(kSAndn2B64, 12, 12, kExec)});
        const std::size_t exit = w.size();
        w.push_back(0);
        push(w, {sop2(kAndB64, kExec, 14, 12), vop2(kVAddF32, 3, 3, kOne), vop2(kVAddI32, 5, 5, kZero + 1)});
        w.push_back(branch_to(w, kBranch, header));
        w[exit] = sopp(kScc0, static_cast<std::int16_t>(w.size() - exit - 1));
        w.push_back(sop1(kMovB64, kExec, 10));
        w[region] = sopp(8 /* s_cbranch_execz */, static_cast<std::int16_t>(w.size() - region - 1));
        w.push_back(sop1(kMovB64, kExec, 18));
        swizzle(w, 6, 3, 0x8055);
        push(w, {sop1(kMovB64, kExec, 2)});
        export_mrt0(w, 6, 6, 6, 6);
        w.push_back(kEndpgm);
        return lift_words(w);
    };
    {
        const Lifted l = loop_in_region(false);
        for (const std::string& why : l.lift.rejections) std::fprintf(stderr, "swizzle after a loop in a region rejected: %s\n", why.c_str());
        CHECK(l.lift.ok() && spirv_valid(l.lift.spirv) && count_op(l.lift.spirv, 365) == 1);
    }
    CHECK(rejected_with(loop_in_region(true).lift, "ds_swizzle_b32 of v3, which the quad's helper lanes may hold differently"));
}

bool valid_spirv(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    const bool valid = tools.Validate(spirv.data(), spirv.size(), vo);
    if (!valid) std::fprintf(stderr, "lifted SPIR-V invalid: %s\n", msg.c_str());
    return valid;
}

bool proof_says(const gcn::LiftResult& r, const char* text) {
    for (const std::string& line : r.proof) {
        if (line.find(text) != std::string::npos) return true;
    }
    std::fprintf(stderr, "expected a proof line containing \"%s\"\n", text);
    return false;
}

bool lifted(const gcn::LiftResult& r) {
    for (const std::string& why : r.rejections) std::fprintf(stderr, "rejected: %s\n", why.c_str());
    return r.ok() && valid_spirv(r.spirv);
}

// ---- texture LOD and whole-quad mode, on hand-encoded programs ---------------------
// The texture T# is in s[4:11] and the sampler S# in s[12:15], both user data;
// texture coordinates come from attribute 0.
namespace lod {

constexpr std::uint32_t kVcc = 106, kExec = 126, kZero = 128, kHalf = 240, kOne = 242;
constexpr std::uint32_t kSample = 32, kSampleLz = 39, kGetLod = 96;

struct Asm {  // Sea Islands encodings of the instructions the cases use
    std::vector<std::uint32_t> w;
    void sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0) { w.push_back(0xbe800000u | sdst << 16 | op << 8 | ssrc0); }
    void sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t ssrc1) {
        w.push_back(0x80000000u | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0);
    }
    void sopc(std::uint32_t op, std::uint32_t ssrc0, std::uint32_t ssrc1) { w.push_back(0xbf000000u | op << 16 | ssrc1 << 8 | ssrc0); }
    void vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | op << 9 | src0); }
    void vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(op << 25 | vdst << 17 | vsrc1 << 9 | src0); }
    void vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(0x7c000000u | op << 17 | vsrc1 << 9 | src0); }
    void interp(std::uint32_t vdst, std::uint32_t attr, std::uint32_t chan) {  // v_interp_p1_f32 / v_interp_p2_f32 from v0, v1
        w.push_back(0xc8000000u | vdst << 18 | attr << 10 | chan << 8 | 0u);
        w.push_back(0xc8000000u | vdst << 18 | 1u << 16 | attr << 10 | chan << 8 | 1u);
    }
    void mimg(std::uint32_t op, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t dmask, bool unorm = false) {
        w.push_back(0xf0000000u | op << 18 | (unorm ? 1u << 12 : 0u) | dmask << 8);
        w.push_back(vaddr | vdata << 8 | (4u / 4) << 16 | (12u / 4) << 21);
    }
    void wqm() { sop1(10, kExec, kExec); }
    void exp_mrt0(std::uint32_t v) {  // exp mrt0, v..v+3 done vm
        w.push_back(0xf800180fu);
        w.push_back(v | (v + 1) << 8 | (v + 2) << 16 | (v + 3) << 24);
    }
    std::size_t branch(std::uint32_t op) {  // a forward SOPP branch, landed later
        w.push_back(0xbf800000u | op << 16);
        return w.size() - 1;
    }
    void land(std::size_t at) { w[at] |= static_cast<std::uint16_t>(w.size() - at - 1); }
};

gcn::LiftResult lift(const std::vector<std::uint32_t>& words, std::uint32_t dim = 1 /* 2D */) {
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
    const gcn::TranslateResult ref = gcn::translate(p, o);
    CHECK(ref.ok());
    return gcn::lift_pixel_shader(p, o, ref);
}

// Whole-quad mode at entry and the coordinates in v2, v3; then `body` in a
// region on the pixel's own condition (0 < v2: EXEC M), `tail` after EXEC is
// restored, and v4..v7 exported.
std::vector<std::uint32_t> divergent(const std::function<void(Asm&)>& body, const std::function<void(Asm&)>& tail = {}) {
    Asm a;
    a.sop1(4, 20, kExec);
    a.wqm();
    a.interp(2, 0, 0);
    a.interp(3, 0, 1);
    a.vopc(1, kZero, 2);    // v_cmp_lt_f32 vcc, 0, v2
    a.sop1(36, 22, kVcc);   // s_and_saveexec_b64 s[22:23], vcc
    const std::size_t skip = a.branch(8);
    body(a);
    a.land(skip);
    a.sop1(4, kExec, 22);   // s_mov_b64 exec, s[22:23]
    if (tail) tail(a);
    a.exp_mrt0(4);
    a.w.push_back(0xbf810000u);
    return a.w;
}

// The compiler's idiom for a sample in a divergent region: save EXEC, widen
// it to whole quads, compute v8 = v2 / 2 and v9 = v3 / 2, restore EXEC.
void quad_coords(Asm& a, std::uint32_t save = kVcc, std::uint32_t restore = kVcc) {
    a.sop1(4, save, kExec);
    a.wqm();
    a.vop2(8, 8, kHalf, 2);  // v_mul_f32 v8, 0.5, v2
    a.vop2(8, 9, kHalf, 3);
    a.sop1(4, kExec, restore);
}

bool has_op(const std::vector<std::uint32_t>& spirv, std::uint32_t opcode) {
    for (std::size_t i = 5; i < spirv.size();) {
        const std::uint32_t n = spirv[i] >> 16;
        if ((spirv[i] & 0xffff) == opcode) return true;
        i += n ? n : 1;
    }
    return false;
}

// image_load(_mip) and image_get_resinfo, which take no sampler and no
// derivatives, and a depth export: lifted as the translator writes them
// (OpImageFetch, OpImageQuerySizeLod with OpImageQueryLevels; FragDepth with
// DepthReplacing, and then no early fragment tests).
void loads_and_sizes() {
    const auto mode = [](const std::vector<std::uint32_t>& spirv, std::uint32_t execution_mode) {
        for (std::size_t i = 5; i < spirv.size();) {
            const std::uint32_t n = spirv[i] >> 16;
            if ((spirv[i] & 0xffff) == 16 && n >= 3 && spirv[i + 2] == execution_mode) return true;  // OpExecutionMode
            i += n ? n : 1;
        }
        return false;
    };
    const auto coords = [](Asm& a) {
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(7, 2, 256 + 2);  // v_cvt_u32_f32 v2, v2
        a.vop1(7, 3, 256 + 3);
    };
    {
        Asm a;
        coords(a);
        a.mimg(0, 4, 2, 0xf);  // image_load v[4:7], v[2:3], s[4:11]
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        const gcn::LiftResult r = lift(a.w);
        CHECK(lifted(r) && has_op(r.spirv, 95) && proof_says(r, "1 image loads"));  // OpImageFetch
    }
    {
        Asm a;
        coords(a);
        a.vop1(1, 4, kZero);   // v_mov_b32 v4, 0: the level
        a.mimg(1, 8, 2, 0xf);  // image_load_mip v[8:11], v[2:4], s[4:11]
        a.exp_mrt0(8);
        a.w.push_back(0xbf810000u);
        CHECK(lifted(lift(a.w)));
    }
    {
        Asm a;
        a.vop1(1, 2, kZero);    // the level
        a.mimg(14, 4, 2, 0xf);  // image_get_resinfo v[4:7], v2, s[4:11]
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        const gcn::LiftResult r = lift(a.w);
        CHECK(lifted(r) && has_op(r.spirv, 103) && has_op(r.spirv, 106));  // OpImageQuerySizeLod, OpImageQueryLevels
    }
    {
        Asm a;
        a.interp(2, 0, 0);
        a.w.push_back(0xf8000000u | 1u << 12 | 0x0fu);  // exp mrt0 v2, v2, v2, v2 vm
        a.w.push_back(2u | 2u << 8 | 2u << 16 | 2u << 24);
        a.w.push_back(0xf8000000u | 1u << 12 | 1u << 11 | 8u << 4 | 0x1u);  // exp mrtz v2 done vm
        a.w.push_back(2u);
        a.w.push_back(0xbf810000u);
        const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Pixel;
        o.ps_input_ena = 2;
        o.descriptor_set = 1;
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.early_fragment_tests = true;  // asked for, and not given with a depth export
        const gcn::TranslateResult ref = gcn::translate(p, o);
        const gcn::LiftResult r = gcn::lift_pixel_shader(p, o, ref);
        CHECK(ref.ok() && lifted(r) && mode(r.spirv, 12) && !mode(r.spirv, 9));  // DepthReplacing, no EarlyFragmentTests
    }
}

// A VCC branch on a texel: the tone map's adaptation samples one texel at
// constant coordinates and branches on comparing it, which every lane does
// alike - the same coordinates sample the same texel - so it is an if. At
// per-pixel coordinates the texels differ and the branch is the wave's.
void uniform_texels() {
    const auto program = [](bool per_pixel) {
        Asm a;
        if (per_pixel) {
            a.interp(0, 0, 0);
            a.interp(1, 0, 1);
        } else {
            a.vop1(1, 0, kZero);  // v_mov_b32 v0, 0
            a.vop1(1, 1, kZero);
        }
        a.mimg(kSampleLz, 2, 0, 0x3);  // image_sample_lz v[2:3], v[0:1], s[4:11], s[12:15]
        a.w.push_back(0xbf8c0070u);   // s_waitcnt vmcnt(0)
        a.vopc(4, 256 + 2, 3);        // v_cmp_gt_f32 vcc, v2, v3
        const std::size_t skip = a.branch(6);  // s_cbranch_vccz
        a.vop1(1, 4, kOne);           // then: v4 = 1.0
        const std::size_t join = a.branch(2);  // s_branch
        a.land(skip);
        a.vop1(1, 4, kHalf);          // else: v4 = 0.5
        a.land(join);
        for (std::uint32_t k = 5; k < 8; ++k) a.vop1(1, k, kZero);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        return a.w;
    };
    const gcn::LiftResult uniform = lift(program(false));
    CHECK(lifted(uniform) && proof_says(uniform, "1 ifs on a bit every lane holds alike"));
    CHECK(rejected_with(lift(program(true)), "s_cbranch_vccz tests a VCC that may differ between lanes"));
}

void cases() {
    // A sample under a varying EXEC whose coordinates were computed for every
    // pixel: lifted for every pixel, its texel written under EXEC.
    {
        const gcn::LiftResult r = lift(divergent([](Asm& a) { a.mimg(kSample, 4, 2, 0xf); }));
        CHECK(lifted(r));
        CHECK(proof_says(r, "1 image samples under a varying EXEC"));
    }
    {
        const gcn::LiftResult r = lift(divergent([](Asm& a) { a.mimg(kSampleLz, 4, 2, 0xf); }));
        CHECK(lifted(r));
        CHECK(proof_says(r, "1 image samples under a varying EXEC"));
    }
    // Its coordinates from a whole-quad block over that EXEC.
    {
        const gcn::LiftResult r = lift(divergent([](Asm& a) {
            quad_coords(a);
            a.mimg(kSample, 4, 8, 0xf);
        }));
        CHECK(lifted(r));
        CHECK(proof_says(r, "whole-quad block"));
    }
    // The block's registers hold GCN's value only in the quads of its EXEC:
    // not for an export or a sample after EXEC is restored.
    CHECK(rejected_with(lift(divergent(
                            [](Asm& a) {
                                quad_coords(a);
                                a.mimg(kSample, 4, 8, 0xf);
                            },
                            [](Asm& a) { a.vop1(1, 4, 256 + 8); })),  // v_mov_b32 v4, v8
                        "an export reads a value the lift may hold differently"));
    CHECK(rejected_with(lift(divergent([](Asm& a) { quad_coords(a); }, [](Asm& a) { a.mimg(kSample, 4, 8, 0xf); })),
                        "an image sample reads an operand the lift may hold differently"));
    CHECK(rejected_with(lift(divergent([](Asm& a) { quad_coords(a); }, [](Asm& a) { a.mimg(kSampleLz, 4, 8, 0xf); })),
                        "an image sample reads an operand the lift may hold differently"));
    // A comparison (or anything else that needs EXEC) inside the block.
    CHECK(rejected_with(lift(divergent([](Asm& a) {
                            a.sop1(4, kVcc, kExec);
                            a.wqm();
                            a.vopc(1, kZero, 3);
                            a.sop1(4, kExec, kVcc);
                        })),
                        "EXEC is used inside a whole-quad block"));
    // A block that restores another mask (here the entry's), or none before a branch.
    CHECK(rejected_with(lift(divergent([](Asm& a) { quad_coords(a, kVcc, 20); })), "does not end by restoring EXEC"));
    CHECK(rejected_with(lift(divergent([](Asm& a) {
                            a.sop1(4, kVcc, kExec);
                            a.wqm();
                            a.vop2(8, 8, kHalf, 2);
                            const std::size_t skip = a.branch(8);
                            a.vop2(8, 9, kHalf, 3);
                            a.land(skip);
                            a.sop1(4, kExec, kVcc);
                        })),
                        "not ended by s_mov_b64 exec"));
    // Nested regions: a block over the inner EXEC may read the outer block's
    // registers (its quads lie within the outer's), not the other way round.
    const auto nested = [](bool outer_reads_inner) {
        return divergent([outer_reads_inner](Asm& a) {
            quad_coords(a);                 // v8, v9 over M
            a.vopc(4, kZero, 3);            // v_cmp_gt_f32 vcc, 0, v3
            a.sop1(36, 24, kVcc);           // EXEC = M2, within M
            const std::size_t skip = a.branch(8);
            a.sop1(4, 26, kExec);
            a.wqm();
            a.vop2(3, 10, kOne, 8);         // v_add_f32 v10, 1.0, v8
            a.vop2(3, 11, kOne, 9);
            a.sop1(4, kExec, 26);
            a.mimg(kSample, 4, 10, 0xf);
            a.land(skip);
            a.sop1(4, kExec, 24);           // EXEC = M
            if (outer_reads_inner) {
                a.sop1(4, 26, kExec);
                a.wqm();
                a.vop2(3, 12, kOne, 10);    // v10 holds GCN's value only in the quads of M2
                a.sop1(4, kExec, 26);
            }
        });
    };
    CHECK(lifted(lift(nested(false))));
    CHECK(rejected_with(lift(nested(true)), "a whole-quad block computes v12 from a value the lift may hold differently"));
    // A register written in whole-quad mode for one region's quads (EXEC M),
    // then by a sample under another region's EXEC (N & ~M), as the G-buffer
    // shaders' two arms do: exact in M's quads and in the N & ~M pixels, so a
    // comparison under N & ~M is lifted. Under the EXEC both regions came
    // from it is still refused: in the pixels outside both, GCN kept the old
    // value where the lift wrote the block's for every pixel.
    const auto two_regions = [](bool compare_after) {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(1, kZero, 3);              // v_cmp_lt_f32 vcc, 0, v3: N
        a.sop1(4, 26, kVcc);
        a.vopc(1, kZero, 2);              // v_cmp_lt_f32 vcc, 0, v2: M
        a.sop1(4, 28, kVcc);
        a.sop1(36, 22, kVcc);             // s_and_saveexec_b64 s[22:23], vcc: EXEC = M
        std::size_t skip = a.branch(8);
        quad_coords(a);                   // v8, v9 in the quads of M
        a.mimg(kSample, 4, 8, 0xf);
        a.land(skip);
        a.sop1(4, kExec, 22);
        a.sop2(21, kExec, 26, 28);        // s_andn2_b64 exec, s[26:27], s[28:29]: EXEC = N & ~M
        skip = a.branch(8);
        a.sop1(4, 24, kExec);
        a.wqm();
        a.vop2(8, 10, kHalf, 3);          // v10, v11 in the quads of N & ~M
        a.vop2(8, 11, kHalf, 2);
        a.sop1(4, kExec, 24);
        a.mimg(kSample, 8, 10, 0xf);      // v8..v11 under N & ~M, over the block's v8, v9
        a.vopc(2, kZero, 9);              // v_cmp_eq_f32 vcc, 0, v9
        a.vop2(0, 4, 256 + 8, 4);         // v_cndmask_b32 v4, v8, v4
        a.land(skip);
        a.sop1(4, kExec, 22);             // EXEC as the regions found it
        if (compare_after) a.vopc(2, kZero, 9);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        return a.w;
    };
    CHECK(lifted(lift(two_regions(false))));
    CHECK(rejected_with(lift(two_regions(true)), "a lane mask is computed from a value the lift may hold differently"));
    // An if on a scalar condition inside the region: the coordinates come from
    // a block on one arm and from plain moves on the other; at the join they
    // hold GCN's value in the quads of the region's EXEC, as the sample needs.
    {
        const auto program = [](bool export_v8) {
            return divergent(
                [](Asm& b) {
                    b.sopc(6, 0, kZero);  // s_cmp_eq_u32 s0, 0
                    const std::size_t to_else = b.branch(4);
                    quad_coords(b);
                    const std::size_t to_join = b.branch(2);
                    b.land(to_else);
                    b.vop1(1, 8, 256 + 2);  // v_mov_b32 v8, v2
                    b.vop1(1, 9, 256 + 3);
                    b.land(to_join);
                    b.mimg(kSample, 4, 8, 0xf);
                },
                [export_v8](Asm& b) {
                    if (export_v8) b.vop1(1, 5, 256 + 8);
                });
        };
        const gcn::LiftResult r = lift(program(false));
        CHECK(lifted(r));
        CHECK(proof_says(r, "if at"));
        CHECK(rejected_with(lift(program(true)), "an export reads a value the lift may hold differently"));
    }

    // Inside a kill region widened to whole quads from a varying EXEC, GCN
    // computes whole quads where the lift keeps each pixel's bit: no implicit
    // derivatives under its EXEC there.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(1, kZero, 2);
        a.sop1(36, 22, kVcc);           // EXEC = M
        a.vopc(1, kZero, 3);
        a.sop1(4, 24, kExec);
        a.sop2(21, 24, 24, kVcc);       // the guard: M and not (0 < v3)
        const std::size_t skip = a.branch(4);
        a.sop2(15, kExec, kExec, 24);   // s_and_b64 exec, exec, s[24:25]
        a.wqm();                        // widened from a varying EXEC
        a.mimg(kSample, 4, 2, 0xf);
        a.land(skip);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(a.w), "an image sample with implicit derivatives under a varying EXEC inside a widened kill region"));
    }

    // image_get_lod: OpImageQueryLod, its implicit derivatives as a sample's.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.mimg(kGetLod, 4, 2, 0x2);  // the unclamped LOD alone, as the game reads it
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        const gcn::LiftResult r = lift(a.w);
        CHECK(lifted(r));
        CHECK(has_op(r.spirv, 105));  // OpImageQueryLod
        CHECK(proof_says(r, "1 image_get_lod"));
    }
    // Only the unclamped LOD (component 1): the clamped one (component 0,
    // Vulkan's level after the view's and the sampler's clamps) is not shown
    // to be GCN's.
    CHECK(rejected_with(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x3); })), "image_get_lod's clamped LOD"));
    CHECK(rejected_with(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x1); })), "image_get_lod's clamped LOD"));
    CHECK(proof_says(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x2); })), "the host's own LOD computation"));
    // TFE and LWE: GCN writes a status VGPR after the data, which the lift does not.
    for (const std::uint32_t bit : {16u, 17u}) {
        for (const std::uint32_t op : {kSample, kGetLod}) {
            Asm a;
            a.sop1(4, 20, kExec);
            a.wqm();
            a.interp(2, 0, 0);
            a.interp(3, 0, 1);
            a.mimg(op, 4, 2, op == kGetLod ? 0x2 : 0x1);
            a.w[a.w.size() - 2] |= 1u << bit;
            a.sop1(4, kExec, 20);
            a.exp_mrt0(4);
            a.w.push_back(0xbf810000u);
            CHECK(rejected_with(lift(a.w), "TFE or LWE"));
        }
    }
    CHECK(lifted(lift(divergent([](Asm& a) {
        quad_coords(a);
        a.mimg(kGetLod, 4, 8, 0x2);
    }))));
    CHECK(rejected_with(lift(divergent([](Asm& a) { quad_coords(a); }, [](Asm& a) { a.mimg(kGetLod, 4, 8, 0x2); })),
                        "image_get_lod reads an operand the lift may hold differently"));
    CHECK(rejected_with(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x4); })), "image_get_lod returns two components"));
    CHECK(rejected_with(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x2, true); })), "image_get_lod with unnormalized coordinates"));
    CHECK(rejected_with(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x2); }), 3 /* cube */), "image_get_lod of a cube map"));
    // After a kill region that leaves EXEC narrower, pixels the lift may hold
    // differently from GCN are only killed at the end: no implicit derivatives.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(1, kZero, 2);
        a.sop2(21, 20, 20, kVcc);      // s_andn2_b64 s[20:21], s[20:21], vcc
        const std::size_t skip = a.branch(4);
        a.sop2(15, kExec, kExec, 20);  // s_and_b64 exec, exec, s[20:21]
        a.vop1(1, 6, kOne);
        a.land(skip);
        a.sop1(4, kExec, 20);          // EXEC set again from the guard, as the compiler does
        a.mimg(kGetLod, 4, 2, 0x2);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(a.w), "image_get_lod after a kill region"));
    }
}


// Implicit derivatives read the quad's helper lanes, which compute the
// coordinates on GCN only in whole-quad mode: interpolated without it, they
// are refused; an explicit-LOD sample reads the pixel's own coordinates only.
void helper_lanes() {
    const auto without_wqm = [](std::uint32_t op, std::uint32_t dmask) {
        Asm a;
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.mimg(op, 4, 2, dmask);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        return a.w;
    };
    CHECK(rejected_with(lift(without_wqm(kSample, 0xf)), "an image sample with implicit derivatives reads coordinates the quad's helper lanes"));
    CHECK(rejected_with(lift(without_wqm(kGetLod, 0x2)), "image_get_lod with implicit derivatives reads coordinates the quad's helper lanes"));
    CHECK(lifted(lift(without_wqm(kSampleLz, 0xf))));
}

// Texture LOD and whole-quad blocks with the lifted loops. s0 is the uniform
// loop count; scratch SGPRs from s20 (s4-s15 hold the T# and S#).
void back(Asm& a, std::uint32_t op, std::size_t target) {  // a backward branch from the next slot
    a.w.push_back(0xbf800000u | op << 16 | static_cast<std::uint16_t>(static_cast<int>(target) - static_cast<int>(a.w.size()) - 1));
}
void loops() {
    // The per-lane exit idiom (v5 counts up to int(x)) around `body`, in whole-quad mode.
    const auto divergent_loop = [](const std::function<void(Asm&)>& body) {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(8, 4, 256 + 2);           // v_cvt_i32_f32 v4, v2
        a.vop1(1, 5, kZero);
        a.sop1(4, 26, kExec);
        a.sop1(4, 24, kExec);            // the loop mask L
        const std::size_t header = a.w.size();
        a.vopc(0x84, 256 + 4, 5);        // v_cmp_gt_i32 vcc, v4, v5: stay
        a.sop1(4, 28, kExec);
        a.sop2(21, kExec, 28, kVcc);     // EXEC = the lanes leaving
        a.sop2(21, 24, 24, kExec);       // L loses them
        const std::size_t exit = a.branch(4);
        a.sop2(15, kExec, 28, 24);       // EXEC = the lanes staying
        body(a);
        a.vop2(37, 5, kZero + 1, 5);     // v_add_i32 v5, 1, v5
        back(a, 2, header);
        a.land(exit);
        a.sop1(4, kExec, 26);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(8);
        a.w.push_back(0xbf810000u);
        return a.w;
    };
    // A whole-quad block over a region's EXEC in its body: the quad's other pixels may have left.
    CHECK(rejected_with(lift(divergent_loop([](Asm& a) {
                            a.vopc(1, kZero, 2);
                            a.sop1(36, 30, kVcc);
                            const std::size_t skip = a.branch(8);
                            a.sop1(4, kVcc, kExec);
                            a.wqm();
                            a.vop2(8, 8, kHalf, 2);
                            a.sop1(4, kExec, kVcc);
                            a.land(skip);
                            a.sop1(4, kExec, 30);
                        })),
                        "whole-quad block inside a loop pixels leave at different iterations"));
    CHECK(rejected_with(lift(divergent_loop([](Asm& a) { a.mimg(kGetLod, 8, 2, 0x2); })),
                        "image_get_lod inside a loop pixels leave at different iterations"));

    // A whole-quad block that would end past a loop header: GCN runs the
    // header's later iterations under the restored mask, not whole quads.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(1, kZero, 2);
        a.sop1(36, 22, kVcc);            // EXEC = M
        const std::size_t skip = a.branch(8);
        a.sop1(4, 30, kExec);
        a.wqm();
        a.vop2(8, 8, kHalf, 2);
        a.sop1(3, 32, kZero);
        const std::size_t header = a.w.size();
        a.vop2(8, 8, kHalf, 2);
        a.sop1(4, kExec, 30);            // the restore, in the loop
        a.sop2(0, 32, 32, kZero + 1);
        a.sopc(3, 32, 0);                // s_cmp_ge_i32 s32, s0
        const std::size_t exit = a.branch(5);
        back(a, 2, header);
        a.land(exit);
        a.land(skip);
        a.sop1(4, kExec, 22);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(a.w), "not ended by s_mov_b64 exec before a branch or block boundary"));
    }

    // In a uniform loop: v8, v9 are the coordinates of an implicit-LOD sample
    // at the top, rewritten in a whole-quad block over a region's EXEC below
    // it, so the next iteration's sample would read them where they hold
    // GCN's value only in that region's quads.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(1, 8, 256 + 2);
        a.vop1(1, 9, 256 + 3);
        a.sop1(3, 32, kZero);
        const std::size_t header = a.w.size();
        a.sopc(3, 32, 0);
        const std::size_t exit = a.branch(5);
        a.mimg(kSample, 4, 8, 0xf);
        a.vopc(1, kZero, 2);
        a.sop1(36, 22, kVcc);
        const std::size_t skip = a.branch(8);
        a.sop1(4, kVcc, kExec);
        a.wqm();
        a.vop2(8, 8, kHalf, 2);
        a.vop2(8, 9, kHalf, 3);
        a.sop1(4, kExec, kVcc);
        a.land(skip);
        a.sop1(4, kExec, 22);
        a.sop2(0, 32, 32, kZero + 1);
        back(a, 2, header);
        a.land(exit);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(a.w), "v8 comes back to the header of the loop at"));
    }
    // The loop's exit carries where its registers hold GCN's value: v8 leaves
    // right after a whole-quad block wrote it, though later in the body it is
    // rewritten for every pixel.
    const auto leaves_partial = [](bool export_v8) {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(1, 8, 256 + 2);
        a.sop1(3, 32, kZero);
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
        a.sopc(3, 32, 0);
        const std::size_t exit = a.branch(5);
        a.vop1(1, 8, 256 + 2);
        a.sop2(0, 32, 32, kZero + 1);
        back(a, 2, header);
        a.land(exit);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(export_v8 ? 8 : 12);
        a.w.push_back(0xbf810000u);
        return a.w;
    };
    {
        const gcn::LiftResult r = lift(leaves_partial(false));
        CHECK(lifted(r));
        CHECK(proof_says(r, "whole-quad block"));
        CHECK(proof_says(r, "uniform, left on a bit every lane holds alike"));
    }
    CHECK(rejected_with(lift(leaves_partial(true)), "an export reads a value the lift may hold differently"));
    // A loop pixels leave at different iterations inside a kill region widened
    // from EXEC known set, where every pixel computes the region: GCN runs it
    // only in the quads holding a pixel the guard keeps, so the lift's other
    // quads leave it at the first exit test (their pixels are killed, and
    // they no longer run iterations on data GCN never looked at).
    {
        Asm a;
        a.sop1(4, 20, kExec);            // the guard starts as the coverage
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vopc(4, kZero, 2);             // v_cmp_gt_f32 vcc, 0, v2: kill
        a.sop2(21, 20, 20, kVcc);        // s_andn2_b64 s[20:21], s[20:21], vcc
        const std::size_t end = a.branch(4);
        a.sop2(15, kExec, kExec, 20);    // s_and_b64 exec, exec, s[20:21]
        a.wqm();                         // widened from EXEC known set
        a.vop1(8, 4, 256 + 3);           // v4 = int(v3): the count
        a.vop1(1, 5, kZero);
        a.vop1(1, 8, kZero);
        a.sop1(4, 26, kExec);
        a.sop1(4, 24, kExec);            // the loop mask L
        const std::size_t header = a.w.size();
        a.vopc(0x84, 256 + 4, 5);        // v_cmp_gt_i32 vcc, v4, v5: stay
        a.sop1(4, 28, kExec);
        a.sop2(21, kExec, 28, kVcc);
        a.sop2(21, 24, 24, kExec);
        const std::size_t exit = a.branch(4);
        a.sop2(15, kExec, 28, 24);
        a.vop2(3, 8, kOne, 8);           // v8 += 1.0
        a.vop2(37, 5, kZero + 1, 5);     // v_add_i32 v5, 1, v5
        back(a, 2, header);
        a.land(exit);
        a.sop1(4, kExec, 26);
        a.land(end);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(8);
        a.w.push_back(0xbf810000u);
        const gcn::LiftResult r = lift(a.w);
        CHECK(lifted(r));
        CHECK(proof_says(r, "quads without a covered pixel the guard keeps"));
        CHECK(has_op(r.spirv, 366));  // OpGroupNonUniformQuadSwap
    }
    // An if whose else arm leaves the loop joins with the then arm's values,
    // and with where they hold GCN's value: v8 from a whole-quad block, so
    // the implicit-LOD sample after the join is refused.
    {
        Asm a;
        a.sop1(4, 20, kExec);
        a.wqm();
        a.interp(2, 0, 0);
        a.interp(3, 0, 1);
        a.vop1(1, 8, 256 + 2);
        a.vop1(1, 9, 256 + 3);
        a.sop1(3, 32, kZero);
        const std::size_t header = a.w.size();
        a.sopc(4, 32, 0);                // s_cmp_lt_i32 s32, s0: another iteration
        const std::size_t to_else = a.branch(4);
        a.vopc(1, kZero, 2);
        a.sop1(36, 22, kVcc);
        const std::size_t skip = a.branch(8);
        a.sop1(4, kVcc, kExec);
        a.wqm();
        a.vop2(8, 8, kHalf, 2);
        a.sop1(4, kExec, kVcc);
        a.land(skip);
        a.sop1(4, kExec, 22);
        const std::size_t to_join = a.branch(2);
        a.land(to_else);
        const std::size_t exit = a.branch(2);  // the else arm leaves the loop
        a.land(to_join);
        a.mimg(kSample, 4, 8, 0xf);
        a.sop2(0, 32, 32, kZero + 1);
        back(a, 2, header);
        a.land(exit);
        a.sop1(4, kExec, 20);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(a.w), "an image sample reads an operand the lift may hold differently"));
    }
}

}  // namespace lod

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
    void exp_mrt0(std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t q) {  // exp mrt0 x, y, z, q done vm
        w.push_back(0xf8000000u | 1u << 12 | 1u << 11 | 0xfu);
        w.push_back(x | y << 8 | z << 16 | q << 24);
    }
};

// Operand codes and opcodes.
constexpr std::uint32_t kV = 256, kC0 = 128, kOne = 242, kTwo = 244, kFour = 246, kVcc = 106, kExec = 126;
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

// ---- one invocation of a lifted pixel shader, control flow and all ----------------------
// Runs the module's blocks from the entry, phis by the edge taken, with attr0
// as given and the params block's user SGPRs, and returns mrt0. It knows the
// instructions the loop and region programs below lift to; anything else fails.
struct CfgRun {
    bool ok = false;
    std::string error;
    std::array<float, 4> out{};
    std::size_t blocks = 0;  // blocks entered
};
CfgRun run_cfg(const std::vector<std::uint32_t>& spv, const std::array<float, 4>& attr0, const std::array<std::uint32_t, 16>& user = {}) {
    CfgRun run;
    using Vec = std::vector<std::uint32_t>;
    std::map<std::uint32_t, Vec> val;
    std::map<std::uint32_t, std::uint32_t> storage, location, builtin, glsl_set;
    struct Ptr {
        std::uint32_t var = 0;
        std::vector<std::uint32_t> idx;
    };
    std::map<std::uint32_t, Ptr> ptrs;
    std::map<std::uint32_t, Vec> memory;  // Function and Private variables
    std::map<std::uint32_t, std::size_t> label_at;  // label -> word index of its OpLabel
    std::size_t entry = 0;
    const auto fail = [&](const std::string& why) {
        run.error = why;
        return run;
    };
    bool in_function = false;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t n = spv[i] >> 16, op = spv[i] & 0xffff;
        if (n == 0 || i + n > spv.size()) return fail("malformed module");
        const std::uint32_t* a = &spv[i + 1];
        switch (op) {
        case 11: glsl_set[a[0]] = 1; break;                          // OpExtInstImport
        case 71:                                                     // OpDecorate
            if (a[1] == 11) builtin[a[0]] = a[2];
            if (a[1] == 30) location[a[0]] = a[2];
            break;
        case 59: if (!in_function) storage[a[1]] = a[2]; else storage[a[1]] = a[2]; break;  // OpVariable
        case 43: val[a[1]] = {a[2]}; break;                          // OpConstant (low word)
        case 41: val[a[1]] = {1}; break;                             // OpConstantTrue
        case 42: val[a[1]] = {0}; break;                             // OpConstantFalse
        case 44: {                                                   // OpConstantComposite
            Vec v;
            for (std::uint32_t k = 3; k < n; ++k) v.push_back(val[a[k - 1]].empty() ? 0 : val[a[k - 1]][0]);
            val[a[1]] = v;
            break;
        }
        case 1: val[a[1]] = {0, 0, 0, 0}; break;                     // OpUndef
        case 54: in_function = true; break;                          // OpFunction
        case 248:                                                    // OpLabel
            label_at[a[0]] = i;
            if (!entry) entry = i;
            break;
        default: break;
        }
        i += n;
    }
    if (!entry) return fail("no function body");
    const auto f = [](std::uint32_t b) { return std::bit_cast<float>(b); };
    const auto fb = [](float x) { return std::bit_cast<std::uint32_t>(x); };
    const auto get = [&](std::uint32_t id) -> const Vec& { return val[id]; };
    const auto s0 = [&](std::uint32_t id) -> std::uint32_t {
        const Vec& v = val[id];
        return v.empty() ? 0u : v[0];
    };
    std::size_t pc = entry;
    std::uint32_t prev = 0, current = spv[entry + 1];
    for (std::size_t steps = 0; steps < 200000; ++steps) {
        const std::uint32_t n = spv[pc] >> 16, op = spv[pc] & 0xffff;
        const std::uint32_t* a = &spv[pc + 1];
        std::size_t next = pc + n;
        const auto go = [&](std::uint32_t label) {
            const auto it = label_at.find(label);
            if (it == label_at.end()) return false;
            prev = current;
            current = label;
            next = it->second;
            ++run.blocks;
            return true;
        };
        const auto bin_f = [&](auto fn) {
            const Vec& x = get(a[2]);
            const Vec& y = get(a[3]);
            Vec r(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) r[k] = fn(f(x[k]), f(k < y.size() ? y[k] : 0));
            val[a[1]] = r;
        };
        const auto bin_u = [&](auto fn) {
            const Vec& x = get(a[2]);
            const Vec& y = get(a[3]);
            Vec r(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) r[k] = fn(x[k], k < y.size() ? y[k] : 0);
            val[a[1]] = r;
        };
        switch (op) {
        case 248: break;                                   // OpLabel
        case 246: case 247: break;                         // OpLoopMerge, OpSelectionMerge
        case 245: {                                        // OpPhi
            bool found = false;
            for (std::uint32_t k = 2; k + 1 < n; k += 2) {
                if (a[k + 1] == prev) {
                    val[a[1]] = get(a[k]);
                    found = true;
                }
            }
            if (!found) return fail("a phi without the edge taken");
            break;
        }
        case 249: if (!go(a[0])) return fail("a branch to no label"); break;
        case 250: if (!go(s0(a[0]) ? a[1] : a[2])) return fail("a branch to no label"); break;
        case 253: case 56: run.ok = true; return run;      // OpReturn
        case 252: return fail("OpKill");
        case 65: {                                         // OpAccessChain
            Ptr p;
            const auto base = ptrs.find(a[2]);
            if (base != ptrs.end()) p = base->second;
            else p.var = a[2];
            for (std::uint32_t k = 3; k + 1 < n; ++k) p.idx.push_back(s0(a[k]));
            ptrs[a[1]] = p;
            break;
        }
        case 61: {                                         // OpLoad
            std::uint32_t var = a[2];
            std::vector<std::uint32_t> idx;
            if (const auto it = ptrs.find(a[2]); it != ptrs.end()) {
                var = it->second.var;
                idx = it->second.idx;
            }
            const std::uint32_t sc = storage.count(var) ? storage[var] : 99;
            if (sc == 1 && location.count(var) && location[var] == 0) {
                Vec v = {fb(attr0[0]), fb(attr0[1]), fb(attr0[2]), fb(attr0[3])};
                val[a[1]] = idx.empty() ? v : Vec{v[idx[0] & 3]};
            } else if (sc == 1 && builtin.count(var) && builtin[var] == 15) {
                val[a[1]] = {fb(0.5f), fb(0.5f), 0, fb(1.0f)};
            } else if (sc == 2) {                          // the params block: user SGPRs are member 2
                if (idx.size() < 3 || idx[0] != 2) return fail("a params load other than a user SGPR");
                val[a[1]] = {user[(idx[1] * 4 + idx[2]) & 15]};
            } else if (sc == 7 || sc == 6) {
                Vec m = memory[var];
                val[a[1]] = idx.empty() ? m : Vec{idx[0] < m.size() ? m[idx[0]] : 0};
            } else {
                return fail("a load the test does not model (storage class " + std::to_string(sc) + ")");
            }
            break;
        }
        case 62: {                                         // OpStore
            std::uint32_t var = a[0];
            std::vector<std::uint32_t> idx;
            if (const auto it = ptrs.find(a[0]); it != ptrs.end()) {
                var = it->second.var;
                idx = it->second.idx;
            }
            const std::uint32_t sc = storage.count(var) ? storage[var] : 99;
            if (sc == 3 && location.count(var) && location[var] == 0) {
                const Vec& v = get(a[1]);
                for (std::size_t k = 0; k < 4 && k < v.size(); ++k) run.out[k] = f(v[k]);
            } else if (sc == 7 || sc == 6) {
                if (idx.empty()) {
                    memory[var] = get(a[1]);
                } else {
                    Vec& m = memory[var];
                    if (m.size() <= idx[0]) m.resize(idx[0] + 1);
                    m[idx[0]] = s0(a[1]);
                }
            } else if (sc != 3) {
                return fail("a store the test does not model");
            }
            break;
        }
        case 59: if (!memory.count(a[1])) memory[a[1]] = n > 4 ? get(a[3]) : Vec{0, 0, 0, 0}; break;  // a Function variable, initialised
        case 81: {                                         // OpCompositeExtract
            const Vec& v = get(a[2]);
            val[a[1]] = {a[3] < v.size() ? v[a[3]] : 0};
            break;
        }
        case 80: {                                         // OpCompositeConstruct
            Vec v;
            for (std::uint32_t k = 2; k < n - 1; ++k) {
                const Vec& c = get(a[k]);
                v.insert(v.end(), c.begin(), c.end());
            }
            val[a[1]] = v;
            break;
        }
        case 124: val[a[1]] = get(a[2]); break;            // OpBitcast
        case 109: val[a[1]] = {static_cast<std::uint32_t>(f(s0(a[2])))}; break;                       // OpConvertFToU
        case 110: val[a[1]] = {static_cast<std::uint32_t>(static_cast<std::int32_t>(f(s0(a[2]))))}; break;  // OpConvertFToS
        case 111: val[a[1]] = {fb(static_cast<float>(static_cast<std::int32_t>(s0(a[2]))))}; break;    // OpConvertSToF
        case 112: val[a[1]] = {fb(static_cast<float>(s0(a[2])))}; break;                               // OpConvertUToF
        case 127: val[a[1]] = {fb(-f(s0(a[2])))}; break;  // OpFNegate
        case 129: bin_f([&](float x, float y) { return fb(x + y); }); break;
        case 131: bin_f([&](float x, float y) { return fb(x - y); }); break;
        case 133: bin_f([&](float x, float y) { return fb(x * y); }); break;
        case 136: bin_f([&](float x, float y) { return fb(x / y); }); break;
        case 128: bin_u([](std::uint32_t x, std::uint32_t y) { return x + y; }); break;
        case 130: bin_u([](std::uint32_t x, std::uint32_t y) { return x - y; }); break;
        case 149: {                                        // OpIAddCarry: {sum, carry}
            const std::uint64_t r = static_cast<std::uint64_t>(s0(a[2])) + s0(a[3]);
            val[a[1]] = {static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(r >> 32)};
            break;
        }
        case 150: {                                        // OpISubBorrow: {difference, borrow}
            const std::uint32_t x = s0(a[2]), y = s0(a[3]);
            val[a[1]] = {x - y, x < y ? 1u : 0u};
            break;
        }
        case 132: bin_u([](std::uint32_t x, std::uint32_t y) { return x * y; }); break;
        case 170: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x == y}; }); break;
        case 171: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x != y}; }); break;
        case 172: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x > y}; }); break;
        case 173: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{static_cast<std::int32_t>(x) > static_cast<std::int32_t>(y)}; }); break;
        case 174: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x >= y}; }); break;
        case 175: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{static_cast<std::int32_t>(x) >= static_cast<std::int32_t>(y)}; }); break;
        case 176: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x < y}; }); break;
        case 177: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{static_cast<std::int32_t>(x) < static_cast<std::int32_t>(y)}; }); break;
        case 178: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x <= y}; }); break;
        case 179: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{static_cast<std::int32_t>(x) <= static_cast<std::int32_t>(y)}; }); break;
        case 180: bin_f([](float x, float y) { return std::uint32_t{x == y}; }); break;
        case 182: bin_f([](float x, float y) { return std::uint32_t{x != y && x == x && y == y}; }); break;
        case 183: bin_f([](float x, float y) { return std::uint32_t{!(x == y)}; }); break;
        case 184: bin_f([](float x, float y) { return std::uint32_t{x < y}; }); break;
        case 185: bin_f([](float x, float y) { return std::uint32_t{!(x >= y)}; }); break;
        case 186: bin_f([](float x, float y) { return std::uint32_t{x > y}; }); break;
        case 187: bin_f([](float x, float y) { return std::uint32_t{!(x <= y)}; }); break;
        case 188: bin_f([](float x, float y) { return std::uint32_t{x <= y}; }); break;
        case 190: bin_f([](float x, float y) { return std::uint32_t{x >= y}; }); break;
        case 164: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{(x != 0) == (y != 0)}; }); break;
        case 165: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{(x != 0) != (y != 0)}; }); break;
        case 166: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x || y}; }); break;
        case 167: bin_u([](std::uint32_t x, std::uint32_t y) { return std::uint32_t{x && y}; }); break;
        case 168: val[a[1]] = {s0(a[2]) ? 0u : 1u}; break;  // OpLogicalNot
        case 169: val[a[1]] = s0(a[2]) ? get(a[3]) : get(a[4]); break;  // OpSelect (scalar condition)
        case 194: bin_u([](std::uint32_t x, std::uint32_t y) { return x >> (y & 31); }); break;
        case 195: bin_u([](std::uint32_t x, std::uint32_t y) { return static_cast<std::uint32_t>(static_cast<std::int32_t>(x) >> (y & 31)); }); break;
        case 196: bin_u([](std::uint32_t x, std::uint32_t y) { return x << (y & 31); }); break;
        case 197: bin_u([](std::uint32_t x, std::uint32_t y) { return x | y; }); break;
        case 198: bin_u([](std::uint32_t x, std::uint32_t y) { return x ^ y; }); break;
        case 199: bin_u([](std::uint32_t x, std::uint32_t y) { return x & y; }); break;
        case 200: val[a[1]] = {~s0(a[2])}; break;          // OpNot
        case 156: val[a[1]] = {f(s0(a[2])) != f(s0(a[2])) ? 1u : 0u}; break;  // OpIsNan
        case 12: {                                         // OpExtInst (GLSL.std.450)
            const std::uint32_t inst = a[3];
            const float x = f(s0(a[4]));
            const float y = n > 6 ? f(s0(a[5])) : 0.0f;
            const float z = n > 7 ? f(s0(a[6])) : 0.0f;
            float r = 0.0f;
            switch (inst) {
            case 4: r = std::fabs(x); break;
            case 8: r = std::floor(x); break;
            case 37: r = std::fmin(x, y); break;
            case 40: r = std::fmax(x, y); break;
            case 43: r = std::fmin(std::fmax(x, y), z); break;
            case 50: r = std::fma(x, y, z); break;
            default: return fail("GLSL.std.450 instruction " + std::to_string(inst));
            }
            val[a[1]] = {fb(r)};
            break;
        }
        default: return fail("unsupported opcode " + std::to_string(op));
        }
        pc = next;
    }
    return fail("did not return");
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
    // The interpreter on the one-exit loop: each pixel adds 1.0 per count.
    {
        const gcn::LiftResult r = lift(divergent_loop());
        for (const float count : {0.0f, 3.0f, 5.0f}) {
            const CfgRun run = run_cfg(r.spirv, {count, 1.0f, 0, 0});
            if (!run.ok) std::fprintf(stderr, "divergent loop: %s\n", run.error.c_str());
            CHECK(run.ok && run.out[0] == count);
        }
    }
    // Two ways out, as the ray-cast shaders leave: each inside an
    // s_cbranch_execz region on the lanes leaving, with moves those lanes make
    // on the way out, and an instruction no branch reaches after the back
    // edge. Each pixel: while i < limit, acc += step and leave once acc > 4;
    // the way out it took marks v6 (1.0 or 2.0), and the first also v7.
    const auto two_ways_out = [](bool distinct_masks) {
        Asm a;
        a.interp(2, 0, 0);                         // v2 = the limit
        a.interp(3, 0, 1);                         // v3 = the step
        a.vop1(kVMov, 4, kC0);                     // v4 = i
        a.vop1(kVMov, 5, kC0);                     // v5 = acc
        a.vop1(kVMov, 6, kC0);                     // v6 = the way out
        a.vop1(kVMov, 7, kC0);                     // v7
        a.sop1(kSMov64, 10, kExec);                // EXEC kept for after the loop
        a.sop1(kSMov64, 12, kExec);                // the loop mask L
        if (distinct_masks) a.sop1(kSMov64, 16, kExec);
        const std::uint32_t header = a.at();
        a.vopc(kCmpGtF32, kV + 2, 4);              // vcc = limit > i: stay
        a.sop1(kSMov64, 14, kExec);
        a.sop2(kSAndn2_64, kExec, 14, kVcc);       // EXEC = the lanes leaving
        a.vop1(kVMov, 6, kOne);                    // on the way out: v6 = 1.0
        const std::size_t skip1 = a.forward(kExecz);
        a.vop2(kVAddF32, 7, kOne, 5);              // v7 = acc + 1
        a.sop2(kSAndn2_64, 12, 12, kExec);         // L loses them
        const std::size_t exit1 = a.forward(kScc0);
        a.land(skip1);
        a.sop2(kSAnd64, kExec, 14, 12);            // EXEC = the lanes staying
        a.vop2(kVAddF32, 5, kV + 3, 5);            // acc += step
        a.vopc(kCmpLtF32, kFour, 5);               // vcc = 4 < acc: leave
        a.sop1(kSAndSaveexec, 14, kVcc);           // EXEC = the lanes leaving
        a.vop1(kVMov, 6, kTwo);                    // on the way out: v6 = 2.0
        const std::size_t skip2 = a.forward(kExecz);
        const std::uint32_t m2 = distinct_masks ? 16 : 12;
        a.sop2(kSAndn2_64, m2, m2, kExec);
        const std::size_t exit2 = a.forward(kScc0);
        a.land(skip2);
        a.sop2(kSAnd64, kExec, 14, 12);
        a.vop2(kVAddF32, 4, kOne, 4);              // i += 1
        a.branch(kBranch, header);
        a.vop1(kVMov, 6, kC0 + 9);                 // no branch reaches it
        a.land(exit1);
        a.land(exit2);
        a.sop1(kSMov64, kExec, 10);
        a.exp_mrt0(5, 6, 4, 7);
        a.sopp(kEndpgm);
        return a;
    };
    {
        const gcn::LiftResult r = lift(two_ways_out(false));
        CHECK(lifted_valid(r, "a divergent loop with two ways out"));
        CHECK(proof_has(r, "2 exits joined"));
        CHECK(proof_has(r, "(in the region at"));
        CHECK(proof_has(r, "after the back edge is reached by no branch and is not lifted"));
        const auto expect = [](float limit, float step) {
            float i = 0, acc = 0, way = 0, v7 = 0;
            for (;;) {
                if (!(limit > i)) {
                    way = 1.0f;
                    v7 = acc + 1.0f;
                    break;
                }
                acc += step;
                if (4.0f < acc) {
                    way = 2.0f;
                    break;
                }
                i += 1.0f;
            }
            return std::array<float, 4>{acc, way, i, v7};
        };
        for (const auto [limit, step] : {std::pair{3.0f, 1.0f}, std::pair{10.0f, 1.5f}, std::pair{0.0f, 1.0f}, std::pair{2.0f, 5.0f}}) {
            const CfgRun run = run_cfg(r.spirv, {limit, step, 0, 0});
            if (!run.ok) std::fprintf(stderr, "a divergent loop with two ways out: %s\n", run.error.c_str());
            CHECK(run.ok);
            const std::array<float, 4> want = expect(limit, step);
            for (std::size_t k = 0; k < 4 && run.ok; ++k) {
                if (run.out[k] != want[k]) {
                    std::fprintf(stderr, "two ways out (limit %g, step %g): output %zu is %g, expected %g\n", limit, step, k, run.out[k], want[k]);
                    ++g_failures;
                }
            }
        }
    }
    // The second way out taking lanes out of another mask: refused.
    CHECK(rejected_with(lift(two_ways_out(true)), "on different masks"));
    // The loop mask must start as EXEC.
    CHECK(rejected_with(lift(divergent_loop(1)), "does not start as EXEC"));
    // GCN may run more iterations than the pixel: a scalar the loop writes is dead after it.
    CHECK(rejected_with(lift(divergent_loop(2)), "written inside the divergent loop"));
    // A second way out of the loop: acc += x until s5 reaches s0 or 7, the
    // exits joined at the merge; run for both ways out.
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
        const gcn::LiftResult r = lift(b);
        CHECK(lifted_valid(r, "a uniform loop with two exits"));
        CHECK(proof_has(r, "2 exits joined"));
        for (const auto [count, want] : {std::pair{3u, 3.0f * 1.5f}, std::pair{12u, 7.0f * 1.5f}}) {
            std::array<std::uint32_t, 16> user{};
            user[0] = count;
            const CfgRun run = run_cfg(r.spirv, {1.5f, 0, 0, 0}, user);
            if (!run.ok) std::fprintf(stderr, "a uniform loop with two exits: %s\n", run.error.c_str());
            CHECK(run.ok && run.out[0] == want);
        }
    }
    // A branch from inside the loop past its exit, over an instruction no
    // branch reaches (the copies of a dead path the compiler leaves after the
    // back edge): the exit is there, and that instruction is not lifted. One
    // a branch does reach is still refused.
    const auto past_exit = [](bool reached) {
        Asm b;
        b.interp(2, 0, 0);
        b.vop1(kVMov, 3, kC0);
        b.sop1(kSMov, 5, kC0);
        b.sopc(kSCmpEqI32, 0, kC0);                // s0 == 0: around the loop (and, reached, into the skipped instruction)
        const std::size_t into = b.forward(kScc1);
        const std::uint32_t header = b.at();
        b.sopc(kSCmpGeI32, 5, 0);
        const std::size_t far = b.forward(kScc1);
        b.vop2(kVAddF32, 3, kV + 2, 3);
        b.sop2(kSAddU32, 5, 5, kC0 + 1);
        b.branch(kBranch, header);
        if (reached) b.land(into);
        b.vop1(kVMov, 3, kC0);  // skipped by the exit
        b.land(far);
        if (!reached) b.land(into);
        b.exp_mrt0(3);
        b.sopp(kEndpgm);
        return b;
    };
    {
        const gcn::LiftResult r = lift(past_exit(false));
        CHECK(lifted_valid(r, "an exit past instructions no branch reaches"));
        CHECK(proof_has(r, "after the back edge is reached by no branch and is not lifted"));
    }
    CHECK(rejected_with(lift(past_exit(true)), "for somewhere other than its exit"));
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

// A uniform loop counting in `counter` up to the user SGPR s1. s0 and s1 are
// registers like any other (only a loop where pixels leave at different
// iterations has a mask pair the lift treats apart).
Asm counted_loop(std::uint32_t counter) {
    Asm a;
    a.interp(2, 0, 0);
    a.sop1(kSMov, counter, kC0);                    // s<counter> = 0
    a.vop1(kVMov, 3, kC0);                          // v3 = 0
    const std::uint32_t header = a.at();
    a.sopc(kSCmpGeI32, counter, 1);                 // s<counter> >= s1: done
    const std::size_t exit = a.forward(kScc1);
    a.vop2(kVAddF32, 3, kOne, 3);                   // v3 += 1.0
    a.sop2(kSAddU32, counter, counter, kC0 + 1);    // s<counter> += 1
    a.branch(kBranch, header);
    a.land(exit);
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

// A lane mask in s[mask:mask+1] flipped each iteration of a loop counted by
// the user SGPR s2, selecting v3 into v4.
Asm flipped_mask(std::uint32_t mask) {
    Asm b;
    b.interp(2, 0, 0);
    b.vop1(kVMov, 3, kC0);
    b.sop1(kSMov, 5, kC0);
    b.vopc(kCmpLtF32, kC0, 2);                      // vcc = 0 < x
    b.sop1(kSMov64, mask, kVcc);
    const std::uint32_t header = b.at();
    b.sopc(kSCmpGeI32, 5, 2);                       // s5 >= s2: done
    const std::size_t exit = b.forward(kScc1);
    b.sop2(kSXor64, mask, mask, kExec);             // flip
    b.w.push_back(0xd2000000u | 4u);                // v_cndmask_b32 v4, 0, v3, s[mask:mask+1] (VOP3 0x100)
    b.w.push_back(kC0 | (kV + 3) << 9 | mask << 18);
    b.vop2(kVAddF32, 3, kV + 4, 3);
    b.sop2(kSAddU32, 5, 5, kC0 + 1);
    b.branch(kBranch, header);
    b.land(exit);
    b.exp_mrt0(3);
    b.sopp(kEndpgm);
    return b;
}

void loop_registers() {
    {
        const gcn::LiftResult s0 = lift(counted_loop(0)), s5 = lift(counted_loop(5));
        CHECK(lifted_valid(s0, "a uniform loop counting in s0"));
        CHECK(proof_has(s0, "2 phis (1 VGPRs, 0 lane masks)"));  // the counter is carried around the loop
        CHECK(s0.ok() && s5.ok() && s0.spirv == s5.spirv);
    }
    for (const std::uint32_t mask : {0u, 20u}) {
        const gcn::LiftResult r = lift(flipped_mask(mask));
        CHECK(lifted_valid(r, "a lane mask flipped around a uniform loop"));
        CHECK(proof_has(r, "1 lane masks)"));
    }
}

constexpr std::uint32_t kSCmpLtI32 = 4, kCmpGeI32 = 0x86, kCmpEqU32 = 0xc2;  // SOPC, VOPC

// A uniform loop counted by s5 whose exit sits inside an s_cbranch_execz
// region of its body (EXEC = the pixels with s5 >= int(x)): GCN tests it
// only in iterations where some pixel enters the region.
Asm exit_in_region(bool in_if_arm) {
    Asm a;
    a.interp(2, 0, 0);
    a.vop1(kVCvtI32F32, 4, kV + 2);
    a.vop1(kVMov, 3, kC0);
    a.sop1(kSMov, 5, kC0);
    const std::uint32_t header = a.at();
    a.vopc(kCmpGeI32, 5, 4);                        // vcc = s5 >= v4
    a.sop1(kSAndSaveexec, 10, kVcc);
    const std::size_t region = a.forward(kExecz);
    std::size_t exit;
    if (in_if_arm) {
        a.sopc(kSCmpLtI32, 5, 0);                   // s5 < s0: stay
        const std::size_t skip = a.forward(kScc1);
        exit = a.forward(kBranch);                  // the then arm leaves the loop
        a.land(skip);
    } else {
        a.sopc(kSCmpGeI32, 5, 0);                   // s5 >= s0: done
        exit = a.forward(kScc1);
    }
    a.vop2(kVAddF32, 6, kOne, 6);
    a.land(region);
    a.sop1(kSMov64, kExec, 10);
    a.vop2(kVAddF32, 3, kOne, 3);
    a.sop2(kSAddU32, 5, 5, kC0 + 1);
    a.branch(kBranch, header);
    a.land(exit);
    a.sop1(kSMov64, kExec, 10);
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

// The per-lane exit idiom (each pixel adds while v5 < int(x)) with `body`
// after the exit test: before EXEC is set to the lanes staying when
// `before_restore`, else after it. Branches the body records in `exits`
// land at the loop's exit.
Asm divergent_body(const std::function<void(Asm&, std::vector<std::size_t>&)>& body, bool before_restore = false,
                   std::uint32_t mask_from = kExec) {
    std::vector<std::size_t> exits;
    Asm a;
    a.interp(2, 0, 0);
    a.vop1(kVCvtI32F32, 4, kV + 2);
    a.vop1(kVMov, 3, kC0);
    a.vop1(kVMov, 5, kC0);
    a.sop1(kSMov64, 10, kExec);                     // EXEC at entry, for after the loop
    a.sop1(kSMov64, 12, mask_from);                 // the loop mask L
    const std::uint32_t header = a.at();
    a.vopc(kCmpGtI32, kV + 4, 5);                   // vcc = v4 > v5: stay
    a.sop1(kSMov64, 14, kExec);
    a.sop2(kSAndn2_64, kExec, 14, kVcc);            // EXEC = the lanes leaving
    a.sop2(kSAndn2_64, 12, 12, kExec);              // L loses them
    const std::size_t exit = a.forward(kScc0);
    if (before_restore) body(a, exits);
    a.sop2(kSAnd64, kExec, 14, 12);                 // EXEC = the lanes staying
    if (!before_restore) body(a, exits);
    a.vop2(kVAddI32, 5, kC0 + 1, 5);
    a.branch(kBranch, header);
    a.land(exit);
    for (const std::size_t e : exits) a.land(e);
    a.sop1(kSMov64, kExec, 10);
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

void loop_exits_and_masks() {
    CHECK(rejected_with(lift(exit_in_region(false)), "inside a region of its body"));
    CHECK(rejected_with(lift(exit_in_region(true)), "inside a region of its body"));

    // The game's idiom in a loop pixels leave at different iterations: EXEC
    // narrowed by s_and_saveexec_b64 around a region and restored from the
    // saved mask, within the loop mask throughout.
    const auto add = [](Asm& a) { a.vop2(kVAddF32, 3, kOne, 3); };
    {
        const gcn::LiftResult r = lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) {
            a.vopc(kCmpLtF32, kC0, 2);              // vcc = 0 < x
            a.sop1(kSAndSaveexec, 16, kVcc);
            add(a);
            const std::size_t skip = a.forward(kExecz);
            add(a);
            a.land(skip);
            a.sop1(kSMov64, kExec, 16);
            add(a);
        }));
        CHECK(lifted_valid(r, "a region in a loop pixels leave at different iterations"));
        CHECK(proof_has(r, "divergent, the per-lane exit idiom on s12"));
    }
    // EXEC widened past the loop mask: GCN runs the lanes that left again.
    // Whole-quad mode (their quad-mates may still be in the loop) ...
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) {
                            a.sop1(kSMov64, 16, kExec);
                            a.sop1(kSWqm, kExec, kExec);
                            add(a);
                            a.sop1(kSMov64, kExec, 16);
                        })),
                        "whole-quad mode inside a loop pixels leave at different iterations"));
    // ... EXEC at the loop's entry restored, after the lanes staying were set or before ...
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) {
                            a.sop1(kSMov64, 16, kExec);
                            a.sop1(kSMov64, kExec, 10);
                            add(a);
                            a.sop1(kSMov64, kExec, 16);
                        })),
                        "beyond the loop mask"));
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) {
                            a.sop1(kSMov64, kExec, 10);
                            add(a);
                        }, true)),
                        "beyond the loop mask"));
    // ... or a write under the lanes leaving, which have left in the lift.
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) { add(a); }, true)), "beyond the loop mask"));
    // The loop mask refilled with EXEC at entry: the lanes that left rejoin it (GCN never ends the loop).
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) {
                            add(a);
                            a.sop1(kSMov64, 12, 10);
                        })),
                        "loop mask"));
    // A wave-wide exit on "some lane left this iteration" (s14 & ~s12): the
    // lift knows the mask only for the pixel it runs.
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>& exits) {
                            add(a);
                            a.sop2(kSAndn2_64, 18, 14, 12);
                            exits.push_back(a.forward(kScc1));
                        })),
                        "leaves the loop at"));
    // A uniform loop's exit on "some lane outside EXEC" (-1 & ~EXEC): set on
    // GCN in the helper and empty lanes, which the lift does not run.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.vop1(kVMov, 3, kC0);
        a.sop1(kSMov, 5, kC0);
        const std::uint32_t header = a.at();
        a.sopc(kSCmpGeI32, 5, 0);
        const std::size_t exit1 = a.forward(kScc1);
        a.vop2(kVAddF32, 3, kOne, 3);
        a.sop2(kSAndn2_64, 20, 193 /* -1 */, kExec);
        const std::size_t exit2 = a.forward(kScc1);
        a.sop2(kSAddU32, 5, 5, kC0 + 1);
        a.branch(kBranch, header);
        a.land(exit1);
        a.land(exit2);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        CHECK(rejected_with(lift(a), "leaves the loop at"));
    }
    // x & ~(x & EXEC) with x = ~vcc, after whole-quad mode: clear for every
    // pixel, but set on GCN in the lanes no pixel occupies (EXEC holds none of
    // them), so the lift must not take x & EXEC for x.
    {
        Asm a;
        a.sop1(kSWqm, kExec, kExec);
        a.interp(2, 0, 0);
        a.vop1(kVMov, 3, kC0);
        a.vopc(kCmpGtF32, kC0, 2);                  // vcc = 0 > x
        a.sop2(kSAndn2_64, 20, 193 /* -1 */, kVcc);  // s[20:21] = ~vcc
        a.sop2(kSAnd64, 22, 20, kExec);             // s[22:23] = ~vcc & EXEC
        a.sop1(kSMov, 5, kC0);
        const std::uint32_t header = a.at();
        a.sopc(kSCmpGeI32, 5, 0);
        const std::size_t exit1 = a.forward(kScc1);
        a.vop2(kVAddF32, 3, kOne, 3);
        a.sop2(kSAndn2_64, 24, 20, 22);             // the lanes outside EXEC
        const std::size_t exit2 = a.forward(kScc1);
        a.sop2(kSAddU32, 5, 5, kC0 + 1);
        a.branch(kBranch, header);
        a.land(exit1);
        a.land(exit2);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        CHECK(rejected_with(lift(a), "leaves the loop at"));
    }
    // The per-lane exit pair inside an if arm of a uniform loop, where the mask
    // is EXEC itself: x & ~x is clear in every lane of the wave, so SCC is
    // clear and the arm leaves the loop.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.vop1(kVMov, 3, kC0);
        a.vopc(kCmpLtF32, kC0, 2);                  // vcc = 0 < x
        a.sop1(kSAndSaveexec, 10, kVcc);            // EXEC = the pixels with 0 < x
        const std::size_t region = a.forward(kExecz);
        a.sop1(kSMov64, 20, kExec);
        a.sop1(kSMov, 5, kC0);
        const std::uint32_t header = a.at();
        a.sopc(kSCmpGeI32, 5, 0);                   // s5 >= s0: the then arm
        const std::size_t skip = a.forward(kScc0);
        a.sop2(kSAndn2_64, 20, 20, kExec);          // EXEC & ~EXEC
        const std::size_t exit = a.forward(kScc0);  // always taken
        a.land(skip);
        a.vop2(kVAddF32, 3, kOne, 3);
        a.sop2(kSAddU32, 5, 5, kC0 + 1);
        a.branch(kBranch, header);
        a.land(exit);
        a.land(region);
        a.sop1(kSMov64, kExec, 10);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        const gcn::LiftResult r = lift(a);
        CHECK(lifted_valid(r, "a uniform loop left through the per-lane exit pair in an if arm"));
        CHECK(proof_has(r, "its then arm leaves the loop"));
    }
    // v_cmpx narrows EXEC in each iteration of a uniform loop: EXEC does not
    // come back to the header as it left.
    {
        Asm a;
        a.interp(2, 0, 0);
        a.vop1(kVMov, 3, kC0);
        a.sop1(kSMov, 5, kC0);
        const std::uint32_t header = a.at();
        a.sopc(kSCmpGeI32, 5, 0);
        const std::size_t exit = a.forward(kScc1);
        a.vopc(0x11 /* v_cmpx_lt_f32 */, kV + 3, 2);  // EXEC &= v3 < x
        a.vop2(kVAddF32, 3, kOne, 3);
        a.sop2(kSAddU32, 5, 5, kC0 + 1);
        a.branch(kBranch, header);
        a.land(exit);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        CHECK(rejected_with(lift(a), "exec comes back to the header of the loop at"));
    }
    // The loop mask must start as a copy of EXEC, not as an equal constant (-1 has every lane, EXEC only the running pixels').
    CHECK(rejected_with(lift(divergent_body([&](Asm& a, std::vector<std::size_t>&) { add(a); }, false, 193 /* -1 */)), "does not start as EXEC"));
}

// An alpha-test kill region; after it EXEC stays as the region narrowed it,
// or is set again from the guard. GCN skips the region when no pixel of the
// wave passes, so EXEC there keeps the pixels the lift kills.
Asm kill_then_export(bool restore_from_guard) {
    Asm a;
    a.sop1(kSMov64, 20, kExec);
    a.interp(2, 0, 0);
    a.vop1(kVMov, 3, kTwo);
    a.vopc(kCmpGtF32, kC0, 2);                      // vcc = 0 > x: kill
    a.sop2(kSAndn2_64, 20, 20, kVcc);
    const std::size_t end = a.forward(kScc0);
    a.sop2(kSAnd64, kExec, kExec, 20);
    a.vop2(kVMulF32, 3, kV + 2, 2);
    a.land(end);
    if (restore_from_guard) a.sop1(kSMov64, kExec, 20);
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

// A uniform loop counted by the user SGPR s1 inside an s_cbranch_execz
// region: the lift runs it for every pixel, also in waves GCN skips.
Asm loop_in_region() {
    Asm a;
    a.interp(2, 0, 0);
    a.vop1(kVMov, 3, kC0);
    a.vopc(kCmpLtF32, kC0, 2);                      // vcc = 0 < x
    a.sop1(kSAndSaveexec, 10, kVcc);
    const std::size_t region = a.forward(kExecz);
    a.sop1(kSMov, 5, kC0);
    const std::uint32_t header = a.at();
    a.sopc(kSCmpGeI32, 5, 1);
    const std::size_t exit = a.forward(kScc1);
    a.vop2(kVAddF32, 3, kOne, 3);
    a.sop2(kSAddU32, 5, 5, kC0 + 1);
    a.branch(kBranch, header);
    a.land(exit);
    a.land(region);
    a.sop1(kSMov64, kExec, 10);
    a.exp_mrt0(3);
    a.sopp(kEndpgm);
    return a;
}

void regions_and_kills() {
    CHECK(lifted_valid(lift(kill_then_export(true)), "a kill region, EXEC set from the guard after it"));
    CHECK(rejected_with(lift(kill_then_export(false)), "EXEC as the kill region at"));
    {
        const gcn::LiftResult r = lift(loop_in_region());
        CHECK(lifted_valid(r, "a uniform loop inside a region"));
        CHECK(proof_has(r, "the count GCN runs in any wave that enters it"));
    }
    // A uniform VGPR written outside whole-quad mode (EXEC = the coverage)
    // and compared in it: GCN's helper lanes kept v5 = 0, and their VCC bits
    // take part in the wave-wide s_cbranch_vccz.
    {
        Asm a;
        a.sop1(kSMov64, 20, kExec);                 // the coverage
        a.sop1(kSWqm, kExec, kExec);
        a.interp(2, 0, 0);
        a.vop1(kVMov, 5, kC0);                      // v5 = 0 in every lane of the quads
        a.sop1(kSAndSaveexec, 22, 20);              // EXEC = the coverage
        a.vop1(kVMov, 5, 0);                        // v5 = s0 in the covered pixels
        a.sop1(kSMov64, kExec, 22);                 // whole-quad mode again
        a.vopc(kCmpEqU32, kC0, 5);                  // vcc = 0 == v5
        const std::size_t to_else = a.forward(kVccz);
        a.vop1(kVMov, 6, kOne);
        const std::size_t to_join = a.forward(kBranch);
        a.land(to_else);
        a.vop1(kVMov, 6, kTwo);
        a.land(to_join);
        a.sop1(kSMov64, kExec, 20);
        a.exp_mrt0(6);
        a.sopp(kEndpgm);
        CHECK(rejected_with(lift(a), "VCC that may differ between lanes"));
    }
    // Whole quads of a mask clear in every covered pixel but set in GCN's
    // helper lanes (the complement of the coverage): GCN sets the covered
    // pixels of every partly covered quad.
    {
        Asm a;
        a.sop2(kSAndn2_64, 0, 193 /* -1 */, kExec);  // s[0:1] = ~EXEC
        a.sop1(kSWqm, 2, 0);                        // s[2:3] = whole quads of it
        a.w.push_back(0xd2000000u | 3u);            // v_cndmask_b32 v3, 0, 1.0, s[2:3]
        a.w.push_back(kC0 | kOne << 9 | 2u << 18);
        a.exp_mrt0(3);
        a.sopp(kEndpgm);
        CHECK(rejected_with(lift(a), "whole-quad mode of a mask"));
    }
}

void run() {
    loops();
    regions_and_ifs();
    loop_registers();
    loop_exits_and_masks();
    regions_and_kills();
}

}  // namespace control_flow

// Small synthetic pixel shaders for the lane writes and the remaining scalar
// and 64-bit instructions: SGPR spills (v_writelane_b32 / v_readlane_b32),
// scalar loads that only fetch descriptors, s_cselect_b32 and v_lshl_b64.
// Each lift is checked against the reasons it must give, and each lifted
// module is validated. No game files needed.
namespace lane_ops {

// Encoders for the fields gcn/decode.cpp reads.
struct Asm {
    std::vector<std::uint32_t> w;
    std::uint32_t at() const { return static_cast<std::uint32_t>(w.size() * 4); }
    void sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc) { w.push_back(0xbe800000u | sdst << 16 | op << 8 | ssrc); }
    void sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t s0, std::uint32_t s1) {
        w.push_back(0x80000000u | op << 23 | sdst << 16 | s1 << 8 | s0);
    }
    void sopc(std::uint32_t op, std::uint32_t s0, std::uint32_t s1) { w.push_back(0xbf000000u | op << 16 | s1 << 8 | s0); }
    void sopp(std::uint32_t op, std::int16_t simm = 0) { w.push_back(0xbf800000u | op << 16 | static_cast<std::uint16_t>(simm)); }
    // A forward branch to `target` (a byte offset), from the next instruction slot.
    void branch(std::uint32_t op, std::uint32_t target) { sopp(op, static_cast<std::int16_t>((static_cast<int>(target) - static_cast<int>(at()) - 4) / 4)); }
    void smrd(std::uint32_t op, std::uint32_t sdst, std::uint32_t sbase, std::uint32_t offset_dw) {
        w.push_back(0xc0000000u | op << 22 | sdst << 15 | (sbase / 2) << 9 | 1u << 8 | offset_dw);
    }
    // The same with the byte offset in SGPR `soff` (IMM clear).
    void smrd_reg(std::uint32_t op, std::uint32_t sdst, std::uint32_t sbase, std::uint32_t soff) {
        w.push_back(0xc0000000u | op << 22 | sdst << 15 | (sbase / 2) << 9 | soff);
    }
    // tbuffer_load_format_* (op 0-3) vdata, vaddr (an index, idxen), s[srsrc..+3], soffset 0, format dfmt/nfmt.
    void tbuffer_load(std::uint32_t op, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t srsrc, std::uint32_t dfmt, std::uint32_t nfmt) {
        w.push_back(0xe8000000u | nfmt << 23 | dfmt << 19 | op << 16 | 1u << 13);
        w.push_back(vaddr | vdata << 8 | (srsrc / 4) << 16 | 128u << 24);
    }
    void vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | op << 9 | src0); }
    // `vsrc1` is the 8-bit field: a VGPR index, or for v_readlane/v_writelane_b32 the lane's scalar code.
    void vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(op << 25 | vdst << 17 | vsrc1 << 9 | src0); }
    void vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(0x7c000000u | op << 17 | vsrc1 << 9 | src0); }
    void vop3(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2 = 0) {
        w.push_back(0xd0000000u | op << 17 | vdst);
        w.push_back(src0 | src1 << 9 | src2 << 18);
    }
    void interp(std::uint32_t vdst, std::uint32_t attr, std::uint32_t chan) {  // v_interp_p1_f32 / p2_f32 over v0, v1
        w.push_back(0xc8000000u | vdst << 18 | 0u << 16 | attr << 10 | chan << 8 | 0u);
        w.push_back(0xc8000000u | vdst << 18 | 1u << 16 | attr << 10 | chan << 8 | 1u);
    }
    void mimg(std::uint32_t op, std::uint32_t dmask, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t srsrc, std::uint32_t ssamp) {
        w.push_back(0xf0000000u | op << 18 | dmask << 8);
        w.push_back(vaddr | vdata << 8 | (srsrc / 4) << 16 | (ssamp / 4) << 21);
    }
    void exp_mrt0(std::uint32_t v0, std::uint32_t v1, std::uint32_t v2, std::uint32_t v3) {  // exp mrt0 ... done vm
        w.push_back(0xf8000000u | 1u << 12 | 1u << 11 | 0xfu);
        w.push_back(v0 | v1 << 8 | v2 << 16 | v3 << 24);
    }
    void endpgm() { sopp(1); }
};

// Operand codes.
constexpr std::uint32_t kV = 256;  // + VGPR index, for 9-bit operands
constexpr std::uint32_t kC0 = 128;  // the inline integer 0; kC0 + n is n (0..64)
constexpr std::uint32_t kHalf = 240, kOne = 242, kTwo = 244;  // 0.5, 1.0, 2.0
constexpr std::uint32_t kVcc = 106, kExec = 126;
// Opcodes.
constexpr std::uint32_t kSMov = 3, kSMov64 = 4, kSAndSaveexec = 36;            // SOP1
constexpr std::uint32_t kSCselect = 10, kSAnd64 = 15;                          // SOP2
constexpr std::uint32_t kSCmpEqU32 = 6;                                        // SOPC
constexpr std::uint32_t kBranch = 2, kScc1 = 5, kExecz = 8, kWaitcnt = 12;     // SOPP
constexpr std::uint32_t kLoadX4 = 2, kLoadX8 = 3;                              // SMRD
constexpr std::uint32_t kVMov = 1, kCvtF32U32 = 6, kCvtU32F32 = 7;             // VOP1
constexpr std::uint32_t kReadlane = 1, kWritelane = 2, kVAdd = 3, kVMul = 8;   // VOP2
constexpr std::uint32_t kCmpLt = 1;                                            // VOPC
constexpr std::uint32_t kLshlB64 = 0x161;                                      // VOP3
constexpr std::uint32_t kSampleLz = 39;                                        // MIMG

// Translates as the renderer's no-fallback variant (user SGPRs s0..s3,
// interpolation at the centre) and lifts against that reference.
// `exclude`: resource paths left to the page table (cb_ssbo_exclude).
gcn::LiftResult lift_words(const std::vector<std::uint32_t>& words, gcn::Program* out = nullptr, gcn::TranslateResult* reference = nullptr,
                           const std::vector<std::string>& exclude = {}) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    CHECK(p.errors.empty());
    if (out) *out = p;
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc2 = 4u << 1;    // four user SGPRs
    o.ps_input_ena = 2;   // PERSP_CENTER: i, j in v0, v1
    o.descriptor_set = 1;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    o.cb_ssbo_exclude = exclude;
    const gcn::TranslateResult ref = gcn::translate(p, o);
    if (!ref.ok()) std::fprintf(stderr, "reference failed: %s\n", ref.errors[0].c_str());
    CHECK(ref.ok());
    if (reference) *reference = ref;
    return gcn::lift_pixel_shader(p, o, ref);
}

bool lifted_valid(const gcn::LiftResult& r) {
    for (const std::string& why : r.rejections) std::fprintf(stderr, "rejected: %s\n", why.c_str());
    if (!r.ok()) return false;
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    if (!tools.Validate(r.spirv.data(), r.spirv.size(), vo)) {
        std::fprintf(stderr, "lifted SPIR-V invalid: %s\n", msg.c_str());
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

// Whether the module loads user SGPR `k` (params member 2, uvec4 k / 4, component k % 4).
bool loads_user_sgpr(const std::vector<std::uint32_t>& spv, std::uint32_t k) {
    std::map<std::uint32_t, std::uint32_t> consts;  // id -> value of 32-bit OpConstants
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t op = spv[i] & 0xffff, len = spv[i] >> 16;
        if (!len || i + len > spv.size()) break;
        if (op == 43 && len == 4) consts[spv[i + 2]] = spv[i + 3];
        if (op == 65 && len == 7) {  // OpAccessChain type id base i0 i1 i2
            const auto c = [&](std::size_t j) { const auto it = consts.find(spv[i + j]); return it == consts.end() ? ~0u : it->second; };
            if (c(4) == 2 && c(5) == k / 4 && c(6) == k % 4) return true;
        }
        i += len;
    }
    return false;
}

// Whether the module has a 64-bit OpShiftLeftLogical (the runtime v_lshl_b64).
bool has_u64_shift(const std::vector<std::uint32_t>& spv) {
    std::set<std::uint32_t> u64;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t op = spv[i] & 0xffff, len = spv[i] >> 16;
        if (!len || i + len > spv.size()) break;
        if (op == 21 && len == 4 && spv[i + 2] == 64) u64.insert(spv[i + 1]);  // OpTypeInt 64
        if (op == 196 && u64.count(spv[i + 1])) return true;
        i += len;
    }
    return false;
}

// v2 = attr0.x, per pixel.
Asm start() {
    Asm a;
    a.interp(2, 0, 0);
    return a;
}

void spills() {
    // Parked in lane 3 of v5, s2 is clobbered and comes back through the cell
    // into s6: the lift reads user SGPR 2 there, as the reference's cell does.
    const auto spill = [](const std::function<void(Asm&)>& between, std::uint32_t read_lane = 3) {
        Asm a = start();
        a.vop2(kWritelane, 5, 2, kC0 + 3);  // v_writelane_b32 v5, s2, 3
        a.sop1(kSMov, 2, kC0);              // s_mov_b32 s2, 0
        between(a);
        a.vop2(kReadlane, 6, kV + 5, kC0 + read_lane);  // v_readlane_b32 s6, v5, read_lane
        a.vop2(kVMul, 3, 6, 2);                          // v_mul_f32 v3, s6, v2
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        return a.w;
    };
    const gcn::LiftResult ok = lift_words(spill([](Asm&) {}));
    CHECK(lifted_valid(ok));
    CHECK(proof_has(ok, "1 v_readlane_b32 of spilled scalars"));
    CHECK(loads_user_sgpr(ok.spirv, 2));
    // The same through the VOP3 forms.
    {
        Asm a = start();
        a.vop3(0x102, 5, 2, kC0 + 3);       // v_writelane_b32 v5, s2, 3
        a.sop1(kSMov, 2, kC0);
        a.vop3(0x101, 6, kV + 5, kC0 + 3);  // v_readlane_b32 s6, v5, 3
        a.vop2(kVMul, 3, 6, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        const gcn::LiftResult r = lift_words(a.w);
        CHECK(lifted_valid(r));
        CHECK(loads_user_sgpr(r.spirv, 2));
    }
    // A lane nothing parked a scalar in: the reference shuffles, reading another pixel.
    CHECK(rejected_with(lift_words(spill([](Asm&) {}, 4)), "no v_writelane_b32 alone reaches"));
    // Read back, v5 is written again under a varying EXEC (an execz region):
    // GCN's value where EXEC is set, the parked word or what the lane held
    // before elsewhere - so exact only where EXEC was set. Read there (inside
    // the region) it lifts, and so does a value made from it under the same
    // EXEC; exported after the region before written whole, it does not.
    {
        const auto reuse = [](bool export_v5_after) {
            Asm a = start();                        // 000000
            a.vop2(kWritelane, 5, 2, kC0 + 3);      // 000008 v_writelane_b32 v5, s2, 3
            a.sop1(kSMov, 2, kC0);                  // 00000c s_mov_b32 s2, 0
            a.vop2(kReadlane, 6, kV + 5, kC0 + 3);  // 000010 v_readlane_b32 s6, v5, 3
            a.vopc(kCmpLt, kHalf, 2);               // 000014 v_cmp_lt_f32 vcc, 0.5, v2
            a.sop1(kSAndSaveexec, 10, kVcc);        // 000018 s_and_saveexec_b64 s[10:11], vcc
            a.branch(kExecz, 0x28);                 // 00001c s_cbranch_execz 0x28
            a.vop1(kVMov, 5, kV + 2);               // 000020 v_mov_b32 v5, v2: over the parked lane
            a.vop2(kVMul, 7, kV + 5, 2);            // 000024 v_mul_f32 v7, v5, v2: v5 read where it is exact
            a.sop1(kSMov64, kExec, 10);             // 000028 s_mov_b64 exec, s[10:11]
            if (!export_v5_after) a.vop1(kVMov, 5, kOne);  // v_mov_b32 v5, 1.0: written whole
            a.vop2(kVMul, 3, 6, 2);                 // v_mul_f32 v3, s6, v2
            a.exp_mrt0(3, 7, 5, 3);
            a.endpgm();
            return a.w;
        };
        const gcn::LiftResult r = lift_words(reuse(false));
        CHECK(lifted_valid(r));
        CHECK(proof_has(r, "1 over a VGPR whose lanes held spilled words, exact only where EXEC was set"));
        CHECK(rejected_with(lift_words(reuse(true)), "an export reads a value the lift may hold differently from GCN in pixels of EXEC"));
    }
    // v5 differs between lanes now: it is not this pixel's word.
    CHECK(rejected_with(lift_words(spill([](Asm& a) { a.vop1(kVMov, 7, kV + 5); })), "v5 holds a word in one lane"));
    // Written whole again it is, but the cell is gone (the reference shuffles).
    {
        Asm a = start();
        a.vop2(kWritelane, 5, 2, kC0 + 3);
        a.vop1(kVMov, 5, kOne);       // v_mov_b32 v5, 1.0
        a.vop2(kVMul, 3, kV + 5, 2);  // v_mul_f32 v3, v5, v2
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        CHECK(lifted_valid(lift_words(a.w)));
    }
    CHECK(rejected_with(lift_words(spill([](Asm& a) { a.vop1(kVMov, 5, kOne); })), "no v_writelane_b32 alone reaches"));
    // A lane held in a register, writing or reading.
    {
        Asm a = start();
        a.vop2(kWritelane, 5, 2, 3);  // v_writelane_b32 v5, s2, s3: every cell of v5 is gone
        a.vop2(kReadlane, 6, kV + 5, kC0 + 3);
        a.vop2(kVMul, 3, 6, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        CHECK(rejected_with(lift_words(a.w), "no v_writelane_b32 alone reaches"));
    }
    {
        Asm a = start();
        a.vop2(kWritelane, 5, 2, kC0 + 3);
        a.vop2(kReadlane, 6, kV + 5, 3);  // v_readlane_b32 s6, v5, s3
        a.vop2(kVMul, 3, 6, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        CHECK(rejected_with(lift_words(a.w), "lane held in a register"));
    }
    // A lane mask is not a word to park.
    {
        Asm a = start();
        a.vop3(kCmpLt, 8, kHalf, kV + 2);   // v_cmp_lt_f32 s[8:9], 0.5, v2
        a.vop2(kWritelane, 5, 8, kC0 + 3);  // v_writelane_b32 v5, s8, 3
        a.vop2(kReadlane, 6, kV + 5, kC0 + 3);
        a.vop2(kVMul, 3, 6, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        CHECK(rejected_with(lift_words(a.w), "s8 holds a lane mask"));
    }
    // Parked on both arms of an if on a scalar condition: the cell joins as a phi.
    {
        Asm a = start();
        a.sopc(kSCmpEqU32, 0, kC0);  // s_cmp_eq_u32 s0, 0
        const std::uint32_t branch = a.at();
        a.sopp(kScc1);                      // s_cbranch_scc1 else (patched below)
        a.vop2(kWritelane, 5, 1, kC0);      // then: v_writelane_b32 v5, s1, 0
        const std::uint32_t jump = a.at();
        a.sopp(kBranch);                    // s_branch join (patched below)
        const std::uint32_t else_at = a.at();
        a.vop2(kWritelane, 5, 2, kC0);      // else: v_writelane_b32 v5, s2, 0
        const std::uint32_t join = a.at();
        a.vop2(kReadlane, 6, kV + 5, kC0);  // v_readlane_b32 s6, v5, 0
        a.vop2(kVMul, 3, 6, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        a.w[branch / 4] |= static_cast<std::uint16_t>((else_at - branch - 4) / 4);
        a.w[jump / 4] |= static_cast<std::uint16_t>((join - jump - 4) / 4);
        const gcn::LiftResult r = lift_words(a.w);
        CHECK(lifted_valid(r));
        CHECK(proof_has(r, "1 ifs on a bit every lane holds alike"));
        CHECK(loads_user_sgpr(r.spirv, 1) && loads_user_sgpr(r.spirv, 2));
    }
    // Parked again inside an s_cbranch_execz region: GCN skips the region
    // where no pixel needs it, keeping the first value; the lift would not.
    {
        Asm a = start();
        a.vop2(kWritelane, 5, 1, kC0);  // v_writelane_b32 v5, s1, 0
        a.vopc(kCmpLt, kHalf, 2);       // v_cmp_lt_f32 vcc, 0.5, v2
        a.sop1(kSAndSaveexec, 8, kVcc);
        a.branch(kExecz, a.at() + 8);
        a.vop2(kWritelane, 5, 2, kC0);  // v_writelane_b32 v5, s2, 0 (the region)
        a.sop1(kSMov64, kExec, 8);      // s_mov_b64 exec, s[8:9]
        a.vop2(kReadlane, 6, kV + 5, kC0);
        a.vop2(kVMul, 3, 6, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        CHECK(rejected_with(lift_words(a.w), "lane 0 of v5 is written inside region"));
    }
}

void descriptor_loads() {
    // s[2:3] copied to s[20:21], the table left to the page table (as the
    // translator leaves a V# it cannot show holds at every load): the
    // reference walks the page table for both loads. They are a T# and an S#,
    // which the sample binds by path.
    const auto sample = [](const std::function<void(Asm&)>& after) {
        Asm a;
        a.sop1(kSMov, 20, 2);
        a.sop1(kSMov, 21, 3);
        a.smrd(kLoadX8, 8, 20, 0);   // s_load_dwordx8 s[8:15], s[20:21], 0x0
        a.smrd(kLoadX4, 16, 20, 8);  // s_load_dwordx4 s[16:19], s[20:21], 0x8
        a.vop1(kVMov, 4, kHalf);
        a.vop1(kVMov, 5, kHalf);
        a.sopp(kWaitcnt, 0);
        a.mimg(kSampleLz, 0xf, 0, 4, 8, 16);  // image_sample_lz v[0:3], v[4:5], s[8:15], s[16:19]
        after(a);
        a.exp_mrt0(0, 1, 2, 3);
        a.endpgm();
        return a.w;
    };
    const std::vector<std::string> table = {"user_sgpr[2]"};
    const gcn::LiftResult ok = lift_words(sample([](Asm&) {}), nullptr, nullptr, table);
    CHECK(lifted_valid(ok));
    CHECK(proof_has(ok, "2 scalar loads the reference walks the page table for hold descriptors only"));
    // A word of them read as data: the reference reads it from guest memory.
    const auto as_data = [](Asm& a) {
        a.vop1(kVMov, 6, 16);         // v_mov_b32 v6, s16 (a copy is still a descriptor)
        a.vop2(kVAdd, 0, kV + 0, 6);  // v_add_f32 v0, v0, v6
    };
    CHECK(rejected_with(lift_words(sample(as_data), nullptr, nullptr, table), "data from a scalar load the reference walks the page table for"));
    // Not excluded, the copies of s[2:3] are its definitions (the translator's
    // resolve_copy_keys): the table is bound, and its words are data like any.
    gcn::TranslateResult ref;
    const gcn::LiftResult bound = lift_words(sample(as_data), nullptr, &ref);
    CHECK(lifted_valid(bound));
    CHECK(ref.buffers.size() == 1 && ref.buffers[0].path.str() == "user_sgpr[2]" && ref.buffer_at.count(8) && ref.buffer_at.count(12));
}

// One invocation of a straight-line lifted pixel shader that reads only the
// params block and storage buffers: `user` the user SGPRs, `bias` and `range`
// StageParams::cb_bias_dw and cb_stride, `buffers` each binding's dwords.
// Returns mrt0's four words, or an error.
struct ScalarRun {
    bool ok = false;
    std::string error;
    std::array<std::uint32_t, 4> mrt0{};
};
ScalarRun run_scalar_ps(const std::vector<std::uint32_t>& spv, const std::array<std::uint32_t, 16>& user, const std::array<std::uint32_t, 16>& bias,
                        const std::array<std::uint32_t, 16>& range, const std::map<std::uint32_t, std::vector<std::uint32_t>>& buffers) {
    ScalarRun run;
    using Vec = std::array<std::uint32_t, 4>;
    std::map<std::uint32_t, Vec> val;
    std::map<std::uint32_t, std::uint32_t> constants, location, binding, storage;
    struct Chain {
        std::uint32_t base;
        std::vector<std::uint32_t> idx;
    };
    std::map<std::uint32_t, Chain> chains;
    const auto fail = [&](const std::string& why) {
        run.error = why;
        return run;
    };
    bool in_body = false, stored = false;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t n = spv[i] >> 16, op = spv[i] & 0xffff;
        if (n == 0 || i + n > spv.size()) return fail("malformed module");
        const std::uint32_t* a = &spv[i + 1];
        i += n;
        switch (op) {
        case 71:  // OpDecorate
            if (a[1] == 30) location[a[0]] = a[2];
            if (a[1] == 33) binding[a[0]] = a[2];
            continue;
        case 59: storage[a[1]] = a[2]; continue;  // OpVariable
        case 43: constants[a[1]] = a[2]; continue;
        case 41: constants[a[1]] = 1; continue;
        case 42: constants[a[1]] = 0; continue;
        case 50: constants[a[1]] = a[2]; continue;  // OpSpecConstant: its default
        case 248: in_body = true; continue;
        default: break;
        }
        if (!in_body) continue;
        const auto get = [&](std::uint32_t id) -> Vec {
            if (const auto c = constants.find(id); c != constants.end()) return {c->second, 0, 0, 0};
            return val[id];
        };
        const auto g0 = [&](std::uint32_t id) { return get(id)[0]; };
        const auto set = [&](std::uint32_t id, std::uint32_t v) { val[id] = {v, 0, 0, 0}; };
        switch (op) {
        case 65: {  // OpAccessChain
            Chain c{a[2], {}};
            for (std::uint32_t k = 3; k < n - 1; ++k) c.idx.push_back(g0(a[k]));
            chains[a[1]] = c;
            break;
        }
        case 132: set(a[1], g0(a[2]) * g0(a[3])); break;   // OpIMul
        case 170: set(a[1], g0(a[2]) == g0(a[3])); break;  // OpIEqual
        case 171: set(a[1], g0(a[2]) != g0(a[3])); break;  // OpINotEqual
        case 61: {  // OpLoad
            const auto c = chains.find(a[2]);
            if (c == chains.end()) return fail("a load of something other than the params or a storage buffer");
            const std::uint32_t base = c->second.base;
            const auto& idx = c->second.idx;
            if (storage[base] == 2) {  // Uniform: the params block
                if (idx.size() != 3) return fail("a params load of another shape");
                const std::uint32_t k = idx[1] * 4 + idx[2];
                if (k >= 16) return fail("a params entry past 16");
                if (idx[0] == 2) set(a[1], user[k]);
                else if (idx[0] == 4) set(a[1], bias[k]);
                else if (idx[0] == 5) set(a[1], range[k]);
                else return fail("a params member the test does not give: " + std::to_string(idx[0]));
            } else if (storage[base] == 12) {  // StorageBuffer
                const auto b = buffers.find(binding[base]);
                if (b == buffers.end() || idx.size() != 2 || idx[0] != 0) return fail("a storage buffer the test does not give");
                if (idx[1] >= b->second.size()) return fail("a storage-buffer read past the binding (dword " + std::to_string(idx[1]) + ")");
                set(a[1], b->second[idx[1]]);
            } else {
                return fail("a load from another storage class");
            }
            break;
        }
        case 62:  // OpStore
            if (storage[a[0]] != 3 || !location.count(a[0]) || location[a[0]] != 0) return fail("a store to something other than mrt0");
            run.mrt0 = get(a[1]);
            stored = true;
            break;
        case 80: {  // OpCompositeConstruct
            Vec v{};
            for (std::uint32_t k = 2; k < n - 1 && k < 6; ++k) v[k - 2] = g0(a[k]);
            val[a[1]] = v;
            break;
        }
        case 124: val[a[1]] = get(a[2]); break;                         // OpBitcast
        case 128: set(a[1], g0(a[2]) + g0(a[3])); break;                // OpIAdd
        case 176: set(a[1], g0(a[2]) < g0(a[3])); break;                // OpULessThan
        case 194: set(a[1], g0(a[2]) >> (g0(a[3]) & 31)); break;        // OpShiftRightLogical
        case 199: set(a[1], g0(a[2]) & g0(a[3])); break;                // OpBitwiseAnd
        case 169: val[a[1]] = g0(a[2]) ? get(a[3]) : get(a[4]); break;  // OpSelect
        case 253: case 56: break;                                       // OpReturn, OpFunctionEnd
        default: return fail("unsupported opcode " + std::to_string(op));
        }
    }
    if (!stored) return fail("mrt0 is not written");
    run.ok = true;
    return run;
}

// s_buffer_load_dwordx4 at a register offset from a V# in a table: the
// reference binds the V#'s whole range (BufferBinding::ranged), and the lift
// reads dword (offset >> 2) + k of it, 0 at or past the range, as GCN's range
// check returns - here through an interpreter of the lifted module, at
// offsets inside, straddling and past the range and with the low bits set.
void ranged_loads() {
    constexpr std::uint32_t kBufferLoadX4 = 10;  // SMRD s_buffer_load_dwordx4
    const auto program = [&] {
        Asm a;
        a.smrd(kLoadX4, 8, 2, 4);                 // s_load_dwordx4 s[8:11], s[2:3], 0x4: the V#
        a.sopp(kWaitcnt, 0);
        a.smrd_reg(kBufferLoadX4, 12, 8, 1);      // s_buffer_load_dwordx4 s[12:15], s[8:11], s1
        a.sopp(kWaitcnt, 0);
        for (std::uint32_t k = 0; k < 4; ++k) a.vop1(kVMov, k, 12 + k);
        a.exp_mrt0(0, 1, 2, 3);
        a.endpgm();
        return a.w;
    };
    gcn::TranslateResult ref;
    const gcn::LiftResult l = lift_words(program(), nullptr, &ref);
    CHECK(lifted_valid(l));
    CHECK(ref.buffers.size() == 2 && ref.buffers[1].ranged && !ref.buffers[1].indexed && !ref.buffers[0].ranged &&
          ref.buffers[1].path.str() == "user_sgpr[2]->load(+16)" && ref.buffer_at.count(8) && ref.buffer_at.at(8) == 1);
    CHECK(proof_has(l, "1 at a run-time offset, each dword checked against the V#'s range"));
    if (!l.ok() || ref.buffers.size() != 2) return;
    const std::uint32_t b0 = ref.buffers[0].binding, b1 = ref.buffers[1].binding;
    std::vector<std::uint32_t> data(64);
    for (std::uint32_t k = 0; k < 64; ++k) data[k] = 0x1000u + k;
    for (const auto& [offset, range, bias] : {std::array<std::uint32_t, 3>{16, 10, 0}, {33, 10, 0}, {40, 10, 0}, {36, 10, 0}, {0, 0, 0},
                                              {8, 10, 6}, {0xfffffffcu, 10, 0}, {28, 64, 2}}) {
        std::array<std::uint32_t, 16> user{}, biases{}, ranges{};
        user[1] = offset;
        biases[1] = bias;
        ranges[1] = range;
        const ScalarRun r = run_scalar_ps(l.spirv, user, biases, ranges, {{b0, std::vector<std::uint32_t>(16, 0)}, {b1, data}});
        if (!r.ok) std::fprintf(stderr, "ranged load at %u: %s\n", offset, r.error.c_str());
        CHECK(r.ok);
        for (std::uint32_t k = 0; k < 4 && r.ok; ++k) {
            const std::uint32_t at = (offset >> 2) + k;
            const std::uint32_t want = at < range ? data[bias + at] : 0;
            if (r.mrt0[k] != want) {
                std::fprintf(stderr, "ranged load at byte %u (range %u dwords, bias %u): dword %u read %08x, expected %08x\n", offset, range, bias, k,
                             r.mrt0[k], want);
                ++g_failures;
            }
        }
    }
    // Left to the page table (the reference walks it, without a binding), the
    // words are data the lift does not have.
    CHECK(rejected_with(lift_words(program(), nullptr, nullptr, {"user_sgpr[2]->load(+16)"}),
                        "data from a scalar load the reference walks the page table for"));
}

// tbuffer_load_format_* by index from a V# in a table: the instruction's
// format with the identity select (gcn::tbuffer_dst_sel), whatever the V#'s
// DST_SEL - the lifted module reads the element and never the params' word 3
// (the interpreter refuses a load of it), and a VGPR past the format's
// components reads 0.
void typed_loads() {
    const auto program = [&](std::uint32_t op, std::uint32_t dfmt) {
        Asm a;
        a.smrd(kLoadX4, 8, 2, 4);  // s_load_dwordx4 s[8:11], s[2:3], 0x4: the V#
        a.sopp(kWaitcnt, 0);
        for (std::uint32_t k = 0; k < 4; ++k) a.vop1(kVMov, k, kC0 + 9);  // v0..v3 = 9, to see what a load leaves
        a.vop1(kVMov, 4, 1);       // v4 = s1: the element's index
        a.tbuffer_load(op, 0, 4, 8, dfmt, 7);
        a.sopp(kWaitcnt, 0);
        a.exp_mrt0(0, 1, 2, 3);
        a.endpgm();
        return a.w;
    };
    std::vector<std::uint32_t> data(64);
    for (std::uint32_t k = 0; k < 64; ++k) data[k] = 0x2000u + k;
    for (const auto& [op, dfmt, comps] : {std::array<std::uint32_t, 3>{3, 14, 4}, {3, 13, 3}, {2, 11, 2}, {0, 4, 1}, {1, 14, 4}}) {
        gcn::TranslateResult ref;
        const gcn::LiftResult l = lift_words(program(op, dfmt), nullptr, &ref);
        CHECK(lifted_valid(l));
        CHECK(ref.buffers.size() == 2 && ref.buffers[1].indexed && !ref.buffers[1].ranged);
        if (!l.ok() || ref.buffers.size() != 2) continue;
        for (const std::uint32_t index : {0u, 3u}) {
            std::array<std::uint32_t, 16> user{}, bias{}, stride{};
            user[1] = index;
            bias[1] = 2;
            stride[1] = 16;  // StageParams::cb_stride: the V#'s stride in bytes
            const ScalarRun r = run_scalar_ps(l.spirv, user, bias, stride, {{ref.buffers[0].binding, std::vector<std::uint32_t>(16, 0)}, {ref.buffers[1].binding, data}});
            if (!r.ok) std::fprintf(stderr, "typed load dfmt %u: %s\n", dfmt, r.error.c_str());
            CHECK(r.ok);
            for (std::uint32_t k = 0; k < 4 && r.ok; ++k) {
                const std::uint32_t want = k > op ? 9u : k < comps ? data[2 + index * 4 + k] : 0u;
                if (r.mrt0[k] != want) {
                    std::fprintf(stderr, "typed load op %u dfmt %u, element %u: component %u read %08x, expected %08x\n", op, dfmt, index, k, r.mrt0[k], want);
                    ++g_failures;
                }
            }
        }
    }
}

// Copies of a V#: a spill to lanes of a VGPR and back (v_writelane_b32 /
// v_readlane_b32 of a spill cell), or s_mov, define what they copy, so a
// constant buffer read through a copy of its V# is bound - where every copy
// reaching the load copies that V#. Otherwise it stays on the page table and
// its data is not lifted.
void copies() {
    constexpr std::uint32_t kBufferLoad = 8;  // SMRD s_buffer_load_dword
    // s[16:19] (the V# at dword 4 of the table s[2:3]) and s[24:27] (at dword
    // 8) loaded; `spill` parks one in lanes 4-7 of v3; s[16:19] clobbered;
    // s[20:23] from the lanes; s_buffer_load_dword s0, s[20:23], 0x2 as data.
    const auto program = [](const std::function<void(Asm&)>& spill) {
        Asm a;
        a.smrd(kLoadX4, 16, 2, 4);
        a.smrd(kLoadX4, 24, 2, 8);
        a.sopp(kWaitcnt, 0);
        spill(a);
        for (std::uint32_t k = 0; k < 4; ++k) a.sop1(kSMov, 16 + k, kC0);  // s16..s19 = 0
        for (std::uint32_t k = 0; k < 4; ++k) a.vop2(kReadlane, 20 + k, kV + 3, kC0 + 4 + k);  // v_readlane_b32 s(20+k), v3, 4+k
        a.smrd(kBufferLoad, 0, 20, 2);
        a.sopp(kWaitcnt, 0);
        a.vop1(kVMov, 0, 0);  // v_mov_b32 v0, s0
        a.exp_mrt0(0, 0, 0, 0);
        a.endpgm();
        return a.w;
    };
    const auto park = [](Asm& a, std::uint32_t s) {
        for (std::uint32_t k = 0; k < 4; ++k) a.vop2(kWritelane, 3, s + k, kC0 + 4 + k);  // v_writelane_b32 v3, s(s+k), 4+k
    };
    const auto bound_at = [](const gcn::TranslateResult& ref, std::uint32_t at, const char* path) {
        const auto it = ref.buffer_at.find(at);
        return it != ref.buffer_at.end() && it->second < ref.buffers.size() && ref.buffers[it->second].path.str() == path;
    };
    {  // straight-line: the cell holds the V# at dword 4
        gcn::TranslateResult ref;
        const gcn::LiftResult r = lift_words(program([&](Asm& a) { park(a, 16); }), nullptr, &ref);
        CHECK(lifted_valid(r));
        CHECK(bound_at(ref, 0x3c, "user_sgpr[2]->load(+16)"));  // the path counts bytes
        CHECK(ref.walks.empty());
    }
    // An if on user data whose arms park different V#s: the cell is one or
    // the other, so the load stays on the page table (the walk's order would
    // have said the else arm's).
    const auto two_arms = [&](std::uint32_t then_v, std::uint32_t else_v) {
        return program([&, then_v, else_v](Asm& a) {
            a.sopc(kSCmpEqU32, 0, kC0);  // s_cmp_eq_u32 s0, 0
            const std::size_t branch = a.w.size();
            a.sopp(kScc1, 0);  // patched below
            park(a, then_v);
            const std::size_t jump = a.w.size();
            a.sopp(kBranch, 0);
            const std::uint32_t else_at = a.at();
            park(a, else_v);
            const std::uint32_t join = a.at();
            a.w[branch] = 0xbf800000u | kScc1 << 16 | static_cast<std::uint16_t>((else_at - static_cast<std::uint32_t>(branch * 4) - 4) / 4);
            a.w[jump] = 0xbf800000u | kBranch << 16 | static_cast<std::uint16_t>((join - static_cast<std::uint32_t>(jump * 4) - 4) / 4);
        });
    };
    {
        gcn::TranslateResult ref;
        const std::vector<std::uint32_t> w = two_arms(16, 24);
        CHECK(rejected_with(lift_words(w, nullptr, &ref), "data from a scalar load the reference walks the page table for"));
        bool walked = false;
        for (const auto& [why, n] : ref.walks) walked |= why.find("registers reused") != std::string::npos;
        CHECK(walked);
    }
    {  // both arms park the same V#: bound
        gcn::TranslateResult ref;
        const std::vector<std::uint32_t> w = two_arms(16, 16);
        CHECK(lifted_valid(lift_words(w, nullptr, &ref)));
        CHECK(ref.walks.empty());
    }
    {  // user data: the V# s[0:3] parked, s0 clobbered, back in s[8:11]
        Asm a;
        for (std::uint32_t k = 0; k < 4; ++k) a.vop2(kWritelane, 3, k, kC0 + k);
        a.sop1(kSMov, 0, kC0);
        for (std::uint32_t k = 0; k < 4; ++k) a.vop2(kReadlane, 8 + k, kV + 3, kC0 + k);
        a.smrd(kBufferLoad, 4, 8, 1);  // s_buffer_load_dword s4, s[8:11], 0x1
        a.sopp(kWaitcnt, 0);
        a.vop1(kVMov, 0, 4);
        a.exp_mrt0(0, 0, 0, 0);
        a.endpgm();
        gcn::TranslateResult ref;
        CHECK(lifted_valid(lift_words(a.w, nullptr, &ref)));
        CHECK(bound_at(ref, 0x24, "user_sgpr[0]"));
        // ... but not when the clobber reaches the load instead of the copy.
        Asm b;
        b.sop1(kSMov, 0, kC0);
        b.smrd(kBufferLoad, 4, 0, 1);  // s_buffer_load_dword s4, s[0:3], 0x1
        b.sopp(kWaitcnt, 0);
        b.vop1(kVMov, 0, 4);
        b.exp_mrt0(0, 0, 0, 0);
        b.endpgm();
        gcn::TranslateResult ref_b;
        lift_words(b.w, nullptr, &ref_b);
        CHECK(!bound_at(ref_b, 4, "user_sgpr[0]"));
    }
}

void cselect() {
    const auto sel = [](const std::function<void(Asm&)>& set_scc) {
        Asm a = start();
        set_scc(a);
        a.sop2(kSCselect, 4, kOne, kTwo);  // s_cselect_b32 s4, 1.0, 2.0
        a.vop2(kVMul, 3, 4, 2);            // v_mul_f32 v3, s4, v2
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        return a.w;
    };
    CHECK(lifted_valid(lift_words(sel([](Asm& a) { a.sopc(kSCmpEqU32, 0, kC0); }))));
    // SCC as "some lane of the mask is set": the other pixels decide it.
    CHECK(rejected_with(lift_words(sel([](Asm& a) {
                            a.vopc(kCmpLt, kHalf, 2);
                            a.sop2(kSAnd64, 6, kVcc, kExec);  // s_and_b64 s[6:7], vcc, exec
                        })),
                        "s_cselect_b32 on an SCC that is not a scalar condition"));
    // After an s_cbranch_execz region that writes s5: s_cselect_b32 redefines
    // s5 (the region's write is dead), or reads it (rejected).
    const auto after_region = [](std::uint32_t src0) {
        Asm a = start();
        a.vopc(kCmpLt, kHalf, 2);
        a.sop1(kSAndSaveexec, 8, kVcc);
        a.branch(kExecz, a.at() + 8);
        a.sop1(kSMov, 5, kC0);  // s_mov_b32 s5, 0 (the region)
        a.sop1(kSMov64, kExec, 8);
        a.sopc(kSCmpEqU32, 0, kC0);
        a.sop2(kSCselect, 5, src0, kTwo);  // s_cselect_b32 s5, src0, 2.0
        a.vop2(kVMul, 3, 5, 2);
        a.exp_mrt0(3, 3, 3, 3);
        a.endpgm();
        return a.w;
    };
    CHECK(lifted_valid(lift_words(after_region(kOne))));
    CHECK(rejected_with(lift_words(after_region(5)), "s5 is written inside region"));
}

void lshl_b64() {
    // v[0:1] = 0 << 0: folded.
    {
        Asm a;
        a.vop3(kLshlB64, 0, kC0, kC0);  // v_lshl_b64 v[0:1], 0, 0
        a.exp_mrt0(0, 1, 0, 1);
        a.endpgm();
        const gcn::LiftResult r = lift_words(a.w);
        CHECK(lifted_valid(r));
        CHECK(!has_u64_shift(r.spirv));
    }
    // A per-pixel pair shifted at run time, as the translator does it.
    {
        Asm a = start();
        a.vop1(kCvtU32F32, 3, kV + 2);           // v_cvt_u32_f32 v3, v2
        a.vop1(kVMov, 4, kC0 + 7);               // v_mov_b32 v4, 7
        a.vop3(kLshlB64, 6, kV + 3, kC0 + 36);   // v_lshl_b64 v[6:7], v[3:4], 36
        a.vop1(kCvtF32U32, 8, kV + 6);
        a.vop1(kCvtF32U32, 9, kV + 7);
        a.exp_mrt0(8, 9, 8, 9);
        a.endpgm();
        const gcn::LiftResult r = lift_words(a.w);
        CHECK(lifted_valid(r));
        CHECK(has_u64_shift(r.spirv));
    }
    // VCC is a lane mask, not a number.
    {
        Asm a = start();
        a.vopc(kCmpLt, kHalf, 2);
        a.vop3(kLshlB64, 0, kVcc, kC0 + 1);  // v_lshl_b64 v[0:1], vcc, 1
        a.exp_mrt0(0, 1, 0, 1);
        a.endpgm();
        CHECK(rejected_with(lift_words(a.w), "unsupported 64-bit operand"));
    }
    // Its first operand is a pair: s1, written in a region, is read after it.
    {
        Asm a = start();
        a.vopc(kCmpLt, kHalf, 2);
        a.sop1(kSAndSaveexec, 8, kVcc);
        a.branch(kExecz, a.at() + 8);
        a.sop1(kSMov, 1, kC0);  // s_mov_b32 s1, 0 (the region)
        a.sop1(kSMov64, kExec, 8);
        a.vop3(kLshlB64, 4, 0, kC0 + 1);  // v_lshl_b64 v[4:5], s[0:1], 1
        a.vop1(kCvtF32U32, 6, kV + 5);
        a.exp_mrt0(6, 6, 6, 6);
        a.endpgm();
        CHECK(rejected_with(lift_words(a.w), "s1 is written inside region"));
    }
}

void run() {
    spills();
    descriptor_loads();
    ranged_loads();
    typed_loads();
    copies();
    cselect();
    lshl_b64();
}

}  // namespace lane_ops

// ---- vertex shaders on vertex input, hand-encoded ---------------------------------------
// The elements load at s_swappc_b64 (the fetch shader's call). With
// vertex_formats_from_params the lift converts each element by its descriptor
// in the params block - or the relink's specialization constant - as the
// translator does; run through a straight-line interpreter of one vertex, it
// must compute what the lift with the same formats as constants computes
// (TranslateOptions::vertex_input's own conversions), for every kind the
// loader serves, every component count and DST_SEL's ones and zeros past them.
namespace vertex {

std::uint32_t sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc) { return 0xbe800000u | sdst << 16 | op << 8 | ssrc; }
// exp <tgt> v0..v3, all four components, done where asked.
void export_vec(std::vector<std::uint32_t>& w, std::uint32_t tgt, std::uint32_t v, bool done = false) {
    w.push_back(0xf8000000u | (done ? 1u << 11 : 0u) | tgt << 4 | 0xfu);
    w.push_back((v + 3) << 24 | (v + 2) << 16 | (v + 1) << 8 | v);
}
constexpr std::uint32_t kEndpgm = 0xbf810000u;

// s_swappc_b64 s[0:1], s[2:3]; the position from v4-v7, param 0 from v8-v11,
// then `extra` exports; two elements at v4 and v8.
std::vector<std::uint32_t> program(const std::function<void(std::vector<std::uint32_t>&)>& extra = {}) {
    std::vector<std::uint32_t> w = {sop1(33, 0, 2)};
    export_vec(w, 12, 4);
    export_vec(w, 32, 8);
    if (extra) extra(w);
    w.push_back(kEndpgm);
    return w;
}

gcn::TranslateOptions options(std::uint32_t w3_a, std::uint32_t w3_b, bool from_params) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.rsrc1 = 3u << 24;  // VGPR_COMP_CNT 3
    o.descriptor_set = 0;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    o.link_outputs = true;
    for (std::uint8_t r = 0; r < 32; ++r) o.output_links.push_back(r);
    o.vertex_input = {{0, 4, 4, from_params ? 0u : w3_a}, {1, 8, 4, from_params ? 0u : w3_b}};
    o.vertex_formats_from_params = from_params;
    return o;
}

struct Lifted {
    gcn::TranslateResult ref;
    gcn::LiftResult lift;
};
Lifted lift_vs(const std::vector<std::uint32_t>& words, const gcn::TranslateOptions& o) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    Lifted r;
    r.ref = gcn::translate(p, o);
    if (!r.ref.ok()) std::fprintf(stderr, "reference failed: %s\n", r.ref.errors[0].c_str());
    r.lift = gcn::lift_vertex_shader(p, o, r.ref);
    return r;
}

// One vertex through the lifted module: `inputs` at their locations, the
// params' vertex_formats, the specialization constants by SpecId. Returns
// the position, the params by location and the clip distances.
struct VertexRun {
    bool ok = false;
    std::string error;
    std::array<std::uint32_t, 4> position{};
    std::map<std::uint32_t, std::array<std::uint32_t, 4>> params;
    std::map<std::uint32_t, std::uint32_t> clip;
};
VertexRun run_vertex(const std::vector<std::uint32_t>& spv, const std::map<std::uint32_t, std::array<std::uint32_t, 4>>& inputs,
                     const std::array<std::uint32_t, 16>& formats, const std::map<std::uint32_t, std::uint32_t>& spec = {}) {
    VertexRun run;
    using Vec = std::array<std::uint32_t, 4>;
    std::map<std::uint32_t, Vec> val;
    std::map<std::uint32_t, std::uint32_t> constants, builtin, location, storage, spec_id, spec_default;
    struct Chain {
        std::uint32_t base;
        std::vector<std::uint32_t> idx;
    };
    std::map<std::uint32_t, Chain> chains;
    std::uint32_t ubo = 0;
    const auto fail = [&](const std::string& why) {
        run.error = why;
        return run;
    };
    bool in_body = false;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t n = spv[i] >> 16, op = spv[i] & 0xffff;
        if (n == 0 || i + n > spv.size()) return fail("malformed module");
        const std::uint32_t* a = &spv[i + 1];
        i += n;
        switch (op) {
        case 71:  // OpDecorate
            if (a[1] == 11) builtin[a[0]] = a[2];
            if (a[1] == 30) location[a[0]] = a[2];
            if (a[1] == 1) spec_id[a[0]] = a[2];
            continue;
        case 59:  // OpVariable
            storage[a[1]] = a[2];
            if (a[2] == 2) ubo = a[1];  // Uniform: the params block
            continue;
        case 43: constants[a[1]] = a[2]; continue;      // OpConstant (low word)
        case 41: constants[a[1]] = 1; continue;         // OpConstantTrue
        case 42: constants[a[1]] = 0; continue;         // OpConstantFalse
        case 50: spec_default[a[1]] = a[2]; continue;   // OpSpecConstant
        case 248: in_body = true; continue;             // OpLabel
        default: break;
        }
        if (!in_body) continue;
        const auto get = [&](std::uint32_t id) -> Vec {
            if (const auto c = constants.find(id); c != constants.end()) return {c->second, 0, 0, 0};
            if (const auto s = spec_default.find(id); s != spec_default.end()) {
                const auto given = spec_id.count(id) ? spec.find(spec_id.at(id)) : spec.end();
                return {given != spec.end() ? given->second : s->second, 0, 0, 0};
            }
            return val[id];
        };
        const auto g0 = [&](std::uint32_t id) { return get(id)[0]; };
        const auto f = [](std::uint32_t b) { return std::bit_cast<float>(b); };
        const auto u = [](float x) { return std::bit_cast<std::uint32_t>(x); };
        const auto set = [&](std::uint32_t id, std::uint32_t v) { val[id] = {v, 0, 0, 0}; };
        switch (op) {
        case 65: {  // OpAccessChain
            Chain c{a[2], {}};
            for (std::uint32_t k = 3; k < n - 1; ++k) c.idx.push_back(g0(a[k]));
            chains[a[1]] = c;
            break;
        }
        case 61: {  // OpLoad
            const std::uint32_t ptr = a[2];
            if (const auto c = chains.find(ptr); c != chains.end()) {
                if (c->second.base != ubo || c->second.idx.size() != 3 || c->second.idx[0] != 7) return fail("a params load outside vertex_formats");
                set(a[1], formats.at(c->second.idx[1] * 4 + c->second.idx[2]));
            } else if (storage[ptr] == 1 && location.count(ptr) && !builtin.count(ptr)) {
                const auto in = inputs.find(location[ptr]);
                if (in == inputs.end()) return fail("an input location the test does not give");
                val[a[1]] = in->second;
            } else {
                return fail("a load of something other than a vertex input or the params");
            }
            break;
        }
        case 62: {  // OpStore
            const std::uint32_t ptr = a[0];
            if (const auto c = chains.find(ptr); c != chains.end()) {
                if (!builtin.count(c->second.base) || builtin[c->second.base] != 3 || c->second.idx.size() != 1) return fail("a chained store outside ClipDistance");
                run.clip[c->second.idx[0]] = g0(a[1]);
            } else if (builtin.count(ptr) && builtin[ptr] == 0) {
                run.position = get(a[1]);
            } else if (storage[ptr] == 3 && location.count(ptr)) {
                run.params[location[ptr]] = get(a[1]);
            } else {
                return fail("a store to something other than the position, a param or a clip distance");
            }
            break;
        }
        case 81: set(a[1], get(a[2])[a[3]]); break;  // OpCompositeExtract
        case 80: {                                   // OpCompositeConstruct
            Vec v{};
            for (std::uint32_t k = 2; k < n - 1 && k < 6; ++k) v[k - 2] = g0(a[k]);
            val[a[1]] = v;
            break;
        }
        case 124: val[a[1]] = get(a[2]); break;                           // OpBitcast
        case 112: set(a[1], u(static_cast<float>(g0(a[2])))); break;      // OpConvertUToF
        case 136: set(a[1], u(f(g0(a[2])) / f(g0(a[3])))); break;         // OpFDiv
        case 133: set(a[1], u(f(g0(a[2])) * f(g0(a[3])))); break;         // OpFMul
        case 129: set(a[1], u(f(g0(a[2])) + f(g0(a[3])))); break;         // OpFAdd
        case 170: set(a[1], g0(a[2]) == g0(a[3])); break;                 // OpIEqual
        case 171: set(a[1], g0(a[2]) != g0(a[3])); break;                 // OpINotEqual
        case 176: set(a[1], g0(a[2]) < g0(a[3])); break;                  // OpULessThan
        case 194: set(a[1], g0(a[2]) >> (g0(a[3]) & 31)); break;          // OpShiftRightLogical
        case 199: set(a[1], g0(a[2]) & g0(a[3])); break;                  // OpBitwiseAnd
        case 166: set(a[1], g0(a[2]) | g0(a[3])); break;                  // OpLogicalOr
        case 169: val[a[1]] = g0(a[2]) ? get(a[3]) : get(a[4]); break;    // OpSelect
        case 202: {                                                       // OpBitFieldSExtract
            const std::uint32_t off = g0(a[3]), count = g0(a[4]);
            if (count == 0 || off + count > 32) return fail("a bit field outside the word");
            const std::int32_t x = static_cast<std::int32_t>(g0(a[2]) << (32 - off - count)) >> (32 - count);
            set(a[1], static_cast<std::uint32_t>(x));
            break;
        }
        case 203: {  // OpBitFieldUExtract (an element's packed components, VertexElement::packed)
            const std::uint32_t off = g0(a[3]), count = g0(a[4]);
            if (count == 0 || off + count > 32) return fail("a bit field outside the word");
            set(a[1], count == 32 ? g0(a[2]) : (g0(a[2]) >> off) & ((1u << count) - 1));
            break;
        }
        case 253: case 56: break;  // OpReturn, OpFunctionEnd
        default: return fail("unsupported opcode " + std::to_string(op));
        }
    }
    run.ok = true;
    return run;
}

void formats() {
    // Word 3 of an element: DST_SEL x,y,z,w in place (0xfac), data format
    // [18:15], number format [14:12]; past a format's components DST_SEL 1 or 0.
    const auto w3 = [](std::uint32_t dfmt, std::uint32_t nfmt, std::uint32_t sel) { return dfmt << 15 | nfmt << 12 | sel; };
    std::vector<std::uint32_t> words3;
    for (const auto [dfmt, nfmt] : {std::pair{14u, 7u}, {13u, 7u}, {11u, 7u}, {4u, 7u},  // 32-bit floats: the raw bits
                                    {14u, 4u}, {11u, 4u}, {4u, 5u}, {13u, 5u},            // 32-bit integers
                                    {14u, 0u}, {4u, 1u},                                  // 32-bit unorm, snorm: the raw bits too
                                    {10u, 0u}, {3u, 0u}, {1u, 0u},                        // unorm8
                                    {12u, 5u}, {5u, 5u}, {2u, 5u}}) {                     // sint16
        int count = dfmt == 14 || dfmt == 10 || dfmt == 12 ? 4 : dfmt == 13 ? 3 : dfmt == 11 || dfmt == 3 || dfmt == 5 ? 2 : 1;
        std::uint32_t in_place = 0;
        for (int k = 0; k < count; ++k) in_place |= (4u + static_cast<std::uint32_t>(k)) << (3 * k);
        // Past the components: all zero, all one, and alternating.
        std::uint32_t ones = in_place, mixed = in_place;
        for (int k = count; k < 4; ++k) {
            ones |= 1u << (3 * k);
            if (k & 1) mixed |= 1u << (3 * k);
        }
        for (const std::uint32_t sel : {in_place, ones, mixed}) {
            words3.push_back(w3(dfmt, nfmt, sel));
            CHECK(gcn::vertex_format_descriptor(words3.back()) != 0);
        }
    }
    const std::vector<std::array<std::uint32_t, 4>> raws = {
        {0x3f800000u, 0xbf000000u, 0x7f7fffffu, 0x00000001u},  // floats, a denormal
        {0, 255, 128, 1},                                      // unorm8 edges
        {0xffffu, 0x8000u, 0x7fffu, 0x1234u},                  // sint16 edges
        {0x12345678u, 0xfffffffeu, 0x80000000u, 0x7fc00000u},  // NaN bits, integers
    };
    const std::vector<std::uint32_t> words = program();
    std::size_t compared = 0;
    for (std::size_t i = 0; i < words3.size(); ++i) {
        const std::uint32_t a = words3[i], b = words3[(i + 7) % words3.size()];
        const Lifted constant = lift_vs(words, options(a, b, false));
        const Lifted params = lift_vs(words, options(a, b, true));
        CHECK(lifted(constant.lift) && lifted(params.lift));
        if (!constant.lift.ok() || !params.lift.ok()) continue;
        CHECK(proof_says(params.lift, "2 vertex elements converted by their descriptor"));
        std::array<std::uint32_t, 16> formats{};
        formats[0] = gcn::vertex_format_descriptor(a);
        formats[1] = gcn::vertex_format_descriptor(b);
        for (const auto& ra : raws) {
            for (const auto& rb : raws) {
                const std::map<std::uint32_t, std::array<std::uint32_t, 4>> inputs = {{0, ra}, {1, rb}};
                const VertexRun want = run_vertex(constant.lift.spirv, inputs, {});
                const VertexRun got = run_vertex(params.lift.spirv, inputs, formats);
                // The relink's specialization constants win over the params.
                const std::array<std::uint32_t, 16> garbage = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
                const VertexRun spec = run_vertex(params.lift.spirv, inputs, garbage,
                                                  {{gcn::kVertexFormatSpecId + 0, formats[0]}, {gcn::kVertexFormatSpecId + 1, formats[1]}});
                for (const VertexRun* r : {&want, &got, &spec}) {
                    if (!r->ok) std::fprintf(stderr, "vertex run failed: %s\n", r->error.c_str());
                    CHECK(r->ok);
                }
                if (!want.ok || !got.ok || !spec.ok) return;
                const bool same = got.position == want.position && got.params == want.params && spec.position == want.position && spec.params == want.params;
                if (!same) {
                    std::fprintf(stderr, "formats %08x/%08x: the lift from params differs from the constant-format lift\n", a, b);
                    ++g_failures;
                    return;
                }
                ++compared;
            }
        }
    }
    CHECK(compared == words3.size() * raws.size() * raws.size());
}

void clip_distances() {
    // pos1 holds clip distances 0 and 1 (PA_CL_VS_OUT_CNTL 0x400003, as 4,762
    // of the game's vertex shaders): ClipDistance, or with the host's clip
    // varying the distances at the clip location for the pixel stage (a
    // disabled one reads 1), and 1s there when nothing exports them.
    const std::vector<std::uint32_t> words = program([](std::vector<std::uint32_t>& w) { export_vec(w, 13, 4); });
    const std::uint32_t fmt = 14u << 15 | 7u << 12 | 0xfac;  // 32_32_32_32 float
    const std::map<std::uint32_t, std::array<std::uint32_t, 4>> inputs = {{0, {0x3f800000u, 0xc0000000u, 0x40400000u, 0x40800000u}},
                                                                          {1, {1, 2, 3, 4}}};
    std::array<std::uint32_t, 16> formats{};
    formats[0] = formats[1] = gcn::vertex_format_descriptor(fmt);
    gcn::TranslateOptions o = options(fmt, fmt, true);
    o.vs_out_cntl = 0x400003;
    const Lifted plain = lift_vs(words, o);
    CHECK(lifted(plain.lift));
    CHECK(plain.ref.vs_clip_count == 2);
    if (plain.lift.ok()) {
        const VertexRun r = run_vertex(plain.lift.spirv, inputs, formats);
        if (!r.ok) std::fprintf(stderr, "vertex run failed: %s\n", r.error.c_str());
        CHECK(r.ok && r.clip.size() == 2 && r.clip.at(0) == 0x3f800000u && r.clip.at(1) == 0xc0000000u);
    }
    o.vs_out_cntl = 0x400003 | gcn::TranslateOptions::kVsOutCntlClipVarying | 5u << gcn::TranslateOptions::kVsOutCntlClipLocationShift;
    const Lifted varying = lift_vs(words, o);
    CHECK(lifted(varying.lift));
    CHECK(varying.ref.vs_params == std::vector<std::uint32_t>({0, 5}));
    if (varying.lift.ok()) {
        const VertexRun r = run_vertex(varying.lift.spirv, inputs, formats);
        CHECK(r.ok && r.clip.empty() && r.params.count(5) &&
              r.params.at(5) == (std::array<std::uint32_t, 4>{0x3f800000u, 0xc0000000u, 0x3f800000u, 0x3f800000u}));
    }
    const Lifted none = lift_vs(program(), o);  // nothing exported at pos1
    CHECK(lifted(none.lift));
    if (none.lift.ok()) {
        const VertexRun r = run_vertex(none.lift.spirv, inputs, formats);
        CHECK(r.ok && r.params.count(5) && r.params.at(5) == (std::array<std::uint32_t, 4>{0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}));
    }
    // A pos1 that PA_CL_VS_OUT_CNTL does not enable is the translator's error.
    o.vs_out_cntl = 0;
    const Lifted stray = lift_vs(words, o);
    CHECK(!stray.ref.ok() && rejected_with(stray.lift, "the reference translation failed"));
}

// The pixel stage's half (TranslateOptions::ps_clip_discard): the translator's
// prologue - discard where a distance at the clip location is negative -
// before the program, and no early fragment tests.
void clip_discard() {
    std::vector<std::uint32_t> w;
    push(w, {vintrp(0, 2, 0, 0, 0), vintrp(1, 2, 0, 0, 1)});  // v2 = attr0.x
    export_mrt0(w, 2, 2, 2, 2);
    w.push_back(kEndpgm);
    gcn::TranslateOptions o = synthetic_options();
    o.ps_clip_discard = 2u | 3u << 8;
    o.early_fragment_tests = true;
    const ::Lifted l = lift_words(w, o);
    CHECK(lifted(l.lift));
    CHECK(proof_says(l.lift, "clip distances (the vertex stage's, at location 3) are negative is discarded at entry"));
    CHECK(count_op(l.lift.spirv, 252) == 1);  // OpKill
    CHECK(count_op(l.lift.spirv, 184) == 2);  // OpFOrdLessThan, distances 0 and 1
    bool clip_input = false, early = false;
    for (std::size_t i = 5; i < l.lift.spirv.size();) {
        const std::uint32_t op = l.lift.spirv[i] & 0xffff, len = l.lift.spirv[i] >> 16;
        if (op == 71 && len == 4 && l.lift.spirv[i + 2] == 30 && l.lift.spirv[i + 3] == 3) clip_input = true;  // Location 3
        if (op == 16 && len >= 3 && l.lift.spirv[i + 2] == 9) early = true;                                   // EarlyFragmentTests
        i += len;
    }
    CHECK(clip_input && !early);
}

void run() {
    formats();
    clip_distances();
    clip_discard();
}

}  // namespace vertex

// ---- buffer stores in a vertex shader -------------------------------------------------
// The particle copies (VS_VFX_StandardParticle, VS_VFX_SparkParticle) store
// with buffer_store_dword* through a V# from a table, at the V#'s base plus an
// SGPR offset plus the instruction's: the reference writes the V#'s binding
// (BufferBinding::written, bound whole: ranged), and the lift writes dword
// (soffset + offset12) >> 2 + k where it is inside the V#'s range and the
// invocation's EXEC bit is set - checked here through an interpreter of the
// lifted module that keeps the storage buffers' words, at offsets inside,
// straddling and past the range, with a bias, and with a constant soffset.
namespace vertex_stores {

struct Asm {
    std::vector<std::uint32_t> w;
    void smrd(std::uint32_t op, std::uint32_t sdst, std::uint32_t sbase, std::uint32_t offset_dw) {
        w.push_back(0xc0000000u | op << 22 | sdst << 15 | (sbase / 2) << 9 | 1u << 8 | offset_dw);
    }
    void vmov(std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | 1u << 9 | src0); }
    // buffer_store_dword* (op 28 x1, 29 x2, 30 x4, 31 x3) vdata, s[srsrc..+3], soffset (a scalar code), offset12;
    // `offen`: the VGPR vaddr adds its value (an address form the reference does not bind).
    void store(std::uint32_t op, std::uint32_t vdata, std::uint32_t srsrc, std::uint32_t soffset, std::uint32_t offset12, bool offen = false,
               std::uint32_t vaddr = 0) {
        w.push_back(0xe0000000u | op << 18 | (offen ? 1u << 12 : 0u) | offset12);
        w.push_back(vaddr | vdata << 8 | (srsrc / 4) << 16 | soffset << 24);
    }
    void waitcnt() { w.push_back(0xbf8c0000u); }
    void endpgm() { w.push_back(0xbf810000u); }
};

gcn::TranslateOptions options(std::vector<std::string> exclude = {}, gcn::Stage stage = gcn::Stage::Vertex) {
    gcn::TranslateOptions o;
    o.stage = stage;
    o.rsrc2 = 8u << 1;  // eight user SGPRs
    if (stage == gcn::Stage::Pixel) o.ps_input_ena = 2;
    o.descriptor_set = stage == gcn::Stage::Pixel ? 1 : 0;
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    o.cb_ssbo_exclude = std::move(exclude);
    return o;
}

// The V# at dword 4 of the table s[2:3] in s[8:11]; v1..v3 from s5..s7; stores
// at s4 + 0 (one dword), s4 + 8 (two) and at the constant 0 + 4 (one).
std::vector<std::uint32_t> program(bool offen = false) {
    Asm a;
    a.smrd(2, 8, 2, 4);  // s_load_dwordx4 s[8:11], s[2:3], 0x4
    a.waitcnt();
    a.vmov(1, 5);
    a.vmov(2, 6);
    a.vmov(3, 7);
    a.store(28, 1, 8, 4, 0, offen, 1);
    a.store(29, 2, 8, 4, 8);
    a.store(28, 1, 8, 128, 4);  // soffset: the inline constant 0
    a.endpgm();
    return a.w;
}

struct Lifted {
    gcn::TranslateResult ref;
    gcn::LiftResult lift;
};
Lifted lift(const std::vector<std::uint32_t>& words, const gcn::TranslateOptions& o) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    Lifted r;
    r.ref = gcn::translate(p, o);
    if (!r.ref.ok()) std::fprintf(stderr, "reference failed: %s\n", r.ref.errors[0].c_str());
    r.lift = o.stage == gcn::Stage::Vertex ? gcn::lift_vertex_shader(p, o, r.ref) : gcn::lift_pixel_shader(p, o, r.ref);
    return r;
}

// One invocation of the lifted module: the params' user SGPRs, biases and
// ranges, and the storage buffers by binding, which its stores change.
struct Run {
    bool ok = false;
    std::string error;
    std::map<std::uint32_t, std::vector<std::uint32_t>> buffers;
};
Run run(const std::vector<std::uint32_t>& spv, const std::array<std::uint32_t, 16>& user, const std::array<std::uint32_t, 16>& bias,
        const std::array<std::uint32_t, 16>& range, std::map<std::uint32_t, std::vector<std::uint32_t>> buffers) {
    Run r;
    r.buffers = std::move(buffers);
    std::map<std::uint32_t, std::uint32_t> val, constants, binding, storage;
    struct Chain {
        std::uint32_t base;
        std::vector<std::uint32_t> idx;
    };
    std::map<std::uint32_t, Chain> chains;
    const auto fail = [&](const std::string& why) {
        r.error = why;
        return r;
    };
    bool in_body = false;
    std::uint32_t skip_to = 0;  // a label: the instructions before it are not run (a then-block whose condition is false)
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t n = spv[i] >> 16, op = spv[i] & 0xffff;
        if (n == 0 || i + n > spv.size()) return fail("malformed module");
        const std::uint32_t* a = &spv[i + 1];
        i += n;
        if (!in_body) {
            if (op == 71 && a[1] == 33) binding[a[0]] = a[2];   // OpDecorate Binding
            if (op == 59) storage[a[1]] = a[2];                  // OpVariable
            if (op == 43) constants[a[1]] = a[2];                // OpConstant
            if (op == 41) constants[a[1]] = 1;                   // OpConstantTrue
            if (op == 42) constants[a[1]] = 0;                   // OpConstantFalse
            if (op == 50) constants[a[1]] = a[2];                // OpSpecConstant: its default
            if (op == 248) in_body = true;                       // OpLabel
            continue;
        }
        if (skip_to) {
            if (op == 248 && a[0] == skip_to) skip_to = 0;
            continue;
        }
        const auto g = [&](std::uint32_t id) {
            const auto c = constants.find(id);
            return c != constants.end() ? c->second : val[id];
        };
        const auto word_of = [&](const Chain& c, bool for_store) -> std::uint32_t* {
            if (storage[c.base] != 12) return nullptr;
            auto b = r.buffers.find(binding[c.base]);
            if (b == r.buffers.end() || c.idx.size() != 2 || c.idx[0] != 0 || c.idx[1] >= b->second.size()) return nullptr;
            (void)for_store;
            return &b->second[c.idx[1]];
        };
        switch (op) {
        case 65: {  // OpAccessChain
            Chain c{a[2], {}};
            for (std::uint32_t k = 3; k < n - 1; ++k) c.idx.push_back(g(a[k]));
            chains[a[1]] = c;
            break;
        }
        case 61: {  // OpLoad
            const auto c = chains.find(a[2]);
            if (c == chains.end()) return fail("a load of something other than the params or a storage buffer");
            if (storage[c->second.base] == 2) {
                const auto& idx = c->second.idx;
                if (idx.size() != 3) return fail("a params load of another shape");
                const std::uint32_t k = idx[1] * 4 + idx[2];
                if (k >= 16) return fail("a params entry past 16");
                if (idx[0] == 2) val[a[1]] = user[k];
                else if (idx[0] == 4) val[a[1]] = bias[k];
                else if (idx[0] == 5) val[a[1]] = range[k];
                else return fail("a params member the test does not give: " + std::to_string(idx[0]));
            } else if (std::uint32_t* w = word_of(c->second, false)) {
                val[a[1]] = *w;
            } else {
                return fail("a load the test does not give");
            }
            break;
        }
        case 62: {  // OpStore
            const auto c = chains.find(a[0]);
            std::uint32_t* w = c == chains.end() ? nullptr : word_of(c->second, true);
            if (!w) return fail("a store outside the storage buffers the test gives");
            *w = g(a[1]);
            break;
        }
        case 124: val[a[1]] = g(a[2]); break;                          // OpBitcast
        case 128: val[a[1]] = g(a[2]) + g(a[3]); break;                // OpIAdd
        case 176: val[a[1]] = g(a[2]) < g(a[3]); break;                // OpULessThan
        case 167: val[a[1]] = g(a[2]) && g(a[3]); break;               // OpLogicalAnd
        case 194: val[a[1]] = g(a[2]) >> (g(a[3]) & 31); break;        // OpShiftRightLogical
        case 196: val[a[1]] = g(a[2]) << (g(a[3]) & 31); break;        // OpShiftLeftLogical
        case 199: val[a[1]] = g(a[2]) & g(a[3]); break;                // OpBitwiseAnd
        case 169: val[a[1]] = g(a[2]) ? g(a[3]) : g(a[4]); break;      // OpSelect
        case 247: break;                                               // OpSelectionMerge
        case 250: if (!g(a[0])) skip_to = a[2]; break;                 // OpBranchConditional: the false label is the merge
        case 249: case 248: case 253: case 56: break;                  // OpBranch, OpLabel, OpReturn, OpFunctionEnd
        default: return fail("unsupported opcode " + std::to_string(op));
        }
    }
    r.ok = true;
    return r;
}

void cases() {
    const Lifted l = lift(program(), options());
    CHECK(lane_ops::lifted_valid(l.lift));
    CHECK(lane_ops::proof_has(l.lift, "3 buffer stores, every one through a binding the reference writes"));
    const std::string path = "user_sgpr[2]->load(+16)";
    int store_binding = -1;
    for (std::size_t k = 0; k < l.ref.buffers.size(); ++k) {
        if (l.ref.buffers[k].path.str() == path) store_binding = static_cast<int>(k);
    }
    CHECK(store_binding >= 0 && l.ref.buffers[static_cast<std::size_t>(store_binding)].written &&
          l.ref.buffers[static_cast<std::size_t>(store_binding)].ranged && !l.ref.buffers[static_cast<std::size_t>(store_binding)].indexed);
    CHECK(l.ref.walks.empty());
    if (!l.lift.ok() || store_binding < 0) return;
    const std::uint32_t b_table = l.ref.buffers[0].binding == l.ref.buffers[static_cast<std::size_t>(store_binding)].binding ? l.ref.buffers[1].binding
                                                                                                                           : l.ref.buffers[0].binding;
    const std::uint32_t b_out = l.ref.buffers[static_cast<std::size_t>(store_binding)].binding;
    for (const auto& [offset, range_dw, bias_dw] : {std::array<std::uint32_t, 3>{0, 10, 0}, {32, 10, 0}, {28, 10, 0}, {4, 3, 0}, {8, 16, 6},
                                                    {0xfffffffcu, 10, 0}, {2, 10, 0}, {0, 0, 0}}) {
        std::array<std::uint32_t, 16> user{}, bias{}, range{};
        user[4] = offset;
        user[5] = 0xa0a0a0a0u;
        user[6] = 0xb1b1b1b1u;
        user[7] = 0xc2c2c2c2u;
        bias[static_cast<std::size_t>(store_binding)] = bias_dw;
        range[static_cast<std::size_t>(store_binding)] = range_dw;
        std::vector<std::uint32_t> out(32, 0x55555555u);
        const Run r = run(l.lift.spirv, user, bias, range, {{b_table, std::vector<std::uint32_t>(16, 0)}, {b_out, out}});
        if (!r.ok) std::fprintf(stderr, "vertex stores at %u: %s\n", offset, r.error.c_str());
        CHECK(r.ok);
        if (!r.ok) continue;
        // GCN: dword (soffset + offset12) >> 2 + k of the V#'s range, written only inside it.
        std::vector<std::uint32_t> want = out;
        const auto put = [&](std::uint32_t byte, std::uint32_t v) {
            const std::uint32_t at = byte >> 2;
            if (at < range_dw) want[bias_dw + at] = v;
        };
        put(offset + 0, user[5]);
        put(offset + 8, user[6]);
        put(offset + 12, user[7]);
        put(0 + 4, user[5]);
        const std::vector<std::uint32_t>& got = r.buffers.at(b_out);
        if (got != want) {
            for (std::size_t k = 0; k < want.size(); ++k) {
                if (got[k] != want[k]) std::fprintf(stderr, "vertex stores at byte %u (range %u, bias %u): dword %zu holds %08x, expected %08x\n",
                                                    offset, range_dw, bias_dw, k, got[k], want[k]);
            }
            ++g_failures;
        }
    }
    // A store with a VGPR offset keeps the page-table walk (its address is
    // per lane), and the lift does not take it; nor one left to the page table,
    // nor one in a pixel shader.
    CHECK(rejected_with(lift(program(true), options()).lift, "is not a store the reference makes through a binding"));
    CHECK(rejected_with(lift(program(), options({path})).lift, "is not a store the reference makes through a binding"));
    {
        std::vector<std::uint32_t> w = program();
        w.pop_back();
        w.push_back(0xf8000000u | 1u << 12 | 1u << 11 | 0xfu);  // exp mrt0 v1, v1, v1, v1 done vm
        w.push_back(0x01010101u);
        w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(w, options({}, gcn::Stage::Pixel)).lift, "outside a vertex shader is not lifted"));
    }
}

void run() { cases(); }

}  // namespace vertex_stores


// ---- compute shaders (gcn::lift_compute_shader) --------------------------------------
// Hand-encoded dispatch programs: the lift binds what the translation binds,
// reads memory through the page table as it does, lifts loops on EXEC, and
// makes the append idiom (ds_append's wave base + the lane's v_mbcnt rank)
// each invocation's own counter step; and refuses what it cannot show.
namespace compute {

struct Asm {
    std::vector<std::uint32_t> w;
    std::uint32_t at() const { return static_cast<std::uint32_t>(w.size() * 4); }
    void sopp(std::uint32_t op, std::int16_t simm = 0) { w.push_back(0xbf800000u | op << 16 | static_cast<std::uint16_t>(simm)); }
    void branch(std::uint32_t op, std::uint32_t target) { sopp(op, static_cast<std::int16_t>((static_cast<int>(target) - static_cast<int>(at()) - 4) / 4)); }
    std::size_t forward(std::uint32_t op) {
        sopp(op);
        return w.size() - 1;
    }
    void land(std::size_t slot) { w[slot] = (w[slot] & 0xffff0000u) | static_cast<std::uint16_t>(w.size() - slot - 1); }
    void sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc) { w.push_back(0xbe800000u | sdst << 16 | op << 8 | ssrc); }
    void sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t s0, std::uint32_t s1) { w.push_back(0x80000000u | op << 23 | sdst << 16 | s1 << 8 | s0); }
    void vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | op << 9 | src0); }
    void vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(op << 25 | vdst << 17 | vsrc1 << 9 | src0); }
    void vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1) { w.push_back(0x7c000000u | op << 17 | vsrc1 << 9 | src0); }
    void vop3(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1) {
        w.push_back(0xd0000000u | op << 17 | vdst);
        w.push_back(src0 | src1 << 9);
    }
    // buffer_store_dword vdata, vaddr, s[srsrc..+3], 0 offen
    void store_offen(std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t srsrc) {
        w.push_back(0xe0000000u | 28u << 18 | 1u << 12);
        w.push_back(vaddr | vdata << 8 | (srsrc / 4) << 16 | 128u << 24);
    }
    // ds_* vdst, vaddr offset:0 [gds]
    void ds(std::uint32_t op, std::uint32_t vdst, std::uint32_t vaddr, bool gds, std::uint32_t data0 = 0) {
        w.push_back(0xd8000000u | op << 18 | (gds ? 1u << 17 : 0u));
        w.push_back(vdst << 24 | data0 << 8 | vaddr);
    }
    void waitcnt() { w.push_back(0xbf8c0000u); }
    void endpgm() { w.push_back(0xbf810000u); }
};

constexpr std::uint32_t kV = 256, kC0 = 128, kVcc = 106, kExec = 126, kExecHi = 127, kM0 = 124, kMinus1 = 193;
constexpr std::uint32_t kEndpgm = 1, kBranch = 2, kExecz = 8, kBarrier = 10;   // SOPP
constexpr std::uint32_t kSMov = 3, kSMov64 = 4;                                 // SOP1
constexpr std::uint32_t kSAnd64 = 15;                                            // SOP2
constexpr std::uint32_t kVMov = 1;                                               // VOP1
constexpr std::uint32_t kVAddI32 = 37, kVLshlrev = 26;                           // VOP2
constexpr std::uint32_t kCmpGtI32 = 0x84;                                        // VOPC
constexpr std::uint32_t kMbcntLo = 0x123, kMbcntHi = 0x124;                      // VOP3
constexpr std::uint32_t kDsAppend = 62, kDsWrite = 13;                           // DS

// Four user SGPRs (a V# in s[0:3]), the fifth M0's GDS range; v0 the x in the workgroup.
gcn::TranslateOptions options(std::uint32_t threads = 64) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Compute;
    o.rsrc2 = 5u << 1;
    o.cs_threads[0] = threads;
    o.gds_counters = true;
    return o;
}

gcn::LiftResult lift(const Asm& a, const gcn::TranslateOptions& o = options()) {
    const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
    CHECK(p.errors.empty());
    const gcn::TranslateResult ref = gcn::translate(p, o);
    if (!ref.ok()) std::fprintf(stderr, "reference failed: %s\n", ref.errors.empty() ? "?" : ref.errors[0].c_str());
    CHECK(ref.ok());
    return gcn::lift_compute_shader(p, o, ref);
}

bool lifted_valid(const gcn::LiftResult& r, const char* what) {
    for (const std::string& why : r.rejections) std::fprintf(stderr, "%s rejected: %s\n", what, why.c_str());
    if (!r.ok()) return false;
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    if (!tools.Validate(r.spirv.data(), r.spirv.size())) {
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

// for (i = x; i < 32; i += 64) store i at base + 4 i: a loop on EXEC. With
// `alike`, i starts at 0 in every lane (every invocation leaves together).
Asm exec_loop(bool barrier, bool alike = false) {
    Asm a;
    a.vop1(kVMov, 1, alike ? kC0 : kV + 0);
    a.sop1(kSMov64, 6, kExec);
    const std::uint32_t header = a.at();
    a.vopc(kCmpGtI32, kC0 + 32, 1);  // v_cmp_gt_i32 vcc, 32, v1
    a.sop2(kSAnd64, kExec, kExec, kVcc);
    const std::size_t exit = a.forward(kExecz);
    a.vop2(kVLshlrev, 2, kC0 + 2, 1);
    a.store_offen(1, 2, 0);
    if (barrier) a.sopp(kBarrier);
    a.vop2(kVAddI32, 1, kC0 + 64, 1);
    a.branch(kBranch, header);
    a.land(exit);
    a.sop1(kSMov64, kExec, 6);
    a.endpgm();
    return a;
}

// The append idiom: rank = v_mbcnt_hi(exec_hi, 0), v_mbcnt_lo(exec_lo, rank);
// base = ds_append (GDS); store x at base + rank. `misuse`: store the base too.
Asm append(bool misuse) {
    Asm a;
    a.sop1(kSMov, kM0, 4);
    a.vop3(kMbcntHi, 1, kExecHi, kC0);
    a.vop3(kMbcntLo, 1, kExec, kV + 1);
    a.ds(kDsAppend, 2, 0, true);
    a.waitcnt();
    if (misuse) {
        a.vop2(kVLshlrev, 3, kC0 + 2, 2);
        a.store_offen(2, 3, 0);
    }
    a.vop2(kVAddI32, 2, kV + 2, 1);
    a.vop2(kVLshlrev, 3, kC0 + 2, 2);
    a.store_offen(0, 3, 0);
    a.endpgm();
    return a;
}

// The decorations a lifted module gives its storage-buffer variables: how
// many carry DescriptorSet `set` and how many Binding `binding`.
std::size_t decorated(const std::vector<std::uint32_t>& spv, std::uint32_t decoration, std::uint32_t value) {
    std::size_t n = 0;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t len = spv[i] >> 16;
        if (!len) break;
        if ((spv[i] & 0xffff) == 71 /* OpDecorate */ && len == 4 && spv[i + 2] == decoration && spv[i + 3] == value) ++n;
        i += len;
    }
    return n;
}

// A lean reference (ComputeLean: cb_ssbo, cb_no_fallback, its buffers in set
// 1): the lift reads what the reference bound from that storage buffer and
// walks the rest as the reference does; a store at the V#'s base plus a
// constant goes through the written binding; with a page-table fallback in
// the reference the lift refuses.
gcn::TranslateOptions lean(gcn::TranslateOptions o) {
    o.cb_ssbo = true;
    o.cb_no_fallback = true;
    o.cb_descriptor_set = 1;
    return o;
}
void lean_reference() {
    {
        Asm a;
        a.w.push_back(0xc2040100u);  // s_buffer_load_dword s8, s[0:3], 0x0
        a.waitcnt();
        a.vop1(kVMov, 1, 8);  // v_mov_b32 v1, s8
        a.vop2(kVLshlrev, 2, kC0 + 2, 0);
        a.store_offen(1, 2, 0);  // at a VGPR offset: walked
        a.endpgm();
        const gcn::TranslateOptions o = lean(options());
        const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok());
        CHECK(ref.buffers.size() == 1);
        CHECK(ref.buffer_at.count(0) == 1);
        const gcn::LiftResult r = gcn::lift_compute_shader(p, o, ref);
        CHECK(lifted_valid(r, "lean reference"));
        CHECK(proof_has(r, "1 scalar and typed loads read from the storage buffer the reference binds"));
        CHECK(proof_has(r, "reaches it: 0 scalar loads, 0 buffer loads"));  // nothing more read through the page table
        CHECK(proof_has(r, "1 buffer stores and"));                           // the walked store
        if (!ref.buffers.empty()) {
            CHECK(decorated(r.spirv, 34 /* DescriptorSet */, 1) >= 1);
            CHECK(decorated(r.spirv, 33 /* Binding */, ref.buffers[0].binding) >= 1);
        }
        gcn::TranslateOptions f = o;
        f.cb_no_fallback = false;
        CHECK(rejected_with(gcn::lift_compute_shader(p, f, gcn::translate(p, f)), "page-table fallback"));
    }
    {
        Asm a;
        a.vop1(kVMov, 1, kC0 + 5);                                // v_mov_b32 v1, 5
        a.w.push_back(0xe0000000u | 28u << 18 | 4u);              // buffer_store_dword v1, off, s[0:3], 0 offset:4
        a.w.push_back(0u | 1u << 8 | 0u << 16 | 128u << 24);
        a.endpgm();
        const gcn::TranslateOptions o = lean(options());
        const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
        const gcn::TranslateResult ref = gcn::translate(p, o);
        CHECK(ref.ok());
        CHECK(ref.buffers.size() == 1 && ref.buffers[0].written);
        const gcn::LiftResult r = gcn::lift_compute_shader(p, o, ref);
        CHECK(lifted_valid(r, "lean reference, bound store"));
        CHECK(proof_has(r, "1 buffer stores through its written binding"));
        CHECK(proof_has(r, "0 buffer stores and"));  // none through the page table
        // The walking reference walks it, and the lift with it.
        const gcn::TranslateResult walk = gcn::translate(p, options());
        const gcn::LiftResult w = gcn::lift_compute_shader(p, options(), walk);
        CHECK(lifted_valid(w, "walking reference, the same store"));
        CHECK(proof_has(w, "1 buffer stores and"));
    }
    // Elements at a VGPR index or offset (translate.cpp bound_compute_access):
    // a typed store of four floats at v0's index, buffer_load_format_x at it,
    // buffer_atomic_add at index and offset with GLC, a typed load at a VGPR
    // offset alone (a ranged binding) - each bound in the lean reference, read
    // and written there by the lift, nothing through the page table.
    {
        const auto mtbuf = [](Asm& a, std::uint32_t op, std::uint32_t dfmt, bool idxen, bool offen, std::uint32_t vdata, std::uint32_t vaddr,
                              std::uint32_t srsrc, std::uint32_t offset12) {
            a.w.push_back(0xe8000000u | 7u << 23 | dfmt << 19 | op << 16 | (idxen ? 1u << 13 : 0u) | (offen ? 1u << 12 : 0u) | offset12);
            a.w.push_back(vaddr | vdata << 8 | (srsrc / 4) << 16 | 128u << 24);
        };
        const auto mubuf = [](Asm& a, std::uint32_t op, bool idxen, bool offen, bool glc, std::uint32_t vdata, std::uint32_t vaddr, std::uint32_t srsrc) {
            a.w.push_back(0xe0000000u | op << 18 | (idxen ? 1u << 13 : 0u) | (offen ? 1u << 12 : 0u) | (glc ? 1u << 14 : 0u));
            a.w.push_back(vaddr | vdata << 8 | (srsrc / 4) << 16 | 128u << 24);
        };
        gcn::TranslateOptions o = lean(options());
        o.rsrc2 = 9u << 1;  // two V#s: s[0:3] and s[4:7] (and s8)
        {
            Asm a;
            a.vop1(kVMov, 1, kC0 + 1);
            a.vop1(kVMov, 2, kC0 + 2);
            a.vop1(kVMov, 3, kC0 + 3);
            a.vop1(kVMov, 4, kC0 + 4);
            mtbuf(a, 7, 14, true, false, 1, 0, 0, 16);  // tbuffer_store_format_xyzw v[1:4], v0, s[0:3] idxen offset:16
            mubuf(a, 0, true, false, false, 5, 0, 4);   // buffer_load_format_x v5, v0, s[4:7] idxen
            a.waitcnt();
            a.vop2(kVLshlrev, 6, kC0 + 2, 0);           // v6 = 4 x
            a.vop1(kVMov, 7, 0);                        // v7 = x: [v7, v8] = index, offset
            a.vop1(kVMov, 8, kV + 6);
            mubuf(a, 50, true, true, true, 5, 7, 4);    // buffer_atomic_add v5, v[7:8], s[4:7] idxen offen glc
            a.waitcnt();
            mtbuf(a, 0, 4, false, true, 9, 6, 0, 0);    // tbuffer_load_format_x v9, v6, s[0:3] offen: no index
            a.waitcnt();
            a.store_offen(9, 6, 4);                     // walked: buffer_store_dword at a VGPR offset of s[4:7]... bound? no: offen
            a.endpgm();
            const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
            CHECK(p.errors.empty());
            const gcn::TranslateResult ref = gcn::translate(p, o);
            CHECK(ref.ok());
            if (!ref.ok() && !ref.errors.empty()) std::fprintf(stderr, "lean elements: %s\n", ref.errors[0].c_str());
            // s[0:3] is read at an index and at an offset alone: the second
            // binds nothing (cb_stride holds the stride) and walks.
            CHECK(ref.buffers.size() == 2);
            bool written = true;
            for (const gcn::BufferBinding& b : ref.buffers) written = written && b.written;
            CHECK(written);
            const gcn::LiftResult r = gcn::lift_compute_shader(p, o, ref);
            CHECK(lifted_valid(r, "lean reference, bound elements"));
            CHECK(proof_has(r, "3 elements at a VGPR index or offset"));
            CHECK(count_op(r.spirv, 234) == 1);  // OpAtomicIAdd, on the binding
            // HLSL: the written bindings as RWByteAddressBuffers, the add as their InterlockedAdd.
            const gcn::HlslResult h = gcn::emit_hlsl(r.spirv);
            CHECK(h.ok());
            CHECK(h.source.find("RWByteAddressBuffer") != std::string::npos);
            CHECK(h.source.find(".InterlockedAdd(") != std::string::npos);
        }
        {
            Asm a;  // the offset-only typed load alone: a ranged binding
            a.vop2(kVLshlrev, 6, kC0 + 2, 0);
            mtbuf(a, 0, 4, false, true, 9, 6, 0, 0);  // tbuffer_load_format_x v9, v6, s[0:3] offen
            a.waitcnt();
            mtbuf(a, 4, 4, false, true, 9, 6, 4, 0);  // tbuffer_store_format_x v9, v6, s[4:7] offen
            a.endpgm();
            const gcn::Program p = gcn::decode(a.w.data(), a.w.size());
            const gcn::TranslateResult ref = gcn::translate(p, o);
            CHECK(ref.ok());
            CHECK(ref.buffers.size() == 2 && ref.buffers[0].ranged && ref.buffers[1].ranged && ref.buffers[1].written && ref.walks.empty());
            const gcn::LiftResult r = gcn::lift_compute_shader(p, o, ref);
            CHECK(lifted_valid(r, "lean reference, ranged elements"));
            CHECK(proof_has(r, "2 elements at a VGPR index or offset"));
            // Walked in the walking reference, and by its lift.
            gcn::TranslateOptions wo = options();
            wo.rsrc2 = o.rsrc2;
            const gcn::TranslateResult walk = gcn::translate(p, wo);
            CHECK(walk.ok() && walk.buffers.empty());
            CHECK(lifted_valid(gcn::lift_compute_shader(p, wo, walk), "walking reference, the same elements"));
        }
    }
}

void run() {
    {
        const gcn::LiftResult r = lift(exec_loop(false));
        CHECK(lifted_valid(r, "loop on EXEC"));
        CHECK(proof_has(r, "on EXEC"));
        CHECK(proof_has(r, "1 buffer stores"));
        CHECK(count_op(r.spirv, 246) == 1);  // OpLoopMerge
    }
    {
        // A barrier needs every invocation that entered the loop: this one's
        // invocations leave at different iterations.
        const gcn::LiftResult r = lift(exec_loop(true));
        CHECK(rejected_with(r, "which invocations leave at different iterations"));
        // ... and this one's leave at the same iteration: the barrier stays.
        const gcn::LiftResult u = lift(exec_loop(true, true));
        CHECK(lifted_valid(u, "barrier in a loop every invocation leaves together"));
        CHECK(proof_has(u, "1 s_barrier"));
        CHECK(count_op(u.spirv, 224) == 1);  // OpControlBarrier
    }
    for (const std::uint32_t threads : {32u, 128u}) {
        // The default (TranslateOptions::gds_subgroup): an elected invocation's
        // atomic for the subgroup's count, each invocation its rank among them.
        const gcn::LiftResult r = lift(append(false), options(threads));
        CHECK(lifted_valid(r, "append idiom"));
        CHECK(r.wave.gds_steps == 1);
        CHECK(count_op(r.spirv, 333) == 1);  // OpGroupNonUniformElect
        CHECK(count_op(r.spirv, 339) == 2);  // OpGroupNonUniformBallot: the count and the rank
        CHECK(count_op(r.spirv, 342) == 2);  // OpGroupNonUniformBallotBitCount
        CHECK(count_op(r.spirv, 338) == 1);  // OpGroupNonUniformBroadcastFirst: the base
        CHECK(count_op(r.spirv, 234) == 1);  // one OpAtomicIAdd
        // gds_subgroup off: each invocation steps the counter itself.
        gcn::TranslateOptions each = options(threads);
        each.gds_subgroup = false;
        const gcn::LiftResult e = lift(append(false), each);
        CHECK(lifted_valid(e, "append idiom, an atomic an invocation"));
        CHECK(count_op(e.spirv, 339) == 0);  // no OpGroupNonUniformBallot
        CHECK(count_op(e.spirv, 333) == 0);
        CHECK(count_op(e.spirv, 234) == 1);  // one OpAtomicIAdd
    }
    CHECK(rejected_with(lift(append(true)), "used other than added to its lanes' ranks"));
    {
        Asm a;  // v_mbcnt_lo of -1: the lane's index, not a rank among a mask's lanes
        a.vop3(kMbcntLo, 1, kMinus1, kC0);
        a.vop2(kVLshlrev, 2, kC0 + 2, 1);
        a.store_offen(1, 2, 0);
        a.endpgm();
        CHECK(rejected_with(lift(a), "not a lane mask"));
    }
    {
        Asm a;  // workgroup memory
        a.ds(kDsWrite, 0, 0, false, 0);
        a.endpgm();
        gcn::TranslateOptions o = options();
        o.rsrc2 |= 1u << 15;  // LDS_SIZE: 128 dwords
        CHECK(rejected_with(lift(a, o), "LDS (groupshared memory) is not lifted"));
    }
    {
        // s_cbranch_execnz over an s_branch to the end: an early-out lifted
        // as the region it skips.
        Asm a;
        a.vopc(kCmpGtI32, kC0 + 16, kV + 0 - kV);  // placeholder replaced below
        a.w.pop_back();
        a.vopc(kCmpGtI32, kC0 + 16, 0);  // v_cmp_gt_i32 vcc, 16, v0
        a.sop1(36 /* s_and_saveexec_b64 */, 6, kVcc);
        const std::size_t nz = a.forward(9 /* s_cbranch_execnz */);
        const std::size_t to_end = a.forward(kBranch);
        a.land(nz);
        a.vop2(kVLshlrev, 2, kC0 + 2, 0);
        a.store_offen(0, 2, 0);
        a.land(to_end);
        a.endpgm();
        const gcn::LiftResult r = lift(a);
        CHECK(lifted_valid(r, "early-out"));
        CHECK(proof_has(r, "s_cbranch_execnz"));
    }
    // An s_cbranch_execz region: a selection an invocation whose EXEC bit is
    // clear skips - unless something in it needs the other invocations (a
    // barrier), when every invocation runs it with its writes masked.
    for (const bool barrier : {false, true}) {
        Asm a;
        a.vopc(kCmpGtI32, kC0 + 16, kV + 0);  // v_cmp_gt_i32 vcc, 16, v0
        a.sop1(36 /* s_and_saveexec_b64 */, 6, kVcc);
        const std::size_t skip = a.forward(kExecz);
        if (barrier) a.sopp(kBarrier);
        a.vop2(kVLshlrev, 2, kC0 + 2, 0);
        a.store_offen(0, 2, 0);
        a.land(skip);
        a.sop1(kSMov64, kExec, 6);
        a.endpgm();
        const gcn::LiftResult r = lift(a);
        CHECK(lifted_valid(r, barrier ? "region with a barrier" : "region"));
        const bool branched = std::any_of(r.proof.begin(), r.proof.end(), [](const std::string& line) {
            return line.find("a selection on the invocation's EXEC bit") != std::string::npos;
        });
        CHECK(branched == !barrier);
        if (barrier) CHECK(proof_has(r, "1 s_barrier"));
    }
    lean_reference();
}

}  // namespace compute

int main(int argc, char** argv) {
    // Hand-encoded programs first: they need no game files.
    compute::run();
    vertex::run();
    vertex_stores::run();
    control_flow::run();
    cndmask_modifiers();
    packed_exports();
    synthetic_swizzles();
    swizzles_in_blocks();
    swizzles_and_loops();
    lane_ops::run();
    lod::cases();
    lod::loads_and_sizes();
    lod::uniform_texels();
    lod::loops();
    lod::helper_lanes();
    if (g_failures) {
        std::fprintf(stderr, "gcn_lift_test: %d check(s) failed in the hand-encoded programs\n", g_failures);
        return 1;
    }
    const std::string path = argc > 1 ? argv[1] : test_app0_file("dvdroot_ps4/shader/gxrenderershader.shaderbnd.dcx");
    std::vector<std::uint8_t> raw;
    if (!gcn::read_file(path, raw)) {
        std::printf("gcn_lift_test: the hand-encoded programs pass; the pilot is skipped: %s not readable\n", path.c_str());
        return kTestSkip;
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

    // Without the restore after the first masked region, EXEC stays the
    // else-mask: the sample at 0x4cc is lifted under it (its coordinates are
    // each pixel's own), but the export then writes pixels whose EXEC bit is
    // clear and s_endpgm does not kill them.
    CHECK(rejected_with(lift(mutate(p, 0x45c, "s_mov_b64", make_nop), o), "a pixel exported where EXEC was clear is not killed"));

    // Without the load at 0x460, s4 written in the second region (0x3ec-0x45c)
    // is read at 0x46c: GCN runs that block for every lane when any lane needs it.
    CHECK(rejected_with(lift(mutate(p, 0x460, "s_buffer_load_dwordx4", make_nop), o), "s4 is written inside region"));

    // An s_branch that closes no if's then arm (here a jump over a region).
    CHECK(rejected_with(lift(mutate(p, 0x3ac, "s_cbranch_execz", [](gcn::Inst& in) { in.op = 2; }), o), "does not close an if"));

    // A VCC branch becomes an if only where every lane holds the same VCC bit;
    // the pilot's VCC comes from per-pixel values.
    CHECK(rejected_with(lift(mutate(p, 0x3ac, "s_cbranch_execz", [](gcn::Inst& in) { in.op = 6; }), o), "VCC that may differ between lanes"));

    // A sample inside a masked region, where EXEC is the region's mask (the
    // instruction at 0x3b0 becomes the sample from 0x068): its coordinates were
    // computed for every pixel, so it is lifted for every pixel in uniform
    // control flow and its texel written under the region's EXEC.
    gcn::Inst sample;
    for (const gcn::Inst& in : p.insts) {
        if (in.offset == 0x68) sample = in;
    }
    const gcn::LiftResult in_region = lift(mutate(p, 0x3b0, "v_mul_f32", [&](gcn::Inst& in) {
        const std::uint32_t at = in.offset;
        const std::uint8_t size = in.size;
        in = sample;
        in.offset = at;
        in.size = size;
    }), o);
    CHECK(lifted(in_region));
    CHECK(proof_says(in_region, "1 image samples under a varying EXEC: 0003b0"));

    if (g_failures) {
        std::fprintf(stderr, "gcn_lift_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("gcn_lift_test ok: the hand-encoded programs pass (every ds_swizzle_b32 quad pattern lifted exactly, v_cndmask_b32's "
                "modifiers and packed exports, loops, regions and ifs, lane writes, texture LOD and whole-quad mode, the unproven ones "
                "refused); pilot lifted (%zu words), its constructs checked\n",
                ok.spirv.size());
    return 0;
}
