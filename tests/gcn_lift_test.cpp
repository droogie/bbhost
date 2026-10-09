// The pilot pixel shader (a22c7f71) lifts from the shipped bundle, and the lifter
// refuses the constructs it cannot prove equal. Skips when the dump is absent.
// Before that, hand-encoded programs check ds_swizzle_b32 (every quad pattern
// run through a small interpreter of the lifted SPIR-V over one quad, and the
// cases the lifter must refuse), texture LOD and whole-quad mode, loops,
// regions and ifs, and lane writes; they run without the dump.
#include "test_app0.h"
#include "gcn/container.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <array>
#include <bit>
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
        case 186:  // OpFOrdGreaterThan
            for (int l = 0; l < 4; ++l) val[a[1]][static_cast<std::size_t>(l)] = {f(get(a[2], l)[0]) > f(get(a[3], l)[0]) ? 1u : 0u, 0, 0, 0};
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
    CHECK(lifted(lift(divergent([](Asm& a) { a.mimg(kGetLod, 4, 2, 0x3); }))));
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
        a.mimg(kGetLod, 4, 2, 0x2);
        a.exp_mrt0(4);
        a.w.push_back(0xbf810000u);
        CHECK(rejected_with(lift(a.w), "image_get_lod after a kill region"));
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
gcn::LiftResult lift_words(const std::vector<std::uint32_t>& words, gcn::Program* out = nullptr) {
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
    const gcn::TranslateResult ref = gcn::translate(p, o);
    if (!ref.ok()) std::fprintf(stderr, "reference failed: %s\n", ref.errors[0].c_str());
    CHECK(ref.ok());
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
    // s[2:3] copied to s[20:21]: the reference stops binding the table (its
    // registers are written) and walks the page table for both loads. They
    // are a T# and an S#, which the sample binds by path.
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
    const gcn::LiftResult ok = lift_words(sample([](Asm&) {}));
    CHECK(lifted_valid(ok));
    CHECK(proof_has(ok, "2 scalar loads the reference walks the page table for hold descriptors only"));
    // A word of them read as data: the reference reads it from guest memory.
    CHECK(rejected_with(lift_words(sample([](Asm& a) {
                            a.vop1(kVMov, 6, 16);       // v_mov_b32 v6, s16 (a copy is still a descriptor)
                            a.vop2(kVAdd, 0, kV + 0, 6);  // v_add_f32 v0, v0, v6
                        })),
                        "data from a scalar load the reference walks the page table for"));
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
    cselect();
    lshl_b64();
}

}  // namespace lane_ops

int main(int argc, char** argv) {
    // Hand-encoded programs first: they need no game files.
    control_flow::run();
    synthetic_swizzles();
    swizzles_and_loops();
    lane_ops::run();
    lod::cases();
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
    std::printf("gcn_lift_test ok: the hand-encoded programs pass (every ds_swizzle_b32 quad pattern lifted exactly, loops, regions and "
                "ifs, lane writes, texture LOD and whole-quad mode, the unproven ones refused); pilot lifted (%zu words), its constructs "
                "checked\n",
                ok.spirv.size());
    return 0;
}
