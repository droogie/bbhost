// A camera cutaway: a clip distance from a stage's clip-space position as it
// leaves the stage (host/vertex_cutaway.h).
#include "host/vertex_cutaway.h"

#include "gcn/spirv.h"

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using u32 = std::uint32_t;

constexpr u32 kBiClipDistance = 3;
// Kept nearer than this (view depth): the full-screen passes the camera's
// depth is bound to - the lighting, the composite, the HUD over it - are
// triangles at w = 1, which the cylinder would otherwise take whole.
constexpr float kNearest = 1.5f;

struct Ins {
    std::size_t at;  // its first word in the module
    u32 op, n;
};

std::string word_str(const std::vector<u32>& w, std::size_t from, std::size_t end) {
    std::string s;
    for (std::size_t k = from; k < end; ++k) {
        for (int b = 0; b < 4; ++b) {
            const char c = static_cast<char>((w[k] >> (8 * b)) & 0xff);
            if (!c) return s;
            s += c;
        }
    }
    return s;
}

std::vector<u32> mk(u32 op, std::initializer_list<u32> operands) {
    std::vector<u32> w;
    w.reserve(operands.size() + 1);
    w.push_back(static_cast<u32>(operands.size() + 1) << 16 | op);
    w.insert(w.end(), operands.begin(), operands.end());
    return w;
}

u32 f32_bits(float f) {
    u32 b;
    static_assert(sizeof b == sizeof f);
    std::memcpy(&b, &f, 4);
    return b;
}

bool annotation(u32 op) { return op == spv::OpDecorate || op == spv::OpMemberDecorate || (op >= 73 && op <= 75) || op == 332 || op == 5632 || op == 5633; }
bool preamble(u32 op) {
    return op == spv::OpCapability || op == spv::OpExtension || op == spv::OpExtInstImport || op == spv::OpMemoryModel || op == spv::OpEntryPoint ||
           op == spv::OpExecutionMode || op == 331 || (op >= 2 && op <= 7) || op == 330 || annotation(op);
}

}  // namespace

bool vertex_cutaway(std::vector<u32>& spirv, const float cut[4], std::string* why) {
    const auto fail = [&](const char* w) {
        if (why) *why = w;
        return false;
    };
    const std::vector<u32>& w = spirv;
    if (w.size() < 5 || w[0] != 0x07230203u) return fail("not a SPIR-V module");
    if (!(cut[0] > 0.0f) || !(cut[1] > 0.0f) || !(cut[3] > 0.0f)) return fail("no cutaway");
    std::vector<Ins> ins;
    for (std::size_t i = 5; i < w.size();) {
        const u32 n = w[i] >> 16;
        if (!n || i + n > w.size()) return fail("a malformed instruction");
        ins.push_back({i, w[i] & 0xffff, n});
        i += n;
    }
    u32 model = ~0u, entry = 0, glsl = 0, t_f32 = 0, t_int = 0, entries = 0;
    bool has_clip_cap = false;
    std::size_t entry_ins = ins.size(), first_cap = ins.size();
    std::map<u32, u32> builtin;
    std::map<std::pair<u32, u32>, u32> member_bi;
    std::map<u32, std::pair<u32, u32>> vec;
    std::map<u32, std::pair<u32, u32>> ptr;
    std::map<u32, u32> out_var;
    std::size_t first_fn = ins.size(), last_annot = ins.size(), first_global = ins.size(), last_import = ins.size(), memory_model = ins.size();
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        const u32* a = &w[x.at];
        if (x.op == spv::OpFunction && first_fn == ins.size()) first_fn = k;
        if (first_fn != ins.size()) continue;
        if (annotation(x.op)) last_annot = k;
        if (!preamble(x.op) && x.op != 8 /* OpLine */ && x.op != 317 /* OpNoLine */ && first_global == ins.size()) first_global = k;
        switch (x.op) {
        case spv::OpCapability:
            if (first_cap == ins.size()) first_cap = k;
            if (x.n >= 2 && a[1] == spv::CapClipDistance) has_clip_cap = true;
            break;
        case spv::OpEntryPoint:
            if (x.n < 4) return fail("a malformed entry point");
            ++entries;
            model = a[1];
            entry = a[2];
            entry_ins = k;
            break;
        case spv::OpExtInstImport:
            last_import = k;
            if (word_str(w, x.at + 2, x.at + x.n) == "GLSL.std.450") glsl = a[1];
            break;
        case spv::OpMemoryModel: memory_model = k; break;
        case spv::OpDecorate:
            if (x.n >= 4 && a[2] == spv::DecBuiltIn) builtin[a[1]] = a[3];
            break;
        case spv::OpMemberDecorate:
            if (x.n >= 5 && a[3] == spv::DecBuiltIn) member_bi[{a[1], a[2]}] = a[4];
            break;
        case spv::OpTypeFloat:
            if (x.n >= 3 && a[2] == 32 && !t_f32) t_f32 = a[1];
            break;
        case spv::OpTypeInt:
            if (x.n >= 4 && a[2] == 32 && !t_int) t_int = a[1];
            break;
        case spv::OpTypeVector:
            if (x.n >= 4) vec[a[1]] = {a[2], a[3]};
            break;
        case spv::OpTypePointer:
            if (x.n >= 4) ptr[a[1]] = {a[2], a[3]};
            break;
        case spv::OpVariable:
            if (x.n >= 4 && a[3] == spv::ScOutput) out_var[a[2]] = a[1];
            break;
        default: break;
        }
    }
    if (entries != 1) return fail("not one entry point");
    if (model != spv::EmVertex && model != spv::EmTessellationEvaluation && model != spv::EmGeometry) return fail("not a stage the rasterizer follows");
    if (first_fn == ins.size() || memory_model == ins.size() || first_global == ins.size() || first_cap == ins.size()) return fail("no function");
    if (!t_f32) return fail("no 32-bit float type");
    for (const auto& [var, bi] : builtin)
        if (bi == kBiClipDistance) return fail("it writes clip distances of its own");
    for (const auto& [sm, bi] : member_bi)
        if (bi == kBiClipDistance) return fail("it writes clip distances of its own");
    const auto is_v4f = [&](u32 t) {
        const auto v = vec.find(t);
        return v != vec.end() && v->second.first == t_f32 && v->second.second == 4;
    };
    u32 pos_var = 0, v4f = 0, block_var = 0, member = 0;
    for (const auto& [var, ptype] : out_var) {
        const auto p = ptr.find(ptype);
        if (p == ptr.end()) continue;
        const auto b = builtin.find(var);
        if (b != builtin.end() && b->second == spv::BiPosition) {
            if (!is_v4f(p->second.second)) return fail("the position is not a vec4 of 32-bit floats");
            pos_var = var;
            v4f = p->second.second;
        }
        for (const auto& [sm, bi] : member_bi) {
            if (sm.first == p->second.second && bi == spv::BiPosition) {
                block_var = var;
                member = sm.second;
            }
        }
    }
    if (!pos_var && !block_var) return fail("no position output");
    u32 v4f_out_ptr = 0, f_out_ptr = 0;
    if (!pos_var) {
        u32 block_struct = 0;
        if (const auto p = ptr.find(out_var[block_var]); p != ptr.end()) block_struct = p->second.second;
        for (const Ins& x : ins) {
            if (x.op != spv::OpTypeStruct || w[x.at + 1] != block_struct) continue;
            if (2 + member >= x.n) return fail("the output block has no such member");
            v4f = w[x.at + 2 + member];
        }
        if (!is_v4f(v4f)) return fail("the position is not a vec4 of 32-bit floats (or the output block is an array)");
    }
    for (const auto& [id, p] : ptr) {
        if (p.first == spv::ScOutput && p.second == v4f) v4f_out_ptr = id;
        if (p.first == spv::ScOutput && p.second == t_f32) f_out_ptr = id;
    }
    std::vector<std::size_t> at;
    {
        bool in_entry = false;
        for (std::size_t k = first_fn; k < ins.size(); ++k) {
            const Ins& x = ins[k];
            if (x.op == spv::OpFunction) in_entry = x.n >= 3 && w[x.at + 2] == entry;
            if (model == spv::EmGeometry ? x.op == spv::OpEmitVertex : (in_entry && x.op == spv::OpReturn)) at.push_back(k);
        }
    }
    if (at.empty()) return fail(model == spv::EmGeometry ? "no emit" : "no return from the entry point");

    // New ids: an int type, the array of one float, its Output pointer, the
    // variable, the float Output pointer, the constants.
    u32 bound = w[3];
    std::vector<u32> globals, imports, decorations, caps;
    const auto add = [&](std::vector<u32>& to, const std::vector<u32>& i) { to.insert(to.end(), i.begin(), i.end()); };
    if (!t_int) {
        t_int = bound++;
        add(globals, mk(spv::OpTypeInt, {t_int, 32, 1}));
    }
    const u32 c_zero = bound++, c_one = bound++;
    add(globals, mk(spv::OpConstant, {t_int, c_zero, 0}));
    add(globals, mk(spv::OpConstant, {t_int, c_one, 1}));
    const u32 t_arr = bound++, t_arr_ptr = bound++, clip_var = bound++;
    add(globals, mk(spv::OpTypeArray, {t_arr, t_f32, c_one}));
    add(globals, mk(spv::OpTypePointer, {t_arr_ptr, spv::ScOutput, t_arr}));
    if (!f_out_ptr) {
        f_out_ptr = bound++;
        add(globals, mk(spv::OpTypePointer, {f_out_ptr, spv::ScOutput, t_f32}));
    }
    if (!pos_var && !v4f_out_ptr) {
        v4f_out_ptr = bound++;
        add(globals, mk(spv::OpTypePointer, {v4f_out_ptr, spv::ScOutput, v4f}));
    }
    u32 c_member = 0;
    if (!pos_var) {
        c_member = bound++;
        add(globals, mk(spv::OpConstant, {t_int, c_member, member}));
    }
    const auto constant = [&](float f) {
        const u32 id = bound++;
        add(globals, mk(spv::OpConstant, {t_f32, id, f32_bits(f)}));
        return id;
    };
    // BBHOST_CUTAWAY_KEEP=1, a test: the distance written is 1 - nothing clipped.
    static const bool keep_all = [] {
        const char* e = std::getenv("BBHOST_CUTAWAY_KEEP");
        return e && e[0] == '1';
    }();
    const u32 c_keep = constant(1.0f), c_w_min = constant(kNearest);
    const u32 c_tx = constant(cut[0]), c_ty = constant(cut[1]), c_near = constant(cut[2]), c_radius = constant(cut[3]);
    add(globals, mk(spv::OpVariable, {t_arr_ptr, clip_var, spv::ScOutput}));
    add(decorations, mk(spv::OpDecorate, {clip_var, spv::DecBuiltIn, kBiClipDistance}));
    if (!has_clip_cap) add(caps, mk(spv::OpCapability, {spv::CapClipDistance}));
    if (!glsl) {
        glsl = bound++;
        std::vector<u32> name = {0, 0, 0, 0};
        const char* s = "GLSL.std.450";
        for (std::size_t k = 0; s[k]; ++k) name[k / 4] |= static_cast<u32>(static_cast<unsigned char>(s[k])) << (8 * (k % 4));
        imports.push_back(static_cast<u32>(2 + name.size()) << 16 | spv::OpExtInstImport);
        imports.push_back(glsl);
        imports.insert(imports.end(), name.begin(), name.end());
    }
    const auto cut_code = [&]() {
        std::vector<u32> c;
        u32 pp = pos_var;
        if (!pp) {
            pp = bound++;
            add(c, mk(spv::OpAccessChain, {v4f_out_ptr, pp, block_var, c_member}));
        }
        const u32 p = bound++, x = bound++, y = bound++, wv = bound++;
        add(c, mk(spv::OpLoad, {v4f, p, pp}));
        add(c, mk(spv::OpCompositeExtract, {t_f32, x, p, 0}));
        add(c, mk(spv::OpCompositeExtract, {t_f32, y, p, 1}));
        add(c, mk(spv::OpCompositeExtract, {t_f32, wv, p, 3}));
        const u32 vx = bound++, vy = bound++, xx = bound++, yy = bound++, r2 = bound++, r = bound++, across = bound++, depth = bound++,
                  d0 = bound++, inside = bound++, d = bound++, slot = bound++;
        add(c, mk(spv::OpFMul, {t_f32, vx, x, c_tx}));
        add(c, mk(spv::OpFMul, {t_f32, vy, y, c_ty}));
        add(c, mk(spv::OpFMul, {t_f32, xx, vx, vx}));
        add(c, mk(spv::OpFMul, {t_f32, yy, vy, vy}));
        add(c, mk(spv::OpFAdd, {t_f32, r2, xx, yy}));
        add(c, mk(spv::OpExtInst, {t_f32, r, glsl, spv::GlslSqrt, r2}));
        add(c, mk(spv::OpFSub, {t_f32, across, r, c_radius}));
        add(c, mk(spv::OpFSub, {t_f32, depth, wv, c_near}));
        add(c, mk(spv::OpExtInst, {t_f32, d0, glsl, spv::GlslFMax, across, depth}));
        add(c, mk(spv::OpFSub, {t_f32, inside, c_w_min, wv}));
        add(c, mk(spv::OpExtInst, {t_f32, d, glsl, spv::GlslFMax, d0, inside}));
        add(c, mk(spv::OpAccessChain, {f_out_ptr, slot, clip_var, c_zero}));
        add(c, mk(spv::OpStore, {slot, keep_all ? c_keep : d}));
        return c;
    };
    std::vector<std::vector<u32>> code;
    for (std::size_t k = 0; k < at.size(); ++k) code.push_back(cut_code());

    std::vector<u32> out(w.begin(), w.begin() + 5);
    out[3] = bound;
    out.reserve(w.size() + globals.size() + imports.size() + decorations.size() + caps.size() + code.size() * code[0].size() + 1);
    std::size_t site = 0;
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        if (k == first_cap) out.insert(out.end(), caps.begin(), caps.end());
        if (last_import == ins.size() && k == memory_model) out.insert(out.end(), imports.begin(), imports.end());
        if (last_annot == ins.size() && k == first_global) out.insert(out.end(), decorations.begin(), decorations.end());
        if (k == first_fn) out.insert(out.end(), globals.begin(), globals.end());
        if (site < at.size() && at[site] == k) {
            out.insert(out.end(), code[site].begin(), code[site].end());
            ++site;
        }
        if (k == entry_ins) {
            // The entry point's interface takes the new output.
            out.push_back(static_cast<u32>(x.n + 1) << 16 | x.op);
            out.insert(out.end(), w.begin() + static_cast<std::ptrdiff_t>(x.at + 1), w.begin() + static_cast<std::ptrdiff_t>(x.at + x.n));
            out.push_back(clip_var);
        } else {
            out.insert(out.end(), w.begin() + static_cast<std::ptrdiff_t>(x.at), w.begin() + static_cast<std::ptrdiff_t>(x.at + x.n));
        }
        if (last_import != ins.size() && k == last_import) out.insert(out.end(), imports.begin(), imports.end());
        if (last_annot != ins.size() && k == last_annot) out.insert(out.end(), decorations.begin(), decorations.end());
    }
    spirv = std::move(out);
    return true;
}
