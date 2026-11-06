// The camera cutaway per pixel: a fragment stage that leaves out what stands
// between the camera and the character, faded (host/pixel_cutaway.h).
#include "host/pixel_cutaway.h"

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

constexpr u32 kBiFragCoord = 15;
constexpr u32 kExEarlyFragmentTests = 9;
constexpr u32 kOpTerminateInvocation = 4416;
constexpr u32 kGlslFract = 10;

struct Ins {
    std::size_t at;
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
    std::memcpy(&b, &f, 4);
    return b;
}

bool annotation(u32 op) { return op == spv::OpDecorate || op == spv::OpMemberDecorate || (op >= 73 && op <= 75) || op == 332 || op == 5632 || op == 5633; }
bool preamble(u32 op) {
    return op == spv::OpCapability || op == spv::OpExtension || op == spv::OpExtInstImport || op == spv::OpMemoryModel || op == spv::OpEntryPoint ||
           op == spv::OpExecutionMode || op == 331 || (op >= 2 && op <= 7) || op == 330 || annotation(op);
}

}  // namespace

bool pixel_cutaway(std::vector<u32>& spirv, const PixelCut& cut, std::string* why) {
    const auto fail = [&](const char* w) {
        if (why) *why = w;
        return false;
    };
    const std::vector<u32>& w = spirv;
    if (w.size() < 5 || w[0] != 0x07230203u) return fail("not a SPIR-V module");
    if (!(cut.scale_x != 0.0f) || !(cut.scale_y != 0.0f) || !(cut.tan_x > 0.0f) || !(cut.tan_y > 0.0f) || !(cut.radius > 0.0f))
        return fail("no cutaway");
    std::vector<Ins> ins;
    for (std::size_t i = 5; i < w.size();) {
        const u32 n = w[i] >> 16;
        if (!n || i + n > w.size()) return fail("a malformed instruction");
        ins.push_back({i, w[i] & 0xffff, n});
        i += n;
    }
    u32 model = ~0u, entry = 0, glsl = 0, t_f32 = 0, t_bool = 0, entries = 0;
    std::size_t entry_ins = ins.size();
    std::map<u32, u32> builtin;
    std::map<u32, std::pair<u32, u32>> vec, ptr;
    std::map<u32, u32> in_var;  // Input variable -> its pointer type
    std::vector<std::size_t> early;  // EarlyFragmentTests modes, left out
    std::size_t first_fn = ins.size(), last_annot = ins.size(), first_global = ins.size(), last_import = ins.size(), memory_model = ins.size();
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        const u32* a = &w[x.at];
        if (x.op == spv::OpFunction && first_fn == ins.size()) first_fn = k;
        if (first_fn != ins.size()) continue;
        if (annotation(x.op)) last_annot = k;
        if (!preamble(x.op) && x.op != 8 && x.op != 317 && first_global == ins.size()) first_global = k;
        switch (x.op) {
        case spv::OpEntryPoint:
            if (x.n < 4) return fail("a malformed entry point");
            ++entries;
            model = a[1];
            entry = a[2];
            entry_ins = k;
            break;
        case spv::OpExecutionMode:
            if (x.n >= 3 && a[2] == kExEarlyFragmentTests) early.push_back(k);
            break;
        case spv::OpExtInstImport:
            last_import = k;
            if (word_str(w, x.at + 2, x.at + x.n) == "GLSL.std.450") glsl = a[1];
            break;
        case spv::OpMemoryModel: memory_model = k; break;
        case spv::OpDecorate:
            if (x.n >= 4 && a[2] == spv::DecBuiltIn) builtin[a[1]] = a[3];
            break;
        case spv::OpTypeFloat:
            if (x.n >= 3 && a[2] == 32 && !t_f32) t_f32 = a[1];
            break;
        case spv::OpTypeBool: t_bool = a[1]; break;
        case spv::OpTypeVector:
            if (x.n >= 4) vec[a[1]] = {a[2], a[3]};
            break;
        case spv::OpTypePointer:
            if (x.n >= 4) ptr[a[1]] = {a[2], a[3]};
            break;
        case spv::OpVariable:
            if (x.n >= 4 && a[3] == spv::ScInput) in_var[a[2]] = a[1];
            break;
        default: break;
        }
    }
    if (entries != 1) return fail("not one entry point");
    if (model != spv::EmFragment) return fail("not a fragment stage");
    if (first_fn == ins.size() || memory_model == ins.size() || first_global == ins.size()) return fail("no function");
    if (!t_f32) return fail("no 32-bit float type");
    // The entry function's first block: its label and the variables that open it.
    std::size_t fn_at = ins.size(), label_at = ins.size(), vars_end = ins.size();
    for (std::size_t k = first_fn; k < ins.size(); ++k) {
        if (ins[k].op == spv::OpFunction && ins[k].n >= 3 && w[ins[k].at + 2] == entry) {
            fn_at = k;
            break;
        }
    }
    if (fn_at == ins.size()) return fail("no entry function");
    for (std::size_t k = fn_at + 1; k < ins.size(); ++k) {
        if (ins[k].op == spv::OpLabel) {
            label_at = k;
            break;
        }
        if (ins[k].op != spv::OpFunctionParameter && ins[k].op != 8 && ins[k].op != 317) return fail("an entry function in a form this does not take");
    }
    if (label_at == ins.size()) return fail("no entry block");
    vars_end = label_at + 1;
    while (vars_end < ins.size() && (ins[vars_end].op == spv::OpVariable || ins[vars_end].op == 8 || ins[vars_end].op == 317)) ++vars_end;
    const u32 old_label = w[ins[label_at].at + 1];

    u32 bound = w[3];
    std::vector<u32> globals, imports, decorations;
    const auto add = [&](std::vector<u32>& to, const std::vector<u32>& i) { to.insert(to.end(), i.begin(), i.end()); };
    // The window position: the stage's FragCoord, else one of ours.
    u32 frag = 0, v4f = 0;
    for (const auto& [var, pt] : in_var) {
        const auto b = builtin.find(var);
        if (b == builtin.end() || b->second != kBiFragCoord) continue;
        const auto p = ptr.find(pt);
        if (p == ptr.end()) continue;
        const auto v = vec.find(p->second.second);
        if (v == vec.end() || v->second.first != t_f32 || v->second.second != 4) return fail("FragCoord is not a vec4 of floats");
        frag = var;
        v4f = p->second.second;
    }
    bool new_frag = false;
    if (!frag) {
        for (const auto& [id, v] : vec)
            if (v.first == t_f32 && v.second == 4) v4f = id;
        if (!v4f) {
            v4f = bound++;
            add(globals, mk(spv::OpTypeVector, {v4f, t_f32, 4}));
        }
        u32 p_in = 0;
        for (const auto& [id, p] : ptr)
            if (p.first == spv::ScInput && p.second == v4f) p_in = id;
        if (!p_in) {
            p_in = bound++;
            add(globals, mk(spv::OpTypePointer, {p_in, spv::ScInput, v4f}));
        }
        frag = bound++;
        add(globals, mk(spv::OpVariable, {p_in, frag, spv::ScInput}));
        add(decorations, mk(spv::OpDecorate, {frag, spv::DecBuiltIn, kBiFragCoord}));
        new_frag = true;
    }
    if (!t_bool) {
        t_bool = bound++;
        add(globals, mk(spv::OpTypeBool, {t_bool}));
    }
    const auto constant = [&](float f) {
        const u32 id = bound++;
        add(globals, mk(spv::OpConstant, {t_f32, id, f32_bits(f)}));
        return id;
    };
    const u32 c_sx = constant(1.0f / cut.scale_x), c_ox = constant(cut.offset_x), c_sy = constant(1.0f / cut.scale_y),
              c_oy = constant(cut.offset_y), c_tx = constant(cut.tan_x), c_ty = constant(cut.tan_y), c_dist = constant(cut.distance),
              c_sp = constant(cut.sin_pitch), c_cp = constant(cut.cos_pitch), c_radius = constant(cut.radius), c_widen = constant(cut.widen),
              c_height = constant(cut.height), c_front = constant(cut.distance - cut.margin),
              c_feather = constant(1.0f / (cut.feather > 0.01f ? cut.feather : 0.01f)),
              c_zero = constant(0.0f), c_one = constant(1.0f), c_n1 = constant(0.06711056f), c_n2 = constant(0.00583715f),
              c_dz = constant(cut.depth_z), c_dw = constant(cut.depth_w),
              c_n3 = constant(52.9829189f);
    if (!glsl) {
        glsl = bound++;
        std::vector<u32> name = {0, 0, 0, 0};
        const char* s = "GLSL.std.450";
        for (std::size_t k = 0; s[k]; ++k) name[k / 4] |= static_cast<u32>(static_cast<unsigned char>(s[k])) << (8 * (k % 4));
        imports.push_back(static_cast<u32>(2 + name.size()) << 16 | spv::OpExtInstImport);
        imports.push_back(glsl);
        imports.insert(imports.end(), name.begin(), name.end());
    }
    // The test, in a block of its own ahead of the function's first.
    std::vector<u32> head;
    const u32 new_label = bound++, kill_label = bound++, near_label = bound++, keep_label = bound++;
    add(head, mk(spv::OpLabel, {new_label}));
    for (std::size_t k = label_at + 1; k < vars_end; ++k)
        head.insert(head.end(), w.begin() + static_cast<std::ptrdiff_t>(ins[k].at), w.begin() + static_cast<std::ptrdiff_t>(ins[k].at + ins[k].n));
    const auto op2 = [&](u32 op, u32 a, u32 b) {
        const u32 id = bound++;
        add(head, mk(op, {t_f32, id, a, b}));
        return id;
    };
    const auto ext1 = [&](u32 f, u32 a) {
        const u32 id = bound++;
        add(head, mk(spv::OpExtInst, {t_f32, id, glsl, f, a}));
        return id;
    };
    const auto ext2 = [&](u32 f, u32 a, u32 b) {
        const u32 id = bound++;
        add(head, mk(spv::OpExtInst, {t_f32, id, glsl, f, a, b}));
        return id;
    };
    const u32 fc = bound++, fx = bound++, fy = bound++, fz = bound++;
    add(head, mk(spv::OpLoad, {v4f, fc, frag}));
    add(head, mk(spv::OpCompositeExtract, {t_f32, fx, fc, 0}));
    add(head, mk(spv::OpCompositeExtract, {t_f32, fy, fc, 1}));
    add(head, mk(spv::OpCompositeExtract, {t_f32, fz, fc, 2}));
    // view depth: the game's decode of its depth
    const u32 wv = op2(spv::OpFDiv, c_one, op2(spv::OpFAdd, op2(spv::OpFMul, fz, c_dz), c_dw));
    // Only a pixel in front of the point looked at can be cut: the rest (the
    // ground beyond the hunter, most of a frame) go straight on.
    const u32 m_front = op2(spv::OpFSub, c_front, wv), in_front = bound++;
    add(head, mk(spv::OpFOrdGreaterThan, {t_bool, in_front, m_front, c_zero}));
    add(head, mk(spv::OpSelectionMerge, {old_label, 0}));
    add(head, mk(spv::OpBranchConditional, {in_front, near_label, old_label}));
    add(head, mk(spv::OpLabel, {near_label}));
    const u32 nx = op2(spv::OpFMul, op2(spv::OpFSub, fx, c_ox), c_sx);           // -1..1 across
    const u32 ny = op2(spv::OpFMul, op2(spv::OpFSub, fy, c_oy), c_sy);           // -1..1 up
    const u32 vx = op2(spv::OpFMul, op2(spv::OpFMul, nx, c_tx), wv);
    const u32 vy = op2(spv::OpFMul, op2(spv::OpFMul, ny, c_ty), wv);
    const u32 r = ext1(spv::GlslSqrt, op2(spv::OpFAdd, op2(spv::OpFMul, vx, vx), op2(spv::OpFMul, vy, vy)));
    const u32 nearer = ext2(spv::GlslFMax, c_zero, op2(spv::OpFSub, c_dist, wv));
    const u32 reach = op2(spv::OpFAdd, c_radius, op2(spv::OpFMul, c_widen, nearer));
    const u32 m_across = op2(spv::OpFSub, reach, r);
    const u32 up = op2(spv::OpFSub, op2(spv::OpFMul, vy, c_cp), op2(spv::OpFMul, op2(spv::OpFSub, wv, c_dist), c_sp));
    const u32 m_up = op2(spv::OpFSub, up, c_height);
    const u32 m = ext2(spv::GlslFMin, ext2(spv::GlslFMin, m_across, m_up), m_front);
    const u32 t = ext2(spv::GlslFMax, c_zero, ext2(spv::GlslFMin, c_one, op2(spv::OpFMul, m, c_feather)));
    const u32 noise = ext1(kGlslFract, op2(spv::OpFMul, c_n3, ext1(kGlslFract, op2(spv::OpFAdd, op2(spv::OpFMul, c_n1, fx), op2(spv::OpFMul, c_n2, fy)))));
    const u32 kill = bound++;
    add(head, mk(spv::OpFOrdLessThan, {t_bool, kill, noise, t}));
    add(head, mk(spv::OpSelectionMerge, {keep_label, 0}));
    add(head, mk(spv::OpBranchConditional, {kill, kill_label, keep_label}));
    add(head, mk(spv::OpLabel, {kill_label}));
    if (w[1] >= 0x00010600u) add(head, mk(kOpTerminateInvocation, {}));
    else add(head, mk(spv::OpKill, {}));
    add(head, mk(spv::OpLabel, {keep_label}));
    add(head, mk(spv::OpBranch, {old_label}));

    std::vector<u32> out(w.begin(), w.begin() + 5);
    out[3] = bound;
    out.reserve(w.size() + globals.size() + imports.size() + decorations.size() + head.size() + 1);
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        if (last_import == ins.size() && k == memory_model) out.insert(out.end(), imports.begin(), imports.end());
        if (last_annot == ins.size() && k == first_global) out.insert(out.end(), decorations.begin(), decorations.end());
        if (k == first_fn) out.insert(out.end(), globals.begin(), globals.end());
        bool skip = false;
        for (std::size_t e : early) skip = skip || e == k;
        if (k > label_at && k < vars_end) skip = true;  // moved into the new first block
        if (k == label_at) out.insert(out.end(), head.begin(), head.end());
        if (skip) continue;
        if (k == entry_ins && new_frag) {
            out.push_back(static_cast<u32>(x.n + 1) << 16 | x.op);
            out.insert(out.end(), w.begin() + static_cast<std::ptrdiff_t>(x.at + 1), w.begin() + static_cast<std::ptrdiff_t>(x.at + x.n));
            out.push_back(frag);
        } else {
            out.insert(out.end(), w.begin() + static_cast<std::ptrdiff_t>(x.at), w.begin() + static_cast<std::ptrdiff_t>(x.at + x.n));
        }
        if (last_import != ins.size() && k == last_import) out.insert(out.end(), imports.begin(), imports.end());
        if (last_annot != ins.size() && k == last_annot) out.insert(out.end(), decorations.begin(), decorations.end());
    }
    spirv = std::move(out);
    return true;
}

std::string g_only_why;  // why the last pixel_cutaway_only came back empty (a test reads it)
std::vector<u32> pixel_cutaway_only(const PixelCut& cut) {
    // main(): an empty fragment stage, then the cutaway's test in front of it.
    const u32 main = 1, t_void = 2, t_fn = 3, label = 4;
    std::vector<u32> m = {0x07230203u, 0x00010000u, 0, 7, 0};
    const auto add = [&](const std::vector<u32>& i) { m.insert(m.end(), i.begin(), i.end()); };
    add(mk(spv::OpCapability, {spv::CapShader}));
    add(mk(spv::OpMemoryModel, {0, 1}));
    m.push_back(4u << 16 | spv::OpEntryPoint);  // EntryPoint Fragment %main "main"
    m.push_back(spv::EmFragment);
    m.push_back(main);
    m.push_back(0x6e69616du);  // "main"
    m.push_back(0);
    m[m.size() - 5] = 5u << 16 | spv::OpEntryPoint;
    add(mk(spv::OpExecutionMode, {main, spv::ExOriginUpperLeft}));
    add(mk(spv::OpTypeVoid, {t_void}));
    add(mk(spv::OpTypeFunction, {t_fn, t_void}));
    add(mk(spv::OpTypeFloat, {6, 32}));  // what the test computes in
    add(mk(spv::OpFunction, {t_void, main, 0, t_fn}));
    add(mk(spv::OpLabel, {label}));
    add(mk(spv::OpReturn, {}));
    add(mk(spv::OpFunctionEnd, {}));
    std::string why;
    if (!pixel_cutaway(m, cut, &why)) {
        g_only_why = why;
        return {};
    }
    return m;
}
