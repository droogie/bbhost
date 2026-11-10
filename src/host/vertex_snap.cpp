// PS1 vertex snapping: a stage's clip-space position moved onto a grid as it
// leaves the stage (host/vertex_snap.h).
#include "host/vertex_snap.h"

#include "gcn/spirv.h"

#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using u32 = std::uint32_t;

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

bool vertex_snap(std::vector<u32>& spirv, float cells_x, float cells_y, std::string* why) {
    const auto fail = [&](const char* w) {
        if (why) *why = w;
        return false;
    };
    const std::vector<u32>& w = spirv;
    if (w.size() < 5 || w[0] != 0x07230203u) return fail("not a SPIR-V module");
    if (!(cells_x >= 1.0f) || !(cells_y >= 1.0f)) return fail("no grid");
    std::vector<Ins> ins;
    for (std::size_t i = 5; i < w.size();) {
        const u32 n = w[i] >> 16;
        if (!n || i + n > w.size()) return fail("a malformed instruction");
        ins.push_back({i, w[i] & 0xffff, n});
        i += n;
    }
    // What the module has: its entry point, the GLSL instructions, the
    // position's variable or block member, the types the snap needs.
    u32 model = ~0u, entry = 0, glsl = 0, t_f32 = 0, t_bool = 0, t_int = 0, entries = 0;
    std::map<u32, u32> builtin;                    // variable -> BuiltIn
    std::map<std::pair<u32, u32>, u32> member_bi;  // (struct, member) -> BuiltIn
    std::map<u32, std::pair<u32, u32>> vec;        // vector type -> (component, count)
    std::map<u32, std::pair<u32, u32>> ptr;        // pointer type -> (storage class, pointee)
    std::map<u32, u32> out_var;                    // an Output variable -> its pointer type
    std::size_t first_fn = ins.size(), last_annot = ins.size(), first_global = ins.size(), last_import = ins.size(), memory_model = ins.size();
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        const u32* a = &w[x.at];
        if (x.op == spv::OpFunction && first_fn == ins.size()) first_fn = k;
        if (first_fn != ins.size()) continue;
        if (annotation(x.op)) last_annot = k;
        if (!preamble(x.op) && x.op != 8 /* OpLine */ && x.op != 317 /* OpNoLine */ && first_global == ins.size()) first_global = k;
        switch (x.op) {
        case spv::OpEntryPoint:
            if (x.n < 4) return fail("a malformed entry point");
            ++entries;
            model = a[1];
            entry = a[2];
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
        case spv::OpTypeBool: t_bool = a[1]; break;
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
    if (first_fn == ins.size() || memory_model == ins.size() || first_global == ins.size()) return fail("no function");
    if (!t_f32) return fail("no 32-bit float type");
    const auto is_v4f = [&](u32 t) {
        const auto v = vec.find(t);
        return v != vec.end() && v->second.first == t_f32 && v->second.second == 4;
    };
    // The position: a variable of its own, or a member of the output block.
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
    u32 v4f_out_ptr = 0;
    if (!pos_var) {
        // The block's member type, from the struct's declaration.
        u32 block_struct = 0;
        if (const auto p = ptr.find(out_var[block_var]); p != ptr.end()) block_struct = p->second.second;
        for (const Ins& x : ins) {
            if (x.op != spv::OpTypeStruct || w[x.at + 1] != block_struct) continue;
            if (2 + member >= x.n) return fail("the output block has no such member");
            v4f = w[x.at + 2 + member];
        }
        if (!is_v4f(v4f)) return fail("the position is not a vec4 of 32-bit floats (or the output block is an array)");
        for (const auto& [id, p] : ptr) {
            if (p.first == spv::ScOutput && p.second == v4f) v4f_out_ptr = id;
        }
    }
    // Where it is final: before each return of the entry point, or each emit.
    std::vector<std::size_t> at;  // instruction indices
    {
        bool in_entry = false;
        for (std::size_t k = first_fn; k < ins.size(); ++k) {
            const Ins& x = ins[k];
            if (x.op == spv::OpFunction) in_entry = x.n >= 3 && w[x.at + 2] == entry;
            if (model == spv::EmGeometry ? x.op == spv::OpEmitVertex : (in_entry && x.op == spv::OpReturn)) at.push_back(k);
        }
    }
    if (at.empty()) return fail(model == spv::EmGeometry ? "no emit" : "no return from the entry point");

    // New ids: the types and constants the snap needs, then each site's values.
    u32 bound = w[3];
    std::vector<u32> globals, imports, decorations;
    if (!t_bool) {
        t_bool = bound++;
        const std::vector<u32> t = mk(spv::OpTypeBool, {t_bool});
        globals.insert(globals.end(), t.begin(), t.end());
    }
    u32 c_member = 0;
    if (!pos_var) {
        if (!t_int) {
            t_int = bound++;
            const std::vector<u32> t = mk(spv::OpTypeInt, {t_int, 32, 1});
            globals.insert(globals.end(), t.begin(), t.end());
        }
        if (!v4f_out_ptr) {
            v4f_out_ptr = bound++;
            const std::vector<u32> t = mk(spv::OpTypePointer, {v4f_out_ptr, spv::ScOutput, v4f});
            globals.insert(globals.end(), t.begin(), t.end());
        }
        c_member = bound++;
        const std::vector<u32> c = mk(spv::OpConstant, {t_int, c_member, member});
        globals.insert(globals.end(), c.begin(), c.end());
    }
    const auto constant = [&](float f) {
        const u32 id = bound++;
        const std::vector<u32> c = mk(spv::OpConstant, {t_f32, id, f32_bits(f)});
        globals.insert(globals.end(), c.begin(), c.end());
        return id;
    };
    const u32 c_hx = constant(cells_x * 0.5f), c_hy = constant(cells_y * 0.5f), c_half = constant(0.5f);
    const u32 c_eps = constant(1e-6f), c_far = constant(8388608.0f);  // 2^23: no fraction left to snap
    if (!glsl) {
        glsl = bound++;
        std::vector<u32> name = {0, 0, 0, 0};
        const char* s = "GLSL.std.450";
        for (std::size_t k = 0; s[k]; ++k) name[k / 4] |= static_cast<u32>(static_cast<unsigned char>(s[k])) << (8 * (k % 4));
        imports.push_back(static_cast<u32>(2 + name.size()) << 16 | spv::OpExtInstImport);
        imports.push_back(glsl);
        imports.insert(imports.end(), name.begin(), name.end());
    }
    const auto snap_code = [&]() {
        std::vector<u32> c;
        const auto emit = [&](std::vector<u32> i) { c.insert(c.end(), i.begin(), i.end()); };
        const auto exact = [&](u32 id) {  // NoContraction
            const std::vector<u32> d = mk(spv::OpDecorate, {id, spv::DecNoContraction});
            decorations.insert(decorations.end(), d.begin(), d.end());
            return id;
        };
        u32 pp = pos_var;
        if (!pp) {
            pp = bound++;
            emit(mk(spv::OpAccessChain, {v4f_out_ptr, pp, block_var, c_member}));
        }
        const u32 p = bound++, x = bound++, y = bound++, wv = bound++;
        emit(mk(spv::OpLoad, {v4f, p, pp}));
        emit(mk(spv::OpCompositeExtract, {t_f32, x, p, 0}));
        emit(mk(spv::OpCompositeExtract, {t_f32, y, p, 1}));
        emit(mk(spv::OpCompositeExtract, {t_f32, wv, p, 3}));
        const u32 ok_w = bound++;
        emit(mk(spv::OpFOrdGreaterThan, {t_bool, ok_w, wv, c_eps}));
        u32 snapped[2] = {};
        const u32 xy[2] = {x, y}, half_cells[2] = {c_hx, c_hy};
        for (int k = 0; k < 2; ++k) {
            const u32 s = exact(bound++), g = exact(bound++), t = exact(bound++), h = exact(bound++), f = bound++, c = exact(bound++),
                      q = exact(bound++);
            const u32 mag = bound++, near = bound++, ok = bound++, out = bound++;
            emit(mk(spv::OpFDiv, {t_f32, s, half_cells[k], wv}));           // cells/2 / w
            emit(mk(spv::OpFMul, {t_f32, g, xy[k], s}));                    // cells from the middle
            emit(mk(spv::OpFAdd, {t_f32, t, g, half_cells[k]}));            // cells from the edge
            emit(mk(spv::OpFAdd, {t_f32, h, t, c_half}));
            emit(mk(spv::OpExtInst, {t_f32, f, glsl, spv::GlslFloor, h}));  // the nearest corner
            emit(mk(spv::OpFSub, {t_f32, c, f, half_cells[k]}));
            emit(mk(spv::OpFDiv, {t_f32, q, c, s}));                        // back to clip space
            emit(mk(spv::OpExtInst, {t_f32, mag, glsl, spv::GlslFAbs, g}));
            emit(mk(spv::OpFOrdLessThan, {t_bool, near, mag, c_far}));
            emit(mk(spv::OpLogicalAnd, {t_bool, ok, ok_w, near}));
            emit(mk(spv::OpSelect, {t_f32, out, ok, q, xy[k]}));
            snapped[k] = out;
        }
        const u32 p1 = bound++, p2 = bound++;
        emit(mk(spv::OpCompositeInsert, {v4f, p1, snapped[0], p, 0}));
        emit(mk(spv::OpCompositeInsert, {v4f, p2, snapped[1], p1, 1}));
        emit(mk(spv::OpStore, {pp, p2}));
        return c;
    };
    std::vector<std::vector<u32>> code;
    for (std::size_t k = 0; k < at.size(); ++k) code.push_back(snap_code());

    // The module again with them in: the import after the last import (or
    // before the memory model), the decorations after the last annotation (or
    // before the first type), the types and constants before the first
    // function, each site's code before its return or emit.
    std::vector<u32> out(w.begin(), w.begin() + 5);
    out[3] = bound;
    out.reserve(w.size() + globals.size() + imports.size() + decorations.size() + code.size() * code[0].size());
    const std::size_t import_after = last_import, annot_after = last_annot;
    std::size_t site = 0;
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        if (import_after == ins.size() && k == memory_model) out.insert(out.end(), imports.begin(), imports.end());
        if (annot_after == ins.size() && k == first_global) out.insert(out.end(), decorations.begin(), decorations.end());
        if (k == first_fn) out.insert(out.end(), globals.begin(), globals.end());
        if (site < at.size() && at[site] == k) {
            out.insert(out.end(), code[site].begin(), code[site].end());
            ++site;
        }
        out.insert(out.end(), w.begin() + static_cast<std::ptrdiff_t>(x.at), w.begin() + static_cast<std::ptrdiff_t>(x.at + x.n));
        if (import_after != ins.size() && k == import_after) out.insert(out.end(), imports.begin(), imports.end());
        if (annot_after != ins.size() && k == annot_after) out.insert(out.end(), decorations.begin(), decorations.end());
    }
    spirv = std::move(out);
    return true;
}

