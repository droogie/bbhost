// PS1 affine texture mapping: a pixel stage's texture coordinates
// interpolated without the perspective (host/pixel_affine.h).
#include "host/pixel_affine.h"

#include "gcn/spirv.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using u32 = std::uint32_t;

struct Ins {
    std::size_t at;  // its first word in the module
    u32 op, n;
};

bool annotation(u32 op) { return op == spv::OpDecorate || op == spv::OpMemberDecorate || (op >= 73 && op <= 75) || op == 332 || op == 5632 || op == 5633; }

// A function body's instructions with no result type: no result at all, or
// (OpLabel) a result alone.
bool no_type(u32 op) {
    switch (op) {
    case spv::OpStore:
    case 63:              // OpCopyMemory
    case 8: case 317:     // OpLine, OpNoLine
    case 56:              // OpFunctionEnd
    case 99:              // OpImageWrite
    case 218: case 219:   // OpEmitVertex, OpEndPrimitive
    case 224: case 225:   // OpControlBarrier, OpMemoryBarrier
    case 228:             // OpAtomicStore
    case 246: case 247:   // OpLoopMerge, OpSelectionMerge
    case spv::OpLabel:
    case 249: case 250: case 251: case 252: case 253: case 254: case 255:  // the branches, returns, kill, unreachable
    case 256: case 257:   // OpLifetimeStart, OpLifetimeStop
    case 4416:            // OpTerminateInvocation
    case 5364: case 5365: // OpBegin/EndInvocationInterlockEXT
    case 5380:            // OpDemoteToHelperInvocation
        return true;
    default: return false;
    }
}

// The image instructions and where their coordinate is (the word after the
// sampled image or image); `projective`: OpImageSampleProj*, whose coordinate
// is divided by its last component.
bool image_coord(u32 op, bool* projective) {
    *projective = op >= 91 && op <= 94;
    return (op >= spv::OpImageSampleImplicitLod && op <= 94) || op == spv::OpImageFetch || op == spv::OpImageGather || op == spv::OpImageDrefGather ||
           op == spv::OpImageRead;
}

}  // namespace

bool pixel_affine(std::vector<u32>& spirv, std::string* why, u32* count) {
    const auto fail = [&](const char* w) {
        if (why) *why = w;
        return false;
    };
    const std::vector<u32>& w = spirv;
    if (w.size() < 5 || w[0] != 0x07230203u) return fail("not a SPIR-V module");
    const u32 bound = w[3];
    std::vector<Ins> ins;
    for (std::size_t i = 5; i < w.size();) {
        const u32 n = w[i] >> 16;
        if (!n || i + n > w.size()) return fail("a malformed instruction");
        ins.push_back({i, w[i] & 0xffff, n});
        i += n;
    }
    // The module's inputs and what they are.
    u32 model = ~0u, entries = 0, t_f32 = 0;
    std::set<u32> skip;                       // built-in, flat or already linear
    std::map<u32, u32> ptr_pointee;           // Input pointer type -> pointee
    std::map<u32, u32> vec_of, arr_of;        // vector / array type -> its element
    std::vector<std::pair<u32, u32>> inputs;  // (variable, pointer type)
    std::size_t first_fn = ins.size(), last_annot = ins.size();
    for (std::size_t k = 0; k < ins.size(); ++k) {
        const Ins& x = ins[k];
        const u32* a = &w[x.at];
        if (x.op == spv::OpFunction && first_fn == ins.size()) first_fn = k;
        if (first_fn != ins.size()) break;
        if (annotation(x.op)) last_annot = k;
        switch (x.op) {
        case spv::OpEntryPoint:
            if (x.n < 4) return fail("a malformed entry point");
            ++entries;
            model = a[1];
            break;
        case spv::OpDecorate:
            if (x.n >= 3 && (a[2] == spv::DecBuiltIn || a[2] == spv::DecFlat || a[2] == spv::DecNoPerspective)) skip.insert(a[1]);
            break;
        case spv::OpTypeFloat:
            if (x.n >= 3 && a[2] == 32 && !t_f32) t_f32 = a[1];
            break;
        case spv::OpTypeVector:
            if (x.n >= 4) vec_of[a[1]] = a[2];
            break;
        case spv::OpTypeArray:
            if (x.n >= 4) arr_of[a[1]] = a[2];
            break;
        case spv::OpTypePointer:
            if (x.n >= 4 && a[2] == spv::ScInput) ptr_pointee[a[1]] = a[3];
            break;
        case spv::OpVariable:
            if (x.n >= 4 && a[3] == spv::ScInput) inputs.push_back({a[2], a[1]});
            break;
        default: break;
        }
    }
    if (entries != 1) return fail("not one entry point");
    if (model != spv::EmFragment) return fail("not a pixel stage");
    if (first_fn == ins.size()) return fail("no function");
    if (last_annot == ins.size()) return fail("no decorations");
    // The inputs that may be made linear: floats (a scalar, a vector, an
    // array of either) that are not built-ins, flat or linear already.
    const auto is_float = [&](u32 t) {
        if (const auto r = arr_of.find(t); r != arr_of.end()) t = r->second;
        if (const auto v = vec_of.find(t); v != vec_of.end()) t = v->second;
        return t && t == t_f32;
    };
    std::vector<u32> vars;
    for (const auto& [var, ptype] : inputs) {
        const auto p = ptr_pointee.find(ptype);
        if (skip.count(var) || p == ptr_pointee.end() || !is_float(p->second)) continue;
        vars.push_back(var);
    }
    if (vars.empty()) return fail("no interpolated input");
    if (vars.size() > 64) return fail("more than 64 inputs");
    // Each id's inputs (a bit each): `from` those its value came from,
    // `divided` those it was divided by on the way. A pointer's are what was
    // stored through it, kept on the variable its access chains start from.
    std::vector<std::uint64_t> from(bound, 0), divided(bound, 0);
    std::vector<u32> root(bound, 0);
    for (std::size_t v = 0; v < vars.size(); ++v) {
        if (vars[v] >= bound) return fail("an id past the bound");
        from[vars[v]] = std::uint64_t{1} << v;
    }
    std::uint64_t coord = 0, projective = 0;
    const auto id = [&](u32 x) { return x < bound ? x : 0u; };
    for (bool changed = true; changed;) {
        changed = false;
        const auto add = [&](std::vector<std::uint64_t>& m, u32 x, std::uint64_t bits) {
            if (x && (m[x] | bits) != m[x]) {
                m[x] |= bits;
                changed = true;
            }
        };
        for (std::size_t k = first_fn; k < ins.size(); ++k) {
            const Ins& x = ins[k];
            const u32* a = &w[x.at];
            if (x.op == spv::OpStore || x.op == 63 /* OpCopyMemory */) {
                if (x.n < 3) continue;
                const u32 dst = id(a[1]), src = id(a[2]);
                const std::uint64_t f = from[src], d = divided[src];
                for (const u32 p : {dst, dst ? root[dst] : 0u}) {
                    add(from, p, f);
                    add(divided, p, d);
                }
                continue;
            }
            if (no_type(x.op) || x.n < 3) continue;
            const u32 result = id(a[2]);
            if (!result) continue;
            // The operands that are ids: all of them but where literals are.
            std::size_t lo = 3, hi = x.n;
            if (x.op == spv::OpCompositeExtract) hi = std::min<std::size_t>(x.n, 4);
            else if (x.op == spv::OpCompositeInsert || x.op == spv::OpVectorShuffle) hi = std::min<std::size_t>(x.n, 5);
            else if (x.op == spv::OpExtInst) lo = 5;
            bool proj = false;
            if (image_coord(x.op, &proj)) {
                // The coordinate's inputs are texture coordinates - or, divided
                // by an input on the way (or projective), a screen position.
                // The result carries none of them.
                if (x.n < 5) continue;
                const u32 c = id(a[4]);
                coord |= from[c];
                projective |= divided[c] | (proj ? from[c] : 0);
                continue;
            }
            std::uint64_t f = 0, d = 0;
            for (std::size_t j = lo; j < hi; ++j) {
                const u32 o = id(a[j]);
                f |= from[o];
                d |= divided[o];
            }
            if (x.op == spv::OpFDiv && x.n >= 5) d |= from[id(a[4])];
            if ((x.op == spv::OpAccessChain || x.op == 66 /* OpInBoundsAccessChain */ || x.op == 67 /* OpPtrAccessChain */) && x.n >= 4) {
                const u32 b = id(a[3]);
                root[result] = b && root[b] ? root[b] : b;
            }
            add(from, result, f);
            add(divided, result, d);
        }
    }
    const std::uint64_t linear = coord & ~projective;
    if (!linear) return fail(coord ? "its texture coordinates are screen positions" : "no texture coordinate from an input");

    // The decorations after the last annotation.
    std::vector<u32> dec;
    u32 made = 0;
    for (std::size_t v = 0; v < vars.size(); ++v) {
        if (!(linear >> v & 1)) continue;
        dec.push_back(3u << 16 | spv::OpDecorate);
        dec.push_back(vars[v]);
        dec.push_back(spv::DecNoPerspective);
        ++made;
    }
    const std::size_t after = ins[last_annot].at + ins[last_annot].n;
    spirv.insert(spirv.begin() + static_cast<std::ptrdiff_t>(after), dec.begin(), dec.end());
    if (count) *count = made;
    return true;
}
