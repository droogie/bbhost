// A geometry-shader draw's ES and GS lifts spliced into one vertex stage, and
// the GS lift with its emits recorded for the check mode (gcn/gs_vertex.h).
#include "gcn/gs_vertex.h"

#include "gcn/spirv.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace gcn {
namespace {

using u32 = std::uint32_t;

struct Ins {
    std::vector<u32> w;  // the header word first
    std::uint16_t op() const { return static_cast<std::uint16_t>(w[0] & 0xffff); }
};

Ins mk(std::uint16_t op, const std::vector<u32>& operands) {
    Ins in;
    in.w.reserve(operands.size() + 1);
    in.w.push_back((static_cast<u32>(operands.size() + 1) << 16) | op);
    in.w.insert(in.w.end(), operands.begin(), operands.end());
    return in;
}

std::vector<u32> str_words(const std::string& s) {
    std::vector<u32> w(s.size() / 4 + 1, 0);
    for (std::size_t k = 0; k < s.size(); ++k) w[k / 4] |= static_cast<u32>(static_cast<unsigned char>(s[k])) << (8 * (k % 4));
    return w;
}

std::string word_str(const std::vector<u32>& w, std::size_t from) {
    std::string s;
    for (std::size_t k = from; k < w.size(); ++k) {
        for (int b = 0; b < 4; ++b) {
            const char c = static_cast<char>((w[k] >> (8 * b)) & 0xff);
            if (!c) return s;
            s += c;
        }
    }
    return s;
}

// A module in its logical layout's sections.
struct Mod {
    u32 bound = 0;
    std::vector<Ins> caps, exts, imports, model, entries, modes, debug, annots, globals;
    std::vector<std::vector<Ins>> functions;  // OpFunction .. OpFunctionEnd each
};

bool parse(const std::vector<u32>& s, Mod& m, std::string& why) {
    if (s.size() < 5 || s[0] != 0x07230203u) return why = "not a SPIR-V module", false;
    m.bound = s[3];
    bool in_fn = false;
    for (std::size_t i = 5; i < s.size();) {
        const u32 n = s[i] >> 16;
        if (!n || i + n > s.size()) return why = "a malformed instruction", false;
        Ins in;
        in.w.assign(s.begin() + static_cast<std::ptrdiff_t>(i), s.begin() + static_cast<std::ptrdiff_t>(i + n));
        i += n;
        const std::uint16_t op = in.op();
        if (op == spv::OpFunction) {
            m.functions.emplace_back();
            in_fn = true;
        }
        if (in_fn) {
            m.functions.back().push_back(std::move(in));
            if (op == spv::OpFunctionEnd) in_fn = false;
            continue;
        }
        switch (op) {
        case spv::OpCapability: m.caps.push_back(std::move(in)); break;
        case spv::OpExtension: m.exts.push_back(std::move(in)); break;
        case spv::OpExtInstImport: m.imports.push_back(std::move(in)); break;
        case spv::OpMemoryModel: m.model.push_back(std::move(in)); break;
        case spv::OpEntryPoint: m.entries.push_back(std::move(in)); break;
        case spv::OpExecutionMode: case 331: m.modes.push_back(std::move(in)); break;
        case 2: case 3: case 4: case spv::OpName: case 6: case 7: case 330: m.debug.push_back(std::move(in)); break;
        case spv::OpDecorate: case spv::OpMemberDecorate: case 73: case 74: case 75: case 332: m.annots.push_back(std::move(in)); break;
        default: m.globals.push_back(std::move(in)); break;
        }
    }
    if (m.entries.size() != 1) return why = "not one entry point", false;
    return true;
}

// Which words of an instruction (1..n-1) are ids. False for an opcode this
// splice does not know: the draw keeps its GS.
bool id_words(const Ins& in, std::vector<bool>& id) {
    const std::size_t n = in.w.size();
    id.assign(n, false);
    const auto ids = [&](std::size_t from, std::size_t to) {
        for (std::size_t k = from; k < std::min(to, n); ++k) id[k] = true;
    };
    const auto image = [&](std::size_t mandatory) {  // ids through `mandatory`, then an image-operands mask, then ids
        ids(1, mandatory + 1);
        ids(mandatory + 2, n);
    };
    switch (in.op()) {
    case spv::OpNop: case spv::OpReturn: case spv::OpKill: case spv::OpUnreachable: case spv::OpEmitVertex: case spv::OpEndPrimitive:
    case spv::OpDemoteToHelperInvocation:
        return true;
    case spv::OpTypeVoid: case spv::OpTypeBool: case spv::OpTypeSampler: case spv::OpLabel: case spv::OpExtInstImport: case spv::OpName:
    case 6 /* OpMemberName */: case spv::OpDecorate: case spv::OpMemberDecorate: case spv::OpBranch:
        ids(1, 2);
        return true;
    case spv::OpTypeInt: case spv::OpTypeFloat:
        ids(1, 2);
        return true;
    case spv::OpTypeVector: case 24 /* OpTypeMatrix */: case spv::OpTypeImage:
        ids(1, 3);
        return true;
    case spv::OpTypePointer:
        id[1] = true;
        if (n > 3) id[3] = true;
        return true;
    case spv::OpVariable:
        ids(1, 3);
        if (n > 4) id[4] = true;
        return true;
    case spv::OpConstant: case spv::OpSpecConstant:
        ids(1, 3);
        return true;
    case spv::OpLoad: case spv::OpCompositeExtract: case spv::OpArrayLength: case spv::OpBranchConditional:
        ids(1, 4);
        return true;
    case spv::OpStore: case spv::OpLoopMerge:
        ids(1, 3);
        return true;
    case spv::OpSelectionMerge:
        ids(1, 2);
        return true;
    case spv::OpCompositeInsert: case spv::OpVectorShuffle:
        ids(1, 5);
        return true;
    case spv::OpExtInst: case spv::OpGroupNonUniformBallotBitCount:
        ids(1, 4);
        ids(5, n);
        return true;
    case spv::OpFunction:
        ids(1, 3);
        if (n > 4) id[4] = true;
        return true;
    case spv::OpImageSampleImplicitLod: case spv::OpImageSampleExplicitLod: case spv::OpImageFetch: case spv::OpImageRead:
        image(4);
        return true;
    case spv::OpImageSampleDrefImplicitLod: case spv::OpImageSampleDrefExplicitLod: case spv::OpImageGather: case spv::OpImageDrefGather:
        image(5);
        return true;
    case spv::OpImageWrite:
        image(3);
        return true;
    // Every word an id: results, operands, labels.
    case spv::OpUndef: case spv::OpTypeSampledImage: case spv::OpTypeArray: case spv::OpTypeRuntimeArray: case spv::OpTypeStruct:
    case spv::OpTypeFunction: case spv::OpConstantTrue: case spv::OpConstantFalse: case spv::OpConstantComposite: case spv::OpConstantNull:
    case spv::OpFunctionParameter: case spv::OpFunctionCall: case spv::OpAccessChain: case 66 /* OpInBoundsAccessChain */:
    case 67 /* OpPtrAccessChain */: case spv::OpCompositeConstruct: case 83 /* OpCopyObject */: case spv::OpSampledImage:
    case 100 /* OpImage */: case spv::OpImageQuerySizeLod: case spv::OpImageQuerySize: case spv::OpImageQueryLod: case spv::OpImageQueryLevels:
    case spv::OpConvertFToU: case spv::OpConvertFToS: case spv::OpConvertSToF: case spv::OpConvertUToF: case spv::OpUConvert:
    case spv::OpSConvert: case spv::OpFConvert: case spv::OpConvertUToPtr: case 117 /* OpConvertPtrToU */: case spv::OpBitcast:
    case spv::OpSNegate: case spv::OpFNegate: case spv::OpIAdd: case spv::OpFAdd: case spv::OpISub: case spv::OpFSub: case spv::OpIMul:
    case spv::OpFMul: case spv::OpUDiv: case spv::OpSDiv: case spv::OpFDiv: case spv::OpUMod: case spv::OpSRem: case spv::OpSMod:
    case 140 /* OpFRem */: case 141 /* OpFMod */: case 142: case 143: case 144: case 145: case 146: case 147: case 148:
    case spv::OpIAddCarry: case spv::OpISubBorrow: case spv::OpUMulExtended: case spv::OpSMulExtended: case spv::OpAny: case spv::OpAll:
    case spv::OpIsNan: case spv::OpIsInf: case spv::OpLogicalEqual: case spv::OpLogicalNotEqual: case spv::OpLogicalOr:
    case spv::OpLogicalAnd: case spv::OpLogicalNot: case spv::OpSelect: case spv::OpIEqual: case spv::OpINotEqual: case spv::OpUGreaterThan:
    case spv::OpSGreaterThan: case spv::OpUGreaterThanEqual: case spv::OpSGreaterThanEqual: case spv::OpULessThan: case spv::OpSLessThan:
    case spv::OpULessThanEqual: case spv::OpSLessThanEqual: case spv::OpFOrdEqual: case spv::OpFUnordEqual: case spv::OpFOrdNotEqual:
    case spv::OpFUnordNotEqual: case spv::OpFOrdLessThan: case spv::OpFUnordLessThan: case spv::OpFOrdGreaterThan:
    case spv::OpFUnordGreaterThan: case spv::OpFOrdLessThanEqual: case spv::OpFUnordLessThanEqual: case spv::OpFOrdGreaterThanEqual:
    case spv::OpFUnordGreaterThanEqual: case spv::OpShiftRightLogical: case spv::OpShiftRightArithmetic: case spv::OpShiftLeftLogical:
    case spv::OpBitwiseOr: case spv::OpBitwiseXor: case spv::OpBitwiseAnd: case spv::OpNot: case spv::OpBitFieldInsert:
    case spv::OpBitFieldSExtract: case spv::OpBitFieldUExtract: case spv::OpBitReverse: case spv::OpBitCount: case spv::OpDPdx: case spv::OpDPdy:
    case spv::OpControlBarrier: case spv::OpMemoryBarrier: case spv::OpAtomicLoad: case spv::OpAtomicStore: case spv::OpAtomicExchange:
    case spv::OpAtomicCompareExchange: case spv::OpAtomicIIncrement: case spv::OpAtomicIDecrement: case spv::OpAtomicIAdd: case spv::OpAtomicISub:
    case spv::OpAtomicSMin: case spv::OpAtomicUMin: case spv::OpAtomicSMax: case spv::OpAtomicUMax: case spv::OpAtomicAnd: case spv::OpAtomicOr:
    case spv::OpAtomicXor: case spv::OpPhi: case spv::OpReturnValue: case spv::OpGroupNonUniformElect: case spv::OpGroupNonUniformAll:
    case spv::OpGroupNonUniformAny: case spv::OpGroupNonUniformBroadcast: case spv::OpGroupNonUniformBroadcastFirst:
    case spv::OpGroupNonUniformBallot: case spv::OpGroupNonUniformShuffle: case spv::OpGroupNonUniformShuffleXor:
    case spv::OpGroupNonUniformQuadBroadcast: case spv::OpGroupNonUniformQuadSwap:
        ids(1, n);
        return true;
    default:
        return false;
    }
}

struct Splice {
    Mod g, e;
    std::string why;
    u32 next = 0;  // the merged module's next free id
    std::vector<Ins> added;  // new types, constants and variables, after the modules' own globals
    std::map<std::vector<u32>, u32> made;  // (opcode, operands but the result) -> id, for find-or-make

    bool fail(const std::string& w) {
        if (why.empty()) why = w;
        return false;
    }
    u32 fresh() { return next++; }
    // A type or constant: the one either module has, or a new one.
    u32 have(std::uint16_t op, std::vector<u32> operands_after_result, bool result_type_first = false) {
        std::vector<u32> key{op};
        key.insert(key.end(), operands_after_result.begin(), operands_after_result.end());
        key.push_back(result_type_first ? 1u : 0u);
        if (const auto it = made.find(key); it != made.end()) return it->second;
        const u32 id = fresh();
        std::vector<u32> ops;
        if (result_type_first) {
            ops.push_back(operands_after_result[0]);
            ops.push_back(id);
            ops.insert(ops.end(), operands_after_result.begin() + 1, operands_after_result.end());
        } else {
            ops.push_back(id);
            ops.insert(ops.end(), operands_after_result.begin(), operands_after_result.end());
        }
        added.push_back(mk(op, ops));
        return made[key] = id;
    }
    void know(const Ins& in) {  // a type or constant already declared: find-or-make finds it
        const std::uint16_t op = in.op();
        if (op == spv::OpTypePointer || op == spv::OpTypeInt || op == spv::OpTypeFloat || op == spv::OpTypeVector || op == spv::OpTypeBool ||
            op == spv::OpTypeVoid) {
            std::vector<u32> key{op};
            key.insert(key.end(), in.w.begin() + 2, in.w.end());
            key.push_back(0u);
            made.emplace(key, in.w[1]);
        } else if (op == spv::OpConstant || op == spv::OpConstantTrue || op == spv::OpConstantFalse || op == spv::OpConstantNull ||
                   op == spv::OpConstantComposite) {
            std::vector<u32> key{op, in.w[1]};
            key.insert(key.end(), in.w.begin() + 3, in.w.end());
            key.push_back(1u);
            made.emplace(key, in.w[2]);
        }
    }
    u32 t_u32() { return have(spv::OpTypeInt, {32, 0}); }
    u32 t_u64() { return have(spv::OpTypeInt, {64, 0}); }
    u32 t_bool() { return have(spv::OpTypeBool, {}); }
    u32 t_f32() { return have(spv::OpTypeFloat, {32}); }
    u32 t_v4f() { return have(spv::OpTypeVector, {t_f32(), 4}); }
    u32 ptr(u32 sc, u32 type) { return have(spv::OpTypePointer, {sc, type}); }
    u32 c_u32(u32 v) { return have(spv::OpConstant, {t_u32(), v}, true); }
    u32 c_u64(std::uint64_t v) { return have(spv::OpConstant, {t_u64(), static_cast<u32>(v), static_cast<u32>(v >> 32)}, true); }
    u32 c_f32(float f) {
        u32 b;
        static_assert(sizeof b == sizeof f);
        std::memcpy(&b, &f, 4);
        return have(spv::OpConstant, {t_f32(), b}, true);
    }
    u32 c_true() { return have(spv::OpConstantTrue, {t_bool()}, true); }
    u32 c_null(u32 type) { return have(spv::OpConstantNull, {type}, true); }
    u32 private_var(u32 type, const char* name, std::vector<Ins>& names) {
        const u32 id = fresh();
        added.push_back(mk(spv::OpVariable, {ptr(spv::ScPrivate, type), id, spv::ScPrivate, c_null(type)}));
        names.push_back(mk(spv::OpName, [&] {
            std::vector<u32> w{id};
            const std::vector<u32> s = str_words(name);
            w.insert(w.end(), s.begin(), s.end());
            return w;
        }()));
        return id;
    }
    // An instruction with a result into `out`, its id the splice's.
    u32 op(std::vector<Ins>& out, std::uint16_t opc, u32 type, const std::vector<u32>& args) {
        const u32 id = fresh();
        std::vector<u32> w{type, id};
        w.insert(w.end(), args.begin(), args.end());
        out.push_back(mk(opc, w));
        return id;
    }
    void vop(std::vector<Ins>& out, std::uint16_t opc, const std::vector<u32>& args) { out.push_back(mk(opc, args)); }
};

// Guest-visible words through physical storage buffer pointers: the draw's
// arguments and its check record.
struct Psb {
    Splice& s;
    u32 t_u32, t_u64, p_u32, p_u64;
    explicit Psb(Splice& sp)
        : s(sp), t_u32(sp.t_u32()), t_u64(sp.t_u64()), p_u32(sp.ptr(spv::ScPhysicalStorageBuffer, t_u32)),
          p_u64(sp.ptr(spv::ScPhysicalStorageBuffer, t_u64)) {}
    u32 at(std::vector<Ins>& out, u32 base, std::uint64_t bytes) { return bytes ? s.op(out, spv::OpIAdd, t_u64, {base, s.c_u64(bytes)}) : base; }
    // base + 4 * idx (a u32 index of words).
    u32 word(std::vector<Ins>& out, u32 base, u32 idx) {
        const u32 wide = s.op(out, spv::OpUConvert, t_u64, {idx});
        return s.op(out, spv::OpIAdd, t_u64, {base, s.op(out, spv::OpShiftLeftLogical, t_u64, {wide, s.c_u32(2)})});
    }
    u32 load32(std::vector<Ins>& out, u32 addr) { return s.op(out, spv::OpLoad, t_u32, {s.op(out, spv::OpConvertUToPtr, p_u32, {addr}), 2 /* Aligned */, 4}); }
    u32 load64(std::vector<Ins>& out, u32 addr) { return s.op(out, spv::OpLoad, t_u64, {s.op(out, spv::OpConvertUToPtr, p_u64, {addr}), 2, 8}); }
    void store32(std::vector<Ins>& out, u32 addr, u32 value) { s.vop(out, spv::OpStore, {s.op(out, spv::OpConvertUToPtr, p_u32, {addr}), value, 2, 4}); }
};

// The ids an OpName names `name` in a module's debug section.
std::vector<u32> named(const Mod& m, const char* name) {
    std::vector<u32> out;
    for (const Ins& in : m.debug) {
        if (in.op() == spv::OpName && in.w.size() >= 3 && word_str(in.w, 2) == name) out.push_back(in.w[1]);
    }
    return out;
}
// Variables of a storage class with their decorations: builtin (or ~0u), location (or ~0u).
struct Var {
    u32 id = 0, type = 0, sc = 0, builtin = ~0u, location = ~0u;
};
std::map<u32, Var> variables(const Mod& m) {
    std::map<u32, Var> out;
    for (const Ins& in : m.globals) {
        if (in.op() == spv::OpVariable && in.w.size() >= 4) out[in.w[2]] = {in.w[2], in.w[1], in.w[3], ~0u, ~0u};
    }
    for (const Ins& in : m.annots) {
        if (in.op() != spv::OpDecorate || in.w.size() < 4) continue;
        const auto it = out.find(in.w[1]);
        if (it == out.end()) continue;
        if (in.w[2] == spv::DecBuiltIn) it->second.builtin = in.w[3];
        if (in.w[2] == spv::DecLocation) it->second.location = in.w[3];
    }
    return out;
}
// The type or constant declaring `id` among a module's globals.
const Ins* type_def(const Mod& m, u32 id) {
    for (const Ins& in : m.globals) {
        const std::uint16_t op = in.op();
        if (op >= spv::OpTypeVoid && op <= spv::OpTypeFunction && in.w.size() > 1 && in.w[1] == id) return &in;
    }
    return nullptr;
}
const Ins* constant_def(const Mod& m, u32 id) {
    for (const Ins& in : m.globals) {
        if (in.op() == spv::OpConstant && in.w.size() > 3 && in.w[2] == id) return &in;
    }
    return nullptr;
}
// The pointee of a pointer type.
u32 pointee(const Mod& m, u32 ptr_type) {
    for (const Ins& in : m.globals) {
        if (in.op() == spv::OpTypePointer && in.w[1] == ptr_type) return in.w[3];
    }
    return 0;
}
// A 32-bit scalar, or a vector or an array of them: its words.
struct Shape {
    u32 words = 0, elem = 0;
    bool is_float = false, aggregate = false;
};
bool shape_of(const Mod& m, u32 type, Shape& sh) {
    const auto scalar = [&](u32 id) {
        const Ins* d = type_def(m, id);
        if (!d || d->w.size() < 3 || d->w[2] != 32 || (d->op() != spv::OpTypeFloat && d->op() != spv::OpTypeInt)) return false;
        sh.elem = id;
        sh.is_float = d->op() == spv::OpTypeFloat;
        return true;
    };
    if (scalar(type)) {
        sh.words = 1;
        return true;
    }
    const Ins* t = type_def(m, type);
    if (!t || t->w.size() < 4 || !scalar(t->w[2])) return false;
    sh.aggregate = true;
    if (t->op() == spv::OpTypeVector) {
        sh.words = t->w[3];
        return true;
    }
    if (t->op() == spv::OpTypeArray) {
        const Ins* len = constant_def(m, t->w[3]);
        if (!len) return false;
        sh.words = len->w[3];
        return sh.words > 0 && sh.words <= 64;
    }
    return false;
}
// A value's words as u32s.
std::vector<u32> words_of(Splice& s, std::vector<Ins>& out, u32 value, const Shape& sh, u32 t_u32) {
    std::vector<u32> w;
    for (u32 i = 0; i < sh.words; ++i) {
        const u32 c = sh.aggregate ? s.op(out, spv::OpCompositeExtract, sh.elem, {value, i}) : value;
        w.push_back(sh.elem == t_u32 ? c : s.op(out, spv::OpBitcast, t_u32, {c}));
    }
    return w;
}
// The GS's outputs as a check record's vertex, in `outs`' order.
bool output_slots(const Mod& g, const std::map<u32, Var>& gv, const std::vector<u32>& outs, std::vector<GsOutputSlot>& slots,
                  std::map<u32, Shape>& shapes, u32& vertex_dwords, std::string& why) {
    vertex_dwords = 0;
    for (const u32 o : outs) {
        Shape sh;
        if (!shape_of(g, pointee(g, gv.at(o).type), sh)) return why = "an output that is not 32-bit words", false;
        GsOutputSlot slot;
        slot.builtin = gv.at(o).builtin;
        slot.location = gv.at(o).location;
        slot.offset = vertex_dwords;
        slot.dwords = sh.words;
        vertex_dwords += sh.words;
        slots.push_back(slot);
        shapes[o] = sh;
    }
    return true;
}

// The most emits on one path from the function's entry to its return (back
// edges, to a block on the current path, not taken: the lift emits in no loop).
bool most_emits(const std::vector<Ins>& fn, u32& most, std::string& why) {
    std::map<u32, std::vector<u32>> succ;
    std::map<u32, u32> emits;
    std::vector<u32> order;
    u32 cur = 0;
    for (const Ins& in : fn) {
        switch (in.op()) {
        case spv::OpLabel: cur = in.w[1]; order.push_back(cur); succ[cur]; break;
        case spv::OpEmitVertex: ++emits[cur]; break;
        case spv::OpBranch: succ[cur].push_back(in.w[1]); break;
        case spv::OpBranchConditional: succ[cur].push_back(in.w[2]); succ[cur].push_back(in.w[3]); break;
        case spv::OpSwitch:
            for (std::size_t k = 2; k < in.w.size(); k += (k == 2 ? 1 : 2)) succ[cur].push_back(in.w[k]);
            break;
        default: break;
        }
    }
    if (order.empty()) return why = "the GS has no code", false;
    std::map<u32, u32> memo;
    std::set<u32> on_path;
    bool deep = false;
    std::function<u32(u32, int)> walk = [&](u32 b, int depth) -> u32 {
        if (depth > 4000) {
            deep = true;
            return 0;
        }
        if (const auto it = memo.find(b); it != memo.end()) return it->second;
        on_path.insert(b);
        u32 best = 0;
        for (const u32 s : succ[b]) {
            if (on_path.count(s)) continue;  // a back edge
            best = std::max(best, walk(s, depth + 1));
        }
        on_path.erase(b);
        return memo[b] = best + emits[b];
    };
    most = walk(order.front(), 0);
    if (deep) return why = "the GS's control flow is too deep to count its emits", false;
    return true;
}

// The most emits an invocation can make, from which emits exclude each other
// (0: not known). Each emit's block is reached under its selections'
// conditions, taken as literals - with the And, Or and Not they are made of,
// as far as those imply - and two emits exclude each other when one holds a
// value true that the other holds false. The invocations making emit i made
// at most the earlier emits that do not exclude it.
u32 exclusive_emits(const std::vector<Ins>& fn) {
    std::map<u32, const Ins*> defs;  // OpLogicalAnd, OpLogicalOr, OpLogicalNot by result
    for (const Ins& in : fn) {
        if ((in.op() == spv::OpLogicalAnd || in.op() == spv::OpLogicalOr || in.op() == spv::OpLogicalNot) && in.w.size() >= 4) defs[in.w[2]] = &in;
    }
    using Lits = std::set<std::pair<u32, bool>>;
    const auto closure = [&](Lits& lits) {
        std::vector<std::pair<u32, bool>> work(lits.begin(), lits.end());
        const auto add = [&](u32 x, bool v) {
            if (lits.insert({x, v}).second) work.push_back({x, v});
        };
        while (!work.empty()) {
            const auto [id, v] = work.back();
            work.pop_back();
            const auto it = defs.find(id);
            if (it == defs.end()) continue;
            const Ins& d = *it->second;
            if (d.op() == spv::OpLogicalNot) {
                add(d.w[3], !v);
            } else if (d.op() == spv::OpLogicalAnd && v && d.w.size() >= 5) {
                add(d.w[3], true);
                add(d.w[4], true);
            } else if (d.op() == spv::OpLogicalOr && !v && d.w.size() >= 5) {
                add(d.w[3], false);
                add(d.w[4], false);
            }
        }
    };
    struct Open {
        u32 cond = 0, then_label = 0, else_label = 0, merge = 0;
        bool loop = false;
        int arm = 0;  // 1 the true arm, 2 the false arm
    };
    std::vector<Open> open;
    std::vector<Lits> emits;
    u32 pending = 0;
    bool pending_loop = false;
    for (const Ins& in : fn) {
        switch (in.op()) {
        case spv::OpLabel: {
            const u32 id = in.w[1];
            while (!open.empty() && open.back().merge == id) open.pop_back();
            if (!open.empty() && !open.back().loop) {
                if (id == open.back().then_label) open.back().arm = 1;
                else if (id == open.back().else_label) open.back().arm = 2;
            }
            break;
        }
        case spv::OpSelectionMerge: pending = in.w[1]; pending_loop = false; break;
        case spv::OpLoopMerge: pending = in.w[1]; pending_loop = true; break;
        case spv::OpBranchConditional:
            if (pending) open.push_back({pending_loop ? 0u : in.w[1], in.w[2], in.w[3], pending, pending_loop, 0});
            pending = 0;
            break;
        case spv::OpBranch: case spv::OpSwitch:
            if (pending) open.push_back({0, 0, 0, pending, true, 0});  // a loop's header, or a switch: no literal
            pending = 0;
            break;
        case spv::OpEmitVertex: {
            Lits lits;
            for (const Open& o : open) {
                if (o.loop) return 0;  // an emit in a loop or a switch: not known
                if (o.arm == 1) lits.insert({o.cond, true});
                if (o.arm == 2) lits.insert({o.cond, false});
            }
            closure(lits);
            emits.push_back(std::move(lits));
            break;
        }
        default: break;
        }
    }
    const auto exclude = [](const Lits& a, const Lits& b) {
        for (const auto& [id, v] : a) {
            if (b.count({id, !v})) return true;
        }
        return false;
    };
    u32 most = 0;
    for (std::size_t i = 0; i < emits.size(); ++i) {
        u32 before = 0;
        for (std::size_t j = 0; j < i; ++j) before += exclude(emits[i], emits[j]) ? 0 : 1;
        most = std::max(most, before + 1);
    }
    return most;
}

// The words of an OpEntryPoint after its name: the interface.
std::vector<u32> interface_of(const Ins& entry) {
    const std::size_t from = 3 + str_words(word_str(entry.w, 3)).size();
    return std::vector<u32>(entry.w.begin() + static_cast<std::ptrdiff_t>(std::min(from, entry.w.size())), entry.w.end());
}

}  // namespace

GsVertexStage gs_vertex_stage(const std::vector<u32>& es_spirv, const std::vector<u32>& gs_spirv, const GsVertexOptions& opt) {
    GsVertexStage res;
    Splice s;
    const auto refuse = [&](const std::string& w) {
        res.spirv.clear();
        res.why = w;
        return res;
    };
    std::string why;
    if (!parse(gs_spirv, s.g, why)) return refuse("GS: " + why);
    if (!parse(es_spirv, s.e, why)) return refuse("ES: " + why);
    Mod& g = s.g;
    Mod& e = s.e;
    if (g.entries[0].w.size() < 4 || g.entries[0].w[1] != 3 /* Geometry */) return refuse("the GS lift is not a geometry stage");
    if (e.entries[0].w.size() < 4 || e.entries[0].w[1] != 0 /* Vertex */) return refuse("the ES lift is not a vertex stage");
    const u32 g_main = g.entries[0].w[2], e_main = e.entries[0].w[2];

    // ---- what each module's interface is ---------------------------------------------
    const std::map<u32, Var> gv = variables(g), ev = variables(e);
    const std::vector<u32> g_es_ids = named(g, "es_gs"), e_es_ids = named(e, "es_gs"), g_params_ids = named(g, "params");
    if (g_es_ids.size() != 1 || !gv.count(g_es_ids[0]) || gv.at(g_es_ids[0]).sc != spv::ScInput) return refuse("the GS has no es_gs input");
    if (e_es_ids.size() != 1 || !ev.count(e_es_ids[0]) || ev.at(e_es_ids[0]).sc != spv::ScOutput) return refuse("the ES has no es_gs output");
    if (g_params_ids.size() != 1 || !gv.count(g_params_ids[0])) return refuse("the GS has no params block");
    const u32 g_es = g_es_ids[0], e_es = e_es_ids[0], g_params = g_params_ids[0];
    u32 g_prim = 0, e_vid = 0, e_iid = 0;
    std::vector<u32> g_outputs;
    for (const auto& [id, v] : gv) {
        if (v.sc == spv::ScInput && id != g_es) {
            if (v.builtin == spv::BiPrimitiveId) g_prim = id;
            else return refuse("the GS reads an input the splice does not give (builtin " + std::to_string(v.builtin) + ")");
        }
        if (v.sc == spv::ScOutput) g_outputs.push_back(id);
    }
    for (const auto& [id, v] : ev) {
        if (v.sc == spv::ScInput) {
            if (v.builtin == spv::BiVertexIndex) e_vid = id;
            else if (v.builtin == spv::BiInstanceIndex) e_iid = id;
            else return refuse("the ES reads an input the splice does not give (builtin " + std::to_string(v.builtin) + ")");
        }
        if (v.sc == spv::ScOutput && id != e_es) return refuse("the ES writes an output besides its ring vertex");
    }
    if (g_outputs.empty()) return refuse("the GS writes no output");
    if (!e_vid) return refuse("the ES reads no vertex index");
    if (g.functions.size() != 1 || e.functions.size() != 1) return refuse("a lift with helper functions");
    const std::vector<Ins>& g_fn = g.functions[0];
    const std::vector<Ins>& e_fn = e.functions[0];
    if (g_fn.size() < 3 || g_fn[0].w[2] != g_main || e_fn.size() < 3 || e_fn[0].w[2] != e_main) return refuse("the entry point is not the module's function");
    for (const std::vector<Ins>* fn : {&g_fn, &e_fn}) {
        const std::size_t returns = static_cast<std::size_t>(std::count_if(fn->begin(), fn->end(), [](const Ins& in) { return in.op() == spv::OpReturn; }));
        if (returns != 1 || (*fn)[fn->size() - 2].op() != spv::OpReturn) return refuse("a lift that does not return once, at its end");
    }
    u32 most = 0;
    if (!most_emits(g_fn, most, why)) return refuse(why);
    res.path_emits = most;
    u32 vertices = most;
    if (const u32 x = exclusive_emits(g_fn); x && x < vertices) vertices = x;
    if (opt.vertices) vertices = opt.vertices;
    // One strip a particle where no emit follows a cut (the lift lays its
    // blocks out in program order); else strips of four.
    bool emit_after_cut = false, cut = false;
    for (const Ins& in : g_fn) {
        if (in.op() == spv::OpEndPrimitive) cut = true;
        if (in.op() == spv::OpEmitVertex && cut) emit_after_cut = true;
    }
    res.strip = !emit_after_cut;
    if (res.strip ? vertices < 3 : vertices % 4 != 0) {
        return refuse("a particle's vertices would be " + std::to_string(vertices) + (res.strip ? ", not a strip's triangle" : ", not whole quads"));
    }
    res.vertices = vertices;
    std::map<u32, Shape> shapes;
    if (!output_slots(g, gv, g_outputs, res.outputs, shapes, res.vertex_dwords, why) && opt.check) return refuse(why);

    // ---- ids: the GS's stay, the ES's move past them (its types the GS has merge) ----
    for (const Ins& in : g.globals) s.know(in);
    const u32 base = g.bound - 1;
    std::map<u32, u32> alias;  // ES id -> merged id, where not id + base
    const auto map_id = [&](u32 id) {
        const auto it = alias.find(id);
        return it != alias.end() ? it->second : id + base;
    };
    std::map<std::vector<u32>, u32> g_types;  // dedupable types by (opcode, operands)
    const auto dedupable = [](std::uint16_t op) {
        return op == spv::OpTypeVoid || op == spv::OpTypeBool || op == spv::OpTypeInt || op == spv::OpTypeFloat || op == spv::OpTypeVector ||
               op == 24 || op == spv::OpTypeImage || op == spv::OpTypeSampler || op == spv::OpTypeSampledImage || op == spv::OpTypeFunction;
    };
    for (const Ins& in : g.globals) {
        if (!dedupable(in.op())) continue;
        std::vector<u32> key{in.op()};
        key.insert(key.end(), in.w.begin() + 2, in.w.end());
        g_types.emplace(key, in.w[1]);
    }
    std::map<std::string, u32> g_imports;
    for (const Ins& in : g.imports) g_imports[word_str(in.w, 2)] = in.w[1];
    s.next = g.bound + e.bound - 1;
    std::vector<bool> idw;
    const auto remap = [&](const Ins& in, Ins& out) {
        out = in;
        if (!id_words(in, idw)) return s.fail("ES: an instruction the splice does not know (opcode " + std::to_string(in.op()) + ")");
        for (std::size_t k = 1; k < out.w.size(); ++k) {
            if (idw[k]) out.w[k] = map_id(out.w[k]);
        }
        return true;
    };

    std::vector<Ins> out_caps, out_exts, out_imports, out_debug, out_annots, out_globals, new_names;
    // Capabilities: both modules', less Geometry; physical storage buffer
    // addresses for the draw's arguments.
    std::set<u32> caps;
    for (const Ins& in : g.caps) caps.insert(in.w[1]);
    for (const Ins& in : e.caps) caps.insert(in.w[1]);
    caps.erase(spv::CapGeometry);
    caps.insert(spv::CapPhysicalStorageBufferAddresses);
    for (const u32 c : caps) out_caps.push_back(mk(spv::OpCapability, {c}));
    std::set<std::string> exts;
    for (const Ins& in : g.exts) exts.insert(word_str(in.w, 1));
    for (const Ins& in : e.exts) exts.insert(word_str(in.w, 1));
    exts.insert("SPV_KHR_physical_storage_buffer");
    for (const std::string& x : exts) out_exts.push_back(mk(spv::OpExtension, str_words(x)));
    out_imports = g.imports;
    for (const Ins& in : e.imports) {
        const std::string name = word_str(in.w, 2);
        if (const auto it = g_imports.find(name); it != g_imports.end()) {
            alias[in.w[1]] = it->second;
        } else {
            Ins c = in;
            c.w[1] = in.w[1] + base;
            out_imports.push_back(c);
        }
    }

    // The ES's globals: its types the GS has, merged; its ring vertex Private;
    // the rest as they are, moved.
    for (const Ins& in : e.globals) {
        if (dedupable(in.op())) {
            Ins r;
            if (!remap(in, r)) return refuse(s.why);
            std::vector<u32> key{in.op()};
            key.insert(key.end(), r.w.begin() + 2, r.w.end());
            if (const auto it = g_types.find(key); it != g_types.end()) {
                alias[in.w[1]] = it->second;
                continue;
            }
            g_types.emplace(key, r.w[1]);
            out_globals.push_back(r);
            continue;
        }
        Ins r;
        if (!remap(in, r)) return refuse(s.why);
        if (in.op() == spv::OpVariable && in.w[2] == e_es) {
            const u32 slots = map_id(pointee(e, in.w[1]));
            for (Ins& a : out_globals) s.know(a);
            r.w[1] = s.ptr(spv::ScPrivate, slots);
            r.w[3] = spv::ScPrivate;
            out_globals.insert(out_globals.end(), s.added.begin(), s.added.end());
            s.added.clear();
        }
        out_globals.push_back(r);
    }
    for (const Ins& in : out_globals) s.know(in);
    const u32 es_vertex = map_id(e_es);

    // ---- the new variables and constants -------------------------------------------
    const u32 t_u32 = s.t_u32(), t_bool = s.t_bool(), t_v4f = s.t_v4f(), t_u64 = s.t_u64();
    Psb psb(s);
    const u32 counter = s.private_var(t_u32, "gs_emitted", new_names);
    const u32 strip = s.private_var(t_u32, "gs_strip", new_names);
    const u32 bad = s.private_var(t_bool, "gs_bad", new_names);
    const u32 cut_seen = res.strip ? s.private_var(t_bool, "gs_cut", new_names) : 0;  // one strip: a cut has ended it
    const u32 es_vid = s.private_var(t_u32, "es_vertex_index", new_names);
    const u32 es_iid = e_iid ? s.private_var(t_u32, "es_instance_index", new_names) : 0;
    u32 prim = 0, prim_type = 0;
    if (g_prim) {
        prim_type = pointee(g, gv.at(g_prim).type);
        prim = s.private_var(prim_type, "gs_primitive", new_names);
    }
    std::map<u32, u32> shadow, final_of;  // output -> its Private stand-ins
    std::map<u32, u32> out_type;
    for (const u32 o : g_outputs) {
        const u32 t = pointee(g, gv.at(o).type);
        out_type[o] = t;
        shadow[o] = s.private_var(t, "gs_out", new_names);
        final_of[o] = s.private_var(t, "gs_kept", new_names);
    }
    u32 position = 0;
    for (const u32 o : g_outputs) {
        if (gv.at(o).builtin == spv::BiPosition) position = o;
    }
    if (!position || out_type[position] != t_v4f) return refuse("the GS writes no vec4 position");
    const u32 c0 = s.c_u32(0), c1 = s.c_u32(1), c4 = s.c_u32(4), c_vpp = s.c_u32(vertices);
    const u32 culled = s.have(spv::OpConstantComposite, {t_v4f, s.c_f32(2.0f), s.c_f32(0.0f), s.c_f32(0.0f), s.c_f32(1.0f)}, true);
    const u32 p_uni_u64 = s.ptr(spv::ScUniform, t_u64);

    // ---- the merged function ------------------------------------------------------------
    std::vector<Ins> body;  // after the entry block's variables
    std::vector<Ins> vars;  // OpVariable Function, both functions'
    const auto first_block = [](const std::vector<Ins>& fn, std::size_t& at) {  // index past the first block's label and variables
        at = 1;
        while (at < fn.size() && fn[at].op() != spv::OpLabel) ++at;
        if (at < fn.size()) ++at;
        while (at < fn.size() && fn[at].op() == spv::OpVariable) ++at;
    };
    std::size_t g_at = 0, e_at = 0;
    first_block(g_fn, g_at);
    first_block(e_fn, e_at);
    if (g_fn[1].op() != spv::OpLabel || e_fn[1].op() != spv::OpLabel) return refuse("a lift's function does not begin with its block");
    const u32 g_first = g_fn[1].w[1], e_first = map_id(e_fn[1].w[1]);
    for (std::size_t k = 2; k < g_at; ++k) vars.push_back(g_fn[k]);
    for (std::size_t k = 2; k < e_at; ++k) {
        Ins r;
        if (!remap(e_fn[k], r)) return refuse(s.why);
        vars.push_back(r);
    }
    const u32 l_end = s.fresh();

    // The entry: which particle and emit this vertex is, the draw's arguments
    // (with the check record: at its header's address).
    std::vector<Ins> entry;
    const u32 vid = s.op(entry, spv::OpLoad, t_u32, {map_id(e_vid)});
    const u32 lds = s.op(entry, spv::OpLoad, t_u64, {s.op(entry, spv::OpAccessChain, p_uni_u64, {g_params, c1})});  // StageParams::lds_address
    const u32 rec = opt.check ? lds : 0;
    const u32 args = opt.check ? psb.load64(entry, rec) : lds;
    const u32 cap = opt.check ? psb.load32(entry, psb.at(entry, rec, 8)) : 0;
    const u32 n = psb.load32(entry, args);
    const u32 instances = psb.load32(entry, psb.at(entry, args, 4));
    const u32 first = psb.load32(entry, psb.at(entry, args, 8));
    const u32 p = s.op(entry, spv::OpUDiv, t_u32, {vid, c_vpp});
    const u32 k = s.op(entry, spv::OpUMod, t_u32, {vid, c_vpp});
    // A particle of the draw's first instance (one with none draws nothing).
    const u32 live = s.op(entry, spv::OpLogicalAnd, t_bool,
                          {s.op(entry, spv::OpULessThan, t_bool, {p, n}), s.op(entry, spv::OpINotEqual, t_bool, {instances, c0})});
    s.vop(entry, spv::OpStore, {es_vid, s.op(entry, spv::OpIAdd, t_u32, {first, p})});
    if (es_iid) s.vop(entry, spv::OpStore, {es_iid, psb.load32(entry, psb.at(entry, args, 12))});
    if (prim) s.vop(entry, spv::OpStore, {prim, prim_type == t_u32 ? p : s.op(entry, spv::OpBitcast, prim_type, {p})});
    s.vop(entry, spv::OpSelectionMerge, {l_end, 0});
    s.vop(entry, spv::OpBranchConditional, {live, e_first, l_end});

    // The ES's blocks: its vertex index the particle's, its ring vertex
    // Private, its return into the GS's first block.
    for (std::size_t i = 1; i + 1 < e_fn.size(); ++i) {
        if (i >= 2 && i < e_at) continue;  // its first block's variables, in the entry
        const Ins& in = e_fn[i];
        Ins r;
        if (!remap(in, r)) return refuse(s.why);
        if (r.op() == spv::OpLoad && r.w[3] == map_id(e_vid)) r.w[3] = es_vid;
        if (e_iid && r.op() == spv::OpLoad && r.w[3] == map_id(e_iid)) r.w[3] = es_iid;
        if ((r.op() == spv::OpAccessChain || r.op() == 66) && r.w[3] == es_vertex) {
            r.w[1] = s.ptr(spv::ScPrivate, map_id(pointee(e, in.w[1])));
        } else if ((r.op() == spv::OpLoad && r.w[3] == es_vertex) || (r.op() == spv::OpStore && r.w[1] == es_vertex)) {
            return refuse("the ES reads or writes its ring vertex whole");
        }
        if (r.op() == spv::OpReturn) r = mk(spv::OpBranch, {g_first});
        body.push_back(r);
    }
    // The GS's blocks: its ring loads from the ES's vertex, its outputs kept at
    // emit k, its return into the end.
    for (std::size_t i = 1; i + 1 < g_fn.size(); ++i) {
        if (i >= 2 && i < g_at) continue;
        const Ins& in = g_fn[i];
        Ins r = in;
        const std::uint16_t op = in.op();
        if ((op == spv::OpAccessChain || op == 66) && in.w[3] == g_es) {
            if (in.w.size() < 5) return refuse("a GS ring access without its vertex index");
            std::vector<u32> ops{s.ptr(spv::ScPrivate, pointee(g, in.w[1])), in.w[2], es_vertex};
            ops.insert(ops.end(), in.w.begin() + 5, in.w.end());  // drop the vertex index (a point has one)
            r = mk(op, ops);
        } else if (op == spv::OpLoad && in.w[3] == g_es) {
            return refuse("the GS reads its input vertex whole");
        } else if (g_prim && op == spv::OpLoad && in.w[3] == g_prim) {
            r.w[3] = prim;
        } else if (op == spv::OpStore && shadow.count(in.w[1])) {
            r.w[1] = shadow[in.w[1]];
        } else if (op == spv::OpEmitVertex) {
            // Every emit up to k: vertex k ends as emit k, or as the last emit.
            const u32 cnt = s.op(body, spv::OpLoad, t_u32, {counter});
            const u32 hit = s.op(body, spv::OpULessThanEqual, t_bool, {cnt, k});
            for (const u32 o : g_outputs) {
                const u32 sv = s.op(body, spv::OpLoad, out_type[o], {shadow[o]});
                const u32 fv = s.op(body, spv::OpLoad, out_type[o], {final_of[o]});
                s.vop(body, spv::OpStore, {final_of[o], s.op(body, spv::OpSelect, out_type[o], {hit, sv, fv})});
            }
            // Quads: a fifth vertex in a strip breaks them. One strip: an emit after a cut.
            const u32 sl = s.op(body, spv::OpLoad, t_u32, {strip});
            const u32 breaks = res.strip ? s.op(body, spv::OpLoad, t_bool, {cut_seen}) : s.op(body, spv::OpUGreaterThanEqual, t_bool, {sl, c4});
            s.vop(body, spv::OpStore, {bad, s.op(body, spv::OpLogicalOr, t_bool, {s.op(body, spv::OpLoad, t_bool, {bad}), breaks})});
            s.vop(body, spv::OpStore, {strip, s.op(body, spv::OpIAdd, t_u32, {sl, c1})});
            s.vop(body, spv::OpStore, {counter, s.op(body, spv::OpIAdd, t_u32, {cnt, c1})});
            continue;
        } else if (op == spv::OpEndPrimitive) {
            const u32 sl = s.op(body, spv::OpLoad, t_u32, {strip});
            if (res.strip) {  // a cut ends the one strip, if it has begun
                const u32 begun = s.op(body, spv::OpINotEqual, t_bool, {sl, c0});
                s.vop(body, spv::OpStore, {cut_seen, s.op(body, spv::OpLogicalOr, t_bool, {s.op(body, spv::OpLoad, t_bool, {cut_seen}), begun})});
            } else {  // quads: a cut after four vertices or none
                const u32 okc = s.op(body, spv::OpLogicalOr, t_bool,
                                     {s.op(body, spv::OpIEqual, t_bool, {sl, c0}), s.op(body, spv::OpIEqual, t_bool, {sl, c4})});
                const u32 nok = s.op(body, spv::OpLogicalNot, t_bool, {okc});
                s.vop(body, spv::OpStore, {bad, s.op(body, spv::OpLogicalOr, t_bool, {s.op(body, spv::OpLoad, t_bool, {bad}), nok})});
            }
            s.vop(body, spv::OpStore, {strip, c0});
            continue;
        } else if (op == spv::OpReturn) {
            r = mk(spv::OpBranch, {l_end});
        } else {
            // Any other use of a replaced variable is refused.
            if (!id_words(in, idw)) return refuse("GS: an instruction the splice does not know (opcode " + std::to_string(op) + ")");
            for (std::size_t w = 1; w < in.w.size(); ++w) {
                if (!idw[w]) continue;
                if (in.w[w] == g_es || (g_prim && in.w[w] == g_prim) || shadow.count(in.w[w]))
                    return refuse("the GS uses its ring input, primitive id or an output other than as the splice reads them");
            }
        }
        body.push_back(r);
    }
    // The end: the vertex kept when its particle emitted any, in the shape its
    // draw has and no more emits than the draw has vertices for; else outside
    // the clip volume. (Kept past the particle's emits, it is its last emit.)
    s.vop(body, spv::OpLabel, {l_end});
    const u32 cnt = s.op(body, spv::OpLoad, t_u32, {counter});
    const u32 any = s.op(body, spv::OpINotEqual, t_bool, {cnt, c0});
    const u32 emitted = s.op(body, spv::OpULessThan, t_bool, {k, cnt});
    const u32 within = s.op(body, spv::OpULessThanEqual, t_bool, {cnt, c_vpp});
    const u32 nb = s.op(body, spv::OpLogicalNot, t_bool, {s.op(body, spv::OpLoad, t_bool, {bad})});
    u32 strips_ok = nb;
    if (!res.strip) {  // quads: the last strip four vertices or none too
        const u32 sl = s.op(body, spv::OpLoad, t_u32, {strip});
        const u32 sok = s.op(body, spv::OpLogicalOr, t_bool, {s.op(body, spv::OpIEqual, t_bool, {sl, c0}), s.op(body, spv::OpIEqual, t_bool, {sl, c4})});
        strips_ok = s.op(body, spv::OpLogicalAnd, t_bool, {sok, nb});
    }
    const u32 ok = s.op(body, spv::OpLogicalAnd, t_bool, {s.op(body, spv::OpLogicalAnd, t_bool, {any, within}), strips_ok});
    // (The check mode's vertex stage draws beside the GS: its every vertex outside.)
    std::map<u32, u32> kept;
    for (const u32 o : g_outputs) {
        const u32 fv = s.op(body, spv::OpLoad, out_type[o], {final_of[o]});
        kept[o] = fv;
        s.vop(body, spv::OpStore, {o, o != position ? fv : opt.check ? culled : s.op(body, spv::OpSelect, t_v4f, {ok, fv, culled})});
    }
    if (opt.check) {
        // The check record: vertex 0 of the draw writes the arguments it read;
        // vertex 0 of each particle in the record its emits and whether it was
        // left out; each kept vertex its outputs.
        const u32 l_head = s.fresh(), l_after_head = s.fresh(), l_rec = s.fresh(), l_count = s.fresh(), l_after_count = s.fresh(),
                  l_out = s.fresh(), l_after_out = s.fresh(), l_ret = s.fresh();
        const u32 first_vertex = s.op(body, spv::OpIEqual, t_bool, {vid, c0});
        s.vop(body, spv::OpSelectionMerge, {l_after_head, 0});
        s.vop(body, spv::OpBranchConditional, {first_vertex, l_head, l_after_head});
        s.vop(body, spv::OpLabel, {l_head});
        psb.store32(body, psb.at(body, rec, 20), n);
        psb.store32(body, psb.at(body, rec, 24), instances);
        psb.store32(body, psb.at(body, rec, 28), first);
        psb.store32(body, psb.at(body, rec, 32), c1);
        s.vop(body, spv::OpBranch, {l_after_head});
        s.vop(body, spv::OpLabel, {l_after_head});
        const u32 in_rec = s.op(body, spv::OpLogicalAnd, t_bool, {live, s.op(body, spv::OpULessThan, t_bool, {p, cap})});
        s.vop(body, spv::OpSelectionMerge, {l_ret, 0});
        s.vop(body, spv::OpBranchConditional, {in_rec, l_rec, l_ret});
        s.vop(body, spv::OpLabel, {l_rec});
        const u32 per = s.c_u32(gs_check_particle_dwords(vertices, res.vertex_dwords));
        const u32 at = psb.word(body, psb.at(body, rec, 4 * kGsCheckHeaderDwords), s.op(body, spv::OpIMul, t_u32, {p, per}));
        const u32 first_of_particle = s.op(body, spv::OpIEqual, t_bool, {k, c0});
        s.vop(body, spv::OpSelectionMerge, {l_after_count, 0});
        s.vop(body, spv::OpBranchConditional, {first_of_particle, l_count, l_after_count});
        s.vop(body, spv::OpLabel, {l_count});
        psb.store32(body, at, cnt);
        const u32 strip_bit = s.op(body, spv::OpSelect, t_u32, {strips_ok, c0, s.c_u32(2)});
        const u32 within_bit = s.op(body, spv::OpSelect, t_u32, {within, c0, c4});
        psb.store32(body, psb.at(body, at, 4), s.op(body, spv::OpBitwiseOr, t_u32, {c1, s.op(body, spv::OpBitwiseOr, t_u32, {strip_bit, within_bit})}));
        s.vop(body, spv::OpBranch, {l_after_count});
        s.vop(body, spv::OpLabel, {l_after_count});
        const u32 recorded = s.op(body, spv::OpLogicalAnd, t_bool, {ok, emitted});  // the particle kept, this vertex one of its emits
        s.vop(body, spv::OpSelectionMerge, {l_after_out, 0});
        s.vop(body, spv::OpBranchConditional, {recorded, l_out, l_after_out});
        s.vop(body, spv::OpLabel, {l_out});
        const u32 vat = psb.word(body, at, s.op(body, spv::OpIAdd, t_u32, {s.c_u32(kGsCheckParticleHead),
                                                                          s.op(body, spv::OpIMul, t_u32, {k, s.c_u32(res.vertex_dwords)})}));
        for (std::size_t o = 0; o < g_outputs.size(); ++o) {
            const std::vector<u32> w = words_of(s, body, kept[g_outputs[o]], shapes[g_outputs[o]], t_u32);
            for (std::size_t j = 0; j < w.size(); ++j) psb.store32(body, psb.at(body, vat, 4ull * (res.outputs[o].offset + j)), w[j]);
        }
        s.vop(body, spv::OpBranch, {l_after_out});
        s.vop(body, spv::OpLabel, {l_after_out});
        s.vop(body, spv::OpBranch, {l_ret});
        s.vop(body, spv::OpLabel, {l_ret});
    }
    s.vop(body, spv::OpReturn, {});

    // ---- the module ------------------------------------------------------------------------
    // Debug names and decorations: the GS's, less its replaced inputs'; the ES's
    // moved, less its function's and its ring vertex's location.
    const std::set<u32> g_gone{g_es, g_prim};
    for (const Ins& in : g.debug) {
        if ((in.op() == spv::OpName || in.op() == 6) && g_gone.count(in.w[1])) continue;
        out_debug.push_back(in);
    }
    for (const Ins& in : e.debug) {
        if (in.op() != spv::OpName && in.op() != 6) continue;  // OpString, OpSource: left out
        if (in.w[1] == e_main) continue;                       // the ES's function is not kept
        Ins r;
        if (!remap(in, r)) return refuse(s.why);
        out_debug.push_back(r);
    }
    out_debug.insert(out_debug.end(), new_names.begin(), new_names.end());
    for (const Ins& in : g.annots) {
        if ((in.op() == spv::OpDecorate || in.op() == spv::OpMemberDecorate) && g_gone.count(in.w[1])) continue;
        out_annots.push_back(in);
    }
    for (const Ins& in : e.annots) {
        if (in.op() == spv::OpDecorate && in.w[1] == e_es) continue;
        if (in.op() != spv::OpDecorate && in.op() != spv::OpMemberDecorate) return refuse("ES: a decoration group");
        if (in.w[1] == e_main) continue;
        Ins r;
        if (!remap(in, r)) return refuse(s.why);
        out_annots.push_back(r);
    }
    // The GS's globals less its replaced inputs.
    std::vector<Ins> g_globals;
    for (const Ins& in : g.globals) {
        if (in.op() == spv::OpVariable && g_gone.count(in.w[2])) continue;
        g_globals.push_back(in);
    }
    // The entry point: a vertex stage, every global variable the function reaches.
    std::vector<u32> iface;
    for (const u32 id : interface_of(g.entries[0])) {
        if (!g_gone.count(id)) iface.push_back(id);
    }
    for (const u32 id : interface_of(e.entries[0])) iface.push_back(map_id(id));
    for (const Ins& in : s.added) {
        if (in.op() == spv::OpVariable) iface.push_back(in.w[2]);
    }
    std::sort(iface.begin(), iface.end());
    iface.erase(std::unique(iface.begin(), iface.end()), iface.end());
    std::vector<u32> ep{0 /* Vertex */, g_main};
    const std::vector<u32> name = str_words("main");
    ep.insert(ep.end(), name.begin(), name.end());
    ep.insert(ep.end(), iface.begin(), iface.end());
    // Execution modes: the float controls either declares, not the geometry's.
    std::set<std::vector<u32>> modes;
    for (const Ins& in : g.modes) {
        const u32 mode = in.w.size() > 2 ? in.w[2] : 0;
        if (mode == spv::ExInputPoints || mode == spv::ExInvocations || mode == spv::ExOutputTriangleStrip || mode == spv::ExOutputLineStrip ||
            mode == spv::ExOutputVertices || mode == 20 || mode == 21 || mode == 22 || mode == 23 || mode == 27)
            continue;
        modes.insert(std::vector<u32>(in.w.begin() + 2, in.w.end()));
    }
    for (const Ins& in : e.modes) modes.insert(std::vector<u32>(in.w.begin() + 2, in.w.end()));

    const u32 l_entry = s.fresh();
    std::vector<u32> outw = {0x07230203u, gs_spirv[1], gs_spirv[2], 0u, 0u};
    const auto put = [&](const Ins& in) { outw.insert(outw.end(), in.w.begin(), in.w.end()); };
    for (const Ins& in : out_caps) put(in);
    for (const Ins& in : out_exts) put(in);
    for (const Ins& in : out_imports) put(in);
    put(mk(spv::OpMemoryModel, {5348 /* PhysicalStorageBuffer64 */, g.model.empty() || g.model[0].w.size() < 3 ? 1u : g.model[0].w[2]}));
    put(mk(spv::OpEntryPoint, ep));
    for (const std::vector<u32>& m : modes) {
        std::vector<u32> ops{g_main};
        ops.insert(ops.end(), m.begin(), m.end());
        put(mk(spv::OpExecutionMode, ops));
    }
    for (const Ins& in : out_debug) put(in);
    for (const Ins& in : out_annots) put(in);
    for (const Ins& in : g_globals) put(in);
    for (const Ins& in : out_globals) put(in);
    for (const Ins& in : s.added) put(in);
    put(g_fn[0]);
    put(mk(spv::OpLabel, {l_entry}));
    for (const Ins& in : vars) put(in);
    for (const Ins& in : entry) put(in);
    for (const Ins& in : body) put(in);
    put(mk(spv::OpFunctionEnd, {}));
    outw[3] = s.next;
    res.spirv = std::move(outw);
    return res;
}

GsVertexStage gs_check_stage(const std::vector<u32>& gs_spirv, u32 vertices) {
    GsVertexStage res;
    Splice s;
    const auto refuse = [&](const std::string& w) {
        res.spirv.clear();
        res.why = w;
        return res;
    };
    std::string why;
    if (!parse(gs_spirv, s.g, why)) return refuse("GS: " + why);
    Mod& g = s.g;
    if (g.entries[0].w.size() < 4 || g.entries[0].w[1] != 3 /* Geometry */) return refuse("the GS lift is not a geometry stage");
    if (!vertices) return refuse("no vertices a particle");
    const u32 g_main = g.entries[0].w[2];
    const std::map<u32, Var> gv = variables(g);
    const std::vector<u32> g_params_ids = named(g, "params");
    if (g_params_ids.size() != 1 || !gv.count(g_params_ids[0])) return refuse("the GS has no params block");
    const u32 g_params = g_params_ids[0];
    u32 g_prim = 0;
    std::vector<u32> g_outputs;
    for (const auto& [id, v] : gv) {
        if (v.sc == spv::ScInput && v.builtin == spv::BiPrimitiveId) g_prim = id;
        if (v.sc == spv::ScOutput) g_outputs.push_back(id);
    }
    if (g_outputs.empty()) return refuse("the GS writes no output");
    if (g.functions.size() != 1) return refuse("a lift with helper functions");
    const std::vector<Ins>& fn = g.functions[0];
    if (fn.size() < 3 || fn[0].w[2] != g_main || fn[1].op() != spv::OpLabel) return refuse("the entry point is not the module's function");
    if (std::count_if(fn.begin(), fn.end(), [](const Ins& in) { return in.op() == spv::OpReturn; }) != 1 || fn[fn.size() - 2].op() != spv::OpReturn)
        return refuse("a lift that does not return once, at its end");
    std::map<u32, Shape> shapes;
    if (!output_slots(g, gv, g_outputs, res.outputs, shapes, res.vertex_dwords, why)) return refuse(why);
    res.vertices = vertices;
    res.path_emits = vertices;

    for (const Ins& in : g.globals) s.know(in);
    s.next = g.bound;
    const u32 t_u32 = s.t_u32(), t_bool = s.t_bool(), t_u64 = s.t_u64();
    Psb psb(s);
    std::vector<Ins> new_names, new_annots;
    const u32 counter = s.private_var(t_u32, "gs_emitted", new_names);
    std::map<u32, u32> shadow, out_type;
    for (const u32 o : g_outputs) {
        out_type[o] = pointee(g, gv.at(o).type);
        shadow[o] = s.private_var(out_type[o], "gs_out", new_names);
    }
    const u32 vd = res.vertex_dwords;
    const u32 acc_words = (vertices + 1) * vd;  // the first `vertices` emits, then one slot every later emit lands in
    const u32 acc = s.private_var(s.have(spv::OpTypeArray, {t_u32, s.c_u32(acc_words)}), "gs_check_vertices", new_names);
    const u32 p_priv_u32 = s.ptr(spv::ScPrivate, t_u32);
    u32 prim_in = g_prim;
    if (!prim_in) {
        prim_in = s.fresh();
        s.added.push_back(mk(spv::OpVariable, {s.ptr(spv::ScInput, t_u32), prim_in, spv::ScInput}));
        new_annots.push_back(mk(spv::OpDecorate, {prim_in, spv::DecBuiltIn, spv::BiPrimitiveId}));
    }
    const u32 prim_type = g_prim ? pointee(g, gv.at(g_prim).type) : t_u32;
    const u32 c1 = s.c_u32(1), c_vpp = s.c_u32(vertices), c_vd = s.c_u32(vd);
    const u32 p_uni_u64 = s.ptr(spv::ScUniform, t_u64);

    std::vector<Ins> out_fn;
    out_fn.push_back(fn[0]);
    out_fn.push_back(fn[1]);
    std::size_t at = 2;
    while (at < fn.size() && fn[at].op() == spv::OpVariable) out_fn.push_back(fn[at++]);
    // The record and the particle, in the first block.
    const u32 rec = s.op(out_fn, spv::OpLoad, t_u64, {s.op(out_fn, spv::OpAccessChain, p_uni_u64, {g_params, c1})});
    const u32 cap = psb.load32(out_fn, psb.at(out_fn, rec, 8));
    u32 p = s.op(out_fn, spv::OpLoad, prim_type, {prim_in});
    if (prim_type != t_u32) p = s.op(out_fn, spv::OpBitcast, t_u32, {p});
    std::vector<bool> idw;
    for (; at + 1 < fn.size(); ++at) {
        const Ins& in = fn[at];
        const std::uint16_t op = in.op();
        if (op == spv::OpStore && shadow.count(in.w[1])) {
            Ins r = in;
            r.w[1] = shadow[in.w[1]];
            out_fn.push_back(r);
            continue;
        }
        if (op == spv::OpEmitVertex) {
            // The outputs from their stand-ins, and into the record's slot.
            const u32 cnt = s.op(out_fn, spv::OpLoad, t_u32, {counter});
            const u32 slot = s.op(out_fn, spv::OpSelect, t_u32, {s.op(out_fn, spv::OpULessThan, t_bool, {cnt, c_vpp}), cnt, c_vpp});
            const u32 slot_at = s.op(out_fn, spv::OpIMul, t_u32, {slot, c_vd});
            for (std::size_t o = 0; o < g_outputs.size(); ++o) {
                const u32 out = g_outputs[o];
                const u32 sv = s.op(out_fn, spv::OpLoad, out_type[out], {shadow[out]});
                s.vop(out_fn, spv::OpStore, {out, sv});
                const std::vector<u32> w = words_of(s, out_fn, sv, shapes[out], t_u32);
                for (std::size_t j = 0; j < w.size(); ++j) {
                    const u32 el = s.op(out_fn, spv::OpIAdd, t_u32, {slot_at, s.c_u32(res.outputs[o].offset + static_cast<u32>(j))});
                    s.vop(out_fn, spv::OpStore, {s.op(out_fn, spv::OpAccessChain, p_priv_u32, {acc, el}), w[j]});
                }
            }
            out_fn.push_back(in);
            s.vop(out_fn, spv::OpStore, {counter, s.op(out_fn, spv::OpIAdd, t_u32, {cnt, c1})});
            continue;
        }
        if (op == spv::OpReturn) {
            // The particle's emits into its record.
            const u32 l_store = s.fresh(), l_ret = s.fresh();
            const u32 cnt = s.op(out_fn, spv::OpLoad, t_u32, {counter});
            const u32 in_rec = s.op(out_fn, spv::OpULessThan, t_bool, {p, cap});
            s.vop(out_fn, spv::OpSelectionMerge, {l_ret, 0});
            s.vop(out_fn, spv::OpBranchConditional, {in_rec, l_store, l_ret});
            s.vop(out_fn, spv::OpLabel, {l_store});
            const u32 per = s.c_u32(gs_check_particle_dwords(vertices, vd));
            const u32 base = psb.word(out_fn, psb.at(out_fn, rec, 4 * kGsCheckHeaderDwords),
                                      s.op(out_fn, spv::OpIMul, t_u32, {s.op(out_fn, spv::OpIAdd, t_u32, {cap, p}), per}));
            psb.store32(out_fn, base, cnt);
            psb.store32(out_fn, psb.at(out_fn, base, 4), c1);
            for (u32 w = 0; w < vertices * vd; ++w) {
                const u32 x = s.op(out_fn, spv::OpLoad, t_u32, {s.op(out_fn, spv::OpAccessChain, p_priv_u32, {acc, s.c_u32(w)})});
                psb.store32(out_fn, psb.at(out_fn, base, 4ull * (kGsCheckParticleHead + w)), x);
            }
            s.vop(out_fn, spv::OpBranch, {l_ret});
            s.vop(out_fn, spv::OpLabel, {l_ret});
            out_fn.push_back(in);
            continue;
        }
        if (!id_words(in, idw)) return refuse("GS: an instruction the splice does not know (opcode " + std::to_string(op) + ")");
        for (std::size_t w = 1; w < in.w.size(); ++w) {
            if (idw[w] && shadow.count(in.w[w])) return refuse("the GS uses an output other than by storing it");
        }
        out_fn.push_back(in);
    }
    out_fn.push_back(fn.back());

    std::set<u32> caps;
    for (const Ins& in : g.caps) caps.insert(in.w[1]);
    caps.insert(spv::CapPhysicalStorageBufferAddresses);
    std::set<std::string> exts;
    for (const Ins& in : g.exts) exts.insert(word_str(in.w, 1));
    exts.insert("SPV_KHR_physical_storage_buffer");
    std::vector<u32> iface = interface_of(g.entries[0]);
    for (const Ins& in : s.added) {
        if (in.op() == spv::OpVariable) iface.push_back(in.w[2]);
    }
    std::sort(iface.begin(), iface.end());
    iface.erase(std::unique(iface.begin(), iface.end()), iface.end());
    std::vector<u32> ep{3 /* Geometry */, g_main};
    const std::vector<u32> name = str_words("main");
    ep.insert(ep.end(), name.begin(), name.end());
    ep.insert(ep.end(), iface.begin(), iface.end());

    std::vector<u32> outw = {0x07230203u, gs_spirv[1], gs_spirv[2], 0u, 0u};
    const auto put = [&](const Ins& in) { outw.insert(outw.end(), in.w.begin(), in.w.end()); };
    for (const u32 c : caps) put(mk(spv::OpCapability, {c}));
    for (const std::string& x : exts) put(mk(spv::OpExtension, str_words(x)));
    for (const Ins& in : g.imports) put(in);
    put(mk(spv::OpMemoryModel, {5348 /* PhysicalStorageBuffer64 */, g.model.empty() || g.model[0].w.size() < 3 ? 1u : g.model[0].w[2]}));
    put(mk(spv::OpEntryPoint, ep));
    for (const Ins& in : g.modes) put(in);
    for (const Ins& in : g.debug) put(in);
    for (const Ins& in : new_names) put(in);
    for (const Ins& in : g.annots) put(in);
    for (const Ins& in : new_annots) put(in);
    for (const Ins& in : g.globals) put(in);
    for (const Ins& in : s.added) put(in);
    for (const Ins& in : out_fn) put(in);
    outw[3] = s.next;
    res.spirv = std::move(outw);
    return res;
}

}  // namespace gcn
