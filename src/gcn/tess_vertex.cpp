// gcn/tess_vertex.h: a tessellated particle draw as one vertex stage.
#include "gcn/tess_vertex.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace gcn {

std::uint32_t tess_vertex_points(std::uint32_t level) { return level <= 1 ? 6u : 12u * level; }

void tess_vertex_domain_point(std::uint32_t level, bool cw, std::uint32_t k, float& u, float& v) {
    if (level <= 1) {
        // (0,0) (1,0) (0,1) and (1,0) (1,1) (0,1): clockwise with v pointing down.
        static const float cu[6] = {0, 1, 0, 1, 1, 0}, cv[6] = {0, 0, 1, 0, 1, 1};
        k %= 6;
        std::uint32_t c = k % 3;
        if (!cw && c) c = 3 - c;  // the other two corners swapped
        const std::uint32_t at = k - k % 3 + c;
        u = cu[at];
        v = cv[at];
        return;
    }
    const std::uint32_t ring = 4 * level, per = 3 * ring;
    k %= per;
    const std::uint32_t tri = k / 3, c = k % 3;
    if (c == 0) {
        u = v = 0.5f;
        return;
    }
    // Clockwise (v down): the centre, ring point r, ring point r + 1, the ring
    // going (0,0) -> (1,0) -> (1,1) -> (0,1).
    const bool first = cw ? c == 1 : c == 2;
    const std::uint32_t r = first ? tri : (tri + 1) % ring;
    const std::uint32_t side = r / level, step = r % level;
    const float t = static_cast<float>(step) / static_cast<float>(level);
    switch (side) {
    case 0: u = t, v = 0.0f; break;
    case 1: u = 1.0f, v = t; break;
    case 2: u = 1.0f - t, v = 1.0f; break;
    default: u = 0.0f, v = 1.0f - t; break;
    }
}

namespace {

// SPIR-V numbers used here (the 1.5 specification).
enum : std::uint16_t {
    kOpSourceContinued = 2, kOpSource = 3, kOpSourceExtension = 4, kOpName = 5, kOpMemberName = 6, kOpString = 7, kOpLine = 8,
    kOpExtension = 10, kOpExtInstImport = 11, kOpExtInst = 12, kOpMemoryModel = 14, kOpEntryPoint = 15, kOpExecutionMode = 16,
    kOpCapability = 17, kOpTypeVoid = 19, kOpTypeInt = 21, kOpTypeFloat = 22, kOpTypeVector = 23, kOpTypeArray = 28,
    kOpTypeStruct = 30, kOpTypePointer = 32, kOpTypeFunction = 33, kOpTypeForwardPointer = 39, kOpConstantTrue = 41,
    kOpConstantFalse = 42, kOpConstant = 43, kOpConstantComposite = 44, kOpConstantNull = 46, kOpSpecConstant = 50,
    kOpSpecConstantOp = 52, kOpFunction = 54, kOpFunctionParameter = 55, kOpFunctionEnd = 56, kOpFunctionCall = 57,
    kOpVariable = 59, kOpImageTexelPointer = 60, kOpLoad = 61, kOpStore = 62, kOpCopyMemory = 63, kOpAccessChain = 65,
    kOpInBoundsAccessChain = 66, kOpPtrAccessChain = 67, kOpArrayLength = 68, kOpDecorate = 71, kOpMemberDecorate = 72,
    kOpDecorationGroup = 73, kOpGroupDecorate = 74, kOpGroupMemberDecorate = 75, kOpVectorShuffle = 79,
    kOpCompositeConstruct = 80, kOpCompositeExtract = 81, kOpCompositeInsert = 82, kOpCopyObject = 83, kOpUDiv = 134,
    kOpUMod = 137, kOpBitcast = 124, kOpSelect = 169, kOpPhi = 245, kOpLoopMerge = 246, kOpSelectionMerge = 247,
    kOpLabel = 248, kOpBranch = 249, kOpBranchConditional = 250, kOpSwitch = 251, kOpReturn = 253, kOpNoLine = 317,
    kOpModuleProcessed = 330, kOpExecutionModeId = 331, kOpDecorateId = 332, kOpDecorateString = 5632,
    kOpMemberDecorateString = 5633,
};
enum : std::uint32_t { kScUniformConstant = 0, kScInput = 1, kScUniform = 2, kScOutput = 3, kScPrivate = 6, kScFunction = 7, kScStorageBuffer = 12 };
enum : std::uint32_t {
    kDecBuiltIn = 11, kDecNoPerspective = 13, kDecFlat = 14, kDecPatch = 15, kDecCentroid = 16, kDecInvariant = 18,
    kDecLocation = 30, kDecComponent = 31, kDecIndex = 32, kDecBinding = 33, kDecDescriptorSet = 34,
};
enum : std::uint32_t {
    kBiPrimitiveId = 7, kBiInvocationId = 8, kBiTessLevelOuter = 11, kBiTessLevelInner = 12, kBiTessCoord = 13,
    kBiPatchVertices = 14, kBiVertexIndex = 42,
};
enum : std::uint32_t { kCapGeometry = 2, kCapTessellation = 3, kCapTessellationPointSize = 23, kCapGeometryPointSize = 24, kCapFloat16 = 9 };

struct Ins {
    std::uint16_t op = 0;
    std::vector<std::uint32_t> w;  // the operands (the words after the first)
};

struct Module {
    std::uint32_t version = 0, bound = 0;
    std::vector<Ins> ins;
};

bool parse(const std::vector<std::uint32_t>& words, Module& m, std::string& err) {
    if (words.size() < 5 || words[0] != 0x07230203u) {
        err = "not a SPIR-V module";
        return false;
    }
    m.version = words[1];
    m.bound = words[3];
    for (std::size_t at = 5; at < words.size();) {
        const std::uint32_t n = words[at] >> 16;
        if (n == 0 || at + n > words.size()) {
            err = "a truncated instruction";
            return false;
        }
        Ins i;
        i.op = static_cast<std::uint16_t>(words[at] & 0xffff);
        i.w.assign(words.begin() + static_cast<std::ptrdiff_t>(at) + 1, words.begin() + static_cast<std::ptrdiff_t>(at + n));
        m.ins.push_back(std::move(i));
        at += n;
    }
    return true;
}

// The words a literal string takes from operand `at` (NUL-terminated, padded).
std::size_t string_words(const std::vector<std::uint32_t>& w, std::size_t at) {
    for (std::size_t k = at; k < w.size(); ++k) {
        const std::uint32_t x = w[k];
        if (!(x & 0xff) || !(x & 0xff00) || !(x & 0xff0000) || !(x & 0xff000000)) return k - at + 1;
    }
    return w.size() - at;
}
std::string string_at(const std::vector<std::uint32_t>& w, std::size_t at) {
    std::string s;
    for (std::size_t k = at; k < w.size(); ++k) {
        for (int b = 0; b < 4; ++b) {
            const char c = static_cast<char>((w[k] >> (8 * b)) & 0xff);
            if (!c) return s;
            s.push_back(c);
        }
    }
    return s;
}

// Which operands are ids; false for an opcode outside the translator's and
// lifter's set, refused rather than risk renumbering a literal.
bool id_operands(const Ins& in, std::vector<bool>& ids) {
    const std::size_t n = in.w.size();
    ids.assign(n, false);
    const auto all = [&](std::size_t from) {
        for (std::size_t k = from; k < n; ++k) ids[k] = true;
    };
    const auto upto = [&](std::size_t end) {  // ids before `end`, literals from it
        for (std::size_t k = 0; k < n && k < end; ++k) ids[k] = true;
    };
    // Image instructions: ids, the image-operands mask at `mask`, ids after it.
    const auto image = [&](std::size_t mask) {
        all(0);
        if (mask < n) ids[mask] = false;
    };
    const std::uint16_t op = in.op;
    switch (op) {
    // Debug and module-level instructions.
    case kOpName: case kOpMemberName: case kOpDecorate: case kOpMemberDecorate: case kOpDecorateString: case kOpMemberDecorateString:
    case kOpExecutionMode: case kOpExtInstImport: case kOpString: case kOpTypeVoid: case 20 /* TypeBool */: case kOpTypeInt:
    case kOpTypeFloat: case 26 /* TypeSampler */: case kOpLabel:
        upto(1);
        return true;
    case kOpCapability: case kOpExtension: case kOpMemoryModel: case kOpSource: case kOpSourceExtension: case kOpSourceContinued:
    case kOpModuleProcessed: case kOpNoLine: case 0 /* Nop */: case kOpFunctionEnd: case 252 /* Kill */: case kOpReturn:
    case 255 /* Unreachable */: case 5380 /* DemoteToHelperInvocation */: case 4416 /* TerminateInvocation */:
        return true;
    case kOpLine:
        upto(1);
        return true;
    case kOpEntryPoint: {
        if (n < 3) return false;
        ids[1] = true;
        all(2 + string_words(in.w, 2));
        return true;
    }
    case kOpExecutionModeId: case kOpDecorateId:
        all(0);
        if (n > 1) ids[1] = false;
        return true;
    case kOpTypeVector: case 24 /* TypeMatrix */:
        upto(2);
        return true;
    case 25 /* TypeImage */:
        upto(2);
        return true;
    case 27 /* TypeSampledImage */: case kOpTypeArray: case 29 /* TypeRuntimeArray */: case kOpTypeStruct: case kOpTypeFunction:
        all(0);
        return true;
    case kOpTypePointer:
        ids[0] = true;
        if (n > 2) ids[2] = true;
        return true;
    case kOpTypeForwardPointer:
        upto(1);
        return true;
    case kOpConstant: case kOpSpecConstant: case kOpConstantTrue: case kOpConstantFalse: case kOpConstantNull: case 1 /* Undef */:
    case 48 /* SpecConstantTrue */: case 49 /* SpecConstantFalse */: case kOpFunctionParameter: case 5381 /* IsHelperInvocationEXT */:
        upto(2);
        return true;
    case kOpSpecConstantOp:
        all(0);
        if (n > 2) ids[2] = false;
        return true;
    case kOpVariable:
        upto(2);
        if (n > 3) ids[3] = true;
        return true;
    case kOpFunction:
        upto(2);
        if (n > 3) ids[3] = true;
        return true;
    case kOpExtInst:
        all(0);
        if (n > 3) ids[3] = false;
        return true;
    case kOpCompositeExtract:
        upto(3);
        return true;
    case kOpCompositeInsert: case kOpVectorShuffle:
        upto(4);
        return true;
    case kOpArrayLength:
        upto(3);
        return true;
    case kOpLoad: {
        // A memory-access mask with only None / Volatile / Aligned / Nontemporal:
        // what follows it is literals.
        if (n > 3 && (in.w[3] & ~0x7u)) return false;
        upto(3);
        return true;
    }
    case kOpStore: {
        if (n > 2 && (in.w[2] & ~0x7u)) return false;
        upto(2);
        return true;
    }
    case kOpCopyMemory: {
        if (n > 2 && (in.w[2] & ~0x7u)) return false;
        upto(2);
        return true;
    }
    case kOpSelectionMerge:
        upto(1);
        return true;
    case kOpLoopMerge:
        upto(2);
        return true;
    case kOpBranchConditional:
        upto(3);
        return true;
    case kOpSwitch:
        // A 32-bit selector: (literal, label) pairs after the default.
        if (n < 2) return false;
        ids[0] = ids[1] = true;
        for (std::size_t k = 3; k < n; k += 2) ids[k] = true;
        return true;
    case 87: case 88: case 95: case 98:  // ImageSample{Implicit,Explicit}Lod, ImageFetch, ImageRead
        image(4);
        return true;
    case 89: case 90: case 96: case 97:  // ImageSampleDref*, ImageGather, ImageDrefGather
        image(5);
        return true;
    case 99:  // ImageWrite
        image(3);
        return true;
    case 342:  // GroupNonUniformBallotBitCount: the group operation is a literal
        all(0);
        if (n > 3) ids[3] = false;
        return true;
    default:
        break;
    }
    if (op >= 349 && op <= 364) {  // GroupNonUniform arithmetic: the group operation is a literal
        all(0);
        if (n > 3) ids[3] = false;
        return true;
    }
    // Everything else these modules hold: every operand an id.
    const bool every_id =
        op == kOpFunctionCall || op == kOpImageTexelPointer || op == kOpAccessChain || op == kOpInBoundsAccessChain ||
        op == kOpPtrAccessChain || op == kOpConstantComposite || op == 51 /* SpecConstantComposite */ || op == 77 || op == 78 ||
        op == kOpCompositeConstruct || op == kOpCopyObject || op == 84 || op == 86 || (op >= 100 && op <= 107) ||
        (op >= 109 && op <= 124) || (op >= 126 && op <= 152) || (op >= 154 && op <= 205) || (op >= 207 && op <= 215) ||
        op == 224 || op == 225 || (op >= 227 && op <= 242) || op == kOpPhi || op == kOpBranch || op == 254 /* ReturnValue */ ||
        (op >= 333 && op <= 366) || op == 400 /* CopyLogical */;
    if (!every_id) return false;
    all(0);
    return true;
}

bool is_type(std::uint16_t op) { return (op >= kOpTypeVoid && op <= kOpTypeFunction) || op == kOpTypeForwardPointer; }
bool is_constant(std::uint16_t op) { return op >= kOpConstantTrue && op <= kOpSpecConstantOp; }
// Types and the constants that may be shared between the two modules.
bool dedupable(std::uint16_t op) {
    return (is_type(op) && op != kOpTypeForwardPointer) || op == kOpConstantTrue || op == kOpConstantFalse || op == kOpConstant ||
           op == kOpConstantComposite || op == kOpConstantNull;
}

// The parts of the merged module, in their layout order.
struct Out {
    std::vector<std::uint32_t> caps;
    std::vector<std::string> exts;
    std::vector<Ins> imports, names, decorations, globals, functions;
    std::deque<Ins> late_globals;  // a deque: Fuser::defs_ points into it as it grows
    std::vector<std::uint32_t> memory_model;
};

class Fuser {
public:
    Fuser(const Module& ls, const Module& ds) : a_(ls), b_(ds) {}

    bool run(std::uint32_t level, bool cw, std::vector<std::uint32_t>& out, std::string& err);

private:
    const Module& a_;
    Module b_;  // renumbered
    std::uint32_t next_ = 1;
    Out o_;
    std::unordered_map<std::uint32_t, std::uint32_t> repl_;  // a B id the merge replaced -> the id kept
    std::map<std::vector<std::uint32_t>, std::uint32_t> keys_;  // a type or constant's (op, operands, decorations) -> its id
    std::unordered_map<std::uint32_t, const Ins*> defs_;          // id -> the global instruction defining it (both modules)
    std::unordered_map<std::uint32_t, std::vector<std::vector<std::uint32_t>>> decos_;  // id -> its decorations (from both)
    std::unordered_set<std::uint32_t> private_ptrs_;  // pointers the merge turned Private: the converted variables and their chains
    std::string err_;

    std::uint32_t map(std::uint32_t id) const {
        const auto it = repl_.find(id);
        return it == repl_.end() ? id : it->second;
    }
    bool fail(const std::string& why) {
        if (err_.empty()) err_ = why;
        return false;
    }
    std::vector<std::uint32_t> key_of(const Ins& in) const {
        std::vector<std::uint32_t> k;
        k.push_back(in.op);
        std::vector<bool> ids;
        id_operands(in, ids);
        // The result id (operand 0 for a type, 1 for a constant) is not part of what the definition is.
        const std::size_t result = is_type(in.op) ? 0 : 1;
        for (std::size_t i = 0; i < in.w.size(); ++i) {
            if (i == result) continue;
            k.push_back(ids[i] ? map(in.w[i]) : in.w[i]);
        }
        k.push_back(0xffffffffu);
        const auto it = decos_.find(in.w[result]);
        if (it != decos_.end()) {
            std::vector<std::vector<std::uint32_t>> d = it->second;
            std::sort(d.begin(), d.end());
            for (const auto& x : d) {
                k.insert(k.end(), x.begin(), x.end());
                k.push_back(0xfffffffeu);
            }
        }
        return k;
    }
    // A type or constant the merged module has, or a new one.
    std::uint32_t make(std::uint16_t op, std::vector<std::uint32_t> operands_without_result, bool is_const_result_type = false) {
        Ins in;
        in.op = op;
        const std::uint32_t id = next_++;
        if (is_const_result_type) {
            in.w.push_back(operands_without_result.front());
            in.w.push_back(id);
            in.w.insert(in.w.end(), operands_without_result.begin() + 1, operands_without_result.end());
        } else {
            in.w.push_back(id);
            in.w.insert(in.w.end(), operands_without_result.begin(), operands_without_result.end());
        }
        const std::vector<std::uint32_t> k = key_of(in);
        if (const auto it = keys_.find(k); it != keys_.end()) {
            --next_;
            return it->second;
        }
        keys_.emplace(k, id);
        o_.late_globals.push_back(in);
        defs_[id] = &o_.late_globals.back();
        return id;
    }
    std::uint32_t private_pointer(std::uint32_t pointer_type) {
        const auto it = defs_.find(pointer_type);
        if (it == defs_.end() || !it->second || it->second->op != kOpTypePointer) return 0;
        if (it->second->w[1] == kScPrivate) return pointer_type;
        return make(kOpTypePointer, {kScPrivate, it->second->w[2]});
    }
    std::uint32_t pointee(std::uint32_t pointer_type) const {
        const auto it = defs_.find(pointer_type);
        return it == defs_.end() || !it->second || it->second->op != kOpTypePointer ? 0 : it->second->w[2];
    }
    bool decorated(std::uint32_t id, std::uint32_t deco, std::uint32_t* value = nullptr) const {
        const auto it = decos_.find(id);
        if (it == decos_.end()) return false;
        for (const auto& d : it->second) {
            if (d.size() >= 2 && d[0] == 0 && d[1] == deco) {  // (member marker 0 = the id itself)
                if (value) *value = d.size() > 2 ? d[2] : 0;
                return true;
            }
        }
        return false;
    }
    bool renumber_b(std::uint32_t offset);
    bool function_body(const Module& m, bool from_b, std::uint32_t main_id, std::vector<Ins>& out);
};

bool Fuser::renumber_b(std::uint32_t offset) {
    for (Ins& in : b_.ins) {
        std::vector<bool> ids;
        if (!id_operands(in, ids)) return fail("an instruction the merge does not know (opcode " + std::to_string(in.op) + ")");
        for (std::size_t k = 0; k < in.w.size(); ++k) {
            if (ids[k]) in.w[k] += offset;
        }
    }
    return true;
}

// A function's instructions with B's ids mapped to the ones kept, and every
// pointer derived from a variable turned Private typed Private.
bool Fuser::function_body(const Module& m, bool from_b, std::uint32_t, std::vector<Ins>& out) {
    bool in_function = false;
    for (const Ins& src : m.ins) {
        if (src.op == kOpFunction) in_function = true;
        if (!in_function) continue;
        if (src.op == kOpLine || src.op == kOpNoLine) continue;
        Ins in = src;
        std::vector<bool> ids;
        if (!id_operands(in, ids)) return fail("an instruction the merge does not know (opcode " + std::to_string(in.op) + ")");
        if (from_b) {
            for (std::size_t k = 0; k < in.w.size(); ++k) {
                if (ids[k]) in.w[k] = map(in.w[k]);
            }
        }
        switch (in.op) {
        case kOpAccessChain: case kOpInBoundsAccessChain: case kOpPtrAccessChain: case kOpCopyObject:
            if (in.w.size() >= 3 && private_ptrs_.count(in.w[2])) {
                const std::uint32_t t = private_pointer(in.w[0]);
                if (!t) return fail("a chain into a stage variable whose type is not a pointer");
                in.w[0] = t;
                private_ptrs_.insert(in.w[1]);
            }
            break;
        case kOpFunctionCall:
            for (std::size_t k = 3; k < in.w.size(); ++k) {
                if (private_ptrs_.count(in.w[k])) return fail("a stage variable passed to a function");
            }
            break;
        case kOpSelect: case kOpPhi:
            for (std::size_t k = 2; k < in.w.size(); ++k) {
                if (private_ptrs_.count(in.w[k])) return fail("a stage variable's pointer selected");
            }
            break;
        default:
            break;
        }
        out.push_back(std::move(in));
        if (src.op == kOpFunctionEnd) in_function = false;
    }
    return true;
}

bool Fuser::run(std::uint32_t level, bool cw, std::vector<std::uint32_t>& words, std::string& err) {
    next_ = a_.bound + b_.bound;  // B's ids move up by a_.bound - 1
    if (!renumber_b(a_.bound - 1)) return err = err_, false;

    // Every instruction of A must be one the merge knows too.
    for (const Ins& in : a_.ins) {
        std::vector<bool> ids;
        if (!id_operands(in, ids)) return err = "an instruction the merge does not know (opcode " + std::to_string(in.op) + ")", false;
    }
    std::uint32_t a_main = 0, b_main = 0, a_model = ~0u, b_model = ~0u;
    std::vector<std::uint32_t> a_modes, b_modes;  // float-control execution modes (mode, literals...)
    for (int which = 0; which < 2; ++which) {
        const Module& m = which ? b_ : a_;
        for (const Ins& in : m.ins) {
            if (in.op == kOpDecorate && !in.w.empty()) {
                std::vector<std::uint32_t> d = {0};
                d.insert(d.end(), in.w.begin() + 1, in.w.end());
                decos_[in.w[0]].push_back(d);
            } else if (in.op == kOpMemberDecorate && in.w.size() >= 2) {
                std::vector<std::uint32_t> d = {1};
                d.insert(d.end(), in.w.begin() + 1, in.w.end());
                decos_[in.w[0]].push_back(d);
            } else if (in.op == kOpDecorationGroup || in.op == kOpGroupDecorate || in.op == kOpGroupMemberDecorate ||
                       in.op == kOpDecorateId || in.op == kOpDecorateString || in.op == kOpMemberDecorateString) {
                return err = "decoration groups or id/string decorations", false;
            } else if (in.op == kOpEntryPoint) {
                if (which ? b_main : a_main) return err = "more than one entry point", false;
                (which ? b_main : a_main) = in.w[1];
                (which ? b_model : a_model) = in.w[0];
            } else if (in.op == kOpExecutionMode && in.w.size() >= 2) {
                const std::uint32_t mode = in.w[1];
                if (mode >= 4459 && mode <= 4463) {
                    auto& v = which ? b_modes : a_modes;
                    v.insert(v.end(), in.w.begin() + 1, in.w.end());
                    v.push_back(~0u);
                }
            }
        }
    }
    if (a_model != 0 /* Vertex */) return err = "the LS module is not a vertex stage", false;
    if (b_model != 2 /* TessellationEvaluation */) return err = "the domain module is not an evaluation stage", false;

    std::set<std::uint32_t> caps;
    std::set<std::string> exts;
    bool a_half = false, b_half = false;
    std::map<std::string, std::uint32_t> imports;
    for (int which = 0; which < 2; ++which) {
        const Module& m = which ? b_ : a_;
        for (const Ins& in : m.ins) {
            if (in.op == kOpCapability) {
                const std::uint32_t c = in.w[0];
                if (c == kCapFloat16) (which ? b_half : a_half) = true;
                if (c != kCapTessellation && c != kCapGeometry && c != kCapTessellationPointSize && c != kCapGeometryPointSize) caps.insert(c);
            } else if (in.op == kOpExtension) {
                exts.insert(string_at(in.w, 0));
            } else if (in.op == kOpExtInstImport) {
                const std::string name = string_at(in.w, 1);
                const auto it = imports.find(name);
                if (it != imports.end()) {
                    repl_[in.w[0]] = it->second;
                } else {
                    imports.emplace(name, in.w[0]);
                    o_.imports.push_back(in);
                }
            } else if (in.op == kOpMemoryModel) {
                if (!which) {
                    o_.memory_model = in.w;
                } else if (in.w != o_.memory_model) {
                    return err = "the modules' memory models differ", false;
                }
            }
        }
    }
    // Float controls apply to a whole entry point: the domain shader's and the
    // LS's must agree wherever both compute in half precision.
    if (a_modes != b_modes && ((a_modes.empty() && a_half) || (b_modes.empty() && b_half) || (!a_modes.empty() && !b_modes.empty()))) {
        return err = "the modules' float controls differ", false;
    }
    const std::vector<std::uint32_t>& modes = a_modes.empty() ? b_modes : a_modes;

    // Globals: B's types, constants, descriptors and built-in inputs shared
    // where A has the same.
    struct Var {
        std::uint32_t id, type, sc;
        bool from_b;
    };
    std::vector<Var> vars;
    std::vector<std::uint32_t> convert;  // variables turned Private, declared last
    std::uint32_t a_vid = 0, a_cp = 0, b_tc = 0, b_pid = 0, b_cp = 0;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> bindings;  // (set, binding) -> variable
    std::vector<Ins> globals;
    globals.reserve(a_.ins.size() + b_.ins.size());
    for (int which = 0; which < 2; ++which) {
        const Module& m = which ? b_ : a_;
        for (const Ins& src : m.ins) {
            if (src.op == kOpFunction) break;
            if (!(is_type(src.op) || is_constant(src.op) || src.op == kOpVariable || src.op == 1 /* Undef */)) continue;
            Ins in = src;
            std::vector<bool> ids;
            id_operands(in, ids);
            if (which) {
                for (std::size_t k = 0; k < in.w.size(); ++k) {
                    if (ids[k]) in.w[k] = map(in.w[k]);
                }
            }
            if (in.op == kOpTypeForwardPointer) return err = "a forward pointer", false;
            if (dedupable(in.op)) {
                const std::vector<std::uint32_t> k = key_of(in);
                const std::uint32_t result = is_type(in.op) ? in.w[0] : in.w[1];
                if (const auto it = keys_.find(k); it != keys_.end()) {
                    if (!which) return err = "the LS module declares a type twice", false;
                    repl_[result] = it->second;
                    continue;
                }
                keys_.emplace(k, result);
                globals.push_back(in);
                continue;
            }
            if (in.op != kOpVariable) {
                globals.push_back(in);
                continue;
            }
            const std::uint32_t id = in.w[1], type = in.w[0], sc = in.w[2];
            std::uint32_t builtin = ~0u, location = ~0u, set = ~0u, binding = ~0u;
            decorated(id, kDecBuiltIn, &builtin);
            decorated(id, kDecLocation, &location);
            decorated(id, kDecDescriptorSet, &set);
            decorated(id, kDecBinding, &binding);
            bool to_private = false;
            if (!which) {
                if (sc == kScOutput) {
                    to_private = true;
                    if (location == 0) a_cp = id;
                } else if (sc == kScInput) {
                    if (builtin == kBiVertexIndex) {
                        to_private = true;
                        a_vid = id;
                    } else if (builtin == ~0u) {
                        return err = "the LS reads vertex input", false;
                    }
                }
            } else {
                if (sc == kScInput) {
                    if (builtin == kBiTessCoord) {
                        to_private = true;
                        b_tc = id;
                    } else if (builtin == kBiPrimitiveId) {
                        to_private = true;
                        b_pid = id;
                    } else if (builtin == kBiPatchVertices || builtin == kBiTessLevelOuter || builtin == kBiTessLevelInner ||
                               builtin == kBiInvocationId) {
                        return err = "the domain shader reads a tessellation built-in other than its coordinate and patch", false;
                    } else if (builtin == ~0u) {
                        if (location != 0) return err = "the domain shader reads an input other than the control point", false;
                        to_private = true;
                        b_cp = id;
                    }
                }
                // A descriptor or built-in input A has the same of: shared.
                const bool shareable = sc == kScUniform || sc == kScUniformConstant || sc == kScStorageBuffer ||
                                       (sc == kScInput && builtin != ~0u && !to_private);
                if (shareable && in.w.size() == 3) {
                    bool merged = false;
                    for (const Var& v : vars) {
                        if (v.from_b || v.type != type || v.sc != sc) continue;
                        std::vector<std::vector<std::uint32_t>> da = decos_[v.id], db = decos_[src.w[1]];
                        // Names are not decorations; NonWritable and the like must agree too.
                        std::sort(da.begin(), da.end());
                        std::sort(db.begin(), db.end());
                        if (da != db) continue;
                        repl_[id] = v.id;
                        merged = true;
                        break;
                    }
                    if (merged) continue;
                }
            }
            if (set != ~0u && binding != ~0u) {
                const auto [it, fresh] = bindings.emplace(std::make_pair(set, binding), id);
                if (!fresh) return err = "two descriptors at set " + std::to_string(set) + " binding " + std::to_string(binding) + " differ", false;
            }
            vars.push_back({id, type, sc, which != 0});
            defs_[id] = nullptr;
            if (to_private) {
                convert.push_back(id);
                private_ptrs_.insert(id);
                Ins v = in;
                globals.push_back(std::move(v));  // placeholder: rewritten below and moved last
                globals.back().op = 0xffff;
                continue;
            }
            globals.push_back(in);
        }
    }
    if (b_cp && !a_cp) return err = "the domain shader reads a control point the LS does not write", false;
    for (const Ins& in : globals) {
        if (in.op == 0xffff) continue;
        const std::uint32_t result = is_type(in.op) ? in.w[0] : in.w.size() > 1 ? in.w[1] : 0;
        if (result) defs_[result] = &in;
    }
    // The converted variables, Private, after every type they now need.
    std::vector<Ins> converted;
    std::unordered_map<std::uint32_t, std::uint32_t> conv_type;  // variable -> its Private pointer type
    for (const Ins& in : globals) {
        if (in.op != 0xffff) continue;
        Ins v = in;
        v.op = kOpVariable;
        v.w[0] = private_pointer(v.w[0]);
        v.w[2] = kScPrivate;
        if (!v.w[0]) return err = "a stage variable whose type is not a pointer", false;
        conv_type[v.w[1]] = v.w[0];
        converted.push_back(std::move(v));
    }
    globals.erase(std::remove_if(globals.begin(), globals.end(), [](const Ins& in) { return in.op == 0xffff; }), globals.end());
    for (const Ins& in : globals) {
        const std::uint32_t result = is_type(in.op) ? in.w[0] : in.w.size() > 1 ? in.w[1] : 0;
        if (result) defs_[result] = &in;
    }

    std::vector<Ins> fa, fb;
    if (!function_body(a_, false, a_main, fa) || !function_body(b_, true, b_main, fb)) return err = err_, false;

    const std::uint32_t t_void = make(kOpTypeVoid, {});
    const std::uint32_t t_fn = make(kOpTypeFunction, {t_void});
    const std::uint32_t t_u32 = make(kOpTypeInt, {32, 0});
    const std::uint32_t t_f32 = make(kOpTypeFloat, {32});
    const std::uint32_t t_v3f = make(kOpTypeVector, {t_f32, 3});
    const auto cu = [&](std::uint32_t v) { return make(kOpConstant, {t_u32, v}, true); };
    const auto cf = [&](float f) {
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        return make(kOpConstant, {t_f32, bits}, true);
    };
    const std::uint32_t points = tess_vertex_points(level);
    // The domain points, as two tables indexed by the vertex's place in its patch.
    std::vector<std::uint32_t> us = {}, vs = {};
    for (std::uint32_t k = 0; k < points; ++k) {
        float u, v;
        tess_vertex_domain_point(level, cw, k, u, v);
        us.push_back(cf(u));
        vs.push_back(cf(v));
    }
    const std::uint32_t c_points = cu(points);
    const std::uint32_t t_table = make(kOpTypeArray, {t_f32, c_points});
    std::vector<std::uint32_t> uc = {t_table}, vc = {t_table};
    uc.insert(uc.end(), us.begin(), us.end());
    vc.insert(vc.end(), vs.begin(), vs.end());
    const std::uint32_t c_us = make(kOpConstantComposite, uc, true), c_vs = make(kOpConstantComposite, vc, true);
    const std::uint32_t p_table = make(kOpTypePointer, {kScPrivate, t_table});
    const std::uint32_t p_f32 = make(kOpTypePointer, {kScPrivate, t_f32});
    const std::uint32_t p_in_u32 = make(kOpTypePointer, {kScInput, t_u32});
    std::vector<Ins> fresh_vars;
    const auto variable = [&](std::uint32_t type, std::uint32_t sc, std::uint32_t init) {
        Ins v;
        v.op = kOpVariable;
        const std::uint32_t id = next_++;
        v.w = {type, id, sc};
        if (init) v.w.push_back(init);
        fresh_vars.push_back(v);
        return id;
    };
    const std::uint32_t tab_u = variable(p_table, kScPrivate, c_us), tab_v = variable(p_table, kScPrivate, c_vs);
    const std::uint32_t vin = variable(p_in_u32, kScInput, 0);
    std::vector<Ins> fm;
    const auto emit = [&](std::uint16_t op, std::vector<std::uint32_t> w) {
        Ins in;
        in.op = op;
        in.w = std::move(w);
        fm.push_back(std::move(in));
    };
    const std::uint32_t fused = next_++;
    emit(kOpFunction, {t_void, fused, 0, t_fn});
    emit(kOpLabel, {next_++});
    const std::uint32_t vi = next_++;
    emit(kOpLoad, {t_u32, vi, vin});
    const std::uint32_t patch = next_++, k = next_++;
    emit(kOpUDiv, {t_u32, patch, vi, c_points});
    emit(kOpUMod, {t_u32, k, vi, c_points});
    if (a_vid) {
        // The LS's vertex index is its patch's control point: the draw is not indexed.
        const std::uint32_t t = pointee(conv_type[a_vid]);
        std::uint32_t value = patch;
        if (t != t_u32) {
            value = next_++;
            emit(kOpBitcast, {t, value, patch});
        }
        emit(kOpStore, {a_vid, value});
    }
    emit(kOpFunctionCall, {t_void, next_++, a_main});
    if (b_cp) {
        // The domain shader's control point array (one point) from the LS's output.
        const std::uint32_t ta = pointee(conv_type[a_cp]), tb = pointee(conv_type[b_cp]);
        const auto bd = defs_.find(tb);
        if (bd == defs_.end() || !bd->second || bd->second->op != kOpTypeArray || bd->second->w[1] != ta) {
            return err = "the domain shader's control point is not an array of the LS's", false;
        }
        const std::uint32_t c0 = cu(0);
        const std::uint32_t cp = next_++, dst = next_++;
        emit(kOpLoad, {ta, cp, a_cp});
        emit(kOpAccessChain, {make(kOpTypePointer, {kScPrivate, ta}), dst, b_cp, c0});
        emit(kOpStore, {dst, cp});
    }
    const std::uint32_t pu = next_++, pv = next_++, u = next_++, v = next_++;
    emit(kOpAccessChain, {p_f32, pu, tab_u, k});
    emit(kOpAccessChain, {p_f32, pv, tab_v, k});
    emit(kOpLoad, {t_f32, u, pu});
    emit(kOpLoad, {t_f32, v, pv});
    if (b_tc) {
        if (pointee(conv_type[b_tc]) != t_v3f) return err = "gl_TessCoord is not a vec3", false;
        const std::uint32_t tc = next_++;
        emit(kOpCompositeConstruct, {t_v3f, tc, u, v, cf(0.0f)});
        emit(kOpStore, {b_tc, tc});
    }
    if (b_pid) {
        // Each invocation of the domain shader reads only its own patch's control point.
        const std::uint32_t t = pointee(conv_type[b_pid]);
        emit(kOpStore, {b_pid, make(kOpConstantNull, {t}, true)});
    }
    emit(kOpFunctionCall, {t_void, next_++, b_main});
    emit(kOpReturn, {});
    emit(kOpFunctionEnd, {});
    if (!err_.empty()) return err = err_, false;

    // ---- assemble ----
    std::vector<std::uint32_t>& w = words;
    w = {0x07230203u, std::max(a_.version, b_.version), 0u, 0u, 0u};
    const auto put = [&](std::uint16_t op, const std::vector<std::uint32_t>& ops) {
        w.push_back(static_cast<std::uint32_t>(ops.size() + 1) << 16 | op);
        w.insert(w.end(), ops.begin(), ops.end());
    };
    const auto put_string = [](std::vector<std::uint32_t>& ops, const std::string& s) {
        const std::size_t n = s.size() / 4 + 1;
        const std::size_t at = ops.size();
        ops.resize(at + n, 0);
        std::memcpy(ops.data() + at, s.data(), s.size());
    };
    for (std::uint32_t c : caps) put(kOpCapability, {c});
    for (const std::string& e : exts) {
        std::vector<std::uint32_t> ops;
        put_string(ops, e);
        put(kOpExtension, ops);
    }
    for (const Ins& in : o_.imports) put(in.op, in.w);
    put(kOpMemoryModel, o_.memory_model);
    // The interface: every global variable (SPIR-V 1.4 and later want them all).
    std::vector<std::uint32_t> ep = {0 /* Vertex */, fused};
    put_string(ep, "main");
    std::set<std::uint32_t> listed;
    const auto list = [&](const auto& v) {
        for (const Ins& in : v) {
            if (in.op == kOpVariable && in.w[2] != kScFunction && listed.insert(in.w[1]).second) ep.push_back(in.w[1]);
        }
    };
    list(globals);
    list(o_.late_globals);
    list(converted);
    list(fresh_vars);
    put(kOpEntryPoint, ep);
    for (std::size_t at = 0; at < modes.size();) {
        std::vector<std::uint32_t> ops = {fused};
        while (at < modes.size() && modes[at] != ~0u) ops.push_back(modes[at++]);
        ++at;
        put(kOpExecutionMode, ops);
    }
    // Names (B's mapped; not those of what B shares with A, nor the old entry points').
    const std::set<std::uint32_t> conv(convert.begin(), convert.end());
    for (int which = 0; which < 2; ++which) {
        const Module& m = which ? b_ : a_;
        for (const Ins& in : m.ins) {
            if (in.op != kOpName && in.op != kOpMemberName) continue;
            if (which && repl_.count(in.w[0])) continue;
            if (in.w[0] == a_main || in.w[0] == b_main) continue;
            put(in.op, in.w);
        }
    }
    {
        std::vector<std::uint32_t> ops = {a_main};
        put_string(ops, "ls_main");
        put(kOpName, ops);
        ops = {b_main};
        put_string(ops, "ds_main");
        put(kOpName, ops);
        ops = {fused};
        put_string(ops, "main");
        put(kOpName, ops);
        ops = {vin};
        put_string(ops, "vertex_index");
        put(kOpName, ops);
    }
    // Decorations: the converted variables lose their stage-interface ones.
    for (int which = 0; which < 2; ++which) {
        const Module& m = which ? b_ : a_;
        for (const Ins& in : m.ins) {
            if (in.op != kOpDecorate && in.op != kOpMemberDecorate) continue;
            if (which && repl_.count(in.w[0])) continue;
            if (in.op == kOpDecorate && conv.count(in.w[0])) {
                const std::uint32_t d = in.w[1];
                if (d == kDecBuiltIn || d == kDecLocation || d == kDecComponent || d == kDecFlat || d == kDecNoPerspective ||
                    d == kDecCentroid || d == kDecInvariant || d == kDecPatch || d == kDecIndex) {
                    continue;
                }
            }
            put(in.op, in.w);
        }
    }
    put(kOpDecorate, {vin, kDecBuiltIn, kBiVertexIndex});
    for (const Ins& in : globals) put(in.op, in.w);
    for (const Ins& in : o_.late_globals) put(in.op, in.w);
    for (const Ins& in : converted) put(in.op, in.w);
    for (const Ins& in : fresh_vars) put(in.op, in.w);
    for (const Ins& in : fa) put(in.op, in.w);
    for (const Ins& in : fb) put(in.op, in.w);
    for (const Ins& in : fm) put(in.op, in.w);
    w[3] = next_;
    return true;
}

}  // namespace

TessVertexFuse fuse_tess_vertex(const std::vector<std::uint32_t>& ls, const std::vector<std::uint32_t>& ds, std::uint32_t level, bool cw) {
    TessVertexFuse r;
    if (level < 1 || level > kTessVertexMaxLevel) {
        r.error = "a level outside 1.." + std::to_string(kTessVertexMaxLevel);
        return r;
    }
    Module a, b;
    if (!parse(ls, a, r.error) || !parse(ds, b, r.error)) return r;
    Fuser f(a, b);
    if (!f.run(level, cw, r.spirv, r.error)) r.spirv.clear();
    return r;
}

}  // namespace gcn
