#include "host/spirv_relax.h"

#include "gcn/spirv.h"

namespace spv {

namespace {

// Function-body instructions without a result id.
bool no_result(std::uint32_t op) {
    switch (op) {
    case OpNop: case 8 /* OpLine */: case 317 /* OpNoLine */: case OpFunctionEnd: case OpStore: case 63 /* OpCopyMemory */:
    case 64 /* OpCopyMemorySized */: case OpImageWrite: case OpControlBarrier: case OpMemoryBarrier: case OpAtomicStore:
    case OpLoopMerge: case OpSelectionMerge: case OpBranch: case OpBranchConditional: case OpSwitch: case OpKill:
    case OpReturn: case OpReturnValue: case OpUnreachable: case OpDemoteToHelperInvocation: case 4416 /* OpTerminateInvocation */:
        return true;
    default:
        return false;
    }
}

// Function-body instructions with a result type and id, as Module emits them;
// another opcode makes relax_contraction leave the module alone.
bool typed_result(std::uint32_t op) {
    if (op == OpLabel || no_result(op)) return false;
    if (op == OpUndef || op == OpExtInst || op == OpFunction || op == OpFunctionParameter || op == OpVariable) return true;
    if (op >= OpLoad && op <= 70 /* OpInBoundsPtrAccessChain */ && op != OpStore && op != 63 && op != 64) return true;
    if (op >= OpVectorShuffle && op <= 83 /* OpCopyObject */) return true;
    if (op >= OpSampledImage && op <= OpImageQueryLevels) return op != OpImageWrite;
    if (op >= OpConvertFToU && op <= 155 /* OpAll */) return true;  // conversions, arithmetic, dot, carries, any/all
    if (op >= OpIsNan && op <= OpBitCount) return true;
    if (op >= OpDPdx && op <= 215 /* OpFwidthCoarse */) return true;
    if (op >= OpAtomicLoad && op <= OpAtomicXor) return op != OpAtomicStore;
    if (op == OpPhi) return true;
    if (op >= OpGroupNonUniformElect && op <= OpGroupNonUniformQuadSwap) return true;
    return false;
}

bool writes_memory(std::uint32_t op) {
    return op == OpStore || op == 63 || op == 64 || (op >= OpAtomicStore && op <= OpAtomicXor);
}

}  // namespace

std::vector<std::uint32_t> relax_contraction(const std::vector<std::uint32_t>& words, RelaxStages stages, RelaxStats* stats) {
    RelaxStats local;
    RelaxStats& st = stats ? *stats : local;
    st = {};
    if (words.size() < 5 || words[0] != 0x07230203u) return words;
    struct Inst {
        std::size_t at;
        std::uint32_t op, count;
    };
    std::vector<Inst> insts;
    for (std::size_t i = 5; i < words.size();) {
        const std::uint32_t count = words[i] >> 16;
        if (count == 0 || i + count > words.size()) return words;
        insts.push_back({i, words[i] & 0xffffu, count});
        i += count;
    }
    const std::uint32_t bound = words[3];
    int model = -1;
    std::vector<std::uint32_t> builtin(bound, ~0u), pointee(bound, 0), def(bound, ~0u), root(bound, 0), var_type(bound, 0);
    std::vector<std::vector<std::uint32_t>> member_builtins(bound);  // struct type -> its members' BuiltIns
    std::vector<std::uint32_t> element(bound, 0);                     // array type -> element type
    std::vector<std::uint32_t> output_vars, conditions;
    bool in_function = false, discards = false;
    const auto id_ok = [&](std::uint32_t id) { return id != 0 && id < bound; };
    for (std::size_t k = 0; k < insts.size(); ++k) {
        const Inst& in = insts[k];
        const std::uint32_t* w = &words[in.at];
        switch (in.op) {
        case OpEntryPoint:
            if (model < 0 && in.count > 1) model = static_cast<int>(w[1]);
            continue;
        case OpDecorate:
            if (in.count >= 4 && id_ok(w[1]) && w[2] == DecBuiltIn) builtin[w[1]] = w[3];
            continue;
        case OpMemberDecorate:
            if (in.count >= 5 && id_ok(w[1]) && w[3] == DecBuiltIn) member_builtins[w[1]].push_back(w[4]);
            continue;
        case OpTypeArray:
        case OpTypeRuntimeArray:
            if (in.count >= 3 && id_ok(w[1])) element[w[1]] = w[2];
            continue;
        case OpTypePointer:
            if (in.count >= 4 && id_ok(w[1])) pointee[w[1]] = w[3];
            continue;
        case OpFunction:
            in_function = true;
            break;
        case OpFunctionEnd:
            in_function = false;
            continue;
        case OpFunctionCall:
        case OpFunctionParameter:
            return words;  // pointers may cross functions: not followed
        default:
            break;
        }
        if (in.op == OpVariable && in.count >= 4 && id_ok(w[2])) {
            root[w[2]] = w[2];
            def[w[2]] = static_cast<std::uint32_t>(k);
            if (w[3] == 3 /* Output */) output_vars.push_back(w[2]);
            var_type[w[2]] = w[1];
            continue;
        }
        if (!in_function) continue;
        if (in.op == OpLabel) continue;
        if (in.op == OpKill || in.op == OpDemoteToHelperInvocation || in.op == 4416) discards = true;
        if (in.op == OpBranchConditional && in.count >= 2) conditions.push_back(w[1]);
        if (in.op == OpSwitch && in.count >= 2) conditions.push_back(w[1]);
        if (no_result(in.op)) continue;
        if (!typed_result(in.op) || in.count < 3 || !id_ok(w[2])) return words;
        def[w[2]] = static_cast<std::uint32_t>(k);
        // Pointers derived from a variable: access chains and copies.
        if ((in.op == OpAccessChain || in.op == 66 || in.op == 67 || in.op == 70 || in.op == 83) && in.count >= 4 && id_ok(w[3]) && root[w[3]]) {
            root[w[2]] = root[w[3]];
        }
    }
    const bool fragment = model == static_cast<int>(EmFragment);
    const bool graphics = model == static_cast<int>(EmVertex) || model == static_cast<int>(EmTessellationControl) ||
                          model == static_cast<int>(EmTessellationEvaluation) || model == 3 /* Geometry */;
    if (!fragment && !(graphics && stages == RelaxStages::Graphics)) return words;
    // The outputs whose writes must stay exact.
    const auto watched = [&](std::uint32_t b) {
        return fragment ? b == BiFragDepth : (b == BiPosition || b == 3 /* ClipDistance */ || b == 4 /* CullDistance */);
    };
    std::vector<std::uint8_t> watched_root(bound, 0);
    bool any_watched = false;
    for (std::uint32_t v : output_vars) {
        bool hit = builtin[v] != ~0u && watched(builtin[v]);
        // The variable's type: pointer -> (arrays of) struct with BuiltIn members.
        std::uint32_t t = id_ok(var_type[v]) ? pointee[var_type[v]] : 0;
        for (int depth = 0; depth < 4 && id_ok(t) && element[t]; ++depth) t = element[t];
        if (id_ok(t)) {
            for (std::uint32_t b : member_builtins[t]) hit = hit || watched(b);
        }
        if (hit) watched_root[v] = 1, any_watched = true;
    }
    std::vector<std::uint8_t> in_slice(bound, 0), root_done(bound, 0);
    std::vector<std::uint32_t> work;
    const auto add = [&](std::uint32_t id) {
        if (id_ok(id) && !in_slice[id]) in_slice[id] = 1, work.push_back(id);
    };
    // Every write into a variable: its operands join the slice.
    std::vector<std::vector<std::uint32_t>> writers(bound);
    for (std::size_t k = 0; k < insts.size(); ++k) {
        const Inst& in = insts[k];
        if (!writes_memory(in.op)) continue;
        const std::uint32_t* w = &words[in.at];
        // OpStore / OpCopyMemory: the target is operand 1; atomics: the pointer follows type and result.
        const std::uint32_t ptr = (in.op == OpStore || in.op == 63 || in.op == 64 || in.op == OpAtomicStore) ? w[1] : (in.count > 3 ? w[3] : 0);
        if (!id_ok(ptr)) return words;
        if (!root[ptr]) return words;  // a pointer not traced to a variable
        writers[root[ptr]].push_back(static_cast<std::uint32_t>(k));
    }
    const auto add_operands = [&](std::size_t k) {
        const Inst& in = insts[k];
        for (std::uint32_t j = 1; j < in.count; ++j) add(words[in.at + j]);  // literals are harmless extra ids
    };
    for (std::uint32_t v = 1; v < bound; ++v) {
        if (watched_root[v]) {
            root_done[v] = 1;
            for (std::uint32_t k : writers[v]) add_operands(k);
        }
    }
    if (!fragment || any_watched || discards) {
        for (std::uint32_t c : conditions) add(c);
    }
    while (!work.empty()) {
        const std::uint32_t id = work.back();
        work.pop_back();
        if (root[id] && !root_done[root[id]]) {
            root_done[root[id]] = 1;
            for (std::uint32_t k : writers[root[id]]) add_operands(k);
        }
        if (def[id] != ~0u) add_operands(def[id]);
    }
    std::vector<std::uint32_t> out(words.begin(), words.begin() + 5);
    out.reserve(words.size());
    for (const Inst& in : insts) {
        const std::uint32_t* w = &words[in.at];
        if (in.op == OpDecorate && in.count == 3 && w[2] == DecNoContraction && id_ok(w[1])) {
            if (!in_slice[w[1]]) {
                ++st.dropped;
                continue;
            }
            ++st.kept;
        }
        out.insert(out.end(), w, w + in.count);
    }
    return out;
}

}  // namespace spv
