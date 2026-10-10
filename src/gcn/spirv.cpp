#include "gcn/spirv.h"

#include <cstring>

namespace spv {

Module::Module() = default;

std::vector<std::uint32_t> Module::encode(const Instr& in) {
    std::vector<std::uint32_t> w;
    const std::uint32_t count = 1 + (in.type ? 1 : 0) + (in.result ? 1 : 0) + static_cast<std::uint32_t>(in.operands.size());
    w.push_back((count << 16) | in.op);
    if (in.type) w.push_back(in.type);
    if (in.result) w.push_back(in.result);
    w.insert(w.end(), in.operands.begin(), in.operands.end());
    return w;
}

static void push_string(std::vector<std::uint32_t>& out, const std::string& s) {
    std::uint32_t word = 0;
    int n = 0;
    for (char c : s) {
        word |= static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << (8 * n);
        if (++n == 4) {
            out.push_back(word);
            word = 0;
            n = 0;
        }
    }
    out.push_back(word);  // includes the terminating zero
}

Id Module::add_type(const std::vector<std::uint32_t>& key, Op op, const std::vector<std::uint32_t>& operands) {
    auto it = type_cache_.find(key);
    if (it != type_cache_.end()) {
        return it->second;
    }
    const Id id = fresh();
    types_consts_globals_.push_back({op, 0, id, operands});
    type_cache_[key] = id;
    return id;
}

Id Module::type_void() { return add_type({OpTypeVoid}, OpTypeVoid, {}); }
Id Module::type_bool() { return add_type({OpTypeBool}, OpTypeBool, {}); }
Id Module::type_int(std::uint32_t width, bool is_signed) {
    return add_type({OpTypeInt, width, is_signed}, OpTypeInt, {width, is_signed ? 1u : 0u});
}
Id Module::type_float(std::uint32_t width) { return add_type({OpTypeFloat, width}, OpTypeFloat, {width}); }
Id Module::type_vector(Id component, std::uint32_t n) {
    return add_type({OpTypeVector, component, n}, OpTypeVector, {component, n});
}
Id Module::type_array(Id element, Id length_const) {
    return add_type({OpTypeArray, element, length_const}, OpTypeArray, {element, length_const});
}
Id Module::type_runtime_array(Id element) {
    return add_type({OpTypeRuntimeArray, element}, OpTypeRuntimeArray, {element});
}
Id Module::type_struct(const std::vector<Id>& members) {
    std::vector<std::uint32_t> key = {OpTypeStruct};
    key.insert(key.end(), members.begin(), members.end());
    return add_type(key, OpTypeStruct, members);
}
Id Module::type_pointer(StorageClass sc, Id pointee) {
    return add_type({OpTypePointer, sc, pointee}, OpTypePointer, {static_cast<std::uint32_t>(sc), pointee});
}
Id Module::type_function(Id ret, const std::vector<Id>& params) {
    std::vector<std::uint32_t> key = {OpTypeFunction, ret};
    key.insert(key.end(), params.begin(), params.end());
    std::vector<std::uint32_t> ops = {ret};
    ops.insert(ops.end(), params.begin(), params.end());
    return add_type(key, OpTypeFunction, ops);
}
Id Module::type_image(Id sampled, Dim dim, bool depth, bool arrayed, bool ms, std::uint32_t sampled_mode) {
    std::vector<std::uint32_t> ops = {sampled, static_cast<std::uint32_t>(dim), depth ? 1u : 0u, arrayed ? 1u : 0u,
                                      ms ? 1u : 0u, sampled_mode, 0u /* Unknown format */};
    std::vector<std::uint32_t> key = {OpTypeImage};
    key.insert(key.end(), ops.begin(), ops.end());
    return add_type(key, OpTypeImage, ops);
}
Id Module::type_sampler() { return add_type({OpTypeSampler}, OpTypeSampler, {}); }
Id Module::type_sampled_image(Id image) {
    return add_type({OpTypeSampledImage, image}, OpTypeSampledImage, {image});
}

Id Module::const_bool(bool v) {
    const Id t = type_bool();
    auto key = std::make_tuple(t, static_cast<std::uint64_t>(v), 0u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({v ? OpConstantTrue : OpConstantFalse, t, id, {}});
    const_cache_[key] = id;
    return id;
}
Id Module::const_u32(std::uint32_t v) {
    const Id t = type_int(32, false);
    auto key = std::make_tuple(t, static_cast<std::uint64_t>(v), 0u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({OpConstant, t, id, {v}});
    const_cache_[key] = id;
    return id;
}
Id Module::const_i32(std::int32_t v) {
    const Id t = type_int(32, true);
    auto key = std::make_tuple(t, static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)), 0u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({OpConstant, t, id, {static_cast<std::uint32_t>(v)}});
    const_cache_[key] = id;
    return id;
}
Id Module::const_u64(std::uint64_t v) {
    const Id t = type_int(64, false);
    auto key = std::make_tuple(t, v, 0u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({OpConstant, t, id, {static_cast<std::uint32_t>(v), static_cast<std::uint32_t>(v >> 32)}});
    const_cache_[key] = id;
    return id;
}
Id Module::const_f32(float v) {
    const Id t = type_float(32);
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    auto key = std::make_tuple(t, static_cast<std::uint64_t>(bits), 0u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({OpConstant, t, id, {bits}});
    const_cache_[key] = id;
    return id;
}
Id Module::const_composite(Id type, const std::vector<Id>& members) {
    std::uint64_t h = type;
    for (Id m : members) h = h * 1000003u + m;
    auto key = std::make_tuple(type, h, static_cast<std::uint32_t>(members.size()) | 0x80000000u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({OpConstantComposite, type, id, members});
    const_cache_[key] = id;
    return id;
}
Id Module::const_null(Id type) {
    auto key = std::make_tuple(type, static_cast<std::uint64_t>(0), 0x40000000u);
    auto it = const_cache_.find(key);
    if (it != const_cache_.end()) return it->second;
    const Id id = fresh();
    types_consts_globals_.push_back({OpConstantNull, type, id, {}});
    const_cache_[key] = id;
    return id;
}

Id Module::spec_const_u32(std::uint32_t value, std::uint32_t spec_id) {
    const Id id = fresh();
    types_consts_globals_.push_back({OpSpecConstant, type_int(32, false), id, {value}});
    decorate(id, DecSpecId, {spec_id});
    return id;
}

void Module::capability(Capability c) {
    for (std::uint32_t existing : capabilities_) {
        if (existing == c) return;
    }
    capabilities_.push_back(c);
}
void Module::extension(const std::string& n) {
    for (const std::string& e : extensions_) {
        if (e == n) return;
    }
    extensions_.push_back(n);
}
Id Module::ext_glsl() {
    if (!glsl_) {
        glsl_ = fresh();
    }
    return glsl_;
}
void Module::memory_model(std::uint32_t addressing, std::uint32_t memory) {
    addressing_ = addressing;
    memory_ = memory;
}
void Module::entry_point(ExecutionModel model, Id func, const std::string& n, const std::vector<Id>& interface) {
    std::vector<std::uint32_t> ops = {static_cast<std::uint32_t>(model), func};
    push_string(ops, n);
    ops.insert(ops.end(), interface.begin(), interface.end());
    entry_points_.push_back({OpEntryPoint, 0, 0, ops});
}
void Module::execution_mode(Id func, ExecutionMode mode, const std::vector<std::uint32_t>& operands) {
    std::vector<std::uint32_t> ops = {func, static_cast<std::uint32_t>(mode)};
    ops.insert(ops.end(), operands.begin(), operands.end());
    execution_modes_.push_back({OpExecutionMode, 0, 0, ops});
}
void Module::decorate(Id target, Decoration d, const std::vector<std::uint32_t>& operands) {
    std::vector<std::uint32_t> ops = {target, static_cast<std::uint32_t>(d)};
    ops.insert(ops.end(), operands.begin(), operands.end());
    decorations_.push_back({OpDecorate, 0, 0, ops});
}
void Module::member_decorate(Id type, std::uint32_t member, Decoration d, const std::vector<std::uint32_t>& operands) {
    std::vector<std::uint32_t> ops = {type, member, static_cast<std::uint32_t>(d)};
    ops.insert(ops.end(), operands.begin(), operands.end());
    decorations_.push_back({OpMemberDecorate, 0, 0, ops});
}
void Module::name(Id target, const std::string& n) {
    std::vector<std::uint32_t> ops = {target};
    push_string(ops, n);
    names_.push_back({OpName, 0, 0, ops});
}
Id Module::global_variable(Id ptr_type, StorageClass sc, Id initializer) {
    const Id id = fresh();
    std::vector<std::uint32_t> ops = {static_cast<std::uint32_t>(sc)};
    if (initializer) ops.push_back(initializer);
    types_consts_globals_.push_back({OpVariable, ptr_type, id, ops});
    return id;
}

Id Module::begin_function(Id ret_type, Id fn_type) {
    const Id id = fresh();
    in_function_ = true;
    local_vars_.clear();
    entry_.clear();
    to_entry_ = false;
    body_.clear();
    body_.push_back({OpFunction, ret_type, id, {0u, fn_type}});
    return id;
}
void Module::end_function() {
    // Hoist local variables into the first block, right after its label.
    std::vector<Instr> out;
    out.reserve(body_.size() + local_vars_.size() + 1);
    bool hoisted = false;
    for (const Instr& in : body_) {
        out.push_back(in);
        if (!hoisted && in.op == OpLabel) {
            out.insert(out.end(), local_vars_.begin(), local_vars_.end());
            out.insert(out.end(), entry_.begin(), entry_.end());
            hoisted = true;
        }
    }
    out.push_back({OpFunctionEnd, 0, 0, {}});
    functions_.insert(functions_.end(), out.begin(), out.end());
    in_function_ = false;
}
Id Module::local_variable(Id ptr_type, Id initializer) {
    const Id id = fresh();
    std::vector<std::uint32_t> ops = {static_cast<std::uint32_t>(ScFunction)};
    if (initializer) ops.push_back(initializer);
    local_vars_.push_back({OpVariable, ptr_type, id, ops});
    return id;
}
Id Module::label(Id id) {
    if (!id) id = fresh();
    ++blocks_;
    body_.push_back({OpLabel, 0, id, {}});
    return id;
}
Id Module::emit(Op op, Id result_type, const std::vector<std::uint32_t>& operands) {
    const Id id = fresh();
    (to_entry_ ? entry_ : body_).push_back({op, result_type, id, operands});
    if (op == OpFNegate || op == OpFAdd || op == OpFSub || op == OpFMul || op == OpFDiv) decorate(id, DecNoContraction);
    return id;
}
void Module::emit_void(Op op, const std::vector<std::uint32_t>& operands) {
    (to_entry_ ? entry_ : body_).push_back({op, 0, 0, operands});
}
bool Module::set_operand(Id result, std::size_t index, std::uint32_t value) {
    for (auto it = body_.rbegin(); it != body_.rend(); ++it) {
        if (it->result != result) continue;
        if (index >= it->operands.size()) return false;
        it->operands[index] = value;
        return true;
    }
    return false;
}
Id Module::ext_inst(Id result_type, Glsl inst, const std::vector<Id>& operands) {
    std::vector<std::uint32_t> ops = {ext_glsl(), static_cast<std::uint32_t>(inst)};
    ops.insert(ops.end(), operands.begin(), operands.end());
    return emit(OpExtInst, result_type, ops);
}
Id Module::access_chain(Id ptr_type, Id base, const std::vector<Id>& indices) {
    std::vector<std::uint32_t> ops = {base};
    ops.insert(ops.end(), indices.begin(), indices.end());
    return emit(OpAccessChain, ptr_type, ops);
}

std::vector<std::uint32_t> Module::assemble() const {
    std::vector<std::uint32_t> out = {0x07230203u, 0x00010500u /* SPIR-V 1.5 */, 0u, next_id_, 0u};
    for (std::uint32_t c : capabilities_) {
        out.push_back((2u << 16) | OpCapability);
        out.push_back(c);
    }
    for (const std::string& e : extensions_) {
        std::vector<std::uint32_t> ops;
        push_string(ops, e);
        out.push_back(((1u + static_cast<std::uint32_t>(ops.size())) << 16) | OpExtension);
        out.insert(out.end(), ops.begin(), ops.end());
    }
    if (glsl_) {
        std::vector<std::uint32_t> ops;
        push_string(ops, "GLSL.std.450");
        out.push_back(((2u + static_cast<std::uint32_t>(ops.size())) << 16) | OpExtInstImport);
        out.push_back(glsl_);
        out.insert(out.end(), ops.begin(), ops.end());
    }
    out.push_back((3u << 16) | OpMemoryModel);
    out.push_back(addressing_);
    out.push_back(memory_);
    auto append = [&](const std::vector<Instr>& v) {
        for (const Instr& in : v) {
            const std::vector<std::uint32_t> w = encode(in);
            out.insert(out.end(), w.begin(), w.end());
        }
    };
    append(entry_points_);
    append(execution_modes_);
    append(names_);
    append(decorations_);
    append(types_consts_globals_);
    append(functions_);
    return out;
}

}  // namespace spv
