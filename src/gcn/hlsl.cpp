// HLSL from a lifted shader: see hlsl.h.
//
// The printer reads the module back (types, constants, globals, decorations,
// the one function's blocks) and walks the function's structured control
// flow: a selection header becomes an if/else whose arms end at its merge
// block, a loop header a `for (;;)` whose body ends at its continue block
// (the back edge) and leaves at its merge block (`break`). Phis are variables
// declared at the top of the function, assigned at the end of each
// predecessor (before the branch that leaves it); every other value is
// declared where it is defined, or at the top when a use lies outside the
// scope it is defined in (a loop header's value used after the loop).
//
// Stage inputs and outputs are static variables the entry point copies in
// from its parameters and out to its result, so the function reads and
// writes them as the module's Input and Output variables are read and
// written. The lifted function itself is `lifted_main`, which DXC inlines.
#include "gcn/hlsl.h"

#include "gcn/names.h"
#include "gcn/spirv.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

namespace gcn {

namespace {

// Opcodes the module builder (gcn/spirv.h) has no name for.
constexpr std::uint16_t kOpSource = 3, kOpSourceExtension = 4, kOpMemberName = 6, kOpString = 7, kOpLine = 8, kOpNoLine = 317,
                        kOpModuleProcessed = 330, kOpSpecConstantTrue = 48, kOpSpecConstantFalse = 49, kOpSpecConstantComposite = 51,
                        kOpSpecConstantOp = 52, kOpCopyObject = 83, kOpExecutionModeId = 331, kOpDecorateId = 332, kOpTypeForwardPointer = 39,
                        kOpTerminateInvocation = 4416, kOpIsHelperInvocation = 5381, kOpSampledImageOp = 86, kOpCopyMemory = 63;

struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& why) { throw Unsupported(why); }

struct Ins {
    std::uint16_t op = 0;
    std::uint32_t type = 0, id = 0;
    std::vector<std::uint32_t> a;  // operands after the result type and id
};

// Whether an opcode has a result type and a result id, for the opcodes a
// module may hold (an unknown one in a function body is taken as an
// operation with both).
void result_shape(std::uint16_t op, bool in_function, bool& has_type, bool& has_id) {
    has_type = has_id = false;
    switch (op) {
    case spv::OpNop: case kOpSource: case kOpSourceExtension: case spv::OpName: case kOpMemberName: case kOpLine: case kOpNoLine:
    case spv::OpExtension: case spv::OpMemoryModel: case spv::OpEntryPoint: case spv::OpExecutionMode: case kOpExecutionModeId:
    case spv::OpCapability: case spv::OpDecorate: case spv::OpMemberDecorate: case kOpDecorateId: case kOpModuleProcessed:
    case kOpTypeForwardPointer: case spv::OpFunctionEnd: case spv::OpStore: case kOpCopyMemory: case spv::OpSelectionMerge:
    case spv::OpLoopMerge: case spv::OpBranch: case spv::OpBranchConditional: case spv::OpSwitch: case spv::OpKill: case spv::OpReturn:
    case spv::OpReturnValue: case spv::OpUnreachable: case spv::OpControlBarrier: case spv::OpMemoryBarrier: case spv::OpImageWrite:
    case spv::OpDemoteToHelperInvocation: case kOpTerminateInvocation: case spv::OpAtomicStore:
        return;
    case spv::OpExtInstImport: case kOpString: case spv::OpLabel:
        has_id = true;
        return;
    default:
        break;
    }
    if (op >= spv::OpTypeVoid && op <= 39) {  // OpTypeVoid .. OpTypeForwardPointer (handled above)
        has_id = true;
        return;
    }
    (void)in_function;
    has_type = has_id = true;
}

enum class TK { Void, Bool, Int, Float, Vector, Array, RuntimeArray, Struct, Pointer, Function, Image, Sampler, SampledImage };
struct Type {
    TK k = TK::Void;
    std::uint32_t width = 0;  // Int, Float
    bool sign = false;        // Int
    std::uint32_t elem = 0;   // Vector component, Array element, Pointer pointee, SampledImage image, Image sampled type
    std::uint32_t count = 0;  // Vector size, Array length
    std::vector<std::uint32_t> members;
    std::uint32_t sc = 0;  // Pointer storage class
    std::uint32_t dim = 0, depth = 0, arrayed = 0, ms = 0, sampled = 0, format = 0;  // Image
};

struct Const {
    std::uint32_t type = 0;
    std::vector<std::uint32_t> words;  // scalar bits (two words for 64-bit)
    std::vector<std::uint32_t> parts;  // composite members
    bool composite = false, null = false, spec = false, undef = false;
};

// The value names StageParams gives the params block's members, by offset
// (gcn/translate.h): the lift declares the block member by member.
const char* params_member_name(std::uint32_t offset) {
    switch (offset) {
    case 0: return "l1_table";
    case 8: return "lds_address";
    case 16: return "user_sgpr";
    case 80: return "cb_valid";
    case 84: return "lds_bytes";
    case 88: return "gds_base";
    case 96: return "cb_bias_dw";
    case 160: return "cb_stride";
    case 224: return "cb_w3";
    case 288: return "vertex_formats";
    case 352: return "patch_cull_w";
    case 384: return "image_index";
    case 448: return "sampler_index";
    default: return nullptr;
    }
}

bool reserved_word(const std::string& s) {
    static const std::set<std::string> words = {
        "AppendStructuredBuffer", "BlendState", "Buffer", "ByteAddressBuffer", "CompileShader", "ComputeShader", "ConsumeStructuredBuffer",
        "DepthStencilState", "DepthStencilView", "DomainShader", "GeometryShader", "Hullshader", "InputPatch", "LineStream", "OutputPatch",
        "PixelShader", "PointStream", "RWBuffer", "RWByteAddressBuffer", "RWStructuredBuffer", "RWTexture1D", "RWTexture1DArray",
        "RWTexture2D", "RWTexture2DArray", "RWTexture3D", "RasterizerState", "RenderTargetView", "SamplerComparisonState", "SamplerState",
        "StructuredBuffer", "Texture1D", "Texture1DArray", "Texture2D", "Texture2DArray", "Texture2DMS", "Texture2DMSArray", "Texture3D",
        "TextureCube", "TextureCubeArray", "TriangleStream", "VertexShader", "abs", "and", "asm", "bool", "break", "case", "cbuffer", "centroid",
        "class", "column_major", "compile", "const", "continue", "default", "discard", "do", "double", "else", "export", "extern", "false",
        "float", "for", "groupshared", "half", "if", "in", "inline", "inout", "int", "interface", "line", "lineadj", "linear", "main",
        "matrix", "min", "max", "namespace", "nointerpolation", "noperspective", "or", "out", "packoffset", "pass", "point", "precise",
        "register", "return", "row_major", "sample", "sampler", "select", "shared", "snorm", "static", "string", "struct", "switch",
        "tbuffer", "technique", "texture", "triangle", "triangleadj", "true", "typedef", "uint", "uniform", "unorm", "unsigned", "vector",
        "vk", "void", "volatile", "while", "lifted_main", "this", "template", "operator", "sizeof", "using", "auto", "char",
        "short", "long", "signed", "enum", "union", "goto", "new", "delete", "private", "public", "protected", "friend", "virtual",
        "Texture", "SamplerState_t", "dword", "half2", "half3", "half4", "float2", "float3", "float4", "int2", "int3", "int4", "uint2",
        "uint3", "uint4", "bool2", "bool3", "bool4"};
    return words.count(s) != 0;
}

// What a value is to the printer.
struct Handle {
    // An image or sampler variable's name (OpLoad of a UniformConstant), a
    // sampled image (image + sampler), or an access chain (an lvalue, or a
    // byte-address buffer's dword).
    enum class Kind { None, Image, Sampler, SampledImage, Lvalue, BufferWord, Physical } kind = Kind::None;
    // Physical: a raw address (a compute lift's OpConvertUToPtr) and what it points at.
    std::string address;
    std::uint32_t pointee = 0;
    std::uint32_t image_var = 0, sampler_var = 0;
    std::string lvalue;           // Lvalue: the expression; BufferWord: the buffer's name
    std::uint32_t index = 0;      // BufferWord: the dword index value
    std::vector<std::uint32_t> uses;  // values the expression reads (for scopes)
};

class Printer {
public:
    explicit Printer(const std::vector<std::uint32_t>& w, const HlslOptions& o) : words(w), opt(o) {}

    HlslResult run() {
        HlslResult r;
        try {
            parse();
            prepare();
            // Each function in two passes: the first finds each block's scope,
            // the second prints, with the values a use outside their scope
            // needs declared at the top.
            for (Function& f : functions) {
                fn = &f;
                use_scopes.clear();
                block_scope.clear();
                dry = true;
                walk_function();
                hoist_values();
                dry = false;
                body_text.clear();
                walk_function();
                f.text = declarations(f) + body_text;
            }
            r.source = assemble();
            r.profile = profile;
            r.arguments = arguments;
            if (runtime_offset) {
                r.arguments.push_back("-Vd");
                r.validate_offsets = true;
            }
            if (hv202x) r.language = "202x";
        } catch (const Unsupported& e) {
            r.errors.push_back(e.what());
            r.source.clear();
        } catch (const std::exception& e) {
            // A module the printer misreads is the printer's bug, but never the game's: keep the lift.
            r.errors.push_back(std::string("internal error: ") + e.what());
            r.source.clear();
        }
        return r;
    }

private:
    const std::vector<std::uint32_t>& words;
    const HlslOptions& opt;

    // ---- the module ----------------------------------------------------------------
    std::vector<Ins> module_ins;
    std::map<std::uint32_t, Type> types;
    std::map<std::uint32_t, Const> consts;
    std::map<std::uint32_t, std::string> names;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::string> member_names;
    std::map<std::uint32_t, std::vector<std::vector<std::uint32_t>>> decos;  // id -> [decoration, operands...]
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::vector<std::uint32_t>>> member_decos;
    std::map<std::uint32_t, std::uint32_t> var_type;  // global variable -> pointer type
    std::vector<std::uint32_t> var_order;
    std::uint32_t glsl = 0, exec_model = 0, entry_fn = 0;
    std::vector<std::uint32_t> interface_ids;
    std::vector<std::vector<std::uint32_t>> exec_modes;
    std::set<std::uint32_t> capabilities;
    std::string profile;
    std::vector<std::string> arguments;
    bool runtime_offset = false;  // a sample takes its offset at run time (HlslResult::validate_offsets)

    struct Block {
        std::uint32_t label = 0;
        std::vector<const Ins*> phis, body;
        const Ins* merge = nullptr;
        const Ins* term = nullptr;
    };
    // The module's functions: the entry point's, and the helpers it calls
    // (a compute lift's page-table walk).
    struct Function {
        std::uint32_t id = 0, result_type = 0;
        std::vector<Ins> ins;  // the body, after OpFunction
        std::vector<std::uint32_t> params;
        std::map<std::uint32_t, Block> blocks;
        std::uint32_t first_block = 0;
        std::map<std::uint32_t, std::size_t> preds;  // block -> branches into it
        std::string name, text;                      // its HLSL name; the printed body with its declarations
    };
    std::vector<Function> functions;  // in module order
    Function* fn = nullptr;           // the one being printed
    std::vector<const Ins*> all_ins;  // every function's instructions
    std::map<std::uint32_t, const Ins*> defs;          // value -> its instruction
    std::map<std::uint32_t, std::uint32_t> def_block;  // value -> its block

    void parse() {
        if (words.size() < 5 || words[0] != 0x07230203u) fail("not a SPIR-V module");
        bool in_function = false;
        for (std::size_t i = 5; i < words.size();) {
            const std::uint32_t n = words[i] >> 16;
            const std::uint16_t op = static_cast<std::uint16_t>(words[i] & 0xffff);
            if (n == 0 || i + n > words.size()) fail("a malformed instruction");
            Ins in;
            in.op = op;
            bool has_type, has_id;
            result_shape(op, in_function, has_type, has_id);
            std::size_t k = i + 1;
            if (has_type) in.type = words[k++];
            if (has_id) in.id = words[k++];
            in.a.assign(words.begin() + static_cast<std::ptrdiff_t>(k), words.begin() + static_cast<std::ptrdiff_t>(i + n));
            i += n;
            if (op == spv::OpFunction) {
                in_function = true;
                Function f;
                f.id = in.id;
                f.result_type = in.type;
                functions.push_back(std::move(f));
                continue;
            }
            if (op == spv::OpFunctionEnd) {
                in_function = false;
                continue;
            }
            if (in_function) functions.back().ins.push_back(std::move(in));
            else module_ins.push_back(std::move(in));
        }
        for (const Ins& in : module_ins) module_level(in);
        if (exec_model == 0 && entry_fn == 0) fail("no entry point");
        if (functions.empty()) fail("no function body");
        for (Function& f : functions) build_blocks(f);
        for (const Function& f : functions) {
            for (const Ins& in : f.ins) all_ins.push_back(&in);
        }
    }

    static std::string str_operand(const std::vector<std::uint32_t>& a, std::size_t from) {
        std::string s;
        for (std::size_t k = from; k < a.size(); ++k) {
            for (int b = 0; b < 4; ++b) {
                const char c = static_cast<char>((a[k] >> (8 * b)) & 0xff);
                if (!c) return s;
                s += c;
            }
        }
        return s;
    }

    void module_level(const Ins& in) {
        switch (in.op) {
        case spv::OpCapability: capabilities.insert(in.a.at(0)); break;
        case spv::OpExtension:
            // A compute lift's raw addresses (vk::RawBufferLoad / vk::BufferPointer in HLSL).
            if (str_operand(in.a, 0) != "SPV_KHR_physical_storage_buffer") fail("extension " + str_operand(in.a, 0));
            break;
        case spv::OpExtInstImport:
            if (str_operand(in.a, 0) != "GLSL.std.450") fail("extended instruction set " + str_operand(in.a, 0));
            glsl = in.id;
            break;
        case spv::OpMemoryModel:
            if ((in.a.at(0) != 0 && in.a.at(0) != 5348 /* PhysicalStorageBuffer64 */) || in.a.at(1) != 1)
                fail("a memory model other than Logical or PhysicalStorageBuffer64 GLSL450");
            break;
        case spv::OpEntryPoint: {
            if (entry_fn) fail("more than one entry point");
            exec_model = in.a.at(0);
            entry_fn = in.a.at(1);
            const std::string nm = str_operand(in.a, 2);
            std::size_t k = 2 + nm.size() / 4 + 1;
            for (; k < in.a.size(); ++k) interface_ids.push_back(in.a[k]);
            break;
        }
        case spv::OpExecutionMode: exec_modes.push_back(in.a); break;
        case spv::OpName: names[in.a.at(0)] = str_operand(in.a, 1); break;
        case kOpMemberName: member_names[{in.a.at(0), in.a.at(1)}] = str_operand(in.a, 2); break;
        case spv::OpDecorate: decos[in.a.at(0)].push_back(std::vector<std::uint32_t>(in.a.begin() + 1, in.a.end())); break;
        case spv::OpMemberDecorate:
            member_decos[{in.a.at(0), in.a.at(1)}].push_back(std::vector<std::uint32_t>(in.a.begin() + 2, in.a.end()));
            break;
        case kOpSource: case kOpSourceExtension: case kOpString: case kOpModuleProcessed: case kOpLine: case kOpNoLine: break;
        case spv::OpTypeVoid: types[in.id].k = TK::Void; break;
        case spv::OpTypeBool: types[in.id].k = TK::Bool; break;
        case spv::OpTypeInt: {
            Type& t = types[in.id];
            t.k = TK::Int;
            t.width = in.a.at(0);
            t.sign = in.a.at(1) != 0;
            if (t.width != 32 && t.width != 64 && t.width != 16) fail("an integer of " + std::to_string(t.width) + " bits");
            break;
        }
        case spv::OpTypeFloat: {
            Type& t = types[in.id];
            t.k = TK::Float;
            t.width = in.a.at(0);
            if (t.width != 32 && t.width != 16) fail("a float of " + std::to_string(t.width) + " bits");
            if (t.width == 16) has_half = true;
            break;
        }
        case spv::OpTypeVector: {
            Type& t = types[in.id];
            t.k = TK::Vector;
            t.elem = in.a.at(0);
            t.count = in.a.at(1);
            break;
        }
        case spv::OpTypeArray: {
            Type& t = types[in.id];
            t.k = TK::Array;
            t.elem = in.a.at(0);
            const auto c = consts.find(in.a.at(1));
            if (c == consts.end() || c->second.spec || c->second.words.empty()) fail("an array whose length is not a constant");
            t.count = c->second.words[0];
            break;
        }
        case spv::OpTypeRuntimeArray: {
            Type& t = types[in.id];
            t.k = TK::RuntimeArray;
            t.elem = in.a.at(0);
            break;
        }
        case spv::OpTypeStruct: {
            Type& t = types[in.id];
            t.k = TK::Struct;
            t.members = in.a;
            break;
        }
        case spv::OpTypePointer: {
            Type& t = types[in.id];
            t.k = TK::Pointer;
            t.sc = in.a.at(0);
            t.elem = in.a.at(1);
            break;
        }
        case spv::OpTypeFunction: types[in.id].k = TK::Function; break;
        case spv::OpTypeImage: {
            Type& t = types[in.id];
            t.k = TK::Image;
            t.elem = in.a.at(0);
            t.dim = in.a.at(1);
            t.depth = in.a.at(2);
            t.arrayed = in.a.at(3);
            t.ms = in.a.at(4);
            t.sampled = in.a.at(5);
            t.format = in.a.at(6);
            break;
        }
        case spv::OpTypeSampler: types[in.id].k = TK::Sampler; break;
        case spv::OpTypeSampledImage: {
            Type& t = types[in.id];
            t.k = TK::SampledImage;
            t.elem = in.a.at(0);
            break;
        }
        case spv::OpConstantTrue: case spv::OpConstantFalse: case kOpSpecConstantTrue: case kOpSpecConstantFalse: {
            Const& c = consts[in.id];
            c.type = in.type;
            c.words = {in.op == spv::OpConstantTrue || in.op == kOpSpecConstantTrue ? 1u : 0u};
            c.spec = in.op == kOpSpecConstantTrue || in.op == kOpSpecConstantFalse;
            break;
        }
        case spv::OpConstant: case spv::OpSpecConstant: {
            Const& c = consts[in.id];
            c.type = in.type;
            c.words = in.a;
            c.spec = in.op == spv::OpSpecConstant;
            break;
        }
        case spv::OpConstantComposite: {
            Const& c = consts[in.id];
            c.type = in.type;
            c.composite = true;
            c.parts = in.a;
            break;
        }
        case spv::OpConstantNull: {
            Const& c = consts[in.id];
            c.type = in.type;
            c.null = true;
            break;
        }
        case spv::OpUndef: {
            Const& c = consts[in.id];
            c.type = in.type;
            c.undef = true;
            break;
        }
        case kOpSpecConstantComposite: case kOpSpecConstantOp: fail("a specialization constant expression");
        case spv::OpVariable: {
            if (in.a.size() > 1) fail("a global variable with an initializer");
            var_type[in.id] = in.type;
            var_order.push_back(in.id);
            break;
        }
        default:
            fail("module-level instruction " + std::to_string(in.op));
        }
    }

    void build_blocks(Function& f) {
        auto& blocks = f.blocks;
        auto& preds = f.preds;
        Block* cur = nullptr;
        for (std::size_t i = 0; i < f.ins.size(); ++i) {
            const Ins& in = f.ins[i];
            if (in.op == spv::OpFunctionParameter) {
                if (cur || f.first_block) fail("a parameter after a block");
                f.params.push_back(in.id);
                defs[in.id] = &in;
                continue;
            }
            if (in.op == spv::OpLabel) {
                cur = &blocks[in.id];
                cur->label = in.id;
                if (!f.first_block) f.first_block = in.id;
                continue;
            }
            if (!cur) fail("an instruction outside a block");
            if (in.id) {
                defs[in.id] = &in;
                def_block[in.id] = cur->label;
            }
            switch (in.op) {
            case spv::OpPhi: cur->phis.push_back(&in); break;
            case spv::OpSelectionMerge: case spv::OpLoopMerge: cur->merge = &in; break;
            case spv::OpBranch: case spv::OpBranchConditional: case spv::OpSwitch: case spv::OpKill: case spv::OpReturn:
            case spv::OpReturnValue: case spv::OpUnreachable: case kOpTerminateInvocation:
                cur->term = &in;
                cur = nullptr;
                break;
            case kOpLine: case kOpNoLine: break;
            case spv::OpVariable: fail("a function-local variable");
            default: cur->body.push_back(&in); break;
            }
        }
        for (const auto& [label, b] : blocks) {
            if (!b.term) fail("a block without a terminator");
            if (b.term->op == spv::OpBranch) ++preds[b.term->a.at(0)];
            if (b.term->op == spv::OpBranchConditional) {
                ++preds[b.term->a.at(1)];
                ++preds[b.term->a.at(2)];
            }
            if (b.term->op == spv::OpSwitch) fail("OpSwitch");
            if (b.term->op == spv::OpReturnValue && f.id == entry_fn) fail("an entry point that returns a value");
            if (b.term->op == kOpTerminateInvocation) fail("OpTerminateInvocation");
        }
    }

    // ---- types and literals ------------------------------------------------------------
    const Type& type(std::uint32_t id) const {
        const auto it = types.find(id);
        if (it == types.end()) fail("an unknown type %" + std::to_string(id));
        return it->second;
    }
    std::uint32_t type_of(std::uint32_t value) const {
        if (const auto c = consts.find(value); c != consts.end()) return c->second.type;
        if (const auto d = defs.find(value); d != defs.end()) return d->second->type;
        if (const auto v = var_type.find(value); v != var_type.end()) return v->second;
        fail("a value with no type %" + std::to_string(value));
    }
    // The scalar type of a scalar or vector.
    std::uint32_t scalar_of(std::uint32_t t) const {
        const Type& ty = type(t);
        return ty.k == TK::Vector ? ty.elem : t;
    }
    std::uint32_t lanes_of(std::uint32_t t) const {
        const Type& ty = type(t);
        return ty.k == TK::Vector ? ty.count : 1;
    }
    // The bits of a scalar or vector of numbers.
    std::uint32_t bits_of(std::uint32_t t) const {
        const Type& s = type(scalar_of(t));
        if (s.k != TK::Int && s.k != TK::Float) fail("a bitcast of a value that is no number");
        return s.width * lanes_of(t);
    }
    // Whether the module declares a 16-bit float (a lift for a device that rounds halves itself).
    bool has_half = false;
    bool is_half(std::uint32_t value) const {
        const Type& s = type(scalar_of(type_of(value)));
        return s.k == TK::Float && s.width == 16;
    }
    // 'u' 'i' 'f' 'b' (and 'h' half, 'U' uint64, 'I' int64) of a scalar or vector type.
    char kind_of(std::uint32_t t) const {
        const Type& s = type(scalar_of(t));
        switch (s.k) {
        case TK::Bool: return 'b';
        case TK::Float: return s.width == 16 ? 'h' : 'f';
        case TK::Int:
            if (s.width == 64) return s.sign ? 'I' : 'U';
            if (s.width == 16) fail("a 16-bit integer");
            return s.sign ? 'i' : 'u';
        default: fail("a value of a type that is no scalar or vector");
        }
    }
    static const char* scalar_name(char k) {
        switch (k) {
        case 'b': return "bool";
        case 'f': return "float";
        case 'h': return "float16_t";
        case 'u': return "uint";
        case 'i': return "int";
        case 'U': return "uint64_t";
        case 'I': return "int64_t";
        }
        return "?";
    }
    bool uses_half = false;
    std::string type_name(std::uint32_t t) {
        const Type& ty = type(t);
        switch (ty.k) {
        case TK::Bool: case TK::Int: case TK::Float: {
            const char k = kind_of(t);
            if (k == 'h') uses_half = true;
            return scalar_name(k);
        }
        case TK::Vector: {
            const char k = kind_of(t);
            if (k == 'h') uses_half = true;
            return std::string(scalar_name(k)) + std::to_string(ty.count);
        }
        case TK::Struct: return struct_name(t);
        default: fail("a value of type kind " + std::to_string(static_cast<int>(ty.k)));
        }
    }
    // A declarator: "float4 name", "float name[4]", "uint4 name[3][2]".
    std::string declare(std::uint32_t t, const std::string& name) {
        std::string dims;
        while (type(t).k == TK::Array) {
            dims += "[" + std::to_string(type(t).count) + "]";
            t = type(t).elem;
        }
        return type_name(t) + " " + name + dims;
    }

    std::map<std::uint32_t, std::string> struct_names;
    std::vector<std::uint32_t> struct_order;
    std::string struct_name(std::uint32_t t) {
        const auto it = struct_names.find(t);
        if (it != struct_names.end()) return it->second;
        fail("a struct value");
    }

    static std::string hex(std::uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
        return b;
    }
    static std::string float_literal(std::uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, 4);
        const std::uint32_t exp = (bits >> 23) & 0xff;
        if (exp == 0xff || (exp == 0 && (bits & 0x7fffff)) || bits == 0x80000000u) return "asfloat(" + hex(bits) + "u)";
        char b[64];
        const auto r = std::to_chars(b, b + sizeof(b), f);
        std::string s(b, r.ptr);
        float back = 0;
        std::from_chars(s.data(), s.data() + s.size(), back);
        std::uint32_t bb;
        std::memcpy(&bb, &back, 4);
        if (bb != bits) return "asfloat(" + hex(bits) + "u)";
        if (s.find_first_of(".e") == std::string::npos) s += ".0";
        return s + "f";
    }
    static std::string uint_literal(std::uint32_t v) { return v < 65536 ? std::to_string(v) + "u" : hex(v) + "u"; }
    static std::string int_literal(std::uint32_t bits) {
        const auto v = static_cast<std::int32_t>(bits);
        if (v >= 0) return std::to_string(v);
        if (v > -65536) return "(" + std::to_string(v) + ")";
        return "asint(" + hex(bits) + "u)";
    }
    // A scalar constant's bits as a literal of kind `k`.
    std::string scalar_literal(std::uint64_t bits, char k) const {
        switch (k) {
        case 'b': return bits ? "true" : "false";
        case 'f': return float_literal(static_cast<std::uint32_t>(bits));
        // A half constant would need 16-bit integers (asfloat16(uint16_t(...))),
        // which the lift does not declare; the lifter writes none.
        case 'h': fail("a half-precision constant");
        case 'u': return uint_literal(static_cast<std::uint32_t>(bits));
        case 'i': return int_literal(static_cast<std::uint32_t>(bits));
        case 'U': return hex(bits) + "ull";
        case 'I': return "int64_t(" + hex(bits) + "ull)";
        }
        return "?";
    }
    std::uint64_t const_bits(const Const& c) const {
        if (c.null || c.undef || c.words.empty()) return 0;
        std::uint64_t v = c.words[0];
        if (c.words.size() > 1) v |= static_cast<std::uint64_t>(c.words[1]) << 32;
        return v;
    }

    // ---- names ---------------------------------------------------------------------------
    std::set<std::string> taken;
    std::map<std::uint32_t, std::string> global_names;  // variables and specialization constants
    std::string unique(std::string s) {
        std::string out;
        for (char c : s) out += (std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_';
        if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0]))) out = "_" + out;
        // vN is a value's own name, phiN and the like the printer's.
        if (out.size() > 1 && out[0] == 'v' && std::all_of(out.begin() + 1, out.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
            out = "_" + out;
        if (out.rfind("gl_", 0) == 0 || out.rfind("spv_", 0) == 0 || out.rfind("in_", 0) == 0 || out.rfind("out_", 0) == 0 || out.rfind("SV_", 0) == 0) out = "_" + out;
        if (reserved_word(out)) out += "_";
        std::string base = out;
        for (int n = 2; taken.count(out); ++n) out = base + "_" + std::to_string(n);
        taken.insert(out);
        return out;
    }

    std::string value_name(std::uint32_t id) const { return "v" + std::to_string(id); }

    // ---- prepare: globals ---------------------------------------------------------------
    struct StageVar {
        std::uint32_t id = 0;
        std::uint32_t pointee = 0;
        bool input = false;
        int location = -1;
        int builtin = -1;
        std::string name;        // the static variable
        std::string interp;      // nointerpolation / noperspective / centroid
        bool patch = false;
        bool invariant = false;  // an output declared Invariant ([[vk::ext_decorate(18)]] on its member)
    };
    std::vector<StageVar> stage_vars;
    std::map<std::uint32_t, std::size_t> stage_index;  // variable -> stage_vars
    struct Resource {
        std::uint32_t id = 0;
        std::string decl;  // the declaration line(s)
    };
    std::vector<std::string> resource_decls, spec_decls, struct_decls, helper_decls;
    std::set<std::uint32_t> compare_samplers, plain_samplers;  // samplers used for depth-compare / other samples
    std::set<std::uint32_t> written_buffers;
    std::map<std::uint32_t, std::string> buffer_names;  // storage buffer variable -> name
    std::map<std::string, const std::vector<ReflectedElement>*> buffer_members;  // its constant-buffer members (HlslOptions::names)
    std::string helper_invocation_name;
    bool early_tests = false;
    std::string tess_domain;                // [domain(...)] of a domain shader
    std::vector<std::uint32_t> tess_modes;  // spacing, winding, point mode
    std::uint32_t output_vertices = 0;      // a hull's OutputVertices
    std::uint32_t local_size[3] = {0, 0, 0};  // a compute shader's [numthreads]
    bool hv202x = false;                      // vk::BufferPointer (an atomic on a raw address) needs -HV 202x
    std::vector<std::uint32_t> float_modes;   // DenormPreserve, RoundingModeRTZ for 16-bit floats, in the lift's order
    // The hull: its per-control-point output and the barrier its body crosses.
    std::uint32_t cp_out_var = 0, invocation_var = 0;
    bool past_barrier = false;

    std::vector<std::uint32_t> decoration(std::uint32_t id, std::uint32_t d) const {
        const auto it = decos.find(id);
        if (it == decos.end()) return {};
        for (const auto& v : it->second) {
            if (!v.empty() && v[0] == d) return std::vector<std::uint32_t>(v.begin() + 1, v.end());
        }
        return {};
    }
    bool decorated(std::uint32_t id, std::uint32_t d) const {
        const auto it = decos.find(id);
        if (it == decos.end()) return false;
        for (const auto& v : it->second) {
            if (!v.empty() && v[0] == d) return true;
        }
        return false;
    }
    std::uint32_t member_offset(std::uint32_t st, std::uint32_t m) const {
        const auto it = member_decos.find({st, m});
        if (it != member_decos.end()) {
            for (const auto& v : it->second) {
                if (v.size() >= 2 && v[0] == spv::DecOffset) return v[1];
            }
        }
        fail("a block member without an offset");
    }

    void prepare() {
        switch (exec_model) {
        case spv::EmFragment: profile = "ps_6_6"; break;
        case spv::EmVertex: profile = "vs_6_6"; break;
        case spv::EmTessellationEvaluation: profile = "ds_6_6"; break;
        case spv::EmTessellationControl: profile = "hs_6_6"; break;
        case spv::EmGLCompute: profile = "cs_6_6"; break;
        default: fail("execution model " + std::to_string(exec_model));
        }
        for (const std::uint32_t c : capabilities) {
            switch (c) {
            case spv::CapShader: case spv::CapInt64: case spv::CapClipDistance: case spv::CapImageQuery: case spv::CapGroupNonUniform:
            case spv::CapGroupNonUniformQuad: case spv::CapDerivativeControl: case spv::CapSampled1D: case spv::CapImage1D: case spv::CapTessellation:
            case spv::CapSampledCubeArray: case spv::CapImageCubeArray: case spv::CapStorageImageReadWithoutFormat:
            case spv::CapStorageImageWriteWithoutFormat: case spv::CapImageGatherExtended: case spv::CapPhysicalStorageBufferAddresses:
                break;
            // A lift for a device that rounds halves itself (gcn/half.h
            // native_half_rtz): Float16 comes with the half types DXC is given
            // (-enable-16bit-types), DenormPreserve and RoundingModeRTZ with
            // the execution modes that need them (float_modes).
            case spv::CapFloat16: case spv::CapDenormPreserve: case spv::CapRoundingModeRTZ: break;
            default: fail("capability " + std::to_string(c));
            }
        }
        for (const auto& m : exec_modes) {
            if (m.size() < 2 || m[0] != entry_fn) fail("an execution mode of another function");
            switch (m[1]) {
            case spv::ExOriginUpperLeft: case spv::ExDepthReplacing: break;  // DXC's own for a pixel shader (with SV_Depth)
            case spv::ExEarlyFragmentTests: early_tests = true; break;
            // Tessellation: the domain is the domain shader's attribute; the
            // spacing, winding and point mode HLSL gives the hull shader, so
            // the evaluation stage declares them as inline SPIR-V, as the lift does.
            case spv::ExQuads: tess_domain = "quad"; break;
            case spv::ExTriangles: tess_domain = "tri"; break;
            case spv::ExIsolines: tess_domain = "isoline"; break;
            case spv::ExSpacingEqual: case spv::ExSpacingFractionalEven: case spv::ExSpacingFractionalOdd: case spv::ExVertexOrderCw:
            case spv::ExVertexOrderCcw: case spv::ExPointMode:
                tess_modes.push_back(m[1]);
                break;
            case spv::ExOutputVertices:
                if (m.size() < 3) fail("OutputVertices without a count");
                output_vertices = m[2];
                break;
            case spv::ExLocalSize:
                if (m.size() < 5) fail("LocalSize without three sizes");
                local_size[0] = m[2], local_size[1] = m[3], local_size[2] = m[4];
                break;
            // The device's own half rounding: f16 conversions round toward
            // zero and keep denormals. HLSL has no spelling for either; the
            // entry point declares them as inline SPIR-V, with their
            // capabilities (float_mode_attributes). For 16-bit floats only:
            // the printer writes no arithmetic on halves, only the lift's
            // conversions and bitcasts as its own instructions, which DXC
            // passes to the driver whatever their operands - an operation in
            // HLSL's own spelling DXC may evaluate itself, rounding to nearest.
            case spv::ExDenormPreserve: case spv::ExRoundingModeRTZ:
                if (m.size() != 3 || m[2] != 16) fail("a float-controls execution mode for floats other than 16-bit");
                if (std::find(float_modes.begin(), float_modes.end(), m[1]) == float_modes.end()) float_modes.push_back(m[1]);
                break;
            default: fail("execution mode " + std::to_string(m[1]));
            }
        }
        // Samplers by use: SampleCmp takes a SamplerComparisonState.
        for (const Ins* ip : all_ins) {
            const Ins& in = *ip;
            if (in.op != kOpSampledImageOp) continue;
            bool compare = false, plain = false;
            for (const Ins* up : all_ins) {
                const Ins& u = *up;
                if (u.a.empty() || u.a[0] != in.id) continue;
                if (u.op == spv::OpImageSampleDrefImplicitLod || u.op == spv::OpImageSampleDrefExplicitLod || u.op == spv::OpImageDrefGather) compare = true;
                else plain = true;
            }
            const std::uint32_t s = sampler_root(in.a.at(1));
            if (compare) compare_samplers.insert(s);
            if (plain) plain_samplers.insert(s);
        }
        for (const Ins* ip : all_ins) {
            const Ins& in = *ip;
            if (in.op == spv::OpStore || in.op == spv::OpAtomicStore || in.op == spv::OpAtomicIAdd) {  // an atomic add writes it too
                if (const std::uint32_t root = pointer_root(in.a.at(0)); root && type(type(var_type.at(root)).elem).k == TK::Struct &&
                                                                            type(var_type.at(root)).sc == spv::ScStorageBuffer) {
                    written_buffers.insert(root);
                }
            }
        }
        for (const std::uint32_t v : var_order) global(v);
        for (Function& f : functions) {
            if (f.id == entry_fn) continue;
            f.name = unique(names.count(f.id) ? names.at(f.id) : "fn" + std::to_string(f.id));
            global_names[f.id] = f.name;
        }
    }

    std::uint32_t sampler_root(std::uint32_t value) const {
        const auto d = defs.find(value);
        if (d == defs.end() || d->second->op != spv::OpLoad) fail("a sampler that is not a loaded variable");
        return d->second->a.at(0);
    }
    std::uint32_t pointer_root(std::uint32_t ptr) const {
        if (var_type.count(ptr)) return ptr;
        const auto d = defs.find(ptr);
        if (d == defs.end()) return 0;
        if (d->second->op == spv::OpAccessChain) return pointer_root(d->second->a.at(0));
        return 0;
    }

    void global(std::uint32_t v) {
        const Type& pt = type(var_type.at(v));
        const std::uint32_t pointee = pt.elem;
        const Type& t = type(pointee);
        const std::string given = names.count(v) ? names.at(v) : std::string();
        const std::vector<std::uint32_t> set = decoration(v, spv::DecDescriptorSet), binding = decoration(v, spv::DecBinding);
        const auto binding_attr = [&] {
            if (set.empty() || binding.empty()) fail("a resource without a descriptor set and binding");
            return "[[vk::binding(" + std::to_string(binding[0]) + ", " + std::to_string(set[0]) + ")]] ";
        };
        switch (pt.sc) {
        case spv::ScUniform: {
            if (t.k != TK::Struct || !decorated(pointee, spv::DecBlock)) fail("a uniform that is not a block");
            const std::string tn = unique((given.empty() ? std::string("Block") : given) + "_t");
            struct_names[pointee] = tn;
            std::string s = "struct " + tn + " {\n";
            for (std::uint32_t m = 0; m < t.members.size(); ++m) {
                const std::uint32_t off = member_offset(pointee, m);
                std::string mn;
                if (const auto it = member_names.find({pointee, m}); it != member_names.end()) mn = it->second;
                if (mn.empty() && params_member_name(off)) mn = params_member_name(off);
                if (mn.empty()) mn = "m" + std::to_string(m);
                member_field[{pointee, m}] = mn;
                std::uint32_t mt = t.members[m];
                // Arrays of vectors need their stride to be the std140 one (16) the lift declares.
                for (std::uint32_t a = mt; type(a).k == TK::Array; a = type(a).elem) {
                    const std::vector<std::uint32_t> stride = decoration(a, spv::DecArrayStride);
                    const std::uint32_t el = type(a).elem;
                    if (stride.empty() || stride[0] != 16 || type(el).k == TK::Array) fail("a block array whose stride is not 16");
                }
                s += "    [[vk::offset(" + std::to_string(off) + ")]] " + declare(mt, mn) + ";\n";
            }
            s += "};\n";
            struct_decls.push_back(s);
            const std::string n = unique(given.empty() ? "block" : given);
            global_names[v] = n;
            resource_decls.push_back(binding_attr() + "ConstantBuffer<" + tn + "> " + n + ";");
            return;
        }
        case spv::ScStorageBuffer: {
            // The lift's constant buffers: a block of one runtime array of uint.
            if (t.k != TK::Struct || t.members.size() != 1 || type(t.members[0]).k != TK::RuntimeArray ||
                kind_of(type(t.members[0]).elem) != 'u' || type(type(t.members[0]).elem).k != TK::Int) {
                fail("a storage buffer other than a block of uint[]");
            }
            const std::string n = unique(given.empty() ? "buffer" : given);
            global_names[v] = n;
            buffer_names[v] = n;
            if (opt.names) {
                for (const BindingName& b : opt.names->buffers) {
                    if (!b.name.empty() && b.name == given && !b.elements.empty()) buffer_members[n] = &b.elements;
                }
            }
            resource_decls.push_back(binding_attr() + (written_buffers.count(v) ? "RWByteAddressBuffer " : "ByteAddressBuffer ") + n + ";");
            return;
        }
        case spv::ScUniformConstant: {
            if (t.k == TK::RuntimeArray || t.k == TK::Array) fail("an array of descriptors (bindless)");
            if (t.k == TK::Sampler) {
                const bool cmp = compare_samplers.count(v) != 0, plain = plain_samplers.count(v) != 0 || !cmp;
                const std::string n = unique(given.empty() ? "smp" : given);
                global_names[v] = n;
                if (plain) resource_decls.push_back(binding_attr() + "SamplerState " + n + ";");
                if (cmp) {
                    const std::string cn = plain ? unique(n + "_cmp") : n;
                    compare_name[v] = cn;
                    resource_decls.push_back(binding_attr() + "SamplerComparisonState " + cn + ";");
                }
                return;
            }
            if (t.k != TK::Image) fail("a uniform constant that is no image or sampler");
            const std::string n = unique(given.empty() ? "img" : given);
            global_names[v] = n;
            resource_decls.push_back(binding_attr() + image_type_name(pointee) + " " + n + ";");
            return;
        }
        case spv::ScInput: case spv::ScOutput: {
            StageVar s;
            s.id = v;
            s.pointee = pointee;
            s.input = pt.sc == spv::ScInput;
            const std::vector<std::uint32_t> loc = decoration(v, spv::DecLocation), bi = decoration(v, spv::DecBuiltIn);
            if (!loc.empty()) s.location = static_cast<int>(loc[0]);
            if (!bi.empty()) s.builtin = static_cast<int>(bi[0]);
            if (s.location < 0 && s.builtin < 0) fail("a stage variable without a location or built-in");
            if (decorated(v, spv::DecFlat)) s.interp = "nointerpolation ";
            if (decorated(v, spv::DecNoPerspective)) s.interp += "noperspective ";
            if (decorated(v, spv::DecCentroid)) s.interp += "centroid ";
            if (decorated(v, 17 /* Sample */)) s.interp += "sample ";
            s.invariant = decorated(v, spv::DecInvariant);
            if (s.invariant && (s.input || exec_model == spv::EmTessellationControl)) fail("an invariant input or hull output");
            s.patch = decorated(v, spv::DecPatch);
            if (s.patch && !(exec_model == spv::EmTessellationEvaluation && s.input) && !(exec_model == spv::EmTessellationControl && !s.input))
                fail("a patch variable outside a domain shader's inputs or a hull shader's outputs");
            if (decorated(v, 4 /* Component */)) fail("a component decoration");
            std::string base = given;
            if (s.builtin >= 0) base = builtin_name(s.builtin, s.input);
            if (base.empty()) base = std::string(s.input ? "in" : "out") + std::to_string(s.location);
            if (s.builtin == spv::BiHelperInvocation) {
                if (exec_model != spv::EmFragment) fail("HelperInvocation outside a pixel shader");
                // As a built-in input of its own: IsHelperLane() would make DXC
                // write every discard as a demote.
                s.name = helper_invocation_name = unique("gl_HelperInvocation_");
                helper_decls.push_back("[[vk::ext_builtin_input(23 /* HelperInvocation */)]] static const bool " + s.name + ";");
            } else if (s.builtin == spv::BiTessCoord) {
                // The tessellation coordinate as the module reads it (SV_DomainLocation is two components of a quad's).
                if (type(pointee).k != TK::Vector || type(pointee).count != 3) fail("a tessellation coordinate that is not a float3");
                s.name = unique("gl_TessCoord_");
                helper_decls.push_back("[[vk::ext_builtin_input(13 /* TessCoord */)]] static const float3 " + s.name + ";");
            } else {
                s.name = unique(base);
            }
            if (exec_model == spv::EmTessellationControl) {
                if (!s.input && s.location >= 0 && !s.patch) {
                    if (cp_out_var) fail("a hull shader with two control-point outputs");
                    cp_out_var = v;
                }
                if (s.input && s.builtin == spv::BiInvocationId) invocation_var = v;
            }
            stage_index[v] = stage_vars.size();
            global_names[v] = s.name;
            stage_vars.push_back(s);
            return;
        }
        default: fail("a global variable in storage class " + std::to_string(pt.sc));
        }
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::string> member_field;
    std::map<std::uint32_t, std::string> compare_name;
    std::set<std::uint32_t> hull_late_cp_access;  // access chains into the output patch after the barrier
    std::map<std::uint32_t, std::uint32_t> hull_early_cp_read;  // ... before it, at an index not the invocation's: chain -> index

    std::string builtin_name(int b, bool input) const {
        switch (b) {
        case spv::BiPosition: return input ? "" : "gl_Position_";
        case 1: return "gl_PointSize_";
        case 3: return "gl_ClipDistance_";
        case spv::BiFragCoord: return "gl_FragCoord_";
        case spv::BiFrontFacing: return "gl_FrontFacing_";
        case spv::BiFragDepth: return "gl_FragDepth_";
        case spv::BiHelperInvocation: return "gl_HelperInvocation_";
        case spv::BiVertexIndex: return "gl_VertexIndex_";
        case spv::BiInstanceIndex: return "gl_InstanceIndex_";
        case spv::BiTessCoord: return "gl_TessCoord_";
        case spv::BiPrimitiveId: return "gl_PrimitiveID_";
        case spv::BiTessLevelOuter: return "gl_TessLevelOuter_";
        case spv::BiTessLevelInner: return "gl_TessLevelInner_";
        case spv::BiInvocationId: return "gl_InvocationID_";
        case spv::BiLocalInvocationId: return "gl_LocalInvocationID_";
        case spv::BiWorkgroupId: return "gl_WorkGroupID_";
        case 28 /* GlobalInvocationId */: return "gl_GlobalInvocationID_";
        case spv::BiLocalInvocationIndex: return "gl_LocalInvocationIndex_";
        default: fail("built-in " + std::to_string(b));
        }
    }

    std::string image_type_name(std::uint32_t t) {
        const Type& im = type(t);
        if (im.format != 0) fail("an image with a format");
        const char k = kind_of(im.elem);
        const std::string texel = std::string(scalar_name(k == 'h' ? 'f' : k)) + "4";
        if (k != 'f' && k != 'u' && k != 'i') fail("an image of texel kind " + std::string(1, k));
        std::string base;
        switch (im.dim) {
        case spv::Dim1D: base = im.arrayed ? "Texture1DArray" : "Texture1D"; break;
        case spv::Dim2D: base = im.ms ? (im.arrayed ? "Texture2DMSArray" : "Texture2DMS") : (im.arrayed ? "Texture2DArray" : "Texture2D"); break;
        case spv::Dim3D: base = "Texture3D"; break;
        case spv::DimCube: base = im.arrayed ? "TextureCubeArray" : "TextureCube"; break;
        case spv::DimBuffer: base = "Buffer"; break;
        default: fail("image dimension " + std::to_string(im.dim));
        }
        if (im.sampled == 2) {
            if (im.ms || im.dim == spv::DimCube) fail("a storage image of this shape");
            return "[[vk::image_format(\"unknown\")]] RW" + base + "<" + texel + ">";
        }
        if (im.sampled != 1) fail("an image whose sampled mode is not known");
        return base + "<" + texel + ">";
    }
    // Coordinates an image of this type takes for a sample (without the array layer: `layer` adds it).
    std::uint32_t coord_count(std::uint32_t image_type, bool layer) const {
        const Type& im = type(image_type);
        std::uint32_t n = im.dim == spv::Dim1D || im.dim == spv::DimBuffer ? 1 : im.dim == spv::Dim2D ? 2 : 3;
        if (layer && im.arrayed) ++n;
        return n;
    }

    // ---- the function: scopes and values ------------------------------------------------
    bool dry = true;
    std::string body_text;
    int indent = 1;
    int scope = 0;
    int scope_count = 0;
    std::vector<int> scope_parent;  // scope -> parent (scope 0: the function's)
    std::map<std::uint32_t, int> block_scope;
    std::map<std::uint32_t, std::set<int>> use_scopes;  // value -> scopes it is read in
    std::set<std::uint32_t> hoisted;
    std::set<std::uint32_t> printed_blocks;
    std::map<std::uint32_t, Handle> handles;
    std::map<std::uint32_t, std::pair<std::string, std::string>> parts;  // a struct value's two members (carry ops)

    void line(const std::string& s) {
        if (dry) return;
        body_text.append(static_cast<std::size_t>(indent) * 4, ' ');
        body_text += s;
        body_text += '\n';
    }
    void open_scope() {
        ++indent;
        const int s = ++scope_count;
        if (static_cast<int>(scope_parent.size()) <= s) scope_parent.resize(static_cast<std::size_t>(s) + 1, 0);
        scope_parent[static_cast<std::size_t>(s)] = scope;
        scope = s;
    }
    void close_scope() {
        --indent;
        scope = scope_parent[static_cast<std::size_t>(scope)];
    }
    bool encloses(int outer, int inner) const {
        for (int s = inner;; s = scope_parent[static_cast<std::size_t>(s)]) {
            if (s == outer) return true;
            if (s == 0) return false;
        }
    }
    void hoist_values() {
        for (const auto& [v, scopes] : use_scopes) {
            const auto d = defs.find(v);
            if (d == defs.end()) continue;
            if (d->second->op == spv::OpPhi || d->second->op == spv::OpFunctionParameter) continue;  // declared at the top anyway
            const auto db = def_block.find(v);
            if (db == def_block.end()) continue;
            const auto bs = block_scope.find(db->second);
            if (bs == block_scope.end()) continue;
            for (const int s : scopes) {
                if (!encloses(bs->second, s)) {
                    hoisted.insert(v);
                    break;
                }
            }
        }
    }
    // Notes that `v` is read here (and what a handle it names reads).
    void use(std::uint32_t v) {
        if (consts.count(v) || var_type.count(v)) return;
        if (const auto h = handles.find(v); h != handles.end()) {
            for (const std::uint32_t u : h->second.uses) use(u);
            return;
        }
        if (dry) use_scopes[v].insert(scope);
    }

    // ---- expressions ---------------------------------------------------------------------
    std::string ref(std::uint32_t v) {
        if (const auto c = consts.find(v); c != consts.end()) return const_expr(v);
        if (const auto g = global_names.find(v); g != global_names.end()) return g->second;
        if (handles.count(v)) fail("a handle used as a value");
        if (parts.count(v)) fail("a struct value used whole");
        use(v);
        return value_name(v);
    }
    std::string const_expr(std::uint32_t v) {
        const Const& c = consts.at(v);
        if (c.spec) {
            if (const auto g = global_names.find(v); g != global_names.end()) return g->second;
            const std::vector<std::uint32_t> id = decoration(v, spv::DecSpecId);
            if (id.empty()) fail("a specialization constant without an id");
            const std::string n = unique("spec" + std::to_string(id[0]));
            global_names[v] = n;
            const char k = kind_of(c.type);
            spec_decls.push_back("[[vk::constant_id(" + std::to_string(id[0]) + ")]] const " + std::string(scalar_name(k)) + " " + n + " = " +
                                 scalar_literal(const_bits(c), k) + ";");
            return n;
        }
        const Type& t = type(c.type);
        if (t.k == TK::Vector) {
            std::string s = type_name(c.type) + "(";
            for (std::uint32_t k = 0; k < t.count; ++k) {
                if (k) s += ", ";
                if (c.null || c.undef) s += scalar_literal(0, kind_of(t.elem));
                else s += const_expr(c.parts.at(k));
            }
            return s + ")";
        }
        if (t.k == TK::Bool || t.k == TK::Int || t.k == TK::Float) return scalar_literal(const_bits(c), kind_of(c.type));
        fail("a constant of an aggregate type");
    }
    // `v` reinterpreted as kind `k` (same width): asuint / asint / asfloat.
    std::string as(std::uint32_t v, char k) {
        const std::uint32_t t = type_of(v);
        const char vk = kind_of(t);
        if (vk == k) return ref(v);
        if (const auto c = consts.find(v); c != consts.end() && !c->second.spec && type(c->second.type).k != TK::Vector) {
            return scalar_literal(const_bits(c->second), k);
        }
        const std::string r = ref(v);
        switch (k) {
        case 'u': if (vk == 'i' || vk == 'f') return "asuint(" + r + ")"; break;
        case 'i': if (vk == 'u' || vk == 'f') return "asint(" + r + ")"; break;
        case 'f': if (vk == 'u' || vk == 'i') return "asfloat(" + r + ")"; break;
        case 'U': if (vk == 'I') return "uint64_t(" + r + ")"; break;
        case 'I': if (vk == 'U') return "int64_t(" + r + ")"; break;
        }
        fail(std::string("a reinterpretation from ") + vk + " to " + k);
    }
    // The type name of `t` with its scalar kind replaced by `k`.
    std::string with_kind(std::uint32_t t, char k) {
        const std::uint32_t n = lanes_of(t);
        return std::string(scalar_name(k)) + (n > 1 ? std::to_string(n) : std::string());
    }

    // Declares (or, hoisted, assigns) value `id` = `expr`.
    void define(const Ins& in, const std::string& expr, const std::string& comment = {}) {
        const std::uint32_t t = in.type;
        const char k = kind_of(t);
        const std::string tail = comment.empty() ? std::string(";") : ";  // " + comment;
        if (hoisted.count(in.id)) {
            line(value_name(in.id) + " = " + expr + tail);
            return;
        }
        line(std::string(k == 'f' || k == 'h' ? "precise " : "") + type_name(t) + " " + value_name(in.id) + " = " + expr + tail);
    }
    // The member a constant-buffer load reads, where its index is the
    // buffer's base (params.cb_bias_dw) plus a constant: "cbSceneParam.FC_MatrixProj[2].w".
    std::string member_comment(const std::string& buffer, std::uint32_t index) {
        const auto members = buffer_members.find(buffer);
        if (members == buffer_members.end()) return {};
        const auto d = defs.find(index);
        if (d == defs.end() || d->second->op != spv::OpIAdd) return {};
        std::uint32_t k = 0, other = 0;
        if (consts.count(d->second->a.at(1)) && !consts.at(d->second->a[1]).spec) {
            k = static_cast<std::uint32_t>(const_bits(consts.at(d->second->a[1])));
            other = d->second->a.at(0);
        } else if (consts.count(d->second->a.at(0)) && !consts.at(d->second->a[0]).spec) {
            k = static_cast<std::uint32_t>(const_bits(consts.at(d->second->a[0])));
            other = d->second->a.at(1);
        } else {
            return {};
        }
        // The other operand: the buffer's base dword, a load of params.cb_bias_dw.
        const auto o = defs.find(other);
        if (o == defs.end() || o->second->op != spv::OpLoad) return {};
        const auto h = handles.find(o->second->a.at(0));
        if (h == handles.end() || h->second.lvalue.find(".cb_bias_dw[") == std::string::npos) return {};
        const std::uint32_t byte = 4 * k;
        for (const ReflectedElement& e : *members->second) {
            if (byte < e.offset || byte >= e.offset + e.size) continue;
            const std::uint32_t at = byte - e.offset;
            static const char* comp[] = {".x", ".y", ".z", ".w"};
            std::string s = buffer + "." + e.name;
            if (e.size > 16) s += "[" + std::to_string(at / 16) + "]";
            if (e.size > 4) s += comp[(at % 16) / 4];
            return s;
        }
        return {};
    }

    // Inline SPIR-V helpers, declared once per signature.
    std::set<std::string> helper_keys;
    std::string inline_op(const std::string& name, int opcode, bool glsl_set, std::uint32_t result_type, const std::vector<std::uint32_t>& args) {
        std::string sig = name;
        std::string decl_args;
        for (std::size_t k = 0; k < args.size(); ++k) {
            const std::string tn = type_name(type_of(args[k]));
            sig += "_" + tn;
            if (k) decl_args += ", ";
            decl_args += tn + " a" + std::to_string(k);
        }
        const std::string rt = type_name(result_type);
        sig += "->" + rt;
        if (helper_keys.insert(sig).second) {
            helper_decls.push_back("[[vk::ext_instruction(" + std::to_string(opcode) + (glsl_set ? ", \"GLSL.std.450\"" : "") + ")]] " + rt + " " +
                                   name + "(" + decl_args + ");");
        }
        std::string call = name + "(";
        for (std::size_t k = 0; k < args.size(); ++k) call += (k ? ", " : "") + ref(args[k]);
        return call + ")";
    }

    // ---- statements ------------------------------------------------------------------------
    void block_body(const Block& b) {
        if (!dry && printed_blocks.count(b.label)) fail("a block printed twice");
        printed_blocks.insert(b.label);
        block_scope[b.label] = scope;
        for (const Ins* in : b.body) statement(*in);
    }

    static const char* swizzle(std::uint32_t k) {
        static const char* s[] = {"x", "y", "z", "w"};
        if (k > 3) fail("a vector component past 3");
        return s[k];
    }

    void statement(const Ins& in) {
        switch (in.op) {
        case spv::OpLoad: return load(in);
        case spv::OpStore: return store(in);
        case spv::OpAccessChain: return access_chain(in);
        case kOpSampledImageOp: {
            Handle h;
            h.kind = Handle::Kind::SampledImage;
            h.image_var = image_root(in.a.at(0));
            h.sampler_var = sampler_root(in.a.at(1));
            handles[in.id] = h;
            return;
        }
        case kOpCopyObject:
            if (handles.count(in.a.at(0))) {
                handles[in.id] = handles.at(in.a.at(0));
                return;
            }
            return define(in, ref(in.a.at(0)));
        case spv::OpUndef: return define(in, zero(in.type));
        case spv::OpNop: return;
        case spv::OpFunctionCall: {
            const std::uint32_t callee = in.a.at(0);
            if (!global_names.count(callee)) fail("a call of an unknown function");
            std::string call = global_names.at(callee) + "(";
            for (std::size_t k = 1; k < in.a.size(); ++k) call += (k > 1 ? ", " : "") + ref(in.a[k]);
            call += ")";
            if (type(in.type).k == TK::Void) line(call + ";");
            else define(in, call);
            return;
        }
        case 120 /* OpConvertUToPtr */: {
            const Type& pt = type(in.type);
            if (pt.k != TK::Pointer || pt.sc != spv::ScPhysicalStorageBuffer) fail("a pointer made of an integer outside physical storage");
            Handle h;
            h.kind = Handle::Kind::Physical;
            h.address = as(in.a.at(0), 'U');
            h.pointee = pt.elem;
            h.uses = {in.a.at(0)};
            handles[in.id] = h;
            return;
        }
        case spv::OpAtomicIAdd: return atomic(in);
        case spv::OpImageWrite: {
            const auto hd = handles.find(in.a.at(0));
            if (hd == handles.end() || !hd->second.image_var) fail("an image write to an image that is not a loaded variable");
            if (in.a.size() > 3) fail("image write operands");
            const std::uint32_t it = type(var_type.at(hd->second.image_var)).elem;
            line(global_names.at(hd->second.image_var) + "[" + coords(in.a.at(1), coord_count(it, true), 'u') + "] = " +
                 as(in.a.at(2), kind_of(type(it).elem) == 'h' ? 'f' : kind_of(type(it).elem)) + ";");
            return;
        }
        case spv::OpControlBarrier:
            if (exec_model == spv::EmGLCompute) {
                // As the lift's: the scopes and semantics exactly (no HLSL barrier has these).
                line(inline_op_void("spv_control_barrier", spv::OpControlBarrier, {in.a.at(0), in.a.at(1), in.a.at(2)}) + ";");
                return;
            }
            [[fallthrough]];
        case 0xffff: {
            // The hull's barrier between its control point and its patch: the
            // control-point phase ends here (see hlsl.h).
            if (exec_model != spv::EmTessellationControl) fail("a barrier outside a hull shader");
            if (scope != 0) fail("a barrier inside a loop or a branch");
            if (past_barrier) fail("a second barrier");
            past_barrier = true;
            line("// The barrier: every control point is written. The control-point phase ends here.");
            line("if (hull_phase_cp) return;");
            return;
        }
        default: break;
        }
        if (in.op >= spv::OpImageSampleImplicitLod && in.op <= spv::OpImageQueryLevels) return image_op(in);
        if (in.op == spv::OpExtInst) return ext_inst(in);
        arithmetic(in);
    }

    std::string zero(std::uint32_t t) {
        const Type& ty = type(t);
        if (ty.k == TK::Vector) {
            std::string s = type_name(t) + "(";
            for (std::uint32_t k = 0; k < ty.count; ++k) s += (k ? ", " : "") + scalar_literal(0, kind_of(ty.elem));
            return s + ")";
        }
        return scalar_literal(0, kind_of(t));
    }

    std::uint32_t image_root(std::uint32_t value) const {
        const auto d = defs.find(value);
        if (d == defs.end() || d->second->op != spv::OpLoad || !var_type.count(d->second->a.at(0))) fail("an image that is not a loaded variable");
        return d->second->a.at(0);
    }

    void load(const Ins& in) {
        const std::uint32_t p = in.a.at(0);
        if (in.a.size() > 1 && !(handles.count(p) && handles.at(p).kind == Handle::Kind::Physical)) fail("a load with memory operands");
        if (var_type.count(p)) {
            const Type& pt = type(var_type.at(p));
            if (pt.sc == spv::ScUniformConstant) {
                Handle h;
                h.kind = type(pt.elem).k == TK::Sampler ? Handle::Kind::Sampler : Handle::Kind::Image;
                (h.kind == Handle::Kind::Sampler ? h.sampler_var : h.image_var) = p;
                handles[in.id] = h;
                return;
            }
            if (pt.sc == spv::ScInput || pt.sc == spv::ScOutput) return define(in, global_names.at(p));
            fail("a load of a whole block");
        }
        const auto h = handles.find(p);
        if (h == handles.end()) fail("a load through a pointer that is no access chain");
        if (h->second.kind == Handle::Kind::Physical) {
            for (const std::uint32_t u : h->second.uses) use(u);
            if (type(in.type).k != TK::Int && type(in.type).k != TK::Float && type(in.type).k != TK::Vector) fail("a raw load of an aggregate");
            return define(in, "vk::RawBufferLoad<" + type_name(in.type) + ">(" + h->second.address + aligned(in, 1) + ")");
        }
        if (h->second.kind == Handle::Kind::BufferWord) {
            for (const std::uint32_t u : h->second.uses) use(u);
            const std::string e = h->second.lvalue + ".Load(4u * " + as(h->second.index, 'u') + ")";
            const char k = kind_of(in.type);
            if (lanes_of(in.type) != 1) fail("a buffer load of a vector");
            return define(in, k == 'u' ? e : k == 'i' ? "asint(" + e + ")" : k == 'f' ? "asfloat(" + e + ")" : (fail("a buffer load of this type"), ""),
                          member_comment(h->second.lvalue, h->second.index));
        }
        for (const std::uint32_t u : h->second.uses) use(u);
        if (const auto e = hull_early_cp_read.find(p); e != hull_early_cp_read.end()) {
            // mine<suffix> where the index is this invocation's, the output patch's element otherwise.
            const std::string own = h->second.lvalue, mine = global_names.at(cp_out_var) + "_mine";
            const std::string suffix = own.substr(mine.size());
            std::string inv;
            for (const StageVar& v : stage_vars) {
                if (v.builtin == spv::BiInvocationId) inv = v.name;
            }
            const std::string idx = as(e->second, 'u');
            return define(in, "select(" + idx + " == " + inv + ", " + own + ", " + global_names.at(cp_out_var) + "[" + idx + "]" + suffix + ")");
        }
        define(in, h->second.lvalue);
    }

    // A raw load's or store's Aligned memory operand (from operand `at`), as RawBuffer*'s alignment argument.
    std::string aligned(const Ins& in, std::size_t at) {
        if (in.a.size() <= at) return "";
        const std::uint32_t mask = in.a[at];
        if (mask != 2 /* Aligned */ || in.a.size() != at + 2) fail("memory operands other than Aligned");
        return ", " + std::to_string(in.a[at + 1]);
    }

    void store(const Ins& in) {
        const std::uint32_t p = in.a.at(0), v = in.a.at(1);
        const bool physical = handles.count(p) && handles.at(p).kind == Handle::Kind::Physical;
        if (in.a.size() > 2 && !physical) fail("a store with memory operands");
        if (var_type.count(p)) {
            const Type& pt = type(var_type.at(p));
            if (pt.sc != spv::ScOutput) fail("a store to a variable that is no output");
            if (p == cp_out_var && cp_out_var) fail("a store to the whole output patch");
            if (type(pt.elem).k == TK::Array) {
                // A whole array (clip distances): element by element.
                const Type& at = type(pt.elem);
                const auto d = defs.find(v);
                if (d == defs.end() || d->second->op != spv::OpCompositeConstruct) fail("a store of an array that is not constructed");
                for (std::uint32_t k = 0; k < at.count; ++k) line(global_names.at(p) + "[" + std::to_string(k) + "] = " + ref(d->second->a.at(k)) + ";");
                return;
            }
            line(global_names.at(p) + " = " + ref(v) + ";");
            return;
        }
        const auto h = handles.find(p);
        if (h == handles.end()) fail("a store through a pointer that is no access chain");
        if (h->second.kind == Handle::Kind::Physical) {
            for (const std::uint32_t u : h->second.uses) use(u);
            const std::uint32_t t = type_of(v);
            line("vk::RawBufferStore<" + type_name(t) + ">(" + h->second.address + ", " + ref(v) + aligned(in, 2) + ");");
            return;
        }
        if (hull_late_cp_access.count(p)) fail("a control point output written after the barrier");
        if (hull_early_cp_read.count(p)) fail("a control point output written at an index other than the invocation's");
        for (const std::uint32_t u : h->second.uses) use(u);
        if (h->second.kind == Handle::Kind::BufferWord) {
            line(h->second.lvalue + ".Store(4u * " + as(h->second.index, 'u') + ", " + as(v, 'u') + ");");
            return;
        }
        line(h->second.lvalue + " = " + ref(v) + ";");
    }

    void access_chain(const Ins& in) {
        const std::uint32_t base = in.a.at(0);
        if (const auto hb = handles.find(base); hb != handles.end() && hb->second.kind == Handle::Kind::Physical) {
            // Offsets from the raw address, by the pointee's layout decorations.
            Handle h = hb->second;
            std::uint32_t t = h.pointee;
            for (std::size_t k = 1; k < in.a.size(); ++k) {
                const std::uint32_t idx = in.a[k];
                const Type& ty = type(t);
                const auto c = consts.find(idx);
                const bool constant = c != consts.end() && !c->second.spec;
                if (ty.k == TK::Struct) {
                    if (!constant) fail("a struct member chosen at run time");
                    const std::uint32_t m = static_cast<std::uint32_t>(const_bits(c->second));
                    const std::uint32_t off = member_offset(t, m);
                    if (off) h.address = "(" + h.address + " + " + std::to_string(off) + "ull)";
                    t = ty.members.at(m);
                } else if (ty.k == TK::Array || ty.k == TK::RuntimeArray) {
                    const std::vector<std::uint32_t> stride = decoration(t, spv::DecArrayStride);
                    if (stride.empty()) fail("a raw array without a stride");
                    h.address = "(" + h.address + " + uint64_t(" + as(idx, kind_of(type_of(idx)) == 'i' ? 'i' : 'u') + ") * " + std::to_string(stride[0]) + "ull)";
                    h.uses.push_back(idx);
                    t = ty.elem;
                } else {
                    fail("a raw access chain into a vector or scalar");
                }
            }
            h.pointee = t;
            handles[in.id] = h;
            return;
        }
        if (!var_type.count(base)) fail("an access chain from another access chain");
        const Type& pt = type(var_type.at(base));
        Handle h;
        std::uint32_t t = pt.elem;
        if (pt.sc == spv::ScStorageBuffer) {
            if (in.a.size() != 3) fail("a storage buffer access that is not one dword");
            const auto c = consts.find(in.a[1]);
            if (c == consts.end() || const_bits(c->second) != 0) fail("a storage buffer access past its first member");
            h.kind = Handle::Kind::BufferWord;
            h.lvalue = global_names.at(base);
            h.index = in.a[2];
            h.uses = {in.a[2]};
            handles[in.id] = h;
            return;
        }
        if (pt.sc != spv::ScUniform && pt.sc != spv::ScInput && pt.sc != spv::ScOutput) fail("an access chain into storage class " + std::to_string(pt.sc));
        h.kind = Handle::Kind::Lvalue;
        h.lvalue = global_names.at(base);
        std::size_t first = 1;
        if (base == cp_out_var && cp_out_var) {
            // Before the barrier an invocation reaches only its own control
            // point (control_point_out_mine), after it every one (the output patch).
            if (in.a.size() < 2) fail("an access to the whole output patch");
            if (!past_barrier) {
                const auto d = defs.find(in.a[1]);
                const bool own = d != defs.end() && d->second->op == spv::OpLoad && d->second->a.at(0) == invocation_var && invocation_var;
                if (!own) {
                    // A read at an index worked out at run time (an LDS address
                    // decoded region by region): its own control point where the
                    // index is the invocation's, another's (unsynchronized in the
                    // lift as well) from the output patch. A store must be its own.
                    hull_early_cp_read[in.id] = in.a[1];
                    h.uses.push_back(in.a[1]);
                }
                h.lvalue = global_names.at(base) + "_mine";
                first = 2;
                t = type(t).elem;
            } else {
                hull_late_cp_access.insert(in.id);
            }
        }
        for (std::size_t k = first; k < in.a.size(); ++k) {
            const std::uint32_t idx = in.a[k];
            const Type& ty = type(t);
            const auto c = consts.find(idx);
            const bool constant = c != consts.end() && !c->second.spec;
            switch (ty.k) {
            case TK::Struct: {
                if (!constant) fail("a struct member chosen at run time");
                const std::uint32_t m = static_cast<std::uint32_t>(const_bits(c->second));
                h.lvalue += "." + member_field.at({t, m});
                t = ty.members.at(m);
                break;
            }
            case TK::Array:
                if (constant) h.lvalue += "[" + std::to_string(const_bits(c->second)) + "]";
                else {
                    h.lvalue += "[" + ref_quiet(idx) + "]";
                    h.uses.push_back(idx);
                }
                t = ty.elem;
                break;
            case TK::Vector:
                if (constant) h.lvalue += std::string(".") + swizzle(static_cast<std::uint32_t>(const_bits(c->second)));
                else {
                    h.lvalue += "[" + ref_quiet(idx) + "]";
                    h.uses.push_back(idx);
                }
                t = ty.elem;
                break;
            default: fail("an access chain into a scalar");
            }
        }
        handles[in.id] = h;
    }
    // A value's name without noting a use (an access chain's index, noted where the chain is used).
    std::string ref_quiet(std::uint32_t v) {
        if (consts.count(v)) return const_expr(v);
        return value_name(v);
    }

    // OpAtomicIAdd: on a raw address (a compute lift's counters) or a buffer word.
    void atomic(const Ins& in) {
        const std::uint32_t p = in.a.at(0), scope = in.a.at(1), sem = in.a.at(2), v = in.a.at(3);
        const auto h = handles.find(p);
        if (h == handles.end()) fail("an atomic through a pointer that is no access chain");
        const auto sc = consts.find(scope), sm = consts.find(sem);
        if (sc == consts.end() || sm == consts.end() || const_bits(sc->second) != spv::ScopeDevice || const_bits(sm->second) != 0)
            fail("an atomic with a scope or semantics HLSL's InterlockedAdd does not write");
        const char k = kind_of(in.type);
        if (k != 'u' && k != 'i') fail("an atomic on a value that is not a 32-bit integer");
        for (const std::uint32_t u : h->second.uses) use(u);
        const std::string orig = value_name(in.id) + "_orig";
        line(type_name(in.type) + " " + orig + ";");
        if (h->second.kind == Handle::Kind::Physical) {
            hv202x = true;
            line("InterlockedAdd(vk::BufferPointer<" + type_name(in.type) + ", 4>(" + h->second.address + ").Get(), " + as(v, k) + ", " + orig + ");");
        } else if (h->second.kind == Handle::Kind::BufferWord) {
            line(h->second.lvalue + ".InterlockedAdd(4u * " + as(h->second.index, 'u') + ", " + as(v, 'u') + ", " + orig + ");");
        } else {
            fail("an atomic on a variable");
        }
        define(in, orig);
    }
    // An inline SPIR-V instruction with no result, declared once per signature.
    std::string inline_op_void(const std::string& name, int opcode, const std::vector<std::uint32_t>& args) {
        std::string sig = name, decl_args;
        for (std::size_t k = 0; k < args.size(); ++k) {
            const std::string tn = type_name(type_of(args[k]));
            sig += "_" + tn;
            decl_args += (k ? ", " : "") + tn + " a" + std::to_string(k);
        }
        if (helper_keys.insert(sig).second)
            helper_decls.push_back("[[vk::ext_instruction(" + std::to_string(opcode) + ")]] void " + name + "(" + decl_args + ");");
        std::string call = name + "(";
        for (std::size_t k = 0; k < args.size(); ++k) call += (k ? ", " : "") + ref(args[k]);
        return call + ")";
    }

    void arithmetic(const Ins& in) {
        const auto bin = [&](const char* o, char k) { return as(in.a.at(0), k) + " " + o + " " + as(in.a.at(1), k); };
        const char rk = in.type && types.count(in.type) && (type(in.type).k == TK::Bool || type(in.type).k == TK::Int ||
                                                              type(in.type).k == TK::Float || type(in.type).k == TK::Vector)
                            ? kind_of(in.type)
                            : 0;
        const auto signed_of = [](char k) { return k == 'U' || k == 'I' ? 'I' : 'i'; };
        const auto unsigned_of = [](char k) { return k == 'U' || k == 'I' ? 'U' : 'u'; };
        const auto back = [&](const std::string& e, char from) {
            // e has kind `from`; the result has kind rk.
            if (from == rk) return e;
            if (rk == 'u' || rk == 'i' || rk == 'f') return std::string(rk == 'u' ? "asuint(" : rk == 'i' ? "asint(" : "asfloat(") + e + ")";
            if (rk == 'U') return "uint64_t(" + e + ")";
            if (rk == 'I') return "int64_t(" + e + ")";
            fail("a result kind");
        };
        const auto operand_kind = [&](std::size_t k) { return kind_of(type_of(in.a.at(k))); };
        if (has_half) {
            // Halves are converted and moved, nothing else (the lifter writes
            // nothing else of them): an operation on halves in HLSL's spelling
            // DXC could evaluate itself, rounding to nearest where the module
            // says toward zero.
            std::size_t ids = in.a.size();  // the operands that are values; the rest are literals
            if (in.op == spv::OpCompositeExtract) ids = 1;
            if (in.op == spv::OpCompositeInsert || in.op == spv::OpVectorShuffle) ids = 2;
            bool half = rk == 'h';
            for (std::size_t k = 0; k < ids && k < in.a.size() && !half; ++k) half = is_half(in.a[k]);
            if (half) {
                switch (in.op) {
                case spv::OpFConvert: case spv::OpBitcast: case spv::OpCompositeConstruct: case spv::OpCompositeExtract:
                case spv::OpCompositeInsert: case spv::OpVectorShuffle: case spv::OpSelect:
                    break;
                default: fail("an operation on halves other than a conversion, a bitcast or a move (opcode " + std::to_string(in.op) + ")");
                }
            }
        }
        switch (in.op) {
        case spv::OpIAdd: return define(in, bin("+", rk));
        case spv::OpISub: return define(in, bin("-", rk));
        case spv::OpIMul: return define(in, bin("*", rk));
        case spv::OpUDiv: return define(in, back(bin("/", unsigned_of(rk)), unsigned_of(rk)));
        case spv::OpUMod: return define(in, back(bin("%", unsigned_of(rk)), unsigned_of(rk)));
        case spv::OpSDiv: return define(in, back(bin("/", signed_of(rk)), signed_of(rk)));
        case spv::OpSRem: return define(in, back(bin("%", signed_of(rk)), signed_of(rk)));
        case spv::OpSNegate: return define(in, back("-(" + as(in.a.at(0), signed_of(rk)) + ")", signed_of(rk)));
        case spv::OpFAdd: return define(in, bin("+", rk));
        case spv::OpFSub: return define(in, bin("-", rk));
        case spv::OpFMul: return define(in, bin("*", rk));
        case spv::OpFDiv: return define(in, bin("/", rk));
        case spv::OpFNegate: return define(in, "-(" + ref(in.a.at(0)) + ")");
        case spv::OpBitwiseAnd: return define(in, bin("&", rk));
        case spv::OpBitwiseOr: return define(in, bin("|", rk));
        case spv::OpBitwiseXor: return define(in, bin("^", rk));
        case spv::OpNot: return define(in, "~(" + as(in.a.at(0), rk) + ")");
        case spv::OpShiftLeftLogical: return define(in, as(in.a.at(0), rk) + " << " + as(in.a.at(1), unsigned_of(operand_kind(1))));
        case spv::OpShiftRightLogical:
            return define(in, back(as(in.a.at(0), unsigned_of(rk)) + " >> " + as(in.a.at(1), unsigned_of(operand_kind(1))), unsigned_of(rk)));
        case spv::OpShiftRightArithmetic:
            return define(in, back(as(in.a.at(0), signed_of(rk)) + " >> " + as(in.a.at(1), unsigned_of(operand_kind(1))), signed_of(rk)));
        case spv::OpIEqual: return define(in, bin("==", operand_kind(0) == 'U' || operand_kind(0) == 'I' ? 'U' : 'u'));
        case spv::OpINotEqual: return define(in, bin("!=", operand_kind(0) == 'U' || operand_kind(0) == 'I' ? 'U' : 'u'));
        case spv::OpULessThan: return define(in, bin("<", unsigned_of(operand_kind(0))));
        case spv::OpULessThanEqual: return define(in, bin("<=", unsigned_of(operand_kind(0))));
        case spv::OpUGreaterThan: return define(in, bin(">", unsigned_of(operand_kind(0))));
        case spv::OpUGreaterThanEqual: return define(in, bin(">=", unsigned_of(operand_kind(0))));
        case spv::OpSLessThan: return define(in, bin("<", signed_of(operand_kind(0))));
        case spv::OpSLessThanEqual: return define(in, bin("<=", signed_of(operand_kind(0))));
        case spv::OpSGreaterThan: return define(in, bin(">", signed_of(operand_kind(0))));
        case spv::OpSGreaterThanEqual: return define(in, bin(">=", signed_of(operand_kind(0))));
        // Ordered comparisons are HLSL's own (DXC: OpFOrd*); an unordered one
        // is the ordered complement negated, which is the same with NaNs.
        case spv::OpFOrdEqual: return define(in, bin("==", operand_kind(0)));
        case spv::OpFOrdNotEqual: return define(in, bin("!=", operand_kind(0)));
        case spv::OpFOrdLessThan: return define(in, bin("<", operand_kind(0)));
        case spv::OpFOrdLessThanEqual: return define(in, bin("<=", operand_kind(0)));
        case spv::OpFOrdGreaterThan: return define(in, bin(">", operand_kind(0)));
        case spv::OpFOrdGreaterThanEqual: return define(in, bin(">=", operand_kind(0)));
        case spv::OpFUnordEqual: return define(in, "!(" + bin("!=", operand_kind(0)) + ")");
        case spv::OpFUnordNotEqual: return define(in, "!(" + bin("==", operand_kind(0)) + ")");
        case spv::OpFUnordLessThan: return define(in, "!(" + bin(">=", operand_kind(0)) + ")");
        case spv::OpFUnordLessThanEqual: return define(in, "!(" + bin(">", operand_kind(0)) + ")");
        case spv::OpFUnordGreaterThan: return define(in, "!(" + bin("<=", operand_kind(0)) + ")");
        case spv::OpFUnordGreaterThanEqual: return define(in, "!(" + bin("<", operand_kind(0)) + ")");
        case spv::OpIsNan: return define(in, "isnan(" + ref(in.a.at(0)) + ")");
        case spv::OpIsInf: return define(in, "isinf(" + ref(in.a.at(0)) + ")");
        case spv::OpLogicalAnd: return define(in, "and(" + ref(in.a.at(0)) + ", " + ref(in.a.at(1)) + ")");
        case spv::OpLogicalOr: return define(in, "or(" + ref(in.a.at(0)) + ", " + ref(in.a.at(1)) + ")");
        case spv::OpLogicalNot: return define(in, "!(" + ref(in.a.at(0)) + ")");
        case spv::OpLogicalEqual: return define(in, ref(in.a.at(0)) + " == " + ref(in.a.at(1)));
        case spv::OpLogicalNotEqual: return define(in, ref(in.a.at(0)) + " != " + ref(in.a.at(1)));
        case spv::OpSelect: {
            if (type(type_of(in.a.at(0))).k == TK::Vector && lanes_of(in.type) != lanes_of(type_of(in.a.at(0)))) fail("a select of mixed widths");
            return define(in, "select(" + ref(in.a.at(0)) + ", " + as(in.a.at(1), rk) + ", " + as(in.a.at(2), rk) + ")");
        }
        case spv::OpBitcast: {
            const std::uint32_t from = type_of(in.a.at(0));
            const char fk = kind_of(from);
            if (fk == 'h' || rk == 'h') {
                // A pair of halves and the 32-bit word that holds them (a
                // packed export's): the lift's own instruction, which HLSL
                // writes only through 16-bit integers.
                if (bits_of(from) != bits_of(in.type)) fail("a bitcast between widths");
                return define(in, inline_op("spv_bitcast", spv::OpBitcast, false, in.type, {in.a.at(0)}));
            }
            if (lanes_of(from) != lanes_of(in.type)) fail("a bitcast between vectors of different sizes");
            if ((fk == 'U' || fk == 'I') != (rk == 'U' || rk == 'I')) fail("a bitcast between widths");
            return define(in, as(in.a.at(0), rk));
        }
        case spv::OpConvertFToU: return define(in, with_kind(in.type, rk) + "(" + ref(in.a.at(0)) + ")");
        case spv::OpConvertFToS: return define(in, with_kind(in.type, rk) + "(" + ref(in.a.at(0)) + ")");
        case spv::OpConvertSToF: return define(in, with_kind(in.type, rk) + "(" + as(in.a.at(0), signed_of(operand_kind(0))) + ")");
        case spv::OpConvertUToF: return define(in, with_kind(in.type, rk) + "(" + as(in.a.at(0), unsigned_of(operand_kind(0))) + ")");
        case spv::OpUConvert:
            return define(in, back(with_kind(in.type, unsigned_of(rk)) + "(" + as(in.a.at(0), unsigned_of(operand_kind(0))) + ")", unsigned_of(rk)));
        case spv::OpSConvert:
            return define(in, back(with_kind(in.type, signed_of(rk)) + "(" + as(in.a.at(0), signed_of(operand_kind(0))) + ")", signed_of(rk)));
        case spv::OpFConvert: {
            // Between f32 and f16 (a lift for a device that rounds halves
            // itself): the lift's own instruction. To half it rounds as the
            // module's execution mode says - toward zero, denormals kept - by
            // the device, at run time; float16_t(x) of a constant DXC would
            // evaluate itself, rounding to nearest. From half it is exact.
            // Undecorated, as the lifter writes it: a Mesa driver (lavapipe;
            // RADV shares its compiler) removes an undecorated f32 -> f16 ->
            // f32 round trip, one decorated NoContraction it keeps - and DXC
            // has no way to decorate a conversion.
            const char fk = operand_kind(0);
            if (!((fk == 'f' && rk == 'h') || (fk == 'h' && rk == 'f'))) fail("a float conversion other than between 32 and 16 bits");
            if (lanes_of(type_of(in.a.at(0))) != lanes_of(in.type)) fail("a float conversion between vectors of different sizes");
            if (decorated(in.id, spv::DecNoContraction)) fail("a half conversion decorated NoContraction, which DXC cannot write");
            return define(in, inline_op(rk == 'h' ? "spv_to_half" : "spv_from_half", spv::OpFConvert, false, in.type, {in.a.at(0)}));
        }
        case spv::OpCompositeConstruct: {
            const Type& t = type(in.type);
            if (t.k != TK::Vector) fail("a constructed aggregate");
            std::string s = type_name(in.type) + "(";
            for (std::size_t k = 0; k < in.a.size(); ++k) s += (k ? ", " : "") + as(in.a[k], rk);
            return define(in, s + ")");
        }
        case spv::OpCompositeExtract: {
            const std::uint32_t src = in.a.at(0);
            if (const auto p = parts.find(src); p != parts.end()) {
                if (in.a.size() != 2) fail("a nested extract from a struct value");
                use(src);
                return define(in, in.a[1] == 0 ? p->second.first : p->second.second);
            }
            std::string e = ref(src);
            std::uint32_t t = type_of(src);
            for (std::size_t k = 1; k < in.a.size(); ++k) {
                const Type& ty = type(t);
                if (ty.k == TK::Vector) e += std::string(".") + swizzle(in.a[k]);
                else if (ty.k == TK::Array) e += "[" + std::to_string(in.a[k]) + "]";
                else fail("an extract from a struct");
                t = ty.elem;
            }
            return define(in, e);
        }
        case spv::OpCompositeInsert: {
            if (in.a.size() != 3 || type(in.type).k != TK::Vector) fail("an insert into an aggregate");
            define(in, ref(in.a.at(1)));
            line(value_name(in.id) + "." + swizzle(in.a[2]) + " = " + as(in.a.at(0), kind_of(type(in.type).elem)) + ";");
            return;
        }
        case spv::OpVectorShuffle: {
            const std::uint32_t n1 = lanes_of(type_of(in.a.at(0)));
            std::string s = type_name(in.type) + "(";
            for (std::size_t k = 2; k < in.a.size(); ++k) {
                const std::uint32_t c = in.a[k];
                if (k > 2) s += ", ";
                if (c == 0xffffffffu) s += scalar_literal(0, rk);
                else if (c < n1) s += ref(in.a[0]) + "." + swizzle(c);
                else s += ref(in.a[1]) + "." + swizzle(c - n1);
            }
            return define(in, s + ")");
        }
        case spv::OpIAddCarry: case spv::OpISubBorrow: case spv::OpUMulExtended: case spv::OpSMulExtended: return extended(in);
        case spv::OpBitFieldUExtract:
            return define(in, inline_op("spv_bitfield_uextract", spv::OpBitFieldUExtract, false, in.type, {in.a.at(0), in.a.at(1), in.a.at(2)}));
        case spv::OpBitFieldSExtract:
            return define(in, inline_op("spv_bitfield_sextract", spv::OpBitFieldSExtract, false, in.type, {in.a.at(0), in.a.at(1), in.a.at(2)}));
        case spv::OpBitFieldInsert:
            return define(in, inline_op("spv_bitfield_insert", spv::OpBitFieldInsert, false, in.type, {in.a.at(0), in.a.at(1), in.a.at(2), in.a.at(3)}));
        case spv::OpBitReverse: return define(in, back("reversebits(" + as(in.a.at(0), 'u') + ")", 'u'));
        case spv::OpBitCount: return define(in, back("countbits(" + as(in.a.at(0), 'u') + ")", 'u'));
        case spv::OpDPdx: return define(in, "ddx(" + ref(in.a.at(0)) + ")");
        case spv::OpDPdy: return define(in, "ddy(" + ref(in.a.at(0)) + ")");
        case spv::OpGroupNonUniformQuadBroadcast: {
            if (!consts.count(in.a.at(0)) || const_bits(consts.at(in.a[0])) != spv::ScopeSubgroup) fail("a quad operation of another scope");
            return define(in, "QuadReadLaneAt(" + ref(in.a.at(1)) + ", " + as(in.a.at(2), 'u') + ")");
        }
        case spv::OpGroupNonUniformQuadSwap: {
            if (!consts.count(in.a.at(0)) || const_bits(consts.at(in.a[0])) != spv::ScopeSubgroup) fail("a quad operation of another scope");
            const auto d = consts.find(in.a.at(2));
            if (d == consts.end()) fail("a quad swap whose direction is not a constant");
            static const char* fn[] = {"QuadReadAcrossX", "QuadReadAcrossY", "QuadReadAcrossDiagonal"};
            const std::uint64_t dir = const_bits(d->second);
            if (dir > 2) fail("a quad swap direction");
            return define(in, std::string(fn[dir]) + "(" + ref(in.a.at(1)) + ")");
        }
        default: break;
        }
        fail("opcode " + std::to_string(in.op));
    }

    // IAddCarry, ISubBorrow, UMulExtended, SMulExtended: the lift's own
    // instruction, inline SPIR-V returning a struct of two members (DXC takes
    // a struct result: spv_pair_uint { lo, hi }), then each member a value of
    // its own. Spelled out instead (the high half of a multiply as a 64-bit
    // multiply and a shift), the RTX's compiler refused some compute modules.
    void extended(const Ins& in) {
        const Type& st = type(in.type);
        if (st.k != TK::Struct || st.members.size() != 2) fail("an extended arithmetic result that is no pair");
        const std::uint32_t mt = st.members[0];
        const char k = kind_of(mt);
        if (k != 'u' && k != 'i') fail("extended arithmetic on 64-bit values");
        if (st.members[1] != mt) fail("an extended arithmetic result of two types");
        const std::string lo = value_name(in.id) + "_lo", hi = value_name(in.id) + "_hi", tn = type_name(mt);
        parts[in.id] = {lo, hi};
        const bool hoist = hoisted.count(in.id) != 0;
        const auto def = [&](const std::string& n, const std::string& e) { line((hoist ? std::string() : tn + " ") + n + " = " + e + ";"); };
        const char* fname = in.op == spv::OpIAddCarry ? "spv_iadd_carry" : in.op == spv::OpISubBorrow ? "spv_isub_borrow"
                          : in.op == spv::OpUMulExtended ? "spv_umul_extended" : "spv_smul_extended";
        const std::string pair = "spv_pair_" + tn;
        if (helper_keys.insert("struct " + pair).second) helper_decls.push_back("struct " + pair + " { " + tn + " lo; " + tn + " hi; };");
        if (helper_keys.insert(std::string(fname) + "_" + tn).second)
            helper_decls.push_back("[[vk::ext_instruction(" + std::to_string(in.op) + ")]] " + pair + " " + fname + "(" + tn + " a0, " + tn + " a1);");
        const std::string r = value_name(in.id) + "_r";
        line(pair + " " + r + " = " + fname + "(" + as(in.a.at(0), k) + ", " + as(in.a.at(1), k) + ");");
        def(lo, r + ".lo");
        def(hi, r + ".hi");
    }

    void ext_inst(const Ins& in) {
        if (in.a.at(0) != glsl) fail("an extended instruction of another set");
        const std::uint32_t op = in.a.at(1);
        const std::vector<std::uint32_t> args(in.a.begin() + 2, in.a.end());
        const char rk = kind_of(in.type);
        const bool wide = rk == 'U' || rk == 'I';
        if (has_half) {  // as arithmetic(): no operation on halves but conversions and moves
            bool half = rk == 'h';
            for (const std::uint32_t a : args) half = half || is_half(a);
            if (half) fail("an operation on halves other than a conversion, a bitcast or a move (GLSL.std.450 " + std::to_string(op) + ")");
        }
        const auto call = [&](const char* fn, char k) {
            std::string s = std::string(fn) + "(";
            for (std::size_t i = 0; i < args.size(); ++i) s += (i ? ", " : "") + as(args[i], k);
            return s + ")";
        };
        switch (op) {
        case spv::GlslRoundEven: return define(in, call("round", rk));
        case spv::GlslTrunc: return define(in, call("trunc", rk));
        case spv::GlslFAbs: return define(in, call("abs", rk));
        case spv::GlslSAbs: return define(in, call("abs", 'i'));
        case spv::GlslFloor: return define(in, call("floor", rk));
        case spv::GlslCeil: return define(in, call("ceil", rk));
        case spv::GlslFract: return define(in, call("frac", rk));
        case spv::GlslSin: return define(in, call("sin", rk));
        case spv::GlslCos: return define(in, call("cos", rk));
        case spv::GlslExp2: return define(in, call("exp2", rk));
        case spv::GlslLog2: return define(in, call("log2", rk));
        case spv::GlslSqrt: return define(in, call("sqrt", rk));
        case spv::GlslInverseSqrt: return define(in, call("rsqrt", rk));
        case spv::GlslFClamp: return define(in, call("clamp", rk));
        // Fma as inline SPIR-V: HLSL's mad() is GLSL.std.450 Fma decorated
        // NoContraction (DXC decorates it whether or not it is precise),
        // which a driver must fuse, while the lift's Fma carries no decoration
        // and a driver may fuse it or not (lavapipe splits it) - the two would
        // round differently.
        case spv::GlslFma: return define(in, inline_op("spv_fma", spv::GlslFma, true, in.type, args));
        case spv::GlslNMin: return define(in, call("min", rk));
        case spv::GlslNMax: return define(in, call("max", rk));
        // (64-bit too: a compute lift's page-table walk clamps an address with UMin.)
        case spv::GlslUMin: return define(in, as_kind_of(call("min", wide ? 'U' : 'u'), wide ? 'U' : 'u', rk));
        case spv::GlslUMax: return define(in, as_kind_of(call("max", wide ? 'U' : 'u'), wide ? 'U' : 'u', rk));
        case spv::GlslSMin: return define(in, as_kind_of(call("min", wide ? 'I' : 'i'), wide ? 'I' : 'i', rk));
        case spv::GlslSMax: return define(in, as_kind_of(call("max", wide ? 'I' : 'i'), wide ? 'I' : 'i', rk));
        case spv::GlslFindILsb: return define(in, as_kind_of(call("firstbitlow", 'u'), 'u', rk));
        case spv::GlslFindUMsb: return define(in, as_kind_of(call("firstbithigh", 'u'), 'u', rk));
        case spv::GlslFindSMsb: return define(in, as_kind_of(call("firstbithigh", 'i'), 'i', rk));
        // The operations whose HLSL spelling DXC lowers to something else.
        case spv::GlslRound: return define(in, inline_op("spv_round", spv::GlslRound, true, in.type, args));
        case spv::GlslFSign: return define(in, inline_op("spv_fsign", spv::GlslFSign, true, in.type, args));
        case 7 /* SSign */: return define(in, inline_op("spv_ssign", 7, true, in.type, args));
        case spv::GlslFMin: return define(in, inline_op("spv_fmin", spv::GlslFMin, true, in.type, args));
        case spv::GlslFMax: return define(in, inline_op("spv_fmax", spv::GlslFMax, true, in.type, args));
        case spv::GlslNClamp: return define(in, inline_op("spv_nclamp", spv::GlslNClamp, true, in.type, args));
        case spv::GlslUClamp: return define(in, inline_op("spv_uclamp", spv::GlslUClamp, true, in.type, args));
        case spv::GlslSClamp: return define(in, inline_op("spv_sclamp", spv::GlslSClamp, true, in.type, args));
        case spv::GlslFMix: return define(in, inline_op("spv_fmix", spv::GlslFMix, true, in.type, args));
        case spv::GlslLdexp: return define(in, inline_op("spv_ldexp", spv::GlslLdexp, true, in.type, args));
        case spv::GlslPackHalf2x16: return define(in, inline_op("spv_pack_half_2x16", spv::GlslPackHalf2x16, true, in.type, args));
        case spv::GlslUnpackHalf2x16: return define(in, inline_op("spv_unpack_half_2x16", spv::GlslUnpackHalf2x16, true, in.type, args));
        case spv::GlslPow: return define(in, inline_op("spv_pow", spv::GlslPow, true, in.type, args));
        case spv::GlslExp: return define(in, inline_op("spv_exp", spv::GlslExp, true, in.type, args));
        case spv::GlslLog: return define(in, inline_op("spv_log", spv::GlslLog, true, in.type, args));
        default: fail("GLSL.std.450 instruction " + std::to_string(op));
        }
    }
    std::string as_kind_of(const std::string& e, char from, char to) {
        if (from == to) return e;
        if ((from == 'U' && to == 'I') || (from == 'I' && to == 'U')) return std::string(to == 'U' ? "uint64_t(" : "int64_t(") + e + ")";
        return std::string(to == 'u' ? "asuint(" : to == 'i' ? "asint(" : "asfloat(") + e + ")";
    }

    // The image operands after `at`: Bias / Lod / Grad / ConstOffset / Offset / Sample.
    struct ImageOperands {
        std::uint32_t mask = 0;
        std::uint32_t bias = 0, lod = 0, dx = 0, dy = 0, offset = 0, sample = 0;
    };
    ImageOperands image_operands(const Ins& in, std::size_t at) {
        ImageOperands o;
        if (in.a.size() <= at) return o;
        o.mask = in.a[at];
        std::size_t k = at + 1;
        const auto next = [&] {
            if (k >= in.a.size()) fail("image operands past the instruction");
            return in.a[k++];
        };
        if (o.mask & ~static_cast<std::uint32_t>(spv::IoBias | spv::IoLod | spv::IoGrad | spv::IoConstOffset | spv::IoOffset | spv::IoSample))
            fail("image operands " + hex(o.mask));
        if (o.mask & spv::IoBias) o.bias = next();
        if (o.mask & spv::IoLod) o.lod = next();
        if (o.mask & spv::IoGrad) {
            o.dx = next();
            o.dy = next();
        }
        if (o.mask & spv::IoConstOffset) o.offset = next();
        if (o.mask & spv::IoOffset) o.offset = next();
        if (o.mask & spv::IoSample) o.sample = next();
        return o;
    }
    // `v` as a float (or int) vector of `n` components: the leading ones of a wider vector.
    std::string coords(std::uint32_t v, std::uint32_t n, char k) {
        const std::uint32_t have = lanes_of(type_of(v));
        if (have < n) fail("coordinates with fewer components than the image takes");
        std::string e = as(v, k);
        if (have == n) return e;
        if (n == 1) return "(" + e + ").x";
        static const char* sw[] = {"", "", ".xy", ".xyz"};
        return "(" + e + ")" + sw[n];
    }
    std::string offset_arg(const ImageOperands& o, std::uint32_t image_type) {
        if (!(o.mask & (spv::IoConstOffset | spv::IoOffset))) return "";
        if (o.mask & spv::IoOffset) runtime_offset = true;
        const std::uint32_t n = coord_count(image_type, false);
        return ", " + coords(o.offset, n, 'i');
    }

    void image_op(const Ins& in) {
        const auto h = handles.find(in.a.at(0));
        if (h == handles.end()) fail("an image operation on an image that is not a loaded variable");
        const Handle& hd = h->second;
        const std::uint32_t img = hd.image_var;
        if (!img) fail("an image operation without an image");
        const std::uint32_t it = type(var_type.at(img)).elem;
        const std::string iname = global_names.at(img);
        const auto sampler = [&](bool compare) {
            if (!hd.sampler_var) fail("a sample without a sampler");
            return compare ? (compare_name.count(hd.sampler_var) ? compare_name.at(hd.sampler_var) : global_names.at(hd.sampler_var))
                           : global_names.at(hd.sampler_var);
        };
        const std::uint32_t nc = coord_count(it, true);
        const char rk = kind_of(in.type);
        const auto texel = [&](const std::string& e) {
            // The image's texel type is the result's (float4 / uint4 / int4).
            const char ik = kind_of(type(it).elem);
            if (ik == rk && lanes_of(in.type) == 4) return e;
            if (lanes_of(in.type) != 4) fail("a sample result that is not four components");
            return as_kind_of(e, ik, rk);
        };
        switch (in.op) {
        case spv::OpImageSampleImplicitLod: {
            if (exec_model != spv::EmFragment) fail("an implicit-LOD sample outside a pixel shader");
            const ImageOperands o = image_operands(in, 2);
            if (o.mask & ~static_cast<std::uint32_t>(spv::IoBias | spv::IoConstOffset | spv::IoOffset)) fail("implicit-LOD sample operands");
            const std::string c = coords(in.a.at(1), nc, 'f');
            if (o.mask & spv::IoBias)
                return define(in, texel(iname + ".SampleBias(" + sampler(false) + ", " + c + ", " + ref(o.bias) + offset_arg(o, it) + ")"));
            return define(in, texel(iname + ".Sample(" + sampler(false) + ", " + c + offset_arg(o, it) + ")"));
        }
        case spv::OpImageSampleExplicitLod: {
            const ImageOperands o = image_operands(in, 2);
            const std::string c = coords(in.a.at(1), nc, 'f');
            if (o.mask & spv::IoLod) return define(in, texel(iname + ".SampleLevel(" + sampler(false) + ", " + c + ", " + as(o.lod, 'f') + offset_arg(o, it) + ")"));
            if (o.mask & spv::IoGrad) {
                const std::uint32_t ng = coord_count(it, false);
                return define(in, texel(iname + ".SampleGrad(" + sampler(false) + ", " + c + ", " + coords(o.dx, ng, 'f') + ", " + coords(o.dy, ng, 'f') +
                                        offset_arg(o, it) + ")"));
            }
            fail("an explicit-LOD sample without a LOD");
        }
        case spv::OpImageSampleDrefImplicitLod: {
            if (exec_model != spv::EmFragment) fail("an implicit-LOD sample outside a pixel shader");
            const ImageOperands o = image_operands(in, 3);
            if (o.mask & ~static_cast<std::uint32_t>(spv::IoConstOffset | spv::IoOffset)) fail("depth-compare sample operands");
            if (rk != 'f' || lanes_of(in.type) != 1) fail("a depth-compare result that is not a float");
            return define(in, iname + ".SampleCmp(" + sampler(true) + ", " + coords(in.a.at(1), nc, 'f') + ", " + as(in.a.at(2), 'f') + offset_arg(o, it) + ")");
        }
        case spv::OpImageSampleDrefExplicitLod: {
            const ImageOperands o = image_operands(in, 3);
            if (!(o.mask & spv::IoLod) || (o.mask & ~static_cast<std::uint32_t>(spv::IoLod | spv::IoConstOffset | spv::IoOffset))) fail("depth-compare sample operands");
            if (rk != 'f' || lanes_of(in.type) != 1) fail("a depth-compare result that is not a float");
            const auto lc = consts.find(o.lod);
            const bool zero_lod = lc != consts.end() && !lc->second.spec && const_bits(lc->second) == 0;
            if (!zero_lod) {
                profile[3] = '6', profile[5] = '7';  // SampleCmpLevel: Shader Model 6.7
                return define(in, iname + ".SampleCmpLevel(" + sampler(true) + ", " + coords(in.a.at(1), nc, 'f') + ", " + as(in.a.at(2), 'f') + ", " +
                                      as(o.lod, 'f') + offset_arg(o, it) + ")");
            }
            return define(in, iname + ".SampleCmpLevelZero(" + sampler(true) + ", " + coords(in.a.at(1), nc, 'f') + ", " + as(in.a.at(2), 'f') +
                                  offset_arg(o, it) + ")");
        }
        case spv::OpImageFetch: {
            const ImageOperands o = image_operands(in, 2);
            if (o.mask & ~static_cast<std::uint32_t>(spv::IoLod | spv::IoConstOffset)) fail("fetch operands");
            const Type& im = type(it);
            if (im.ms || im.dim == spv::DimBuffer) fail("a fetch from a multisampled or buffer image");
            const std::uint32_t n = coord_count(it, true);
            const std::string c = coords(in.a.at(1), n, 'i');
            const std::string lod = (o.mask & spv::IoLod) ? as(o.lod, 'i') : std::string("0");
            const std::string arg = "int" + std::to_string(n + 1) + "(" + c + ", " + lod + ")";
            return define(in, texel(iname + ".Load(" + arg + offset_arg(o, it) + ")"));
        }
        case spv::OpImageRead: {
            if (in.a.size() > 2) fail("image read operands");
            return define(in, texel(iname + "[" + coords(in.a.at(1), coord_count(it, true), 'u') + "]"));
        }
        case spv::OpImageQuerySizeLod: {
            const Type& im = type(it);
            const std::string id = value_name(in.id);
            std::vector<std::string> outs;
            const std::uint32_t n = coord_count(it, true);
            static const char* comp[] = {"_w", "_h", "_d"};
            for (std::uint32_t k = 0; k < n; ++k) outs.push_back(id + comp[k]);
            line("uint " + join(outs) + ", " + id + "_levels;");
            std::string call = iname + ".GetDimensions(" + as(in.a.at(1), 'u');
            for (const std::string& s : outs) call += ", " + s;
            line(call + ", " + id + "_levels);");
            (void)im;
            if (lanes_of(in.type) != n) fail("a size query of another width");
            const std::string v = n == 1 ? outs[0] : "uint" + std::to_string(n) + "(" + join(outs) + ")";
            return define(in, as_kind_of(v, 'u', rk));
        }
        case spv::OpImageQueryLevels: {
            const std::string id = value_name(in.id);
            const std::uint32_t n = coord_count(it, true);
            std::vector<std::string> outs;
            static const char* comp[] = {"_w", "_h", "_d"};
            for (std::uint32_t k = 0; k < n; ++k) outs.push_back(id + comp[k]);
            line("uint " + join(outs) + ", " + id + "_levels;");
            std::string call = iname + ".GetDimensions(0u";
            for (const std::string& s : outs) call += ", " + s;
            line(call + ", " + id + "_levels);");
            return define(in, as_kind_of(id + "_levels", 'u', rk));
        }
        case spv::OpImageQueryLod: {
            if (exec_model != spv::EmFragment) fail("a LOD query outside a pixel shader");
            if (rk != 'f' || lanes_of(in.type) != 2) fail("a LOD query result that is not a float2");
            const std::string c = coords(in.a.at(1), coord_count(it, false), 'f'), s = sampler(false);
            return define(in, "float2(" + iname + ".CalculateLevelOfDetail(" + s + ", " + c + "), " + iname + ".CalculateLevelOfDetailUnclamped(" + s + ", " +
                                  c + "))");
        }
        default: fail("image opcode " + std::to_string(in.op));
        }
    }
    static std::string join(const std::vector<std::string>& v) {
        std::string s;
        for (std::size_t k = 0; k < v.size(); ++k) s += (k ? ", " : "") + v[k];
        return s;
    }

    // ---- control flow -----------------------------------------------------------------------
    struct Ctx {
        std::uint32_t sel_merge = 0;
        std::uint32_t loop_merge = 0, loop_continue = 0, loop_header = 0;
    };

    // The phis of `to` given their values for the edge from `from`.
    void copies(std::uint32_t from, std::uint32_t to) {
        const auto b = fn->blocks.find(to);
        if (b == fn->blocks.end()) return;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> moves;  // phi, value
        for (const Ins* phi : b->second.phis) {
            for (std::size_t k = 0; k + 1 < phi->a.size(); k += 2) {
                if (phi->a[k + 1] == from) moves.push_back({phi->id, phi->a[k]});
            }
        }
        if (moves.empty()) return;
        // A value that is itself one of these phis must be read before any is written.
        bool parallel = false;
        std::set<std::uint32_t> targets;
        for (const auto& m : moves) targets.insert(m.first);
        for (const auto& m : moves) parallel |= targets.count(m.second) && m.second != m.first;
        if (!parallel) {
            for (const auto& [phi, v] : moves) {
                if (phi == v) continue;
                line(value_name(phi) + " = " + as(v, kind_of(type_of(phi))) + ";");
            }
            return;
        }
        line("{");
        open_scope();
        for (std::size_t k = 0; k < moves.size(); ++k) {
            line(type_name(type_of(moves[k].first)) + " t" + std::to_string(k) + " = " + as(moves[k].second, kind_of(type_of(moves[k].first))) + ";");
        }
        for (std::size_t k = 0; k < moves.size(); ++k) line(value_name(moves[k].first) + " = t" + std::to_string(k) + ";");
        close_scope();
        line("}");
    }

    void walk_function() {
        handles.clear();
        parts.clear();
        printed_blocks.clear();
        past_barrier = false;
        hull_late_cp_access.clear();
        hull_early_cp_read.clear();
        scope = 0;
        scope_count = 0;
        scope_parent.assign(1, 0);
        indent = 1;
        chain(fn->first_block, Ctx{});
    }

    // Prints blocks from `b` on until control leaves the construct `ctx` describes.
    void chain(std::uint32_t b, const Ctx& ctx) {
        for (;;) {
            if (b == ctx.sel_merge) return;
            if (b == ctx.loop_merge) {
                line("break;");
                return;
            }
            if (b == ctx.loop_header) return;  // the back edge
            if (b == ctx.loop_continue) {
                continue_block(ctx);
                return;
            }
            const Block& blk = fn->blocks.at(b);
            if (blk.merge && blk.merge->op == spv::OpLoopMerge) {
                b = loop(b);
                continue;
            }
            block_body(blk);
            const Ins& t = *blk.term;
            switch (t.op) {
            case spv::OpBranch:
                copies(b, t.a.at(0));
                b = t.a.at(0);
                continue;
            case spv::OpBranchConditional: {
                if (!blk.merge || blk.merge->op != spv::OpSelectionMerge) fail("a conditional branch without a selection merge");
                const std::uint32_t m = blk.merge->a.at(0), tt = t.a.at(1), ff = t.a.at(2);
                if (t.a.size() > 3) fail("branch weights");
                copies(b, tt);
                if (ff != tt) copies(b, ff);
                Ctx inner = ctx;
                inner.sel_merge = m;
                const std::string c = ref(t.a.at(0));
                if (tt != m && ff != m) {
                    line("if (" + c + ") {");
                    open_scope();
                    chain(tt, inner);
                    close_scope();
                    line("} else {");
                    open_scope();
                    chain(ff, inner);
                    close_scope();
                    line("}");
                } else if (tt != m) {
                    line("if (" + c + ") {");
                    open_scope();
                    chain(tt, inner);
                    close_scope();
                    line("}");
                } else if (ff != m) {
                    line("if (!" + c + ") {");
                    open_scope();
                    chain(ff, inner);
                    close_scope();
                    line("}");
                }
                b = m;
                continue;
            }
            case spv::OpKill: line("discard;"); return;
            case spv::OpReturn: line("return;"); return;
            case spv::OpReturnValue: line("return " + ref(t.a.at(0)) + ";"); return;
            case spv::OpUnreachable: return;
            default: fail("terminator " + std::to_string(t.op));
            }
        }
    }

    // A loop from its header; returns its merge block.
    std::uint32_t loop(std::uint32_t h) {
        const Block& hb = fn->blocks.at(h);
        const std::uint32_t m = hb.merge->a.at(0), c = hb.merge->a.at(1);
        if (c == h) fail("a loop whose header is its continue target");
        if (fn->preds[c] != 1) fail("a continue block reached from more than one block");
        line("[loop] for (;;) {");
        open_scope();
        Ctx ctx;
        ctx.loop_merge = m;
        ctx.loop_continue = c;
        ctx.loop_header = h;
        block_body(hb);
        const Ins& t = *hb.term;
        if (t.op != spv::OpBranch) fail("a loop header that does not branch on unconditionally");
        copies(h, t.a.at(0));
        if (t.a.at(0) == c) continue_block(ctx);
        else chain(t.a.at(0), ctx);
        close_scope();
        line("}");
        return m;
    }

    void continue_block(const Ctx& ctx) {
        const Block& cb = fn->blocks.at(ctx.loop_continue);
        if (cb.merge) fail("a continue block that is a header");
        block_body(cb);
        const Ins& t = *cb.term;
        if (t.op == spv::OpBranch && t.a.at(0) == ctx.loop_header) {
            copies(cb.label, ctx.loop_header);
            line("continue;");
            return;
        }
        if (t.op == spv::OpBranchConditional) {
            const std::uint32_t tt = t.a.at(1), ff = t.a.at(2);
            const std::string c = ref(t.a.at(0));
            if (tt == ctx.loop_header && ff == ctx.loop_merge) {
                line("if (" + c + ") {");
            } else if (ff == ctx.loop_header && tt == ctx.loop_merge) {
                line("if (!" + c + ") {");
            } else {
                fail("a continue block that leaves for somewhere other than its header or merge");
            }
            open_scope();
            copies(cb.label, ctx.loop_header);
            line("continue;");
            close_scope();
            line("}");
            copies(cb.label, ctx.loop_merge);
            line("break;");
            return;
        }
        fail("a continue block that does not branch back to its header");
    }

    // ---- the whole source -----------------------------------------------------------------
    // The device's half rounding as the lift declares it (float_modes): each
    // mode's capability on the entry point, ahead of HLSL's own attributes
    // (DXC takes them there), and the mode itself in the entry point's body.
    std::string float_mode_attributes() const {
        std::string s;
        for (const std::uint32_t m : float_modes) {
            s += m == spv::ExDenormPreserve ? "[[vk::ext_capability(" + std::to_string(spv::CapDenormPreserve) + " /* DenormPreserve */)]]\n"
                                            : "[[vk::ext_capability(" + std::to_string(spv::CapRoundingModeRTZ) + " /* RoundingModeRTZ */)]]\n";
        }
        return s;
    }
    std::string float_mode_calls() const {
        std::string s;
        for (const std::uint32_t m : float_modes) {
            s += "    vk::ext_execution_mode(" + std::to_string(m) + ", 16);  // " +
                 (m == spv::ExDenormPreserve ? "DenormPreserve: f16 denormals kept" : "RoundingModeRTZ: f32 to f16 rounds toward zero") + "\n";
        }
        return s;
    }

    // The hull shader's two phases from the one body (hlsl.h).
    std::string assemble_hull() {
        if (!output_vertices) fail("a hull shader without OutputVertices");
        if (!cp_out_var || !invocation_var) fail("a hull shader without its control-point output or invocation id");
        if (!past_barrier) fail("a hull shader without a barrier");
        const std::string n = std::to_string(output_vertices);
        std::string s, cp_in, cp_out, pc, copy_in, copy_out_cp, copy_pc, params, call_params;
        const bool quads = opt.tess_quads;
        for (const StageVar& v : stage_vars) {
            if (v.builtin == spv::BiInvocationId) {
                s += "static " + declare(v.pointee, v.name) + ";\n";
                continue;
            }
            if (v.input && v.builtin == spv::BiPrimitiveId) {
                s += "static " + declare(v.pointee, v.name) + ";\n";
                params += ", " + declare(v.pointee, "si_" + v.name) + " : SV_PrimitiveID";
                copy_in += "    " + v.name + " = si_" + v.name + ";\n";
                continue;
            }
            if (v.input && v.location >= 0) {
                // An input of each control point (the LS's): the input patch.
                const Type& at = type(v.pointee);
                if (at.k != TK::Array || at.count != output_vertices) fail("a hull input that is not an array of the patch's control points");
                s += "static " + declare(v.pointee, v.name) + ";\n";
                cp_in += "    [[vk::location(" + std::to_string(v.location) + ")]] " + v.interp + declare(at.elem, v.name) + " : CP" +
                         std::to_string(v.location) + ";\n";
                for (std::uint32_t k = 0; k < at.count; ++k)
                    copy_in += "    " + v.name + "[" + std::to_string(k) + "] = ip[" + std::to_string(k) + "]." + v.name + ";\n";
                continue;
            }
            if (!v.input && v.id == cp_out_var) {
                const Type& at = type(v.pointee);
                if (at.k != TK::Array || at.count != output_vertices) fail("a hull output that is not an array of the patch's control points");
                s += "static " + declare(v.pointee, v.name) + ";  // the output patch, after the barrier\n";
                s += "static " + declare(at.elem, v.name + "_mine") + ";  // this invocation's control point\n";
                cp_out += "    [[vk::location(" + std::to_string(v.location) + ")]] " + declare(at.elem, v.name) + " : CPO" + std::to_string(v.location) + ";\n";
                copy_out_cp += "    o." + v.name + " = " + v.name + "_mine;\n";
                for (std::uint32_t k = 0; k < at.count; ++k)
                    copy_pc += "    " + v.name + "[" + std::to_string(k) + "] = op[" + std::to_string(k) + "]." + v.name + ";\n";
                continue;
            }
            if (!v.input && v.patch && v.location >= 0) {
                s += "static " + declare(v.pointee, v.name) + ";\n";
                pc += "    [[vk::location(" + std::to_string(v.location) + ")]] " + declare(v.pointee, v.name) + " : PC" + std::to_string(v.location) + ";\n";
                continue;
            }
            if (!v.input && (v.builtin == spv::BiTessLevelOuter || v.builtin == spv::BiTessLevelInner)) {
                s += "static " + declare(v.pointee, v.name) + ";\n";
                continue;
            }
            fail("a hull shader variable HLSL has no place for");
        }
        // The factors HLSL declares for the domain, from the lift's four outer and two inner levels.
        std::string factor_copy;
        std::string outer, inner;
        for (const StageVar& v : stage_vars) {
            if (v.builtin == spv::BiTessLevelOuter) outer = v.name;
            if (v.builtin == spv::BiTessLevelInner) inner = v.name;
        }
        if (outer.empty() || inner.empty()) fail("a hull shader that writes no tessellation levels");
        const std::uint32_t n_outer = quads ? 4 : 3;
        for (std::uint32_t k = 0; k < n_outer; ++k) factor_copy += "    pc.tess_outer[" + std::to_string(k) + "] = " + outer + "[" + std::to_string(k) + "];\n";
        if (quads) factor_copy += "    pc.tess_inner[0] = " + inner + "[0];\n    pc.tess_inner[1] = " + inner + "[1];\n";
        else factor_copy += "    pc.tess_inner = " + inner + "[0];\n";
        for (const StageVar& v : stage_vars) {
            if (!v.input && v.patch && v.location >= 0) factor_copy += "    pc." + v.name + " = " + v.name + ";\n";
        }
        s += "static bool hull_phase_cp;  // the control-point phase: the body returns at the barrier\n\n";
        s += "struct ControlPointIn {\n" + cp_in + "};\n";
        s += "struct ControlPointOut {\n" + cp_out + "};\n";
        s += "struct PatchConstants {\n";
        s += quads ? "    float tess_outer[4] : SV_TessFactor;\n    float tess_inner[2] : SV_InsideTessFactor;\n"
                   : "    float tess_outer[3] : SV_TessFactor;\n    float tess_inner : SV_InsideTessFactor;\n";
        s += pc + "};\n";
        std::string invocation;
        for (const StageVar& v : stage_vars) {
            if (v.builtin == spv::BiInvocationId) invocation = v.name;
        }
        s += "\n" + helper_functions() + "void lifted_main() {\n" + body_text + "}\n\n";
        s += "// The patch's constants and factors: the body again as invocation 0, with every control point's output.\n";
        s += "PatchConstants patch_main(InputPatch<ControlPointIn, " + n + "> ip, OutputPatch<ControlPointOut, " + n + "> op" + params + ") {\n";
        s += copy_in + copy_pc;
        s += "    " + invocation + " = 0u;\n    hull_phase_cp = false;\n    lifted_main();\n    PatchConstants pc;\n" + factor_copy + "    return pc;\n}\n\n";
        static const char* const partitioning[] = {"integer", "fractional_odd", "fractional_even"};
        s += float_mode_attributes();
        s += std::string("[domain(\"") + (quads ? "quad" : "tri") + "\")]\n";
        s += std::string("[partitioning(\"") + partitioning[std::clamp(opt.tess_spacing, 0, 2)] + "\")]\n";
        s += std::string("[outputtopology(\"") + (opt.tess_cw ? "triangle_cw" : "triangle_ccw") + "\")]\n";
        s += "[outputcontrolpoints(" + n + ")]\n[patchconstantfunc(\"patch_main\")]\n";
        s += "ControlPointOut main(InputPatch<ControlPointIn, " + n + "> ip, uint id : SV_OutputControlPointID" + params + ") {\n";
        s += float_mode_calls();
        s += copy_in + "    " + invocation + " = id;\n    hull_phase_cp = true;\n    lifted_main();\n    ControlPointOut o;\n" + copy_out_cp + "    return o;\n}\n";
        if (uses_half) arguments.push_back("-enable-16bit-types");
        return s;
    }

    // A function's phis and the values a use outside their scope reaches, declared at its top.
    std::string declarations(const Function& f) {
        std::string top;
        for (const Ins& in : f.ins) {
            if (!in.id || !in.type || in.op == spv::OpFunctionParameter) continue;
            const bool phi = in.op == spv::OpPhi;
            if (!phi && !hoisted.count(in.id)) continue;
            if (parts.count(in.id)) {
                const std::string tn = type_name(type(in.type).members.at(0));
                top += "    " + tn + " " + parts.at(in.id).first + ", " + parts.at(in.id).second + ";\n";
                continue;
            }
            const char k = kind_of(in.type);
            top += std::string("    ") + (k == 'f' || k == 'h' ? "precise " : "") + type_name(in.type) + " " + value_name(in.id) + ";\n";
        }
        return top;
    }
    // The helper functions, prototypes first (one may call another declared after it).
    std::string helper_functions() {
        std::string protos, bodies;
        for (Function& f : functions) {
            if (f.id == entry_fn) continue;
            const std::string ret = type(f.result_type).k == TK::Void ? std::string("void") : type_name(f.result_type);
            std::string sig = ret + " " + f.name + "(";
            for (std::size_t k = 0; k < f.params.size(); ++k) {
                sig += (k ? ", " : "") + type_name(defs.at(f.params[k])->type) + " " + value_name(f.params[k]);
            }
            sig += ")";
            protos += sig + ";\n";
            bodies += sig + " {\n" + f.text + "}\n\n";
        }
        return protos.empty() ? std::string() : protos + "\n" + bodies;
    }

    // A compute shader: [numthreads], its built-ins as the entry point's parameters.
    std::string assemble_compute() {
        if (!local_size[0]) fail("a compute shader without LocalSize");
        std::string s, params, copy;
        for (const StageVar& v : stage_vars) {
            if (!v.input || v.builtin < 0) fail("a compute shader stage variable that is no built-in input");
            const char* sem = nullptr;
            switch (v.builtin) {
            case spv::BiLocalInvocationId: sem = "SV_GroupThreadID"; break;
            case spv::BiWorkgroupId: sem = "SV_GroupID"; break;
            case 28: sem = "SV_DispatchThreadID"; break;
            case spv::BiLocalInvocationIndex: sem = "SV_GroupIndex"; break;
            default: fail("compute built-in " + std::to_string(v.builtin));
            }
            s += "static " + declare(v.pointee, v.name) + ";\n";
            params += std::string(params.empty() ? "" : ", ") + declare(v.pointee, "si_" + v.name) + " : " + sem;
            copy += "    " + v.name + " = si_" + v.name + ";\n";
        }
        s += "\n" + helper_functions() + "void lifted_main() {\n" + body_text + "}\n\n";
        s += float_mode_attributes();
        s += "[numthreads(" + std::to_string(local_size[0]) + ", " + std::to_string(local_size[1]) + ", " + std::to_string(local_size[2]) + ")]\n";
        s += "void main(" + params + ") {\n" + float_mode_calls() + copy + "    lifted_main();\n}\n";
        if (uses_half) arguments.push_back("-enable-16bit-types");
        return s;
    }

    std::string assemble() {
        for (const Function& f : functions) {
            if (f.id == entry_fn) body_text = f.text;  // lifted_main's, declarations included
        }
        std::string s;
        s += "// " + (opt.title.empty() ? std::string("lifted shader") : opt.title) + "\n";
        s += "// HLSL printed from the lifted SPIR-V by bbhost (gcn/hlsl.h), on this machine, from this game's own shader.\n";
        s += "// Values are named by their SPIR-V ids (v123); the module's names are kept for resources and stage variables.\n\n";
        for (const std::string& d : helper_decls) s += d + "\n";
        if (!helper_decls.empty()) s += "\n";
        for (const std::string& d : struct_decls) s += d + "\n";
        for (const std::string& d : resource_decls) s += d + "\n";
        if (!resource_decls.empty()) s += "\n";
        for (const std::string& d : spec_decls) s += d + "\n";
        if (!spec_decls.empty()) s += "\n";

        if (exec_model == spv::EmTessellationControl) return s + assemble_hull();
        if (exec_model == spv::EmGLCompute) return s + assemble_compute();

        // Stage variables: statics, filled from and written to the entry point's.
        const bool tes = exec_model == spv::EmTessellationEvaluation;
        std::string in_struct, out_struct, copy_in, copy_out, cp_struct, pc_struct, params;
        std::uint32_t patch_size = 0;
        for (const StageVar& v : stage_vars) {
            if (v.builtin == spv::BiHelperInvocation || v.builtin == spv::BiTessCoord) continue;  // built-in inputs of their own (helper_decls)
            s += "static " + declare(v.pointee, v.name) + ";\n";
            if (tes && v.input && v.location >= 0 && !v.patch) {
                // An input of each control point: the patch's array of it, one member of a control point.
                const Type& at = type(v.pointee);
                if (at.k != TK::Array) fail("a domain shader input that is not an array of control points");
                if (patch_size && patch_size != at.count) fail("control-point inputs of different patch sizes");
                patch_size = at.count;
                cp_struct += "    [[vk::location(" + std::to_string(v.location) + ")]] " + v.interp + declare(at.elem, v.name) + " : CP" +
                             std::to_string(v.location) + ";\n";
                for (std::uint32_t k = 0; k < at.count; ++k)
                    copy_in += "    " + v.name + "[" + std::to_string(k) + "] = patch[" + std::to_string(k) + "]." + v.name + ";\n";
                continue;
            }
            if (tes && v.input && v.patch && v.location >= 0) {
                pc_struct += "    [[vk::location(" + std::to_string(v.location) + ")]] " + declare(v.pointee, v.name) + " : PC" + std::to_string(v.location) + ";\n";
                copy_in += "    " + v.name + " = pc." + v.name + ";\n";
                continue;
            }
            if (tes && v.input && (v.builtin == spv::BiTessLevelOuter || v.builtin == spv::BiTessLevelInner)) {
                copy_in += "    " + v.name + " = pc." + (v.builtin == spv::BiTessLevelOuter ? "tess_outer" : "tess_inner") + ";\n";
                continue;
            }
            if (tes && v.input && v.builtin == spv::BiPrimitiveId) {
                params += ", " + declare(v.pointee, "si_" + v.name) + " : SV_PrimitiveID";
                copy_in += "    " + v.name + " = si_" + v.name + ";\n";
                continue;
            }
            if (v.patch) fail("a patch variable");
            std::string sem, attr;
            if (v.location >= 0) {
                attr = "[[vk::location(" + std::to_string(v.location) + ")]] ";
                if (!v.input && exec_model == spv::EmFragment) sem = "SV_Target" + std::to_string(v.location);
                else sem = std::string(v.input ? "IN" : "OUT") + std::to_string(v.location);
            } else {
                switch (v.builtin) {
                case spv::BiPosition: sem = "SV_Position"; break;
                case 1: attr = "[[vk::builtin(\"PointSize\")]] "; sem = "PSIZE"; break;
                case spv::BiFragCoord: sem = "SV_Position"; break;
                case spv::BiFrontFacing: sem = "SV_IsFrontFace"; break;
                case spv::BiFragDepth: sem = "SV_Depth"; break;
                case spv::BiVertexIndex: sem = "SV_VertexID"; break;
                case spv::BiInstanceIndex: sem = "SV_InstanceID"; break;
                case 3: break;  // below
                default: fail("built-in " + std::to_string(v.builtin));
                }
            }
            std::string& st = v.input ? in_struct : out_struct;
            std::string& cp = v.input ? copy_in : copy_out;
            // Invariant (BBHOST_VS_INVARIANT): DXC takes the decoration on the member.
            if (v.invariant) {
                if (v.builtin == 3) fail("invariant clip distances");
                attr = "[[vk::ext_decorate(18 /* Invariant */)]] " + attr;
            }
            if (v.builtin == 3) {
                // ClipDistance: float[n] as SV_ClipDistance0 (and 1 past four).
                const Type& at = type(v.pointee);
                if (at.k != TK::Array || at.count > 8 || kind_of(at.elem) != 'f') fail("clip distances that are not float[1..8]");
                for (std::uint32_t g = 0; g * 4 < at.count; ++g) {
                    const std::uint32_t n = std::min<std::uint32_t>(4, at.count - g * 4);
                    const std::string mn = v.name + std::to_string(g);
                    st += "    " + std::string(n == 1 ? "float" : "float" + std::to_string(n)) + " " + mn + " : SV_ClipDistance" + std::to_string(g) + ";\n";
                    for (std::uint32_t k = 0; k < n; ++k) {
                        const std::string el = v.name + "[" + std::to_string(g * 4 + k) + "]";
                        const std::string mm = n == 1 ? mn : mn + "." + swizzle(k);
                        cp += "    " + (v.input ? el + " = si." + mm : "so." + mm + " = " + el) + ";\n";
                    }
                }
                continue;
            }
            if (type(v.pointee).k == TK::Array && (v.location < 0 || type(type(v.pointee).elem).k == TK::Array)) fail("an array stage variable");
            st += "    " + attr + v.interp + declare(v.pointee, v.name) + " : " + sem + ";\n";
            cp += "    " + (v.input ? v.name + " = si." + v.name : "so." + v.name + " = " + v.name) + ";\n";
        }
        s += "\n";
        if (!in_struct.empty()) {
            if (tes) fail("a domain shader input HLSL has no place for");
            s += "struct StageInput {\n" + in_struct + "};\n";
        }
        if (tes) {
            // The patch constants a domain shader takes, with the factors HLSL requires there.
            if (tess_domain.empty()) fail("a domain shader without a domain");
            const char* factors = tess_domain == "quad" ? "    float tess_outer[4] : SV_TessFactor;\n    float tess_inner[2] : SV_InsideTessFactor;\n"
                                  : tess_domain == "tri" ? "    float tess_outer[3] : SV_TessFactor;\n    float tess_inner : SV_InsideTessFactor;\n"
                                                         : "    float tess_outer[2] : SV_TessFactor;\n";
            s += "struct PatchConstants {\n" + std::string(factors) + pc_struct + "};\n";
            if (patch_size) s += "struct ControlPoint {\n" + cp_struct + "};\n";
        }
        if (!out_struct.empty()) s += "struct StageOutput {\n" + out_struct + "};\n";
        s += "\n" + helper_functions() + "void lifted_main() {\n" + body_text + "}\n\n";
        s += float_mode_attributes();
        if (early_tests) s += "[earlydepthstencil]\n";
        std::string args = in_struct.empty() ? "" : "StageInput si";
        if (tes) {
            s += "[domain(\"" + tess_domain + "\")]\n";
            args = "PatchConstants pc";
            if (patch_size) args += ", const OutputPatch<ControlPoint, " + std::to_string(patch_size) + "> patch";
            args += params;
        }
        s += std::string(out_struct.empty() ? "void" : "StageOutput") + " main(" + args + ") {\n";
        s += float_mode_calls();
        for (const std::uint32_t m : tess_modes) {
            static const std::map<std::uint32_t, const char*> mode_names = {
                {spv::ExSpacingEqual, "SpacingEqual"}, {spv::ExSpacingFractionalEven, "SpacingFractionalEven"},
                {spv::ExSpacingFractionalOdd, "SpacingFractionalOdd"}, {spv::ExVertexOrderCw, "VertexOrderCw"},
                {spv::ExVertexOrderCcw, "VertexOrderCcw"}, {spv::ExPointMode, "PointMode"}};
            s += "    vk::ext_execution_mode(" + std::to_string(m) + ");  // " + mode_names.at(m) + "\n";
        }
        s += copy_in;
        s += "    lifted_main();\n";
        if (!out_struct.empty()) {
            s += "    StageOutput so;\n" + copy_out + "    return so;\n";
        }
        s += "}\n";
        if (uses_half) arguments.push_back("-enable-16bit-types");
        return s;
    }
};

// ---- interfaces ---------------------------------------------------------------------------

struct Iface {
    std::uint32_t model = 0;
    std::set<std::vector<std::uint32_t>> modes;  // each execution mode with its literals (RoundingModeRTZ 16, LocalSize 64 1 1)
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::string> bindings;  // (set, binding) -> description
    std::map<std::string, std::string> io;                                  // "in loc 3" / "out builtin 0" -> description
    std::set<std::uint32_t> spec_ids;
};

Iface interface_of(const std::vector<std::uint32_t>& w) {
    Iface f;
    if (w.size() < 5) return f;
    std::map<std::uint32_t, std::vector<std::uint32_t>> ins;  // id -> its words (type and constant declarations)
    std::map<std::uint32_t, std::map<std::uint32_t, std::vector<std::uint32_t>>> dec;  // id -> decoration -> operands
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> member_offset;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> vars;  // id, pointer type
    std::set<std::uint32_t> iface_ids;
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || i + n > w.size()) break;
        const std::uint32_t* a = &w[i + 1];
        switch (op) {
        case spv::OpEntryPoint: {
            f.model = a[0];
            std::size_t k = 2;
            for (; k < n - 1; ++k) {
                const std::uint32_t x = a[k];
                if (!(x & 0xffu) || !(x & 0xff00u) || !(x & 0xff0000u) || !(x & 0xff000000u)) {
                    ++k;
                    break;
                }
            }
            for (; k < n - 1; ++k) iface_ids.insert(a[k]);
            break;
        }
        case spv::OpExecutionMode:
            if (n >= 3) f.modes.insert(std::vector<std::uint32_t>(a + 1, a + n - 1));
            break;
        case spv::OpDecorate: dec[a[0]][a[1]] = std::vector<std::uint32_t>(a + 2, a + n - 1); break;
        case spv::OpMemberDecorate:
            if (n >= 5 && a[2] == spv::DecOffset) member_offset[{a[0], a[1]}] = a[3];
            break;
        case spv::OpVariable: vars.push_back({a[1], a[0]}); break;
        default:
            if ((op >= spv::OpTypeVoid && op <= 38) || op == spv::OpConstant || op == spv::OpSpecConstant) ins[op == spv::OpConstant || op == spv::OpSpecConstant ? a[1] : a[0]] = std::vector<std::uint32_t>(&w[i], &w[i] + n);
            if (op == spv::OpSpecConstant || op == 48 || op == 49) {
                // spec id recorded below from the decoration
            }
            break;
        }
        i += n;
    }
    for (const auto& [id, d] : dec) {
        const auto s = d.find(spv::DecSpecId);
        if (s != d.end() && !s->second.empty()) f.spec_ids.insert(s->second[0]);
    }
    // A type, described without its ids.
    std::function<std::string(std::uint32_t)> desc = [&](std::uint32_t t) -> std::string {
        const auto it = ins.find(t);
        if (it == ins.end()) return "?";
        const std::vector<std::uint32_t>& x = it->second;
        const std::uint32_t op = x[0] & 0xffff;
        switch (op) {
        case spv::OpTypeVoid: return "void";
        case spv::OpTypeBool: return "bool";
        case spv::OpTypeInt: return std::string(x[3] ? "i" : "u") + std::to_string(x[2]);
        case spv::OpTypeFloat: return "f" + std::to_string(x[2]);
        case spv::OpTypeVector: return desc(x[2]) + "x" + std::to_string(x[3]);
        case spv::OpTypeArray: {
            const auto c = ins.find(x[3]);
            return desc(x[2]) + "[" + (c != ins.end() ? std::to_string(c->second[3]) : std::string("?")) + "]";
        }
        case spv::OpTypeRuntimeArray: return desc(x[2]) + "[]";
        case spv::OpTypeStruct: {
            std::string s = "{";
            for (std::size_t k = 2; k < x.size(); ++k) {
                const auto o = member_offset.find({t, static_cast<std::uint32_t>(k - 2)});
                s += (k > 2 ? "," : "") + desc(x[k]) + (o != member_offset.end() ? "@" + std::to_string(o->second) : std::string());
            }
            return s + "}";
        }
        case spv::OpTypeImage:
            // Depth is left out: Vulkan ignores it, DXC writes 2 (unknown).
            return "image(" + desc(x[2]) + ",dim" + std::to_string(x[3]) + ",arr" + std::to_string(x[5]) + ",ms" + std::to_string(x[6]) + ",s" +
                   std::to_string(x[7]) + ",fmt" + std::to_string(x[8]) + ")";
        case spv::OpTypeSampler: return "sampler";
        case spv::OpTypeSampledImage: return "sampled(" + desc(x[2]) + ")";
        case spv::OpTypePointer: return desc(x[3]);
        default: return "op" + std::to_string(op);
        }
    };
    for (const auto& [id, ptr] : vars) {
        const auto pt = ins.find(ptr);
        if (pt == ins.end()) continue;
        const std::uint32_t sc = pt->second[2];
        const auto d = dec.find(id);
        const auto get = [&](std::uint32_t k) -> const std::vector<std::uint32_t>* {
            if (d == dec.end()) return nullptr;
            const auto e = d->second.find(k);
            return e == d->second.end() ? nullptr : &e->second;
        };
        if (sc == spv::ScUniform || sc == spv::ScStorageBuffer || sc == spv::ScUniformConstant) {
            const auto* set = get(spv::DecDescriptorSet);
            const auto* binding = get(spv::DecBinding);
            if (!set || !binding || set->empty() || binding->empty()) continue;
            std::string what = std::string(sc == spv::ScUniform ? "uniform " : sc == spv::ScStorageBuffer ? "storage " : "") + desc(pt->second[3]);
            // A block's struct: only its layout matters, not that it is a block of one uint[] or a ByteAddressBuffer.
            auto& slot = f.bindings[{(*set)[0], (*binding)[0]}];
            if (slot.empty()) slot = what;
            else if (slot != what) slot += " | " + what;  // aliases (SamplerState and SamplerComparisonState)
            continue;
        }
        if (sc != spv::ScInput && sc != spv::ScOutput) continue;
        if (!iface_ids.count(id)) continue;
        std::string key = sc == spv::ScInput ? "in " : "out ";
        if (const auto* bi = get(spv::DecBuiltIn)) key += "builtin " + std::to_string((*bi)[0]);
        else if (const auto* loc = get(spv::DecLocation)) key += "location " + std::to_string((*loc)[0]);
        else key += "?";
        std::string what = desc(pt->second[3]);
        // Flat on an integer or built-in input is DXC's own requirement; the lift's
        // vertex inputs and built-ins carry none. Interpolation matters for a
        // pixel shader's float inputs.
        if (get(spv::DecNoPerspective)) what += " noperspective";
        if (get(spv::DecCentroid)) what += " centroid";
        if (get(spv::DecFlat) && !get(spv::DecBuiltIn) && what.rfind("f", 0) == 0) what += " flat";
        if (get(spv::DecInvariant)) what += " invariant";
        if (get(spv::DecPatch)) what += " patch";
        f.io[key] = what;
    }
    return f;
}

struct OpCensus {
    std::map<std::string, std::size_t> ops;  // "op 129" / "glsl 79" -> count
    std::size_t contractible = 0;              // float arithmetic without NoContraction
    std::size_t fused_fma = 0;                 // Fma decorated NoContraction (a driver must fuse it)
    // Constants of 16-bit floats (scalars, vectors, composites of them): a
    // conversion to half that DXC evaluated itself - rounding to nearest,
    // where a lift for a device that rounds halves itself says toward zero -
    // leaves one where the lift has none.
    std::size_t half_constants = 0;
};

OpCensus op_census(const std::vector<std::uint32_t>& w) {
    // Counted per function, a call counting its callee's operations again
    // (DXC inlines every function: a compute lift's page-table walk is in
    // its module once, in DXC's once a call).
    struct Fn {
        std::map<std::string, std::size_t> ops;
        std::vector<std::uint32_t> calls;
    };
    std::map<std::uint32_t, Fn> fns;
    std::uint32_t entry = 0, current = 0;
    OpCensus c;
    std::set<std::uint32_t> nocontract, consts, quad_results, half_types;
    std::vector<std::uint32_t> float_arith, fma_ids;
    std::uint32_t glsl_set = 0;
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || i + n > w.size()) break;
        const std::uint32_t* a = &w[i + 1];
        if (op == spv::OpEntryPoint) entry = a[1];
        if (op == spv::OpTypeFloat && n >= 3 && a[1] == 16) half_types.insert(a[0]);
        if ((op == spv::OpTypeVector || op == spv::OpTypeArray) && n >= 3 && half_types.count(a[1])) half_types.insert(a[0]);
        if ((op == spv::OpConstant || op == spv::OpConstantComposite || op == spv::OpConstantNull || op == spv::OpSpecConstant ||
             op == kOpSpecConstantComposite || op == kOpSpecConstantOp) &&
            n >= 3 && half_types.count(a[0])) {
            ++c.half_constants;
        }
        if (op == spv::OpFunction) current = a[1];
        if (op == spv::OpFunctionCall && current) fns[current].calls.push_back(a[2]);
        auto& ops = fns[current].ops;
        if (op == spv::OpExtInstImport) glsl_set = a[0];
        if (op == spv::OpGroupNonUniformQuadSwap || op == spv::OpGroupNonUniformQuadBroadcast) quad_results.insert(a[1]);
        if (op == spv::OpConstant || op == spv::OpConstantComposite || op == spv::OpConstantNull || op == spv::OpSpecConstant) consts.insert(a[1]);
        if (op == spv::OpDecorate && n >= 3 && a[1] == spv::DecNoContraction) nocontract.insert(a[0]);
        // (Conversions between float and integer and between float widths: not
        // UConvert/SConvert, which are exact whatever spells them.)
        const bool semantic = (op >= spv::OpConvertFToU && op <= spv::OpConvertUToF) || op == spv::OpFConvert || op == 116 /* QuantizeToF16 */ ||
                              (op >= spv::OpFNegate && op <= 141 && (op == spv::OpFNegate || op == spv::OpFAdd || op == spv::OpFSub || op == spv::OpFMul ||
                                                                    op == spv::OpFDiv || op == 140 || op == 141)) ||
                              op == spv::OpIsNan || op == spv::OpIsInf || (op >= spv::OpFOrdEqual && op <= spv::OpFUnordGreaterThanEqual) ||
                              (op >= 207 && op <= 215) || (op >= spv::OpImageSampleImplicitLod && op <= spv::OpImageQueryLevels) ||
                              (op >= 333 && op <= 366) || op == spv::OpKill || op == spv::OpDemoteToHelperInvocation || op == kOpIsHelperInvocation ||
                              op == kOpTerminateInvocation || (op >= spv::OpAtomicLoad && op <= 242) || op == spv::OpControlBarrier;
        if (semantic) {
            if (op >= spv::OpFOrdEqual && op <= spv::OpFUnordGreaterThanEqual) {
                // A comparison and the negation of its complement are the same
                // (with NaNs too): spirv-opt folds !(a < b) into FUnordGreaterThanEqual.
                // A quad read of a bool is one of a float DXC makes of it (1.0 or
                // 0.0), compared with 0 again: part of the quad operation.
                if (!(op == spv::OpFOrdNotEqual && (quad_results.count(a[2]) || quad_results.count(a[3])))) ops["float comparisons"]++;
            } else if (op == spv::OpFNegate && consts.count(a[2])) {
                // -0.5f in a precise expression: an exact negation DXC does not fold.
            } else {
                ops["op " + std::to_string(op)]++;
            }
        }
        if (op == spv::OpExtInst && n >= 5 && a[2] == glsl_set) {
            ops["glsl " + std::to_string(a[3])]++;
            if (a[3] == spv::GlslFma) fma_ids.push_back(a[1]);
        }
        if (op == spv::OpFAdd || op == spv::OpFSub || op == spv::OpFMul || op == spv::OpFDiv || (op == spv::OpFNegate && !consts.count(a[2]))) {
            // Float scalars and vectors only reach these opcodes. A negated
            // literal (-1.0f) is exact whether or not it may contract.
            float_arith.push_back(a[1]);
        }
        i += n;
    }
    for (const std::uint32_t id : float_arith) c.contractible += nocontract.count(id) ? 0 : 1;
    for (const std::uint32_t id : fma_ids) c.fused_fma += nocontract.count(id) ? 1 : 0;
    // The entry point's counts with every call's callee counted at the call (depth-limited).
    std::function<void(std::uint32_t, int)> add = [&](std::uint32_t f, int depth) {
        if (depth > 8 || !fns.count(f)) return;
        for (const auto& [k, n] : fns.at(f).ops) c.ops[k] += n;
        for (const std::uint32_t callee : fns.at(f).calls) add(callee, depth + 1);
    };
    add(entry, 0);
    return c;
}

}  // namespace

std::string hlsl_operation_diff(const std::vector<std::uint32_t>& lifted, const std::vector<std::uint32_t>& compiled) {
    const OpCensus a = op_census(lifted), b = op_census(compiled);
    // A hull runs its body in both of HLSL's phases (hlsl.h): each operation up to twice.
    const std::size_t copies = interface_of(lifted).model == spv::EmTessellationControl ? 2 : 1;
    for (const auto& [k, n] : b.ops) {
        const auto it = a.ops.find(k);
        const std::size_t have = (it == a.ops.end() ? 0 : it->second) * copies;
        if (n <= have) continue;
        // GetDimensions(mip, ...) asks for the level count too; CalculateLevelOfDetail
        // and its unclamped twin are one OpImageQueryLod each before DXC merges them.
        if (k == "op " + std::to_string(spv::OpImageQueryLevels) && a.ops.count("op " + std::to_string(spv::OpImageQuerySizeLod))) continue;
        if (k == "op " + std::to_string(spv::OpImageQuerySizeLod) && a.ops.count("op " + std::to_string(spv::OpImageQueryLevels))) continue;
        if (k == "op " + std::to_string(spv::OpImageQueryLod) && n <= 2 * have) continue;
        return k + ": " + std::to_string(n) + " where the lift has " + std::to_string(have);
    }
    if (b.half_constants > a.half_constants)
        return std::to_string(b.half_constants) + " half-precision constants where the lift has " + std::to_string(a.half_constants) + " (a conversion DXC evaluated)";
    if (b.contractible) return std::to_string(b.contractible) + " float operations without NoContraction";
    // Fma the lift leaves to the driver (fused or not) must not be one DXC's module requires fused.
    if (b.fused_fma > a.fused_fma) return std::to_string(b.fused_fma) + " Fma decorated NoContraction where the lift has " + std::to_string(a.fused_fma);
    return {};
}

HlslResult emit_hlsl(const std::vector<std::uint32_t>& spirv, const HlslOptions& options) { return Printer(spirv, options).run(); }

std::string hlsl_interface_diff(const std::vector<std::uint32_t>& lifted, const std::vector<std::uint32_t>& compiled) {
    const Iface a = interface_of(lifted), b = interface_of(compiled);
    if (a.model != b.model) return "execution model " + std::to_string(a.model) + " against " + std::to_string(b.model);
    const auto mode_text = [](const std::vector<std::uint32_t>& m) {
        std::string s = "execution mode";
        for (const std::uint32_t x : m) s += " " + std::to_string(x);
        return s;
    };
    for (const std::vector<std::uint32_t>& m : a.modes) {
        if (!b.modes.count(m)) return mode_text(m) + " missing";
    }
    for (const std::vector<std::uint32_t>& m : b.modes) {
        // HLSL declares a hull's domain, partitioning and winding (the
        // evaluation stage declares the same; HlslOptions passes them).
        const std::uint32_t k = m.at(0);
        const bool hull_tess_mode = b.model == spv::EmTessellationControl && m.size() == 1 &&
                                    (k == spv::ExTriangles || k == spv::ExQuads || k == spv::ExIsolines || k == spv::ExSpacingEqual ||
                                     k == spv::ExSpacingFractionalEven || k == spv::ExSpacingFractionalOdd || k == spv::ExVertexOrderCw ||
                                     k == spv::ExVertexOrderCcw);
        if (!a.modes.count(m) && !hull_tess_mode) return mode_text(m) + " added";
    }
    for (const auto& [k, v] : a.bindings) {
        const auto it = b.bindings.find(k);
        if (it == b.bindings.end()) return "binding " + std::to_string(k.first) + "." + std::to_string(k.second) + " (" + v + ") missing";
        if (it->second != v) {
            // A sampler aliased for depth compares is the same descriptor.
            if (v == "sampler" && it->second == "sampler | sampler") continue;
            return "binding " + std::to_string(k.first) + "." + std::to_string(k.second) + ": " + v + " against " + it->second;
        }
    }
    for (const auto& [k, v] : b.bindings) {
        if (!a.bindings.count(k)) return "binding " + std::to_string(k.first) + "." + std::to_string(k.second) + " (" + v + ") added";
    }
    for (const auto& [k, v] : a.io) {
        const auto it = b.io.find(k);
        if (it == b.io.end()) return k + " (" + v + ") missing";
        if (it->second != v) return k + ": " + v + " against " + it->second;
    }
    for (const auto& [k, v] : b.io) {
        // HLSL's domain shader declares its patch's factors whether it reads them or not.
        if (b.model == spv::EmTessellationEvaluation && (k == "in builtin 11" || k == "in builtin 12")) continue;
        if (!a.io.count(k)) return k + " (" + v + ") added";
    }
    if (a.spec_ids != b.spec_ids) return "specialization constant ids differ";
    return {};
}

}  // namespace gcn
