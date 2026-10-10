// Minimal SPIR-V module builder: enough of the binary form to emit the
// translated shaders (types, constants, variables, one function with
// arbitrary blocks). Opcode numbers are from the SPIR-V 1.5 specification.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace spv {

using Id = std::uint32_t;

enum Op : std::uint16_t {
    OpNop = 0, OpUndef = 1, OpName = 5, OpExtension = 10, OpExtInstImport = 11, OpExtInst = 12,
    OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
    OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27, OpTypeArray = 28,
    OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32, OpTypeFunction = 33,
    OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44,
    OpConstantNull = 46, OpSpecConstant = 50, OpFunction = 54, OpFunctionParameter = 55, OpFunctionEnd = 56,
    OpFunctionCall = 57, OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
    OpDecorate = 71, OpMemberDecorate = 72, OpVectorShuffle = 79, OpCompositeConstruct = 80,
    OpCompositeExtract = 81, OpCompositeInsert = 82, OpSampledImage = 86,
    OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88, OpImageSampleDrefImplicitLod = 89,
    OpImageSampleDrefExplicitLod = 90, OpImageFetch = 95, OpImageGather = 96, OpImageDrefGather = 97,
    OpImageRead = 98, OpImageWrite = 99, OpImageQuerySizeLod = 103, OpImageQuerySize = 104,
    OpImageQueryLod = 105, OpImageQueryLevels = 106, OpConvertFToU = 109, OpConvertFToS = 110,
    OpConvertSToF = 111, OpConvertUToF = 112, OpUConvert = 113, OpSConvert = 114, OpFConvert = 115,
    OpConvertUToPtr = 120, OpBitcast = 124, OpSNegate = 126, OpFNegate = 127, OpIAdd = 128, OpFAdd = 129,
    OpISub = 130, OpFSub = 131, OpIMul = 132, OpFMul = 133, OpUDiv = 134, OpSDiv = 135, OpFDiv = 136,
    OpUMod = 137, OpSRem = 138, OpSMod = 139, OpIAddCarry = 149, OpISubBorrow = 150, OpUMulExtended = 151,
    OpSMulExtended = 152, OpAny = 154, OpAll = 155, OpIsNan = 156, OpIsInf = 157, OpLogicalEqual = 164,
    OpLogicalNotEqual = 165, OpLogicalOr = 166, OpLogicalAnd = 167, OpLogicalNot = 168, OpSelect = 169,
    OpIEqual = 170, OpINotEqual = 171, OpUGreaterThan = 172, OpSGreaterThan = 173, OpUGreaterThanEqual = 174,
    OpSGreaterThanEqual = 175, OpULessThan = 176, OpSLessThan = 177, OpULessThanEqual = 178,
    OpSLessThanEqual = 179, OpFOrdEqual = 180, OpFUnordEqual = 181, OpFOrdNotEqual = 182,
    OpFUnordNotEqual = 183, OpFOrdLessThan = 184, OpFUnordLessThan = 185, OpFOrdGreaterThan = 186,
    OpFUnordGreaterThan = 187, OpFOrdLessThanEqual = 188, OpFUnordLessThanEqual = 189,
    OpFOrdGreaterThanEqual = 190, OpFUnordGreaterThanEqual = 191, OpShiftRightLogical = 194,
    OpShiftRightArithmetic = 195, OpShiftLeftLogical = 196, OpBitwiseOr = 197, OpBitwiseXor = 198,
    OpBitwiseAnd = 199, OpNot = 200, OpBitFieldInsert = 201, OpBitFieldSExtract = 202,
    OpBitFieldUExtract = 203, OpBitReverse = 204, OpBitCount = 205, OpDPdx = 207, OpDPdy = 208,
    OpControlBarrier = 224, OpMemoryBarrier = 225, OpAtomicLoad = 227, OpAtomicStore = 228,
    OpAtomicExchange = 229, OpAtomicCompareExchange = 230, OpAtomicIIncrement = 232,
    OpAtomicIDecrement = 233, OpAtomicIAdd = 234, OpAtomicISub = 235, OpAtomicSMin = 236,
    OpAtomicUMin = 237, OpAtomicSMax = 238, OpAtomicUMax = 239, OpAtomicAnd = 240, OpAtomicOr = 241,
    OpAtomicXor = 242, OpPhi = 245, OpLoopMerge = 246, OpSelectionMerge = 247, OpLabel = 248,
    OpBranch = 249, OpBranchConditional = 250, OpSwitch = 251, OpKill = 252, OpReturn = 253, OpReturnValue = 254,
    OpUnreachable = 255, OpGroupNonUniformElect = 333, OpGroupNonUniformAll = 334,
    OpGroupNonUniformAny = 335, OpGroupNonUniformBroadcast = 337, OpGroupNonUniformBroadcastFirst = 338,
    OpGroupNonUniformBallot = 339, OpGroupNonUniformBallotBitCount = 342, OpGroupNonUniformShuffle = 345,
    OpGroupNonUniformShuffleXor = 346, OpDemoteToHelperInvocation = 5380,
    OpGroupNonUniformQuadBroadcast = 365, OpGroupNonUniformQuadSwap = 366,
};

enum Glsl : std::uint32_t {
    GlslRound = 1, GlslRoundEven = 2, GlslTrunc = 3, GlslFAbs = 4, GlslSAbs = 5, GlslFSign = 6, GlslFloor = 8,
    GlslCeil = 9, GlslFract = 10, GlslSin = 13, GlslCos = 14, GlslPow = 26, GlslExp = 27, GlslLog = 28,
    GlslExp2 = 29, GlslLog2 = 30, GlslSqrt = 31, GlslInverseSqrt = 32, GlslFMin = 37, GlslUMin = 38,
    GlslSMin = 39, GlslFMax = 40, GlslUMax = 41, GlslSMax = 42, GlslFClamp = 43, GlslUClamp = 44,
    GlslSClamp = 45, GlslFMix = 46, GlslFma = 50, GlslFrexp = 51, GlslLdexp = 53, GlslPackHalf2x16 = 58,
    GlslUnpackHalf2x16 = 62, GlslFindILsb = 73, GlslFindSMsb = 74, GlslFindUMsb = 75, GlslNMin = 79,
    GlslNMax = 80, GlslNClamp = 81,
};

enum Capability : std::uint32_t {
    CapShader = 1, CapFloat16 = 9, CapImageGatherExtended = 25, CapClipDistance = 32, CapCullDistance = 33, CapInt64 = 11, CapImageQuery = 50, CapDerivativeControl = 51,
    CapStorageImageReadWithoutFormat = 55, CapStorageImageWriteWithoutFormat = 56, CapGroupNonUniform = 61,
    CapGroupNonUniformVote = 62, CapGroupNonUniformArithmetic = 63, CapGroupNonUniformBallot = 64,
    CapGroupNonUniformShuffle = 65, CapSampled1D = 43, CapImage1D = 44, CapSampledCubeArray = 45,
    CapImageCubeArray = 34, CapDemoteToHelperInvocation = 5379, CapPhysicalStorageBufferAddresses = 5347,
    CapTessellation = 3, CapSampledImageArrayDynamicIndexing = 29, CapStorageImageArrayDynamicIndexing = 31,
    CapRuntimeDescriptorArray = 5302, CapDenormPreserve = 4464, CapRoundingModeRTZ = 4468,
    CapGroupNonUniformQuad = 68,
};

enum StorageClass : std::uint32_t {
    ScUniformConstant = 0, ScInput = 1, ScUniform = 2, ScOutput = 3, ScWorkgroup = 4, ScPrivate = 6,
    ScFunction = 7, ScPushConstant = 9, ScStorageBuffer = 12, ScPhysicalStorageBuffer = 5349,
};

enum Decoration : std::uint32_t {
    DecSpecId = 1, DecBlock = 2, DecArrayStride = 6, DecBuiltIn = 11, DecNoPerspective = 13, DecFlat = 14, DecPatch = 15, DecCentroid = 16,
    DecInvariant = 18,
    DecNonWritable = 24, DecLocation = 30, DecBinding = 33, DecDescriptorSet = 34, DecOffset = 35,
    DecNoContraction = 42,
};

enum BuiltIn : std::uint32_t {
    BiPosition = 0, BiFragCoord = 15, BiFrontFacing = 17, BiSampleId = 18, BiSampleMask = 20, BiFragDepth = 22,
    BiWorkgroupId = 26, BiLocalInvocationId = 27, BiLocalInvocationIndex = 29, BiSubgroupSize = 36,
    BiSubgroupLocalInvocationId = 41, BiVertexIndex = 42, BiInstanceIndex = 43, BiHelperInvocation = 23,
    // Tessellation
    BiTessLevelOuter = 11, BiTessLevelInner = 12, BiTessCoord = 13, BiPatchVertices = 14,
    BiInvocationId = 8, BiPrimitiveId = 7,
};

enum ExecutionModel : std::uint32_t {
    EmVertex = 0, EmTessellationControl = 1, EmTessellationEvaluation = 2, EmFragment = 4, EmGLCompute = 5,
};
enum ExecutionMode : std::uint32_t {
    ExOriginUpperLeft = 7, ExEarlyFragmentTests = 9, ExDepthReplacing = 12, ExLocalSize = 17,
    // Tessellation: VGT_TF_PARAM's domain, partitioning and topology.
    ExSpacingEqual = 1, ExSpacingFractionalEven = 2, ExSpacingFractionalOdd = 3,
    ExVertexOrderCw = 4, ExVertexOrderCcw = 5, ExPointMode = 27,
    ExTriangles = 22, ExQuads = 24, ExIsolines = 25, ExOutputVertices = 26,
    // SPIR-V 1.4 float controls: operand the bit width they apply to.
    ExDenormPreserve = 4459, ExRoundingModeRTZ = 4463,
};
enum Scope : std::uint32_t { ScopeDevice = 1, ScopeWorkgroup = 2, ScopeSubgroup = 3, ScopeInvocation = 4 };
enum MemorySemantics : std::uint32_t {
    MsNone = 0, MsAcquire = 2, MsRelease = 4, MsAcquireRelease = 8, MsUniformMemory = 0x40,
    MsWorkgroupMemory = 0x100, MsImageMemory = 0x800,
};
enum Dim : std::uint32_t { Dim1D = 0, Dim2D = 1, Dim3D = 2, DimCube = 3, DimBuffer = 5 };
enum ImageOperands : std::uint32_t {
    IoNone = 0, IoBias = 1, IoLod = 2, IoGrad = 4, IoConstOffset = 8, IoOffset = 0x10, IoSample = 0x40,
};

class Module {
public:
    Module();

    Id fresh() { return next_id_++; }
    Id bound() const { return next_id_; }

    // Types (deduplicated).
    Id type_void();
    Id type_bool();
    Id type_int(std::uint32_t width, bool is_signed);
    Id type_float(std::uint32_t width);
    Id type_vector(Id component, std::uint32_t n);
    Id type_array(Id element, Id length_const);
    Id type_runtime_array(Id element);
    Id type_struct(const std::vector<Id>& members);
    Id type_pointer(StorageClass sc, Id pointee);
    Id type_function(Id ret, const std::vector<Id>& params);
    Id type_image(Id sampled, Dim dim, bool depth, bool arrayed, bool ms, std::uint32_t sampled_mode);
    Id type_sampler();
    Id type_sampled_image(Id image);

    // Constants (deduplicated).
    Id const_bool(bool v);
    Id const_u32(std::uint32_t v);
    Id const_i32(std::int32_t v);
    Id const_u64(std::uint64_t v);
    Id const_f32(float v);
    Id const_composite(Id type, const std::vector<Id>& members);
    Id const_null(Id type);
    // A 32-bit unsigned specialization constant (not deduplicated): `value`
    // unless the pipeline gives SpecId `spec_id` another.
    Id spec_const_u32(std::uint32_t value, std::uint32_t spec_id);

    // Module-level declarations.
    void capability(Capability c);
    void capability(std::uint32_t c) { capability(static_cast<Capability>(c)); }
    void extension(const std::string& name);
    Id ext_glsl();  // imports GLSL.std.450 once
    void memory_model(std::uint32_t addressing, std::uint32_t memory);
    void entry_point(ExecutionModel model, Id func, const std::string& name, const std::vector<Id>& interface);
    void execution_mode(Id func, ExecutionMode mode, const std::vector<std::uint32_t>& operands = {});
    void decorate(Id target, Decoration d, const std::vector<std::uint32_t>& operands = {});
    void member_decorate(Id type, std::uint32_t member, Decoration d, const std::vector<std::uint32_t>& operands = {});
    void name(Id target, const std::string& n);
    Id global_variable(Id ptr_type, StorageClass sc, Id initializer = 0);

    // Function body emission. Function-local variables are collected and
    // hoisted into the first block as SPIR-V requires.
    Id begin_function(Id ret_type, Id fn_type);
    void end_function();
    Id local_variable(Id ptr_type, Id initializer = 0);
    // While on, instructions go to the function's first block, after its
    // variables: loads of inputs and uniforms that every later block may use.
    void emit_to_entry(bool on) { to_entry_ = on; }
    bool emitting_to_entry() const { return to_entry_; }
    Id label(Id id = 0);  // starts a block; returns its label id
    std::uint32_t blocks() const { return blocks_; }  // blocks started so far: an id made since the last label is in the current block
    // Generic instruction with a result. Float arithmetic results are decorated
    // NoContraction: GCN computes a multiply and a later add separately, and a
    // driver that fuses them does so differently in differently structured
    // modules (the translator's dispatcher and a lifted shader differed by one
    // ulp in a few texels).
    Id emit(Op op, Id result_type, const std::vector<std::uint32_t>& operands);
    // Instruction without a result.
    void emit_void(Op op, const std::vector<std::uint32_t>& operands);
    // Replaces operand `index` of the function-body instruction that defined
    // `result`: a loop header's OpPhi gets its back-edge value once the loop
    // body is emitted. False when there is no such instruction or operand.
    bool set_operand(Id result, std::size_t index, std::uint32_t value);
    Id ext_inst(Id result_type, Glsl inst, const std::vector<Id>& operands);

    Id load(Id type, Id ptr) { return emit(OpLoad, type, {ptr}); }
    void store(Id ptr, Id value) { emit_void(OpStore, {ptr, value}); }
    Id access_chain(Id ptr_type, Id base, const std::vector<Id>& indices);

    std::vector<std::uint32_t> assemble() const;

private:
    struct Instr {
        Op op;
        Id type;
        Id result;
        std::vector<std::uint32_t> operands;
    };
    static std::vector<std::uint32_t> encode(const Instr& in);
    Id add_type(const std::vector<std::uint32_t>& key, Op op, const std::vector<std::uint32_t>& operands);

    Id next_id_ = 1;
    Id glsl_ = 0;
    std::uint32_t blocks_ = 0;
    std::vector<std::uint32_t> capabilities_;
    std::vector<std::string> extensions_;
    std::uint32_t addressing_ = 0, memory_ = 1;
    std::vector<Instr> entry_points_, execution_modes_, names_, decorations_, types_consts_globals_;
    std::vector<Instr> functions_;
    std::map<std::vector<std::uint32_t>, Id> type_cache_;
    std::map<std::tuple<Id, std::uint64_t, std::uint32_t>, Id> const_cache_;
    // Current function state.
    bool in_function_ = false;
    std::size_t function_start_ = 0;
    std::vector<Instr> local_vars_;
    std::vector<Instr> entry_;  // emit_to_entry
    bool to_entry_ = false;
    std::vector<Instr> body_;
};

}  // namespace spv
