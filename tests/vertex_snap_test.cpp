// PS1 vertex snapping (host/vertex_snap.h) on small hand-made modules: a
// vertex stage that writes its own BuiltIn Position variable (DXC's form) and
// one that writes gl_PerVertex's member through an access chain with two
// returns and no GLSL import (the GLSL compilers' form), a geometry stage's
// emits, and the stages it must leave alone. Each rewrite is valid SPIR-V,
// snaps before every return or emit, keeps its arithmetic NoContraction, and
// a refusal leaves the module as it was.
#include "gcn/spirv.h"
#include "host/vertex_snap.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

using u32 = std::uint32_t;

struct Asm {
    std::vector<u32> w{0x07230203u, 0x00010000u, 0, 0, 0};
    u32 next = 1;
    u32 id() { return next++; }
    void op(u32 opcode, std::initializer_list<u32> operands) {
        w.push_back(static_cast<u32>(operands.size() + 1) << 16 | opcode);
        w.insert(w.end(), operands.begin(), operands.end());
    }
    void op_str(u32 opcode, std::initializer_list<u32> before, const char* s, std::initializer_list<u32> after = {}) {
        std::vector<u32> str(std::strlen(s) / 4 + 1, 0);
        for (std::size_t k = 0; s[k]; ++k) str[k / 4] |= static_cast<u32>(static_cast<unsigned char>(s[k])) << (8 * (k % 4));
        w.push_back(static_cast<u32>(1 + before.size() + str.size() + after.size()) << 16 | opcode);
        w.insert(w.end(), before.begin(), before.end());
        w.insert(w.end(), str.begin(), str.end());
        w.insert(w.end(), after.begin(), after.end());
    }
    std::vector<u32> done() {
        w[3] = next;
        return w;
    }
};

bool valid(const std::vector<u32>& m, std::string* msg) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_0);
    std::string text;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m2) { text += m2; });
    const bool ok = tools.Validate(m);
    if (msg) *msg = text;
    return ok;
}

// How many instructions of `opcode` (and, for OpExtInst, of GLSL instruction `glsl`).
int count(const std::vector<u32>& m, u32 opcode, u32 glsl = ~0u) {
    int n = 0;
    for (std::size_t i = 5; i < m.size();) {
        const u32 len = m[i] >> 16;
        if (!len) break;
        if ((m[i] & 0xffff) == opcode && (glsl == ~0u || (len > 4 && m[i + 4] == glsl))) ++n;
        i += len;
    }
    return n;
}

int no_contraction(const std::vector<u32>& m) {
    int n = 0;
    for (std::size_t i = 5; i < m.size();) {
        const u32 len = m[i] >> 16;
        if (!len) break;
        if ((m[i] & 0xffff) == spv::OpDecorate && len == 3 && m[i + 2] == spv::DecNoContraction) ++n;
        i += len;
    }
    return n;
}

// DXC's form: Position a variable of its own, one return.
std::vector<u32> vertex_own_variable() {
    Asm a;
    const u32 glsl = a.id(), main = a.id(), in_pos = a.id(), out_pos = a.id(), t_void = a.id(), t_fn = a.id(), t_f32 = a.id(),
              t_v4 = a.id(), p_in = a.id(), p_out = a.id(), label = a.id(), v = a.id();
    a.op(spv::OpCapability, {spv::CapShader});
    a.op_str(spv::OpExtInstImport, {glsl}, "GLSL.std.450");
    a.op(spv::OpMemoryModel, {0, 1});
    a.op_str(spv::OpEntryPoint, {spv::EmVertex, main}, "main", {in_pos, out_pos});
    a.op(spv::OpDecorate, {in_pos, spv::DecLocation, 0});
    a.op(spv::OpDecorate, {out_pos, spv::DecBuiltIn, spv::BiPosition});
    a.op(spv::OpTypeVoid, {t_void});
    a.op(spv::OpTypeFunction, {t_fn, t_void});
    a.op(spv::OpTypeFloat, {t_f32, 32});
    a.op(spv::OpTypeVector, {t_v4, t_f32, 4});
    a.op(spv::OpTypePointer, {p_in, spv::ScInput, t_v4});
    a.op(spv::OpTypePointer, {p_out, spv::ScOutput, t_v4});
    a.op(spv::OpVariable, {p_in, in_pos, spv::ScInput});
    a.op(spv::OpVariable, {p_out, out_pos, spv::ScOutput});
    a.op(spv::OpFunction, {t_void, main, 0, t_fn});
    a.op(spv::OpLabel, {label});
    a.op(spv::OpLoad, {t_v4, v, in_pos});
    a.op(spv::OpStore, {out_pos, v});
    a.op(spv::OpReturn, {});
    a.op(spv::OpFunctionEnd, {});
    return a.done();
}

// The GLSL compilers' form: gl_PerVertex's member 0 through an access chain,
// a branch with a return on each side, no GLSL import and no bool type.
std::vector<u32> vertex_block_member() {
    Asm a;
    const u32 main = a.id(), in_pos = a.id(), block = a.id(), t_void = a.id(), t_fn = a.id(), t_f32 = a.id(), t_v4 = a.id(),
              t_s = a.id(), t_int = a.id(), c0 = a.id(), p_in = a.id(), p_block = a.id(), p_v4 = a.id(), entry = a.id(), yes = a.id(),
              no = a.id(), v = a.id(), x = a.id(), chain = a.id(), chain2 = a.id(), t_bool = a.id(), cond = a.id();
    a.op(spv::OpCapability, {spv::CapShader});
    a.op(spv::OpMemoryModel, {0, 1});
    a.op_str(spv::OpEntryPoint, {spv::EmVertex, main}, "main", {in_pos, block});
    a.op(spv::OpDecorate, {in_pos, spv::DecLocation, 0});
    a.op(spv::OpMemberDecorate, {t_s, 0, spv::DecBuiltIn, spv::BiPosition});
    a.op(spv::OpDecorate, {t_s, spv::DecBlock});
    a.op(spv::OpTypeVoid, {t_void});
    a.op(spv::OpTypeFunction, {t_fn, t_void});
    a.op(spv::OpTypeFloat, {t_f32, 32});
    a.op(spv::OpTypeVector, {t_v4, t_f32, 4});
    a.op(spv::OpTypeStruct, {t_s, t_v4});
    a.op(spv::OpTypeInt, {t_int, 32, 1});
    a.op(spv::OpConstant, {t_int, c0, 0});
    a.op(spv::OpTypePointer, {p_in, spv::ScInput, t_v4});
    a.op(spv::OpTypePointer, {p_block, spv::ScOutput, t_s});
    a.op(spv::OpTypePointer, {p_v4, spv::ScOutput, t_v4});
    a.op(spv::OpTypeBool, {t_bool});
    a.op(spv::OpVariable, {p_in, in_pos, spv::ScInput});
    a.op(spv::OpVariable, {p_block, block, spv::ScOutput});
    a.op(spv::OpFunction, {t_void, main, 0, t_fn});
    a.op(spv::OpLabel, {entry});
    a.op(spv::OpLoad, {t_v4, v, in_pos});
    a.op(spv::OpAccessChain, {p_v4, chain, block, c0});
    a.op(spv::OpStore, {chain, v});
    a.op(spv::OpCompositeExtract, {t_f32, x, v, 0});
    a.op(spv::OpFOrdGreaterThan, {t_bool, cond, x, x});
    a.op(spv::OpSelectionMerge, {no, 0});
    a.op(spv::OpBranchConditional, {cond, yes, no});
    a.op(spv::OpLabel, {yes});
    a.op(spv::OpAccessChain, {p_v4, chain2, block, c0});
    a.op(spv::OpStore, {chain2, v});
    a.op(spv::OpReturn, {});
    a.op(spv::OpLabel, {no});
    a.op(spv::OpReturn, {});
    a.op(spv::OpFunctionEnd, {});
    return a.done();
}

// A geometry stage: two emits of its own Position variable.
std::vector<u32> geometry_two_emits() {
    Asm a;
    const u32 main = a.id(), out_pos = a.id(), t_void = a.id(), t_fn = a.id(), t_f32 = a.id(), t_v4 = a.id(), p_out = a.id(),
              c1 = a.id(), cv = a.id(), label = a.id();
    a.op(spv::OpCapability, {spv::CapGeometry});
    a.op(spv::OpMemoryModel, {0, 1});
    a.op_str(spv::OpEntryPoint, {spv::EmGeometry, main}, "main", {out_pos});
    a.op(spv::OpExecutionMode, {main, spv::ExInputPoints});
    a.op(spv::OpExecutionMode, {main, spv::ExInvocations, 1});
    a.op(spv::OpExecutionMode, {main, spv::ExOutputTriangleStrip});
    a.op(spv::OpExecutionMode, {main, spv::ExOutputVertices, 2});
    a.op(spv::OpDecorate, {out_pos, spv::DecBuiltIn, spv::BiPosition});
    a.op(spv::OpTypeVoid, {t_void});
    a.op(spv::OpTypeFunction, {t_fn, t_void});
    a.op(spv::OpTypeFloat, {t_f32, 32});
    a.op(spv::OpTypeVector, {t_v4, t_f32, 4});
    a.op(spv::OpTypePointer, {p_out, spv::ScOutput, t_v4});
    a.op(spv::OpConstant, {t_f32, c1, 0x3f800000u});
    a.op(spv::OpConstantComposite, {t_v4, cv, c1, c1, c1, c1});
    a.op(spv::OpVariable, {p_out, out_pos, spv::ScOutput});
    a.op(spv::OpFunction, {t_void, main, 0, t_fn});
    a.op(spv::OpLabel, {label});
    a.op(spv::OpStore, {out_pos, cv});
    a.op(spv::OpEmitVertex, {});
    a.op(spv::OpStore, {out_pos, cv});
    a.op(spv::OpEmitVertex, {});
    a.op(spv::OpEndPrimitive, {});
    a.op(spv::OpReturn, {});
    a.op(spv::OpFunctionEnd, {});
    return a.done();
}

// A pixel stage: nothing to snap.
std::vector<u32> fragment() {
    Asm a;
    const u32 main = a.id(), t_void = a.id(), t_fn = a.id(), label = a.id();
    a.op(spv::OpCapability, {spv::CapShader});
    a.op(spv::OpMemoryModel, {0, 1});
    a.op_str(spv::OpEntryPoint, {spv::EmFragment, main}, "main");
    a.op(spv::OpExecutionMode, {main, spv::ExOriginUpperLeft});
    a.op(spv::OpTypeVoid, {t_void});
    a.op(spv::OpTypeFunction, {t_fn, t_void});
    a.op(spv::OpFunction, {t_void, main, 0, t_fn});
    a.op(spv::OpLabel, {label});
    a.op(spv::OpReturn, {});
    a.op(spv::OpFunctionEnd, {});
    return a.done();
}

void check_snapped(const char* what, std::vector<u32> m, int sites) {
    std::string msg;
    CHECK(valid(m, &msg));
    const std::vector<u32> before = m;
    std::string why;
    const bool ok = vertex_snap(m, 426.0f, 240.0f, &why);
    if (!ok) std::fprintf(stderr, "%s: refused: %s\n", what, why.c_str());
    CHECK(ok);
    if (!ok) return;
    const bool v = valid(m, &msg);
    if (!v) std::fprintf(stderr, "%s: invalid after the rewrite: %s\n", what, msg.c_str());
    CHECK(v);
    CHECK(m[3] > before[3]);                                     // the bound grew
    CHECK(count(m, spv::OpExtInst, spv::GlslFloor) == 2 * sites);  // x and y at each site
    CHECK(count(m, spv::OpExtInstImport) == 1);
    CHECK(no_contraction(m) == 12 * sites);  // div, mul, add, add, sub, div for each of x and y
    CHECK(count(m, spv::OpStore) == count(before, spv::OpStore) + sites);
}

void check_refused(const char* what, std::vector<u32> m) {
    const std::vector<u32> before = m;
    std::string why;
    CHECK(!vertex_snap(m, 320.0f, 240.0f, &why));
    CHECK(!why.empty());
    CHECK(m == before);
    std::printf("%s: refused (%s)\n", what, why.c_str());
}

}  // namespace

int main() {
    check_snapped("vertex, own variable", vertex_own_variable(), 1);
    check_snapped("vertex, gl_PerVertex member", vertex_block_member(), 2);
    check_snapped("geometry", geometry_two_emits(), 2);
    check_refused("fragment", fragment());
    {
        std::vector<u32> m = vertex_own_variable();
        CHECK(!vertex_snap(m, 0.0f, 240.0f));  // no grid
    }
    {
        // A vertex stage with no position output (its BuiltIn decoration gone).
        std::vector<u32> m = vertex_own_variable();
        for (std::size_t i = 5; i < m.size(); i += m[i] >> 16) {
            if ((m[i] & 0xffff) == spv::OpDecorate && m[i + 2] == spv::DecBuiltIn) m[i + 3] = 1;  // PointSize
        }
        check_refused("vertex without a position", m);
    }
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("vertex snap: all checks passed\n");
    return 0;
}
