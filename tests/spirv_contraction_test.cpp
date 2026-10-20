// spv::relax_contraction: which NoContraction decorations a vertex and a
// pixel shader keep, and that the result is still valid SPIR-V.
#include "host/spirv_relax.h"

#include "gcn/spirv.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <set>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool valid(const std::vector<std::uint32_t>& w) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    const bool ok = tools.Validate(w);
    if (!ok) std::fprintf(stderr, "invalid: %s\n", msg.c_str());
    return ok;
}

std::set<std::uint32_t> no_contraction(const std::vector<std::uint32_t>& w) {
    std::set<std::uint32_t> ids;
    for (std::size_t i = 5; i < w.size(); i += w[i] >> 16) {
        if ((w[i] & 0xffff) == spv::OpDecorate && (w[i] >> 16) == 3 && w[i + 2] == spv::DecNoContraction) ids.insert(w[i + 1]);
    }
    return ids;
}

struct Shader {
    std::vector<std::uint32_t> words;
    spv::Id exact = 0;      // must keep NoContraction
    spv::Id through = 0;    // reaches the exact output through a function variable: must keep it too
    spv::Id condition = 0;  // feeds a branch condition
    spv::Id free = 0;       // feeds only an ordinary output: may lose it
};

// A vertex shader (or, with `fragment`, a pixel shader that may discard and
// writes FragDepth): position = (a*b + c) through a Function variable, a
// branch on (a*c > 0), and a varying a*a + b.
Shader build(bool fragment) {
    using namespace spv;
    Module m;
    m.capability(1 /* Shader */);
    m.memory_model(0, 1);
    const Id f32 = m.type_float(32), v4 = m.type_vector(f32, 4), vd = m.type_void(), b = m.type_bool();
    const Id in_ptr = m.type_pointer(StorageClass(1) /* Input */, v4), out_ptr = m.type_pointer(StorageClass(3) /* Output */, v4);
    const Id out_f = m.type_pointer(StorageClass(3), f32), fn_ptr = m.type_pointer(StorageClass(7) /* Function */, f32);
    const Id in = m.global_variable(in_ptr, StorageClass(1));
    m.decorate(in, DecLocation, {0});
    const Id varying = m.global_variable(out_ptr, StorageClass(3));
    m.decorate(varying, DecLocation, {0});
    Id exact_var = 0;
    if (fragment) {
        exact_var = m.global_variable(out_f, StorageClass(3));
        m.decorate(exact_var, DecBuiltIn, {BiFragDepth});
    } else {
        exact_var = m.global_variable(out_ptr, StorageClass(3));
        m.decorate(exact_var, DecBuiltIn, {BiPosition});
    }
    const Id fn = m.begin_function(vd, m.type_function(vd, {}));
    m.label();
    const Id tmp = m.local_variable(fn_ptr);
    const Id v = m.load(v4, in);
    const Id a = m.emit(OpCompositeExtract, f32, {v, 0}), bb = m.emit(OpCompositeExtract, f32, {v, 1}), c = m.emit(OpCompositeExtract, f32, {v, 2});
    Shader s;
    const Id ab = m.emit(OpFMul, f32, {a, bb});
    s.through = ab;
    s.exact = m.emit(OpFAdd, f32, {ab, c});
    m.store(tmp, s.exact);
    const Id ac = m.emit(OpFMul, f32, {a, c});
    s.condition = ac;
    const Id positive = m.emit(OpFOrdGreaterThan, b, {ac, m.const_f32(0.0f)});
    const Id aa = m.emit(OpFMul, f32, {a, a});
    s.free = m.emit(OpFAdd, f32, {aa, bb});
    m.store(varying, m.emit(OpCompositeConstruct, v4, {s.free, s.free, s.free, s.free}));
    const Id l_then = m.fresh(), l_merge = m.fresh();
    m.emit_void(OpSelectionMerge, {l_merge, 0u});
    m.emit_void(OpBranchConditional, {positive, l_then, l_merge});
    m.label(l_then);
    if (fragment) {
        m.emit_void(OpKill, {});
    } else {
        m.emit_void(OpBranch, {l_merge});
    }
    m.label(l_merge);
    const Id e = m.load(f32, tmp);
    if (fragment) {
        m.store(exact_var, e);
    } else {
        m.store(exact_var, m.emit(OpCompositeConstruct, v4, {e, e, e, m.const_f32(1.0f)}));
    }
    m.emit_void(OpReturn, {});
    m.end_function();
    m.entry_point(fragment ? EmFragment : EmVertex, fn, "main", {in, varying, exact_var});
    if (fragment) m.execution_mode(fn, ExOriginUpperLeft);
    if (fragment) m.execution_mode(fn, ExDepthReplacing);
    s.words = m.assemble();
    return s;
}

}  // namespace

int main() {
    for (const bool fragment : {false, true}) {
        const Shader s = build(fragment);
        expect(valid(s.words), "the test module is valid");
        const std::set<std::uint32_t> before = no_contraction(s.words);
        expect(before.count(s.exact) && before.count(s.free), "Module::emit decorates float arithmetic");
        spv::RelaxStats st;
        const std::vector<std::uint32_t> frag_only = spv::relax_contraction(s.words, spv::RelaxStages::Fragment, &st);
        if (!fragment) expect(frag_only == s.words && st.dropped == 0, "a vertex stage is unchanged in fragment scope");
        const std::vector<std::uint32_t> r = spv::relax_contraction(s.words, spv::RelaxStages::Graphics, &st);
        expect(valid(r), "the relaxed module is valid");
        const std::set<std::uint32_t> after = no_contraction(r);
        expect(after.count(s.exact) == 1, "the exact output's value keeps NoContraction");
        expect(after.count(s.through) == 1, "a value reaching the exact output through a variable keeps it");
        expect(after.count(s.condition) == 1, "a branch condition's value keeps it");
        expect(after.count(s.free) == 0, "a value feeding only a varying loses it");
        expect(st.dropped == before.size() - after.size() && st.kept == after.size(), "the counts add up");
        if (fragment) {
            // Discard decisions free: the FragDepth write still pins the branch condition.
            const std::set<std::uint32_t> free_discards = no_contraction(spv::relax_contraction(s.words, spv::RelaxStages::Graphics, &st, false));
            expect(free_discards == after, "a pixel shader writing FragDepth keeps its condition with discards free");
        }
        if (!fragment) {
            // Invariant in place of exact arithmetic.
            const std::vector<std::uint32_t> inv = spv::relax_contraction(s.words, spv::RelaxStages::GraphicsInvariant, &st);
            expect(valid(inv), "the invariant-relaxed module is valid");
            const std::set<std::uint32_t> kept = no_contraction(inv);
            expect(kept.size() == 1 && kept.count(s.condition) == 1, "invariant scope keeps only the branch condition's");
            int invariant = 0;
            for (std::size_t i = 5; i < inv.size(); i += inv[i] >> 16) {
                if ((inv[i] & 0xffff) == spv::OpDecorate && (inv[i] >> 16) == 3 && inv[i + 2] == spv::DecInvariant) ++invariant;
            }
            expect(invariant == 1, "the Position output is decorated Invariant once");
            expect(spv::relax_contraction(inv, spv::RelaxStages::GraphicsInvariant) == inv, "a second pass changes nothing");
        }
    }
    // A pixel shader that neither discards nor writes depth loses every decoration.
    {
        using namespace spv;
        Module m;
        m.capability(1);
        m.memory_model(0, 1);
        const Id f32 = m.type_float(32), v4 = m.type_vector(f32, 4), vd = m.type_void();
        const Id in = m.global_variable(m.type_pointer(StorageClass(1), v4), StorageClass(1));
        m.decorate(in, DecLocation, {0});
        const Id out = m.global_variable(m.type_pointer(StorageClass(3), v4), StorageClass(3));
        m.decorate(out, DecLocation, {0});
        const Id fn = m.begin_function(vd, m.type_function(vd, {}));
        m.label();
        const Id v = m.load(v4, in);
        const Id x = m.emit(OpCompositeExtract, f32, {v, 0});
        const Id y = m.emit(OpFAdd, f32, {m.emit(OpFMul, f32, {x, x}), x});
        m.store(out, m.emit(OpCompositeConstruct, v4, {y, y, y, y}));
        m.emit_void(OpReturn, {});
        m.end_function();
        m.entry_point(EmFragment, fn, "main", {in, out});
        m.execution_mode(fn, ExOriginUpperLeft);
        const std::vector<std::uint32_t> w = m.assemble();
        RelaxStats st;
        const std::vector<std::uint32_t> r = relax_contraction(w, RelaxStages::Fragment, &st);
        expect(valid(r), "the relaxed pixel shader is valid");
        expect(no_contraction(r).empty() && st.dropped == 2 && st.kept == 0, "a plain pixel shader loses every NoContraction");
    }
    // A pixel shader that discards when x*x + x > 0 and writes no depth: its
    // condition stays exact only with exact_discards.
    {
        using namespace spv;
        Module m;
        m.capability(1);
        m.memory_model(0, 1);
        const Id f32 = m.type_float(32), v4 = m.type_vector(f32, 4), vd = m.type_void(), b = m.type_bool();
        const Id in = m.global_variable(m.type_pointer(StorageClass(1), v4), StorageClass(1));
        m.decorate(in, DecLocation, {0});
        const Id out = m.global_variable(m.type_pointer(StorageClass(3), v4), StorageClass(3));
        m.decorate(out, DecLocation, {0});
        const Id fn = m.begin_function(vd, m.type_function(vd, {}));
        m.label();
        const Id v = m.load(v4, in);
        const Id x = m.emit(OpCompositeExtract, f32, {v, 0});
        const Id y = m.emit(OpFAdd, f32, {m.emit(OpFMul, f32, {x, x}), x});
        const Id positive = m.emit(OpFOrdGreaterThan, b, {y, m.const_f32(0.0f)});
        const Id l_kill = m.fresh(), l_merge = m.fresh();
        m.emit_void(OpSelectionMerge, {l_merge, 0u});
        m.emit_void(OpBranchConditional, {positive, l_kill, l_merge});
        m.label(l_kill);
        m.emit_void(OpKill, {});
        m.label(l_merge);
        m.store(out, v);
        m.emit_void(OpReturn, {});
        m.end_function();
        m.entry_point(EmFragment, fn, "main", {in, out});
        m.execution_mode(fn, ExOriginUpperLeft);
        const std::vector<std::uint32_t> w = m.assemble();
        expect(valid(w), "the discarding pixel shader is valid");
        RelaxStats st;
        const std::vector<std::uint32_t> exact = relax_contraction(w, RelaxStages::Fragment, &st);
        expect(no_contraction(exact).count(y) == 1 && st.kept == 2, "exact discards keep the condition's slice");
        const std::vector<std::uint32_t> free = relax_contraction(w, RelaxStages::Fragment, &st, false);
        expect(valid(free), "the relaxed discarding pixel shader is valid");
        expect(no_contraction(free).empty() && st.dropped == 2 && st.kept == 0, "free discards drop every NoContraction");
    }
    if (failures) return 1;
    std::printf("spirv_contraction_test: ok\n");
    return 0;
}
