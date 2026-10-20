// gcn/tess_vertex.h: the domain points tile the unit square, wound as asked;
// a fused LS and domain shader is one valid vertex stage with no tessellation
// left; params blocks of different types are refused.
#include "gcn/isa.h"
#include "gcn/tess_vertex.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

// ds_<op> in the DS encoding: op, offset; then vdst, data0, addr.
void ds(std::vector<std::uint32_t>& w, std::uint32_t op, std::uint32_t offset, std::uint32_t dst, std::uint32_t data0, std::uint32_t addr) {
    w.push_back(0xd8000000u | (op << 18) | (offset & 0xffff));
    w.push_back((dst << 24) | (data0 << 8) | addr);
}
// exp <tgt> v[a] v[b] v[c] v[d], all four channels, done.
void exp(std::vector<std::uint32_t>& w, std::uint32_t tgt, std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
    w.push_back(0xf8000000u | (1u << 11) | (tgt << 4) | 0xf);
    w.push_back(a | (b << 8) | (c << 16) | (d << 24));
}

struct Counts {
    int entry_points = 0, vertex_entry = 0, tess_modes = 0, params = 0, vertex_index = 0, inputs = 0, tess_builtins = 0;
};
Counts count(const std::vector<std::uint32_t>& spv) {
    Counts c;
    std::vector<std::uint32_t> input_vars;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t op = spv[i] & 0xffff, len = spv[i] >> 16;
        if (!len || i + len > spv.size()) break;
        if (op == 15) {  // OpEntryPoint
            ++c.entry_points;
            if (spv[i + 1] == 0) ++c.vertex_entry;
        }
        if (op == 16 && (spv[i + 2] == 24 || spv[i + 2] == 22 || spv[i + 2] == 1 || spv[i + 2] == 4)) ++c.tess_modes;  // Quads...
        if (op == 71 && spv[i + 2] == 33 && spv[i + 3] == gcn::kBindingParams) ++c.params;                              // Binding
        if (op == 71 && spv[i + 2] == 11 && spv[i + 3] == 42) ++c.vertex_index;                                       // BuiltIn VertexIndex
        if (op == 71 && spv[i + 2] == 11 && (spv[i + 3] == 13 || spv[i + 3] == 7)) ++c.tess_builtins;                 // TessCoord, PrimitiveId
        if (op == 59 && spv[i + 3] == 1) ++c.inputs;                                                                   // OpVariable Input
        i += len;
    }
    return c;
}

void domain_points() {
    for (std::uint32_t level : {1u, 2u, 3u, 8u}) {
        for (bool cw : {true, false}) {
            const std::uint32_t n = gcn::tess_vertex_points(level);
            CHECK(n == (level == 1 ? 6u : 12u * level));
            double area = 0.0;
            bool wound = true, inside = true;
            for (std::uint32_t t = 0; t < n; t += 3) {
                float u[3], v[3];
                for (int k = 0; k < 3; ++k) {
                    gcn::tess_vertex_domain_point(level, cw, t + k, u[k], v[k]);
                    inside = inside && u[k] >= 0.0f && u[k] <= 1.0f && v[k] >= 0.0f && v[k] <= 1.0f;
                }
                // Positive: clockwise on a domain whose v points down.
                const double a = 0.5 * ((u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]));
                wound = wound && (cw ? a > 0.0 : a < 0.0);
                area += std::fabs(a);
            }
            std::printf("gcn_tess_vertex_test: level %u %s: %u points, area %.6f%s\n", level, cw ? "cw" : "ccw", n, area,
                        wound ? "" : ", wound wrong");
            CHECK(inside);
            CHECK(wound);
            CHECK(std::fabs(area - 1.0) < 1e-5);
        }
    }
}

}  // namespace

int main() {
    domain_points();

    // LS: ds_write_b32 v1, v0 (its vertex id at its control point's dword 0); s_endpgm.
    std::vector<std::uint32_t> ls_words;
    ds(ls_words, 13, 0, 0, 0, 1);
    ls_words.push_back(0xbf810000u);
    // Domain shader: ds_read_b32 v4, v2 (that dword); exp pos0 v0 v1 v4 v4; exp param0 v4 v4 v4 v4; s_endpgm.
    std::vector<std::uint32_t> ds_words;
    ds(ds_words, 54, 0, 4, 0, 2);
    exp(ds_words, 12, 0, 1, 4, 4);
    exp(ds_words, 32, 4, 4, 4, 4);
    ds_words.push_back(0xbf810000u);
    const gcn::Program ls = gcn::decode(ls_words.data(), ls_words.size());
    const gcn::Program dom = gcn::decode(ds_words.data(), ds_words.size());
    CHECK(ls.errors.empty() && dom.errors.empty());

    gcn::TranslateOptions lo;
    lo.stage = gcn::Stage::Vertex;
    lo.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
    lo.tess_lds_attributes = true;
    lo.tess_attr_vec4s = 16;
    lo.rsrc1 = 1u << 24;  // VGPR_COMP_CNT 1
    lo.descriptor_set = 0;
    gcn::TranslateOptions dop;
    dop.stage = gcn::Stage::TessEval;
    dop.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
    dop.tess_lds_attributes = true;
    dop.tess_attr_vec4s = 16;
    dop.descriptor_set = 0;
    dop.params_vertex_formats = true;
    const gcn::TranslateResult lt = gcn::translate(ls, lo), dt = gcn::translate(dom, dop);
    CHECK(lt.ok() && dt.ok());

    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    CHECK(tools.Validate(lt.spirv) && tools.Validate(dt.spirv));
    for (std::uint32_t level : {1u, 3u}) {
        const gcn::TessVertexFuse f = gcn::fuse_tess_vertex(lt.spirv, dt.spirv, level, true);
        msg.clear();
        const bool valid = f.ok() && tools.Validate(f.spirv);
        const Counts c = count(f.spirv);
        std::printf("gcn_tess_vertex_test: fused at level %u: %s, %zu words%s; %d entry point(s), %d params binding(s), %d stage input(s)\n",
                    level, f.ok() ? "ok" : f.error.c_str(), f.spirv.size(), valid ? ", valid" : (", invalid: " + msg).c_str(),
                    c.entry_points, c.params, c.inputs);
        CHECK(f.ok());
        CHECK(valid);
        CHECK(c.entry_points == 1 && c.vertex_entry == 1);
        CHECK(c.tess_modes == 0);
        CHECK(c.params == 1);
        CHECK(c.vertex_index == 1);
        CHECK(c.tess_builtins == 0);
    }
    // The domain shader's params block without the LS's member: two blocks at one binding.
    gcn::TranslateOptions plain = dop;
    plain.params_vertex_formats = false;
    const gcn::TessVertexFuse bad = gcn::fuse_tess_vertex(lt.spirv, gcn::translate(dom, plain).spirv, 1, true);
    std::printf("gcn_tess_vertex_test: without params_vertex_formats: %s\n", bad.ok() ? "fused" : bad.error.c_str());
    CHECK(!bad.ok() && bad.error.find("binding") != std::string::npos);
    // The stages the wrong way round.
    CHECK(!gcn::fuse_tess_vertex(dt.spirv, lt.spirv, 1, true).ok());

    if (g_fail) {
        std::printf("%d failed\n", g_fail);
        return 1;
    }
    std::printf("gcn_tess_vertex_test: ok\n");
    return 0;
}
