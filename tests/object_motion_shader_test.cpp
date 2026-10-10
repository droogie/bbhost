// Tests for object motion GCN -> SPIR-V instrumentation.
// Verifies StageParams layout, synthetic VS/PS compilation, physical storage buffer atomic exchanges,
// bounds checking, varying locations, and validates with SPIRV-Tools.
#include "test_app0.h"
#include "gcn/container.h"
#include "gcn/isa.h"
#include "gcn/spirv.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* desc) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", desc);
        ++g_failures;
    }
}

// Sea Islands encodings for synthetic shaders.
constexpr std::uint32_t kEndpgm = 0xbf810000u;

std::vector<std::uint32_t> make_synthetic_vs() {
    // exp pos0 (tgt=12, dmask=0xf, done=1), vsrc = v0, v1, v2, v3
    // w0 = (0x3eu << 26) | (1u << 11) | (12u << 4) | 0xfu
    // w1 = 0 | (1u << 8) | (2u << 16) | (3u << 24)
    const std::uint32_t exp_w0 = (0x3eu << 26) | (1u << 11) | (12u << 4) | 0xfu;
    const std::uint32_t exp_w1 = 0u | (1u << 8) | (2u << 16) | (3u << 24);
    return {exp_w0, exp_w1, kEndpgm};
}

std::vector<std::uint32_t> make_synthetic_ps() {
    // exp mrt0 (tgt=0, dmask=0xf, done=1), vsrc = v0, v1, v2, v3
    const std::uint32_t exp_w0 = (0x3eu << 26) | (1u << 11) | (0u << 4) | 0xfu;
    const std::uint32_t exp_w1 = 0u | (1u << 8) | (2u << 16) | (3u << 24);
    return {exp_w0, exp_w1, kEndpgm};
}

bool contains_opcode(const std::vector<std::uint32_t>& spirv, std::uint16_t opcode) {
    std::size_t i = 5;  // skip header
    while (i < spirv.size()) {
        const std::uint32_t word = spirv[i];
        const std::uint16_t op = static_cast<std::uint16_t>(word & 0xffff);
        const std::uint16_t word_count = static_cast<std::uint16_t>(word >> 16);
        if (word_count == 0) break;
        if (op == opcode) return true;
        i += word_count;
    }
    return false;
}

bool has_decoration_location(const std::vector<std::uint32_t>& spirv, std::uint32_t loc) {
    std::size_t i = 5;
    while (i < spirv.size()) {
        const std::uint32_t word = spirv[i];
        const std::uint16_t op = static_cast<std::uint16_t>(word & 0xffff);
        const std::uint16_t word_count = static_cast<std::uint16_t>(word >> 16);
        if (word_count == 0) break;
        // OpDecorate: target, DecorationLocation (30), location_value
        if (op == spv::OpDecorate && word_count >= 4) {
            if (spirv[i + 2] == spv::DecLocation && spirv[i + 3] == loc) {
                return true;
            }
        }
        i += word_count;
    }
    return false;
}

bool has_member_decoration_offset(const std::vector<std::uint32_t>& spirv, std::uint32_t offset) {
    std::size_t i = 5;
    while (i < spirv.size()) {
        const std::uint32_t word = spirv[i];
        const std::uint16_t op = static_cast<std::uint16_t>(word & 0xffff);
        const std::uint16_t word_count = static_cast<std::uint16_t>(word >> 16);
        if (word_count == 0) break;
        // OpMemberDecorate: type, member, DecorationOffset (35), byte_offset
        if (op == spv::OpMemberDecorate && word_count >= 5) {
            if (spirv[i + 3] == spv::DecOffset && spirv[i + 4] == offset) {
                return true;
            }
        }
        i += word_count;
    }
    return false;
}

void test_stage_params_layout() {
    static_assert(sizeof(gcn::StageParams) == 576, "StageParams size must be 576 bytes");
    static_assert(offsetof(gcn::StageParams, image_index) == 384, "image_index offset");
    static_assert(offsetof(gcn::StageParams, motion_positions) == 512, "motion_positions offset");
    static_assert(offsetof(gcn::StageParams, motion_store) == 520, "motion_store offset");
    static_assert(offsetof(gcn::StageParams, motion_load) == 524, "motion_load offset");
    static_assert(offsetof(gcn::StageParams, motion_vertices) == 528, "motion_vertices offset");
    static_assert(offsetof(gcn::StageParams, motion_first_vertex) == 532, "motion_first_vertex offset");
    static_assert(offsetof(gcn::StageParams, motion_instances) == 536, "motion_instances offset");
    static_assert(offsetof(gcn::StageParams, motion_first_instance) == 540, "motion_first_instance offset");
    static_assert(offsetof(gcn::StageParams, motion_flags) == 544, "motion_flags offset");
    static_assert(offsetof(gcn::StageParams, motion_pad) == 548, "motion_pad offset");
    static_assert(offsetof(gcn::StageParams, motion_scale) == 552, "motion_scale offset");
    static_assert(offsetof(gcn::StageParams, motion_reserved) == 560, "motion_reserved offset");

    check(sizeof(gcn::StageParams) == 576, "sizeof(StageParams) == 576");
    check(offsetof(gcn::StageParams, motion_positions) == 512, "offsetof(motion_positions) == 512");
    std::printf("test_stage_params_layout: OK (size=576, offset=512)\n");
}

void test_synthetic_vs(spvtools::SpirvTools& tools, const spvtools::ValidatorOptions& vo) {
    const auto vs_words = make_synthetic_vs();
    const gcn::Program prog = gcn::decode(vs_words.data(), vs_words.size());

    // 1. Uninstrumented VS
    {
        gcn::TranslateOptions opt;
        opt.stage = gcn::Stage::Vertex;
        opt.object_motion = false;
        const gcn::TranslateResult res = gcn::translate(prog, opt);
        check(res.ok(), "Uninstrumented VS translation succeeds");
        check(tools.Validate(res.spirv.data(), res.spirv.size(), vo), "Uninstrumented VS spirv-val passes");
        check(!contains_opcode(res.spirv, spv::OpAtomicExchange), "Uninstrumented VS has no OpAtomicExchange");
    }

    // 2. Instrumented VS at motion_location = 2
    {
        gcn::TranslateOptions opt;
        opt.stage = gcn::Stage::Vertex;
        opt.object_motion = true;
        opt.motion_location = 2;
        const gcn::TranslateResult res = gcn::translate(prog, opt);
        check(res.ok(), "Instrumented VS (loc=2) translation succeeds");
        check(tools.Validate(res.spirv.data(), res.spirv.size(), vo), "Instrumented VS (loc=2) spirv-val passes");
        check(contains_opcode(res.spirv, spv::OpAtomicExchange), "Instrumented VS has OpAtomicExchange");
        check(contains_opcode(res.spirv, spv::OpConvertUToPtr), "Instrumented VS has OpConvertUToPtr");
        check(contains_opcode(res.spirv, spv::OpSelectionMerge), "Instrumented VS has OpSelectionMerge");
        check(has_decoration_location(res.spirv, 2), "Instrumented VS outputs at motion_location 2");
        check(has_decoration_location(res.spirv, 3), "Instrumented VS outputs at motion_location 3");
        check(has_member_decoration_offset(res.spirv, 512), "Instrumented VS StageParams has offset 512");
        check(has_member_decoration_offset(res.spirv, 520), "Instrumented VS StageParams has offset 520");
        check(has_member_decoration_offset(res.spirv, 552), "Instrumented VS StageParams has offset 552");
    }

    // 3. Instrumented VS at custom motion_location = 10 (proves no hardcoded 30/31)
    {
        gcn::TranslateOptions opt;
        opt.stage = gcn::Stage::Vertex;
        opt.object_motion = true;
        opt.motion_location = 10;
        const gcn::TranslateResult res = gcn::translate(prog, opt);
        check(res.ok(), "Instrumented VS (loc=10) translation succeeds");
        check(tools.Validate(res.spirv.data(), res.spirv.size(), vo), "Instrumented VS (loc=10) spirv-val passes");
        check(has_decoration_location(res.spirv, 10), "Instrumented VS outputs at motion_location 10");
        check(has_decoration_location(res.spirv, 11), "Instrumented VS outputs at motion_location 11");
        check(!has_decoration_location(res.spirv, 30), "Instrumented VS does NOT hardcode location 30");
        check(!has_decoration_location(res.spirv, 31), "Instrumented VS does NOT hardcode location 31");
    }
    std::printf("test_synthetic_vs: OK\n");
}

void test_synthetic_ps(spvtools::SpirvTools& tools, const spvtools::ValidatorOptions& vo) {
    const auto ps_words = make_synthetic_ps();
    const gcn::Program prog = gcn::decode(ps_words.data(), ps_words.size());

    // 1. Uninstrumented PS
    {
        gcn::TranslateOptions opt;
        opt.stage = gcn::Stage::Pixel;
        opt.object_motion = false;
        const gcn::TranslateResult res = gcn::translate(prog, opt);
        check(res.ok(), "Uninstrumented PS translation succeeds");
        check(tools.Validate(res.spirv.data(), res.spirv.size(), vo), "Uninstrumented PS spirv-val passes");
        check(!has_decoration_location(res.spirv, 7), "Uninstrumented PS has no MRT 7");
    }

    // 2. Instrumented PS at motion_location = 2
    {
        gcn::TranslateOptions opt;
        opt.stage = gcn::Stage::Pixel;
        opt.object_motion = true;
        opt.motion_location = 2;
        const gcn::TranslateResult res = gcn::translate(prog, opt);
        check(res.ok(), "Instrumented PS (loc=2) translation succeeds");
        check(tools.Validate(res.spirv.data(), res.spirv.size(), vo), "Instrumented PS (loc=2) spirv-val passes");
        check(has_decoration_location(res.spirv, 2), "Instrumented PS inputs at motion_location 2");
        check(has_decoration_location(res.spirv, 3), "Instrumented PS inputs at motion_location 3");
        check(has_decoration_location(res.spirv, 7), "Instrumented PS exports to MRT 7");
        check(contains_opcode(res.spirv, spv::OpFDiv), "Instrumented PS has OpFDiv for NDC");
        check(contains_opcode(res.spirv, spv::OpFMul), "Instrumented PS has OpFMul for scale");
        check(contains_opcode(res.spirv, spv::OpFOrdGreaterThan), "Instrumented PS has OpFOrdGreaterThan for epsilon check");
    }

    // 3. Instrumented PS at custom motion_location = 8
    {
        gcn::TranslateOptions opt;
        opt.stage = gcn::Stage::Pixel;
        opt.object_motion = true;
        opt.motion_location = 8;
        const gcn::TranslateResult res = gcn::translate(prog, opt);
        check(res.ok(), "Instrumented PS (loc=8) translation succeeds");
        check(tools.Validate(res.spirv.data(), res.spirv.size(), vo), "Instrumented PS (loc=8) spirv-val passes");
        check(has_decoration_location(res.spirv, 8), "Instrumented PS inputs at motion_location 8");
        check(has_decoration_location(res.spirv, 9), "Instrumented PS inputs at motion_location 9");
        check(has_decoration_location(res.spirv, 7), "Instrumented PS exports to MRT 7");
    }
    std::printf("test_synthetic_ps: OK\n");
}

void test_game_shaders_if_available(spvtools::SpirvTools& tools, const spvtools::ValidatorOptions& vo) {
    std::string candidate_paths[] = {
        test_app0_file("dvdroot_ps4/shader/gxflvershader.shaderbnd.dcx"),
        test_app0_file("dvdroot_ps4/shader/gxgui.shaderbnd.dcx"),
        test_app0_file("dvdroot_ps4/shader/gxshader.shaderbnd.dcx"),
    };

    std::string path;
    for (const auto& cp : candidate_paths) {
        if (!cp.empty()) {
            std::vector<std::uint8_t> test_raw;
            if (gcn::read_file(cp, test_raw)) {
                path = cp;
                break;
            }
        }
    }

    if (path.empty()) {
        std::printf("test_game_shaders: skipped (no dump available)\n");
        return;
    }

    std::vector<std::uint8_t> raw;
    const bool read = gcn::read_file(path, raw);
    check(read, "Read game shader bundle");
    if (!read) return;
    std::string err;
    const std::vector<std::uint8_t> b = gcn::dcx_decompress(raw, &err);
    std::vector<gcn::BundleEntry> entries;
    const bool unpacked = !b.empty() && gcn::bnd4_entries(b, entries, &err);
    check(unpacked, "Decompress game shader bundle");
    if (!unpacked) return;

    std::size_t vs_tested = 0, ps_tested = 0;
    for (const auto& e : entries) {
        gcn::ShaderCode code;
        if (!gcn::shader_code(e.data, code)) continue;
        std::size_t shdr = e.data.size();
        for (std::size_t i = 0; i + 4 <= e.data.size(); ++i) {
            if (std::memcmp(e.data.data() + i, "Shdr", 4) == 0) { shdr = i; break; }
        }
        std::uint32_t regs[16];
        if (shdr == e.data.size() || e.data.size() - shdr < 16 + sizeof(regs)) continue;
        std::memcpy(regs, e.data.data() + shdr + 16, 64);
        gcn::TranslateOptions opt;
        opt.stage = code.type == 2 ? gcn::Stage::Pixel : code.type == 4 ? gcn::Stage::Compute : gcn::Stage::Vertex;
        opt.rsrc1 = regs[4];
        opt.rsrc2 = regs[5];
        if (opt.stage == gcn::Stage::Pixel) opt.ps_input_ena = regs[8];
        if (opt.stage == gcn::Stage::Vertex) opt.vs_out_cntl = regs[8];

        if (opt.stage == gcn::Stage::Vertex) {
            opt.object_motion = true;
            opt.motion_location = 14;
            const gcn::Program p = gcn::decode(code.words.data(), code.words.size());
            const gcn::TranslateResult r = gcn::translate(p, opt);
            if (r.ok()) {
                check(tools.Validate(r.spirv.data(), r.spirv.size(), vo), "Game VS spirv-val passes");
                ++vs_tested;
            }
        } else if (opt.stage == gcn::Stage::Pixel) {
            opt.object_motion = true;
            opt.motion_location = 14;
            const gcn::Program p = gcn::decode(code.words.data(), code.words.size());
            const gcn::TranslateResult r = gcn::translate(p, opt);
            if (r.ok()) {
                check(tools.Validate(r.spirv.data(), r.spirv.size(), vo), "Game PS spirv-val passes");
                ++ps_tested;
            }
        }
    }
    check(vs_tested > 0 && ps_tested > 0, "Game bundle exercised vertex and pixel motion variants");
    std::printf("test_game_shaders: OK (%s: %zu VS, %zu PS tested)\n", path.c_str(), vs_tested, ps_tested);
}

}  // namespace

int main() {
    std::printf("=== object_motion_shader_test ===\n");
    test_stage_params_layout();

    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        std::fprintf(stderr, "SPIRV-Tools: %s\n", message);
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);

    test_synthetic_vs(tools, vo);
    test_synthetic_ps(tools, vo);
    test_game_shaders_if_available(tools, vo);

    if (g_failures != 0) {
        std::fprintf(stderr, "object_motion_shader_test FAILED with %d errors\n", g_failures);
        return 1;
    }
    std::printf("object_motion_shader_test ALL PASSED\n");
    return 0;
}
