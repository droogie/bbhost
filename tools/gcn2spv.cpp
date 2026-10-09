// gcn2spv: translate shaders from From's bundles to SPIR-V and validate them.
//
//   gcn2spv --check <bundle.dcx>...          translate + validate every shader
//   gcn2spv <bundle.dcx> <name-substr> [out.spv]   translate matching entries (prints errors, writes the first)
//   gcn2spv --raw <code.bin> <vs|ps|cs|ls|ds[:level]> <rsrc1> <rsrc2> [fetch.bin] [out.spv]
//                                            a program saved from guest memory (BBHOST_TESS_PROBE's tmp/tess/*.bin):
//                                            ls = LS as the tessellation compute pass, ds = domain shader as a vertex stage;
//                                            tri:<window hex> for a triangle patch of 3 with its own LDS window (hs, tes, ls)
#include "gcn/container.h"
#include "gcn/half.h"
#include "gcn/isa.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Header {
    gcn::Stage stage;
    std::uint32_t regs[16];
};

bool read_header(const gcn::BundleEntry& e, const gcn::ShaderCode& code, Header& h) {
    // Stage registers follow the 16-byte ShaderFileHeader.
    std::size_t shdr = 0;
    for (std::size_t i = 0; i + 4 <= e.data.size(); ++i) {
        if (std::memcmp(e.data.data() + i, "Shdr", 4) == 0) { shdr = i; break; }
    }
    if (shdr + 16 + 64 > e.data.size()) return false;
    std::memcpy(h.regs, e.data.data() + shdr + 16, 64);
    switch (code.type) {
    case 1: h.stage = gcn::Stage::Vertex; break;
    case 2: h.stage = gcn::Stage::Pixel; break;
    case 4: h.stage = gcn::Stage::Compute; break;
    default: return false;
    }
    return true;
}

// GCN2SPV_LANE_SCAN=1: as on MoltenVK, no subgroup built-ins in these stages.
bool lane_from_scan(gcn::Stage stage) {
    static const bool on = [] {
        const char* e = std::getenv("GCN2SPV_LANE_SCAN");
        return e && e[0] == '1';
    }();
    return on && (stage == gcn::Stage::Vertex || stage == gcn::Stage::TessEval);
}

gcn::TranslateOptions options_for(const Header& h) {
    gcn::TranslateOptions o;
    o.stage = h.stage;
    o.rsrc1 = h.regs[4];
    o.rsrc2 = h.regs[5];
    // GCN2SPV_CB_SSBO=1: translate as the renderer does, with constant buffers
    // read through storage-buffer bindings (graphics stages only).
    static const bool cb_ssbo = [] {
        const char* e = std::getenv("GCN2SPV_CB_SSBO");
        return e && e[0] == '1';
    }();
    o.cb_ssbo = cb_ssbo && h.stage != gcn::Stage::Compute;
    // GCN2SPV_CB_LEAN=1: also drop the page-table fallback, as the renderer's
    // variant for draws whose constant buffers all bind.
    static const bool cb_lean = [] {
        const char* e = std::getenv("GCN2SPV_CB_LEAN");
        return e && e[0] == '1';
    }();
    o.cb_no_fallback = o.cb_ssbo && cb_lean;
    // GCN2SPV_EXEC_KNOWN=0: keep every EXEC-masked write (BBHOST_EXEC_KNOWN=0).
    static const bool exec_known = [] {
        const char* e = std::getenv("GCN2SPV_EXEC_KNOWN");
        return !(e && e[0] == '0');
    }();
    o.exec_known = exec_known;
    o.lane_from_scan = lane_from_scan(o.stage);
    if (h.stage == gcn::Stage::Pixel) {
        o.ps_input_ena = h.regs[8];
        o.descriptor_set = 1;
    } else if (h.stage == gcn::Stage::Vertex) {
        o.vs_out_cntl = h.regs[8];
    } else if (h.stage == gcn::Stage::Compute) {
        o.cs_threads[0] = h.regs[6];
        o.cs_threads[1] = h.regs[7];
        o.cs_threads[2] = h.regs[8];
    }
    return o;
}

bool load_bundle(const std::string& path, std::vector<gcn::BundleEntry>& entries) {
    std::vector<std::uint8_t> raw;
    if (!gcn::read_file(path, raw)) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        return false;
    }
    std::string err;
    const std::vector<std::uint8_t> b = gcn::dcx_decompress(raw, &err);
    if (b.empty() || !gcn::bnd4_entries(b, entries, &err)) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    return true;
}

std::string validate(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& pos, const char* message) {
        if (msg.empty()) msg = "word " + std::to_string(pos.index) + ": " + message;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);  // VK_KHR_maintenance8 lets sample ops take per-lane offsets
    if (!tools.Validate(spirv.data(), spirv.size(), vo)) {
        return msg.empty() ? "validation failed" : msg;
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    // GCN2SPV_HALF_NATIVE=1: translate as on a device that rounds f16 toward
    // zero itself (gcn/half.h native_half_rtz), to validate that form.
    if (const char* e = std::getenv("GCN2SPV_HALF_NATIVE"); e && e[0] == '1') gcn::set_native_half_rtz(true);
    if (argc < 2) {
        std::fprintf(stderr, "usage: gcn2spv --check <bundle.dcx>... | <bundle> <name> [out.spv]\n");
        return 2;
    }
    if (std::string(argv[1]) == "--tess-stubs") {
        // The generated tessellation and rect-list stages, validated.
        const float outer[4] = {1.0f, 1.0f, 1.0f, 1.0f}, inner[2] = {1.0f, 1.0f};
        const gcn::RectTessShaders rect = gcn::make_rect_tess_shaders({0, 1, 4}, 2);
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
        tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* msg) {
            std::fprintf(stderr, "  %s\n", msg);
        });
        int bad = 0;
        for (const auto& [name, words] : std::vector<std::pair<const char*, std::vector<std::uint32_t>>>{
                 {"tessellation vertex stage", gcn::make_tess_passthrough_vs()},
                 {"tessellation control stage", gcn::make_tess_constant_tcs(outer, inner, 1)},
                 {"tessellation control stage, attributes", gcn::make_tess_constant_tcs(outer, inner, 1, 8)},
                 {"rect list control stage", rect.control},
                 {"rect list evaluation stage", rect.evaluation}}) {
            const bool ok = tools.Validate(words);
            std::printf("%s: %zu words, %s\n", name, words.size(), ok ? "valid" : "INVALID");
            bad += !ok;
        }
        return bad ? 1 : 0;
    }
    if (std::string(argv[1]) == "--raw") {
        if (argc < 6) {
            std::fprintf(stderr, "usage: gcn2spv --raw <code.bin> <vs|ps|cs|ls|ds[:level]|hs|tes> <rsrc1> <rsrc2> [fetch.bin] [out.spv]\n");
            return 2;
        }
        auto words_of = [](const char* path, std::vector<std::uint32_t>& w) {
            std::vector<std::uint8_t> raw;
            if (!gcn::read_file(path, raw)) return false;
            w.resize(raw.size() / 4);
            std::memcpy(w.data(), raw.data(), w.size() * 4);
            return true;
        };
        std::vector<std::uint32_t> words, fetch_words;
        if (!words_of(argv[2], words)) {
            std::fprintf(stderr, "cannot read %s\n", argv[2]);
            return 1;
        }
        const std::string stage = argv[3];
        gcn::TranslateOptions o;
        o.rsrc1 = static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 16));
        o.rsrc2 = static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 16));
        if (stage == "ps") o.stage = gcn::Stage::Pixel, o.descriptor_set = 1;
        else if (stage == "cs") o.stage = gcn::Stage::Compute;
        else if (stage == "ls") o.stage = gcn::Stage::Compute, o.tess_role = gcn::TranslateOptions::TessRole::LsCompute;
        else if (stage == "lsv") {
            o.stage = gcn::Stage::Vertex;
            o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
            o.tess_lds_attributes = true;
        } else if (stage == "tesa") {
            o.stage = gcn::Stage::TessEval;
            o.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
            o.tess_lds_attributes = true;
        } else if (stage == "hs") o.stage = gcn::Stage::TessControl, o.tess_role = gcn::TranslateOptions::TessRole::HullTcs;
        else if (stage == "tes") o.stage = gcn::Stage::TessEval, o.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
        else if (stage.rfind("ds", 0) == 0) {
            o.stage = gcn::Stage::Vertex;
            o.tess_role = gcn::TranslateOptions::TessRole::Domain;
            o.domain_level = stage.size() > 3 ? static_cast<std::uint32_t>(std::atoi(stage.c_str() + 3)) : 1;
        } else o.stage = gcn::Stage::Vertex;
        o.lane_from_scan = lane_from_scan(o.stage);
        const char* out = nullptr;
        gcn::Program fetch;
        for (int a = 6; a < argc; ++a) {
            const std::string arg = argv[a];
            if (arg.rfind("tri:", 0) == 0) {
                // The game's own hull (the Forbidden Woods' meshes): a
                // triangle domain, three control points, a window a patch.
                o.tess_quads = false;
                o.tess_spacing = 1;
                o.tess_patch_control_points = 3;
                o.tess_window = static_cast<std::uint32_t>(std::strtoul(arg.c_str() + 4, nullptr, 16));
            } else if (arg.size() > 4 && arg.compare(arg.size() - 4, 4, ".spv") == 0) {
                out = argv[a];
            } else if (words_of(argv[a], fetch_words)) {
                fetch = gcn::decode(fetch_words.data(), fetch_words.size());
                o.fetch = &fetch;
            }
        }
        const gcn::Program p = gcn::decode(words.data(), words.size());
        const gcn::TranslateResult r = gcn::translate(p, o);
        std::printf("%s: %zu instructions, %zu words, %zu images, %zu errors\n", argv[2], p.insts.size(), r.spirv.size(), r.images.size(),
                    r.errors.size());
        for (const std::string& err : r.errors) std::printf("  error: %s\n", err.c_str());
        const std::string v = validate(r.spirv);
        std::printf("  %s\n", v.empty() ? "valid" : ("INVALID: " + v).c_str());
        if (out) {
            if (FILE* f = std::fopen(out, "wb")) {
                std::fwrite(r.spirv.data(), 4, r.spirv.size(), f);
                std::fclose(f);
            }
        }
        return r.ok() && v.empty() ? 0 : 1;
    }
    if (std::string(argv[1]) == "--check") {
        std::size_t shaders = 0, translated = 0, valid = 0;
        std::map<std::string, std::size_t> errors;
        std::map<std::string, std::size_t> val_errors;
        for (int a = 2; a < argc; ++a) {
            std::vector<gcn::BundleEntry> entries;
            if (!load_bundle(argv[a], entries)) return 1;
            for (const gcn::BundleEntry& e : entries) {
                gcn::ShaderCode code;
                Header h;
                if (!gcn::shader_code(e.data, code) || !read_header(e, code, h)) continue;
                ++shaders;
                const gcn::Program p = gcn::decode(code.words.data(), code.words.size());
                const gcn::TranslateResult r = gcn::translate(p, options_for(h));
                if (!r.ok()) {
                    for (const std::string& err : r.errors) {
                        const std::string key = err.substr(err.find(": ") + 2);
                        if (errors[key]++ == 0) std::printf("first: %s (%s)\n", err.c_str(), e.name.c_str());
                    }
                    continue;
                }
                ++translated;
                const std::string v = validate(r.spirv);
                if (!v.empty()) {
                    const std::string key = v.substr(0, 200);
                    if (val_errors[key]++ == 0) std::printf("first invalid: %s: %s\n", e.name.c_str(), v.c_str());
                    continue;
                }
                ++valid;
            }
        }
        std::printf("shaders=%zu translated=%zu valid=%zu\n", shaders, translated, valid);
        for (const auto& [k, n] : errors) std::printf("  translate: %-60s x%zu\n", k.c_str(), n);
        for (const auto& [k, n] : val_errors) std::printf("  validate:  %-60s x%zu\n", k.c_str(), n);
        return valid == shaders ? 0 : 1;
    }
    std::vector<gcn::BundleEntry> entries;
    if (!load_bundle(argv[1], entries)) return 1;
    const std::string needle = argc > 2 ? argv[2] : "";
    const char* out = argc > 3 ? argv[3] : nullptr;
    for (const gcn::BundleEntry& e : entries) {
        if (e.name.find(needle) == std::string::npos) continue;
        gcn::ShaderCode code;
        Header h;
        if (!gcn::shader_code(e.data, code) || !read_header(e, code, h)) continue;
        const gcn::Program p = gcn::decode(code.words.data(), code.words.size());
        const gcn::TranslateResult r = gcn::translate(p, options_for(h));
        std::printf("%s: %zu words, %zu images, %zu samplers, %zu errors\n", e.name.c_str(), r.spirv.size(),
                    r.images.size(), r.samplers.size(), r.errors.size());
        for (const gcn::ImageBinding& b : r.images) {
            std::printf("  image binding %u: %s (dim %u%s)\n", b.binding, b.path.str().c_str(), b.dim, b.arrayed ? " array" : "");
        }
        for (const gcn::SamplerBinding& b : r.samplers) std::printf("  sampler binding %u: %s\n", b.binding, b.path.str().c_str());
        for (const gcn::BufferBinding& b : r.buffers) {
            std::printf("  buffer binding %u: %s (%s, %u dwords)\n", b.binding, b.path.str().c_str(),
                        b.pointer ? "table" : "constant buffer", b.max_dw);
        }
        for (const std::string& err : r.errors) std::printf("  error: %s\n", err.c_str());
        const std::string v = validate(r.spirv);
        std::printf("  %s\n", v.empty() ? "valid" : ("INVALID: " + v).c_str());
        if (out) {
            FILE* f = std::fopen(out, "wb");
            if (f) {
                std::fwrite(r.spirv.data(), 4, r.spirv.size(), f);
                std::fclose(f);
            }
            out = nullptr;
        }
    }
    return 0;
}
