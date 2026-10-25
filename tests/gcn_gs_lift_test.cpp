// The geometry-shader lifts (gcn/lift.h GsLift) on the game's own programs,
// from the shipped effects bundle: each GPU particle system's export shader
// (VS_VFX_*, container type 5) as the pipeline's vertex stage and its geometry
// shader (GS_VFX_*, type 3, with the copy shader it carries) as the geometry
// stage. Each lifts to valid SPIR-V, the copy shader's map is the one its
// exports make, and what the lift cannot show is refused; each lift is printed
// as HLSL and - where DXC is installed (host/dxc.h) - compiled, with the
// lift's interface and operations. Skips when the dump is absent (BBHOST_APP0).
#include "test_app0.h"
#include "gcn/container.h"
#include "gcn/gs_vertex.h"
#include "gcn/hlsl.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/translate.h"
#include "host/dxc.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

gpu::Dxc g_dxc;
bool g_have_dxc = false;
int g_compiled = 0;

bool valid_spirv(const std::vector<std::uint32_t>& spirv, const std::string& what) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    const bool ok = tools.Validate(spirv.data(), spirv.size(), vo);
    if (!ok) std::fprintf(stderr, "%s: lifted SPIR-V invalid: %s\n", what.c_str(), msg.c_str());
    return ok;
}

bool lifted(const gcn::LiftResult& r, const std::string& what) {
    if (!r.ok()) {
        std::fprintf(stderr, "%s: not lifted: %s\n", what.c_str(), r.rejections.empty() ? "(no SPIR-V)" : r.rejections[0].c_str());
        return false;
    }
    // GS_LIFT_TEST_DUMP=<dir>: each lifted module and its listing, for spirv-dis.
    if (const char* dir = std::getenv("GS_LIFT_TEST_DUMP"); dir && *dir) {
        std::string base = std::string(dir) + "/" + what.substr(what.rfind(' ') + 1);
        if (FILE* f = std::fopen((base + ".spv").c_str(), "wb")) {
            std::fwrite(r.spirv.data(), 4, r.spirv.size(), f);
            std::fclose(f);
        }
        if (FILE* f = std::fopen((base + ".txt").c_str(), "wb")) {
            std::fwrite(r.listing.data(), 1, r.listing.size(), f);
            std::fclose(f);
        }
    }
    return valid_spirv(r.spirv, what);
}

// The lift as HLSL and, with DXC, compiled: valid, with the lift's interface and operations.
void through_hlsl(const gcn::LiftResult& lift, const std::string& what) {
    if (!lift.ok()) return;
    gcn::HlslOptions o;
    o.title = what;
    const gcn::HlslResult h = gcn::emit_hlsl(lift.spirv, o);
    CHECK(h.ok());
    if (!h.ok()) {
        std::fprintf(stderr, "%s: not printed: %s\n", what.c_str(), h.errors.empty() ? "?" : h.errors[0].c_str());
        return;
    }
    if (const char* dir = std::getenv("GS_LIFT_TEST_DUMP"); dir && *dir) {
        if (FILE* f = std::fopen((std::string(dir) + "/" + what.substr(what.rfind(' ') + 1) + ".hlsl").c_str(), "wb")) {
            std::fwrite(h.source.data(), 1, h.source.size(), f);
            std::fclose(f);
        }
    }
    if (!g_have_dxc) return;
    std::vector<std::uint32_t> spirv;
    std::string error;
    const bool compiled = g_dxc.compile(h, gpu::DxcOptions{}, spirv, error);
    CHECK(compiled);
    if (!compiled) {
        std::fprintf(stderr, "%s: DXC: %s\n", what.c_str(), error.c_str());
        return;
    }
    CHECK(valid_spirv(spirv, what + " (DXC)"));
    const std::string iface = gcn::hlsl_interface_diff(lift.spirv, spirv), ops = gcn::hlsl_operation_diff(lift.spirv, spirv);
    CHECK(iface.empty());
    CHECK(ops.empty());
    if (!iface.empty()) std::fprintf(stderr, "%s: interface: %s\n", what.c_str(), iface.c_str());
    if (!ops.empty()) std::fprintf(stderr, "%s: operations: %s\n", what.c_str(), ops.c_str());
    ++g_compiled;
}

bool rejected_with(const gcn::LiftResult& r, const char* text) {
    for (const std::string& why : r.rejections) {
        if (why.find(text) != std::string::npos) return true;
    }
    std::fprintf(stderr, "expected a rejection containing \"%s\"; got %zu rejection(s)%s%s\n", text, r.rejections.size(),
                 r.rejections.empty() ? "" : ": ", r.rejections.empty() ? "" : r.rejections[0].c_str());
    return false;
}

// One program of the bundle: its code, its register block (16 dwords after
// "Shdr" - the Gnm shader structure the GX objects point at) and, for a
// geometry shader, the copy shader its container carries after its own
// footer: from the first 'OrbShdr' + 0x1c to the second.
struct Shader {
    std::string name;
    gcn::ShaderCode code;
    gcn::Program program;
    std::uint32_t regs[16] = {};
    std::vector<std::uint32_t> copy_words;
    gcn::Program copy;
};

bool load(const std::string& bundle, std::vector<Shader>& out) {
    std::vector<std::uint8_t> raw;
    if (!gcn::read_file(bundle, raw)) return false;
    std::string err;
    const std::vector<std::uint8_t> b = gcn::dcx_decompress(raw, &err);
    std::vector<gcn::BundleEntry> entries;
    if (b.empty() || !gcn::bnd4_entries(b, entries, &err)) {
        std::fprintf(stderr, "%s: %s\n", bundle.c_str(), err.c_str());
        return false;
    }
    for (const gcn::BundleEntry& e : entries) {
        Shader s;
        s.name = e.name;
        if (!gcn::shader_code(e.data, s.code)) continue;
        std::size_t shdr = 0;
        while (shdr + 0x60 <= e.data.size() && shdr < 0x100 && std::memcmp(e.data.data() + shdr, "Shdr", 4) != 0) ++shdr;
        if (shdr + 0x60 > e.data.size()) continue;
        std::memcpy(s.regs, e.data.data() + shdr + 16, sizeof(s.regs));
        s.program = gcn::decode(s.code.words.data(), s.code.words.size());
        if (s.code.type == 3) {
            std::size_t second = 0;
            for (std::size_t at = s.code.footer + 7; at + 7 <= e.data.size(); ++at) {
                if (std::memcmp(e.data.data() + at, "OrbShdr", 7) == 0) {
                    second = at;
                    break;
                }
            }
            const std::size_t first = s.code.footer + 0x1c;
            if (second > first && (second - first) % 4 == 0) {
                s.copy_words.resize((second - first) / 4);
                std::memcpy(s.copy_words.data(), e.data.data() + first, second - first);
                s.copy = gcn::decode(s.copy_words.data(), s.copy_words.size());
            }
        }
        out.push_back(std::move(s));
    }
    return true;
}

const Shader* find(const std::vector<Shader>& all, const std::string& name) {
    for (const Shader& s : all) {
        if (s.name.find(name) != std::string::npos) return &s;
    }
    std::fprintf(stderr, "%s not in the bundle\n", name.c_str());
    return nullptr;
}

// The options a GX draw's stage gets (render.cpp): constant buffers through
// storage-buffer bindings with no page-table fallback, set 2, outputs linked.
gcn::TranslateOptions stage_options(const Shader& s, gcn::TranslateOptions::GsRole role) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.gs_role = role;
    o.rsrc1 = s.regs[4];
    o.rsrc2 = s.regs[5];
    o.cb_descriptor_set = 2;
    o.cb_ssbo = true;
    o.link_outputs = true;
    for (std::uint8_t k = 0; k < 32; ++k) o.output_links.push_back(k);
    const gcn::TranslateResult paths = gcn::translate(s.program, o);
    for (const gcn::PredictedImage& pi : gcn::predict_image_dims(s.program, paths)) o.image_dims.emplace_back(pi.dim, pi.arrayed);
    o.sampler_force_unnormalized.assign(paths.samplers.size(), false);
    o.cb_no_fallback = true;
    return o;
}

void report_reference(const char* what, const Shader& s, const gcn::TranslateResult& ref) {
    std::printf("%s %s: reference %s, %zu buffers, %zu images, %zu samplers, %zu params\n", what, s.name.c_str(), ref.ok() ? "ok" : "failed",
                ref.buffers.size(), ref.images.size(), ref.samplers.size(), ref.vs_params.size());
    for (const std::string& e : ref.errors) std::printf("    error: %s\n", e.c_str());
    for (const gcn::BufferBinding& b : ref.buffers)
        std::printf("    buffer %u: %s%s%s%s%s, %u dwords\n", b.binding, b.path.str().c_str(), b.pointer ? " pointer" : "", b.indexed ? " indexed" : "",
                    b.ranged ? " ranged" : "", b.written ? " written" : "", b.max_dw);
}

void splice(const Shader& gs, const gcn::LiftResult& el, const gcn::LiftResult& glr, bool particle);

// ---- the particle systems: an ES and a GS each ----
void particles(const std::vector<Shader>& all) {
    struct Pair {
        const char* es;
        const char* gs;
    };
    const Pair pairs[] = {{"VS_VFX_StandardParticle.vpo", "GS_VFX_StandardParticle.gpo"},
                          {"VS_VFX_SparkParticle.vpo", "GS_VFX_SparkParticle.gpo"},
                          {"VS_VFX_WanderingVision.vpo", "GS_VFX_WanderingVision.gpo"}};
    for (const Pair& p : pairs) {
        const Shader* es = find(all, p.es);
        const Shader* gs = find(all, p.gs);
        CHECK(es && gs);
        if (!es || !gs) continue;
        CHECK(es->code.type == 5 && gs->code.type == 3);
        CHECK(!gs->copy.insts.empty());
        const gcn::TranslateOptions eo = stage_options(*es, gcn::TranslateOptions::GsRole::EsVertex);
        const gcn::TranslateResult eref = gcn::translate(es->program, eo);
        report_reference("ES", *es, eref);
        const gcn::TranslateOptions go = stage_options(*gs, gcn::TranslateOptions::GsRole::Geometry);
        const gcn::TranslateResult gref = gcn::translate(gs->program, go);
        report_reference("GS", *gs, gref);
        // The Gnm structures' words: the ES's ring vertex size (7), the GS's
        // output primitive (7), GSVS vertex size (9) and most vertices (13).
        gcn::GsLift gl;
        gl.es_dwords = es->regs[7];
        gl.max_vertices = gs->regs[13];
        gl.vertex_dwords = gs->regs[9];
        gl.out_prim = gs->regs[7];
        std::string why;
        const bool copy_ok = gcn::parse_gs_copy(gs->copy, gl.max_vertices, gl.copy, why);
        if (!copy_ok) std::fprintf(stderr, "%s: copy shader not parsed: %s\n", gs->name.c_str(), why.c_str());
        CHECK(copy_ok);
        std::string map;
        for (const gcn::GsCopyOut& c : gl.copy) {
            char one[48];
            std::snprintf(one, sizeof(one), " %s%u.%c=%s%u", c.target == 12 ? "pos" : "param", c.target == 12 ? 0u : c.target - 32u, "xyzw"[c.comp],
                          c.constant ? "#" : "d", c.value);
            map += one;
        }
        std::printf("  copy (%u vertices, %u ES dwords):%s\n", gl.max_vertices, gl.es_dwords, map.c_str());
        const gcn::LiftResult el = gcn::lift_es_shader(es->program, eo, eref, gl);
        if (lifted(el, "ES " + es->name)) {
            for (const std::string& line : el.proof) std::printf("    %s\n", line.c_str());
        }
        CHECK(el.ok());
        through_hlsl(el, "ES " + es->name);
        const gcn::LiftResult glr = gcn::lift_gs_shader(gs->program, go, gref, gl);
        if (lifted(glr, "GS " + gs->name)) {
            for (const std::string& line : glr.proof) std::printf("    %s\n", line.c_str());
        }
        CHECK(glr.ok());
        through_hlsl(glr, "GS " + gs->name);
        // The two as one vertex stage (gcn/gs_vertex.h): valid, a particle's
        // vertices one strip for the standard particle, ten quads for the
        // spark's trail. The particles' pairs must splice; the others are
        // reported.
        const bool particle = std::strstr(p.gs, "StandardParticle") || std::strstr(p.gs, "SparkParticle");
        if (el.ok() && glr.ok()) splice(*gs, el, glr, particle);
    }
}

void dump_words(const std::vector<std::uint32_t>& w, const std::string& name) {
    if (const char* dir = std::getenv("GS_LIFT_TEST_DUMP"); dir && *dir) {
        if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".spv").c_str(), "wb")) {
            std::fwrite(w.data(), 4, w.size(), f);
            std::fclose(f);
        }
    }
}

// The GS lift with its emit instructions cut to `keep` of them (the rest
// become OpNop): a mutant whose particles' vertices are no longer whole quads.
std::vector<std::uint32_t> without_emits(const std::vector<std::uint32_t>& spirv, int drop) {
    std::vector<std::uint32_t> w = spirv;
    for (std::size_t i = 5; i < w.size() && drop > 0;) {
        const std::uint32_t n = w[i] >> 16;
        if (!n) break;
        if ((w[i] & 0xffff) == 218 /* OpEmitVertex */) {
            w[i] = (1u << 16) | 0 /* OpNop */;
            --drop;
        }
        i += n;
    }
    return w;
}

void splice(const Shader& gs, const gcn::LiftResult& el, const gcn::LiftResult& glr, bool particle) {
    const gcn::GsVertexStage vs = gcn::gs_vertex_stage(el.spirv, glr.spirv);
    if (!vs.ok()) std::fprintf(stderr, "%s: not as a vertex stage: %s\n", gs.name.c_str(), vs.why.c_str());
    if (particle) CHECK(vs.ok());
    if (!vs.ok()) return;
    std::printf("  as a vertex stage: %u vertices a particle as %s (%u emits on its longest path), %zu words\n", vs.vertices,
                vs.strip ? "one strip" : "quads", vs.path_emits, vs.spirv.size());
    CHECK(valid_spirv(vs.spirv, "VS " + gs.name));
    dump_words(vs.spirv, "vs-" + gs.name);
    // The standard particle: path A's emits, then the common ones, one cut at
    // the end - one strip of up to eight. The spark: ten strips of four.
    if (std::strstr(gs.name.c_str(), "StandardParticle")) CHECK(vs.vertices == 8 && vs.strip);
    if (std::strstr(gs.name.c_str(), "SparkParticle")) CHECK(vs.vertices == 40 && !vs.strip);
    // A GS one emit short of whole quads on its every path is refused.
    if (particle) {
        const gcn::GsVertexStage cut = gcn::gs_vertex_stage(el.spirv, without_emits(glr.spirv, 1));
        if (cut.ok() && cut.vertices % 4 == 0 && cut.vertices == vs.vertices && std::strstr(gs.name.c_str(), "SparkParticle"))
            std::fprintf(stderr, "%s: a GS one emit short still splices as %u vertices\n", gs.name.c_str(), cut.vertices);
        if (std::strstr(gs.name.c_str(), "SparkParticle")) CHECK(!cut.ok());
        const gcn::GsVertexStage none = gcn::gs_vertex_stage(el.spirv, without_emits(glr.spirv, 1000));
        CHECK(!none.ok());
    }
}

// ---- GS_LIFT_TEST_ALL=1: every geometry shader in the bundles ----
// Each with the export shader of its name (.gpo -> .vpo; the effects' shadow
// and array variants with the ES GX binds them with), lifted as the renderer
// would; the counts by result, for docs/rendering.md. Not a check: a census.
void census(const std::vector<Shader>& all, const char* bundle) {
    std::map<std::string, int> results;
    int lifted_gs = 0, total = 0;
    for (const Shader& gs : all) {
        if (gs.code.type != 3) continue;
        ++total;
        std::string es_name = gs.name.substr(0, gs.name.size() - 4) + ".vpo";
        if (gs.name.find("GS_VFX_") != std::string::npos) {
            es_name = gs.name.find("StandardParticle") != std::string::npos ? "VS_VFX_StandardParticle.vpo"
                      : gs.name.find("GsArray") != std::string::npos    ? "VS_VFX_Grid.vpo"
                                                                         : "VS_VFX_" + gs.name.substr(gs.name.find("GS_VFX_") + 7, std::string::npos);
            es_name = es_name.substr(0, es_name.rfind('.')) + ".vpo";
        }
        const Shader* es = nullptr;
        for (const Shader& s : all) {
            if (s.code.type == 5 && s.name.size() >= es_name.size() && s.name.compare(s.name.size() - es_name.size(), es_name.size(), es_name) == 0) {
                es = &s;
                break;
            }
        }
        if (!es) {
            ++results["no export shader of its name"];
            continue;
        }
        gcn::GsLift gl;
        gl.es_dwords = es->regs[7];
        gl.max_vertices = gs.regs[13];
        gl.vertex_dwords = gs.regs[9];
        gl.out_prim = gs.regs[7];
        std::string why;
        if (!gcn::parse_gs_copy(gs.copy, gl.max_vertices, gl.copy, why)) {
            ++results["copy shader: " + why.substr(0, 90)];
            continue;
        }
        const gcn::TranslateOptions go = stage_options(gs, gcn::TranslateOptions::GsRole::Geometry);
        const gcn::LiftResult glr = gcn::lift_gs_shader(gs.program, go, gcn::translate(gs.program, go), gl);
        if (!glr.ok()) {
            const std::string r = glr.rejections.empty() ? "(no SPIR-V)" : glr.rejections[0];
            const std::size_t colon = r.find(": ");
            ++results["GS: " + (colon != std::string::npos && colon < 12 ? r.substr(colon + 2) : r).substr(0, 90)];
            continue;
        }
        const gcn::TranslateOptions eo = stage_options(*es, gcn::TranslateOptions::GsRole::EsVertex);
        const gcn::LiftResult el = gcn::lift_es_shader(es->program, eo, gcn::translate(es->program, eo), gl);
        if (!el.ok()) {
            ++results["ES: " + (el.rejections.empty() ? std::string("(no SPIR-V)") : el.rejections[0].substr(0, 90))];
            continue;
        }
        ++lifted_gs;
        ++results["lifted (ES and GS)"];
    }
    std::printf("%s: %d of %d geometry shaders lifted with their export shaders\n", bundle, lifted_gs, total);
    for (const auto& [k, n] : results) std::printf("  %5d  %s\n", n, k.c_str());
}

}  // namespace

int main() {
    const std::string bundle = test_app0_file("dvdroot_ps4/shader/gxffxshader.shaderbnd.dcx");
    std::vector<Shader> all;
    if (bundle.empty() || !load(bundle, all)) {
        std::printf("skipped: no dump (BBHOST_APP0)\n");
        return kTestSkip;
    }
    g_have_dxc = g_dxc.load();
    if (!g_have_dxc) std::printf("DXC not found: the HLSL is printed, not compiled\n");
    particles(all);
    if (const char* e = std::getenv("GS_LIFT_TEST_ALL"); e && e[0] == '1') {
        census(all, "gxffxshader");
        std::vector<Shader> flver;
        const std::string fb = test_app0_file("dvdroot_ps4/shader/gxflvershader.shaderbnd.dcx");
        if (!fb.empty() && load(fb, flver)) census(flver, "gxflvershader");
    }
    std::printf("%d lifts through DXC\n", g_compiled);
    if (g_failures) std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return g_failures ? 1 : 0;
}
