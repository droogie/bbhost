// The tessellation lifts (gcn/lift.h TessLift) on the game's own programs,
// from the shipped shader bundles: the particles' LS and domain shaders with
// their control point as stage attributes and their constant-factor hulls,
// and the Forbidden Woods' DOL PN meshes - LS, hull and domain shader - in the
// window a patch's LDS is laid out in, as a native hull pipeline runs them.
// Each lifts to valid SPIR-V, and what the lift cannot show is refused.
// Skips when the dump is absent (BBHOST_APP0).
#include "test_app0.h"
#include "gcn/container.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/tess_vertex.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstring>
#include <set>
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
    return valid_spirv(r.spirv, what);
}

bool rejected_with(const gcn::LiftResult& r, const char* text) {
    for (const std::string& why : r.rejections) {
        if (why.find(text) != std::string::npos) return true;
    }
    std::fprintf(stderr, "expected a rejection containing \"%s\"; got %zu rejection(s)%s%s\n", text, r.rejections.size(),
                 r.rejections.empty() ? "" : ": ", r.rejections.empty() ? "" : r.rejections[0].c_str());
    return false;
}

bool proof_has(const gcn::LiftResult& r, const char* text) {
    for (const std::string& p : r.proof) {
        if (p.find(text) != std::string::npos) return true;
    }
    std::fprintf(stderr, "expected a proof line containing \"%s\"\n", text);
    return false;
}

// One program of a bundle, with the register block and input usage slots its
// container keeps after "Shdr" (the Gnm shader structure the GX objects point
// at): words 4-5 its RSRC1/2; the LS's word 7 its LDS stride; the hull's
// words 9-12 and 15 its control points in and out, patch constants,
// control-point stride and first edge factor.
struct Shader {
    std::string name;
    gcn::ShaderCode code;
    gcn::Program program;
    std::uint32_t regs[16] = {};
    std::vector<std::uint32_t> slots;  // input usage slots: type, API slot, first user SGPR, flags (a byte each)
    std::vector<std::uint8_t> data;
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
        s.data = e.data;
        s.program = gcn::decode(s.code.words.data(), s.code.words.size());
        out.push_back(std::move(s));
    }
    return true;
}

const Shader* find(const std::vector<Shader>& all, const char* name) {
    for (const Shader& s : all) {
        if (s.name.find(name) != std::string::npos) return &s;
    }
    std::fprintf(stderr, "%s not in the bundle\n", name);
    return nullptr;
}

// Where the register block starts: 16 bytes past "Shdr".
std::size_t block(const Shader& s) {
    std::size_t shdr = 0;
    while (shdr + 0x60 <= s.data.size() && shdr < 0x100 && std::memcmp(s.data.data() + shdr, "Shdr", 4) != 0) ++shdr;
    return shdr + 16;
}

// The constant buffer a stage reads the tessellation constants through: the
// input usage slot of an immediate constant buffer (type 2) at API slot 0x13,
// its first user SGPR - or, past the sixteenth, its place in the extended user
// data table whose pointer a slot of type 0x1b holds. The slots follow the
// stage's registers, which are fewer for a vertex-slot stage than for a hull.
std::string constants_path(const Shader& s) {
    std::uint32_t reg = ~0u, table = ~0u;
    for (std::size_t at = block(s) + 24; at + 4 <= s.code.offset && at + 4 <= s.data.size(); at += 4) {
        std::uint32_t w;
        std::memcpy(&w, s.data.data() + at, 4);
        if ((w & 0xff) == 0x02 && ((w >> 8) & 0xff) == 0x13 && reg == ~0u) reg = (w >> 16) & 0xff;
        if ((w & 0xff) == 0x1b && table == ~0u) table = (w >> 16) & 0xff;
    }
    if (reg < 16) return "user_sgpr[" + std::to_string(reg) + "]";
    if (reg != ~0u && table < 16) return "user_sgpr[" + std::to_string(table) + "]->load(+" + std::to_string((reg - 16) * 4) + ")";
    return std::string();
}

// An LS's vertex inputs from its container's input table, after its usage
// slots: entries of (semantic, first VGPR, count, 0), semantics 0, 1, 2...,
// taken as 32-bit floats.
std::vector<gcn::VertexElement> ls_inputs(const Shader& s) {
    for (std::size_t first = block(s) + 24; first + 4 <= s.code.offset; first += 4) {
        std::vector<gcn::VertexElement> els;
        for (std::size_t at = first; at + 4 <= s.code.offset && els.size() < 16; at += 4) {
            const std::uint8_t sem = s.data[at], vgpr = s.data[at + 1], cnt = s.data[at + 2], pad = s.data[at + 3];
            if (sem != els.size() || !cnt || cnt > 4 || vgpr < 4 || pad) break;
            gcn::VertexElement el;
            el.location = static_cast<std::uint32_t>(els.size());
            el.vdata = vgpr;
            el.count = cnt;
            const std::uint32_t dfmt = cnt == 1 ? 4 : cnt == 2 ? 11 : cnt == 3 ? 13 : 14;
            el.w3 = dfmt << 15 | 7u << 12 | 0xfacu;
            els.push_back(el);
        }
        if (!els.empty()) return els;
    }
    return {};
}

bool reads_lds(const gcn::Program& p) {
    for (const gcn::Inst& in : p.insts) {
        if (in.enc == gcn::Enc::DS && in.op != 53) return true;  // not ds_swizzle_b32
    }
    return false;
}

// ---- the particles: one control point a patch, carried as stage attributes ----
void particles(const std::vector<Shader>& all) {
    std::set<std::vector<std::uint32_t>> seen;
    int ls_n = 0, ls_ok = 0, ds_n = 0, ds_ok = 0, hs_n = 0, hs_ok = 0;
    for (const Shader& s : all) {
        if (s.name.find("tessellate") == std::string::npos || !seen.insert(s.code.words).second) continue;
        if (s.code.type == 7) {
            ++hs_n;
            const gcn::HullLift h = gcn::lift_constant_hull(s.program);
            if (h.ok()) {
                bool positive = true;
                for (float f : h.factors) positive = positive && f > 0.0f;
                CHECK(positive);
                ++hs_ok;
            } else {
                std::fprintf(stderr, "HS %s: %s\n", s.name.c_str(), h.rejections[0].c_str());
            }
            continue;
        }
        if (s.code.type == 6) {
            const std::vector<gcn::VertexElement> els = ls_inputs(s);
            if (els.empty()) continue;  // an input table this reading does not follow
            ++ls_n;
            gcn::TranslateOptions o;
            o.stage = gcn::Stage::Vertex;
            o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
            o.tess_lds_attributes = true;
            o.tess_attr_vec4s = 16;
            o.rsrc1 = s.regs[4];
            o.rsrc2 = s.regs[5];
            o.vertex_input = els;
            o.cb_ssbo = true;
            o.cb_no_fallback = true;
            const gcn::TranslateResult ref = gcn::translate(s.program, o);
            gcn::TessLift tl;
            tl.attr_vec4s = 16;
            if (ref.ok() && lifted(gcn::lift_ls_shader(s.program, o, ref, tl), "LS " + s.name)) ++ls_ok;
            continue;
        }
        if (s.code.type != 1 || !reads_lds(s.program)) continue;
        ++ds_n;
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::TessEval;
        o.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
        o.tess_lds_attributes = true;
        o.tess_attr_vec4s = 16;
        o.tess_patch_control_points = 1;
        o.rsrc1 = s.regs[4];
        o.rsrc2 = s.regs[5];
        o.vs_out_cntl = s.regs[8];
        o.cb_descriptor_set = 2;
        o.cb_ssbo = true;
        o.link_outputs = true;
        for (std::uint8_t k = 0; k < 32; ++k) o.output_links.push_back(k);
        const gcn::TranslateResult paths = gcn::translate(s.program, o);
        for (const gcn::PredictedImage& pi : gcn::predict_image_dims(s.program, paths)) o.image_dims.emplace_back(pi.dim, pi.arrayed);
        o.sampler_force_unnormalized.assign(paths.samplers.size(), false);
        o.cb_no_fallback = true;
        const gcn::TranslateResult ref = gcn::translate(s.program, o);
        gcn::TessLift tl;
        tl.attr_vec4s = 16;
        const gcn::LiftResult l = ref.ok() ? gcn::lift_domain_shader(s.program, o, ref, tl) : gcn::LiftResult{};
        if (ref.ok() && lifted(l, "DS " + s.name)) {
            ++ds_ok;
            CHECK(proof_has(l, "LDS"));
        }
        // With the page-table fallback the lift has nothing to declare its bindings by.
        if (ds_n == 1 && ref.ok()) {
            gcn::TranslateOptions fb = o;
            fb.cb_no_fallback = false;
            CHECK(!gcn::lift_domain_shader(s.program, fb, gcn::translate(s.program, fb), tl).ok());
        }
    }
    std::printf("particles: LS %d/%d, domain shaders %d/%d, constant hulls %d/%d lifted\n", ls_ok, ls_n, ds_ok, ds_n, hs_ok, hs_n);
    CHECK(ls_n >= 6 && ls_ok == ls_n);
    CHECK(ds_n >= 16 && ds_ok == ds_n);
    CHECK(hs_n >= 4 && hs_ok == hs_n);
}

// ---- the particles as one vertex stage (gcn/tess_vertex.h) ----
// Where tessellation runs through compute, every LS x domain shader pair,
// translated as the host does, must fuse and validate at both hull levels.
void fused_particles(const std::vector<Shader>& all) {
    std::set<std::vector<std::uint32_t>> seen;
    std::vector<gcn::TranslateResult> ls_t, ds_t;
    for (const Shader& s : all) {
        if (s.name.find("tessellate") == std::string::npos || !seen.insert(s.code.words).second) continue;
        if (s.code.type == 6) {
            gcn::TranslateOptions o;
            o.stage = gcn::Stage::Vertex;
            o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
            o.tess_lds_attributes = true;
            o.tess_attr_vec4s = 16;
            o.rsrc1 = s.regs[4];
            o.rsrc2 = s.regs[5];
            o.descriptor_set = 0;
            gcn::TranslateResult t = gcn::translate(s.program, o);
            CHECK(t.ok());
            if (t.ok()) ls_t.push_back(std::move(t));
            continue;
        }
        if (s.code.type != 1 || !reads_lds(s.program)) continue;
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::TessEval;
        o.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
        o.tess_lds_attributes = true;
        o.tess_attr_vec4s = 16;
        o.tess_patch_control_points = 1;
        o.rsrc1 = s.regs[4];
        o.rsrc2 = s.regs[5];
        o.vs_out_cntl = s.regs[8];
        o.descriptor_set = 0;
        o.cb_descriptor_set = 2;
        o.cb_ssbo = true;
        o.params_vertex_formats = true;
        gcn::TranslateResult t = gcn::translate(s.program, o);
        CHECK(t.ok());
        if (t.ok()) ds_t.push_back(std::move(t));
    }
    int n = 0, ok = 0;
    for (const gcn::TranslateResult& d : ds_t) {
        for (const gcn::TranslateResult& l : ls_t) {
            for (std::uint32_t level : {1u, 3u}) {
                ++n;
                const gcn::TessVertexFuse f = gcn::fuse_tess_vertex(l.spirv, d.spirv, level, true);
                if (!f.ok()) {
                    std::fprintf(stderr, "fused particle stage: %s\n", f.error.c_str());
                    continue;
                }
                if (valid_spirv(f.spirv, "fused particle stage")) ++ok;
            }
        }
    }
    std::printf("particles as one vertex stage: %d/%d LS and domain shader pairs fused and valid (%zu LS, %zu domain shaders)\n", ok, n,
                ls_t.size(), ds_t.size());
    CHECK(!ls_t.empty() && !ds_t.empty());
    CHECK(ok == n);
}

// ---- the game's own hulls: a patch of control points in its LDS window ----
// One program set - LS, hull, domain shader - as a native hull pipeline runs
// it (render.cpp kTessNativeSet): the window from the containers (the layout
// gx_hull_layout reads from the GX objects at a draw), the LS on vertex input
// with its params and buffers in set 4 from 16, the hull as the control stage
// with its own in set 4 from 32, the domain shader as the evaluation stage.
struct Native {
    std::uint32_t window = 0;
    gcn::TessLift tl;
    gcn::TranslateOptions lo, ho, dop;
    gcn::TranslateResult lref, href, dref;
    gcn::TessLift ll, hl, dl;
    gcn::LiftResult ls, hs, ds;
};

Native lift_native(const Shader& ls, const Shader& hs, const Shader& ds) {
    Native n;
    const std::uint32_t ls_stride = ls.regs[7], cp = hs.regs[9], npc = hs.regs[11], cp_stride = hs.regs[12];
    const std::uint32_t inputs = cp * ls_stride, outputs = cp * cp_stride, consts = npc * 16;
    const std::uint32_t tc[12] = {ls_stride, cp_stride, 1, inputs, consts, inputs + outputs, outputs, 0, hs.regs[15], 0, 0, 0};
    n.window = (inputs + outputs + consts + 15) & ~15u;
    n.tl.window = true;
    n.tl.control_points = cp;
    n.tl.ls_stride = ls_stride;
    n.tl.cp_stride = cp_stride;
    n.tl.patch_constants = npc;
    std::memcpy(n.tl.constants, tc, sizeof(tc));
    // VGT_TF_PARAM (register word 6): the domain, its partitioning and winding.
    const std::uint32_t tf = hs.regs[6];
    const bool quads = (tf & 3) == 2;
    const int spacing = ((tf >> 2) & 7) == 2 ? 1 : ((tf >> 2) & 7) == 3 ? 2 : 0;
    const bool cw = ((tf >> 5) & 7) != 3;

    n.ho.stage = gcn::Stage::TessControl;
    n.ho.tess_role = gcn::TranslateOptions::TessRole::HullTcs;
    n.ho.rsrc1 = hs.regs[4];
    n.ho.rsrc2 = hs.regs[5];
    n.ho.tess_patch_control_points = cp;
    n.ho.tess_window = n.window;
    n.ho.tess_lds_bound = true;
    n.ho.tess_quads = quads;
    n.ho.tess_spacing = spacing;
    n.ho.tess_cw = cw;
    n.ho.descriptor_set = 4;
    n.ho.cb_ssbo = true;
    n.ho.cb_no_fallback = true;
    n.ho.cb_descriptor_set = 4;
    n.ho.cb_binding_base = 32;
    n.href = gcn::translate(hs.program, n.ho);
    n.hl = n.tl;
    n.hl.own_params = true;
    n.hl.params_binding = 1;
    n.hl.constants_path = constants_path(hs);
    if (n.href.ok()) n.hs = gcn::lift_hull_shader(hs.program, n.ho, n.href, n.hl);

    n.dop.stage = gcn::Stage::TessEval;
    n.dop.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
    n.dop.rsrc1 = ds.regs[4];
    n.dop.rsrc2 = ds.regs[5];
    n.dop.vs_out_cntl = ds.regs[8];
    n.dop.tess_patch_control_points = cp;
    n.dop.tess_window = n.window;
    n.dop.tess_lds_bound = true;
    n.dop.tess_quads = quads;
    n.dop.tess_spacing = spacing;
    n.dop.tess_cw = cw;
    n.dop.cb_descriptor_set = 2;
    n.dop.cb_ssbo = true;
    n.dop.link_outputs = true;
    for (std::uint8_t k = 0; k < 32; ++k) n.dop.output_links.push_back(k);
    const gcn::TranslateResult dpaths = gcn::translate(ds.program, n.dop);
    for (const gcn::PredictedImage& pi : gcn::predict_image_dims(ds.program, dpaths)) n.dop.image_dims.emplace_back(pi.dim, pi.arrayed);
    n.dop.sampler_force_unnormalized.assign(dpaths.samplers.size(), false);
    n.dop.cb_no_fallback = true;
    n.dref = gcn::translate(ds.program, n.dop);
    n.dl = n.tl;
    n.dl.constants_path = constants_path(ds);
    if (n.dref.ok()) n.ds = gcn::lift_domain_shader(ds.program, n.dop, n.dref, n.dl);

    n.lo.stage = gcn::Stage::Vertex;
    n.lo.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
    n.lo.tess_lds_attributes = true;
    n.lo.tess_attr_vec4s = ls_stride / 16;
    n.lo.rsrc1 = ls.regs[4];
    n.lo.rsrc2 = ls.regs[5];
    n.lo.vertex_input = ls_inputs(ls);
    n.lo.cb_ssbo = true;
    n.lo.cb_no_fallback = true;
    n.lo.descriptor_set = 4;
    n.lo.cb_descriptor_set = 4;
    n.lo.cb_binding_base = 16;
    n.lref = gcn::translate(ls.program, n.lo);
    n.ll = n.tl;
    n.ll.attr_vec4s = ls_stride / 16;
    n.ll.own_params = true;
    n.ll.params_binding = 0;
    if (n.lref.ok() && !n.lo.vertex_input.empty()) n.ls = gcn::lift_ls_shader(ls.program, n.lo, n.lref, n.ll);
    return n;
}

// The Forbidden Woods' DOL PN meshes, the hull draws seen in game: three
// control points, and what the lift must refuse.
void dol_pn(const std::vector<Shader>& all) {
    const Shader* ls = find(all, "GXFlver_ColDifSpcBumpDOL_Gbuf_PN.vpo");
    const Shader* hs = find(all, "GXFlver_ColDifSpcBumpDOL_Gbuf_PN.hpo");
    const Shader* ds = find(all, "GXFlver_ColDifSpcBumpDOL_Gbuf_PN.dpo");
    CHECK(ls && hs && ds);
    if (!ls || !hs || !ds) return;
    CHECK(ls->regs[7] == 112 && hs->regs[9] == 3 && hs->regs[10] == 3 && hs->regs[11] == 14 && hs->regs[12] == 96);
    CHECK((hs->regs[6] & 3) == 1);  // a triangle domain
    const Native n = lift_native(*ls, *hs, *ds);
    CHECK(n.hl.constants_path == "user_sgpr[12]" && n.dl.constants_path == "user_sgpr[12]");
    CHECK(lifted(n.hs, "hull " + hs->name));
    CHECK(lifted(n.ds, "domain shader " + ds->name));
    CHECK(lifted(n.ls, "LS " + ls->name));
    CHECK(proof_has(n.hs, "window"));
    std::printf("DOL PN: hull %zu words (translated %zu), domain shader %zu (%zu), LS %zu (%zu)\n", n.hs.spirv.size(), n.href.spirv.size(),
                n.ds.spirv.size(), n.dref.spirv.size(), n.ls.spirv.size(), n.lref.spirv.size());
    // The attribute model's layout is not a hull's window.
    {
        gcn::TessLift attrs;
        attrs.attr_vec4s = 16;
        CHECK(rejected_with(gcn::lift_hull_shader(hs->program, n.ho, n.href, attrs), "only the game's own hull"));
    }
    // A reference that runs the hull on the LDS ring without a window is not
    // the layout the lift turns into stage I/O.
    {
        gcn::TranslateOptions ring = n.ho;
        ring.tess_window = 0;
        CHECK(rejected_with(gcn::lift_hull_shader(hs->program, ring, gcn::translate(hs->program, ring), n.hl), "window"));
    }
    // Without the constants folded the domain shader's control-point addresses
    // are not ones the lift knows, and it has no run-time decode.
    {
        gcn::TessLift unfolded = n.dl;
        unfolded.constants_path.clear();
        CHECK(rejected_with(gcn::lift_domain_shader(ds->program, n.dop, n.dref, unfolded), "address"));
    }
    // On a params block shared with the evaluation stage, the LS's user data
    // would be where that stage's vertex formats are.
    {
        gcn::TranslateOptions shared = n.lo;
        shared.vertex_formats_from_params = true;
        CHECK(!gcn::lift_ls_shader(ls->program, shared, gcn::translate(ls->program, shared), n.ll).ok());
    }
}

// Every game hull in the bundle with the LS and domain shader of its name,
// each distinct program set once: all three stages lift - including the
// domain shaders that read the tessellation constants from the extended user
// data table (patches of 17 constants, the water).
void every_hull(const std::vector<Shader>& all) {
    std::set<std::string> seen;
    int sets = 0, ok = 0, extended = 0;
    for (const Shader& hs : all) {
        if (hs.code.type != 7 || hs.code.words.size() < 300) continue;  // past the particles' constant hulls
        const std::string stem = hs.name.substr(0, hs.name.rfind('.'));
        const Shader *ls = nullptr, *ds = nullptr;
        for (const Shader& s : all) {
            if (s.name == stem + ".vpo") ls = &s;
            if (s.name == stem + ".dpo") ds = &s;
        }
        if (!ls || !ds) continue;
        std::string key;
        for (const Shader* s : {ls, &hs, ds}) key.append(reinterpret_cast<const char*>(s->code.words.data()), s->code.words.size() * 4).push_back('|');
        if (!seen.insert(key).second) continue;
        ++sets;
        const Native n = lift_native(*ls, hs, *ds);
        if (n.dl.constants_path.find("->load") != std::string::npos) ++extended;
        if (lifted(n.hs, "hull " + hs.name) && lifted(n.ds, "domain shader " + ds->name) && lifted(n.ls, "LS " + ls->name)) ++ok;
    }
    std::printf("game hulls: %d of %d program sets lifted whole (%d read the constants from the extended table)\n", ok, sets, extended);
    CHECK(sets >= 100 && ok == sets);
    CHECK(extended > 0);
}

}  // namespace

int main() {
    const std::string ffx = test_app0_file("dvdroot_ps4/shader/gxffxshader.shaderbnd.dcx");
    const std::string flver = test_app0_file("dvdroot_ps4/shader/gxflvershader.shaderbnd.dcx");
    std::vector<Shader> particles_all, flver_all;
    if (ffx.empty() || !load(ffx, particles_all) || !load(flver, flver_all)) {
        std::printf("gcn_tess_lift_test: skipped, the shader bundles are not readable (BBHOST_APP0)\n");
        return kTestSkip;
    }
    particles(particles_all);
    fused_particles(particles_all);
    dol_pn(flver_all);
    every_hull(flver_all);
    if (g_failures) {
        std::fprintf(stderr, "gcn_tess_lift_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("gcn_tess_lift_test ok\n");
    return 0;
}
