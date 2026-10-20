// gcnlift: lift a captured draw's pixel or vertex shader.
//
//   gcnlift <capture-dir> <out-dir> [--vs | --ls | --ds | --native]
//
// Rebuilds the stage's TranslateOptions the renderer used from the capture
// manifest, translates the captured GCN words again and requires the result to
// equal the captured SPIR-V word for word (so the options are the real ones),
// then lifts the program against that reference. Writes lifted.spv (validated),
// listing.txt and report.json with the proof or the rejections.
// Replay the result with `drawreplay <capture-dir> --ps|--vs <out-dir>/lifted.spv`.
// A vertex shader lifts on vertex input only (captures record it). A
// tessellated draw's LS (--ls, on the vertex input the capture records for it)
// and domain shader (--ds) replay with drawreplay --ls / --ds.
// Exit 0: lifted and valid; 1: rejected or invalid; 2: bad input.
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/translate.h"
#include "replay/json.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) fail("cannot read " + p.string());
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary);
    if (!f || !f.write(text.data(), static_cast<std::streamsize>(text.size()))) fail("cannot write " + p.string());
}

std::vector<std::uint32_t> words_of(const std::vector<std::uint8_t>& b, const std::string& what) {
    if (b.size() % 4) fail(what + " is not a whole number of dwords");
    std::vector<std::uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), b.size());
    return w;
}

std::string validate(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& pos, const char* message) {
        if (msg.empty()) msg = "word " + std::to_string(pos.index) + ": " + message;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    return tools.Validate(spirv.data(), spirv.size(), vo) ? std::string() : (msg.empty() ? "invalid" : msg);
}

// Whether a module decorates anything with DescriptorSet `set`: the captured
// stage's constant buffers in set 2 say the renderer pushed them
// (render.cpp cb_push_on), which its translation then assumed.
bool spirv_uses_set(const std::vector<std::uint32_t>& w, std::uint32_t set) {
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (n == 0) break;
        if (op == 71 /* OpDecorate */ && n >= 4 && i + 3 < w.size() && w[i + 2] == 34 /* DescriptorSet */ && w[i + 3] == set) return true;
        i += n;
    }
    return false;
}

std::uint32_t word_at(const json::Value& list, std::size_t k, const char* what) {
    if (list.type != json::Value::Type::Array || k >= list.array.size()) fail(std::string("manifest ") + what + " is too short");
    return static_cast<std::uint32_t>(json::as_u64(list.array[k], what));
}

// The draw's clip varying (TranslateOptions::kVsOutCntlClipVarying): the pixel
// shader's ps_clip_discard, 0 in captures from before it was recorded.
std::uint32_t clip_discard(const json::Value& tr) {
    const json::Value* c = tr.find("ps_clip_discard");
    return c ? static_cast<std::uint32_t>(json::as_u64(*c, "ps_clip_discard")) : 0;
}

// The pixel-stage options render.cpp's gfx_pipeline() builds, from what the
// capture recorded about the draw.
gcn::TranslateOptions pixel_options(const json::Value& m) {
    const json::Value& reg = json::member(m, "registers");
    const json::Value& tr = json::member(m, "translate");
    const json::Value& ps = json::arr(m, "stages").at(1);
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc1 = word_at(json::member(reg, "ps_pgm"), 2, "ps_pgm");
    o.rsrc2 = word_at(json::member(reg, "ps_pgm"), 3, "ps_pgm");
    o.ps_input_ena = static_cast<std::uint32_t>(json::u64(reg, "ps_input_ena"));
    o.descriptor_set = 1;
    for (const json::Value& b : json::arr(ps, "images")) {
        // The renderer asks for a cube; the binding records the 2D array it became.
        const json::Value* cube = b.find("cube");
        const bool is_cube = cube && cube->type == json::Value::Type::Bool && cube->boolean;
        o.image_dims.push_back(is_cube ? std::pair<std::uint32_t, bool>{3, false}
                                       : std::pair<std::uint32_t, bool>{json::u32(b, "dim"), json::flag(b, "arrayed")});
    }
    for (const json::Value& b : json::arr(ps, "samplers")) {
        o.sampler_force_unnormalized.push_back((word_at(json::member(b, "ssharp"), 0, "ssharp") >> 15) & 1);
    }
    o.cb_ssbo = json::flag(tr, "cb_ssbo");
    o.cb_no_fallback = json::flag(tr, "cb_no_fallback");
    o.exec_known = json::flag(tr, "exec_known");
    o.early_fragment_tests = json::flag(tr, "early_fragment_tests");
    o.debug_ps = static_cast<int>(json::u64(tr, "debug_ps"));
    const json::Value& cntl = json::member(reg, "ps_input_cntl");
    for (std::size_t k = 0; k < 32; ++k) {
        const std::uint32_t c = word_at(cntl, k, "ps_input_cntl");
        if ((c >> 10) & 1) o.ps_flat_mask |= 1u << k;
        // Input k is read at the location of the register the input map names
        // (render.cpp ps_input_locations); the vertex shader writes register r at r.
        o.ps_input_map.push_back(static_cast<std::uint8_t>(c & 0x1f));
    }
    o.ps_clip_discard = clip_discard(tr);
    // As render.cpp ps_translate_options: with push descriptors (MoltenVK) the
    // constant buffers sit in set 2, which the captured bindings show.
    if (const json::Value* f = tr.find("ps_lane_from_scan"); f && f->type == json::Value::Type::Bool) o.lane_from_scan = f->boolean;
    for (const json::Value& b : json::arr(ps, "buffers")) {
        if (json::u32(b, "binding") >= gcn::kBindingStorageBuffer0 + gcn::kMaxBuffers) {
            o.cb_descriptor_set = 2;
            o.cb_binding_base = gcn::kBindingStorageBuffer0 + gcn::kMaxBuffers;
        }
    }
    return o;
}

// The vertex-stage options gfx_pipeline() builds. Captures before vs_out_cntl
// and vs_invariant were recorded default them (0, off).
gcn::TranslateOptions vertex_options(const json::Value& m) {
    const json::Value& reg = json::member(m, "registers");
    const json::Value& tr = json::member(m, "translate");
    const json::Value& vs = json::arr(m, "stages").at(0);
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.rsrc1 = word_at(json::member(reg, "vs_pgm"), 2, "vs_pgm");
    o.rsrc2 = word_at(json::member(reg, "vs_pgm"), 3, "vs_pgm");
    if (const json::Value* cntl = tr.find("vs_out_cntl")) o.vs_out_cntl = static_cast<std::uint32_t>(json::as_u64(*cntl, "vs_out_cntl"));
    // build_gfx_pipeline adds the clip varying's host bits to the register's.
    if (const std::uint32_t clip = clip_discard(tr)) {
        o.vs_out_cntl |= gcn::TranslateOptions::kVsOutCntlClipVarying | ((clip >> 8) & 31) << gcn::TranslateOptions::kVsOutCntlClipLocationShift;
    }
    const json::Value* inv = tr.find("vs_invariant");
    o.invariant_position = inv && inv->type == json::Value::Type::Bool && inv->boolean;
    o.descriptor_set = 0;
    for (const json::Value& b : json::arr(vs, "images")) {
        const json::Value* cube = b.find("cube");
        const bool is_cube = cube && cube->type == json::Value::Type::Bool && cube->boolean;
        o.image_dims.push_back(is_cube ? std::pair<std::uint32_t, bool>{3, false}
                                       : std::pair<std::uint32_t, bool>{json::u32(b, "dim"), json::flag(b, "arrayed")});
    }
    for (const json::Value& b : json::arr(vs, "samplers")) {
        o.sampler_force_unnormalized.push_back((word_at(json::member(b, "ssharp"), 0, "ssharp") >> 15) & 1);
    }
    o.cb_ssbo = json::flag(tr, "cb_ssbo");
    o.cb_no_fallback = json::flag(tr, "cb_no_fallback");
    o.exec_known = json::flag(tr, "exec_known");
    if (const json::Value* f = tr.find("vs_lane_from_scan"); f && f->type == json::Value::Type::Bool) o.lane_from_scan = f->boolean;  // MoltenVK: no subgroup built-ins in vertex stages
    const json::Value* vi = m.find("vertex_input");
    if (vi && vi->type == json::Value::Type::Object) {
        for (const json::Value& e : json::arr(*vi, "elements")) {
            gcn::VertexElement el;
            el.location = json::u32(e, "location");
            el.vdata = json::u32(e, "vdata");
            el.count = json::u32(e, "count");
            el.w3 = static_cast<std::uint32_t>(json::u64(e, "w3"));
            o.vertex_input.push_back(el);
        }
        // As the capture records it. Captures from before step 6c do not, and are
        // rebuilt with formats from params wherever the loader supports them
        // (drawreplay fills StageParams::vertex_formats from these elements).
        if (const json::Value* p = tr.find("vertex_formats_from_params"); p && p->type == json::Value::Type::Bool) {
            o.vertex_formats_from_params = p->boolean;
        } else {
            o.vertex_formats_from_params = gcn::vertex_formats_from_params_supported(o.vertex_input);
        }
    }
    return o;
}

// A tessellated draw's evaluation stage (stages[0]) as render.cpp
// vs_translate_options builds it for the host's tessellator.
const json::Value& tessellation(const json::Value& m) {
    const json::Value* t = m.find("tessellation");
    if (!t || t->type != json::Value::Type::Object) fail("the capture is not of a tessellated draw");
    return *t;
}
bool flag_or(const json::Value& obj, const char* key, bool otherwise) {
    const json::Value* v = obj.find(key);
    return v && v->type == json::Value::Type::Bool ? v->boolean : otherwise;
}
gcn::TranslateOptions domain_options(const json::Value& m) {
    const json::Value& t = tessellation(m);
    const json::Value& tr = json::member(m, "translate");
    gcn::TranslateOptions o = vertex_options(m);
    o.vertex_input.clear();
    o.vertex_formats_from_params = false;
    o.force_dispatcher = false;
    o.bindless = flag_or(tr, "bindless", false);
    o.stage = gcn::Stage::TessEval;
    o.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
    o.tess_patch_control_points = json::u32(t, "control_points");
    o.tess_quads = json::flag(t, "quads");
    o.tess_spacing = static_cast<int>(json::u64(t, "spacing"));
    o.tess_cw = json::flag(t, "cw");
    o.tess_window = json::u32(t, "window");
    o.tess_lds_bound = flag_or(tr, "tess_lds_bound", true);
    o.tess_lds_attributes = json::str(t, "kind") == "attributes";
    o.tess_attr_vec4s = json::u32(t, "attr_vec4s");
    return o;
}
// The LS as the vertex stage (render.cpp tess_ls_options): the captured
// module inlines its fetch shader; the lift's reference takes the elements as
// vertex input (tessellation.ls_vertex_input) and binds every buffer.
gcn::TranslateOptions ls_options(const json::Value& m) {
    const json::Value& t = tessellation(m);
    const json::Value& tr = json::member(m, "translate");
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.bindless = flag_or(tr, "bindless", false);
    o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
    o.tess_lds_attributes = true;
    o.tess_attr_vec4s = json::u32(t, "attr_vec4s");
    o.rsrc1 = word_at(json::member(t, "ls_rsrc"), 0, "ls_rsrc");
    o.rsrc2 = word_at(json::member(t, "ls_rsrc"), 1, "ls_rsrc");
    o.exec_known = json::flag(tr, "exec_known");
    o.descriptor_set = 0;
    return o;
}

// bbhost's set 4 (render.cpp kTessNativeSet...): a native hull pipeline's LS
// and hull - the LS's params at binding 0 and buffers from 16, the hull's
// params at 1 and buffers from 32 - and a particle LS's constant buffers.
constexpr std::uint32_t kNativeSet = 4, kNativeLsParams = 0, kNativeHsParams = 1, kNativeLsBuffers = 16, kNativeHsBuffers = 32;
json::Value bindings_json(const gcn::TranslateResult& r) {
    json::Value a = json::Value::make_array();
    for (const gcn::BufferBinding& b : r.buffers) {
        json::Value j = json::Value::make_object();
        j.set("binding", b.binding);
        j.set("path", b.path.str());
        j.set("user_sgpr", b.path.user_sgpr);
        j.set("immediate", b.path.immediate);
        json::Value loads = json::Value::make_array();
        for (const gcn::ResourceStep& st : b.path.loads) {
            json::Value l = json::Value::make_object();
            l.set("offset_dw", st.offset_dw);
            l.set("vsharp", st.vsharp);
            loads.push(l);
        }
        j.set("loads", loads);
        j.set("final_offset_dw", b.path.final_offset_dw);
        j.set("final_vsharp", b.path.final_vsharp);
        j.set("max_dw", b.max_dw);
        j.set("pointer", b.pointer);
        j.set("indexed", b.indexed);
        j.set("ranged", b.ranged);
        a.push(j);
    }
    return a;
}

// gcnlift --ls / --ds: a tessellated draw's LS or domain shader.
int run_tess(const fs::path& capture, const fs::path& out, bool ls) {
    const std::vector<std::uint8_t> text = read_file(capture / "manifest.json");
    json::Value m;
    std::string error;
    if (!json::parse(std::string(text.begin(), text.end()), m, error)) fail("manifest.json: " + error);
    if (json::str(m, "schema") != "bbhost-draw-capture") fail("not a draw capture");
    const json::Value& blobs = json::member(m, "blobs");
    const auto blob = [&](const std::string& name) { return read_file(capture / json::str(json::member(blobs, name.c_str()), "file")); };
    const json::Value& t = tessellation(m);
    if (json::str(t, "kind") != "attributes") fail("only the attribute model (one control point as stage attributes) is lifted so far");
    std::vector<std::uint32_t> gcn_words, captured;
    gcn::TranslateOptions options;
    gcn::Program fetch_program;
    if (ls) {
        gcn_words = words_of(blob(json::str(t, "ls_gcn")), "ls-gcn");
        captured = words_of(blob(json::str(t, "vertex_spirv")), "tess-vs-spirv");
        options = ls_options(m);
        const json::Value* f = t.find("ls_fetch_gcn");
        if (f && f->type == json::Value::Type::String) {
            const std::vector<std::uint32_t> fetch_words = words_of(blob(f->string), "ls-fetch-gcn");
            fetch_program = gcn::decode(fetch_words.data(), fetch_words.size());
            options.fetch = &fetch_program;
        }
    } else {
        const json::Value& stage = json::arr(m, "stages").at(0);
        gcn_words = words_of(blob(json::str(stage, "gcn")), "vs-gcn");
        captured = words_of(blob(json::str(stage, "spirv")), "vs-spirv");
        options = domain_options(m);
        options.link_outputs = true;
        for (std::uint8_t r = 0; r < 32; ++r) options.output_links.push_back(r);
        if (spirv_uses_set(captured, 2)) options.cb_descriptor_set = 2;
    }
    const gcn::Program program = gcn::decode(gcn_words.data(), gcn_words.size());
    // The captured module first: the rebuilt options must reproduce it.
    const gcn::TranslateResult captured_ref = gcn::translate(program, options);
    const bool same_reference = captured_ref.ok() && captured_ref.spirv == captured;
    // The lift's reference: the LS on vertex input, every buffer bound.
    gcn::TranslateOptions lift_options = options;
    if (ls) {
        const json::Value* vi = t.find("ls_vertex_input");
        if (!vi || vi->type != json::Value::Type::Object) fail("the capture has no vertex input for the LS");
        lift_options.fetch = nullptr;
        for (const json::Value& e : json::arr(*vi, "elements")) {
            gcn::VertexElement el;
            el.location = json::u32(e, "location");
            el.vdata = json::u32(e, "vdata");
            el.count = json::u32(e, "count");
            el.w3 = static_cast<std::uint32_t>(json::u64(e, "w3"));
            lift_options.vertex_input.push_back(el);
        }
        lift_options.cb_ssbo = true;
        lift_options.cb_no_fallback = true;
        // An LS that reads constant buffers binds them in set 4 from 16, as
        // bbhost lays its pipeline out (render.cpp, the point sprites' LS).
        if (!gcn::translate(program, lift_options).buffers.empty()) {
            lift_options.cb_descriptor_set = static_cast<int>(kNativeSet);
            lift_options.cb_binding_base = kNativeLsBuffers;
        }
    }
    const gcn::TranslateResult reference = ls ? gcn::translate(program, lift_options) : captured_ref;
    gcn::TessLift tl;
    tl.attr_vec4s = json::u32(t, "attr_vec4s");
    const gcn::LiftResult lifted = ls ? gcn::lift_ls_shader(program, lift_options, reference, tl)
                                      : gcn::lift_domain_shader(program, lift_options, reference, tl);
    const std::string invalid = lifted.ok() ? validate(lifted.spirv) : std::string();
    fs::create_directories(out);
    json::Value report = json::Value::make_object();
    report.set("schema", "bbhost-gcn-lift");
    report.set("version", 1);
    report.set("capture", capture.string());
    report.set("pipeline", json::str(json::member(m, "identity"), "pipeline"));
    report.set("stage", ls ? "ls" : "domain");
    report.set("reference_matches_capture", same_reference);
    report.set("captured_module", "translated");
    report.set("lifted", lifted.ok());
    report.set("spirv_valid", lifted.ok() && invalid.empty());
    if (!invalid.empty()) report.set("spirv_error", invalid);
    json::Value proof = json::Value::make_array();
    for (const std::string& p : lifted.proof) proof.push(p);
    report.set("proof", proof);
    json::Value rejections = json::Value::make_array();
    for (const std::string& r : lifted.rejections) rejections.push(r);
    report.set("rejections", rejections);
    report.set("reference_words", static_cast<unsigned long>(reference.spirv.size()));
    report.set("lifted_words", static_cast<unsigned long>(lifted.spirv.size()));
    write_file(out / "report.json", json::dump(report));
    write_file(out / "listing.txt", lifted.listing);
    const auto spv_text = [](const std::vector<std::uint32_t>& w) { return std::string(reinterpret_cast<const char*>(w.data()), w.size() * 4); };
    if (lifted.ok()) write_file(out / "lifted.spv", spv_text(lifted.spirv));
    if (reference.ok()) write_file(out / "reference.spv", spv_text(reference.spirv));
    if (ls && reference.ok() && !reference.buffers.empty()) {
        // drawreplay --ls-set4: the buffers it binds there.
        json::Value set4 = json::Value::make_object(), st = json::Value::make_object();
        st.set("params_binding", kNativeLsParams);
        st.set("buffers", bindings_json(reference));
        set4.set("ls", st);
        write_file(out / "ls-set4.json", json::dump(set4));
        std::printf("its %zu constant buffers are bound in set 4 (ls-set4.json)\n", reference.buffers.size());
    }
    std::printf("rebuilt %s module %s the captured SPIR-V; lift %s%s\n", ls ? "LS" : "domain", same_reference ? "matches" : "DOES NOT MATCH",
                lifted.ok() ? "ok" : "rejected", invalid.empty() ? "" : (" but INVALID: " + invalid).c_str());
    for (const std::string& r : lifted.rejections) std::printf("  rejected: %s\n", r.c_str());
    for (const std::string& p : lifted.proof) std::printf("  proof: %s\n", p.c_str());
    if (!same_reference) {
        std::printf("  the rebuilt options do not reproduce the pipeline's shader; the lift is not comparable\n");
        return 1;
    }
    return lifted.ok() && invalid.empty() ? 0 : 1;
}

// gcnlift --native: a game-hull draw's three stages as one pipeline on stage
// I/O without the LDS ring (TessLift::window), as bbhost builds it: the LS on
// the captured vertex input and the hull as the control stage in a set of
// their own (kNativeSet), the domain shader as the evaluation stage (the
// captured stage's bindings). The
// tessellation constants fold where each stage reads them. Writes ls.spv,
// hs.spv, ds.spv and native.json (the LS's and hull's buffer bindings, for
// drawreplay --native to resolve from the captured memory).
int run_native(const fs::path& capture, const fs::path& out) {
    const std::vector<std::uint8_t> text = read_file(capture / "manifest.json");
    json::Value m;
    std::string error;
    if (!json::parse(std::string(text.begin(), text.end()), m, error)) fail("manifest.json: " + error);
    const json::Value& blobs = json::member(m, "blobs");
    const auto blob = [&](const std::string& name) { return read_file(capture / json::str(json::member(blobs, name.c_str()), "file")); };
    const json::Value& t = tessellation(m);
    if (json::str(t, "kind") != "hull") fail("--native lifts a game-hull draw");
    const json::Value& tr = json::member(m, "translate");
    // The window's layout, from the tessellation constants the host built
    // (gx_hull_layout): LS stride, CP stride, patches, inputs, patch-constant
    // bytes, their base, outputs.
    std::uint32_t tc[12];
    for (std::size_t k = 0; k < 12; ++k) tc[k] = word_at(json::member(t, "tess_constants"), k, "tess_constants");
    gcn::TessLift tl;
    tl.window = true;
    tl.control_points = json::u32(t, "control_points");
    tl.ls_stride = tc[0];
    tl.cp_stride = tc[1];
    tl.patch_constants = tc[4] / 16;
    std::memcpy(tl.constants, tc, sizeof(tc));
    if (tc[2] != 1 || tc[3] != tl.control_points * tl.ls_stride || tc[6] != tl.control_points * tl.cp_stride) fail("tessellation constants that are not one window");
    // Where each stage reads them: the evaluation stage's binding holding
    // these words, and the user-data V# with that binding's address.
    const json::Value& ds_stage = json::arr(m, "stages").at(0);
    std::string ds_path;
    std::uint64_t tc_va = 0;
    for (const json::Value& b : json::arr(ds_stage, "buffers")) {
        const std::vector<std::uint8_t> bytes = blob(json::str(b, "blob"));
        const std::size_t n = std::min<std::size_t>(bytes.size() & ~std::size_t{3}, sizeof(tc));
        if (n >= 16 && std::memcmp(bytes.data(), tc, n) == 0) {
            ds_path = json::str(b, "path");
            if (const json::Value* va = b.find("guest_va")) tc_va = json::as_u64(*va, "guest_va");
        }
    }
    std::string hs_path;
    for (std::uint32_t k = 0; k + 1 < 16 && tc_va; ++k) {
        const std::uint64_t base = word_at(json::member(t, "hs_user"), k, "hs_user") |
                                   (static_cast<std::uint64_t>(word_at(json::member(t, "hs_user"), k + 1, "hs_user") & 0xffff) << 32);
        if (base == tc_va) {
            hs_path = "user_sgpr[" + std::to_string(k) + "]";
            break;
        }
    }
    fs::create_directories(out);
    json::Value report = json::Value::make_object();
    report.set("schema", "bbhost-gcn-lift-native");
    report.set("set", kNativeSet);
    report.set("capture", capture.string());
    report.set("constants_path_ds", ds_path);
    report.set("constants_path_hs", hs_path);
    bool all = true;
    const auto spv_text = [](const std::vector<std::uint32_t>& w) { return std::string(reinterpret_cast<const char*>(w.data()), w.size() * 4); };
    const auto finish = [&](const char* stage, const gcn::LiftResult& l, const gcn::TranslateResult& ref, json::Value extra) {
        const std::string invalid = l.ok() ? validate(l.spirv) : std::string();
        extra.set("lifted", l.ok());
        extra.set("spirv_valid", l.ok() && invalid.empty());
        if (!invalid.empty()) extra.set("spirv_error", invalid);
        json::Value pr = json::Value::make_array();
        for (const std::string& x : l.proof) pr.push(x);
        extra.set("proof", pr);
        json::Value rj = json::Value::make_array();
        for (const std::string& x : l.rejections) rj.push(x);
        extra.set("rejections", rj);
        extra.set("reference_words", static_cast<unsigned long>(ref.spirv.size()));
        extra.set("lifted_words", static_cast<unsigned long>(l.spirv.size()));
        report.set(stage, extra);
        if (l.ok() && invalid.empty()) write_file(out / (std::string(stage) + ".spv"), spv_text(l.spirv));
        write_file(out / (std::string(stage) + "-listing.txt"), l.listing);
        std::printf("%s: %s%s\n", stage, l.ok() ? "lifted" : ("rejected: " + l.rejections[0]).c_str(), invalid.empty() ? "" : (" but INVALID: " + invalid).c_str());
        all = all && l.ok() && invalid.empty();
    };
    {  // LS
        const std::vector<std::uint32_t> words = words_of(blob(json::str(t, "ls_gcn")), "ls-gcn");
        const gcn::Program program = gcn::decode(words.data(), words.size());
        gcn::TranslateOptions o = ls_options(m);
        o.tess_attr_vec4s = tl.ls_stride / 16;
        const json::Value* vi = t.find("ls_vertex_input");
        if (!vi || vi->type != json::Value::Type::Object) fail("the capture has no vertex input for the LS");
        for (const json::Value& e : json::arr(*vi, "elements")) {
            gcn::VertexElement el;
            el.location = json::u32(e, "location");
            el.vdata = json::u32(e, "vdata");
            el.count = json::u32(e, "count");
            el.w3 = static_cast<std::uint32_t>(json::u64(e, "w3"));
            o.vertex_input.push_back(el);
        }
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.descriptor_set = kNativeSet;
        o.cb_descriptor_set = static_cast<int>(kNativeSet);
        o.cb_binding_base = kNativeLsBuffers;
        const gcn::TranslateResult ref = gcn::translate(program, o);
        gcn::TessLift ll = tl;
        ll.attr_vec4s = tl.ls_stride / 16;
        ll.own_params = true;
        ll.params_binding = kNativeLsParams;
        ll.constants_path.clear();
        const gcn::LiftResult l = ref.ok() ? gcn::lift_ls_shader(program, o, ref, ll) : gcn::LiftResult{{}, {"the reference translation failed: " + ref.errors[0]}};
        json::Value extra = json::Value::make_object();
        extra.set("params_binding", kNativeLsParams);
        extra.set("buffers", bindings_json(ref));
        extra.set("images", static_cast<unsigned long>(ref.images.size() + ref.samplers.size()));
        finish("ls", l, ref, extra);
    }
    {  // hull
        const std::vector<std::uint32_t> words = words_of(blob(json::str(t, "hs_gcn")), "hs-gcn");
        const gcn::Program program = gcn::decode(words.data(), words.size());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::TessControl;
        o.tess_role = gcn::TranslateOptions::TessRole::HullTcs;
        o.rsrc1 = word_at(json::member(t, "hs_rsrc"), 0, "hs_rsrc");
        o.rsrc2 = word_at(json::member(t, "hs_rsrc"), 1, "hs_rsrc");
        o.tess_patch_control_points = tl.control_points;
        o.tess_window = json::u32(t, "window");
        o.tess_lds_bound = flag_or(tr, "tess_lds_bound", true);
        o.tess_quads = json::flag(t, "quads");
        o.tess_spacing = static_cast<int>(json::u64(t, "spacing"));
        o.tess_cw = json::flag(t, "cw");
        o.exec_known = json::flag(tr, "exec_known");
        o.descriptor_set = kNativeSet;
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.cb_descriptor_set = static_cast<int>(kNativeSet);
        o.cb_binding_base = kNativeHsBuffers;
        const gcn::TranslateResult ref = gcn::translate(program, o);
        gcn::TessLift hl = tl;
        hl.own_params = true;
        hl.params_binding = kNativeHsParams;
        hl.constants_path = hs_path;
        const gcn::LiftResult l = ref.ok() ? gcn::lift_hull_shader(program, o, ref, hl) : gcn::LiftResult{{}, {"the reference translation failed: " + ref.errors[0]}};
        json::Value extra = json::Value::make_object();
        extra.set("params_binding", kNativeHsParams);
        extra.set("buffers", bindings_json(ref));
        extra.set("images", static_cast<unsigned long>(ref.images.size() + ref.samplers.size()));
        finish("hs", l, ref, extra);
    }
    {  // domain shader: the captured stage's options and bindings
        const std::vector<std::uint32_t> words = words_of(blob(json::str(ds_stage, "gcn")), "vs-gcn");
        const gcn::Program program = gcn::decode(words.data(), words.size());
        const std::vector<std::uint32_t> captured = words_of(blob(json::str(ds_stage, "spirv")), "vs-spirv");
        gcn::TranslateOptions o = domain_options(m);
        o.link_outputs = true;
        for (std::uint8_t r = 0; r < 32; ++r) o.output_links.push_back(r);
        if (spirv_uses_set(captured, 2)) o.cb_descriptor_set = 2;
        const gcn::TranslateResult ref = gcn::translate(program, o);
        gcn::TessLift dl = tl;
        dl.constants_path = ds_path;
        const gcn::LiftResult l = ref.ok() ? gcn::lift_domain_shader(program, o, ref, dl) : gcn::LiftResult{{}, {"the reference translation failed: " + ref.errors[0]}};
        json::Value extra = json::Value::make_object();
        extra.set("reference_matches_capture", ref.ok() && ref.spirv == captured);
        finish("ds", l, ref, extra);
        if (!(ref.ok() && ref.spirv == captured)) {
            std::printf("  the rebuilt domain options do not reproduce the captured module; the lift is not comparable\n");
            all = false;
        }
    }
    write_file(out / "native.json", json::dump(report));
    return all ? 0 : 1;
}

int run(const fs::path& capture, const fs::path& out, bool vs) {
    const std::vector<std::uint8_t> text = read_file(capture / "manifest.json");
    json::Value m;
    std::string error;
    if (!json::parse(std::string(text.begin(), text.end()), m, error)) fail("manifest.json: " + error);
    if (json::str(m, "schema") != "bbhost-draw-capture") fail("not a draw capture");
    const json::Value& blobs = json::member(m, "blobs");
    const auto blob = [&](const std::string& name) { return read_file(capture / json::str(json::member(blobs, name.c_str()), "file")); };
    const json::Value& stage = json::arr(m, "stages").at(vs ? 0 : 1);
    if (!json::flag(stage, "present")) fail(vs ? "the capture has no vertex shader" : "the capture has no pixel shader");
    const std::vector<std::uint32_t> gcn_words = words_of(blob(json::str(stage, "gcn")), vs ? "vs-gcn" : "ps-gcn");
    const std::vector<std::uint32_t> captured = words_of(blob(json::str(stage, "spirv")), vs ? "vs-spirv" : "ps-spirv");

    const gcn::Program program = gcn::decode(gcn_words.data(), gcn_words.size());
    gcn::TranslateOptions options = vs ? vertex_options(m) : pixel_options(m);
    if (spirv_uses_set(captured, 2)) options.cb_descriptor_set = 2;
    if (vs) {
        // Linking by register (render.cpp register_links): the vertex shader
        // writes param register r at location r, whatever older captures recorded.
        options.link_outputs = true;
        for (std::uint8_t r = 0; r < 32; ++r) options.output_links.push_back(r);
    }
    gcn::Program fetch_program;  // a vertex shader without vertex input inlines its fetch shader
    if (vs && options.vertex_input.empty()) {
        const json::Value* fetch = json::member(m, "shaders").find("fetch_gcn");
        if (fetch && fetch->type == json::Value::Type::String) {
            const std::vector<std::uint32_t> fetch_words = words_of(blob(fetch->string), "fetch-gcn");
            fetch_program = gcn::decode(fetch_words.data(), fetch_words.size());
            options.fetch = &fetch_program;
        }
    }
    const gcn::TranslateResult reference = gcn::translate(program, options);
    const gcn::LiftResult lifted = vs ? gcn::lift_vertex_shader(program, options, reference) : gcn::lift_pixel_shader(program, options, reference);
    // A capture taken with BBHOST_DECOMP bound the lifted stage; that is what
    // the rebuilt options have to reproduce.
    const json::Value* stage_lifted = json::member(m, "translate").find(vs ? "vs_lifted" : "ps_lifted");
    const bool captured_lifted = stage_lifted && stage_lifted->type == json::Value::Type::Bool && stage_lifted->boolean;
    // Debug names aside: the renderer puts the engine's names on a lift
    // (gcn::name_lift, from the container's reflection, which a capture does
    // not carry), and they change nothing the module computes.
    const auto unnamed = [](const std::vector<std::uint32_t>& m) {
        if (m.size() < 5) return m;
        std::vector<std::uint32_t> out(m.begin(), m.begin() + 5);
        for (std::size_t i = 5; i < m.size();) {
            const std::uint32_t n = m[i] >> 16, op = m[i] & 0xffff;
            if (!n || i + n > m.size()) return m;
            if (op != 5 && op != 6) out.insert(out.end(), m.begin() + static_cast<std::ptrdiff_t>(i), m.begin() + static_cast<std::ptrdiff_t>(i + n));  // OpName, OpMemberName
            i += n;
        }
        return out;
    };
    const bool same_reference = captured_lifted ? lifted.ok() && unnamed(lifted.spirv) == unnamed(captured)
                                                : reference.ok() && unnamed(reference.spirv) == unnamed(captured);
    const std::string invalid = lifted.ok() ? validate(lifted.spirv) : std::string();

    fs::create_directories(out);
    json::Value report = json::Value::make_object();
    report.set("schema", "bbhost-gcn-lift");
    report.set("version", 1);
    report.set("capture", capture.string());
    report.set("pipeline", json::str(json::member(m, "identity"), "pipeline"));
    report.set("stage", vs ? "vertex" : "pixel");
    report.set("reference_matches_capture", same_reference);  // the rebuilt module of the captured kind
    report.set("captured_module", captured_lifted ? "lifted" : "translated");
    report.set("lifted", lifted.ok());
    report.set("spirv_valid", lifted.ok() && invalid.empty());
    if (!invalid.empty()) report.set("spirv_error", invalid);
    json::Value proof = json::Value::make_array();
    for (const std::string& p : lifted.proof) proof.push(p);
    report.set("proof", proof);
    json::Value rejections = json::Value::make_array();
    for (const std::string& r : lifted.rejections) rejections.push(r);
    report.set("rejections", rejections);
    report.set("reference_words", static_cast<unsigned long>(reference.spirv.size()));
    report.set("lifted_words", static_cast<unsigned long>(lifted.spirv.size()));
    {
        // Step 2 of per-shader compiles: the dimensions predicted from the
        // program alone against the ones the captured draw bound.
        const std::vector<gcn::PredictedImage> predicted = gcn::predict_image_dims(program, reference);
        json::Value dims = json::Value::make_object();
        json::Value pred = json::Value::make_array(), bound = json::Value::make_array();
        bool match = true, conflict = false;
        for (std::size_t k = 0; k < predicted.size(); ++k) {
            const std::pair<std::uint32_t, bool> a = k < options.image_dims.size() ? options.image_dims[k] : std::pair<std::uint32_t, bool>{1, false};
            pred.push(std::to_string(predicted[k].dim) + (predicted[k].arrayed ? "a" : "") + (predicted[k].conflict ? "!" : ""));
            bound.push(std::to_string(a.first) + (a.second ? "a" : ""));
            match &= predicted[k].dim == a.first && predicted[k].arrayed == a.second;
            conflict |= predicted[k].conflict;
        }
        dims.set("predicted", pred);
        dims.set("bound", bound);
        dims.set("match", match);
        dims.set("conflict", conflict);
        report.set("image_dims", dims);
    }
    write_file(out / "report.json", json::dump(report));
    write_file(out / "listing.txt", lifted.listing);
    const auto spv_text = [](const std::vector<std::uint32_t>& w) { return std::string(reinterpret_cast<const char*>(w.data()), w.size() * 4); };
    if (lifted.ok()) write_file(out / "lifted.spv", spv_text(lifted.spirv));
    if (reference.ok()) write_file(out / "reference.spv", spv_text(reference.spirv));
    // Diagnostic floor for replay benches: the same translation with every
    // export magenta (BBHOST_DEBUG_PS_COLOR=1), so the rest of the pixel shader
    // is dead code while the pipeline, bindings and fragment tests stay the same.
    if (!vs) {
        gcn::TranslateOptions magenta_options = options;
        magenta_options.debug_ps = 1;
        const gcn::TranslateResult magenta = gcn::translate(program, magenta_options);
        if (magenta.ok() && validate(magenta.spirv).empty()) write_file(out / "magenta.spv", spv_text(magenta.spirv));
    }
    std::printf("rebuilt %s module %s the captured SPIR-V; lift %s%s\n", captured_lifted ? "lifted" : "translated",
                same_reference ? "matches" : "DOES NOT MATCH", lifted.ok() ? "ok" : "rejected",
                invalid.empty() ? "" : (" but INVALID: " + invalid).c_str());
    for (const std::string& r : lifted.rejections) std::printf("  rejected: %s\n", r.c_str());
    for (const std::string& p : lifted.proof) std::printf("  proof: %s\n", p.c_str());
    if (!same_reference) {
        std::printf("  the rebuilt options do not reproduce the pipeline's shader; the lift is not comparable\n");
        return 1;
    }
    return lifted.ok() && invalid.empty() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const bool vs = argc == 4 && std::strcmp(argv[3], "--vs") == 0;
    const bool ls = argc == 4 && std::strcmp(argv[3], "--ls") == 0;
    const bool ds = argc == 4 && std::strcmp(argv[3], "--ds") == 0;
    const bool native = argc == 4 && std::strcmp(argv[3], "--native") == 0;
    if (argc != 3 && !vs && !ls && !ds && !native) {
        std::fprintf(stderr, "usage: gcnlift <capture-dir> <out-dir> [--vs | --ls | --ds | --native]\n");
        return 2;
    }
    try {
        if (native) return run_native(argv[1], argv[2]);
        if (ls || ds) return run_tess(argv[1], argv[2], ls);
        return run(argv[1], argv[2], vs);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gcnlift: %s\n", e.what());
        return 2;
    }
}
