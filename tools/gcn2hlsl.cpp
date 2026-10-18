// gcn2hlsl: every lifted shader in the game's bundles printed as HLSL, compiled
// with DXC to SPIR-V for Vulkan and checked against the lift.
//
//   gcn2hlsl [--ps] [--vs] [--tess] [--cs] [--clip] [--runtime-offsets] [--native-half-rtz] [--invariant] [--O0] [--jobs N] [--only NAME[,NAME]]
//            [--out DIR] [--list FILE] [--no-dxc] [--sample N] [--oconfig PASSES] <bundle.dcx>...
//   gcn2hlsl --spv <lifted.spv> --out <prefix> [--O0]
//            one lifted module (a draw capture's bound one, tools/hlsl_verify.py):
//            writes <prefix>.hlsl and <prefix>.dxc.spv; exit 0 when the module
//            compiled, validated and kept the lift's interface and operations.
//
// Each program is translated and lifted with the options the renderer uses
// when GX creates it (render.cpp precompile_ps / precompile_vs: the
// no-fallback variant, constant buffers pushed in set 2), named from its
// container's reflection, printed as HLSL (gcn/hlsl.h) and compiled by DXC
// (host/dxc.h, libdxcompiler loaded at run time). The DXC module must pass
// spirv-val and declare the lift's interface (gcn::hlsl_interface_diff).
// --cs lifts the compute shaders as the renderer does when GX creates them
// (gpu.cpp precompile_compute). --clip lifts the pixel shaders with the clip-distance discard and the
// vertex shaders with the clip varying, as the draws that use them do.
// --invariant declares the vertex shaders' position Invariant, as
// BBHOST_VS_INVARIANT=1 does.
// --runtime-offsets lifts a sample's offset as the run-time operand, as on a
// device with VK_KHR_maintenance8 (gcn::set_runtime_sample_offsets); without
// it the offset moves the coordinates, as elsewhere. --native-half-rtz lifts
// as on a device that rounds f16 toward zero itself (gcn::set_native_half_rtz:
// RADV, the Steam Deck's driver). --device and --sweep set both from the GPU.
//
// --out writes, per program, <name>.<stage>.lift.spv, .hlsl and .dxc.spv.
// They are made from the game's own shaders: they stay on this machine.
// --list writes one line per program: stage, name, result, compile ms, reason.
// --app0 DIR takes the bundles from the game folder (DIR/dvdroot_ps4/shader,
// the 1.09 update's copies from DIR-UPDATE where it is kept beside it).
// Exit 0: every lift compiled to a valid module with the lift's interface.
//
//   gcn2hlsl --sweep --app0 <game folder> [--gpu N|NAME] [--list FILE] [--validate] [--monolithic]
//            the driver-acceptance sweep: every vertex, pixel, tessellation and
//            compute program lifted as the renderer lifts it on this machine's GPU (its
//            run-time sample offsets and half rounding), printed, compiled by
//            DXC and checked as the HLSL path checks it, and then the lift's
//            module and DXC's each built into the pipelines the renderer makes
//            of the stage on that GPU (tools/pipeline_probe.h: graphics
//            pipeline libraries, a fast and an optimized link; a game hull's
//            LS, hull and domain shader as one native hull pipeline). The
//            summary counts, by stage, what the driver built and refused, and
//            names every DXC module the driver refused where it builds the
//            lift - the stages the game's fallback would give back the lift.
//            --device does the same for the stages chosen with --ps/--vs/--tess.
//            The driver's own shader disk cache is off for the run
//            (__GL_SHADER_DISK_CACHE, MESA_SHADER_CACHE_DISABLE), so the sweep
//            neither reads the game's pipelines from it nor pushes them out;
//            --driver-cache keeps it. Exit 0: as above, and no DXC module
//            refused where its lift was built.
#include "gcn/container.h"
#include "gcn/half.h"
#include "gcn/hlsl.h"
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/names.h"
#include "gcn/translate.h"
#include "host/dxc.h"
#ifdef GCN2HLSL_HAVE_VULKAN
#include "pipeline_probe.h"
#endif

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

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
    vo.SetAllowOffsetTextureOperand(true);
    return tools.Validate(spirv.data(), spirv.size(), vo) ? std::string() : (msg.empty() ? "invalid" : msg);
}

bool write_bytes(const fs::path& p, const void* data, std::size_t n) {
    FILE* f = std::fopen(p.string().c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data, 1, n, f) == n;
    return std::fclose(f) == 0 && ok;
}

// A reason without its offsets and numbers, so like failures count together.
std::string reason_key(const std::string& r) {
    std::string k;
    bool digits = false;
    for (char c : r.substr(0, 120)) {
        const bool d = (c >= '0' && c <= '9');
        if (d) {
            if (!digits) k += '#';
            digits = true;
            continue;
        }
        digits = false;
        k += c;
    }
    return k;
}

enum class Kind { Pixel, Vertex, Ls, Hull, Domain, Compute };
const char* kind_name(Kind k) {
    switch (k) {
    case Kind::Pixel: return "ps";
    case Kind::Vertex: return "vs";
    case Kind::Ls: return "ls";
    case Kind::Hull: return "hs";
    case Kind::Domain: return "ds";
    case Kind::Compute: return "cs";
    }
    return "?";
}
struct Job {
    std::string bundle;  // the bundle's file name without its extensions
    std::string name;    // the program's name in its bundle
    Kind kind = Kind::Pixel;
    gcn::Program program;
    gcn::TranslateOptions options;
    gcn::TessLift tess;
    std::vector<std::uint8_t> container;
    int set = -1;  // a game hull's LS, hull and domain shader share a set (one native hull pipeline)
};

// ---- tessellation: the stages as tests/gcn_tess_lift_test.cpp lifts them ----
struct TessShader {
    std::string name;
    gcn::ShaderCode code;
    gcn::Program program;
    std::uint32_t regs[16] = {};
    std::vector<std::uint8_t> data;
};
std::size_t block_of(const TessShader& s) {
    std::size_t shdr = 0;
    while (shdr + 0x60 <= s.data.size() && shdr < 0x100 && std::memcmp(s.data.data() + shdr, "Shdr", 4) != 0) ++shdr;
    return shdr + 16;
}
// The constant buffer a stage reads the tessellation constants through (API slot 0x13).
std::string tess_constants_path(const TessShader& s) {
    std::uint32_t reg = ~0u, table = ~0u;
    for (std::size_t at = block_of(s) + 24; at + 4 <= s.code.offset && at + 4 <= s.data.size(); at += 4) {
        std::uint32_t w;
        std::memcpy(&w, s.data.data() + at, 4);
        if ((w & 0xff) == 0x02 && ((w >> 8) & 0xff) == 0x13 && reg == ~0u) reg = (w >> 16) & 0xff;
        if ((w & 0xff) == 0x1b && table == ~0u) table = (w >> 16) & 0xff;
    }
    if (reg < 16) return "user_sgpr[" + std::to_string(reg) + "]";
    if (reg != ~0u && table < 16) return "user_sgpr[" + std::to_string(table) + "]->load(+" + std::to_string((reg - 16) * 4) + ")";
    return std::string();
}
// An LS's vertex inputs from its container's input table, as 32-bit floats.
std::vector<gcn::VertexElement> tess_ls_inputs(const TessShader& s) {
    for (std::size_t first = block_of(s) + 24; first + 4 <= s.code.offset; first += 4) {
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
bool tess_reads_lds(const gcn::Program& p) {
    for (const gcn::Inst& in : p.insts) {
        if (in.enc == gcn::Enc::DS && in.op != 53) return true;  // not ds_swizzle_b32
    }
    return false;
}
// The particles' LS and domain shaders (one control point, carried as stage
// attributes) and every game hull's LS, hull and domain shader in its window.
void tess_jobs(const std::string& bundle_path, std::vector<gcn::BundleEntry>& entries, std::vector<Job>& out, int& next_set) {
    std::string bundle = fs::path(bundle_path).filename().string();
    bundle = bundle.substr(0, bundle.find('.'));
    std::vector<TessShader> all;
    for (gcn::BundleEntry& e : entries) {
        TessShader s;
        s.name = e.name;
        if (!gcn::shader_code(e.data, s.code)) continue;
        std::size_t shdr = 0;
        while (shdr + 0x60 <= e.data.size() && shdr < 0x100 && std::memcmp(e.data.data() + shdr, "Shdr", 4) != 0) ++shdr;
        if (shdr + 0x60 > e.data.size()) continue;
        std::memcpy(s.regs, e.data.data() + shdr + 16, sizeof(s.regs));
        s.data = e.data;
        s.program = gcn::decode(s.code.words.data(), s.code.words.size());
        all.push_back(std::move(s));
    }
    const auto flat = [](std::string n) {
        for (char& c : n) {
            if (c == '\\' || c == '/') c = '~';
        }
        return n;
    };
    std::set<std::vector<std::uint32_t>> seen;
    for (const TessShader& s : all) {
        if (s.name.find("tessellate") == std::string::npos || !seen.insert(s.code.words).second) continue;
        if (s.code.type == 6) {
            const std::vector<gcn::VertexElement> els = tess_ls_inputs(s);
            if (els.empty()) continue;
            Job j;
            j.bundle = bundle;
            j.name = flat(s.name);
            j.kind = Kind::Ls;
            j.program = s.program;
            gcn::TranslateOptions& o = j.options;
            o.stage = gcn::Stage::Vertex;
            o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
            o.tess_lds_attributes = true;
            o.tess_attr_vec4s = 16;
            o.rsrc1 = s.regs[4];
            o.rsrc2 = s.regs[5];
            o.vertex_input = els;
            o.cb_ssbo = true;
            o.cb_no_fallback = true;
            j.tess.attr_vec4s = 16;
            out.push_back(std::move(j));
            continue;
        }
        if (s.code.type != 1 || !tess_reads_lds(s.program)) continue;
        Job j;
        j.bundle = bundle;
        j.name = flat(s.name);
        j.kind = Kind::Domain;
        j.program = s.program;
        gcn::TranslateOptions& o = j.options;
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
        j.tess.attr_vec4s = 16;
        out.push_back(std::move(j));
    }
    std::set<std::string> sets;
    for (const TessShader& hs : all) {
        if (hs.code.type != 7 || hs.code.words.size() < 300) continue;  // past the particles' constant hulls
        const std::string stem = hs.name.substr(0, hs.name.rfind('.'));
        const TessShader *ls = nullptr, *ds = nullptr;
        for (const TessShader& s : all) {
            if (s.name == stem + ".vpo") ls = &s;
            if (s.name == stem + ".dpo") ds = &s;
        }
        if (!ls || !ds) continue;
        std::string key;
        for (const TessShader* s : {ls, &hs, ds}) key.append(reinterpret_cast<const char*>(s->code.words.data()), s->code.words.size() * 4).push_back('|');
        if (!sets.insert(key).second) continue;
        const std::uint32_t ls_stride = ls->regs[7], cp = hs.regs[9], npc = hs.regs[11], cp_stride = hs.regs[12];
        const std::uint32_t inputs = cp * ls_stride, outputs = cp * cp_stride, consts = npc * 16;
        const std::uint32_t tc[12] = {ls_stride, cp_stride, 1, inputs, consts, inputs + outputs, outputs, 0, hs.regs[15], 0, 0, 0};
        const std::uint32_t window = (inputs + outputs + consts + 15) & ~15u;
        gcn::TessLift tl;
        tl.window = true;
        tl.control_points = cp;
        tl.ls_stride = ls_stride;
        tl.cp_stride = cp_stride;
        tl.patch_constants = npc;
        std::memcpy(tl.constants, tc, sizeof(tc));
        const std::uint32_t tf = hs.regs[6];
        const bool quads = (tf & 3) == 2;
        const int spacing = ((tf >> 2) & 7) == 2 ? 1 : ((tf >> 2) & 7) == 3 ? 2 : 0;
        const bool cw = ((tf >> 5) & 7) != 3;
        const int set = next_set++;
        {
            Job j;
            j.bundle = bundle;
            j.name = flat(hs.name);
            j.kind = Kind::Hull;
            j.program = hs.program;
            gcn::TranslateOptions& o = j.options;
            o.stage = gcn::Stage::TessControl;
            o.tess_role = gcn::TranslateOptions::TessRole::HullTcs;
            o.rsrc1 = hs.regs[4];
            o.rsrc2 = hs.regs[5];
            o.tess_patch_control_points = cp;
            o.tess_window = window;
            o.tess_lds_bound = true;
            o.tess_quads = quads;
            o.tess_spacing = spacing;
            o.tess_cw = cw;
            o.descriptor_set = 4;
            o.cb_ssbo = true;
            o.cb_no_fallback = true;
            o.cb_descriptor_set = 4;
            o.cb_binding_base = 32;
            j.tess = tl;
            j.tess.own_params = true;
            j.tess.params_binding = 1;
            j.tess.constants_path = tess_constants_path(hs);
            j.set = set;
            out.push_back(std::move(j));
        }
        {
            Job j;
            j.bundle = bundle;
            j.name = flat(ds->name);
            j.kind = Kind::Domain;
            j.program = ds->program;
            gcn::TranslateOptions& o = j.options;
            o.stage = gcn::Stage::TessEval;
            o.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
            o.rsrc1 = ds->regs[4];
            o.rsrc2 = ds->regs[5];
            o.vs_out_cntl = ds->regs[8];
            o.tess_patch_control_points = cp;
            o.tess_window = window;
            o.tess_lds_bound = true;
            o.tess_quads = quads;
            o.tess_spacing = spacing;
            o.tess_cw = cw;
            o.cb_descriptor_set = 2;
            o.cb_ssbo = true;
            o.link_outputs = true;
            for (std::uint8_t k = 0; k < 32; ++k) o.output_links.push_back(k);
            const gcn::TranslateResult dpaths = gcn::translate(ds->program, o);
            for (const gcn::PredictedImage& pi : gcn::predict_image_dims(ds->program, dpaths)) o.image_dims.emplace_back(pi.dim, pi.arrayed);
            o.sampler_force_unnormalized.assign(dpaths.samplers.size(), false);
            o.cb_no_fallback = true;
            j.tess = tl;
            j.tess.constants_path = tess_constants_path(*ds);
            j.set = set;
            out.push_back(std::move(j));
        }
        {
            Job j;
            j.bundle = bundle;
            j.name = flat(ls->name);
            j.kind = Kind::Ls;
            j.program = ls->program;
            gcn::TranslateOptions& o = j.options;
            o.stage = gcn::Stage::Vertex;
            o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
            o.tess_lds_attributes = true;
            o.tess_attr_vec4s = ls_stride / 16;
            o.rsrc1 = ls->regs[4];
            o.rsrc2 = ls->regs[5];
            o.vertex_input = tess_ls_inputs(*ls);
            o.cb_ssbo = true;
            o.cb_no_fallback = true;
            o.descriptor_set = 4;
            o.cb_descriptor_set = 4;
            o.cb_binding_base = 16;
            j.tess = tl;
            j.tess.attr_vec4s = ls_stride / 16;
            j.tess.own_params = true;
            j.tess.params_binding = 0;
            j.set = set;
            if (!o.vertex_input.empty()) out.push_back(std::move(j));
        }
    }
}

struct Totals {
    std::mutex mu;
    std::size_t programs = 0, lifted = 0, emitted = 0, compiled = 0, valid = 0, same_interface = 0;
    std::map<std::string, std::size_t> not_lifted, not_emitted, not_compiled, invalid, interface_diffs;
    std::uint64_t emit_us = 0, compile_us = 0, max_compile_us = 0;
    std::string slowest;
    std::vector<std::uint32_t> compile_ms;  // per compiled program
    std::uint64_t lift_words = 0, dxc_words = 0, hlsl_bytes = 0;
};

// The game's shader bundles under a game folder: DIR/dvdroot_ps4/shader (or
// DIR/shader, or DIR itself), each from DIR-UPDATE when the 1.09 update is
// kept beside the game, as bbhost reads it.
std::vector<std::string> app0_bundles(const std::string& app0) {
    fs::path root(app0);
    while (!root.empty() && !root.has_filename()) root = root.parent_path();  // a trailing separator
    fs::path dir;
    std::string rel;
    for (const char* sub : {"dvdroot_ps4/shader", "shader", ""}) {
        const fs::path d = *sub ? root / sub : root;
        std::error_code ec;
        if (fs::is_directory(d, ec)) {
            dir = d;
            rel = sub;
            break;
        }
    }
    std::vector<std::string> out;
    if (dir.empty()) return out;
    const fs::path update = root.parent_path() / (root.filename().string() + "-UPDATE");
    std::set<std::string> names;
    for (const fs::path& d : {update / rel, dir}) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        for (const auto& e : fs::directory_iterator(d, ec)) {
            const std::string n = e.path().filename().string();
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".dcx") == 0 && n.find(".shaderbnd") != std::string::npos && names.insert(n).second)
                out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) { return fs::path(a).filename() < fs::path(b).filename(); });
    return out;
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// The programs each worker is building pipelines of, for a driver that
// crashes inside one: the handler names them before the process ends.
constexpr int kMaxWorkers = 64;
std::atomic<const char*> g_in_flight[kMaxWorkers];
extern "C" void on_crash(int sig) {
    std::fputs("gcn2hlsl: crashed while building the pipelines of:", stderr);
    for (std::atomic<const char*>& f : g_in_flight) {
        const char* n = f.load();
        if (n) {
            std::fputs(" ", stderr);
            std::fputs(n, stderr);
        }
    }
    std::fputs("\n", stderr);
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

#ifdef GCN2HLSL_HAVE_VULKAN
// The driver's verdicts on one program, the lift's module and DXC's.
struct Probed {
    bool lift_tried = false, dxc_tried = false;
    probe::Verdict lift, dxc;
    std::uint64_t us = 0;  // the driver's time on both
    std::vector<std::uint32_t> lift_spirv, dxc_spirv;  // a hull set's member, kept for the set's pipeline
};
// Driver tallies for one kind of stage.
struct DriverTally {
    std::size_t lifts = 0, lift_built = 0, lift_refused = 0, lift_untried = 0;
    std::size_t dxc = 0, dxc_built = 0, dxc_refused = 0, dxc_untried = 0, dxc_only = 0, both = 0, lift_only = 0, notes = 0, warnings = 0;
    std::uint64_t us = 0;
    void add(const Probed& p) {
        us += p.us;
        if (p.lift_tried) {
            ++lifts;
            (p.lift.ok ? lift_built : p.lift.skipped ? lift_untried : lift_refused)++;
        }
        if (p.dxc_tried) {
            ++dxc;
            (p.dxc.ok ? dxc_built : p.dxc.skipped ? dxc_untried : dxc_refused)++;
            if (p.dxc.refused() && p.lift_tried && p.lift.ok) ++dxc_only;
            if (p.dxc.refused() && p.lift_tried && p.lift.refused()) ++both;
            if (p.dxc.ok && p.lift_tried && p.lift.refused()) ++lift_only;
        }
        notes += !p.lift.note.empty() + !p.dxc.note.empty();
        warnings += !p.lift.warning.empty() + !p.dxc.warning.empty();
    }
};
#endif

}  // namespace

// --spv: one module from a file.
int one_module(const char* in, const char* prefix, int opt_level) {
    std::vector<std::uint8_t> bytes;
    if (!gcn::read_file(in, bytes) || bytes.size() < 20 || bytes.size() % 4) {
        std::fprintf(stderr, "cannot read %s as SPIR-V\n", in);
        return 2;
    }
    std::vector<std::uint32_t> lifted(bytes.size() / 4);
    std::memcpy(lifted.data(), bytes.data(), bytes.size());
    gcn::HlslOptions hopt;
    hopt.title = fs::path(in).filename().string();
    const gcn::HlslResult h = gcn::emit_hlsl(lifted, hopt);
    if (!h.source.empty()) write_bytes(std::string(prefix) + ".hlsl", h.source.data(), h.source.size());
    if (!h.ok()) {
        std::printf("not emitted: %s\n", h.errors[0].c_str());
        return 1;
    }
    gpu::Dxc dxc;
    if (!dxc.load()) {
        std::fprintf(stderr, "DXC: %s\n", dxc.error().c_str());
        return 2;
    }
    gpu::DxcOptions dopt;
    dopt.optimize = opt_level;
    std::vector<std::uint32_t> spirv;
    std::string error;
    const auto t0 = std::chrono::steady_clock::now();
    if (!dxc.compile(h, dopt, spirv, error)) {
        std::printf("not compiled: %s\n", error.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    write_bytes(std::string(prefix) + ".dxc.spv", spirv.data(), spirv.size() * 4);
    if (const std::string v = validate(spirv); !v.empty()) {
        std::printf("invalid: %s\n", v.c_str());
        return 1;
    }
    if (const std::string d = gcn::hlsl_interface_diff(lifted, spirv); !d.empty()) {
        std::printf("interface: %s\n", d.c_str());
        return 1;
    }
    if (const std::string d = gcn::hlsl_operation_diff(lifted, spirv); !d.empty()) {
        std::printf("operations: %s\n", d.c_str());
        return 1;
    }
    std::printf("ok %.0f ms, %zu words (lifted %zu)\n", ms, spirv.size(), lifted.size());
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::string(argv[1]) == "--spv" && std::string(argv[3]) == "--out") {
        return one_module(argv[2], argv[4], argc >= 6 && std::string(argv[5]) == "--O0" ? 0 : 3);
    }
    bool ps = false, vs = false, clip = false, use_dxc = true, tess = false, invariant = false, cs = false;
    int opt_level = 3;
    int sample = 1;  // every n-th program
    std::string passes;
    unsigned jobs = 0;  // --jobs; else half the machine's threads, or all of them for --sweep (the machine is the sweep's)
    std::set<std::string> only;
    const char* out_dir = nullptr;
    const char* list_path = nullptr;
    std::vector<std::string> bundles;
    bool device = false, validate_layer = false, monolithic = false, driver_cache = false, sweep = false;
    std::string gpu_select, app0;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--sweep") sweep = device = true;
        else if (arg == "--device") device = true;
        else if (arg == "--gpu" && a + 1 < argc) gpu_select = argv[++a];
        else if (arg == "--validate") validate_layer = true;
        else if (arg == "--monolithic") monolithic = true;
        else if (arg == "--driver-cache") driver_cache = true;
        else if (arg == "--app0" && a + 1 < argc) app0 = argv[++a];
        else if (arg == "--ps") ps = true;
        else if (arg == "--vs") vs = true;
        else if (arg == "--clip") clip = true;
        else if (arg == "--tess") tess = true;
        else if (arg == "--cs") cs = true;
        else if (arg == "--runtime-offsets") gcn::set_runtime_sample_offsets(true);
        else if (arg == "--native-half-rtz") gcn::set_native_half_rtz(true);
        else if (arg == "--invariant") invariant = true;
        else if (arg == "--O0") opt_level = 0;
        else if (arg == "--oconfig" && a + 1 < argc) passes = argv[++a];
        else if (arg == "--sample" && a + 1 < argc) sample = std::max(1, std::atoi(argv[++a]));
        else if (arg == "--no-dxc") use_dxc = false;
        else if (arg == "--jobs" && a + 1 < argc) jobs = static_cast<unsigned>(std::max(1, std::atoi(argv[++a])));
        else if (arg == "--out" && a + 1 < argc) out_dir = argv[++a];
        else if (arg == "--list" && a + 1 < argc) list_path = argv[++a];
        else if (arg == "--only" && a + 1 < argc) {
            const std::string s = argv[++a];
            for (std::size_t at = 0; at <= s.size();) {
                const std::size_t comma = std::min(s.find(',', at), s.size());
                if (comma > at) only.insert(s.substr(at, comma - at));
                at = comma + 1;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            return 2;
        } else {
            bundles.push_back(arg);
        }
    }
    if (!app0.empty()) {
        const std::vector<std::string> found = app0_bundles(app0);
        if (found.empty()) {
            std::fprintf(stderr, "%s: no shader bundles (dvdroot_ps4/shader/*.shaderbnd*.dcx)\n", app0.c_str());
            return 2;
        }
        bundles.insert(bundles.end(), found.begin(), found.end());
    }
    if (bundles.empty()) {
        std::fprintf(stderr, "usage: gcn2hlsl [--ps] [--vs] [--tess] [--clip] [--runtime-offsets] [--native-half-rtz] [--invariant] [--O0] [--jobs N] [--only NAME,..] [--out DIR] [--list FILE] [--no-dxc] <bundle.dcx>...\n"
                             "       gcn2hlsl --sweep --app0 <game folder> [--gpu N|NAME] [--list FILE] [--validate] [--monolithic] [--driver-cache]\n");
        return 2;
    }
    if (sweep) ps = vs = tess = cs = true;
    if (!jobs) jobs = std::max(1u, std::thread::hardware_concurrency() / (sweep ? 1 : 2));
    if (!ps && !vs && !tess && !cs) ps = vs = true;
    if (out_dir) fs::create_directories(out_dir);
    if (device && !use_dxc) {
        std::fprintf(stderr, "--device and --sweep need DXC\n");
        return 2;
    }
    if (device && !driver_cache) {
        // The driver's shader disk cache: neither read nor filled by the sweep.
        set_env("__GL_SHADER_DISK_CACHE", "0");
        set_env("MESA_SHADER_CACHE_DISABLE", "true");
    }
    gpu::Dxc dxc;
    if (use_dxc && !dxc.load()) {
        std::fprintf(stderr, "DXC: %s\n", dxc.error().c_str());
        return 2;
    }
    if (use_dxc) std::printf("DXC: %s (%s)\n", dxc.version().c_str(), dxc.path().c_str());
#ifdef GCN2HLSL_HAVE_VULKAN
    // Before any program is lifted: the device sets the lifter's switches.
    std::unique_ptr<probe::Prober> prober;
    if (device) {
        prober = std::make_unique<probe::Prober>(dxc);
        std::string error;
        if (!prober->init(gpu_select, validate_layer, monolithic, error)) {
            std::fprintf(stderr, "device: %s\n", error.c_str());
            return 2;
        }
        std::printf("%s\n", prober->info().describe().c_str());
        std::signal(SIGSEGV, on_crash);
        std::signal(SIGABRT, on_crash);
        std::signal(SIGILL, on_crash);
    }
#else
    if (device) {
        std::fprintf(stderr, "--device and --sweep: this gcn2hlsl was built without Vulkan\n");
        return 2;
    }
    (void)validate_layer;
    (void)monolithic;
    (void)gpu_select;
#endif

    // The programs, with the options the renderer lifts them with at creation.
    std::vector<Job> work;
    std::size_t seen = 0;
    int next_set = 0;
    for (const std::string& path : bundles) {
        std::vector<gcn::BundleEntry> entries;
        if (!load_bundle(path, entries)) return 1;
        if (tess) {
            std::vector<Job> t;
            tess_jobs(path, entries, t, next_set);
            for (Job& j : t) {
                if (!only.empty() && !only.count(j.name)) continue;
                if (seen++ % static_cast<std::size_t>(sample) == 0) work.push_back(std::move(j));
            }
        }
        for (gcn::BundleEntry& e : entries) {
            gcn::ShaderCode code;
            if (!gcn::shader_code(e.data, code)) continue;
            std::size_t shdr = 0;
            while (shdr + 0x60 <= e.data.size() && shdr < 0x100 && std::memcmp(e.data.data() + shdr, "Shdr", 4) != 0) ++shdr;
            if (shdr + 0x60 > e.data.size() || shdr >= 0x100) continue;
            const bool is_ps = code.type == 2, is_vs = code.type == 1, is_cs = code.type == 4;
            if (!(is_ps && ps) && !(is_vs && vs) && !(is_cs && cs)) continue;
            std::string base = e.name;
            if (const std::size_t slash = base.find_last_of("\\/"); slash != std::string::npos) base = base.substr(slash + 1);
            if (!only.empty() && !only.count(base) && !only.count(e.name)) continue;
            // The entry's folders too: one program name can be in several (Lightprepass0\, Deferred\).
            std::string full = e.name;
            for (char& c : full) {
                if (c == '\\' || c == '/') c = '~';
            }
            std::uint32_t regs[16];
            std::memcpy(regs, e.data.data() + shdr + 16, sizeof(regs));
            Job j;
            j.bundle = fs::path(path).filename().string();
            j.bundle = j.bundle.substr(0, j.bundle.find('.'));
            j.name = full;
            j.kind = is_cs ? Kind::Compute : is_vs ? Kind::Vertex : Kind::Pixel;
            j.program = gcn::decode(code.words.data(), code.words.size());
            const std::size_t gnm = shdr + 16;
            if (is_cs) {
                // gpu.cpp precompile_compute: the dispatch's options from what the container gives.
                gcn::TranslateOptions paths_options;
                paths_options.stage = gcn::Stage::Compute;
                paths_options.rsrc1 = regs[4];
                paths_options.rsrc2 = regs[5];
                const gcn::TranslateResult paths = gcn::translate(j.program, paths_options);
                gcn::TranslateOptions& o = j.options;
                o.stage = gcn::Stage::Compute;
                o.gds_counters = true;
                o.rsrc1 = regs[4];
                o.rsrc2 = regs[5];
                for (const gcn::PredictedImage& pi : gcn::predict_image_dims(j.program, paths)) o.image_dims.emplace_back(pi.dim, pi.arrayed);
                o.sampler_force_unnormalized.assign(paths.samplers.size(), false);
                o.max_lds_bytes = 32768;
                o.cs_threads[0] = regs[6];
                o.cs_threads[1] = regs[7];
                o.cs_threads[2] = regs[8];
                o.exec_known = true;
            } else if (is_ps) {
                // render.cpp precompile_ps
                gcn::TranslateOptions paths_options;
                paths_options.stage = gcn::Stage::Pixel;
                paths_options.rsrc1 = regs[4];
                paths_options.rsrc2 = regs[5];
                const gcn::TranslateResult paths = gcn::translate(j.program, paths_options);
                gcn::TranslateOptions& o = j.options;
                o.stage = gcn::Stage::Pixel;
                o.rsrc1 = regs[4];
                o.rsrc2 = regs[5];
                o.ps_input_ena = regs[8];
                o.descriptor_set = 1;
                o.cb_descriptor_set = 2;
                o.cb_binding_base = gcn::kBindingStorageBuffer0 + gcn::kMaxBuffers;
                o.cb_ssbo = true;
                o.cb_no_fallback = true;
                o.exec_known = true;
                for (const gcn::PredictedImage& pi : gcn::predict_image_dims(j.program, paths)) o.image_dims.emplace_back(pi.dim, pi.arrayed);
                o.sampler_force_unnormalized.assign(paths.samplers.size(), false);
                const std::uint32_t declared_inputs = e.data[gnm + 0x38];
                const std::size_t input_table = gnm + 0x3c + static_cast<std::size_t>(e.data[gnm + 3]) * 4;
                std::uint32_t read_inputs = 0, registers[32] = {};
                bool inputs_ok = true;
                for (std::uint32_t k : paths.ps_inputs) {
                    if (k >= 32 || k >= declared_inputs || input_table + 2 * (k + 1) > e.data.size()) {
                        inputs_ok = false;
                        break;
                    }
                    std::uint16_t word;
                    std::memcpy(&word, e.data.data() + input_table + 2 * k, 2);
                    if (((word >> 12) | (word >> 10)) & 1) o.ps_flat_mask |= 1u << k;
                    const std::uint32_t semantic = word & 0xff;
                    if (semantic < 0xf || semantic - 0xf >= 32) {
                        inputs_ok = false;
                        break;
                    }
                    read_inputs |= 1u << k;
                    registers[k] = semantic - 0xf;
                }
                if (inputs_ok) {
                    for (std::uint32_t k = 0; k < 32; ++k) {
                        if (!((read_inputs >> k) & 1)) continue;
                        while (o.ps_input_map.size() < k) o.ps_input_map.push_back(static_cast<std::uint8_t>(o.ps_input_map.size()));
                        o.ps_input_map.push_back(static_cast<std::uint8_t>(registers[k] & 0x1f));
                    }
                }
                if (clip) {
                    int used = -1;
                    for (std::uint8_t l : o.ps_input_map) used = std::max(used, static_cast<int>(l));
                    for (const gcn::Inst& in : j.program.insts) {
                        if (in.enc == gcn::Enc::VINTRP && in.attr < 32) used = std::max(used, static_cast<int>(in.attr));
                    }
                    if (used + 1 <= 29) o.ps_clip_discard = 4u | static_cast<std::uint32_t>(used + 1) << 8;
                }
            } else {
                // render.cpp precompile_vs
                const std::size_t inputs = e.data[gnm + 0x24], input_table = gnm + 0x28 + static_cast<std::size_t>(e.data[gnm + 3]) * 4;
                bool ok = inputs > 0 && inputs <= 16 && input_table + inputs * 4 <= e.data.size();
                std::vector<gcn::VertexElement> elements;
                for (std::size_t k = 0; ok && k < inputs; ++k) {
                    gcn::VertexElement el;
                    el.location = static_cast<std::uint32_t>(k);
                    el.vdata = e.data[input_table + 4 * k + 1];
                    el.count = e.data[input_table + 4 * k + 2];
                    if (el.count == 0 || el.count > 4) ok = false;
                    elements.push_back(el);
                }
                if (!ok) continue;  // the renderer compiles none at creation (no usable input table)
                const std::uint32_t rsrc1 = (regs[4] & ~(3u << 24)) | (3u << 24), rsrc2 = regs[5];
                gcn::TranslateOptions paths_options;
                paths_options.stage = gcn::Stage::Vertex;
                paths_options.rsrc1 = rsrc1;
                paths_options.rsrc2 = rsrc2;
                const gcn::TranslateResult paths = gcn::translate(j.program, paths_options);
                gcn::TranslateOptions& o = j.options;
                o.stage = gcn::Stage::Vertex;
                o.rsrc1 = rsrc1;
                o.rsrc2 = rsrc2;
                o.vs_out_cntl = regs[8];
                if (clip) {
                    const std::uint32_t c = regs[8];
                    if ((c & (1u << 22)) && (c & 0xf) && !(c & 0xfff0) && !(c & (1u << 23))) {
                        int used = -1;
                        for (const gcn::Inst& in : j.program.insts) {
                            if (in.enc == gcn::Enc::EXP && in.tgt >= 32 && in.tgt < 64) used = std::max(used, static_cast<int>(in.tgt) - 32);
                        }
                        if (used + 1 <= 29) {
                            o.vs_out_cntl |= gcn::TranslateOptions::kVsOutCntlClipVarying |
                                             static_cast<std::uint32_t>(used + 1) << gcn::TranslateOptions::kVsOutCntlClipLocationShift;
                        }
                    }
                }
                o.vertex_input = elements;
                o.vertex_formats_from_params = true;
                o.link_outputs = true;
                o.invariant_position = invariant;
                for (std::uint8_t k = 0; k < 32; ++k) o.output_links.push_back(k);
                o.descriptor_set = 0;
                o.cb_descriptor_set = 2;
                for (const gcn::PredictedImage& pi : gcn::predict_image_dims(j.program, paths)) o.image_dims.emplace_back(pi.dim, pi.arrayed);
                o.sampler_force_unnormalized.assign(paths.samplers.size(), false);
                o.cb_ssbo = true;
                o.cb_no_fallback = true;
                o.exec_known = true;
            }
            j.container = std::move(e.data);
            if (seen++ % static_cast<std::size_t>(sample) == 0) work.push_back(std::move(j));
        }
    }

    Totals t;
    FILE* list = list_path ? std::fopen(list_path, "w") : nullptr;
    std::mutex list_mu;
    std::atomic<std::size_t> next{0};
    gpu::DxcOptions dopt;
    dopt.optimize = opt_level;
    dopt.passes = passes;
#ifdef GCN2HLSL_HAVE_VULKAN
    // --device: the driver's verdicts by program, and the time spent on them.
    std::vector<Probed> probed(prober ? work.size() : 0);
    std::atomic<std::uint64_t> probe_us{0};
    const auto probe_one = [&](int wid, const Job& j, const std::vector<std::uint32_t>& m, std::uint32_t wave64_needs,
                               const std::vector<std::uint32_t>& lift, std::uint64_t& us) {
        g_in_flight[wid].store(j.name.c_str());
        const auto p0 = std::chrono::steady_clock::now();
        probe::Verdict v;
        switch (j.kind) {
        case Kind::Pixel: v = prober->pixel(m, wave64_needs); break;
        case Kind::Vertex:
        case Kind::Ls: v = prober->vertex(m); break;
        case Kind::Domain: v = prober->domain(m, lift); break;
        case Kind::Compute: v = prober->compute(m); break;
        case Kind::Hull:
            v.ok = false;
            v.skipped = true;
            v.what = "a hull shader with no LS and domain shader of its own";
            break;
        }
        const std::uint64_t spent = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - p0).count());
        probe_us.fetch_add(spent, std::memory_order_relaxed);
        us += spent;
        g_in_flight[wid].store(nullptr);
        return v;
    };
#endif
    const auto worker = [&](int wid) {
        (void)wid;
        for (;;) {
            const std::size_t i = next.fetch_add(1);
            if (i >= work.size()) return;
            const Job& j = work[i];
            const char* stage = kind_name(j.kind);
            std::string result, reason;
            std::uint64_t hlsl_hash = 0;  // FNV-1a of the printed source (a regression baseline for the printer)
            std::uint64_t ms = 0;
            const auto line = [&] {
                if (!list) return;
                std::string driver;
#ifdef GCN2HLSL_HAVE_VULKAN
                if (prober) {
                    const Probed& p = probed[i];
                    driver = j.set >= 0 ? "lift=set dxc=set " : "lift=" + (p.lift_tried ? p.lift.token() : std::string("-")) + " dxc=" +
                                                                    (p.dxc_tried ? p.dxc.token() : std::string("-")) + " ";
                    // What the driver said of a refusal, a retry or a validation error, after the HLSL path's own reason.
                    for (const probe::Verdict* v : {&p.lift, &p.dxc}) {
                        const std::string said = v == &p.lift ? "lift: " : "dxc: ";
                        if (!v->ok) reason += (reason.empty() ? "" : "; ") + said + v->str();
                        else if (!v->note.empty()) reason += (reason.empty() ? "" : "; ") + said + v->note;
                        if (!v->warning.empty()) reason += (reason.empty() ? "" : "; ") + said + "validation: " + v->warning;
                    }
                }
#endif
                std::lock_guard<std::mutex> lk(list_mu);
                std::fprintf(list, "%s %s/%s %s %llu %016llx %s%s\n", stage, j.bundle.c_str(), j.name.c_str(), result.c_str(), static_cast<unsigned long long>(ms),
                             static_cast<unsigned long long>(hlsl_hash), driver.c_str(), reason.c_str());
                std::fflush(list);
            };
            {
                std::lock_guard<std::mutex> lk(t.mu);
                ++t.programs;
            }
            if (!j.program.errors.empty()) continue;
            const gcn::TranslateResult ref = gcn::translate(j.program, j.options);
            if (!ref.ok()) {
                result = "untranslated", reason = ref.errors[0];
                line();
                continue;
            }
            gcn::LiftResult l;
            switch (j.kind) {
            case Kind::Pixel: l = gcn::lift_pixel_shader(j.program, j.options, ref); break;
            case Kind::Vertex: l = gcn::lift_vertex_shader(j.program, j.options, ref); break;
            case Kind::Ls: l = gcn::lift_ls_shader(j.program, j.options, ref, j.tess); break;
            case Kind::Hull: l = gcn::lift_hull_shader(j.program, j.options, ref, j.tess); break;
            case Kind::Domain: l = gcn::lift_domain_shader(j.program, j.options, ref, j.tess); break;
            case Kind::Compute: l = gcn::lift_compute_shader(j.program, j.options, ref); break;
            }
            if (!l.ok()) {
                result = "not-lifted", reason = l.rejections.empty() ? "no module" : l.rejections[0];
                std::lock_guard<std::mutex> lk(t.mu);
                t.not_lifted[reason_key(reason)]++;
                line();
                continue;
            }
            if (j.kind == Kind::Pixel || j.kind == Kind::Vertex) {
                gcn::ShaderReflection refl;
                if (gcn::parse_reflection(j.container, refl)) gcn::name_lift(l, ref, refl);
            }
            {
                std::lock_guard<std::mutex> lk(t.mu);
                ++t.lifted;
                t.lift_words += l.spirv.size();
            }
#ifdef GCN2HLSL_HAVE_VULKAN
            // The lift's own pipelines: what the game builds where it keeps the lift.
            if (prober) {
                Probed& p = probed[i];
                if (j.set >= 0) {
                    p.lift_spirv = l.spirv;  // the set's pipeline, below
                } else {
                    p.lift = probe_one(wid, j, l.spirv, ref.wave64_needs, l.spirv, p.us);
                    p.lift_tried = true;
                }
            }
#endif
            const fs::path stem = out_dir ? fs::path(out_dir) / j.bundle / (j.name + "." + stage) : fs::path();
            if (out_dir) fs::create_directories(stem.parent_path());
            if (out_dir) write_bytes(stem.string() + ".lift.spv", l.spirv.data(), l.spirv.size() * 4);
            const auto e0 = std::chrono::steady_clock::now();
            gcn::HlslOptions hopt;
            static const char* const what[] = {"pixel shader", "vertex shader", "LS as the vertex stage", "hull shader as the control stage",
                                               "domain shader as the evaluation stage", "compute shader"};
            hopt.title = j.name + " (" + what[static_cast<int>(j.kind)] + ")";
            hopt.names = l.names.buffers.empty() && l.names.images.empty() ? nullptr : &l.names;
            hopt.tess_quads = j.options.tess_quads;
            hopt.tess_spacing = j.options.tess_spacing;
            hopt.tess_cw = j.options.tess_cw;
            const gcn::HlslResult h = gcn::emit_hlsl(l.spirv, hopt);
            const std::uint64_t emit_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - e0).count());
            if (out_dir && !h.source.empty()) write_bytes(stem.string() + ".hlsl", h.source.data(), h.source.size());
            hlsl_hash = 1469598103934665603ull;
            for (const char c : h.source) hlsl_hash = (hlsl_hash ^ static_cast<std::uint8_t>(c)) * 1099511628211ull;
            if (!h.ok()) {
                result = "not-emitted", reason = h.errors[0];
                std::lock_guard<std::mutex> lk(t.mu);
                t.not_emitted[reason_key(reason)]++;
                t.emit_us += emit_us;
                line();
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(t.mu);
                ++t.emitted;
                t.emit_us += emit_us;
                t.hlsl_bytes += h.source.size();
            }
            if (!use_dxc) {
                result = "emitted";
                line();
                continue;
            }
            std::vector<std::uint32_t> spirv;
            std::string error;
            const auto c0 = std::chrono::steady_clock::now();
            const bool compiled = dxc.compile(h, dopt, spirv, error);
            const std::uint64_t us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - c0).count());
            ms = us / 1000;
            if (!compiled) {
                result = "not-compiled", reason = error;
                std::lock_guard<std::mutex> lk(t.mu);
                t.not_compiled[reason_key(error.substr(0, error.find('\n')))]++;
                line();
                continue;
            }
            if (out_dir) write_bytes(stem.string() + ".dxc.spv", spirv.data(), spirv.size() * 4);
            {
                std::lock_guard<std::mutex> lk(t.mu);
                ++t.compiled;
                t.compile_us += us;
                t.compile_ms.push_back(static_cast<std::uint32_t>(ms));
                t.dxc_words += spirv.size();
                if (us > t.max_compile_us) {
                    t.max_compile_us = us;
                    t.slowest = j.name + "." + stage;
                }
            }
            const std::string v = validate(spirv);
            if (!v.empty()) {
                result = "invalid", reason = v;
                std::lock_guard<std::mutex> lk(t.mu);
                t.invalid[reason_key(v)]++;
                line();
                continue;
            }
            std::string diff = gcn::hlsl_interface_diff(l.spirv, spirv);
            if (diff.empty()) {
                const std::string ops = gcn::hlsl_operation_diff(l.spirv, spirv);
                if (!ops.empty()) diff = "operations: " + ops;
            }
#ifdef GCN2HLSL_HAVE_VULKAN
            // DXC's module, where the HLSL path would take it.
            if (prober && diff.empty()) {
                Probed& p = probed[i];
                if (j.set >= 0) {
                    p.dxc_spirv = spirv;
                } else {
                    p.dxc = probe_one(wid, j, spirv, ref.wave64_needs, l.spirv, p.us);
                    p.dxc_tried = true;
                }
            }
#endif
            {
                std::lock_guard<std::mutex> lk(t.mu);
                ++t.valid;
                if (diff.empty()) {
                    ++t.same_interface;
                    result = "ok";
                } else {
                    t.interface_diffs[reason_key(diff)]++;
                    result = diff.rfind("operations", 0) == 0 ? "operations" : "interface", reason = diff;
                }
            }
            line();
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    jobs = std::min<unsigned>(jobs, kMaxWorkers);
    for (unsigned k = 0; k < jobs; ++k) threads.emplace_back(worker, static_cast<int>(k));
    for (std::thread& th : threads) th.join();
    threads.clear();
#ifdef GCN2HLSL_HAVE_VULKAN
    // A game hull's LS, hull and domain shader as one native hull pipeline: the lifts', and DXC's where all three are taken.
    struct SetProbe {
        std::size_t ls = SIZE_MAX, hs = SIZE_MAX, ds = SIZE_MAX;
        bool lift_tried = false, dxc_tried = false;
        probe::Verdict lift, dxc;
        std::uint64_t us = 0;
    };
    std::vector<SetProbe> sets;
    if (prober) {
        std::map<int, SetProbe> by_set;
        for (std::size_t i = 0; i < work.size(); ++i) {
            const Job& j = work[i];
            if (j.set < 0) continue;
            SetProbe& sp = by_set[j.set];
            (j.kind == Kind::Ls ? sp.ls : j.kind == Kind::Hull ? sp.hs : sp.ds) = i;
        }
        for (auto& kv : by_set) sets.push_back(kv.second);
        std::atomic<std::size_t> next_probe{0};
        const auto set_worker = [&](int wid) {
            for (;;) {
                const std::size_t k = next_probe.fetch_add(1);
                if (k >= sets.size()) return;
                SetProbe& sp = sets[k];
                if (sp.ls == SIZE_MAX || sp.hs == SIZE_MAX || sp.ds == SIZE_MAX) continue;  // part of a set left out (--only, --sample)
                const Probed &pl = probed[sp.ls], &ph = probed[sp.hs], &pd = probed[sp.ds];
                if (pl.lift_spirv.empty() || ph.lift_spirv.empty() || pd.lift_spirv.empty()) continue;
                const std::uint32_t cp = work[sp.hs].tess.control_points;
                g_in_flight[wid].store(work[sp.hs].name.c_str());
                const auto p0 = std::chrono::steady_clock::now();
                sp.lift = prober->hull_set(pl.lift_spirv, ph.lift_spirv, pd.lift_spirv, cp);
                sp.lift_tried = true;
                if (!pl.dxc_spirv.empty() && !ph.dxc_spirv.empty() && !pd.dxc_spirv.empty()) {
                    sp.dxc = prober->hull_set(pl.dxc_spirv, ph.dxc_spirv, pd.dxc_spirv, cp);
                    sp.dxc_tried = true;
                }
                sp.us = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - p0).count());
                probe_us.fetch_add(sp.us, std::memory_order_relaxed);
                g_in_flight[wid].store(nullptr);
            }
        };
        for (unsigned k = 0; k < jobs; ++k) threads.emplace_back(set_worker, static_cast<int>(k));
        for (std::thread& th : threads) th.join();
        if (list) {
            for (const SetProbe& sp : sets) {
                if (!sp.lift_tried) continue;
                const Job& h = work[sp.hs];
                std::string said;
                for (const probe::Verdict* v : {&sp.lift, &sp.dxc}) {
                    if (v == &sp.dxc && !sp.dxc_tried) continue;
                    const std::string who = v == &sp.lift ? "lift: " : "dxc: ";
                    if (!v->ok) said += (said.empty() ? "" : "; ") + who + v->str();
                    if (!v->warning.empty()) said += (said.empty() ? "" : "; ") + who + "validation: " + v->warning;
                }
                std::fprintf(list, "set %s/%s - 0 0 lift=%s dxc=%s %s\n", h.bundle.c_str(), h.name.c_str(), sp.lift.token().c_str(),
                             sp.dxc_tried ? sp.dxc.token().c_str() : "-", said.c_str());
            }
        }
    }
#endif
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (list) std::fclose(list);

    std::printf("programs=%zu lifted=%zu emitted=%zu compiled=%zu valid=%zu same_interface=%zu (%.1f s wall, %u jobs)\n", t.programs, t.lifted,
                t.emitted, t.compiled, t.valid, t.same_interface, wall, jobs);
    std::printf("words: lifted %llu, dxc %llu; HLSL %.1f MiB; emit %.2f s total\n", static_cast<unsigned long long>(t.lift_words),
                static_cast<unsigned long long>(t.dxc_words), t.hlsl_bytes / 1048576.0, t.emit_us / 1e6);
    if (!t.compile_ms.empty()) {
        std::vector<std::uint32_t> ms = t.compile_ms;
        std::sort(ms.begin(), ms.end());
        std::printf("DXC %s: %.1f s summed over %zu programs; per program median %u ms, p90 %u ms, p99 %u ms, max %.0f ms (%s)\n",
                    (passes.empty() ? "-O" + std::to_string(opt_level) : "-Oconfig=" + passes).c_str(),
                    t.compile_us / 1e6, ms.size(), ms[ms.size() / 2], ms[ms.size() * 9 / 10], ms[ms.size() * 99 / 100], t.max_compile_us / 1e3,
                    t.slowest.c_str());
    }
    const auto dump = [](const char* what, const std::map<std::string, std::size_t>& m) {
        std::vector<std::pair<std::size_t, std::string>> v;
        for (const auto& [k, n] : m) v.push_back({n, k});
        std::sort(v.rbegin(), v.rend());
        for (std::size_t i = 0; i < v.size() && i < 30; ++i) std::printf("  %s x%zu: %s\n", what, v[i].first, v[i].second.c_str());
    };
    dump("not lifted", t.not_lifted);
    dump("not emitted", t.not_emitted);
    dump("not compiled", t.not_compiled);
    dump("invalid", t.invalid);
    dump("interface or operations", t.interface_diffs);
    std::size_t dxc_only = 0;
#ifdef GCN2HLSL_HAVE_VULKAN
    if (prober) {
        // The driver's verdicts: each stage kind on its own, the hull sets as one pipeline each.
        std::map<std::string, DriverTally> by_kind;
        std::map<std::string, std::size_t> untried;
        std::vector<std::string> refused_dxc, refused_lift;
        const auto note_untried = [&](const probe::Verdict& v) {
            if (v.skipped) untried[reason_key(v.what)]++;
        };
        for (std::size_t i = 0; i < work.size(); ++i) {
            const Job& j = work[i];
            const Probed& p = probed[i];
            if (j.set >= 0 || (!p.lift_tried && !p.dxc_tried)) continue;
            by_kind[kind_name(j.kind)].add(p);
            note_untried(p.lift);
            note_untried(p.dxc);
            const std::string name = std::string(kind_name(j.kind)) + " " + j.bundle + "/" + j.name;
            if (p.dxc_tried && p.dxc.refused() && p.lift_tried && p.lift.ok) refused_dxc.push_back(name + ": " + p.dxc.str());
            if (p.lift_tried && p.lift.refused()) refused_lift.push_back(name + ": " + p.lift.str());
        }
        DriverTally set_tally;
        for (const SetProbe& sp : sets) {
            if (!sp.lift_tried) continue;
            Probed p;
            p.lift_tried = sp.lift_tried;
            p.lift = sp.lift;
            p.dxc_tried = sp.dxc_tried;
            p.dxc = sp.dxc;
            p.us = sp.us;
            set_tally.add(p);
            note_untried(p.lift);
            note_untried(p.dxc);
            const std::string name = "hull set " + work[sp.hs].bundle + "/" + work[sp.hs].name;
            if (p.dxc_tried && p.dxc.refused() && p.lift.ok) refused_dxc.push_back(name + ": " + p.dxc.str());
            if (p.lift.refused()) refused_lift.push_back(name + ": " + p.lift.str());
        }
        const probe::DeviceInfo& d = prober->info();
        std::printf("driver, %s (%s): each lift's module and each DXC module the HLSL path takes, built as the renderer builds the stage "
                    "(%.1f s of pipelines over %u threads)\n",
                    d.name.c_str(), d.driver.c_str(), probe_us.load() / 1e6, jobs);
        const auto row = [](const char* what, const DriverTally& k) {
            std::printf("  %-27s lifts %5zu: built %5zu, refused %zu, not tried %zu;  DXC modules %5zu: built %5zu, refused %zu (%zu where the lift is "
                        "built), not tried %zu;  %.0f s\n",
                        what, k.lifts, k.lift_built, k.lift_refused, k.lift_untried, k.dxc, k.dxc_built, k.dxc_refused, k.dxc_only, k.dxc_untried, k.us / 1e6);
        };
        static const char* const kinds[][2] = {{"ps", "pixel shaders"}, {"vs", "vertex shaders"}, {"ls", "LS (particles', alone)"},
                                               {"ds", "domain shaders (particles')"}, {"cs", "compute shaders"}};
        DriverTally total = set_tally;
        for (const auto& k : kinds) {
            const auto it = by_kind.find(k[0]);
            if (it == by_kind.end()) continue;
            row(k[1], it->second);
            const DriverTally& x = it->second;
            total.dxc_only += x.dxc_only;
            total.notes += x.notes;
            total.warnings += x.warnings;
        }
        if (set_tally.lifts) row("hull sets (LS+hull+domain)", set_tally);
        if (total.notes) std::printf("  subgroup size refused and built at the driver's default, as the renderer retries: %zu\n", total.notes);
        if (total.warnings) std::printf("  built, but with a validation error: %zu (--list has them)\n", total.warnings);
        for (const auto& [why, n] : untried) std::printf("  not tried x%zu: %s\n", n, why.c_str());
        for (const std::string& r : refused_lift) std::printf("  the lift refused: %s\n", r.c_str());
        dxc_only = refused_dxc.size();
        std::printf("DXC modules the driver refused where it builds the lift - the stages the game's fallback gives the lift: %zu\n", dxc_only);
        for (const std::string& r : refused_dxc) std::printf("  %s\n", r.c_str());
    }
#endif
    return t.same_interface == t.lifted && !dxc_only ? 0 : 1;
}
