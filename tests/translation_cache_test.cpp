// The translation cache (host/translation_cache.cpp) against the shipped
// shader bundles: every entry reads back as the translation and the lift it
// was written from, a file written by one "start" serves the next one entirely
// from the cache with the same results, and a changed translation switch, a
// damaged entry and another translator's file each fall back to translating.
// Skips (77) when the dump is absent.
//
//   translation_cache_test [bundle.dcx...]   (default: BBHOST_APP0's dvdroot_ps4/shader bundles)
//   BBHOST_TC_TEST_ALL=1: every shader, not a sample of ~600
#include "test_app0.h"

// The cache's internals (its anonymous namespace) are what is tested.
#include "host/translation_cache.cpp"

#include "gcn/container.h"
#include "gcn/translate.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

std::uint32_t host_thread_id() { return 1; }

using namespace gpu;

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            if (++g_failures > 20) std::exit(1);                                           \
        }                                                                                  \
    } while (0)

namespace {

struct Case {
    std::string name;
    gcn::Program prog;
    gcn::TranslateOptions o;
    bool lift_ps = false;
};

bool same_path(const gcn::ResourcePath& a, const gcn::ResourcePath& b) {
    return a.user_sgpr == b.user_sgpr && a.loads == b.loads && a.final_offset_dw == b.final_offset_dw && a.final_vsharp == b.final_vsharp &&
           a.immediate == b.immediate;
}

bool same(const gcn::TranslateResult& a, const gcn::TranslateResult& b) {
    if (a.spirv != b.spirv || a.bindless != b.bindless || a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size() ||
        a.buffers.size() != b.buffers.size() || a.image_at != b.image_at || a.sampler_at != b.sampler_at || a.buffer_at != b.buffer_at ||
        a.errors != b.errors || a.lds_bytes != b.lds_bytes || a.store_buffers.size() != b.store_buffers.size() ||
        a.untraced_stores != b.untraced_stores || a.unnormalized_samples != b.unnormalized_samples || a.walks != b.walks ||
        a.vs_params != b.vs_params || a.ps_inputs != b.ps_inputs || a.vs_clip_count != b.vs_clip_count || a.vs_point_size != b.vs_point_size) {
        return false;
    }
    for (std::size_t i = 0; i < a.images.size(); ++i) {
        const gcn::ImageBinding &x = a.images[i], &y = b.images[i];
        if (!same_path(x.path, y.path) || x.sgpr != y.sgpr || x.binding != y.binding || x.storage != y.storage || x.r128 != y.r128 ||
            x.da != y.da || x.dim != y.dim || x.arrayed != y.arrayed || x.depth != y.depth || x.cube != y.cube || x.kind != y.kind) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.samplers.size(); ++i) {
        const gcn::SamplerBinding &x = a.samplers[i], &y = b.samplers[i];
        if (!same_path(x.path, y.path) || x.sgpr != y.sgpr || x.binding != y.binding || x.compare != y.compare) return false;
    }
    for (std::size_t i = 0; i < a.buffers.size(); ++i) {
        const gcn::BufferBinding &x = a.buffers[i], &y = b.buffers[i];
        if (!same_path(x.path, y.path) || x.binding != y.binding || x.max_dw != y.max_dw || x.pointer != y.pointer || x.indexed != y.indexed) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.store_buffers.size(); ++i) {
        if (!same_path(a.store_buffers[i], b.store_buffers[i])) return false;
    }
    return true;
}

bool same_lift(const gcn::LiftResult& a, const gcn::LiftResult& b) { return a.spirv == b.spirv && a.rejections == b.rejections; }

gcn::TranslateResult without_spirv(gcn::TranslateResult t) {
    t.spirv = {};
    return t;
}

// The cache as a new process finds it: nothing loaded.
void reset_cache() {
    std::lock_guard<std::mutex> lk(g_tc.mu);
    g_tc.ready.store(false);
    g_tc.on = false;
    g_tc.slots.clear();
    g_tc.made.clear();
    g_tc.file.clear();
    g_tc.bytes = 0;
    g_tc.dirty = false;
    g_tc.loaded = 0;
    for (std::atomic<std::uint64_t>* c : {&g_tc.hits, &g_tc.hit_us, &g_tc.misses, &g_tc.miss_us, &g_tc.stored, &g_tc.store_us, &g_tc.bad,
                                          &g_tc.early, &g_tc.raw_hit_bytes, &g_tc.lift_hits, &g_tc.lift_hit_us, &g_tc.lift_misses,
                                          &g_tc.lift_miss_us}) {
        c->store(0);
    }
    g_tc_loaded.store(false);
}

// One "start": every case through the cache (stage, paths, and the pixel
// shaders' lifts), each compared with what the translator makes now.
void run_start(const std::vector<Case>& cases, const std::vector<gcn::TranslateResult>& fresh, const std::vector<gcn::LiftResult>& lifts) {
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        const gcn::TranslateResult t = translate_cached(c.prog, c.o);
        CHECK(same(t, fresh[i]));
        const gcn::TranslateResult p = translate_cached(c.prog, c.o, TranslationUse::kPaths);
        CHECK(p.spirv.empty() && same(p, without_spirv(fresh[i])));
        if (c.lift_ps) CHECK(same_lift(lift_cached(false, c.prog, c.o, t), lifts[i]));
    }
}

}  // namespace

int main(int argc, char** argv) {
    // These key checks also run without a game dump in CI. Motion-enabled
    // shaders and their output locations must not reuse ordinary translations.
    gcn::Program empty;
    gcn::TranslateOptions plain, motion;
    motion.object_motion = true;
    const auto plain_key = key_for(empty, plain, TranslationUse::kStage);
    const auto motion_key = key_for(empty, motion, TranslationUse::kStage);
    CHECK(!(plain_key == motion_key));
    motion.motion_location = 3;
    CHECK(!(motion_key == key_for(empty, motion, TranslationUse::kStage)));
    if (g_failures) return 1;
    std::vector<std::string> bundles;
    for (int a = 1; a < argc; ++a) bundles.emplace_back(argv[a]);
    if (bundles.empty()) {
        for (const char* b : {"gxrenderershader", "gxposteffect", "gxffxshader", "gxdecal", "gxgui", "gxshader", "gxflvershader"}) {
            bundles.push_back(test_app0_file((std::string("dvdroot_ps4/shader/") + b + ".shaderbnd.dcx").c_str()));
        }
    }
    std::vector<Case> all;
    std::set<std::string> seen;  // a shader in two bundles is one entry
    for (const std::string& path : bundles) {
        std::vector<std::uint8_t> raw;
        if (!gcn::read_file(path, raw)) continue;
        std::string err;
        std::vector<gcn::BundleEntry> entries;
        const std::vector<std::uint8_t> b = gcn::dcx_decompress(raw, &err);
        if (b.empty() || !gcn::bnd4_entries(b, entries, &err)) continue;
        for (const gcn::BundleEntry& e : entries) {
            gcn::ShaderCode code;
            if (!gcn::shader_code(e.data, code) || (code.type != 1 && code.type != 2 && code.type != 4)) continue;
            std::size_t shdr = 0;
            while (shdr + 0x50 <= e.data.size() && std::memcmp(e.data.data() + shdr, "Shdr", 4) != 0) ++shdr;
            if (shdr + 0x50 > e.data.size()) continue;
            std::uint32_t regs[16];
            std::memcpy(regs, e.data.data() + shdr + 16, sizeof(regs));
            // What the options below take: one case per program and registers.
            const std::uint32_t used[6] = {code.type, regs[4], regs[5], regs[8], code.type == 4 ? regs[6] : 0, code.type == 4 ? regs[7] : 0};
            std::string id(reinterpret_cast<const char*>(used), sizeof(used));
            id.append(reinterpret_cast<const char*>(code.words.data()), code.words.size() * 4);
            if (!seen.insert(std::move(id)).second) continue;
            Case c;
            c.name = e.name;
            c.prog = gcn::decode(code.words.data(), code.words.size());
            if (!c.prog.errors.empty()) continue;
            c.o.rsrc1 = regs[4];
            c.o.rsrc2 = regs[5];
            if (code.type == 4) {
                c.o.stage = gcn::Stage::Compute;
                for (int k = 0; k < 3; ++k) c.o.cs_threads[k] = regs[6 + k];
            } else {
                // As the renderer's no-fallback variants: constant buffers on
                // storage buffers, in the stage's own set.
                c.o.stage = code.type == 1 ? gcn::Stage::Vertex : gcn::Stage::Pixel;
                c.o.cb_ssbo = c.o.cb_no_fallback = true;
                if (code.type == 2) {
                    c.o.ps_input_ena = regs[8];
                    c.o.descriptor_set = 1;
                    c.lift_ps = true;
                } else {
                    c.o.vs_out_cntl = regs[8];
                }
            }
            all.push_back(std::move(c));
        }
    }
    if (all.empty()) {
        std::fprintf(stderr, "no shader bundles (BBHOST_APP0); skipped\n");
        return kTestSkip;
    }
    std::vector<Case> cases;
    const char* every = std::getenv("BBHOST_TC_TEST_ALL");
    const std::size_t step = every && every[0] == '1' ? 1 : std::max<std::size_t>(1, all.size() / 600);
    for (std::size_t i = 0; i < all.size(); i += step) cases.push_back(std::move(all[i]));

    // 1. Every entry reads back as what was written.
    std::vector<gcn::TranslateResult> fresh;
    std::vector<gcn::LiftResult> lifts(cases.size());
    std::size_t lifted = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        fresh.push_back(gcn::translate(c.prog, c.o));
        for (const bool with_spirv : {true, false}) {
            Writer w;
            write_result(w, fresh[i], with_spirv);
            Reader r{w.out.data(), w.out.data() + w.out.size()};
            gcn::TranslateResult back;
            CHECK(read_result(r, back));
            CHECK(same(back, with_spirv ? fresh[i] : without_spirv(fresh[i])));
        }
        if (c.lift_ps) {
            lifts[i] = gcn::lift_pixel_shader(c.prog, c.o, fresh[i]);
            lifted += lifts[i].ok();
            Writer w;
            write_lift(w, lifts[i]);
            Reader r{w.out.data(), w.out.data() + w.out.size()};
            gcn::LiftResult back;
            CHECK(read_lift(r, back));
            CHECK(same_lift(back, lifts[i]));
        }
    }
    std::size_t lift_cases = 0;
    for (const Case& c : cases) lift_cases += c.lift_ps;
    std::printf("%zu shaders (of %zu), %zu pixel shaders lifted of %zu\n", cases.size(), all.size(), lifted, lift_cases);

    // 2. A first start fills the file, a second takes everything from it.
    const std::string path = (std::filesystem::temp_directory_path() / "bbhost-translation-cache-test.bin").generic_string();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    const std::uint64_t entries = cases.size() * 2 + lift_cases;
    reset_cache();
    translation_cache_load(path);
    CHECK(g_tc.on && g_tc.loaded == 0);
    run_start(cases, fresh, lifts);
    CHECK(g_tc.misses.load() == cases.size() * 2 && g_tc.hits.load() == 0 && g_tc.lift_misses.load() == lift_cases);
    const std::uint64_t made_us = g_tc.miss_us.load() + g_tc.lift_miss_us.load();
    run_start(cases, fresh, lifts);  // the same run: from memory
    CHECK(g_tc.hits.load() == cases.size() * 2 && g_tc.lift_hits.load() == lift_cases && g_tc.stored.load() == entries);
    translation_cache_save(path, true);
    reset_cache();
    translation_cache_load(path);
    CHECK(g_tc.loaded == entries);
    run_start(cases, fresh, lifts);
    CHECK(g_tc.misses.load() == 0 && g_tc.lift_misses.load() == 0 && g_tc.hits.load() == cases.size() * 2 && g_tc.lift_hits.load() == lift_cases);
    std::printf("second start: %llu translations and %llu lifts from the file (%.1f MiB) in %.2f s, none translated; the first start "
                "made them in %.2f s\n",
                static_cast<unsigned long long>(g_tc.hits.load()), static_cast<unsigned long long>(g_tc.lift_hits.load()),
                g_tc.file.size() / 1048576.0, (g_tc.hit_us.load() + g_tc.lift_hit_us.load()) / 1e6, made_us / 1e6);

    // 3. Another switch for the translator: nothing is taken.
    gcn::set_runtime_sample_offsets(!gcn::runtime_sample_offsets());
    reset_cache();
    translation_cache_load(path);
    CHECK(g_tc.loaded == entries);
    {
        const gcn::TranslateResult t = translate_cached(cases[0].prog, cases[0].o);
        CHECK(same(t, gcn::translate(cases[0].prog, cases[0].o)));
        CHECK(g_tc.hits.load() == 0 && g_tc.misses.load() == 1);
    }
    gcn::set_runtime_sample_offsets(!gcn::runtime_sample_offsets());

    // 4. A damaged entry is translated again, and left out of the next write.
    {
        std::vector<std::uint8_t> bytes;
        CHECK(gcn::read_file(path, bytes));
        Reader r{bytes.data(), bytes.data() + bytes.size()};
        r.u32();
        r.u32();
        r.str();
        r.u32();
        r.u64();
        r.u64();
        r.u32();
        const std::uint32_t size = r.count(1);
        CHECK(r.ok && size > 8);
        bytes[static_cast<std::size_t>(r.p - bytes.data()) + size / 2] ^= 0x5a;
        std::ofstream(path, std::ios::binary | std::ios::trunc).write(reinterpret_cast<const char*>(bytes.data()),
                                                                       static_cast<std::streamsize>(bytes.size()));
        reset_cache();
        translation_cache_load(path);
        run_start(cases, fresh, lifts);
        CHECK(g_tc.bad.load() == 1);
        CHECK(g_tc.misses.load() + g_tc.lift_misses.load() == 1);
        translation_cache_save(path, true);
        reset_cache();
        translation_cache_load(path);
        CHECK(g_tc.loaded == entries);  // translated again in 4, written whole
        run_start(cases, fresh, lifts);
        CHECK(g_tc.bad.load() == 0 && g_tc.misses.load() == 0 && g_tc.lift_misses.load() == 0);
    }

    // 5. Another translator's file starts again.
    {
        std::vector<std::uint8_t> bytes;
        CHECK(gcn::read_file(path, bytes));
        bytes[12] ^= 1;  // the fingerprint's first character, after magic, version and its length
        std::ofstream(path, std::ios::binary | std::ios::trunc).write(reinterpret_cast<const char*>(bytes.data()),
                                                                       static_cast<std::streamsize>(bytes.size()));
        reset_cache();
        translation_cache_load(path);
        CHECK(g_tc.loaded == 0 && g_tc.on);
        run_start(cases, fresh, lifts);
        CHECK(g_tc.hits.load() == 0 && g_tc.misses.load() == cases.size() * 2);
    }
    std::filesystem::remove(path, ec);
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("translation cache: ok\n");
    return 0;
}
