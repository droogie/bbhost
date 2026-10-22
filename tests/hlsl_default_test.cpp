// The HLSL path as players run it by default (host/translation_cache.h,
// BBHOST_HLSL unset): a stage keeps its lift while bb-hlsl compiles DXC's
// module behind it, and takes that module from the cache afterwards; the
// queue's lanes (what draws took before what GX only created, a drawn stage
// moved up); BBHOST_HLSL_DUMP=1's folder; edited HLSL in shaders/, taken
// when it compiles with the lift's interface and refused (the stage keeps its
// own module) when it does not; and the files' stable names (bundle entry, GX
// name, stage), which a lift with other words still finds. Synthetic programs only: no game shader is
// read. Skips (77) without DXC (host/dxc.h: BBHOST_HLSL_DXC, beside the
// executable, or the system's).
#include "host/translation_cache.cpp"

#include "gcn/isa.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

std::uint32_t host_thread_id() { return 1; }
void host_thread_set_name(const char*) {}
void host_thread_background() {}

using namespace gpu;

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

constexpr std::uint32_t kEndpgm = 0xbf810000u;

// attr0's channels interpolated into v2..v5 in the order `chans` gives,
// exported to mrt0: a different order is a different program and lift.
struct Ps {
    gcn::Program prog;
    gcn::TranslateOptions o;
    gcn::TranslateResult ref;
};
Ps make_ps(const std::uint32_t (&chans)[4]) {
    std::vector<std::uint32_t> w;
    for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t vdst = 2 + k, attr = 0, chan = chans[k];
        w.push_back(0xc8000000u | vdst << 18 | attr << 10 | chan << 8 | 0u);
        w.push_back(0xc8000000u | vdst << 18 | 1u << 16 | attr << 10 | chan << 8 | 1u);
    }
    w.push_back(0xf800180fu);  // exp mrt0, v2..v5 done vm
    w.push_back(2u | 3u << 8 | 4u << 16 | 5u << 24);
    w.push_back(kEndpgm);
    Ps p;
    p.prog = gcn::decode(w.data(), w.size());
    CHECK(p.prog.errors.empty());
    p.o.stage = gcn::Stage::Pixel;
    p.o.rsrc2 = 16u << 1;
    p.o.ps_input_ena = 2;
    p.o.descriptor_set = 1;
    p.o.cb_ssbo = true;
    p.o.cb_no_fallback = true;
    p.ref = gcn::translate(p.prog, p.o);
    CHECK(p.ref.ok());
    return p;
}

bool busy_flag = false;
bool busy() { return __atomic_load_n(&busy_flag, __ATOMIC_ACQUIRE); }
void set_busy(bool v) { __atomic_store_n(&busy_flag, v, __ATOMIC_RELEASE); }

// Waits until bb-hlsl has done `n` compiles (or 60 s).
bool wait_done(std::uint64_t n) {
    for (int i = 0; i < 600 && g_hlsl_behind.done.load() < n; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return g_hlsl_behind.done.load() >= n;
}

}  // namespace

int main() {
    // One bb-hlsl thread, so the queue's order is what it takes; the dump in
    // the data folder.
    setenv("BBHOST_HLSL_THREADS", "1", 1);
    setenv("BBHOST_HLSL_DUMP", "1", 1);
    unsetenv("BBHOST_HLSL");
    if (!hlsl_dxc().loaded()) {
        std::fprintf(stderr, "no DXC (%s); skipped\n", hlsl_dxc().error().c_str());
        return 77;
    }
    CHECK(hlsl_mode() == 2 && hlsl_on());  // the default: behind the game

    // The folder is this process's own: two test runs at once each have theirs.
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("bbhost-hlsl-default-test-" + std::to_string(getpid()));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir / "shaders", ec);
    hlsl_set_folders(dir.string());
    translation_cache_load((dir / "translation-cache.bin").string());
    hlsl_set_busy_probe(&busy);

    // 1. The first use keeps the lift; bb-hlsl compiles DXC's module behind
    // it; the next use (a later start's) takes DXC's module from the cache.
    const Ps a = make_ps({0, 1, 2, 3});
    const gcn::LiftResult plain = gcn::lift_pixel_shader(a.prog, a.o, a.ref);
    CHECK(plain.ok());
    const gcn::LiftResult first = lift_cached(false, a.prog, a.o, a.ref);
    CHECK(first.spirv == plain.spirv);
    CHECK(g_hlsl_behind.deferred.load() == 1);
    CHECK(wait_done(1));
    CHECK(g_hlsl_behind.failed.load() == 0);
    const gcn::LiftResult second = lift_cached(false, a.prog, a.o, a.ref);
    CHECK(second.ok() && second.spirv != plain.spirv);
    CHECK(gcn::hlsl_interface_diff(plain.spirv, second.spirv).empty());
    std::vector<std::uint32_t> behind;
    CHECK(hlsl_lift_of(second.spirv, behind) && behind == plain.spirv);  // a refused pipeline is built from the lift
    CHECK(g_hlsl.cached.load() == 1 && g_hlsl.taken.load() == 1);
    // BBHOST_HLSL_DUMP=1: the module's HLSL under the lift's hash.
    char name[40];
    std::snprintf(name, sizeof(name), "%016llx.ps.hlsl", static_cast<unsigned long long>(lift_hash(plain.spirv)));
    const std::filesystem::path dumped = dir / "hlsl-dump" / name;
    CHECK(std::filesystem::exists(dumped));

    // 2. Lanes. With the start's compiles busy, bb-hlsl takes one job and
    // waits; a created stage then a drawn one queue in their lanes, and a
    // draw that takes the created stage moves it up.
    set_busy(true);
    const Ps x = make_ps({1, 0, 2, 3}), y = make_ps({2, 1, 0, 3}), z = make_ps({3, 2, 1, 0});
    hlsl_lane(kHlslLaneCreated);
    (void)lift_cached(false, x.prog, x.o, x.ref);  // taken at once, waiting for the start
    // Until bb-hlsl has taken x, which the busy probe then holds: the lanes
    // below are only what the test says once it has. The queue's lock is held
    // to look, never across the sleep - sleeping with it kept a worker that
    // had not run before the first look from ever taking x (a loaded CI
    // runner), and the checks below then found x still queued.
    bool x_taken = false;
    for (int i = 0; i < 6000 && !x_taken; ++i) {
        {
            std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
            x_taken = g_hlsl_behind.jobs[kHlslLaneCreated].empty() && g_hlsl_behind.waiting == 0;
        }
        if (!x_taken) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(x_taken);
    (void)lift_cached(false, y.prog, y.o, y.ref);
    hlsl_tag_stage(0x1234);  // the stage cache's key for y's stage
    hlsl_lane(kHlslLaneDrawn);
    (void)lift_cached(false, z.prog, z.o, z.ref);
    {
        std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
        CHECK(g_hlsl_behind.jobs[kHlslLaneDrawn].size() == 1 && g_hlsl_behind.jobs[kHlslLaneCreated].size() == 1);
        CHECK(g_hlsl_behind.waiting == 2);
    }
    hlsl_note_drawn(0x1234);
    {
        std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
        CHECK(g_hlsl_behind.jobs[kHlslLaneDrawn].size() == 2);  // z, then y
        CHECK(g_hlsl_behind.waiting == 2);                      // y counted once
    }
    CHECK(g_hlsl_behind.promoted.load() == 1);
    set_busy(false);
    CHECK(wait_done(4));
    CHECK(g_hlsl_behind.lane_done[kHlslLaneCreated].load() == 1 && g_hlsl_behind.lane_done[kHlslLaneDrawn].load() == 3);  // x; a, z, y
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(g_hlsl_behind.done.load() == 4);  // y's stale copy in its old lane is not compiled again
    {
        std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
        CHECK(g_hlsl_behind.waiting == 0);
    }

    // 3. Edited HLSL in shaders/: the dumped file with an edit is taken; one
    // that does not compile is refused and the stage keeps its own module.
    std::string source;
    {
        std::ifstream in(dumped, std::ios::binary);
        source.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    CHECK(!source.empty());
    std::ofstream(dir / "shaders" / name, std::ios::binary) << "// edited\n" << source;
    std::snprintf(name, sizeof(name), "%016llx.ps.hlsl",
                  static_cast<unsigned long long>(lift_hash(gcn::lift_pixel_shader(x.prog, x.o, x.ref).spirv)));
    std::ofstream(dir / "shaders" / name, std::ios::binary) << "this is not HLSL\n";
    hlsl_set_folders(dir.string());
    const gcn::LiftResult edited = lift_cached(false, a.prog, a.o, a.ref);
    CHECK(edited.ok() && edited.spirv != plain.spirv);
    CHECK(g_hlsl_overrides.taken.load() == 1);
    CHECK(gcn::hlsl_interface_diff(plain.spirv, edited.spirv).empty());
    const gcn::LiftResult x_lift = gcn::lift_pixel_shader(x.prog, x.o, x.ref);
    const gcn::LiftResult x_now = lift_cached(false, x.prog, x.o, x.ref);
    CHECK(g_hlsl_overrides.refused.load() == 1);
    CHECK(x_now.ok() && x_now.spirv != x_lift.spirv);  // refused: DXC's module of the lift, from the cache
    (void)lift_cached(false, x.prog, x.o, x.ref);
    CHECK(g_hlsl_overrides.refused.load() == 1);  // decided once a run

    // 4. Stable names. A lift whose program is known is dumped under the
    // shader's identity - its bundle entry and GX's name, and the stage - as
    // Test_Shader.ppo@deadbeef.ps.hlsl, another lift of the same program as
    // ...ps~<lift hash>.hlsl, the hash on each file's second line. An edited
    // file under the identity's name is taken by a lift with other words (what
    // a translator update makes), and a variant's own file comes first.
    hlsl_set_name_lookup(+[](const std::string& gx) -> std::string { return gx == "deadbeef" ? "Test_Shader.ppo" : std::string(); });
    const Ps p1 = make_ps({0, 2, 1, 3}), p2 = make_ps({1, 3, 0, 2}), p3 = make_ps({3, 1, 2, 0});
    const gcn::LiftResult l1 = gcn::lift_pixel_shader(p1.prog, p1.o, p1.ref), l2 = gcn::lift_pixel_shader(p2.prog, p2.o, p2.ref),
                          l3 = gcn::lift_pixel_shader(p3.prog, p3.o, p3.ref);
    const std::uint64_t done_before = g_hlsl_behind.done.load();
    hlsl_name_next("deadbeef");
    (void)lift_cached(false, p1.prog, p1.o, p1.ref);
    CHECK(wait_done(done_before + 1));
    hlsl_name_next("deadbeef");
    (void)lift_cached(false, p2.prog, p2.o, p2.ref);
    CHECK(wait_done(done_before + 2));
    const std::filesystem::path stable_dump = dir / "hlsl-dump" / "Test_Shader.ppo@deadbeef.ps.hlsl";
    const std::filesystem::path variant_dump = dir / "hlsl-dump" / ("Test_Shader.ppo@deadbeef.ps~" + lift_hash_hex(l2.spirv) + ".hlsl");
    CHECK(std::filesystem::exists(stable_dump));
    CHECK(std::filesystem::exists(variant_dump));
    CHECK(dump_is_lift(stable_dump, lift_hash_hex(l1.spirv)));
    CHECK(dump_is_lift(variant_dump, lift_hash_hex(l2.spirv)));
    {
        std::ifstream in(stable_dump, std::ios::binary);
        std::string first;
        std::getline(in, first);
        CHECK(first == "// Test_Shader.ppo (deadbeef)");
    }
    std::string stable_source;
    {
        std::ifstream in(stable_dump, std::ios::binary);
        stable_source.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::ofstream(dir / "shaders" / "Test_Shader.ppo@deadbeef.ps.hlsl", std::ios::binary) << "// edited\n" << stable_source;
    std::ofstream(dir / "shaders" / ("Test_Shader.ppo@deadbeef.ps~" + lift_hash_hex(l2.spirv) + ".hlsl"), std::ios::binary) << "not HLSL either\n";
    hlsl_set_folders(dir.string());
    const std::uint64_t taken_before = g_hlsl_overrides.taken.load(), refused_before = g_hlsl_overrides.refused.load();
    hlsl_name_next("deadbeef");
    const gcn::LiftResult e3 = lift_cached(false, p3.prog, p3.o, p3.ref);  // never met: other words, the same program and stage
    CHECK(g_hlsl_overrides.taken.load() == taken_before + 1);
    CHECK(e3.ok() && e3.spirv != l3.spirv);
    CHECK(gcn::hlsl_interface_diff(l3.spirv, e3.spirv).empty());
    hlsl_name_next("deadbeef");
    const gcn::LiftResult e2 = lift_cached(false, p2.prog, p2.o, p2.ref);  // its own variant file, which does not compile
    CHECK(g_hlsl_overrides.refused.load() == refused_before + 1);
    CHECK(e2.ok() && e2.spirv != l2.spirv);  // refused: DXC's module of the lift, from the cache
    hlsl_name_next("deadbee0");
    (void)lift_cached(false, p1.prog, p1.o, p1.ref);  // another program's name: the identity's file is not its
    CHECK(g_hlsl_overrides.taken.load() == taken_before + 1);

    translation_cache_report();
    std::filesystem::remove_all(dir, ec);
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("hlsl default: ok\n");
    return 0;
}
