// Edited shaders as variants (host/shader_overrides.h): a program's sources
// named by one salt, which the stages built with it compile, whatever the
// sources are by then; the player's folder before other owners, and an
// owner's texts before its folders; a return to an earlier source compiling
// nothing; the variant's file before the shader's before the lift hash's,
// which ties itself to its program at that lift's first sight; a file changed
// after it was listed, listed again; an owner turned off or gone; a change
// compiled ahead of its switch, for the variants met, off the drawing thread;
// states, the report and the off switch; the watch's listing; an owner's
// shader data read through bbhost.hlsli (its device address compiled in, its
// writes latched a frame at a time); the game's own source at run time; a
// geometry lift's stage. Synthetic pixel shaders through the HLSL path
// (BBHOST_HLSL=1, compiled where they are met). Skips (77) without DXC.
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
int host_opt_motion_rebuilt_bits() { return 0; }

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

// attr0's channels interpolated in the order `chans` gives, exported to mrt0
// (as hlsl_default_test makes them).
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

// The lift's own HLSL with its output scaled by `scale`: a source that
// compiles with the lift's interface and draws differently.
std::string scaled(const std::vector<std::uint32_t>& lift, const char* scale) {
    std::string src = gcn::emit_hlsl(lift).source;
    const std::size_t at = src.rfind("    lifted_main();\n");
    CHECK(at != std::string::npos && src.find("static float4 mrt0;") != std::string::npos);
    if (at != std::string::npos) src.insert(at + 19, std::string("    mrt0 = mrt0 * ") + scale + ";\n");
    return src;
}

// The renderer's maker of an owner's shader data buffer: an address of its own.
std::uint64_t fake_buffer(int owner, const std::uint8_t*) { return 0x0000001234560000ull + static_cast<std::uint64_t>(owner) * 0x10000u; }

// Until no changed source waits for its compile ahead (60 s at most).
bool settle() {
    for (int i = 0; i < 6000 && !overrides_ahead_idle(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return overrides_ahead_idle();
}

// A stage of `ps` named `name`, keyed with the program's salt as it stands now
// (the renderer's draw), and whether it took an edited shader's module.
gcn::LiftResult stage(const Ps& ps, const char* name, std::uint64_t salt, bool& took) {
    hlsl_name_next(name, salt);
    gcn::LiftResult l = lift_cached(false, ps.prog, ps.o, ps.ref);
    took = hlsl_took_override();
    return l;
}

}  // namespace

int main() {
    host_log_decide(true);          // the log to stderr: what the overrides say, for a reader of the test's output
    setenv("BBHOST_HLSL", "1", 1);  // compiled where met: nothing behind the test
    unsetenv("BBHOST_HLSL_DUMP");
    if (!hlsl_dxc().loaded()) {
        std::fprintf(stderr, "no DXC (%s); skipped\n", hlsl_dxc().error().c_str());
        return 77;
    }
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("bbhost-shader-overrides-test-" + std::to_string(getpid()));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir / "shaders", ec);
    std::filesystem::create_directories(dir / "plugin1" / "shaders", ec);

    // 1. Nothing registered: generation 0, every salt 0 - the renderer's fast path.
    CHECK(override_gen() == 0);
    CHECK(override_salt("12345678") == 0);
    hlsl_set_folders(dir.string());  // the player's folder, empty
    CHECK(override_gen() == 0 && overrides_on());

    const Ps p = make_ps({0, 1, 2, 3}), q = make_ps({1, 0, 2, 3});
    const gcn::LiftResult lp = gcn::lift_pixel_shader(p.prog, p.o, p.ref), lq = gcn::lift_pixel_shader(q.prog, q.o, q.ref);
    CHECK(lp.ok() && lq.ok());
    const std::string a = scaled(lp.spirv, "0.5f"), b = scaled(lp.spirv, "0.25f");
    bool took = false;
    const gcn::LiftResult base = stage(p, "aaaa0001", 0, took);
    CHECK(base.ok() && !took);

    // 2. Content: a text, another, the first again - its salt and module come
    // back, compiled once each.
    overrides_owner(1, "plugin one", (dir / "plugin1").string(), true);
    overrides_owner(2, "plugin two", (dir / "plugin2").string(), true);
    // Its program was met: the text is compiled ahead on bb-hlsl-ahead and
    // applies once it is - the stage then compiles nothing.
    CHECK(overrides_set_text(1, "aaaa0001.ps", a.c_str()) == 0);
    CHECK(settle() && override_gen() != 0);
    CHECK(g_hlsl_ahead.compiles.load() == 1);
    const std::uint64_t s_a = override_salt("aaaa0001");
    CHECK(s_a != 0);
    const std::uint64_t before_a = g_hlsl_overrides.compiles.load();
    const gcn::LiftResult ma = stage(p, "aaaa0001", s_a, took);
    CHECK(took && ma.ok() && ma.spirv != base.spirv);
    CHECK(g_hlsl_overrides.compiles.load() == before_a);
    std::string why;
    CHECK(overrides_state(1, "aaaa0001.ps", &why) == kOverrideTaken);
    CHECK(overrides_state(1, "aaaa0002.ps", &why) == kOverrideNone);
    const std::uint64_t gen_a = override_gen();
    CHECK(overrides_set_text(1, "aaaa0001.ps", a.c_str()) == 0 && override_gen() == gen_a);  // the same text again: no change
    CHECK(overrides_set_text(1, "aaaa0001.ps", b.c_str()) == 0 && settle() && override_gen() != gen_a);
    const std::uint64_t s_b = override_salt("aaaa0001");
    CHECK(s_b != 0 && s_b != s_a);
    const gcn::LiftResult mb = stage(p, "aaaa0001", s_b, took);
    CHECK(took && mb.spirv != ma.spirv);
    // The race: a draw keyed before the change builds after it - with its own salt's source.
    const gcn::LiftResult late = stage(p, "aaaa0001", s_a, took);
    CHECK(took && late.spirv == ma.spirv);
    CHECK(overrides_set_text(1, "aaaa0001.ps", a.c_str()) == 0 && settle());
    CHECK(override_salt("aaaa0001") == s_a);
    const std::uint64_t compiles = g_hlsl_overrides.compiles.load();
    const gcn::LiftResult again = stage(p, "aaaa0001", s_a, took);
    CHECK(took && again.spirv == ma.spirv && g_hlsl_overrides.compiles.load() == compiles);  // nothing compiled
    CHECK(compiles == 2);
    // A stage keyed without the salt keeps its own module, overrides or not.
    const gcn::LiftResult plain = stage(p, "aaaa0001", 0, took);
    CHECK(!took && plain.spirv == base.spirv);

    // 2b. The game's own source at run time: the dump's text of the first
    // variant met; a shader not met yet has none.
    std::string text;
    CHECK(hlsl_override_source("aaaa0001.ps", text) == 0);
    gcn::HlslOptions titled;
    titled.title = "aaaa0001";
    CHECK(text == hlsl_dump_text(gcn::emit_hlsl(lp.spirv, titled).source, lift_hash_hex(lp.spirv)));
    CHECK(hlsl_override_source("aaaa0001.ps~" + lift_hash_hex(lp.spirv), text) == 0);
    CHECK(hlsl_override_source("aaaa0001.ps~0123456789abcdef", text) == 1);
    CHECK(hlsl_override_source("dddd0004.ps", text) == 1 && hlsl_override_source("not a name", text) == 1);

    // 2c. Shader data: bbhost.hlsli included, the owner's data read by device
    // address - compiled in, its interface the lift's - and a DXC error past
    // the include cites the file's own line.
    overrides_set_data_maker(&fake_buffer);
    std::string reads = a;
    reads.insert(0, "// reads its plugin's numbers\n#include \"bbhost.hlsli\"\n");
    const std::size_t main_at = reads.rfind("    lifted_main();\n");
    reads.insert(main_at + 19, "    mrt0 = mrt0 * bb_data_float(4) + bb_data_float4x4(64)[1];\n");
    overrides_owner(5, "plugin five", std::string(), true);
    CHECK(overrides_set_text(5, "eeee0005.ps", reads.c_str()) == 0 && settle());
    const std::uint64_t s_data = override_salt("eeee0005");
    const gcn::LiftResult md = stage(p, "eeee0005", s_data, took);
    CHECK(took && md.ok() && gcn::hlsl_interface_diff(lp.spirv, md.spirv).empty());
    CHECK(override_data_address(5) == fake_buffer(5, nullptr));
    bool addressed = false;  // the address in the module: OpConstant of a 64-bit integer
    for (std::size_t i = 5; i + 4 < md.spirv.size(); ++i) {
        const std::uint64_t at = fake_buffer(5, nullptr);
        if (md.spirv[i] == (5u << 16 | 43u) && md.spirv[i + 3] == static_cast<std::uint32_t>(at + 4) && md.spirv[i + 4] == static_cast<std::uint32_t>(at >> 32))
            addressed = true;
    }
    CHECK(addressed);
    std::string broken = reads;
    broken.insert(broken.find("\n", broken.find("bbhost.hlsli")) + 1, "\n\nthis line is not HLSL\n");  // the file's line 5
    CHECK(overrides_set_text(5, "eeee0005.ps", broken.c_str()) == 0 && settle());
    CHECK(overrides_state(5, "eeee0005.ps", &why) == kOverrideRefused && why.find("eeee0005.ps.hlsl:5:") != std::string::npos);
    // Writes: into the shadow at once, uploaded only once a frame's end
    // latched them - one copy a frame - and only what was written.
    std::vector<OverrideUpload> ups;
    const float k = 0.5f;
    CHECK(overrides_data_write(5, 4, &k, sizeof(k)) == 0);
    CHECK(overrides_data_write(5, kOverrideDataBytes - 2, &k, sizeof(k)) == 1 && overrides_data_write(kOverridePlayer, 0, &k, 4) == 1);
    override_data_latch();
    CHECK(override_data_take(ups) && ups.size() == 1 && ups[0].owner == 5 && ups[0].offset == 4 && ups[0].bytes.size() == 4);
    const std::uint16_t half = 0x3c00;
    CHECK(overrides_data_write(5, 70, &half, sizeof(half)) == 0 && overrides_data_write(5, 64, &k, sizeof(k)) == 0);
    CHECK(!override_data_take(ups));  // not latched yet: the frame under way keeps its copy
    override_data_latch();
    CHECK(override_data_take(ups) && ups.size() == 1 && ups[0].offset == 64 && ups[0].bytes.size() == 8);  // 64..72, by fours
    override_data_latch();
    CHECK(!override_data_take(ups));  // nothing new

    // 3. Owners: the player's folder first, then the owners in their order;
    // an owner's texts before its folders.
    const std::uint64_t compiles2 = g_hlsl_overrides.compiles.load();
    CHECK(overrides_set_text(2, "aaaa0001.ps", b.c_str()) == 0 && settle());
    CHECK(g_hlsl_overrides.compiles.load() == compiles2);  // shadowed by owner 1's: nothing compiled ahead
    std::shared_ptr<const OverrideSource> src;
    CHECK(override_pick(override_salt("aaaa0001"), "aaaa0001", "ps", lift_hash_hex(lp.spirv), src) && src->owner == 1);
    std::ofstream(dir / "plugin1" / "shaders" / "aaaa0001.ps.hlsl", std::ios::binary) << b;
    CHECK(overrides_add_folder(1, "shaders") == 1 && settle());  // relative: from the owner's own folder
    CHECK(overrides_add_folder(1, "missing") == -1);
    CHECK(override_salt("aaaa0001") == s_a);         // its text comes first
    CHECK(override_pick(s_a, "aaaa0001", "ps", lift_hash_hex(lp.spirv), src) && src->owner == 1 && src->text);
    std::ofstream(dir / "shaders" / "aaaa0001.ps.hlsl", std::ios::binary) << b;
    hlsl_set_folders(dir.string());  // listed again: the player's file
    const std::uint64_t s_player = override_salt("aaaa0001");
    CHECK(s_player != s_a && override_pick(s_player, "aaaa0001", "ps", lift_hash_hex(lp.spirv), src) && src->owner == kOverridePlayer);
    const gcn::LiftResult mp = stage(p, "aaaa0001", s_player, took);
    CHECK(took && mp.spirv == mb.spirv);  // b's text, from the player's file: DXC's same module
    std::filesystem::remove(dir / "shaders" / "aaaa0001.ps.hlsl", ec);
    hlsl_set_folders(dir.string());
    CHECK(override_salt("aaaa0001") == s_a);
    CHECK(settle());
    overrides_owner_on(1, false);  // turned off: the next owner's
    CHECK(override_pick(override_salt("aaaa0001"), "aaaa0001", "ps", lift_hash_hex(lp.spirv), src) && src->owner == 2);
    overrides_owner_on(1, true);
    CHECK(override_salt("aaaa0001") == s_a);
    overrides_drop_owner(2);
    CHECK(overrides_set_text(2, "aaaa0001.ps", a.c_str()) == 1);  // gone
    CHECK(overrides_set_text(1, "not a shader", a.c_str()) == 1 && overrides_set_text(1, "aaaa0001.xx", a.c_str()) == 1);

    // 4. Names: the variant's file comes before the shader's, the shader's
    // before the lift hash's.
    const std::string hp = lift_hash_hex(lp.spirv);
    CHECK(overrides_set_text(1, ("aaaa0001.ps~" + hp).c_str(), b.c_str()) == 0 && settle());
    const std::uint64_t s_var = override_salt("aaaa0001");
    CHECK(override_pick(s_var, "aaaa0001", "ps", hp, src) && src->name == "aaaa0001.ps~" + hp);
    CHECK(override_pick(s_var, "aaaa0001", "ps", lift_hash_hex(lq.spirv), src) && src->name == "aaaa0001.ps");  // another variant
    CHECK(overrides_set_text(1, ("aaaa0001.ps~" + hp).c_str(), nullptr) == 0 && override_salt("aaaa0001") == s_a);
    // A lift hash's file: tied to its program at that lift's first sight (the
    // program's next salt names it), never before.
    const std::string hq = lift_hash_hex(lq.spirv);
    const std::string qa = scaled(lq.spirv, "0.5f");
    CHECK(overrides_set_text(1, (hq + ".ps").c_str(), qa.c_str()) == 0 && settle());
    CHECK(override_salt("bbbb0002") == 0);
    const gcn::LiftResult q0 = stage(q, "bbbb0002", 0, took);
    CHECK(!took);
    const std::uint64_t s_q = override_salt("bbbb0002");
    CHECK(s_q != 0);
    const gcn::LiftResult q1 = stage(q, "bbbb0002", s_q, took);
    CHECK(took && q1.spirv != q0.spirv);
    CHECK(override_salt("aaaa0001") == s_a);  // another program's salt as it was

    // 5. A file changed after it was listed: refused for now, listed again,
    // and built with its new text at the next key.
    overrides_drop_owner(1);
    std::ofstream(dir / "shaders" / "aaaa0001.ps.hlsl", std::ios::binary) << a;
    hlsl_set_folders(dir.string());
    const std::uint64_t s_file = override_salt("aaaa0001");
    CHECK(s_file != 0);
    std::ofstream(dir / "shaders" / "aaaa0001.ps.hlsl", std::ios::binary | std::ios::app) << "// one more line\n";
    const std::uint64_t gen_file = override_gen();
    const gcn::LiftResult stale = stage(p, "aaaa0001", s_file, took);
    CHECK(!took && stale.spirv == base.spirv);  // its stage's own module; listed again, and compiled ahead
    CHECK(settle() && override_gen() != gen_file);
    const std::uint64_t s_file2 = override_salt("aaaa0001");
    CHECK(s_file2 != 0 && s_file2 != s_file);
    const gcn::LiftResult fresh = stage(p, "aaaa0001", s_file2, took);
    CHECK(took && fresh.spirv == ma.spirv);  // a's text and a comment
    std::filesystem::remove(dir / "shaders" / "aaaa0001.ps.hlsl", ec);
    hlsl_set_folders(dir.string());
    CHECK(override_salt("aaaa0001") == 0);

    // 6. A refused text: the stage keeps its own module, decided once.
    overrides_owner(3, "plugin three", std::string(), true);
    const std::uint64_t refused = g_hlsl_overrides.refused.load();
    CHECK(overrides_set_text(3, "aaaa0001.ps", "this is not HLSL\n") == 0 && settle());
    CHECK(g_hlsl_overrides.refused.load() == refused + 1);  // refused as it was compiled ahead
    const gcn::LiftResult r1 = stage(p, "aaaa0001", override_salt("aaaa0001"), took);
    CHECK(!took && r1.spirv == base.spirv && g_hlsl_overrides.refused.load() == refused + 1);
    (void)stage(p, "aaaa0001", override_salt("aaaa0001"), took);
    CHECK(g_hlsl_overrides.refused.load() == refused + 1);
    CHECK(overrides_state(3, "aaaa0001.ps", &why) == kOverrideRefused && why.find("DXC") != std::string::npos);

    // 7. States, the report and the off switch: a later owner's text of the
    // same name is not used; a source not met waits; off, none applies.
    overrides_owner(4, "plugin four", std::string(), true);
    CHECK(overrides_set_text(4, "aaaa0001.ps", a.c_str()) == 0 && settle());
    CHECK(overrides_state(4, "aaaa0001.ps", &why) == kOverrideShadowed && why.find("plugin three") != std::string::npos);
    CHECK(overrides_set_text(4, "cccc0003.ps", a.c_str()) == 0 && settle());
    CHECK(overrides_state(4, "cccc0003.ps", &why) == kOverrideWaiting);
    overrides_owner_on(4, false);
    CHECK(overrides_state(4, "cccc0003.ps", &why) == kOverrideWaiting && why == "its owner is off");
    overrides_owner_on(4, true);
    const std::vector<std::string> report = overrides_report();
    CHECK(report.size() == 5 && report[0].find("at exit: 4 (0 taken, 2 refused, 1 not met, 1 not used, 0 off)") != std::string::npos);
    hlsl_set_overrides(false, "switched off for the test");
    hlsl_set_folders(dir.string());
    CHECK(override_salt("aaaa0001") == 0 && !overrides_on());
    CHECK(hlsl_override_source("aaaa0001.ps", text) == 2);
    CHECK(overrides_state(3, "aaaa0001.ps", &why) == kOverrideOff && why == "switched off for the test");
    const gcn::LiftResult off = stage(p, "aaaa0001", 0, took);
    CHECK(!took && off.spirv == base.spirv);
    hlsl_set_overrides(true, std::string());
    hlsl_set_folders(dir.string());
    CHECK(overrides_on() && override_salt("aaaa0001") != 0);

    CHECK(hlsl_override_source("aaaa0001.ps", text) == 0);  // (on again)

    // 8. The watch (BBHOST_SHADER_WATCH's listing, called here): a file put in
    // the player's folder while the game runs applies once compiled ahead -
    // before the owners' texts of its name - an edit too, and its removal.
    const std::uint64_t s_three = override_salt("aaaa0001");
    CHECK(!overrides_watch_scan());  // nothing changed
    std::ofstream(dir / "shaders" / "aaaa0001.ps.hlsl", std::ios::binary) << b;
    CHECK(overrides_watch_scan() && settle());
    const std::uint64_t s_new = override_salt("aaaa0001");
    CHECK(s_new != s_three && override_pick(s_new, "aaaa0001", "ps", hp, src) && src->owner == kOverridePlayer);
    CHECK(stage(p, "aaaa0001", s_new, took).spirv == mb.spirv && took);
    std::ofstream(dir / "shaders" / "aaaa0001.ps.hlsl", std::ios::binary | std::ios::app) << "// again\n";
    CHECK(overrides_watch_scan() && settle());
    const std::uint64_t s_edit = override_salt("aaaa0001");
    CHECK(s_edit != s_new && s_edit != s_three);
    std::filesystem::remove(dir / "shaders" / "aaaa0001.ps.hlsl", ec);
    CHECK(overrides_watch_scan() && settle());
    CHECK(override_salt("aaaa0001") == s_three);
    CHECK(!overrides_watch_scan());

    // 9. A geometry lift's stage is "gs", as the dump names its files.
    const std::vector<std::uint32_t> gs = {0x07230203u, 0x00010500u, 0, 10, 0, (5u << 16) | 15u, 3u, 1u, 0x6e69616du /* "main" */, 0};
    CHECK(lift_stage_tag(gs) == "gs");

    std::filesystem::remove_all(dir, ec);
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("shader overrides: ok\n");
    return 0;
}
