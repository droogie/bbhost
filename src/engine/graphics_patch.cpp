#include "guest_abi.h"
#include "engine/graphics_patch.h"

#include "core/elf.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "engine/camera.h"
#include "engine/fmod_probe.h"
#include "engine/fmod_ring.h"
#include "engine/sf_text_layout.h"
#include "engine/frame_rate.h"
#include "engine/live_resolution.h"
#include "engine/menu_memory.h"
#include "engine/sf_heap_probe.h"
#include "hle/modules.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/window.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

// Binary Ninja addresses, which are the shadPS4 patch list's **unchanged**:
// that list gives ELF VAs, and this database is loaded at the preferred
// 0x400000 base, so the two already agree. Adding 0x400000 "to convert" landed
// 4 MiB away, in the middle of unrelated code, and the byte check caught it -
// which is the whole reason every site here checks.
//
// Measured against a warmed stock run at the same flip, floor 0.00%:
//
//   SSAO off        27.84% of pixels, mean 16.5 -> 18.3   recesses lighten
//   AA off           1.16%, worst 224                     edge pixels only
//   depth of field   0.00%                                nothing, in a corridor
//   motion blur off 99.77%, mean 16.5 -> 245.7            the frame blows out
//
// ---- Live ------------------------------------------------------------------
//
// The community patch list turns these off as the game is loaded: SSAO and AA
// as bytes of the renderer constructor's `mov dword [rbx+0x2968], 0x01010101`,
// depth of field and chromatic aberration as code. That only ever takes effect
// on the next run. Every one of them is also a value the game reads again each
// frame, so here they follow their settings live:
//
//   - SSAO and AA are two of the renderer's bools (+0x2968, +0x296a), read by
//     the scene renderer every frame. Their resources do not depend on them:
//     the resource setup (sub_26c39f0) creates SSAO's under capability bit 0x8
//     of the renderer's +0x48 and AA's under 0x10, and only clears a bool when
//     its capability is missing. So a hook on the render-view entry
//     (sub_269e990, the renderer in r9) writes each bool as setting && that
//     bit, for every renderer - there are several (two in the graphics
//     manager, a child of each, and menu views) - before it draws.
//   - Chromatic aberration is an amount the area's lens parameters give each
//     frame: sub_128a710 fills a block on the stack from two parameter sets,
//     and the same function then copies it field by field into the global
//     post-process block at 0x59406e0 - the amount by
//     `mov eax, [rbp-0xa70]; mov [rbx+0xac], eax` at 0x269faa8, the copy the
//     community patch replaces with a store of 0. Here those 12 bytes become
//     a jump to a stub that does the same copy with the amount ANDed with a
//     mask the host owns: all ones for on, 0 for off. The code is written
//     once, at load; turning it on or off is one store to the mask.
//   - Motion blur and depth of field are shader patches (host/shader_patch.h).
//   - Bloom and vignette are two of the area's draw parameters, changed in the
//     block the game hands YEBIS each frame - "Draw parameters" below.
constexpr std::uint64_t kRenderView = 0x269e990;
// push rbp; mov rbp, rsp; push r15..r12; push rbx; sub rsp, 0xea8
const std::uint8_t kRenderViewPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                            0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xa8, 0x0e, 0x00, 0x00};
constexpr std::size_t kCapabilities = 0x48, kSsaoFlag = 0x2968, kAaFlag = 0x296a;
constexpr std::uint64_t kSsaoCapability = 0x8, kAaCapability = 0x10;
constexpr std::uint64_t kCaCopy = 0x269faa8;
// mov eax, [rbp-0xa70] ; mov [rbx+0xac], eax
const std::uint8_t kCaCopyBytes[] = {0x8b, 0x85, 0x90, 0xf5, 0xff, 0xff, 0x89, 0x83, 0xac, 0x00, 0x00, 0x00};

std::atomic<std::uint32_t>* g_ca_mask = nullptr;  // in the stub's page
std::atomic<bool> g_ssao{true}, g_aa{true};
// Bumped when the SSAO or AA setting changes: each renderer is given the
// values once a generation (render_view_hook). 0: never changed - the game's.
std::atomic<std::uint32_t> g_flags_gen{0};
// The renderers given them, and the generation each was given (render_view_hook,
// the render thread alone).
struct GivenRenderer {
    std::uint8_t* renderer;
    std::uint32_t gen;
};
GivenRenderer g_given[16] = {};
unsigned g_given_next = 0;
// Scene contexts constructed since the last view (graphics_scene_context_made):
// the constructor turns SSAO and AA on again, so one made where an earlier
// one stood - a menu's context made on demand, freed and made again at the
// same address - is given the settings again at its next view.
std::mutex g_made_mu;
std::vector<std::uint64_t> g_made;  // under g_made_mu
std::atomic<bool> g_made_any{false};

// ---- Draw parameters ---------------------------------------------------------
//
// Post-processing comes from the area's GPARAM (param/drawparam, read and
// written by tools/gparam.py): the graphics manager keeps one bank per group -
// Tone Map at +0x5978, Bloom at +0x5980, and so on - and each view, the render
// entry (sub_269e990) looks up two entries per bank by the ids the renderer
// holds and blends them field by field (sub_128a710 for the tone map,
// sub_126d790 for bloom) into a block on its stack. From there the values go
// to the global post-process block at 0x59406e0: the tone map's copied by the
// same function, bloom's by sub_25d5050. And then sub_25d4860 records the
// YEBIS pass, which starts by copying all 0x370 bytes of that block into the
// command it records.
//
// So its entry is the one place where every value is this view's and none has
// been used yet, and a change made there is made again the next view from the
// area's own numbers - nothing to keep, nothing to restore, and the area's
// value is what a setting at its default leaves alone. When the YEBIS object's
// +0x2710 byte is set, the game skips all of those copies (the same flag
// guards the tone map's, bloom's and light shafts'), so the block holds what an
// earlier view left - already changed - and is left as it is.
//
//   +0xc8  saturation         tone map, entry +0x20 ("Yebis-ColorS")
//   +0xd0  vignette power     tone map, entry +0x70 ("Yebis-Vignette Power")
//   +0xdc  glare luminance    bloom, entry +0x10 ("Yebis-GlareLuminance")
//
// The tone map's entry is its GPARAM group in order, four bytes a parameter
// from +0x8, and the block takes it from its +0x10 ("Yebis-FilmicContrast",
// +0xa8) through +0x28 ("Yebis-Contrust", +0x248). Neither of those two makes
// a contrast setting: Contrust is 1.0 in every area and so steep that 0.8
// washes the clinic out to grey (mean 15 -> 83) and 1.2 crushes it to black;
// FilmicContrast mostly moves the overall brightness, which the game's own
// Brightness already does.
//
// Nor does motion blur's strength, though "Yebis-MotionBlurScale" (+0x1ec) is
// right there: at half and at double the area's value the game never fades in
// from its loading screen (twice at half, three times at double; the glare
// that eases in as the world starts stays at its first value), while 0, 1x
// and 1.4x load normally. Motion blur keeps its on/off setting.
constexpr std::uint64_t kYebisRecord = 0x25d4860;
// push rbp; mov rbp, rsp; push r15..r12; push rbx; sub rsp, 0x28; mov r13, r9
const std::uint8_t kYebisRecordPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                             0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28, 0x4d, 0x89, 0xcd};
constexpr std::uint64_t kPostBlock = 0x59406e0;
constexpr std::size_t kSaturation = 0xc8, kVignettePower = 0xd0, kGlareLuminance = 0xdc;
constexpr std::size_t kYebisHeld = 0x2710;  // on the YEBIS object: the block was not refreshed
std::uint8_t* g_post_block = nullptr;
std::atomic<float> g_bloom{1.0f}, g_saturation{1.0f};
std::atomic<bool> g_vignette{true};

// Ambient occlusion strength. The SSAO bank (+0x5968) is blended the same way
// (sub_1285500, into a 0x6c-byte block) and copied into the SSAO object the
// renderer keeps at +0x10d0 - to its +0x14c, unless its +0x148 byte holds it -
// right before sub_12669d0 draws the pass with it. Every area uses the second
// version of the pass (UseNewVersion 1), whose output is the visibility times
// "Ver2 AO Scale" (+0x44, 1.0 everywhere) plus "Ver2 AO Offset" (+0x48, 0):
// scale 0 left the clinic black (mean 15.4 -> 3.1) and 2 brightened it. So
// strength k makes the output 1 - k(1 - out) - the scale times k, the offset
// times k plus 1 - k: 0 is no occlusion, 1 the area's, 2 twice as dark.
constexpr std::uint64_t kSsaoDraw = 0x12669d0;
// push rbp; mov rbp, rsp; push r15..r12; push rbx; sub rsp, 0x2a8
const std::uint8_t kSsaoDrawPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                          0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xa8, 0x02, 0x00, 0x00};
constexpr std::size_t kSsaoHeld = 0x148, kSsaoBlock = 0x14c;  // on the SSAO object
constexpr std::size_t kSsaoScale = 0x44, kSsaoOffset = 0x48;  // in its block
std::atomic<float> g_ao_strength{1.0f};

// Fog - the haze over distant streets outdoors (Fog 0 in Central Yharnam:
// 41% of the frame, mean 36.7 -> 29.4, the far buildings crisp; the clinic's
// interiors have none, only particles read it there). Not a bank on the
// graphics manager and not one block: the fog manager
// at its +0x5888 (sub_26d7c90 registers its parameters) keeps a table of
// 0x60-byte blocks at +0x250 - 681 of them, count at +0x258 - one per fog the
// area defines, already in the form the shaders read. Each drawable carries
// fog ids that become an index into it (sub_264d020 passes the index in its
// per-draw constants), and a few kinds of object copy their block out
// directly (sub_26dab00: the id map at +0x248, then table + index * 0x60).
// Read against the clinic's GPARAM (Dist 15/80, heightFogDensity 14/6,
// DistOffset 20/15, HeightOffset 20/45, fogContrast 0.46), a block is
//
//   [0] opacity  1/Dist   ?        1/heightFogDensity
//   [1] colour r g b               contrast
//   [2] opacity  DistOffset 1/Dist  1/heightFogDensity
//   [3] height opacity  HeightOffset  ?   -39.1
//   [4] height colour r g b        contrast
//   [5] an int (1)
//
// and sub_26da690, which builds it, says what the opacities are: the fog
// colour's alpha times the area's fogAlpha for [0] and [2], the height
// colour's for [3]. So the Fog setting scales those three.
//
// It has to be done at the right moment. sub_26da690 runs as a command the
// render entry queues (sub_261c270), rebuilds every block from the area's fog
// entries each time, and ends with a tail call - `jmp sub_122e6b0` at
// 0x26daadf, with the table in rdx - that copies the table to the GPU.
// Written any earlier, the rebuild undoes it (tried: the table changed and
// the frame did not). So that jump goes through a stub that scales the fresh
// table and carries on into the copy: nothing to keep, nothing to restore.
// sub_122e6b0 is shared with two other managers' uploads, which is why the
// jump rather than its entry.
constexpr std::uint64_t kGraphicsManager = 0x5940dd8;
constexpr std::size_t kFogManager = 0x5888, kFogMap = 0x248, kFogTable = 0x250, kFogCount = 0x258;
constexpr std::size_t kFogBlock = 0x60;
constexpr int kFogOpacity[] = {0, 8, 12};  // floats within a block
constexpr std::uint64_t kFogUploadJump = 0x26daadf, kTableUpload = 0x122e6b0;
std::atomic<float> g_fog{1.0f};
// BBHOST_FOG_TEST=1=10,10=10: multiply those floats of every block too, at
// the upload, to see which ones the frame answers to; `0+0.8` adds instead,
// for a place whose fog has no opacity to scale.
struct FogTest {
    int index[8];
    float mul[8];
    bool add[8];
    int n = 0;
};
const FogTest g_fog_test = [] {
    FogTest t;
    const char* e = std::getenv("BBHOST_FOG_TEST");
    while (e && *e && t.n < 8) {
        char* end = nullptr;
        const long i = std::strtol(e, &end, 0);
        if (!end || (*end != '=' && *end != '+') || i < 0 || i >= static_cast<long>(kFogBlock / 4)) break;
        t.index[t.n] = static_cast<int>(i);
        t.add[t.n] = *end == '+';
        t.mul[t.n] = std::strtof(end + 1, &end);
        ++t.n;
        e = end && *end == ',' ? end + 1 : nullptr;
    }
    return t;
}();

// Shadow distance. The ShadowParam bank (+0x59d0) goes the same way into a
// block the renderer keeps at +0x80 - sub_266c600 blends it there - and
// sub_269cb70, which draws the directional light's cascades, hands that block
// straight to sub_12ac210 to set the shadow object up. That is the only reader
// of the splits: the lighting pass takes them from the shadow object, so
// everything that uses them sees the same ones. It reads
//
//   +0x34  near             +0x40  cascade 0->1   (kept: close shadows' detail)
//   +0x38  far fade start   +0x44  cascade 1->2
//   +0x3c  far fade length  +0x48  cascade 2->3
//
// and the last cascade reaches to fade start + length, fade start being at
// least the 2->3 split (2 / 8 / 15 m and a 10-15 m fade in the first areas,
// so shadows end near 30 m). The renderer's +0x297c byte, when set, keeps the
// block from being blended again - then it holds what was scaled last time and
// is left alone. Its +0x90 is a shadow-map size index (2048, 1024, 2048, 4096
// from 0x4ca9668); most areas already ask for 4096.
constexpr std::uint64_t kShadowSetup = 0x12ac210;
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; sub rsp, 0x18;
// mov r14, rsi; mov rbx, rdi
const std::uint8_t kShadowSetupPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48,
                                             0x83, 0xec, 0x18, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
constexpr std::size_t kShadowBlock = 0x80, kShadowHeld = 0x297c;  // on the renderer
constexpr std::size_t kShadowScaled[] = {0x38, 0x3c, 0x44, 0x48};
std::atomic<float> g_shadow_scale{1.0f};
// BBHOST_SHADOW_SCALE=0.1 stands in for the setting's x1.0, for an A/B run:
// start there, move the setting in the same run, and the one frame pair shows
// whether this place's shadows depend on the splits at all.
const float g_shadow_scale_env = [] {
    const char* e = std::getenv("BBHOST_SHADOW_SCALE");
    return e && *e ? std::strtof(e, nullptr) : 0.0f;
}();

void set_chromatic_aberration(bool on) {
    if (!g_ca_mask) return;
    const std::uint32_t want = on ? 0xffffffffu : 0u;
    if (g_ca_mask->exchange(want) != want) host_log("graphics: chromatic aberration %s", on ? "on" : "off");
}

void* alloc_exec_near(std::uint64_t at);
void free_exec_page(void* page);

// The copy, through the mask. The stub has to be within a rel32 of the code,
// so its page comes from near it (alloc_exec_near). On Windows there was no
// such page here until 2026-10-14: the stub was never placed, every run said
// "it stays on", and the setting did nothing (Kyo: "chromatic aberration
// disable actually doesn't even currently work").
bool install_ca_stub(ElfImage* image, std::uint64_t at) {
    void* page = alloc_exec_near(at);
    if (!page) return false;
    auto* c = static_cast<std::uint8_t*>(page);
    const std::uint64_t stub = reinterpret_cast<std::uint64_t>(c);
    const std::uint64_t back = at + sizeof(kCaCopyBytes);
    auto rel32 = [](std::uint64_t from_end, std::uint64_t to, std::int32_t* out) {
        const std::int64_t d = static_cast<std::int64_t>(to) - static_cast<std::int64_t>(from_end);
        if (d < INT32_MIN || d > INT32_MAX) return false;
        *out = static_cast<std::int32_t>(d);
        return true;
    };
    std::int32_t to_stub = 0, to_back = 0, to_mask = 0;
    constexpr std::size_t kMaskAt = 0x40;
    // mov eax, [rbp-0xa70]
    std::memcpy(c, kCaCopyBytes, 6);
    // and eax, [rip+mask]
    c[6] = 0x23;
    c[7] = 0x05;
    // mov [rbx+0xac], eax
    std::memcpy(c + 12, kCaCopyBytes + 6, 6);
    // jmp back
    c[18] = 0xe9;
    if (!rel32(stub + 12, stub + kMaskAt, &to_mask) || !rel32(stub + 23, back, &to_back) ||
        !rel32(at + 5, stub, &to_stub)) {
        free_exec_page(page);
        return false;
    }
    std::memcpy(c + 8, &to_mask, 4);
    std::memcpy(c + 19, &to_back, 4);
    g_ca_mask = reinterpret_cast<std::atomic<std::uint32_t>*>(c + kMaskAt);
    g_ca_mask->store(0xffffffffu);
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, at));
    const std::uint64_t lo = at & ~0xfffull, hi = (at + sizeof(kCaCopyBytes) + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&image->mem, lo, hi - lo)) {
        g_ca_mask = nullptr;
        free_exec_page(page);
        return false;
    }
    p[0] = 0xe9;  // jmp stub
    std::memcpy(p + 1, &to_stub, 4);
    std::memset(p + 5, 0x90, sizeof(kCaCopyBytes) - 5);
    guest_protect_rx(&image->mem, lo, hi - lo);
    return true;
}

std::atomic<bool> g_motion_blur{true};
std::atomic<int> g_lod_bias{0};  // the model detail bias to keep (lod_bias_tick, below)
bool g_lod_bias_env = false;     // BBHOST_LOD_BIAS set: the setting does not move it

// Motion blur's velocity post-pass. Before YEBIS blurs, the scene renderer
// runs sub_1264430 (its only caller, 0x26ab381) when the graphics manager's
// +0x78b4 is set (`cmp byte [r12+0x78b4], 0; je` at 0x26ab345): two
// ping-pong passes over the velocity map into a half-res pair (+0xcf8/+0xd50)
// that only the motion blur reads. With the Motion Blur setting off the
// blur's shader is patched away (host/shader_patch.cpp), so the pass is work
// nobody looks at. The switch is written at its source instead of the pass
// being skipped by a hook on every call: the manager's constructor's
// immediate (`mov byte [rbx+0x78b4], 1` at 0x2695ce6) and the live manager's
// byte, once when the setting changes - the debug menu's "Enable VeloMap
// Dilation" is the same byte and keeps it in between. The velocity map itself
// is still drawn: leaving it out whites out the frame.
constexpr std::uint64_t kVelocityCtor = 0x2695ce6;
const std::uint8_t kVelocityCtorBytes[] = {0xc6, 0x83, 0xb4, 0x78, 0x00, 0x00, 0x01};
constexpr std::size_t kVelocitySwitch = 0x78b4;
std::uint8_t* g_velocity_ctor = nullptr;  // the constructor's immediate, when its bytes were the game's
GuestMemory g_velocity_mem{};
std::uint64_t g_velocity_slide = 0;
constexpr std::uint64_t kGraphicsManagerOfBlur = 0x5940dd8;

void motion_blur_apply(bool on) {
    const std::uint8_t want = on ? 1 : 0;
    if (g_velocity_ctor && *g_velocity_ctor != want) {
        const std::uint64_t at = reinterpret_cast<std::uint64_t>(g_velocity_ctor);
        if (guest_protect_rwx(&g_velocity_mem, at & ~0xfffull, 0x1000)) {
            *g_velocity_ctor = want;
            guest_protect_rx(&g_velocity_mem, at & ~0xfffull, 0x1000);
        }
    }
    std::uint64_t gm = 0;
    if (g_velocity_slide)
        std::memcpy(&gm, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g_velocity_slide + (kGraphicsManagerOfBlur - kPreferredGuestSlide))), 8);
    if (gm) *reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(gm + kVelocitySwitch)) = want;
    host_log("graphics: motion blur %s (its velocity pass %s)", on ? "on" : "off", on ? "runs" : "skipped at the graphics manager's switch");
}

// The settings' listener (host_settings_listen, from live_install): told each
// change once, on the thread that made it (the store calls its listeners one
// at a time), it keeps the values the hooks below apply - the hooks read these
// and never the settings, so a view costs no settings lookup at all. The
// hooks that scale an area's blended values (bloom, vignette, saturation, fog,
// AO strength, shadow distance) write every view, as they must: the game
// derives those values again for each view (the area's draw parameters), and
// a value applied once would last one view. SSAO and AA are given to each
// renderer once a generation (render_view_hook) and motion blur's switch is
// written once (motion_blur_apply), so the debug menu's switches over the same
// bytes keep what they set in between.
void settings_changed(const HostSettings& hs) {
    // SSAO and AA go to the renderers once each when they change - or at the
    // first view when the file asks for other than the game's own
    // (render_view_hook); never while they are the game's.
    static bool ssao_was = true, aa_was = true;
    if (hs.ssao != ssao_was || hs.anti_alias != aa_was) {
        ssao_was = hs.ssao;
        aa_was = hs.anti_alias;
        g_ssao.store(hs.ssao);
        g_aa.store(hs.anti_alias);
        g_flags_gen.fetch_add(1, std::memory_order_release);
    }
    static bool blur_was = true;
    if (hs.motion_blur != blur_was) {
        blur_was = hs.motion_blur;
        motion_blur_apply(hs.motion_blur);
    }
    set_chromatic_aberration(hs.chromatic_aberration && !hs.performance_mode);
    g_bloom.store(hs.bloom);
    g_saturation.store(hs.saturation);
    g_fog.store(hs.fog);
    g_ao_strength.store(hs.ao_strength);
    g_vignette.store(hs.vignette);
    g_motion_blur.store(hs.motion_blur);
    if (!g_lod_bias_env) g_lod_bias.store(hs.lod_bias);
    g_shadow_scale.store(hs.shadow_scale > 1.0f ? hs.shadow_scale
                                                : g_shadow_scale_env > 0.0f ? g_shadow_scale_env : 1.0f);
}

// sub_12669d0's entry: the SSAO object in rdi, its block just copied in.
GUEST_ABI std::int64_t ssao_draw_hook(std::uint64_t, const std::uint64_t* saved) {
    auto* ssao = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(saved[5]));
    const float k = g_ao_strength.load(std::memory_order_relaxed);
    if (!ssao || k == 1.0f || ssao[kSsaoHeld]) return 0;
    float v = 0, o = 0;
    std::memcpy(&v, ssao + kSsaoBlock + kSsaoScale, 4);
    std::memcpy(&o, ssao + kSsaoBlock + kSsaoOffset, 4);
    const float scale = v * k, offset = o * k + (1.0f - k);
    std::memcpy(ssao + kSsaoBlock + kSsaoScale, &scale, 4);
    std::memcpy(ssao + kSsaoBlock + kSsaoOffset, &offset, 4);
    static float logged = -1, logged_k = -1;
    static std::atomic<int> logs{0};
    if ((v != logged || k != logged_k) && logs.fetch_add(1) < 16) {
        host_log("graphics: area AO scale %g offset %g, strength x%g", v, o, k);
        logged = v;
        logged_k = k;
    }
    return 0;
}

// sub_12ac210's entry: the shadow object in rdi, the renderer's block in rsi.
GUEST_ABI std::int64_t shadow_setup_hook(std::uint64_t, const std::uint64_t* saved) {
    auto* block = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(saved[4]));
    const float k = g_shadow_scale.load(std::memory_order_relaxed);
    if (!block || k == 1.0f || block[kShadowHeld - kShadowBlock]) return 0;
    float v[4];
    for (int i = 0; i < 4; ++i) {
        std::memcpy(&v[i], block + kShadowScaled[i], 4);
        const float scaled = v[i] * k;
        std::memcpy(block + kShadowScaled[i], &scaled, 4);
    }
    static float logged = -1, logged_k = -1;
    static std::atomic<int> logs{0};
    if ((v[3] != logged || k != logged_k) && logs.fetch_add(1) < 16) {
        host_log("graphics: shadow splits 1->2 %g, 2->3 %g, fade %g + %g, x%g", v[2], v[3], v[0], v[1], k);
        logged = v[3];
        logged_k = k;
    }
    return 0;
}

// BBHOST_POST_POKE=0xc8=0,0xa8=0.8: floats written into the post-process
// block at the YEBIS record, after the settings - to see what a field does
// before a setting is built on it.
struct PostPoke {
    std::uint32_t off[8];
    float value[8];
    int n = 0;
};
const PostPoke g_post_poke = [] {
    PostPoke p;
    const char* e = std::getenv("BBHOST_POST_POKE");
    while (e && *e && p.n < 8) {
        char* end = nullptr;
        const unsigned long off = std::strtoul(e, &end, 0);
        if (!end || *end != '=' || off + 4 > 0x370) break;
        p.off[p.n] = static_cast<std::uint32_t>(off);
        p.value[p.n] = std::strtof(end + 1, &end);
        ++p.n;
        e = end && *end == ',' ? end + 1 : nullptr;
    }
    return p;
}();

// What Bloom does to YEBIS's glare, read from the eboot (2026-10-08). The
// executor of the pass this hook records (sub_25d5f80) hands the block's
// glare luminance (+0xdc) to YEBIS (0x25d72eb -> sub_fd4b60, which keeps it
// at +0x6bc of YEBIS's context), and YEBIS's glare (sub_fdd560) draws a
// camera's glare - the bright pass, the blurred pyramid and its sum - only
// when that camera's glare is enabled and the luminance is above 0: at
// 0xfdd688 `0 >= luminance` skips the camera's whole glare. Anywhere else the
// luminance is one factor of a shader constant (0xfed5f5, divided by +0x710).
// So:
//   - Bloom 0 is the game's own no-glare path: none of the glare's passes
//     run, and the frame is cheaper by all of them. That is the whole
//     difference players see between 0 and the rest.
//   - Bloom 1-9 is the same passes as 10, at another strength: it costs what
//     the game's own glare costs, no more.
//   - Neither a change of the setting nor the area's own value easing from one
//     lighting set to the next (a new value every view for a second or so)
//     makes YEBIS rebuild anything: the value is stored and read, and this
//     hook only multiplies it in place.
// What it did cost: the log line below, on every change of either value - so
// the first transition used up its 24 lines in under a second and nothing
// after it said what Bloom was doing. A line now comes once per value that
// settles (unchanged for kGlareSettle views) or per setting changed, with the
// views since the last and how many had no glare; the 300-flip report
// counts them too (graphics_glare_report).
constexpr int kGlareSettle = 30;     // views a value holds before it is logged
constexpr int kGlareLogLines = 200;  // per run
std::atomic<std::uint64_t> g_glare_views{0}, g_glare_skipped{0};

struct GlareLog {
    float seen_glare = -1, seen_power = -1;  // the area's values in the last view
    int stable = 0;                          // views they have held
    float glare = -1, power = -1;            // as last logged
    float bloom = -1;
    bool vignette = true;
    int lines = 0;
    std::uint64_t views = 0, skipped = 0;  // the counters at the last line
};
GlareLog g_glare_log;  // the YEBIS record hook's
std::atomic_flag g_glare_log_busy = ATOMIC_FLAG_INIT;  // one render thread records the views; a second skips the line

void log_glare(float glare, float power, float bloom, bool vignette) {
    if (g_glare_log_busy.test_and_set(std::memory_order_acquire)) return;
    struct Release {
        ~Release() { g_glare_log_busy.clear(std::memory_order_release); }
    } release;
    GlareLog& l = g_glare_log;
    if (glare != l.seen_glare || power != l.seen_power) {
        l.seen_glare = glare;
        l.seen_power = power;
        l.stable = 0;
    } else if (l.stable < kGlareSettle) {
        ++l.stable;
    }
    const bool first = l.lines == 0;
    if (!first && (l.stable < kGlareSettle || l.lines >= kGlareLogLines)) return;
    if (!first && glare == l.glare && power == l.power && bloom == l.bloom && vignette == l.vignette) return;
    const std::uint64_t views = g_glare_views.load(std::memory_order_relaxed);
    const std::uint64_t skipped = g_glare_skipped.load(std::memory_order_relaxed);
    const float sent = glare * bloom;
    // The area's own values, as they arrive: 0 vignette means the place has
    // none, which is why turning it off there changes nothing.
    host_log("graphics: area glare luminance %g (bloom %d/10: YEBIS gets %g%s), vignette power %g (%s); "
             "%llu views since the last line, %llu of them with no glare",
             glare, static_cast<int>(bloom * 10.0f + 0.5f), sent, sent > 0.0f ? "" : " - its glare passes skipped", power,
             vignette ? "on" : "off", static_cast<unsigned long long>(views - l.views),
             static_cast<unsigned long long>(skipped - l.skipped));
    if (++l.lines == kGlareLogLines) host_log("graphics: no more glare lines this run (the 300-flip report still counts them)");
    l.glare = glare;
    l.power = power;
    l.bloom = bloom;
    l.vignette = vignette;
    l.views = views;
    l.skipped = skipped;
}

// sub_25d4860's entry: the YEBIS object in rdi.
GUEST_ABI std::int64_t yebis_record_hook(std::uint64_t, const std::uint64_t* saved) {
    const auto* yebis = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(saved[5]));
    if (!g_post_block || !yebis || yebis[kYebisHeld]) return 0;
    const float bloom = g_bloom.load(std::memory_order_relaxed);
    const bool vignette = g_vignette.load(std::memory_order_relaxed);
    const float saturation = g_saturation.load(std::memory_order_relaxed);
    if (saturation != 1.0f) {
        float s = 0;
        std::memcpy(&s, g_post_block + kSaturation, 4);
        s *= saturation;
        std::memcpy(g_post_block + kSaturation, &s, 4);
    }
    float glare = 0, power = 0;
    std::memcpy(&glare, g_post_block + kGlareLuminance, 4);
    std::memcpy(&power, g_post_block + kVignettePower, 4);
    g_glare_views.fetch_add(1, std::memory_order_relaxed);
    if (!(glare * bloom > 0.0f)) g_glare_skipped.fetch_add(1, std::memory_order_relaxed);
    log_glare(glare, power, bloom, vignette);
    if (bloom != 1.0f) {
        glare *= bloom;
        std::memcpy(g_post_block + kGlareLuminance, &glare, 4);
    }
    if (!vignette && power != 0.0f) {
        power = 0.0f;
        std::memcpy(g_post_block + kVignettePower, &power, 4);
    }
    for (int i = 0; i < g_post_poke.n; ++i) {
        std::memcpy(g_post_block + g_post_poke.off[i], &g_post_poke.value[i], 4);
    }
    return 0;
}

// Where the copy lands: the post-process block's chromatic aberration amount.
constexpr std::uint64_t kCaAmount = 0x594078c;
const float* g_ca_amount = nullptr;

// BBHOST_DRAWPARAM_PROBE=1: which draw-parameter entries each view asks for,
// and whether they exist. sub_269e990 fetches two entries from each of twelve
// banks on the graphics manager (+0x5970..+0x59c8) by two ids the renderer
// holds (from +0x2a78 and +0x2ae0, blended by a float from +0x2b48 - but not
// in bank order, so the table below has them pair by pair), and a bank that
// has no entry for an id - nor for bank 1's id 0, which the lookup tries next
// - answers with the built-in entry it carries. That entry is at a different
// place in each bank: the lookup's last `add r15, N` before it returns says
// where (+0x1f0 Setting Info, +0x230 Dof, +0x2d8 Tone Map ...), and the
// Setting Info's - DepthOfFieldQuality 1 - is how a view can end up with a
// depth-of-field context that refuses its area's edge quality.
// Logged when the ids change. The banks follow the GPARAM's groups: +0x5968
// SSAO, +0x5970 Setting Info, +0x5978 Tone Map, +0x5980 Bloom, +0x5988
// LightShaft, +0x5990 Dof, +0x5998 Anti-Alias, +0x59a0 MotionBlur, +0x59a8
// FeedBackBlur, +0x59b0 ColorGrading, +0x59d0 ShadowParam.
struct BankLookup {
    std::uint32_t bank;      // on the manager
    std::uint64_t lookup;    // the function sub_269e990 calls for it
    std::uint32_t a, b;      // the renderer's two ids for it
    std::uint32_t weight;    // and the float that blends them
    std::uint32_t fallback;  // its built-in entry, what an id it has no entry for gets
};
// Pairs as sub_269e990 makes them - the ids are not in bank order.
const BankLookup kBanks[12] = {
    {0x5970, 0x25fe970, 0x2a78, 0x2ae0, 0x2b48, 0x1f0}, {0x5978, 0x26044a0, 0x2a80, 0x2ae8, 0x2b4c, 0x2d8},
    {0x5980, 0x25e62d0, 0x2a90, 0x2af8, 0x2b54, 0x2d0}, {0x5988, 0x25f68b0, 0x2ac0, 0x2b28, 0x2b6c, 0x210},
    {0x5990, 0x25ef2e0, 0x2a88, 0x2af0, 0x2b50, 0x230}, {0x5998, 0x25e14c0, 0x2ab0, 0x2b18, 0x2b64, 0x168},
    {0x59a0, 0x25fa460, 0x2aa8, 0x2b10, 0x2b60, 0x190}, {0x59a8, 0x25f2b90, 0x2aa0, 0x2b08, 0x2b5c, 0x160},
    {0x59b0, 0x25ea1e0, 0x2a98, 0x2b00, 0x2b58, 0x5a0}, {0x59b8, 0x127b3b0, 0x2ac8, 0x2b30, 0x2b70, 0x140},
    {0x59c0, 0x1274bd0, 0x2ad0, 0x2b38, 0x2b74, 0x1f0}, {0x59c8, 0x127dad0, 0x2ad8, 0x2b40, 0x2b78, 0x1c0},
};
std::uint64_t g_slide = 0;

// BBHOST_DRAWPARAM_POKE=<bank>:<offset>:<float>:<expect>: write that float
// into both entries a view asks that bank for, every view - to see whether a
// field is read live, before anything is built on it. It is also how to test a
// setting where the area has nothing to change: 0x5978:0x70:0.4:0 gives the
// clinic a vignette, which Vignette Off then has to take away again. Only into an entry whose
// field holds <expect> (or the value already): the title's views ask with ids
// ffffffff:ffffffff and get an entry laid out differently, and a float in its
// int field fails an assert.
struct Poke {
    std::uint32_t bank = 0, offset = 0;
    float value = 0, expect = 0;
};
const Poke g_poke = [] {
    Poke p;
    if (const char* e = std::getenv("BBHOST_DRAWPARAM_POKE")) {
        char* end = nullptr;
        p.bank = static_cast<std::uint32_t>(std::strtoul(e, &end, 0));
        if (end && *end == ':') p.offset = static_cast<std::uint32_t>(std::strtoul(end + 1, &end, 0));
        if (end && *end == ':') p.value = std::strtof(end + 1, &end);
        if (end && *end == ':') p.expect = std::strtof(end + 1, nullptr);
    }
    return p;
}();

void drawparam_poke(const std::uint8_t* mgr, const std::uint8_t* r) {
    if (!g_poke.bank || !mgr || !r) return;
    for (int b = 0; b < 12; ++b) {
        if (kBanks[b].bank != g_poke.bank) continue;
        std::uint64_t bank = 0;
        std::memcpy(&bank, mgr + kBanks[b].bank, 8);
        if (!bank) return;
        void* fn = reinterpret_cast<void*>(static_cast<std::uintptr_t>(g_slide + (kBanks[b].lookup - kPreferredGuestSlide)));
        for (std::uint32_t id : {kBanks[b].a, kBanks[b].b}) {
            std::uint32_t key[2];
            std::memcpy(key, r + id, sizeof(key));
            if (key[0] == 0xffffffffu || key[1] == 0xffffffffu) continue;
            const std::uint64_t e = hle_call_guest<std::uint64_t>(fn, bank, r + id);
            if (!e) continue;
            float* f = reinterpret_cast<float*>(static_cast<std::uintptr_t>(e + g_poke.offset));
            if (*f != g_poke.value && std::fabs(*f - g_poke.expect) < 1e-4f) {
                static int logs = 0;
                if (logs++ < 4) host_log("drawparam: poke bank +0x%x entry %p +0x%x: %g -> %g", g_poke.bank,
                                         reinterpret_cast<void*>(static_cast<std::uintptr_t>(e)), g_poke.offset, *f,
                                         g_poke.value);
                *f = g_poke.value;
            }
        }
    }
}

// BBHOST_FOG_PROBE=1: the fog table, once the world is up - the fog manager
// at the graphics manager's +0x5888, its 0x60-byte blocks at +0x250, their
// count at +0x258.
void fog_probe(const std::uint8_t* mgr, const std::uint8_t* r) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_FOG_PROBE");
        return e && e[0] == '1';
    }();
    static int views = 0;
    if (!on || !mgr || !r || views < 0) return;
    std::uint64_t probe_id = 0;
    std::memcpy(&probe_id, r + 0x2a80, 8);
    if (static_cast<std::uint32_t>(probe_id) == 0xffffffffu || ++views < 200) return;
    views = -1;
    std::uint64_t gm = 0, fm = 0, table = 0;
    std::memcpy(&gm, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g_slide + (0x5940dd8 - kPreferredGuestSlide))), 8);
    if (gm) std::memcpy(&fm, reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(gm)) + 0x5888, 8);
    host_log("fog: graphics manager %p (render view's %p), fog manager %p", reinterpret_cast<void*>(static_cast<std::uintptr_t>(gm)),
             static_cast<const void*>(mgr), reinterpret_cast<void*>(static_cast<std::uintptr_t>(fm)));
    if (!fm) return;
    std::memcpy(&table, reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(fm)) + 0x250, 8);
    std::uint32_t count = 0;
    std::memcpy(&count, reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(fm)) + 0x258, 4);
    host_log("fog: manager %p table %p count %u", reinterpret_cast<void*>(static_cast<std::uintptr_t>(fm)),
             reinterpret_cast<void*>(static_cast<std::uintptr_t>(table)), count);
    if (!table || count > 4096) return;
    int shown = 0;
    for (std::uint32_t i = 0; i < count && shown < 24; ++i) {
        const auto* q = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(table + i * 0x60ull));
        bool any = false;
        for (int k = 0; k < 24; ++k) any |= q[k] != 0.0f;
        if (!any) continue;
        ++shown;
        char line[600];
        int n = std::snprintf(line, sizeof(line), "fog: [%u]", i);
        for (int k = 0; k < 24 && n < 560; ++k) n += std::snprintf(line + n, sizeof(line) - n, "%s%g", k % 4 ? " " : " | ", static_cast<double>(q[k]));
        host_log("%s", line);
    }
}

void drawparam_probe(const std::uint8_t* mgr, const std::uint8_t* r) {
    drawparam_poke(mgr, r);
    fog_probe(mgr, r);
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_DRAWPARAM_PROBE");
        return e && e[0] == '1';
    }();
    if (!on || !mgr || !r) return;
    // Once: each bank's built-in entry, the one a missing id gets.
    static bool dumped = false;
    std::uint64_t probe_id = 0;
    std::memcpy(&probe_id, r + 0x2a80, 8);
    if (!dumped && static_cast<std::uint32_t>(probe_id) != 0xffffffffu) {
        dumped = true;
        for (const BankLookup& bk : kBanks) {
            std::uint64_t bank = 0;
            std::memcpy(&bank, mgr + bk.bank, 8);
            if (!bank || !hle_kernel_va_mapped(bank + bk.fallback, 0x60)) {
                host_log("drawparam: slot +0x%x = %p (not a bank)", bk.bank, reinterpret_cast<void*>(static_cast<std::uintptr_t>(bank)));
                continue;
            }
            const auto* q = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(bank + bk.fallback));
            char line[400];
            int n = std::snprintf(line, sizeof(line), "drawparam: slot +0x%x default (+0x%x):", bk.bank, bk.fallback);
            for (int k = 0; k < 20 && n < 380; ++k) {
                float f;
                std::memcpy(&f, &q[k], 4);
                const bool looks_float = q[k] > 0x100 && (q[k] & 0x7fffffffu) < 0x7f800000u && (f > 1e-4f || f < -1e-4f);
                if (looks_float) {
                    n += std::snprintf(line + n, sizeof(line) - n, " %g", static_cast<double>(f));
                } else {
                    n += std::snprintf(line + n, sizeof(line) - n, " %x", q[k]);
                }
            }
            host_log("%s", line);
        }
    }
    static std::uint64_t last[12][2] = {};
    static int logs = 0;
    for (int b = 0; b < 12 && logs < 400; ++b) {
        std::uint64_t ida = 0, idb = 0;
        float w = 0;
        std::memcpy(&ida, r + kBanks[b].a, 8);
        std::memcpy(&idb, r + kBanks[b].b, 8);
        std::memcpy(&w, r + kBanks[b].weight, 4);
        if (ida == last[b][0] && idb == last[b][1]) continue;
        last[b][0] = ida;
        last[b][1] = idb;
        std::uint64_t bank = 0;
        std::memcpy(&bank, mgr + kBanks[b].bank, 8);
        if (!bank) continue;
        void* fn = reinterpret_cast<void*>(static_cast<std::uintptr_t>(g_slide + (kBanks[b].lookup - kPreferredGuestSlide)));
        const std::uint64_t ea = hle_call_guest<std::uint64_t>(fn, bank, r + kBanks[b].a);
        const std::uint64_t eb = hle_call_guest<std::uint64_t>(fn, bank, r + kBanks[b].b);
        ++logs;
        const std::uint64_t fallback = bank + kBanks[b].fallback;
        if (ea && ea != fallback) {
            const auto* q = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(ea));
            host_log("drawparam:   entry %p: %08x %08x %08x %08x %08x %08x %08x %08x | %08x %08x %08x %08x %08x %08x %08x %08x",
                     reinterpret_cast<const void*>(q), q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10],
                     q[11], q[12], q[13], q[14], q[15]);
        }
        host_log("drawparam: bank +0x%x (%p): A %08x:%08x -> %s, B %08x:%08x -> %s, weight %g", kBanks[b].bank,
                 reinterpret_cast<void*>(static_cast<std::uintptr_t>(bank)), static_cast<unsigned>(ida),
                 static_cast<unsigned>(ida >> 32), ea == fallback ? "DEFAULT" : "entry",
                 static_cast<unsigned>(idb), static_cast<unsigned>(idb >> 32), eb == fallback ? "DEFAULT" : "entry",
                 w);
    }
}

// The render-view entry, before the renderer in r9 draws the view.
// At the fog table's upload: rdi the id map, rsi the command context, rdx
// the table, all three checked against the fog manager before anything is
// written.
GUEST_ABI std::int64_t fog_upload_hook(std::uint64_t, const std::uint64_t* saved) {
    const float k = g_fog.load(std::memory_order_relaxed);
    if (k == 1.0f && g_fog_test.n == 0) return 0;
    std::uint64_t gm = 0, fm = 0;
    std::memcpy(&gm, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g_slide + (kGraphicsManager - kPreferredGuestSlide))), 8);
    if (gm) std::memcpy(&fm, reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(gm)) + kFogManager, 8);
    if (!fm) return 0;
    const auto* f = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(fm));
    std::uint64_t map = 0, table = 0;
    std::uint32_t count = 0;
    std::memcpy(&map, f + kFogMap, 8);
    std::memcpy(&table, f + kFogTable, 8);
    std::memcpy(&count, f + kFogCount, 4);
    if (map != saved[5] || table != saved[3] || !table || count == 0 || count > 4096) return 0;
    auto* blocks = reinterpret_cast<float*>(static_cast<std::uintptr_t>(table));
    for (std::uint32_t i = 0; i < count; ++i) {
        float* b = blocks + i * (kFogBlock / 4);
        for (int o : kFogOpacity) b[o] *= k;
        for (int t = 0; t < g_fog_test.n; ++t) {
            float& v = b[g_fog_test.index[t]];
            v = g_fog_test.add[t] ? v + g_fog_test.mul[t] : v * g_fog_test.mul[t];
        }
    }
    static float logged = -1.0f;
    if (k != logged) {
        host_log("graphics: fog x%g over %u blocks", k, count);
        logged = k;
    }
    return 0;
}

void free_exec_page(void* page) {
#if defined(_WIN32)
    VirtualFree(page, 0, MEM_RELEASE);
#else
    munmap(page, 0x1000);
#endif
}

// An executable page within a rel32 of `at`, for a stub a jump or call there
// can reach. Null when none is free.
void* alloc_exec_near(std::uint64_t at) {
#if defined(_WIN32)
    // Downwards from `at` and then upwards, a 64 KiB allocation granule at a
    // time, inside the reach of a rel32.
    const std::uint64_t reach = 0x7ff00000ull, base = at & ~0xffffull;
    for (std::uint64_t a = base - 0x10000; a > 0x10000 && base - a < reach; a -= 0x10000)
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)) return p;
    for (std::uint64_t a = base + 0x10000; a - base < reach; a += 0x10000)
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)) return p;
    return nullptr;
#else
    void* page = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (page == MAP_FAILED) return nullptr;
    const std::int64_t d = static_cast<std::int64_t>(reinterpret_cast<std::uint64_t>(page)) - static_cast<std::int64_t>(at);
    if (d < INT32_MIN + 0x1000ll || d > INT32_MAX - 0x1000ll) {
        munmap(page, 0x1000);
        return nullptr;
    }
    return page;
#endif
}

// A `jmp rel32` (or `call rel32`) at `at` sent through a stub that calls
// `host` with the argument registers and then goes on to `target` - or, when
// host returns nonzero, returns to the call's return address instead (for a
// call: host ran the target itself, around its own work). For a jump the
// stack is as at a function's entry (a tail call).
bool install_jump_hook(ElfImage* image, std::uint64_t at, std::uint64_t target, void* host) {
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, at));
    std::int32_t rel = 0;
    std::memcpy(&rel, p + 1, 4);
    if ((p[0] != 0xe9 && p[0] != 0xe8) || at + 5 + static_cast<std::int64_t>(rel) != target) return false;
    void* page = alloc_exec_near(at);
    if (!page) return false;
    const std::uint64_t stub = reinterpret_cast<std::uint64_t>(page);
    const std::int64_t d = static_cast<std::int64_t>(stub) - static_cast<std::int64_t>(at + 5);
    thunk_emit_prologue_stub(static_cast<std::uint8_t*>(page), 0, thunk_wrap(host), target, nullptr, 0);
    const std::uint64_t lo = at & ~0xfffull, hi = (at + 5 + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&image->mem, lo, hi - lo)) return false;  // the page stays; it is one
    const std::int32_t to_stub = static_cast<std::int32_t>(d);
    std::memcpy(p + 1, &to_stub, 4);
    guest_protect_rx(&image->mem, lo, hi - lo);
    return true;
}

// ---- Light grid ----------------------------------------------------------------
//
// The deferred renderer's local lights (types 0x65-0x6d) are not drawn once
// each: every frame sub_26a0a30 cuts the light pass's target - its width
// (+0x7c of the pass's camera record) and height (+0x80) plus twice a guard
// band (+0x78) - into tiles of the GraphicsManager's "DynamicLightGridSize"
// (+0x78a8 / +0x78ac, 128 x 128 from its constructor sub_26958d0; the
// manager is at 0x5940dd8), splits the tiles into "Light Grid Div" x by y
// jobs (0x553ae80 / 0x553ae84, 3 and 3 in .data), and each job (sub_26b4660)
// bins every light into each tile it touches (sub_12a4410) and draws the
// tile's lights under the tile's scissor. A light over k tiles is drawn k
// times: a light filling a 1920x1080 frame is 135 draws of the same pixels.
//
// Bigger tiles are the same image - a tile's rect is clamped to the frame
// (0x26a5e3e) - with fewer draws and less binning; the community patches
// "REAL LightGrid 1080p" and "Light Grid Div 1x1" make it one 1920x1080 tile
// in one job. But a tile's lights are shaded as the tile's quad, so a light
// in a frame-sized tile shades the whole frame. Measured (2026-10-03, pinned,
// the frozen seed in Cathedral Ward, 1080p; GPU ms a frame / main-loop work
// ms; the game's own 5.8-5.9 / 12.7-13.1, 454k draws per 300 frames):
//
//   one tile, the frame        6.32-6.40 / 12.2-12.3   363k draws
//   2 x 2 tiles (960x540)      6.14 / 12.1
//   4 x 4 tiles (480x270)      5.98 / 12.5             373k draws
//
// so the tile here is a quarter of the frame on each axis - most of the CPU
// saved, none of the GPU added - and the frame is the larger of the
// presented image (window, or display in fullscreen) and the render size,
// with one job ("Div" 1x1). Written on every view when it changed: the size
// is read each frame (0x26a5d22, 0x26b4cb7) and the bins are allocated per
// frame, so a resize or a new render size takes effect on the next frame.
// Never 0 - it is a divisor (0x26a5d43). BBHOST_LIGHT_GRID=0 keeps the
// game's tiles and jobs; =<N> N tiles a side (1: one tile, the frame); =WxH
// a fixed tile, for measuring.
constexpr std::uint64_t kGraphicsManagerSlot = 0x5940dd8;
constexpr std::uint64_t kLightGridDiv = 0x553ae80;  // x, y
constexpr std::size_t kLightTileW = 0x78a8, kLightTileH = 0x78ac;
const std::uint64_t* g_graphics_manager = nullptr;  // the slot, slid; null when off
const std::uint32_t* g_res_words = nullptr;         // render width, height
std::uint32_t g_tile_fixed[2] = {0, 0};             // BBHOST_LIGHT_GRID=WxH: that tile instead (experiments)
// Tiles a side: 4 - a quarter of the window each way, in one job ("Light
// Grid Div 1x1"); 1 with BBHOST_LIGHT_GRID=window: one tile the window's size,
// Kyo's "REAL LightGrid 1080p" (the constructor's 1920x1080) at whatever size
// the window or display is, each light binned and drawn once (=quarter names
// the default). The picture is the same - a tile only bounds where a light is
// shaded, and its rect is clamped to the frame - but a light in a frame-sized
// tile shades the whole frame. Measured: a Radeon 8060S, 2026-10-08, the
// Hunter's Dream, GPU ms per 300 flips, 4 a side 1,878-2,027 and one tile
// 1,960-2,096; 2026-10-12, Kyo's area uncapped and GPU-bound, Model detail
// Normal, 45 s each: one tile 160.2 fps, 4 a side 159.2, the game's own
// 128x128 tiles in 3x3 jobs (BBHOST_LIGHT_GRID=0) 165.1. 2026-10-14 at the
// Central Yharnam lamp, Uncapped: a Steam Deck at its 6 W limit (GPU-bound)
// one tile 23.3-24.0 fps, 4 a side 25.8-26.2, the game's 25.5-26.3, and the
// same picture (eight frames of each, twice: the pairs across configurations
// differ no more than two runs of one); an RTX 3070 Ti laptop one tile
// 83.0-83.5, 4 a side 90.7-91.5; an RTX 4070 (main-loop-bound) the same fps,
// one tile +10% GPU time. 2026-10-14 (Kyo), the room NVIDIA players drop to
// ~35 fps in (Kyo's save; 1080p, headless, the light pass's GPU ms a frame /
// draws a frame / the command processor's ms a frame, two runs each in
// A-B-C-C-B-A order): one tile 1.45-1.51 / 885, the game's tiles 0.87 / 1162,
// 4 a side 0.83-0.86 / 918 / 6.8, 6 a side 0.74-0.76 / 961 / 6.5-6.7, 8 a side
// 0.73-0.76 / 1020 / 7.3. BBHOST_LIGHT_GRID=<N> sets N tiles a side (=1 or
// =window one tile, =quarter 4).
std::uint32_t g_tiles_per_axis = 4;
// The tile is the GraphicsManager's own, written once by its constructor
// (sub_26958d0: mov dword [rbx+0x78a8], 128 at 0x2695cb6 and [rbx+0x78ac] at
// 0x2695cc0 - Kyo's "REAL LightGrid 1080p" puts 1920 and 1080 there) and
// never by the game again, so bbhost sets it the same way: the constructor's
// immediates at install, the live manager's two words only when the window's
// or display's size changes (light_grid_window, told by the window on a
// resize or fullscreen toggle). Nothing for it runs in a frame.
constexpr std::uint64_t kTileCtorW = 0x2695cb6, kTileCtorH = 0x2695cc0;  // 10 bytes each, the imm32 at +6
constexpr std::uint8_t kTileCtorWas[2][10] = {{0xc7, 0x83, 0xa8, 0x78, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00},
                                              {0xc7, 0x83, 0xac, 0x78, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00}};
GuestMemory g_tile_mem{};
std::uint64_t g_tile_ctor[2] = {0, 0};  // the immediates' addresses, slid (0: not ours to write)
std::mutex g_tile_mu;
bool g_task_steal = false;                          // BBHOST_TASK_TUNE=s (task_tune_install, task_steal_tick, below)

// The tile for a frame of fw x fh (the window, the display in fullscreen, or
// the render size headless): the frame over the tiles a side, or a fixed
// tile (BBHOST_LIGHT_GRID=WxH). Written into the constructor's immediates and
// the live manager, under g_tile_mu; nothing happens when they hold it already.
void light_grid_apply(std::uint32_t fw, std::uint32_t fh) {
    if (!fw || !fh) return;
    const std::uint32_t tiles = g_tiles_per_axis;
    std::uint32_t tw = std::clamp<std::uint32_t>((fw + tiles - 1) / tiles, 128u, 16384u);
    std::uint32_t th = std::clamp<std::uint32_t>((fh + tiles - 1) / tiles, 128u, 16384u);
    if (g_tile_fixed[0]) {
        tw = std::clamp<std::uint32_t>(g_tile_fixed[0], 16u, 16384u);
        th = std::clamp<std::uint32_t>(g_tile_fixed[1], 16u, 16384u);
    }
    const std::uint32_t want[2] = {tw, th};
    for (int k = 0; k < 2; ++k) {
        if (!g_tile_ctor[k]) continue;
        auto* imm = static_cast<std::uint32_t*>(guest_ptr(g_tile_mem, g_tile_ctor[k] + 6));
        if (*imm == want[k]) continue;
        const std::uint64_t page = g_tile_ctor[k] & ~0xfffull;
        if (!guest_protect_rwx(&g_tile_mem, page, 0x2000)) continue;
        std::memcpy(imm, &want[k], 4);
        guest_protect_rx(&g_tile_mem, page, 0x2000);
    }
    const std::uint64_t gm = g_graphics_manager ? *g_graphics_manager : 0;
    if (!gm) return;  // not made yet: its constructor writes the size
    auto* tile = reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(gm + kLightTileW));
    if (tile[0] == tw && tile[1] == th) return;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 8) host_log("graphics: light grid tile %ux%u -> %ux%u (a %ux%u frame)", tile[0], tile[1], tw, th, fw, fh);
    tile[0] = tw;
    tile[1] = th;
    static_assert(kLightTileH == kLightTileW + 4, "the tile's height follows its width");
}

// The presented image's size (light_grid_window) and the render size the tile
// was last made for, under g_tile_mu.
std::uint32_t g_window_px[2] = {0, 0};
std::uint32_t g_render_px[2] = {0, 0};

// The tile for the larger of the presented image and the render size on each
// side: sized to the window alone, a window a little smaller than the render
// (a 1910x1015 one over 1920x1080, a Linux player's) made the grid 2x2 tiles,
// every light drawn once a tile it touches and each tile's lights a pass of
// their own - ~100 render passes a frame against ~25.
void light_grid_refresh_locked() {
    g_render_px[0] = g_res_words ? g_res_words[0] : 0;
    g_render_px[1] = g_res_words ? g_res_words[1] : 0;
    light_grid_apply(std::max(g_window_px[0], g_render_px[0]), std::max(g_window_px[1], g_render_px[1]));
}

// The window's pixel size, now and whenever it changes (host_window_on_pixels).
void light_grid_window(int w, int h) {
    std::lock_guard<std::mutex> lock(g_tile_mu);
    g_window_px[0] = static_cast<std::uint32_t>(std::max(w, 0));
    g_window_px[1] = static_cast<std::uint32_t>(std::max(h, 0));
    light_grid_refresh_locked();
}

// A view's check that the render size is the one the tile was made for (a
// live resolution change, PC Graphics' Resolution): two words a view.
void light_grid_render_size_tick() {
    if (!g_res_words || (g_res_words[0] == g_render_px[0] && g_res_words[1] == g_render_px[1])) return;
    std::lock_guard<std::mutex> lock(g_tile_mu);
    light_grid_refresh_locked();
}

// Model detail. Every mesh draw (sub_216f9d0, 0x216fc09) picks its level as
// clamp(bias + the model's own distance level, 0, 2), the bias a global
// (0x5987d20) that sub_2141d60 copies every frame from the render manager's
// settings (*(RendMan 0x5940298 + 0x10) + 0x530, the debug menu's "LodLvBias")
// - five other readers take it from there too. So the bias is written at its
// source, where all of them agree: the Model detail setting (Full -2,
// Normal 0, Low 1, Lowest 2), or BBHOST_LOD_BIAS=N for an experiment.
// Measured (2026-10-03, pinned, the frozen seed at the Cathedral Ward
// doors): Lowest -7.5% GPU and 3.5% fewer draws, the relief over the doors
// melted to blobs; Full no measurable cost there.
constexpr std::uint64_t kRendManSlot = 0x5940298;
const std::uint64_t* g_rend_man = nullptr;

// Written when the setting changes (or the render manager's settings are
// made again), not held every frame: the debug menu's LodLvBias changes it in
// between and keeps it.
void lod_bias_tick() {
    if (!g_rend_man || !*g_rend_man) return;
    const int want = g_lod_bias.load(std::memory_order_relaxed);
    const std::uint64_t settings = *reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(*g_rend_man + 0x10));
    if (!settings) return;
    static std::uint64_t applied_at = 0;
    static int applied = 0;
    if (settings == applied_at && want == applied) return;
    applied_at = settings;
    applied = want;
    auto* bias = reinterpret_cast<std::int32_t*>(static_cast<std::uintptr_t>(settings + 0x530));
    if (*bias == want) return;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 8) host_log("graphics: model detail bias %d -> %d", *bias, want);
    *bias = want;
}

void lod_bias_install(ElfImage* image) {
    g_rend_man = static_cast<const std::uint64_t*>(guest_ptr(image->mem, image->mem.slide + (kRendManSlot - kPreferredGuestSlide)));
    if (const char* e = std::getenv("BBHOST_LOD_BIAS"); e && *e) {
        g_lod_bias.store(std::clamp(std::atoi(e), -2, 2));
        g_lod_bias_env = true;
        host_log("graphics: model detail bias %d (BBHOST_LOD_BIAS)", g_lod_bias.load());
    }
}

// BBHOST_TASK_TUNE (experiment, from the community "Performance Patch v3"):
// "g" sets the render job granularities in .data - EntityDrawTask 4 -> 1
// (0x5522750), InstanceTask 2 -> 1 (0x553acd4), Matrix 5 -> 30 (0x553ac90) -
// and "s" turns on the graphics manager's render task stealing (+0x78bd).
void task_tune_install(ElfImage* image) {
    const char* e = std::getenv("BBHOST_TASK_TUNE");
    if (!e || !*e) return;
    const auto at = [&](std::uint64_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    if (std::strchr(e, 'g')) {
        struct W { std::uint64_t bn; std::uint32_t was, now; } const ws[] = {{0x5522750, 4, 1}, {0x553acd4, 2, 1}, {0x553ac90, 5, 30}};
        for (const W& w : ws) {
            auto* p = static_cast<std::uint32_t*>(guest_ptr(image->mem, at(w.bn)));
            if (*p != w.was || !guest_protect_rw(&image->mem, at(w.bn) & ~0xfffull, 0x1000)) continue;
            *p = w.now;
            host_log("graphics: task granularity 0x%llx %u -> %u (BBHOST_TASK_TUNE)", static_cast<unsigned long long>(w.bn), w.was, w.now);
        }
    }
    g_task_steal = std::strchr(e, 's') != nullptr;
}

// BBHOST_TASK_TUNE=s: the render task stealing held on in the live manager
// each view (the experiment's switch rode on the light grid's per-frame
// write, which is gone now that the grid is set at its source).
void task_steal_tick() {
    const std::uint64_t gm = g_graphics_manager ? *g_graphics_manager : 0;
    if (gm) *reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(gm + 0x78bd)) = 1;
}

void light_grid_install(ElfImage* image) {
    if (const char* e = std::getenv("BBHOST_LIGHT_GRID"); e && e[0] == '0') {
        host_log("graphics: light grid as the game has it (BBHOST_LIGHT_GRID=0)");
        return;
    }
    if (const char* e = std::getenv("BBHOST_LIGHT_GRID"); e && std::strcmp(e, "window") == 0) g_tiles_per_axis = 1;
    // =<N>: N tiles a side (1 is one tile the frame's size).
    if (const char* e = std::getenv("BBHOST_LIGHT_GRID"); e && *e >= '1' && *e <= '9' && !std::strchr(e, 'x')) {
        g_tiles_per_axis = static_cast<std::uint32_t>(std::clamp(std::atoi(e), 1, 32));
    }
    if (const char* e = std::getenv("BBHOST_LIGHT_GRID"); e && std::strchr(e, 'x')) {
        g_tile_fixed[0] = static_cast<std::uint32_t>(std::atoi(e));
        g_tile_fixed[1] = static_cast<std::uint32_t>(std::atoi(std::strchr(e, 'x') + 1));
    }
    const auto at = [&](std::uint64_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    // The game's 3x3, or 1x1 already: a patch list's (Performance Patch v3
    // writes the split and a 1920x1080 tile, and with it applied this
    // refused, so the tile never followed the window - Kyo's log, 2026-10-14).
    auto* div = static_cast<std::uint32_t*>(guest_ptr(image->mem, at(kLightGridDiv)));
    const bool patched = div[0] == 1 && div[1] == 1;
    if (!patched && (div[0] != 3 || div[1] != 3)) {
        host_log("graphics: refused, the light grid's job split at 0x%llx holds %ux%u, not 3x3",
                 static_cast<unsigned long long>(kLightGridDiv), div[0], div[1]);
        return;
    }
    if (!patched && !guest_protect_rw(&image->mem, at(kLightGridDiv) & ~0xfffull, 0x1000)) {
        host_log("graphics: cannot unprotect the light grid's job split");
        return;
    }
    if (!patched) {
        div[0] = 1;
        div[1] = 1;
    }
    g_res_words = static_cast<const std::uint32_t*>(guest_ptr(image->mem, at(0x55289f8)));  // res_width, res_height (below)
    g_graphics_manager = static_cast<const std::uint64_t*>(guest_ptr(image->mem, at(kGraphicsManagerSlot)));
    g_tile_mem = image->mem;
    const std::uint64_t ctor[2] = {at(kTileCtorW), at(kTileCtorH)};
    for (int k = 0; k < 2; ++k) {
        // The instruction with any immediate: a patch list may have put its own size there already.
        if (std::memcmp(guest_ptr(image->mem, ctor[k]), kTileCtorWas[k], 6) == 0) g_tile_ctor[k] = ctor[k];
    }
    if (!g_tile_ctor[0] || !g_tile_ctor[1]) {
        host_log("graphics: the GraphicsManager constructor's 128x128 tile is not at 0x%llx/0x%llx; the size is written into the "
                 "manager alone", static_cast<unsigned long long>(kTileCtorW), static_cast<unsigned long long>(kTileCtorH));
        g_tile_ctor[0] = g_tile_ctor[1] = 0;
    }
    {
        // The render size until the window says otherwise (headless, it stays).
        std::lock_guard<std::mutex> lock(g_tile_mu);
        light_grid_refresh_locked();
    }
    host_window_on_pixels(&light_grid_window);
    host_log("graphics: light grid - one job of %ux%u tiles sized to the larger of the window and the render size, set by the "
             "manager's constructor and again only when either changes (was 3x3 jobs of 128x128 tiles%s; BBHOST_LIGHT_GRID=<N>: N "
             "a side, =1 or =window one tile, =0 the game's)",
             g_tiles_per_axis, g_tiles_per_axis, patched ? "; the split was 1x1 already, a patch's" : "");
}

// The flip during which the scene's view was last drawn, plus one (0: none
// yet): the title, loading screens and movies draw none (engine_scene_view_flip).
std::atomic<std::uint64_t> g_scene_view_flip{0};

GUEST_ABI std::int64_t render_view_hook(std::uint64_t, const std::uint64_t* saved) {
    g_scene_view_flip.store(hle_video_flip_count() + 1, std::memory_order_relaxed);
    camera_tick();
    lod_bias_tick();
    if (g_task_steal) task_steal_tick();
    if (g_graphics_manager) light_grid_render_size_tick();
    drawparam_probe(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(saved[5])),
                    reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(saved[0])));
    if (g_ca_amount) {
        // What the last view copied, logged when it changes: the area's own
        // amount through the mask. 0 with the setting on means this place has
        // none, which is why a test of the setting there shows nothing.
        static float last = -1.0f;
        static std::atomic<int> logs{0};
        const float now = *reinterpret_cast<const volatile float*>(g_ca_amount);
        if (now != last && logs.fetch_add(1) < 16) {
            host_log("graphics: chromatic aberration amount %g", now);
        }
        last = now;
    }
    // SSAO and AA: a renderer's two flags, written once a generation - when
    // the setting changes, or at its first view when the file asks for other
    // than the game's own - and left alone in between, so the debug menu's
    // "Enable SSAO" and "Enable AntiAliasing" (the same bytes) keep what they
    // set. Held every view until 2026-10-14, the PC graphics settings off
    // included, which turned the debug menu's switches back on.
    const std::uint32_t gen = g_flags_gen.load(std::memory_order_acquire);
    if (g_made_any.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_made_mu);
        for (const std::uint64_t c : g_made) {
            for (GivenRenderer& e : g_given) {
                if (reinterpret_cast<std::uint64_t>(e.renderer) == c) e = GivenRenderer{};
            }
        }
        g_made.clear();
        g_made_any.store(false, std::memory_order_relaxed);
    }
    auto* r = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(saved[0]));
    if (r && gen) {
        GivenRenderer* g = nullptr;
        for (GivenRenderer& c : g_given) g = c.renderer == r ? &c : g;
        if (!g || g->gen != gen) {
            if (!g) g = &g_given[g_given_next++ % 16];
            g->renderer = r;
            g->gen = gen;
            std::uint64_t caps = 0;
            std::memcpy(&caps, r + kCapabilities, sizeof(caps));
            const std::uint8_t ssao = g_ssao.load(std::memory_order_relaxed) && (caps & kSsaoCapability);
            const std::uint8_t aa = g_aa.load(std::memory_order_relaxed) && (caps & kAaCapability);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 24) {
                host_log("graphics: renderer %p (capabilities 0x%llx): SSAO %u -> %u, AA %u -> %u", static_cast<void*>(r),
                         static_cast<unsigned long long>(caps), r[kSsaoFlag], ssao, r[kAaFlag], aa);
            }
            r[kSsaoFlag] = ssao;
            r[kAaFlag] = aa;
        }
    }
    return 0;  // and the view renders as it would
}

}  // namespace

void graphics_scene_context_made(std::uint64_t ctx) {
    std::lock_guard<std::mutex> lk(g_made_mu);
    if (g_made.size() < 64) g_made.push_back(ctx);
    g_made_any.store(true, std::memory_order_release);
}

namespace {

// A jump at a function's entry to a prologue stub (core/thunk.h) that calls
// `host` with its argument registers, then runs the `n` bytes it displaced.
// `exec`/`m`: the bytes the stub runs in place of the displaced prologue when
// it cannot run them verbatim (a rip-relative lea rewritten as a movabs of the
// slid address); null runs the prologue itself.
bool install_prologue_hook(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n,
                           void* host, std::uint64_t id = 0, const std::uint8_t* exec = nullptr, std::size_t m = 0,
                           bool keep_rax = false, bool site = false) {
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, at));
    if (n < 14 || std::memcmp(p, prologue, n) != 0) return false;
    if (!exec) {
        exec = prologue;
        m = n;
    }
#if defined(_WIN32)
    void* page = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    const bool placed = page != nullptr;
#else
    void* page = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const bool placed = page != MAP_FAILED;
#endif
    const std::uint64_t lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
    if (!placed || !guest_protect_rwx(&image->mem, lo, hi - lo)) return false;
    auto* stub = static_cast<std::uint8_t*>(page);
    thunk_emit_prologue_stub(stub, id, thunk_wrap(host), at + n, exec, m, keep_rax, site);
    const std::uint64_t dest = reinterpret_cast<std::uint64_t>(stub);
    p[0] = 0xff;
    p[1] = 0x25;  // jmp [rip+0]
    std::memset(p + 2, 0, 4);
    std::memcpy(p + 6, &dest, 8);
    std::memset(p + 14, 0xcc, n - 14);
    guest_protect_rx(&image->mem, lo, hi - lo);
    return true;
}

void live_install(ElfImage* image) {
    g_slide = image->mem.slide;
    light_grid_install(image);
    lod_bias_install(image);
    task_tune_install(image);
    const auto at = [&](std::uint64_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    const std::uint64_t ca = at(kCaCopy);
    {
        auto* v = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(kVelocityCtor)));
        g_velocity_slide = image->mem.slide;
        if (std::memcmp(v, kVelocityCtorBytes, sizeof(kVelocityCtorBytes)) == 0) {
            g_velocity_mem = image->mem;
            g_velocity_ctor = v + 6;
        } else {
            host_log("graphics: 0x%llx is not the graphics manager's motion blur switch; the velocity pass runs with blur off too",
                     static_cast<unsigned long long>(kVelocityCtor));
        }
    }
    if (std::memcmp(guest_ptr(image->mem, ca), kCaCopyBytes, sizeof(kCaCopyBytes)) != 0) {
        host_log("graphics: refused, 0x%llx is not chromatic aberration's copy; it stays on",
                 static_cast<unsigned long long>(kCaCopy));
    } else if (!install_ca_stub(image, ca)) {
        host_log("graphics: cannot place chromatic aberration's stub; it stays on");
    } else {
        g_ca_amount = static_cast<const float*>(guest_ptr(image->mem, at(kCaAmount)));
    }
    if (!install_prologue_hook(image, at(kRenderView), kRenderViewPrologue, sizeof(kRenderViewPrologue),
                               reinterpret_cast<void*>(&render_view_hook))) {
        host_log("graphics: refused, 0x%llx is not the render-view prologue (or no room for its hook); SSAO "
                 "and AA stay on",
                 static_cast<unsigned long long>(kRenderView));
    }
    if (!install_prologue_hook(image, at(kYebisRecord), kYebisRecordPrologue, sizeof(kYebisRecordPrologue),
                               reinterpret_cast<void*>(&yebis_record_hook))) {
        host_log("graphics: refused, 0x%llx is not the YEBIS record's prologue (or no room for its hook); "
                 "bloom and vignette stay as the area has them",
                 static_cast<unsigned long long>(kYebisRecord));
    } else {
        g_post_block = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(kPostBlock)));
    }
    if (!install_prologue_hook(image, at(kSsaoDraw), kSsaoDrawPrologue, sizeof(kSsaoDrawPrologue),
                               reinterpret_cast<void*>(&ssao_draw_hook))) {
        host_log("graphics: refused, 0x%llx is not the SSAO draw's prologue (or no room for its hook); "
                 "AO strength stays as the area has it",
                 static_cast<unsigned long long>(kSsaoDraw));
    }
    if (!install_jump_hook(image, at(kFogUploadJump), at(kTableUpload), reinterpret_cast<void*>(&fog_upload_hook))) {
        host_log("graphics: refused, 0x%llx is not the fog table's jump to its upload (or no room for a stub); "
                 "fog stays as the area has it",
                 static_cast<unsigned long long>(kFogUploadJump));
    }
    if (!install_prologue_hook(image, at(kShadowSetup), kShadowSetupPrologue, sizeof(kShadowSetupPrologue),
                               reinterpret_cast<void*>(&shadow_setup_hook))) {
        host_log("graphics: refused, 0x%llx is not the shadow setup's prologue (or no room for its hook); "
                 "shadow distance stays as the area has it",
                 static_cast<unsigned long long>(kShadowSetup));
    }
    host_settings_listen(&settings_changed);  // and at once with the settings as they are
    host_log("graphics: SSAO, anti-aliasing and motion blur's pass are written when they change (the debug menu's "
             "switches over the same bytes keep theirs in between); AO strength, bloom, vignette, saturation, fog and "
             "shadow distance follow their settings every view; chromatic aberration through its mask%s",
             g_ca_mask ? "" : " (not this run: its stub could not be placed)");
}

// The rest of the flag block, for probing.
//
// The constructor writes **seven** adjacent bools, not four, and it writes
// them with three different stores, so their immediates are not contiguous:
//
//   26c2532  c6 83 6e 29 00 00 01            mov byte  [rbx+0x296e], 1
//   26c2539  66 c7 83 6c 29 00 00 01 01      mov word  [rbx+0x296c], 0x0101
//   26c2542  c7 83 68 29 00 00 01 01 01 01   mov dword [rbx+0x2968], 0x01010101
//
// Reading the four bytes after 0x26c254b as more flags is reading the next
// instruction (`48 c7 83 77 29 ...`), which is what a first pass at this did:
// three probe runs refused with "holds 72/199/131, not 1" and quietly measured
// stock against stock. So the probe is addressed by **field offset** and the
// mapping lives here, where it can be checked against the disassembly above.
struct Flag {
    std::uint16_t field;   // +0x29xx on the object
    std::uint64_t at;      // the immediate byte in the constructor
    const char* what;      // "" when nothing names it
};
const Flag kFlags[] = {
    {0x2968, 0x26c2548, "SSAO"},
    {0x2969, 0x26c2549, "motion blur"},
    {0x296a, 0x26c254a, "anti-aliasing"},
    {0x296b, 0x26c254b, ""},
    {0x296c, 0x26c2540, ""},
    {0x296d, 0x26c2541, ""},
    {0x296e, 0x26c2538, ""},
};

// BBHOST_GFX_FLAG=0x296c zeroes one of them. It refuses a field that is not in
// the table and any byte that is not currently 1.
void probe_patch(ElfImage* image) {
    const char* env = std::getenv("BBHOST_GFX_FLAG");
    if (!env || !*env) {
        return;
    }
    const std::uint64_t field = std::strtoull(env, nullptr, 0);
    const Flag* f = nullptr;
    for (const Flag& c : kFlags) {
        if (c.field == field) {
            f = &c;
            break;
        }
    }
    if (!f) {
        host_log("graphics: BBHOST_GFX_FLAG=%s is not one of +0x2968..+0x296e", env);
        return;
    }
    const std::uint64_t at = image->mem.slide + (f->at - kPreferredGuestSlide);
    auto* q = static_cast<std::uint8_t*>(guest_ptr(image->mem, at));
    if (*q != 1) {
        host_log("graphics: refused, the immediate for +0x%llx (0x%llx) holds %u, not 1",
                 static_cast<unsigned long long>(f->field),
                 static_cast<unsigned long long>(f->at), *q);
        return;
    }
    if (!guest_protect_rwx(&image->mem, at & ~0xfffull, 0x1000)) {
        host_log("graphics: cannot unprotect the flag block");
        return;
    }
    *q = 0;
    guest_protect_rx(&image->mem, at & ~0xfffull, 0x1000);
    host_log("graphics: probe - +0x%llx (%s, immediate at 0x%llx) forced to 0",
             static_cast<unsigned long long>(f->field), f->what[0] ? f->what : "unnamed",
             static_cast<unsigned long long>(f->at));
}

// The render resolution, which is two int32s in .data.
//
//   0x55289f8  res_width   0x780 (1920)
//   0x55289fc  res_height  0x438 (1080)
//
// Fifty-odd sites read them, nearly every one as `lea rax, [rip+res_width];
// mov eax, [rax]`. The game writes them once, at start: SprjInitStuff
// (0x241786e) stores back what a SystemProperties lookup returns - its
// default is the current value - and derives the UI scale and the window
// size (0x59404d8/dc) from them, so a value written here before it runs is
// the one the game keeps (two setters, sub_2417700 and sub_219e850, have no
// callers). So they are a size the build put in .data rather than a mode the
// game negotiates. The
// shadPS4 resolution patches rewrite each of those reads into a `mov eax,
// imm32`, which is how they give different passes different sizes (their
// "Optimal 1080p" runs the globals at 160x90 and pins the main renders to
// 1920x1080). Changing the two words instead scales every pass together,
// which is one 8-byte data write and no code patched at all.
//
// Chosen on the PC Graphics row, or by BBHOST_RES=WxH for an A/B run.
constexpr std::uint64_t kResWidth = 0x55289f8, kResHeight = 0x55289fc;

// The Scaleform stage stays 1920x1080 whatever the render size: SprjInitStuff
// derives the UI scale (0x59404d0/d4 = res/1920, res/1080) from the two words
// above and the menus draw through it. Nine reads of the words are not render
// passes but stage-space users, and they go wrong the moment the words change:
//
//   sub_1a439b0  0x1a44357/65  floating plates over other players: clip xy
//                              times (res_width, res_height) is the plate's
//                              stage position, so at 720p a cooperator's name
//                              and HP landed two thirds of the way to the
//                              top-left corner
//   sub_1a448f0  0x1a44c55/63  the same projection for the enemy gauges
//   sub_1a44cf0  0x1a452c7/d5  and for the lock-on marker
//   sub_1ffb2a0  0x1ffd491/d1  "is the plate on screen": 0 <= x < res_width
//   sub_19e7ce0  0x19e83af     a cursor placed relative to res_width / 2
//   sub_1a54d80  0x1a54df6/e1c "F20_open_eneny_HP": the enemy gauge is shown
//                0x1a55611/28  only when 0 <= its stage position < res, and
//                              placed in a res-sized rect
//   sub_1ab3320  0x1ab3372/98  "F20_LockCursor": the lock-on marker the same,
//                              its position written by sub_1a44cf0 above in
//                              stage space - so at 1280x720 a marker past
//                              x 1280 or y 720 was hidden
//
// Each is `lea rax, [rip+res_width]; mov eax, [rax]` (9 bytes) - one, the
// lock cursor's width, the same through rcx - and becomes `mov eax, 1920`
// (or 1080; `mov ecx` for that one) and a 4-byte nop. The shadPS4 community
// patches pin the first nine (facts from Bloodborne.xml, "proper lock-on/
// enemy/ally hp bar coordinates"), which is how they were confirmed; the
// last six turned up reading that list's "Optimal 1080p" (2026-10-03,
// docs/modding.md), which does not pin them either.
struct StageSite {
    std::uint64_t at;
    std::uint64_t reads;  // res_width or res_height
    std::uint32_t value;
    bool rcx = false;     // lea rcx / mov ecx, [rcx]
};
constexpr StageSite kStageSites[] = {
    {0x19e83af, kResWidth, 1920},  {0x1ffd491, kResWidth, 1920},  {0x1ffd4d1, kResHeight, 1080},
    {0x1a44357, kResWidth, 1920},  {0x1a44365, kResHeight, 1080}, {0x1a44c55, kResWidth, 1920},
    {0x1a44c63, kResHeight, 1080}, {0x1a452c7, kResWidth, 1920},  {0x1a452d5, kResHeight, 1080},
    {0x1a54df6, kResWidth, 1920},  {0x1a54e1c, kResHeight, 1080}, {0x1a55611, kResHeight, 1080},
    {0x1a55628, kResWidth, 1920},  {0x1ab3372, kResWidth, 1920, true}, {0x1ab3398, kResHeight, 1080},
};

void ui_stage_pin(ElfImage* image) {
    int done = 0;
    for (const StageSite& site : kStageSites) {
        const std::uint64_t at = image->mem.slide + (site.at - kPreferredGuestSlide);
        auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, at));
        // lea rax, [rip+disp32]; mov eax, [rax]
        const std::int32_t disp = static_cast<std::int32_t>(static_cast<std::int64_t>(site.reads) -
                                                            static_cast<std::int64_t>(site.at + 7));
        std::uint8_t want[9] = {0x48, 0x8d, 0x05, 0, 0, 0, 0, 0x8b, 0x00};
        if (site.rcx) {
            want[2] = 0x0d;
            want[8] = 0x09;
        }
        std::memcpy(want + 3, &disp, 4);
        if (std::memcmp(p, want, 9) != 0) {
            host_log("graphics: stage site 0x%llx is not the read expected, left alone",
                     static_cast<unsigned long long>(site.at));
            continue;
        }
        const std::uint64_t lo = at & ~0xfffull, hi = (at + 9 + 0xfff) & ~0xfffull;
        if (!guest_protect_rwx(&image->mem, lo, hi - lo)) {
            host_log("graphics: cannot unprotect the stage site 0x%llx", static_cast<unsigned long long>(site.at));
            continue;
        }
        std::uint8_t code[9] = {0xb8, 0, 0, 0, 0, 0x0f, 0x1f, 0x40, 0x00};  // mov eax, imm32; nop4
        if (site.rcx) code[0] = 0xb9;                                         // mov ecx, imm32
        std::memcpy(p, code, 9);
        std::memcpy(p + 1, &site.value, 4);
        guest_protect_rx(&image->mem, lo, hi - lo);
        ++done;
    }
    host_log("graphics: UI stage pinned to 1920x1080 at %d of %d sites", done,
             static_cast<int>(sizeof(kStageSites) / sizeof(kStageSites[0])));
}

void resolution_patch(ElfImage* image, bool live) {
    int w = 0, h = 0;
    if (!host_opt_resolution(&w, &h) || (w == 1920 && h == 1080)) {
        // What it shipped with: nothing to write - but the words change
        // under a live resolution change, so the stage readers are pinned.
        if (live) ui_stage_pin(image);
        return;
    }
    if (w < 256 || h < 144 || w > 7680 || h > 4320) {
        host_log("graphics: %dx%d is outside the range this will write", w, h);
        return;
    }
    const std::uint64_t at = image->mem.slide + (kResWidth - kPreferredGuestSlide);
    auto* q = static_cast<std::uint32_t*>(guest_ptr(image->mem, at));
    if (q[0] != 1920 || q[1] != 1080) {
        host_log("graphics: refused, 0x%llx holds %ux%u, not 1920x1080",
                 static_cast<unsigned long long>(kResWidth), q[0], q[1]);
        return;
    }
    if (!guest_protect_rw(&image->mem, at & ~0xfffull, 0x1000)) {
        host_log("graphics: cannot unprotect the resolution");
        return;
    }
    q[0] = static_cast<std::uint32_t>(w);
    q[1] = static_cast<std::uint32_t>(h);
    host_log("graphics: render resolution %dx%d (was 1920x1080)", w, h);
    ui_stage_pin(image);
}

}  // namespace

bool engine_prologue_hook(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n, void* host,
                          std::uint64_t id) {
    return install_prologue_hook(image, at, prologue, n, host, id);
}

bool engine_prologue_hook_exec(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n,
                               const std::uint8_t* exec, std::size_t m, void* host, std::uint64_t id) {
    return install_prologue_hook(image, at, prologue, n, host, id, exec, m);
}

bool engine_site_hook(ElfImage* image, std::uint64_t at, const std::uint8_t* bytes, std::size_t n, void* host,
                      std::uint64_t id) {
    return install_prologue_hook(image, at, bytes, n, host, id, nullptr, 0, false, true);
}

bool engine_call_site_hook(ElfImage* image, std::uint64_t at, std::uint64_t target, void* host) {
    return install_jump_hook(image, at, target, host);
}

bool engine_prologue_hook_variadic(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n,
                                   void* host, std::uint64_t id) {
    return install_prologue_hook(image, at, prologue, n, host, id, nullptr, 0, true);
}

std::uint64_t engine_scene_view_flip() { return g_scene_view_flip.load(std::memory_order_relaxed); }

std::string graphics_glare_report() {
    static std::uint64_t last_views = 0, last_skipped = 0;
    const std::uint64_t views = g_glare_views.load(std::memory_order_relaxed);
    const std::uint64_t skipped = g_glare_skipped.load(std::memory_order_relaxed);
    if (views == last_views) return {};
    char buf[224];
    std::snprintf(buf, sizeof(buf),
                  "glare: bloom %d/10; %llu YEBIS views, %llu with a glare luminance above 0, %llu with none (YEBIS's "
                  "no-glare path: its glare passes skipped)",
                  static_cast<int>(g_bloom.load(std::memory_order_relaxed) * 10.0f + 0.5f),
                  static_cast<unsigned long long>(views - last_views),
                  static_cast<unsigned long long>((views - last_views) - (skipped - last_skipped)),
                  static_cast<unsigned long long>(skipped - last_skipped));
    last_views = views;
    last_skipped = skipped;
    return buf;
}

void graphics_patch_install(ElfImage* image) {
    resolution_patch(image, live_resolution_install(image));
    menu_memory_install(image);
    frame_rate_install(image);
    camera_install(image);
    fmod_probe_install(image);
    fmod_ring_install(image);
    sf_text_layout_install(image);
    sf_heap_probe_install(image);
    probe_patch(image);
    live_install(image);
}
