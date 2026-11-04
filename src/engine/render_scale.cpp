#include "engine/render_scale.h"

#include "core/elf.h"
#include "core/memory.h"
#include "engine/addr.h"
#include "engine/patch_manifest.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/upscaler.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

// Addresses are Binary Ninja's (0x400000 base).
//
// Kyo's decoupled render resolution patches (his shadPS4 list, "Decoupled
// Render Res", 106 sites) draw the scene contexts at one size while YEBIS and
// the UI stay at another. Most of their sites are the res words at 160x90 and
// the reads that must not follow them pinned back to the UI's size - the
// window, the Scaleform stage, the plates - which bbhost leaves alone: the res
// words stay the Resolution, so every size and shape the Resolution setting
// has keeps its UI, plates and cameras as they are (engine/live_resolution.cpp
// apply_aspect, engine/graphics_patch.cpp ui_stage_pin). What is left is the
// scene contexts' side, here:
//
//   - the size every scene context is made at. The GX device's output entry
//     (device +0x90, 0xb0 bytes each) holds the render size at +0x80/+0x84,
//     and seven places build or rebuild a context from it: the graphics
//     manager's build (sub_26979b0), its resize listener (sub_269c200: the
//     size the broadcast gave), a registered context's resize slot
//     (sub_26c6540, twice: its own entry's size, or the broadcast's for the
//     manager's own entry), and the contexts made on demand (sub_26f8c70,
//     sub_26f9330, sub_26f9940). Each read becomes the render size; the
//     resize slot's two tests that would take the broadcast's size instead
//     always go to the entry's (now pinned) read;
//   - the colour target a context is given. Initialize (sub_26c39f0) keeps
//     the target its description names - the manager's names the output
//     entry's DefRenderTarget, the display buffer - only when it shares a
//     side with the context's size, and otherwise makes one of its own at
//     that size. Kept always (0x26c3af3), the display buffer stays the
//     scene's back buffer: YEBIS makes its whole chain at its back buffer's
//     size, so it stays at the output size, and its copy of the scene
//     (0b0acf50+ccbf44a6) is where the picture goes from the render size to
//     the output's. That is why the display buffers must be exactly the
//     output size here (engine/live_resolution.cpp), as they are on the
//     console. The depth is still the context's own at the render size;
//   - the flip pass's full-screen viewport (sub_25d5f80, 0x25d818d): the
//     manager's size (+0x1e0, the render size now) becomes the output entry's;
//   - one YEBIS-side pass into the scene's main target set (sub_1258d30,
//     0x1258f2c), whose screen-size constants (w, h, 1/w, 1/h) came from the
//     set (the render size) but whose target is the back buffer: the output
//     size.
//
// The rest of Kyo's list - the asserts and DL_PANIC let through, Performance
// Patch v3's 80x45 secondary output entries (0x2594aa6), the glow buffer at
// 80x45 (0x26c45ba: never smaller than its pass here, or the lit world went
// black; it stays the context's size) - is either Performance Patch v3's,
// which bbhost carries, or not the decoupling's.

namespace {

constexpr std::uint64_t kManagerBuild = 0x2697aea;  // the first pinned read: whose manifest it is

enum Value : std::uint8_t { kNone, kRenderW, kRenderH, kOutputW, kOutputH };

// One site: the eboot's bytes, the bytes written, and the 32-bit values in
// them (an offset into the bytes for each, 0xff none).
struct PinSite {
    const char* id;
    std::uint64_t at;
    std::vector<std::uint8_t> game, ours;
    std::uint8_t off[2] = {0xff, 0xff};
    Value value[2] = {kNone, kNone};
};

// mov r32, imm32 with nops over a read of the output entry's size.
PinSite mov_imm(const char* id, std::uint64_t at, std::vector<std::uint8_t> game, std::uint8_t opcode, Value v) {
    PinSite s{id, at, std::move(game), {}};
    s.ours.assign(s.game.size(), 0x90);
    s.ours[0] = opcode;
    s.off[0] = 1;
    s.value[0] = v;
    return s;
}

std::vector<PinSite> pin_sites() {
    std::vector<PinSite> v;
    // The graphics manager's build: ecx, edx = the output entry's size.
    v.push_back(mov_imm("manager-build-w", kManagerBuild, {0x8b, 0x88, 0x80, 0x00, 0x00, 0x00}, 0xb9, kRenderW));
    v.push_back(mov_imm("manager-build-h", 0x2697af0, {0x8b, 0x90, 0x84, 0x00, 0x00, 0x00}, 0xba, kRenderH));
    // A registered context's resize slot (sub_26c6540), its two paths: the
    // broadcast's size for the manager's own entry becomes the entry's.
    v.push_back({"resize-a-entry", 0x26c65f6, {0x74, 0x18}, {0xeb, 0x18}});  // je -> jmp
    v.push_back(mov_imm("resize-a-w", 0x26c6624, {0x8b, 0x94, 0x08, 0x80, 0x00, 0x00, 0x00}, 0xba, kRenderW));
    v.push_back(mov_imm("resize-a-h", 0x26c662e, {0x8b, 0x84, 0x08, 0x84, 0x00, 0x00, 0x00}, 0xb8, kRenderH));
    v.push_back({"resize-b-entry", 0x26c6735, {0x74, 0x16}, {0xeb, 0x16}});
    v.push_back(mov_imm("resize-b-w", 0x26c6760, {0x8b, 0x94, 0x08, 0x80, 0x00, 0x00, 0x00}, 0xba, kRenderW));
    v.push_back(mov_imm("resize-b-h", 0x26c676a, {0x8b, 0x84, 0x08, 0x84, 0x00, 0x00, 0x00}, 0xb8, kRenderH));
    // The contexts made on demand.
    v.push_back(mov_imm("demand-a-w", 0x26f8ce1, {0x8b, 0x88, 0x80, 0x00, 0x00, 0x00}, 0xb9, kRenderW));
    v.push_back(mov_imm("demand-a-h", 0x26f8ce7, {0x8b, 0xb0, 0x84, 0x00, 0x00, 0x00}, 0xbe, kRenderH));
    v.push_back(mov_imm("demand-b-w", 0x26f93d8, {0x8b, 0x9c, 0x3e, 0x80, 0x00, 0x00, 0x00}, 0xbb, kRenderW));
    v.push_back(mov_imm("demand-b-h", 0x26f93df, {0x8b, 0xb4, 0x3e, 0x84, 0x00, 0x00, 0x00}, 0xbe, kRenderH));
    v.push_back(mov_imm("demand-c-w", 0x26f99db, {0x8b, 0xb4, 0x08, 0x80, 0x00, 0x00, 0x00}, 0xbe, kRenderW));
    v.push_back(mov_imm("demand-c-h", 0x26f99e2, {0x8b, 0x8c, 0x08, 0x84, 0x00, 0x00, 0x00}, 0xb9, kRenderH));
    // The manager's resize listener: its context's description takes the
    // broadcast's size (r15d, [rsp+0x1c]) and 0x800 x 0x800; written as the
    // render size and the same 0x800s through edx, which the call after
    // loads again (mov rdx, r13).
    {
        PinSite s{"listener",
                  0x269c7d5,
                  {0x44, 0x89, 0x7c, 0x24, 0x20, 0x8b, 0x54, 0x24, 0x1c, 0x89, 0x54, 0x24, 0x24, 0xc7, 0x44,
                   0x24, 0x28, 0x00, 0x08, 0x00, 0x00, 0xc7, 0x44, 0x24, 0x2c, 0x00, 0x08, 0x00, 0x00},
                  {0xc7, 0x44, 0x24, 0x20, 0, 0, 0, 0, 0xc7, 0x44, 0x24, 0x24, 0, 0, 0, 0,  // mov dword [rsp+0x20/0x24], imm32
                   0xba, 0x00, 0x08, 0x00, 0x00, 0x89, 0x54, 0x24, 0x28, 0x89, 0x54, 0x24, 0x2c}};  // mov edx, 0x800; two stores
        s.off[0] = 4;
        s.value[0] = kRenderW;
        s.off[1] = 12;
        s.value[1] = kRenderH;
        v.push_back(s);
    }
    // Initialize keeps the colour target its description names at any size.
    v.push_back({"keep-colour", 0x26c3af3, {0x74, 0x36}, {0xeb, 0x36}});
    // The flip pass's viewport: the output entry's size (entry 0) for the
    // manager's (+0x1e0/+0x1e4), the same 45 bytes rearranged (the vmovss's
    // displacement moved with it; mov esi/edx, 0 as xor).
    v.push_back({"flip-viewport", 0x25d818d,
                 {0x48, 0x8d, 0x05, 0x44, 0x8c, 0x36, 0x03, 0x48, 0x8b, 0x00, 0x8b, 0x88, 0xe0, 0x01, 0x00, 0x00, 0x44, 0x8b, 0x80, 0xe4, 0x01, 0x00, 0x00,
                  0xc5, 0xfa, 0x10, 0x05, 0xe8, 0x15, 0x75, 0x02, 0xbe, 0x00, 0x00, 0x00, 0x00, 0xba, 0x00, 0x00, 0x00, 0x00, 0xc5, 0xf0, 0x57, 0xc9},
                 {0x48, 0x8b, 0x05, 0x34, 0x85, 0x36, 0x03, 0x48, 0x8b, 0x80, 0x90, 0x00, 0x00, 0x00, 0x8b, 0x88, 0x80, 0x00, 0x00, 0x00, 0x44, 0x8b, 0x80,
                  0x84, 0x00, 0x00, 0x00, 0xc5, 0xfa, 0x10, 0x05, 0xe4, 0x15, 0x75, 0x02, 0x31, 0xf6, 0x31, 0xd2, 0xc5, 0xf0, 0x57, 0xc9, 0x66, 0x90}});
    // sub_1258d30's screen-size constants: the output size for its target set's.
    {
        PinSite s{"pass-constants", 0x1258f2c, {0x41, 0x8b, 0x44, 0x24, 0x50, 0x41, 0x8b, 0x4c, 0x24, 0x54}, {0xb8, 0, 0, 0, 0, 0xb9, 0, 0, 0, 0}};
        s.off[0] = 1;
        s.value[0] = kOutputW;
        s.off[1] = 6;
        s.value[1] = kOutputH;
        v.push_back(s);
    }
    return v;
}

std::uint32_t value_of(Value v, const RenderScale& rs) {
    switch (v) {
        case kRenderW: return rs.render_w;
        case kRenderH: return rs.render_h;
        case kOutputW: return rs.output_w;
        case kOutputH: return rs.output_h;
        default: return 0;
    }
}

int quality_asked() {
    if (const char* e = std::getenv("BBHOST_UPSCALE_QUALITY"); e && *e) {
        const std::string v = e;
        if (v == "native") return 0;
        if (v == "quality") return 1;
        if (v == "balanced") return 2;
        if (v == "performance") return 3;
        return std::clamp(std::atoi(e), 0, 3);
    }
    return std::clamp(host_settings_ref().upscale_quality, 0, 3);
}

bool upscaler_asked() {
    if (const char* e = std::getenv("BBHOST_UPSCALE"); e && *e)
        return std::strcmp(e, "dlss") == 0 || std::strcmp(e, "fsr") == 0 || std::strcmp(e, "fsr3") == 0;
    return host_settings_ref().upscaler != 0;
}

ElfImage* g_image = nullptr;
RenderScale g_now;                    // the sizes in the code (the plan's, or the output's after a fallback)
std::atomic<bool> g_scaled{false};    // the 3D is at the render size (render_scale_on)

bool write_values(const RenderScale& rs) {
    if (!g_image) return false;
    for (const PinSite& s : pin_sites()) {
        for (int k = 0; k < 2; ++k) {
            if (s.value[k] == kNone) continue;
            const std::uint64_t at = g_image->mem.slide + (s.at - kPreferredGuestSlide) + s.off[k];
            const std::uint64_t lo = at & ~0xfffull, hi = (at + 4 + 0xfff) & ~0xfffull;
            if (!guest_protect_rwx(&g_image->mem, lo, hi - lo)) return false;
            const std::uint32_t value = value_of(s.value[k], rs);
            std::memcpy(guest_ptr(g_image->mem, at), &value, 4);
            guest_protect_rx(&g_image->mem, lo, hi - lo);
        }
    }
    return true;
}

}  // namespace

void render_scale_size(int quality, std::uint32_t output_w, std::uint32_t output_h, std::uint32_t* render_w, std::uint32_t* render_h) {
    static const double kRatio[4] = {1.0, 1.5, 1.7, 2.0};
    const double r = kRatio[std::clamp(quality, 0, 3)];
    const auto down4 = [&](std::uint32_t o) {
        if (r == 1.0) return o;
        const auto v = static_cast<std::uint32_t>(std::floor(static_cast<double>(o) / r / 4.0 + 1e-9)) * 4;
        return std::max<std::uint32_t>(v, 4);
    };
    *render_w = down4(output_w);
    *render_h = down4(output_h);
}

bool render_scale_plan(RenderScale* out) {
    static const RenderScale plan = [] {
        RenderScale rs;
        if (const char* e = std::getenv("BBHOST_RENDER_SCALE"); e && e[0] == '0') return rs;
        // The display buffers are made at the output size by the live
        // resolution's GX init hook, which also checks that an upscaler runs.
        if (const char* e = std::getenv("BBHOST_LIVE_RESOLUTION"); e && e[0] == '0') return rs;
        const int q = quality_asked();
        if (!q || !upscaler_asked()) return rs;
        int w = 1920, h = 1080;
        if (!host_opt_resolution(&w, &h)) w = 1920, h = 1080;
        rs.output_w = static_cast<std::uint32_t>(w);
        rs.output_h = static_cast<std::uint32_t>(h);
        rs.quality = q;
        render_scale_size(q, rs.output_w, rs.output_h, &rs.render_w, &rs.render_h);
        return rs;
    }();
    if (out) *out = plan;
    return plan.quality > 0;
}

std::vector<patch_file::Site> render_scale_sites(const RenderScale& rs) {
    std::vector<patch_file::Site> out;
    for (const PinSite& s : pin_sites()) {
        std::vector<std::uint8_t> bytes = s.ours;
        for (int k = 0; k < 2; ++k) {
            if (s.value[k] == kNone) continue;
            const std::uint32_t value = value_of(s.value[k], rs);
            std::memcpy(bytes.data() + s.off[k], &value, 4);
        }
        out.push_back(patch_file::Site{s.id, s.at, s.game, bytes, true, {}});
    }
    return out;
}

bool render_scale_applied() {
    const char* owner = patch_manifest_owner(kManagerBuild, 1);
    return owner && std::strcmp(owner, "render-scale") == 0;
}

void render_scale_gx_init(ElfImage* image) {
    RenderScale rs;
    if (!render_scale_plan(&rs) || !render_scale_applied()) return;
    g_image = image;
    g_now = rs;
    if (!upscaler_method()) {
        // No upscaler runs here: the plain picture at the full size, not a
        // stretch of the render size.
        RenderScale full = rs;
        full.render_w = rs.output_w;
        full.render_h = rs.output_h;
        if (write_values(full)) g_now = full;
        host_log("render scale: no upscaler runs here (%s); the 3D is drawn at the full %ux%u", upscaler_caps().loaded ? "its method is not available" : "no bridge",
                 rs.output_w, rs.output_h);
        return;
    }
    g_scaled.store(true);
    host_log("render scale: the 3D is drawn at %ux%u and upscaled to %ux%u (Upscaling quality %d); YEBIS, the HUD and the menus at %ux%u",
             rs.render_w, rs.render_h, rs.output_w, rs.output_h, rs.quality, rs.output_w, rs.output_h);
}

bool render_scale_on(RenderScale* out) {
    if (!g_scaled.load(std::memory_order_acquire)) return false;
    if (out) *out = g_now;
    return true;
}
