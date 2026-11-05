#include "engine/render_scale.h"

#include "core/elf.h"
#include "core/memory.h"
#include "engine/addr.h"
#include "core/thunk.h"
#include "engine/graphics_patch.h"
#include "engine/live_resolution.h"
#include "engine/patch_manifest.h"
#include "hle/modules.h"
#include "host/options.h"
#include "host/scene_look.h"
#include "host/settings.h"
#include "host/upscaler.h"
#include "host/window.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

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
//   - the colour target the graphics manager's scene is given. Initialize
//     (sub_26c39f0) keeps the target its description names - DefRenderTarget,
//     the display buffer - only when it matches the context's size, and
//     otherwise makes one of its own at that size; YEBIS makes its whole
//     chain at that back buffer's size. Kyo's patch keeps the display buffer
//     at any size (its display buffers are the output size). bbhost's are the
//     largest a live change can reach, so when they are not the output size
//     the scene is given an output-sized colour target of ours (made as the
//     target sets make theirs: a 0x80-byte object, vtable 0x57b6940,
//     sub_25a32f0), handed in at its two builds (calls at 0x2697b75 and
//     0x269c812), and Initialize's colour test (0x26c3af0) keeps the one
//     target this file names at any size - every other the game's way. Its
//     copy into the display buffer is the flip pass's, as for any picture of
//     a size other than the display buffers'. The depth is the context's own
//     at the render size;
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
std::uint64_t guest(std::uint64_t bn) { return g_image->mem.slide + (bn - kPreferredGuestSlide); }
void* guest_fn_ptr(std::uint64_t bn) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest(bn))); }
template <typename T>
T& at_(std::uint64_t a) { return *reinterpret_cast<T*>(static_cast<std::uintptr_t>(a)); }

std::mutex g_mu;                    // g_now
RenderScale g_now;                  // the sizes in the code now
std::atomic<bool> g_scaled{false};  // the 3D is at the render size (render_scale_on)
std::atomic<bool> g_ready{false};   // render_scale_gx_init ran

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


// ---- The scene's own back buffer ---------------------------------------------------

constexpr std::uint64_t kGxDeviceSlot = 0x59406c8, kGxAllocatorSlot = 0x5940680;
constexpr std::uint64_t kColourTargetVtable = 0x57b6940, kColourTargetDefaults = 0x4d21f50, kColourTargetInit = 0x25a32f0;
constexpr std::uint64_t kInitialize = 0x26c39f0, kBuildInitializeCall = 0x2697b75;
constexpr std::uint64_t kReinitialize = 0x26c58e0, kListenerReinitializeCall = 0x269c812;
constexpr std::uint64_t kColourTest = 0x26c3af0;  // cmp eax, [r15]; je/jne (5 bytes)
constexpr std::uint64_t kColourKeep = 0x26c3b2b;  // where the test goes to keep the target

std::uint64_t* g_keep = nullptr;  // in the stub's page: the colour target Initialize keeps at any size (0 none)
std::uint64_t g_back = 0;         // ours, when the scene has one
std::uint32_t g_back_w = 0, g_back_h = 0;
struct OldBack {
    std::uint64_t obj, flip;
};
std::vector<OldBack> g_old_back;  // main thread

void* vslot(std::uint64_t obj, int off) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(at_<std::uint64_t>(at_<std::uint64_t>(obj) + off)));
}

void release_ref(std::uint64_t obj) {
    std::int32_t& refs = at_<std::int32_t>(obj + 8);
    if (refs-- <= 1) hle_call_guest6(vslot(obj, 0), static_cast<std::int64_t>(obj), 0, 0, 0, 0, 0);  // slot 0: delete
}

// A colour target of w x h in DefRenderTarget's format, made as a target set
// makes its own (sub_2666420): our reference held.
std::uint64_t make_back_buffer(std::uint32_t w, std::uint32_t h) {
    const std::uint64_t dev = at_<std::uint64_t>(guest(kGxDeviceSlot));
    const std::uint64_t out = dev ? at_<std::uint64_t>(dev + 0x90) : 0;
    const std::uint64_t def_rt = out ? at_<std::uint64_t>(out + 0x68) : 0;
    const std::uint64_t alloc = at_<std::uint64_t>(guest(kGxAllocatorSlot));
    if (!def_rt || !alloc) return 0;
    const auto format = static_cast<std::uint32_t>(hle_call_guest6(vslot(def_rt, 0x38), static_cast<std::int64_t>(def_rt), 0, 0, 0, 0, 0));
    const auto obj = static_cast<std::uint64_t>(hle_call_guest6(vslot(alloc, 0x58), static_cast<std::int64_t>(alloc), 0x80, 0x10, 0, 0, 0));
    if (!obj) return 0;
    std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(obj)), 0, 0x80);
    at_<std::uint64_t>(obj) = guest(kColourTargetVtable);
    at_<float>(obj + 0x10) = 1.0f;
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(obj + 0x20)), guest_ptr(g_image->mem, guest(kColourTargetDefaults)), 16);
    at_<std::uint64_t>(obj + 0x68) = ~0ull;
    at_<std::uint32_t>(obj + 0x70) = ~0u;
    alignas(16) std::uint32_t samples[2] = {1, 0};
    GuestCallRegs r{};
    r.fn = guest_fn_ptr(kColourTargetInit);
    r.ints[0] = obj;
    r.ints[1] = w;
    r.ints[2] = h;
    r.ints[3] = format;
    r.ints[4] = 1;  // mips
    r.ints[5] = reinterpret_cast<std::uintptr_t>(samples);
    r.stack[0] = 0;  // flags
    hle_call_guest_regs(&r);
    if (!(r.ret & 0xff)) {
        release_ref(obj);  // no reference taken yet: deleted
        host_log("render scale: no %ux%u colour target could be made for the scene", w, h);
        return 0;
    }
    at_<std::int32_t>(obj + 8) += 1;
    host_log("render scale: the scene's back buffer is a %ux%u target of its own (format %u): the display buffers are another size", w, h,
             format);
    return obj;
}

// The scene's back buffer for the render and output sizes: none at the full
// size (the game's own test), DefRenderTarget when the display buffers are
// the output size, else ours (made, or kept from before). Main thread, with
// nothing recording.
void choose_back_buffer(const RenderScale& rs, std::uint32_t display_w, std::uint32_t display_h, std::uint64_t flip) {
    const bool scaled = rs.render_w != rs.output_w || rs.render_h != rs.output_h;
    const bool ours = scaled && (display_w != rs.output_w || display_h != rs.output_h);
    if (g_back && (!ours || g_back_w != rs.output_w || g_back_h != rs.output_h)) {
        g_old_back.push_back({g_back, flip});  // the scene lets go of it in its rebuild
        g_back = 0;
    }
    if (ours && !g_back) {
        g_back = make_back_buffer(rs.output_w, rs.output_h);
        g_back_w = rs.output_w;
        g_back_h = rs.output_h;
    }
    if (!g_keep) return;
    if (!scaled) {
        *g_keep = 0;
    } else if (g_back) {
        *g_keep = g_back;
    } else {
        const std::uint64_t dev = at_<std::uint64_t>(guest(kGxDeviceSlot));
        *g_keep = dev ? at_<std::uint64_t>(at_<std::uint64_t>(dev + 0x90) + 0x68) : 0;  // DefRenderTarget
    }
}

// The manager's scene's build and rebuild: our back buffer for its colour.
void build_with_back_buffer(std::uint64_t fn, const std::uint64_t* saved) {
    const std::uint64_t desc = saved[4];  // rsi
    if (g_back && desc) at_<std::uint64_t>(desc + 0x20) = g_back;
    hle_call_guest6(guest_fn_ptr(fn), static_cast<std::int64_t>(saved[5]), static_cast<std::int64_t>(saved[4]), static_cast<std::int64_t>(saved[3]),
                    static_cast<std::int64_t>(saved[2]), static_cast<std::int64_t>(saved[1]), static_cast<std::int64_t>(saved[0]));
}
GUEST_ABI std::int64_t build_call_hook(std::uint64_t, const std::uint64_t* saved) {
    build_with_back_buffer(kInitialize, saved);
    return 1;
}
GUEST_ABI std::int64_t rebuild_call_hook(std::uint64_t, const std::uint64_t* saved) {
    build_with_back_buffer(kReinitialize, saved);
    return 1;
}

// A page within a rel32 of `at`.
void* page_near(std::uint64_t at) {
#if defined(_WIN32)
    const std::uint64_t reach = 0x7ff00000ull, base = at & ~0xffffull;
    for (std::uint64_t a = base - 0x10000; a > 0x10000 && base - a < reach; a -= 0x10000)
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)) return p;
    for (std::uint64_t a = base + 0x10000; a - base < reach; a += 0x10000)
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)) return p;
    return nullptr;
#else
    void* p = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    const std::int64_t d = static_cast<std::int64_t>(reinterpret_cast<std::uint64_t>(p)) - static_cast<std::int64_t>(at);
    if (d < INT32_MIN + 0x1000ll || d > INT32_MAX - 0x1000ll) {
        munmap(p, 0x1000);
        return nullptr;
    }
    return p;
#endif
}

bool rel32(std::uint64_t from_end, std::uint64_t to, std::uint8_t* out) {
    const std::int64_t d = static_cast<std::int64_t>(to) - static_cast<std::int64_t>(from_end);
    if (d < INT32_MIN || d > INT32_MAX) return false;
    const auto v = static_cast<std::int32_t>(d);
    std::memcpy(out, &v, 4);
    return true;
}

// Initialize's colour test (cmp eax, [r15]: the target's width against the
// context's, then the game's je to keep it or live resolution's both-sides
// jne to make one) through a stub that first keeps the one target g_keep
// names, at any size.
bool install_keep_stub() {
    const std::uint64_t at = guest(kColourTest);
    auto* p = static_cast<std::uint8_t*>(guest_ptr(g_image->mem, at));
    if (p[0] != 0x41 || p[1] != 0x3b || p[2] != 0x07 || (p[3] != 0x74 && p[3] != 0x75)) {
        host_log("render scale: Initialize's colour test is not as expected; the scene's back buffer stays the game's choice");
        return false;
    }
    const std::uint64_t branch_to = at + 5 + static_cast<std::int8_t>(p[4]);
    const std::uint8_t jcc = p[3] == 0x74 ? 0x84 : 0x85;
    auto* page = static_cast<std::uint8_t*>(page_near(at));
    if (!page) return false;
    const auto base = reinterpret_cast<std::uint64_t>(page);
    const std::uint64_t code_at = base + 64;  // the slot at +0
    std::uint8_t code[40];
    std::size_t n = 0;
    const auto emit = [&](std::initializer_list<std::uint8_t> b) {
        for (std::uint8_t x : b) code[n++] = x;
    };
    bool ok = true;
    emit({0x4c, 0x3b, 0x35, 0, 0, 0, 0});  // cmp r14, [rip+keep]
    ok &= rel32(code_at + n, base, code + n - 4);
    emit({0x0f, 0x84, 0, 0, 0, 0});  // je keep
    ok &= rel32(code_at + n, guest(kColourKeep), code + n - 4);
    emit({0x41, 0x3b, 0x07});        // cmp eax, [r15]
    emit({0x0f, jcc, 0, 0, 0, 0});   // the test's own branch
    ok &= rel32(code_at + n, branch_to, code + n - 4);
    emit({0xe9, 0, 0, 0, 0});        // on to the height's test
    ok &= rel32(code_at + n, at + 5, code + n - 4);
    std::uint8_t jmp[5] = {0xe9, 0, 0, 0, 0};
    ok &= rel32(at + 5, code_at, jmp + 1);
    if (!ok) return false;
    g_keep = reinterpret_cast<std::uint64_t*>(page);
    *g_keep = 0;
    std::memcpy(page + 64, code, n);
    const std::uint64_t lo = at & ~0xfffull, hi = (at + 5 + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&g_image->mem, lo, hi - lo)) return false;
    std::memcpy(p, jmp, 5);
    guest_protect_rx(&g_image->mem, lo, hi - lo);
    return true;
}

// What the settings ask for at an output size: the quality's render size
// with an upscaler that runs here, the full size without. A plugin's 3D at a
// size of its own (host/scene_look.h, the PS1 Look's) has no upscaler: the
// Resolution fitted to the output's shape as an upscaler's would be, or the
// plugin's height and as wide as the output's shape gives, each way a
// multiple of 4 - YEBIS's copy brings it up nearest (render.cpp).
RenderScale wanted(std::uint32_t output_w, std::uint32_t output_h, bool upscaler, int quality) {
    RenderScale rs;
    rs.output_w = output_w;
    rs.output_h = output_h;
    if (std::uint32_t lines = 0; scene_look_scale(&lines)) {
        rs.quality = 0;
        double k = 1.0;
        int res_w = 0, res_h = 0;
        if (lines) {
            k = std::min(1.0, static_cast<double>(lines) / output_h);
        } else if (host_opt_resolution(&res_w, &res_h) && res_w > 0 && res_h > 0 &&
                   static_cast<std::uint64_t>(res_w) * static_cast<std::uint32_t>(res_h) < static_cast<std::uint64_t>(output_w) * output_h) {
            k = std::min(static_cast<double>(res_w) / output_w, static_cast<double>(res_h) / output_h);
        }
        rs.render_w = k < 1.0 ? std::max<std::uint32_t>(static_cast<std::uint32_t>(output_w * k + 0.5) & ~3u, 4) : output_w;
        rs.render_h = k < 1.0 ? std::max<std::uint32_t>(static_cast<std::uint32_t>(output_h * k + 0.5) & ~3u, 4) : output_h;
        return rs;
    }
    rs.quality = upscaler ? quality : 0;
    // The 3D's base: the output, or with an upscaler a Resolution smaller
    // than it (host_opt_output_size: the output is then the window's), fitted
    // to the output's shape - then the quality's share of that.
    std::uint32_t base_w = output_w, base_h = output_h;
    int res_w = 0, res_h = 0;
    if (upscaler && host_opt_resolution(&res_w, &res_h) && res_w > 0 && res_h > 0 &&
        static_cast<std::uint64_t>(res_w) * static_cast<std::uint32_t>(res_h) < static_cast<std::uint64_t>(output_w) * output_h) {
        const double k = std::min(static_cast<double>(res_w) / output_w, static_cast<double>(res_h) / output_h);
        base_w = std::max<std::uint32_t>(static_cast<std::uint32_t>(output_w * k) & ~3u, 4);
        base_h = std::max<std::uint32_t>(static_cast<std::uint32_t>(output_h * k) & ~3u, 4);
        rs.quality = std::max(rs.quality, 1);  // drawn smaller than the output: the upscaler brings it up
    }
    render_scale_size(quality && upscaler ? quality : 0, base_w, base_h, &rs.render_w, &rs.render_h);
    return rs;
}

void publish(const RenderScale& rs) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_now = rs;
    }
    g_scaled.store(rs.render_w != rs.output_w || rs.render_h != rs.output_h, std::memory_order_release);
}

// Settings: a change of the upscaler or its quality asks for a rebuild at the
// same Resolution (engine/live_resolution.cpp), whose quiet point retargets.
void settings_changed(const HostSettings& s) {
    if (!g_ready.load(std::memory_order_acquire)) return;
    RenderScale now;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        now = g_now;
    }
    bool runs = false;
    if (scene_look_scale()) {
        runs = false;  // a plugin's 3D size: no upscaler (host/upscaler.cpp)
    } else if (std::getenv("BBHOST_UPSCALE")) {
        runs = upscaler_method() != 0;
    } else {
        const UpscalerCaps caps = upscaler_caps();
        runs = s.upscaler == 1 ? caps.dlss : s.upscaler >= 2 ? caps.fsr : false;
    }
    const int quality = std::getenv("BBHOST_UPSCALE_QUALITY") ? now.quality : std::clamp(s.upscale_quality, 0, 3);
    // The upscaler on or off moves the game's own size too (host_opt_output_size:
    // the window's with one and a smaller Resolution, the Resolution without).
    int ow = 0, oh = 0;
    if (host_opt_output_size(&ow, &oh) && (static_cast<std::uint32_t>(ow) != now.output_w || static_cast<std::uint32_t>(oh) != now.output_h)) {
        live_resolution_request(static_cast<unsigned>(ow), static_cast<unsigned>(oh));
        return;
    }
    const RenderScale want = wanted(now.output_w, now.output_h, runs, quality);
    if (want.render_w != now.render_w || want.render_h != now.render_h) live_resolution_rebuild();
}

// A plugin's look changed what the scene's targets are (host/scene_look.h),
// once the game runs (main thread, its on_option): its 8-bit colour patch in
// or out (engine/patch_manifest.cpp scene-rgba8) - Initialize reads those
// bytes at each build, Reinitialize calls it whole - then the settings told
// again on the options' thread: settings_changed above asks for the 3D's new size, and
// engine/graphics_patch.cpp turns its AA off or back. The targets are made
// again when the bytes moved (the rgba8f patch file on as well moves none).
void look_changed() {
    const bool rgba8 = scene_look_colour_8bit();
    bool formats = false;
    std::string why;
    patch_manifest_refresh_live("scene-rgba8", rgba8, &formats, &why);
    // The settings are the options' thread's (the window's pump): published
    // there. With no window (headless) nothing else writes them, and nothing
    // pumps: published here.
    if (host_window_active())
        host_options_post([] { host_opt_republish(); });
    else
        host_opt_republish();
    if (formats && live_resolution_available()) live_resolution_rebuild();
    host_log("render scale: a plugin's look changed: the 3D's colour %s%s%s", rgba8 ? "8-bit" : "as the game makes it",
             formats ? ", its targets made again" : why.empty() ? "" : " at the next start - ", formats ? "" : why.c_str());
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
        // Its changes are the live resolution's rebuilds.
        if (const char* e = std::getenv("BBHOST_LIVE_RESOLUTION"); e && e[0] == '0') return rs;
        int w = 1920, h = 1080;
        if (!host_opt_output_size(&w, &h)) w = 1920, h = 1080;
        return wanted(static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), upscaler_asked(), quality_asked());
    }();
    if (out) *out = plan;
    return plan.output_w != 0;
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

void render_scale_install(ElfImage* image) {
    if (!render_scale_applied()) return;
    g_image = image;
    const bool keep = install_keep_stub();
    const bool hooks =
        engine_call_site_hook(image, guest(kBuildInitializeCall), guest(kInitialize), reinterpret_cast<void*>(&build_call_hook)) &&
        engine_call_site_hook(image, guest(kListenerReinitializeCall), guest(kReinitialize), reinterpret_cast<void*>(&rebuild_call_hook));
    if (!keep || !hooks) host_log("render scale: the scene's back-buffer hooks did not all go in; with display buffers of another size it misdraws");
    host_settings_listen(&settings_changed);
}

void render_scale_gx_init(std::uint32_t display_w, std::uint32_t display_h, std::uint64_t flip) {
    if (!g_image) return;
    RenderScale plan;
    render_scale_plan(&plan);
    render_scale_retarget(plan.output_w, plan.output_h, display_w, display_h, flip, nullptr, nullptr);
    g_ready.store(true, std::memory_order_release);
    scene_look_on_targets_change(&look_changed);
}

bool render_scale_retarget(std::uint32_t output_w, std::uint32_t output_h, std::uint32_t display_w, std::uint32_t display_h, std::uint64_t flip,
                           RenderScale* before, RenderScale* after) {
    if (!g_image) return false;
    RenderScale was;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        was = g_now;
    }
    const bool runs = upscaler_method() != 0;
    const RenderScale rs = wanted(output_w, output_h, runs, quality_asked());
    if (!was.output_w) was = rs;
    write_values(rs);
    choose_back_buffer(rs, display_w, display_h, flip);
    publish(rs);
    if (before) *before = was;
    if (after) *after = rs;
    const bool scaled = rs.render_w != rs.output_w || rs.render_h != rs.output_h;
    host_log("render scale: the 3D at %ux%u for %ux%u (%s)", rs.render_w, rs.render_h, output_w, output_h,
             scene_look_scale() ? (scaled ? "a plugin's look: brought up nearest" : "a plugin's look: the full size")
             : !runs            ? "no upscaler running: the full size"
             : scaled           ? "upscaled"
                                : "Native AA");
    return true;
}

void render_scale_flip(std::uint64_t flip) {
    std::size_t kept = 0;
    for (const OldBack& o : g_old_back) {
        if (flip >= o.flip + 3)
            release_ref(o.obj);
        else
            g_old_back[kept++] = o;
    }
    g_old_back.resize(kept);
}

bool render_scale_on(RenderScale* out) {
    if (!g_scaled.load(std::memory_order_acquire)) return false;
    if (out) {
        std::lock_guard<std::mutex> lk(g_mu);
        *out = g_now;
    }
    return true;
}
