// YEBIS's ClearRenderTargetView and its compute clear (decomp/yebis/clear.cpp)
// beside the game's own (tests/decomp/eboot_kit.h). YEBIS's state queries
// and setters, the dispatch, the packet writers, the fast clear and the clear
// colour's writer are recorders (a buffer on the caller's stack recorded by
// what it holds; the command buffer's room from a scratch arena); Gnm's own
// texture, buffer and render-target methods run as the game's in both, as do
// the helpers on the decomp list that one of them calls. On generated
// textures (every type, mip and slice ranges on both sides of empty), render
// targets (with and without the fast-clear mask, slice ranges, fragment
// counts) and colours (1, NaN, values past 1 and past an int, negatives),
// ours must make the game's calls with the game's arguments in the game's
// order and leave the context and the arena as the game's leave them. Three
// deliberate mistakes must be caught, and the entries hold the instructions
// written against. Skips without the 1.09 eboot.
#define DECOMP_YEBIS_CLEAR_MUTANTS
int g_clear_mutant = 0;
#include "decomp/yebis/clear.cpp"

#include "eboot_kit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0xc1ea);

struct Call {
    int who;
    std::vector<u64> a;
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;

enum Who {
    kShadersId, kStateTexId, kStateBufId, kSetCsId, kSetBufsId, kSetTexsId, kDispatchId, kEmbedId, kEosId, kWaitId, kAcquireId,
    kAcquireRangeId, kFastClearId, kClearColorId
};

constexpr std::size_t kCtxBytes = 0x3600, kArenaBytes = 0x200;
alignas(64) u8 g_ctx[kCtxBytes], g_view[0x100], g_tex[0x100], g_rt[0x40], g_cb[0x40], g_shaders[0x40], g_colour[16], g_in[0x20];
alignas(64) u8 g_arena[kArenaBytes];
u64 g_arena_used = 0;
u64 g_state_tex[4], g_buf_lo, g_buf_hi;

u64 A(const void* p) { return reinterpret_cast<u64>(p); }
std::vector<u64> words(u64 at, int n) {
    std::vector<u64> v;
    for (int i = 0; i < n; ++i) v.push_back(load<u64>(at + 8ull * i));
    return v;
}

GUEST_ABI u64 rec_shaders(u64 state) {
    g_rec.push_back({kShadersId, {state}});
    return A(g_shaders);
}
GUEST_ABI u64 rec_state_tex(u64 out, u64 state, u64 stage, u64 slot) {
    g_rec.push_back({kStateTexId, {state, stage & 0xffffffffull, slot & 0xffffffffull}});
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(out)), g_state_tex, sizeof g_state_tex);
    return 0;
}
GUEST_ABI Pair rec_state_buf(u64 state, u64 stage, u64 slot) {
    g_rec.push_back({kStateBufId, {state, stage & 0xffffffffull, slot & 0xffffffffull}});
    return {g_buf_lo, g_buf_hi};
}
GUEST_ABI u64 rec_set_cs(u64 y, u64 shader) {
    g_rec.push_back({kSetCsId, {y, shader}});
    return 0;
}
GUEST_ABI u64 rec_set_bufs(u64 y, u64 stage, u64 slot, u64 count, u64 bufs) {
    Call c{kSetBufsId, {y, stage & 0xffffffffull, slot & 0xffffffffull, count & 0xffffffffull, bufs != 0}};
    if (bufs) for (u64 w : words(bufs, 2)) c.a.push_back(w);
    g_rec.push_back(c);
    return 0;
}
GUEST_ABI u64 rec_set_texs(u64 y, u64 stage, u64 slot, u64 count, u64 texs) {
    Call c{kSetTexsId, {y, stage & 0xffffffffull, slot & 0xffffffffull, count & 0xffffffffull, texs != 0}};
    if (texs) for (u64 w : words(texs, 4)) c.a.push_back(w);
    g_rec.push_back(c);
    return 0;
}
GUEST_ABI u64 rec_dispatch(u64 y, u64 x, u64 gy, u64 z) {
    g_rec.push_back({kDispatchId, {y, x & 0xffffffffull, gy & 0xffffffffull, z & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_embed(u64 cb, u64 bytes, u64 align) {
    g_rec.push_back({kEmbedId, {cb, bytes & 0xffffffffull, align & 0xffffffffull}});
    const u64 at = A(g_arena) + g_arena_used;
    g_arena_used += 0x20;
    return at;
}
GUEST_ABI u64 rec_eos(u64 cb, u64 event, u64 label, u64 value) {
    g_rec.push_back({kEosId, {cb, event & 0xffffffffull, label, value & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_wait(u64 cb, u64 address, u64 mask, u64 function, u64 reference) {
    g_rec.push_back({kWaitId, {cb, address, mask & 0xffffffffull, function & 0xffffffffull, reference & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_acquire(u64 cb, u64 a, u64 b, u64 c) {
    g_rec.push_back({kAcquireId, {cb, a & 0xffffffffull, b & 0xffffffffull, c & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_acquire_range(u64 cb, u64 base, u64 size, u64 action, u64 extended, u64 cache, u64 engine) {
    g_rec.push_back({kAcquireRangeId, {cb, base & 0xffffffffull, size & 0xffffffffull, action & 0xffffffffull, extended & 0xffffffffull,
                                       cache & 0xffffffffull, engine & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_fast_clear(u64 y, u64 rt) {
    g_rec.push_back({kFastClearId, {y, rt}});
    return 0;
}
GUEST_ABI u64 rec_clear_color(u64 cb, u64 slot, u64 words_at) {
    g_rec.push_back({kClearColorId, {cb, slot & 0xffffffffull, load<u64>(words_at)}});
    return 0;
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

float colour_value() {
    switch (g_rng.range(0, 7)) {
        case 0: return 1.0f;
        case 1: return std::numeric_limits<float>::quiet_NaN();
        case 2: return g_rng.uniform(1.0f, 2.0f);
        case 3: return 3.0e9f;
        case 4: return -g_rng.uniform(0.0f, 1.0f);
        case 5: return 0.0f;
        default: return g_rng.uniform(0.0f, 1.0f);
    }
}

// A Gnm texture: random but for its type and small mip and slice ranges.
void make_texture(u8* t) {
    for (int i = 0; i < 0x20; ++i) t[i] = static_cast<u8>(g_rng.u32());
    const u32 type = g_rng.chance(0.9) ? static_cast<u32>(g_rng.range(8, 13)) : static_cast<u32>(g_rng.range(0, 15));
    const u32 base = static_cast<u32>(g_rng.range(0, 3)), last = static_cast<u32>(g_rng.chance(0.85) ? g_rng.range(static_cast<int>(base), 4) : 0);
    u32 d3 = load<u32>(A(t) + 0xc);
    d3 = (d3 & 0x0ff00fffu) | base << 12 | last << 16 | type << 28;
    store<u32>(A(t) + 0xc, d3);
    store<u32>(A(t) + 0x10, (load<u32>(A(t) + 0x10) & ~0x1fffu) | static_cast<u32>(g_rng.range(0, 9)));
    const u32 bs = static_cast<u32>(g_rng.range(0, 3)), ls = static_cast<u32>(g_rng.chance(0.85) ? g_rng.range(static_cast<int>(bs), 4) : 0);
    store<u32>(A(t) + 0x14, (load<u32>(A(t) + 0x14) & 0xfc000000u) | bs | ls << 13);
}

void generate() {
    for (u8& b : g_ctx) b = static_cast<u8>(g_rng.u32());
    for (u8* p : {g_view, g_tex, g_rt, g_cb, g_shaders}) for (int i = 0; i < 0x40; ++i) p[i] = static_cast<u8>(g_rng.u32());
    for (int i = 0x40; i < 0x100; ++i) g_view[i] = static_cast<u8>(g_rng.u32());
    for (u8& b : g_arena) b = static_cast<u8>(g_rng.u32());
    const u64 ctx = A(g_ctx), y = ctx + kGnm;
    store<u64>(y + 0x8, A(g_cb));  // ctx + kCb
    store<u64>(y + 0x28, g_rng.next());  // the state, handed to the recorders only
    store<u64>(A(g_view) + 0xd8, A(g_view) + 0xf0);
    store<u64>(A(g_view) + 0xf0, A(g_rt));
    store<u64>(A(g_view) + 0x80, A(g_tex));
    // The render target: slices and fragments small, the fast-clear mask on or off.
    store<u32>(A(g_rt) + 0xc, (load<u32>(A(g_rt) + 0xc) & 0xff000000u) | static_cast<u32>(g_rng.range(0, 3)) | static_cast<u32>(g_rng.range(0, 4)) << 13);
    store<u32>(A(g_rt) + 0x1c, g_rng.chance(0.5) ? 0 : g_rng.u32() | 1);
    store<u16>(A(g_rt) + 0x18, static_cast<u16>(g_rng.range(0, 300)));
    store<u16>(A(g_rt) + 0x1a, static_cast<u16>(g_rng.range(0, 300)));
    for (int i = 0; i < 4; ++i) store<float>(A(g_colour) + 4ull * i, colour_value());
    make_texture(g_in);
    for (u64& w : g_state_tex) w = g_rng.chance(0.3) ? 0 : g_rng.next();
    g_buf_lo = g_rng.chance(0.3) ? 0 : g_rng.next();
    g_buf_hi = g_rng.next();
}

constexpr int kFns = 3;
const char* const kNames[kFns] = {"yebis_compute_clear", "yebis_clear_target_compute", "yebis_context_clear_render_target_view"};
const u64 kBn[kFns] = {0x15e8930, 0x15e8e90, 0x2e41320};

u64 snapshot(std::vector<u8>& out) {
    out.assign(g_ctx, g_ctx + kCtxBytes);
    out.insert(out.end(), g_arena, g_arena + kArenaBytes);
    out.insert(out.end(), g_cb, g_cb + 0x40);
    out.insert(out.end(), g_tex, g_tex + 0x40);
    return out.size();
}

// What the cases covered: clears that dispatched, that ended with the wait,
// that stopped early; views cleared fast and by compute.
long g_dispatched = 0, g_waited = 0, g_stopped = 0, g_fast = 0, g_slow = 0;

long run(int cases) {
    long n = 0;
    for (int f = 0; f < kFns; ++f) {
        for (int t = 0; t < cases; ++t) {
            generate();
            const u64 ctx = A(g_ctx), y = ctx + kGnm;
            u64 a[3];
            if (f == 0) a[0] = y, a[1] = A(g_in), a[2] = A(g_colour);
            else if (f == 1) a[0] = y, a[1] = A(g_rt), a[2] = A(g_colour);
            else a[0] = ctx, a[1] = A(g_view), a[2] = A(g_colour);
            std::vector<u8> before, game, ours;
            snapshot(before);
            u8 in_saved[0x20];
            std::memcpy(in_saved, g_in, sizeof in_saved);
            using F3 = GUEST_ABI void (*)(u64, u64, u64);
            g_rec.clear();
            g_arena_used = 0;
            eboot_kit::fn<F3>(kBn[f])(a[0], a[1], a[2]);
            const std::vector<Call> calls = g_rec;
            snapshot(game);
            if (!g_quiet) {
                bool dispatched = false, waited = false, fast = false;
                for (const Call& c : calls) {
                    dispatched |= c.who == kDispatchId;
                    waited |= c.who == kEosId;
                    fast |= c.who == kFastClearId;
                }
                if (f == 0) {
                    g_dispatched += dispatched;
                    g_waited += waited;
                    g_stopped += !waited;
                } else if (f == 2) {
                    g_fast += fast;
                    g_slow += !fast;
                }
            }
            std::memcpy(g_ctx, before.data(), kCtxBytes);
            std::memcpy(g_arena, before.data() + kCtxBytes, kArenaBytes);
            std::memcpy(g_cb, before.data() + kCtxBytes + kArenaBytes, 0x40);
            std::memcpy(g_tex, before.data() + kCtxBytes + kArenaBytes + 0x40, 0x40);
            std::memcpy(g_in, in_saved, sizeof in_saved);
            g_rec.clear();
            g_arena_used = 0;
            void* ours_fn = f == 0 ? reinterpret_cast<void*>(&yebis_compute_clear)
                                   : f == 1 ? reinterpret_cast<void*>(&yebis_clear_target_compute)
                                            : reinterpret_cast<void*>(&yebis_context_clear_render_target_view);
            reinterpret_cast<F3>(ours_fn)(a[0], a[1], a[2]);
            snapshot(ours);
            if (g_rec != calls) fail(kNames[f], t, "the calls");
            if (ours != game) fail(kNames[f], t, "the memory");
            ++n;
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_clear_mutant = mutant;
    g_quiet = true;
    run(300);
    g_quiet = false;
    g_clear_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("yebis_clear_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 3000;
    eboot_kit::stub(kStateShaders, reinterpret_cast<const void*>(&rec_shaders));
    eboot_kit::stub(kStateTexture, reinterpret_cast<const void*>(&rec_state_tex));
    eboot_kit::stub(kStateBuffer, reinterpret_cast<const void*>(&rec_state_buf));
    eboot_kit::stub(kSetComputeShader, reinterpret_cast<const void*>(&rec_set_cs));
    eboot_kit::stub(kSetBuffers, reinterpret_cast<const void*>(&rec_set_bufs));
    eboot_kit::stub(kSetTextures, reinterpret_cast<const void*>(&rec_set_texs));
    eboot_kit::stub(kDispatch, reinterpret_cast<const void*>(&rec_dispatch));
    eboot_kit::stub(kEmbedData, reinterpret_cast<const void*>(&rec_embed));
    eboot_kit::stub(kEndOfShader, reinterpret_cast<const void*>(&rec_eos));
    eboot_kit::stub(kWaitRegMem, reinterpret_cast<const void*>(&rec_wait));
    eboot_kit::stub(kAcquireMem, reinterpret_cast<const void*>(&rec_acquire));
    eboot_kit::stub(kAcquireRange, reinterpret_cast<const void*>(&rec_acquire_range));
    eboot_kit::stub(kFastClear, reinterpret_cast<const void*>(&rec_fast_clear));
    eboot_kit::stub(kSetClearColor, reinterpret_cast<const void*>(&rec_clear_color));
    const u8* entries[kFns] = {kEntryComputeClear, kEntryClearTarget, kEntryClearView};
    const std::size_t sizes[kFns] = {sizeof kEntryComputeClear, sizeof kEntryClearTarget, sizeof kEntryClearView};
    for (int f = 0; f < kFns; ++f) {
        const u8* at = eboot_kit::file_at("yebis_clear_test", kBn[f]);
        if (!at || std::memcmp(at, entries[f], sizes[f]) != 0) fail(kNames[f], -1, "the entry's instructions");
    }
    const long total = run(cases);
    for (const int mutant : {1, 2, 3}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("yebis_clear_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("yebis_clear_test: ok - %ld cases (compute clears: %ld dispatched, %ld ended with the wait, %ld stopped early; views: %ld "
                "fast, %ld by compute), 3 mutants caught\n",
                total, g_dispatched, g_waited, g_stopped, g_fast, g_slow);
    return 0;
}
