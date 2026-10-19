// GX's present (decomp/gx/present.cpp) beside the game's own
// (tests/decomp/eboot_kit.h). VideoOut's flip status, the kernel's event
// wait and the thread query are the test's (bound in place of the imports'
// traps), scripted per case: the flip status's argument stepping up to the
// count the wait needs, a failing status or wait now and then, the calling
// thread the render thread's or another's. The engine's callees are
// recorders (the flip's result scripted too). On generated swap chains -
// one to three display buffers, flip counts on both sides of the limit,
// sync intervals 0-6 with garbage above them - ours must make the game's
// calls with the game's arguments in the game's order, leave the swap chain
// and the context as the game's leaves them and return what the game's
// returns. Three deliberate mistakes must be caught, and the entries hold the
// instructions written against. Skips without the 1.09 eboot.
#define DECOMP_GX_PRESENT_MUTANTS
int g_present_mutant = 0;
#include "decomp/gx/present.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x9f11);

struct Call {
    int who;
    u64 a[5];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;

// The script: what each call of the imports and the flip answers.
struct Script {
    u64 self[2];
    int status_rc[8];
    s64 flip_arg[8];
    int wait_rc[8];
    u64 flip_ret;
} g_script;
int g_status_calls = 0, g_wait_calls = 0, g_self_calls = 0;

enum Who { kSelfId, kStatusId, kWaitId, kEndOfFrameId, kCopyId, kFlipId, kReleaseId, kResetId, kFlushId };

GUEST_ABI u64 fake_self() {
    g_rec.push_back({kSelfId, {}});
    return g_script.self[g_self_calls++ & 1];
}
GUEST_ABI u64 fake_status(u64 handle, u64 status) {
    const int k = g_status_calls++ & 7;
    g_rec.push_back({kStatusId, {handle & 0xffffffffull}});
    u8* s = reinterpret_cast<u8*>(static_cast<std::uintptr_t>(status));
    for (int i = 0; i < 0x40; ++i) s[i] = static_cast<u8>(0x5a + i);  // the whole SceVideoOutFlipStatus, as VideoOut writes it
    std::memcpy(s + kStatusFlipArg, &g_script.flip_arg[k], 8);
    return static_cast<u64>(static_cast<s64>(g_script.status_rc[k]));
}
GUEST_ABI u64 fake_wait(u64 queue, u64 events, u64 count, u64 out, u64 timeout) {
    (void)events;
    (void)out;
    if (g_wait_calls > 64) {
        std::printf("FAIL: the wait loop does not end\n");
        std::exit(1);
    }
    g_rec.push_back({kWaitId, {queue, count & 0xffffffffull, timeout}});
    return static_cast<u64>(static_cast<s64>(g_script.wait_rc[g_wait_calls++ & 7]));
}
template <int W, int N, u32 Narrow>
GUEST_ABI u64 recorder(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4) {
    Call c{W, {a0, a1, a2, a3, a4}};
    for (int k = 0; k < 5; ++k) {
        if (k >= N) c.a[k] = 0;
        else if ((Narrow >> k) & 1) c.a[k] &= 0xffffffffull;
    }
    if (W == kReleaseId) c.a[1] = load<u64>(a1);  // the command buffer, by pointer
    g_rec.push_back(c);
    return W == kFlipId ? g_script.flip_ret : 0x5eed0000u + W;
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

// The swap chain, the context and the block pools' holder.
constexpr std::size_t kScBytes = 0x3400, kCtxBytes = 0xb800, kPoolBytes = 0x40;
alignas(64) u8 g_sc[kScBytes], g_ctx[kCtxBytes], g_pool[kPoolBytes];
alignas(64) u8 g_saved[kScBytes + kCtxBytes + kPoolBytes], g_game[kScBytes + kCtxBytes + kPoolBytes];
void save(u8* to) {
    std::memcpy(to, g_sc, kScBytes);
    std::memcpy(to + kScBytes, g_ctx, kCtxBytes);
    std::memcpy(to + kScBytes + kCtxBytes, g_pool, kPoolBytes);
}
void put(const u8* from) {
    std::memcpy(g_sc, from, kScBytes);
    std::memcpy(g_ctx, from + kScBytes, kCtxBytes);
    std::memcpy(g_pool, from + kScBytes + kCtxBytes, kPoolBytes);
}

u64 generate() {
    for (u8& b : g_sc) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_ctx) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_pool) b = static_cast<u8>(g_rng.u32());
    const u64 sc = reinterpret_cast<u64>(g_sc), ctx = reinterpret_cast<u64>(g_ctx);
    const u32 buffers = static_cast<u32>(g_rng.range(1, 3));
    const s64 flips = g_rng.range(0, 6);
    store<u64>(sc + kContext, ctx);
    store<s64>(sc + kFlips, flips);
    store<u32>(sc + kInFlight, buffers);
    store<u64>(ctx + 0x8, reinterpret_cast<u64>(g_pool));  // [ctx + 8] + 8: the block pools
    const u64 render = g_rng.next();
    store<u64>(ctx + kRenderThread, render);
    g_script.self[0] = g_rng.chance(0.8) ? render : render ^ 1;
    g_script.self[1] = g_rng.chance(0.9) ? render : render ^ 2;
    // The flip status: a few answers under the count the wait needs, then one at it or past it.
    const int waits = g_rng.range(0, 3);
    const s64 need = 1 - static_cast<s64>(buffers) + flips;
    for (int k = 0; k < 8; ++k) {
        // The last answer well past the count, so a wrong threshold (a mutant) ends too.
        g_script.flip_arg[k] = k < waits ? need - 1 - g_rng.range(0, 2) : k == 7 ? need + 8 : need + g_rng.range(0, 1);
        g_script.status_rc[k] = g_rng.chance(0.05) ? -0x7f000000 - g_rng.range(0, 9) : g_rng.range(0, 1);
        g_script.wait_rc[k] = g_rng.chance(0.05) ? -0x7ffffff0 : 0;
    }
    g_script.flip_ret = g_rng.chance(0.1) ? g_rng.u32() | 1 : 0;
    return sc;
}

void reset_script() { g_status_calls = g_wait_calls = g_self_calls = 0; }

long run(int cases) {
    long n = 0;
    for (int w = 0; w < 2; ++w) {
        const char* name = w ? "gx_swap_chain_present" : "gx_swap_chain_present_frame";
        for (int t = 0; t < cases; ++t) {
            const u64 sc = generate();
            const u64 frame = g_rng.next();
            const u64 sync = (g_rng.next() & ~0xffffffffull) | static_cast<u64>(g_rng.range(0, 6));
            using F3 = GUEST_ABI u64 (*)(u64, u64, u64);
            using F2 = GUEST_ABI u64 (*)(u64, u64);
            save(g_saved);
            g_rec.clear();
            reset_script();
            const u32 game_ret = static_cast<u32>(w ? eboot_kit::fn<F2>(0x2ad5dd0)(sc, sync) : eboot_kit::fn<F3>(kPresentFrame)(sc, frame, sync));
            const std::vector<Call> game = g_rec;
            save(g_game);
            put(g_saved);
            g_rec.clear();
            reset_script();
            const u32 our_ret = static_cast<u32>(w ? reinterpret_cast<F2>(&gx_swap_chain_present)(sc, sync)
                                                   : reinterpret_cast<F3>(&gx_swap_chain_present_frame)(sc, frame, sync));
            u8 ours[sizeof g_saved];
            save(ours);
            if (our_ret != game_ret) fail(name, t, "the return value");
            if (g_rec != game) fail(name, t, "the calls");
            if (std::memcmp(ours, g_game, sizeof ours) != 0) fail(name, t, "the memory");
            ++n;
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_present_mutant = mutant;
    g_quiet = true;
    run(500);
    g_quiet = false;
    g_present_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_present_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 5000;
    bool bound = eboot_kit::bind("scePthreadSelf", reinterpret_cast<const void*>(&fake_self));
    bound = eboot_kit::bind("sceVideoOutGetFlipStatus", reinterpret_cast<const void*>(&fake_status)) && bound;
    bound = eboot_kit::bind("sceKernelWaitEqueue", reinterpret_cast<const void*>(&fake_wait)) && bound;
    if (!bound) {
        std::printf("FAIL: an import not bound\n");
        return 1;
    }
    eboot_kit::stub(kEndOfFrame, reinterpret_cast<const void*>(&recorder<kEndOfFrameId, 3, 0>));
    eboot_kit::stub(kCopyToDisplay, reinterpret_cast<const void*>(&recorder<kCopyId, 4, 0>));
    eboot_kit::stub(kFlip, reinterpret_cast<const void*>(&recorder<kFlipId, 5, 0xa>));
    eboot_kit::stub(kReleaseBlocks, reinterpret_cast<const void*>(&recorder<kReleaseId, 2, 0>));
    eboot_kit::stub(kResetState, reinterpret_cast<const void*>(&recorder<kResetId, 1, 0>));
    eboot_kit::stub(kFlush, reinterpret_cast<const void*>(&recorder<kFlushId, 1, 0>));
    const u8* at = eboot_kit::file_at("gx_present_test", 0x2ad5dd0);
    if (!at || std::memcmp(at, kEntryPresent, sizeof kEntryPresent) != 0) fail("gx_swap_chain_present", -1, "the entry's instructions");
    at = eboot_kit::file_at("gx_present_test", kPresentFrame);
    if (!at || std::memcmp(at, kEntryPresentFrame, sizeof kEntryPresentFrame) != 0) fail("gx_swap_chain_present_frame", -1, "the entry's instructions");
    const long total = run(cases);
    for (const int mutant : {1, 2, 3}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("gx_present_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("gx_present_test: ok - %ld cases, 3 mutants caught\n", total);
    return 0;
}
