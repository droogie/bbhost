// YEBIS's PPFX entry points (decomp/yebis/ppfx.cpp) beside the game's own
// (tests/decomp/eboot_kit.h). The effect's SetRenderTarget and ApplyEffects,
// YEBIS's debug print, the error text and _Assert are the test's recorders;
// the effect's result (success or an HRESULT failure) and YEBIS's two debug
// flags are generated. Ours must make the game's calls with the game's
// arguments in the game's order - garbage above the 32-bit ones - and return
// what the game's returns (a bool, in al). Two deliberate mistakes must be
// caught, and the entries hold the instructions written against. Skips
// without the 1.09 eboot.
#define DECOMP_YEBIS_PPFX_MUTANTS
int g_ppfx_mutant = 0;
#include "decomp/yebis/ppfx.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x99f);

struct Call {
    int who;
    u64 a[6];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;
u64 g_hr = 0, g_text = 0;

enum Who { kSetRtId, kApplyId, kPrintId, kTextId, kAssertId };

GUEST_ABI u64 fake_set_rt(u64 effect, u64 a, u64 b, u64 c, u64 d, u64 f) {
    g_rec.push_back({kSetRtId, {effect, a & 0xffffffffull, b, c & 0xffffffffull, d & 0xffffffffull, f}});
    return g_hr;
}
GUEST_ABI u64 fake_apply(u64 effect, u64 a, u64 b) {
    g_rec.push_back({kApplyId, {effect, a & 0xffffffffull, b & 0xffffffffull}});
    return g_hr;
}
GUEST_ABI u64 fake_print(u64 format, u64 a, u64 b) {
    // The second line's two values are there only when its format has them.
    const bool two = format != static_cast<u64>(kErrorPrefix);
    g_rec.push_back({kPrintId, {format, a & 0xffffffffull, two ? b : 0}});
    return 0;
}
GUEST_ABI u64 fake_text(u64 hr) {
    g_rec.push_back({kTextId, {hr & 0xffffffffull}});
    return g_text;
}
GUEST_ABI u64 fake_assert(u64 text, u64 function) {
    g_rec.push_back({kAssertId, {text, function}});
    return 0;
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

u64 garbage32() { return (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32(); }

long run(int cases) {
    long n = 0;
    for (int w = 0; w < 2; ++w) {
        const char* name = w ? "ppfxApplyEffects" : "ppfxBindFramebuffer";
        for (int t = 0; t < cases; ++t) {
            store<u64>(kSinglePostEffect, g_rng.next());
            store<u32>(kDebugOut, g_rng.chance(0.5) ? 0 : g_rng.u32() | 1);
            store<u32>(kDebugOutToo, g_rng.chance(0.5) ? 0 : g_rng.u32() | 1);
            // Success (0, positive) or failure (negative); the upper half is garbage, as rax's can be.
            const s32 hrs[] = {0, 1, static_cast<s32>(g_rng.u32() & 0x7fffffff), static_cast<s32>(0x80000000u | g_rng.u32()), -1};
            g_hr = (static_cast<u64>(g_rng.u32()) << 32) | static_cast<u32>(hrs[g_rng.range(0, 4)]);
            g_text = g_rng.next();
            u64 a[6];
            for (u64& v : a) v = garbage32();
            using F6 = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64);
            using F0 = GUEST_ABI u64 (*)();
            g_rec.clear();
            const u64 game_ret = w ? eboot_kit::fn<F0>(0xfb6980)() : eboot_kit::fn<F6>(0xfb68f0)(a[0], a[1], a[2], a[3], a[4], a[5]);
            const std::vector<Call> game = g_rec;
            g_rec.clear();
            const u64 our_ret = w ? reinterpret_cast<F0>(&ppfxApplyEffects)()
                                  : reinterpret_cast<F6>(&ppfxBindFramebuffer)(a[0], a[1], a[2], a[3], a[4], a[5]);
            if ((our_ret & 0xff) != (game_ret & 0xff)) fail(name, t, "the return value");
            if (g_rec != game) fail(name, t, "the calls");
            ++n;
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_ppfx_mutant = mutant;
    g_quiet = true;
    run(400);
    g_quiet = false;
    g_ppfx_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("yebis_ppfx_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 5000;
    eboot_kit::stub(kSetRenderTarget, reinterpret_cast<const void*>(&fake_set_rt));
    eboot_kit::stub(kApplyEffects, reinterpret_cast<const void*>(&fake_apply));
    eboot_kit::stub(kPrint, reinterpret_cast<const void*>(&fake_print));
    eboot_kit::stub(kErrorText, reinterpret_cast<const void*>(&fake_text));
    if (!eboot_kit::bind("_Assert", reinterpret_cast<const void*>(&fake_assert))) {
        std::printf("FAIL: _Assert not bound\n");
        return 1;
    }
    const u8* at = eboot_kit::file_at("yebis_ppfx_test", 0xfb68f0);
    if (!at || std::memcmp(at, kEntryBind, sizeof kEntryBind) != 0) fail("ppfxBindFramebuffer", -1, "the entry's instructions");
    at = eboot_kit::file_at("yebis_ppfx_test", 0xfb6980);
    if (!at || std::memcmp(at, kEntryApply, sizeof kEntryApply) != 0) fail("ppfxApplyEffects", -1, "the entry's instructions");
    const long total = run(cases);
    for (const int mutant : {1, 2}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("yebis_ppfx_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("yebis_ppfx_test: ok - %ld cases, 2 mutants caught\n", total);
    return 0;
}
