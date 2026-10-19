// Gnmx's constant update engine around a draw (decomp/gnmx/cue.cpp) beside
// the game's own (tests/decomp/eboot_kit.h). Every callee is a recorder; one
// given a buffer on the caller's stack (the built input table, the constant
// command buffer's copied header) is recorded by what the buffer holds, and
// the stage apply changes the dirty and constant-engine bytes as the real one
// does. On generated engines - shaders present or not, a given input table
// or none, the tessellation and geometry flags, each stage's dirty bits and
// counts, the global table's and the constant engine's bytes - ours must make
// the game's calls with the game's arguments in the game's order and leave
// the engine as the game's leaves it. Three deliberate mistakes must be
// caught, and the entries hold the instructions written against. Skips
// without the 1.09 eboot.
#define DECOMP_GNMX_CUE_MUTANTS
int g_cue_mutant = 0;
#include "decomp/gnmx/cue.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0xc0e);

struct Call {
    int who;
    std::vector<u64> a;
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;

enum Who { kCeWaitId, kBuildId, kSetPsId, kPrefetchId, kApplyId, kWriteDataId, kAcquireId, kCeIncId, kDeWaitId, kNopId, kDeIncId };

constexpr std::size_t kCueBytes = 0xe500;
alignas(64) u8 g_cue[kCueBytes], g_saved[kCueBytes], g_game[kCueBytes];
alignas(64) u8 g_ccb[0x40], g_vs[0x100], g_ps[0x100], g_table[0x80];
u8 g_apply_script[16][3];
int g_apply_calls = 0;

u64 cue_addr() { return reinterpret_cast<u64>(g_cue); }

GUEST_ABI u64 rec_ce_wait(u64 ccb, u64 diff) {
    g_rec.push_back({kCeWaitId, {ccb, diff & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_build(u64 out, u64 vs, u64 vs_count, u64 ps, u64 inputs) {
    g_rec.push_back({kBuildId, {vs, vs_count & 0xffffffffull, ps, inputs & 0xffffffffull}});
    u8* o = reinterpret_cast<u8*>(static_cast<std::uintptr_t>(out));
    for (int i = 0; i < 0x80; ++i) o[i] = static_cast<u8>(0x31 * i + 7);
    return 0;
}
GUEST_ABI u64 rec_set_ps(u64 dcb, u64 table, u64 count) {
    Call c{kSetPsId, {dcb, count & 0xffffffffull}};
    const u32 n = static_cast<u32>(count) < 32 ? static_cast<u32>(count) : 32;
    for (u32 i = 0; i < n; ++i) c.a.push_back(load<u32>(table + 4ull * i));
    g_rec.push_back(c);
    return 0;
}
GUEST_ABI u64 rec_prefetch(u64 dcb, u64 address, u64 bytes) {
    g_rec.push_back({kPrefetchId, {dcb, address, bytes & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_apply(u64 cue, u64 stage) {
    g_rec.push_back({kApplyId, {cue, stage & 0xffffffffull}});
    const u8* s = g_apply_script[g_apply_calls++ & 15];
    store<u8>(cue + kStagesDirty, s[0]);
    store<u8>(cue + kCeCounted, s[1]);
    store<u8>(cue + kCeUsed, s[2]);
    return 0;
}
GUEST_ABI u64 rec_write_data(u64 dcb, u64 dst, u64 src, u64 dwords, u64 sel, u64 confirm) {
    g_rec.push_back({kWriteDataId, {dcb, dst, src, dwords & 0xffffffffull, sel & 0xffffffffull, confirm & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_acquire(u64 dcb, u64 a, u64 b, u64 c) {
    g_rec.push_back({kAcquireId, {dcb, a & 0xffffffffull, b & 0xffffffffull, c & 0xffffffffull}});
    return 0;
}
GUEST_ABI u64 rec_ce_inc(u64 ccb) {
    g_rec.push_back({kCeIncId, {ccb}});
    return 0;
}
GUEST_ABI u64 rec_de_wait(u64 dcb) {
    g_rec.push_back({kDeWaitId, {dcb}});
    return 0;
}
GUEST_ABI u64 rec_nop(u64 cb, u64 n) {
    Call c{kNopId, {n & 0xffffffffull}};
    for (int i = 0; i < 8; ++i) c.a.push_back(load<u64>(cb + 8ull * i));
    g_rec.push_back(c);
    return 0;
}
GUEST_ABI u64 rec_de_inc(u64 dcb) {
    g_rec.push_back({kDeIncId, {dcb}});
    return 0;
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

void generate() {
    for (u8& b : g_cue) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_ccb) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_vs) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_ps) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_table) b = static_cast<u8>(g_rng.u32());
    const u64 cue = cue_addr();
    store<u64>(cue + kCcb, reinterpret_cast<u64>(g_ccb));
    store<u64>(cue + kVsShader, reinterpret_cast<u64>(g_vs));
    store<u64>(cue + kPsShader, g_rng.chance(0.15) ? 0 : reinterpret_cast<u64>(g_ps));
    store<u64>(cue + kPsInputs, g_rng.chance(0.5) ? 0 : reinterpret_cast<u64>(g_table));
    store<u8>(reinterpret_cast<u64>(g_ps) + 0x38, static_cast<u8>(g_rng.chance(0.2) ? 0 : g_rng.range(1, 32)));
    store<u8>(reinterpret_cast<u64>(g_vs) + 3, static_cast<u8>(g_rng.range(0, 8)));
    store<u8>(reinterpret_cast<u64>(g_vs) + 0x24, static_cast<u8>(g_rng.range(0, 8)));
    store<u8>(reinterpret_cast<u64>(g_ps) + 3, static_cast<u8>(g_rng.range(0, 8)));
    for (int s = 0; s < 8; ++s) store<u16>(cue + kStages + s * kStageBytes + 8, static_cast<u16>(g_rng.chance(0.3) ? 0 : g_rng.range(1, 9)));
    store<u8>(cue + kCeUsed, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    store<u8>(cue + kCeCounted, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    store<u8>(cue + kGlobalDirty, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    store<u8>(cue + kPrefetchOn, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    for (auto& s : g_apply_script) {
        s[0] = static_cast<u8>(g_rng.u32());
        s[1] = static_cast<u8>(g_rng.chance(0.5) ? 0 : 1);
        s[2] = static_cast<u8>(g_rng.chance(0.5) ? 0 : 1);
    }
}

long run(int cases) {
    long n = 0;
    for (int w = 0; w < 2; ++w) {
        const char* name = w ? "gnmx_cue_post_draw" : "gnmx_cue_pre_draw";
        const u64 bn = w ? 0x1493150 : 0x1492e80;
        void* ours = w ? reinterpret_cast<void*>(&gnmx_cue_post_draw) : reinterpret_cast<void*>(&gnmx_cue_pre_draw);
        for (int t = 0; t < cases; ++t) {
            generate();
            using F1 = GUEST_ABI void (*)(u64);
            std::memcpy(g_saved, g_cue, kCueBytes);
            u8 ccb_saved[sizeof g_ccb];
            std::memcpy(ccb_saved, g_ccb, sizeof g_ccb);
            g_rec.clear();
            g_apply_calls = 0;
            eboot_kit::fn<F1>(bn)(cue_addr());
            const std::vector<Call> game = g_rec;
            std::memcpy(g_game, g_cue, kCueBytes);
            std::memcpy(g_cue, g_saved, kCueBytes);
            std::memcpy(g_ccb, ccb_saved, sizeof g_ccb);
            g_rec.clear();
            g_apply_calls = 0;
            reinterpret_cast<F1>(ours)(cue_addr());
            if (g_rec != game) fail(name, t, "the calls");
            if (std::memcmp(g_cue, g_game, kCueBytes) != 0) fail(name, t, "the engine");
            ++n;
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_cue_mutant = mutant;
    g_quiet = true;
    run(400);
    g_quiet = false;
    g_cue_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gnmx_cue_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 5000;
    eboot_kit::stub(kCeWait, reinterpret_cast<const void*>(&rec_ce_wait));
    eboot_kit::stub(kBuildPsInputs, reinterpret_cast<const void*>(&rec_build));
    eboot_kit::stub(kSetPsInputs, reinterpret_cast<const void*>(&rec_set_ps));
    eboot_kit::stub(kPrefetch, reinterpret_cast<const void*>(&rec_prefetch));
    eboot_kit::stub(kApplyStage, reinterpret_cast<const void*>(&rec_apply));
    eboot_kit::stub(kWriteData, reinterpret_cast<const void*>(&rec_write_data));
    eboot_kit::stub(kAcquireMem, reinterpret_cast<const void*>(&rec_acquire));
    eboot_kit::stub(kCeIncrement, reinterpret_cast<const void*>(&rec_ce_inc));
    eboot_kit::stub(kDeWaitOnCe, reinterpret_cast<const void*>(&rec_de_wait));
    eboot_kit::stub(kNop, reinterpret_cast<const void*>(&rec_nop));
    eboot_kit::stub(kDeIncrement, reinterpret_cast<const void*>(&rec_de_inc));
    const u8* at = eboot_kit::file_at("gnmx_cue_test", 0x1492e80);
    if (!at || std::memcmp(at, kEntryPreDraw, sizeof kEntryPreDraw) != 0) fail("gnmx_cue_pre_draw", -1, "the entry's instructions");
    at = eboot_kit::file_at("gnmx_cue_test", 0x1493150);
    if (!at || std::memcmp(at, kEntryPostDraw, sizeof kEntryPostDraw) != 0) fail("gnmx_cue_post_draw", -1, "the entry's instructions");
    const long total = run(cases);
    for (const int mutant : {1, 2, 3}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("gnmx_cue_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("gnmx_cue_test: ok - %ld cases, 3 mutants caught\n", total);
    return 0;
}
