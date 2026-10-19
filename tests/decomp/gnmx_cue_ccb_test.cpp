// The constant command buffer around Gnmx's pre-draw (decomp/gnmx/cue.cpp)
// with the constant engine's writers ours (decomp/gx/constant_engine.cpp).
// The pre-draw - ours and the game's own (tests/decomp/eboot_kit.h) - copies
// the buffer's header right after the wait on the draw engine and, when the
// draw left the constant engine unused, writes a two-dword NOP through the
// copy 8 bytes below its write pointer, with the game's NOP writer. That NOP
// has to land on the wait in both modes - the dead packets on (the counter
// step writes nothing) and off: nothing below the buffer touched (in the game
// the heap block's header in front of it, which the flip pass's teardown
// frees through), no dword of a reservation already in the buffer touched,
// the wait's two dwords a NOP when the engine stayed unused and still the
// wait when it was used, nothing written past the write pointer, and ours
// leaving everything as the game's leaves it. Runs of one to four draws on
// one buffer, empty or holding a reservation's payload. Three deliberate
// mistakes must be caught: the wait written as nothing while the dead
// packets are on (as from PR #149 to the fix), the NOP 4 bytes off, the NOP
// left out. Skips without the 1.09 eboot.
#define DECOMP_GNMX_CUE_MUTANTS
#define DECOMP_CE_MUTANTS
int g_cue_mutant = 0;
int g_ce_mutant = 0;
#include "decomp/gnmx/cue.cpp"
#include "decomp/gx/constant_engine.cpp"

// decomp/gx/labels.cpp's, which gnm_ce_configure sets with the constant engine.
u8 g_ce_counters_dead = 0;

#include "eboot_kit.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0xcc0b);

constexpr std::size_t kCueBytes = 0xe500;
constexpr u32 kNopHeader = 0xc0001000u, kWaitHeader = 0xc0008800u;

// The constant command buffer's memory as the flip pass's lies: the block in
// front of it (in the game a heap block, its header the 16 bytes before the
// chunk), the chunk, and what follows it.
struct alignas(64) Memory {
    u8 below[64];
    u8 chunk[2048];
    u8 above[64];
};
struct alignas(64) State {
    Memory mem;
    u8 ccb[0x40];  // the command buffer: +0 the chunk, +8 its end, +0x10 the write pointer, +0x18 the refill, +0x20 its argument
    u8 cue[kCueBytes];
    u8 vs[0x100], ps[0x100], table[0x80];
};
State g_s, g_start, g_game;

struct Script {
    u8 dirty, counted, used, reserve_dwords;
};
Script g_script[16];
int g_applies = 0;
u8 g_last_used = 0;  // the engine's "used" byte as the pre-draw will read it at its end

// What a draw's checks need: where its wait goes and the payloads in the buffer.
struct Payload {
    u64 at;
    u32 dwords;
};
std::vector<Payload> g_payloads;
bool g_refilled = false;

u64 ccb() { return reinterpret_cast<u64>(g_s.ccb); }
u64 cue() { return reinterpret_cast<u64>(g_s.cue); }
u64 chunk() { return reinterpret_cast<u64>(g_s.mem.chunk); }

GUEST_ABI u64 refill(u64, u32, u64, u64) {
    g_refilled = true;  // never with room to spare: a refill here is a failure
    return 0;
}
GUEST_ABI u64 nothing() { return 0; }

// A reservation of `dwords` with its payload filled, as the constant update
// engine fills one.
void reserve(u32 dwords) {
    const u32 offset = static_cast<u32>(g_rng.range(0, 0x2f00)) * 4;
    const u64 payload = gnm_ce_reserve_const_ram(ccb(), offset, dwords);
    if (!payload) return;
    for (u32 k = 0; k < dwords; ++k) store<u32>(payload + 4ull * k, g_rng.u32() | 1u);
    g_payloads.push_back({payload, dwords});
}

// The stage apply (0x1491f10): the dirty and constant-engine bytes from the
// script, and a reservation when the stage used the constant engine.
GUEST_ABI u64 apply(u64 c, u64) {
    const Script& s = g_script[g_applies++ & 15];
    store<u8>(c + kStagesDirty, s.dirty);
    store<u8>(c + kCeCounted, s.counted);
    store<u8>(c + kCeUsed, s.used);
    g_last_used = s.used;
    if (s.used && s.reserve_dwords) reserve(s.reserve_dwords);
    return 0;
}

void generate_cue() {
    for (u8& b : g_s.cue) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_s.vs) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_s.ps) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_s.table) b = static_cast<u8>(g_rng.u32());
    const u64 c = cue();
    store<u64>(c + kCcb, ccb());
    store<u64>(c + kDcb, 0x7000000000ull);  // the draw engine's writers are stubs: never read
    store<u64>(c + kVsShader, reinterpret_cast<u64>(g_s.vs));
    store<u64>(c + kPsShader, g_rng.chance(0.15) ? 0 : reinterpret_cast<u64>(g_s.ps));
    store<u64>(c + kPsInputs, g_rng.chance(0.5) ? 0 : reinterpret_cast<u64>(g_s.table));
    store<u8>(reinterpret_cast<u64>(g_s.ps) + 0x38, static_cast<u8>(g_rng.chance(0.2) ? 0 : g_rng.range(1, 32)));
    for (int s = 0; s < 8; ++s) store<u16>(c + kStages + s * kStageBytes + 8, static_cast<u16>(g_rng.chance(0.3) ? 0 : g_rng.range(1, 9)));
    store<u8>(c + kCeUsed, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    store<u8>(c + kCeCounted, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    store<u8>(c + kGlobalDirty, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    store<u8>(c + kPrefetchOn, static_cast<u8>(g_rng.chance(0.5) ? 0 : 1));
    for (Script& s : g_script) {
        s.dirty = static_cast<u8>(g_rng.u32());
        s.counted = static_cast<u8>(g_rng.chance(0.5) ? 0 : 1);
        s.used = static_cast<u8>(g_rng.chance(0.4) ? 1 : 0);
        s.reserve_dwords = static_cast<u8>(g_rng.chance(0.3) ? 0 : g_rng.range(1, 24));
    }
}

// The buffer: empty, or holding a reservation's payload (the flip pass's
// constants), with the memory around it random.
void generate_buffer() {
    for (u8& b : g_s.mem.below) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_s.mem.chunk) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_s.mem.above) b = static_cast<u8>(g_rng.u32());
    store<u64>(ccb(), chunk());
    store<u64>(ccb() + 0x8, chunk() + sizeof g_s.mem.chunk);
    store<u64>(ccb() + 0x10, chunk());
    store<u64>(ccb() + 0x18, reinterpret_cast<u64>(&refill));
    store<u64>(ccb() + 0x20, 0x1234);
    for (int k = 0x28; k < 0x40; k += 8) store<u64>(ccb() + k, (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32());
    g_payloads.clear();
    if (g_rng.chance(0.5)) reserve(static_cast<u32>(g_rng.range(1, 64)));
}

int g_failed = 0;
bool g_quiet = false;
std::map<std::string, int> g_by_reason;  // who: why, for one mistake's run
void fail(const char* who, int t, int draw, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d, draw %d: %s\n", who, t, draw, why);
    ++g_by_reason[std::string(who) + ": " + (t < 0 || draw >= 0 ? why : "memory unlike the game's pre-draw's")];
}

// One draw's checks, against the memory before it.
void check(const char* who, int t, int d, const Memory& before, u64 wait_at, u32 diff, u8 used_at_end,
           const std::vector<Payload>& payloads_before) {
    if (std::memcmp(g_s.mem.below, before.below, sizeof before.below) != 0) fail(who, t, d, "memory below the buffer written");
    if (std::memcmp(g_s.mem.above, before.above, sizeof before.above) != 0) fail(who, t, d, "memory past the buffer written");
    for (const Payload& p : payloads_before) {
        const std::size_t off = static_cast<std::size_t>(p.at - chunk());
        if (std::memcmp(g_s.mem.chunk + off, before.chunk + off, 4ull * p.dwords) != 0) fail(who, t, d, "a reservation's payload written");
    }
    const u64 wp = load<u64>(ccb() + 0x10);
    if (wp < wait_at + 8) {
        fail(who, t, d, "the write pointer not past the wait");
        return;
    }
    const u32 h = load<u32>(wait_at), b = load<u32>(wait_at + 4);
    if (used_at_end) {
        if (h != kWaitHeader || b != diff) fail(who, t, d, "the wait not left in place when the engine was used");
    } else {
        if (h != kNopHeader || b != 0) fail(who, t, d, "no NOP over the wait when the engine stayed unused");
    }
    const std::size_t end = static_cast<std::size_t>(wp - chunk());
    if (std::memcmp(g_s.mem.chunk + end, before.chunk + end, sizeof before.chunk - end) != 0) fail(who, t, d, "written past the write pointer");
    if (g_refilled) fail(who, t, d, "a refill with room to spare");
}

// A run of draws on one buffer by `pre_draw`; the property checks after each.
using PreDraw = GUEST_ABI void (*)(u64);
void run_draws(const char* who, PreDraw pre_draw, int t, int draws, u64 seed) {
    eboot_kit::Rng saved = g_rng;
    g_rng = eboot_kit::Rng(seed);
    for (int d = 0; d < draws; ++d) {
        if (d) generate_cue();  // the next draw's engine state, the same buffer
        const Memory before = g_s.mem;
        const std::vector<Payload> payloads = g_payloads;
        const u64 wait_at = load<u64>(ccb() + 0x10);
        const u32 diff = load<u32>(cue() + kDiff) >> 2;
        g_applies = 0;
        g_last_used = load<u8>(cue() + kCeUsed);
        g_refilled = false;
        pre_draw(cue());
        check(who, t, d, before, wait_at, diff, g_last_used, payloads);
    }
    g_rng = saved;
}

long run(int cases) {
    long n = 0;
    const PreDraw game = eboot_kit::fn<PreDraw>(0x1492e80);
    const PreDraw ours = &gnmx_cue_pre_draw;
    for (int t = 0; t < cases; ++t) {
        const bool dead = g_rng.chance(0.5);
        gnm_ce_configure(dead);
        const int draws = static_cast<int>(g_rng.range(1, 4));
        const u64 seed = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
        g_payloads.clear();
        generate_cue();
        generate_buffer();
        g_start = g_s;
        const std::vector<Payload> payloads = g_payloads;
        Script script[16];
        std::memcpy(script, g_script, sizeof script);
        run_draws("the game's pre-draw", game, t, draws, seed);
        g_game = g_s;
        g_s = g_start;
        g_payloads = payloads;
        std::memcpy(g_script, script, sizeof script);
        run_draws("gnmx_cue_pre_draw", ours, t, draws, seed);
        if (std::memcmp(&g_s, &g_game, sizeof g_s) != 0) {
            const u8* a = reinterpret_cast<const u8*>(&g_s);
            const u8* b = reinterpret_cast<const u8*>(&g_game);
            std::size_t k = 0;
            while (a[k] == b[k]) ++k;
            char why[160];
            std::snprintf(why, sizeof why, "memory unlike the game's pre-draw's, first at +0x%zx (the chunk at +0x%zx, the object +0x%zx, the engine +0x%zx)", k,
                          offsetof(State, mem) + offsetof(Memory, chunk), offsetof(State, ccb), offsetof(State, cue));
            fail("gnmx_cue_pre_draw", t, -1, why);
        }
        gnm_ce_configure(false);
        n += draws;
    }
    return n;
}

bool caught(int ce, int cue) {
    const int before = g_failed;
    g_ce_mutant = ce;
    g_cue_mutant = cue;
    g_quiet = true;
    run(600);
    g_quiet = false;
    g_ce_mutant = g_cue_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gnmx_cue_ccb_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 6000;
    // The constant engine's writers ours, as bbhost places them; the game's
    // NOP writer (0x15fd4a0) as it is; the rest stubs.
    eboot_kit::stub(kCeWait, reinterpret_cast<const void*>(&gnm_ce_wait_on_de_counter_diff));
    eboot_kit::stub(kCeIncrement, reinterpret_cast<const void*>(&gnm_ce_increment_ce_counter));
    eboot_kit::stub(kApplyStage, reinterpret_cast<const void*>(&apply));
    for (const u64 a : {kBuildPsInputs, kSetPsInputs, kPrefetch, kWriteData, kAcquireMem, kDeWaitOnCe, kDeIncrement})
        eboot_kit::stub(a, reinterpret_cast<const void*>(&nothing));
    const u8* at = eboot_kit::file_at("gnmx_cue_ccb_test", 0x15fd4a0);
    // push rbp; mov rbp, rsp; push r14; push rbx; mov r14d, esi; mov rbx, rdi
    static const u8 kNopEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb};
    if (!at || std::memcmp(at, kNopEntry, sizeof kNopEntry) != 0) fail("the game's NOP writer", -1, -1, "the entry's instructions");
    // decomp_gnmx_cue_ccb N ce1|cue2|cue4: one mistake's failures shown, nothing else run.
    if (argc > 2) {
        const std::string m = argv[2];
        g_ce_mutant = m == "ce1" ? 1 : 0;
        g_cue_mutant = m == "cue2" ? 2 : m == "cue4" ? 4 : 0;
        g_quiet = true;
        run(cases);
        for (const auto& [why, n] : g_by_reason) std::printf("  %6d  %s\n", n, why.c_str());
        std::printf("gnmx_cue_ccb_test: mutant %s - %d failures\n", argv[2], g_failed);
        return 0;
    }
    const long total = run(cases);
    struct Mutant {
        int ce, cue;
        const char* what;
    };
    const Mutant mutants[] = {{1, 0, "the wait written as nothing while the dead packets are on"},
                              {0, 2, "the NOP 4 bytes off"},
                              {0, 4, "the NOP left out"}};
    for (const Mutant& m : mutants) {
        if (!caught(m.ce, m.cue)) {
            ++g_failed;
            std::printf("FAIL: mutant not caught: %s\n", m.what);
        }
    }
    if (g_failed) {
        std::printf("gnmx_cue_ccb_test: %d failures (%ld draws)\n", g_failed, total);
        return 1;
    }
    std::printf("gnmx_cue_ccb_test: ok - %ld draws in %d runs, the game's pre-draw and ours, 3 mutants caught\n", total, cases);
    return 0;
}
