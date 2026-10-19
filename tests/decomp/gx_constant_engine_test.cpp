// Gnm's constant-engine writers (decomp/gx/constant_engine.cpp) beside the
// game's own (tests/decomp/eboot_kit.h): on generated command buffers and
// arguments - room to spare, exactly enough, short, none, past 32 bits, a
// refill that gives a chunk and one that gives nothing, offsets and sizes in
// and out of the reserve's and the dump's ranges - ours writing the game's
// packets must leave everything as the game's leaves it: the return value,
// the command buffer object, both chunks and the refill's calls. Then what
// bbhost runs them with when nothing dumps constant RAM: the counter's packet
// not written, the wait's, the reserve's and the dump's written as the game's
// (the wait's callers NOP over it - tests/decomp/gnmx_cue_ccb_test.cpp - and
// the reserve's payload is guest memory callers keep and draws read), the
// dump counted.
// And the entries hold the instructions written against. Skips without the
// 1.09 eboot.
#include "decomp/gx/constant_engine.cpp"

// decomp/gx/labels.cpp's, which gnm_ce_configure sets with the constant engine.
u8 g_ce_counters_dead = 0;

#include "eboot_kit.h"

#include <cstdio>

namespace {

eboot_kit::Rng g_rng(0xce0ce0);

struct alignas(64) Memory {
    u8 object[0x40];
    u8 chunk[4096];
    u8 next[4096];
};
Memory g_mem, g_saved, g_game;

struct RefillCall {
    u64 cb;
    u32 dwords;
    u64 arg, room;
    bool operator==(const RefillCall&) const = default;
};
std::vector<RefillCall> g_calls;
bool g_refill_gives = true;
u64 g_refill_result = 0;

u64 cb() { return reinterpret_cast<u64>(g_mem.object); }

GUEST_ABI u64 stub_refill(u64 c, u32 dwords, u64 arg, u64 room) {
    g_calls.push_back({c, dwords, arg, room});
    if (!g_refill_gives) return g_refill_result;
    const u64 next = reinterpret_cast<u64>(g_mem.next);
    store<u64>(c + 0x8, next + sizeof g_mem.next);
    store<u64>(c + 0x10, next);
    return g_refill_result;
}

u64 high(u32 v) { return (static_cast<u64>(g_rng.u32()) << 32) | v; }

Args random_args(std::size_t i) {
    Args a{};
    a.r[0] = cb();
    for (int k = 1; k < 6; ++k) a.r[k] = high(g_rng.u32());
    if (i == kReserve || i == kDump) {
        const int o = i == kReserve ? 1 : 2, d = i == kReserve ? 2 : 3;
        static const u32 kOffsets[] = {0, 4, 0xbffc, 0xbfff, 0xc000, 0xfffc, 0x10000, 2, 0x8000};
        static const u32 kDwords[] = {0, 1, 0x40, 0x3000, 0x3001, 0x2fff, 0x8000, 0xffffffffu};
        const u32 off = g_rng.chance(0.5) ? (static_cast<u32>(g_rng.range(0, 0x2fff)) * 4) : kOffsets[g_rng.range(0, 8)];
        const u32 dw = g_rng.chance(0.6) ? static_cast<u32>(g_rng.range(0, 0x400)) : kDwords[g_rng.range(0, 7)];
        a.r[o] = high(off);
        a.r[d] = high(dw);
        if (i == kDump) a.r[1] = (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
    }
    return a;
}

void random_buffer(u32 need, u32 writes) {
    for (u8& b : g_mem.object) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.chunk) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.next) b = static_cast<u8>(g_rng.u32());
    // The reserve writes only its header: its payload's room is the end
    // pointer's arithmetic, not memory.
    const u64 span = (static_cast<u64>(writes) + 2) * 4;
    const u64 off = static_cast<u64>(g_rng.range(0, static_cast<int>(sizeof g_mem.chunk - span))) & ~3ull;
    const u64 wp = reinterpret_cast<u64>(g_mem.chunk) + off;
    u64 room;
    switch (g_rng.range(0, 6)) {
        case 0: room = 0; break;
        case 1: room = need ? need - 1 : 0; break;
        case 2: room = need; break;
        case 3: room = static_cast<u64>(need) + 1; break;
        case 4: room = static_cast<u64>(g_rng.range(0, 300)); break;
        case 5: room = static_cast<u64>(g_rng.u32()) + 0x100000000ull * g_rng.range(0, 1); break;
        default: room = 0x100000000ull * static_cast<u64>(g_rng.range(1, 3)) + static_cast<u64>(g_rng.range(0, need > 1 ? static_cast<int>(need) - 1 : 0)); break;
    }
    store<u64>(cb() + 0x8, wp + room * 4 + static_cast<u64>(g_rng.range(0, 3)));
    store<u64>(cb() + 0x10, wp);
    store<u64>(cb() + 0x18, reinterpret_cast<u64>(&stub_refill));
    store<u64>(cb() + 0x20, high(g_rng.u32()));
    g_refill_gives = g_rng.chance(0.75);
    g_refill_result = g_refill_gives ? ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu) | 1u)
                                     : ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu));
}

u64 call(std::size_t i, bool game, const Args& a) {
    const u64 c = a.r[0];
    const auto d = [&](int k) { return static_cast<u32>(a.r[k]); };
    const u64 bn = kTable[i].bn;
#define CALL(F, ours, ...) (game ? eboot_kit::fn<u64(GUEST_ABI*) F>(bn)(__VA_ARGS__) : ours(__VA_ARGS__))
    switch (i) {
        case kReserve: return CALL((u64, u32, u32), gnm_ce_reserve_const_ram, c, d(1), d(2));
        case kWaitDiff: return CALL((u64, u32), gnm_ce_wait_on_de_counter_diff, c, d(1));
        case kIncrementCe: return CALL((u64), gnm_ce_increment_ce_counter, c);
        case kDump: return CALL((u64, u64, u32, u32), gnm_ce_dump_const_ram, c, a.r[1], d(2), d(3));
        default: return 0;
    }
#undef CALL
}

int g_failed = 0;
void fail(std::size_t i, int t, const char* what) {
    if (++g_failed <= 40) std::printf("FAIL: %s, case %d: %s\n", kTable[i].name, t, what);
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_constant_engine_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 20000;
    for (std::size_t i = 0; i < kWriters; ++i) {
        const u8* at = eboot_kit::file_at("gx_constant_engine_test", kTable[i].bn);
        if (!at || std::memcmp(at, kTable[i].bytes, kTable[i].n) != 0) fail(i, -1, "the entry's instructions");
    }
    std::memset(&g_mem, 0, sizeof g_mem);
    for (std::size_t i = 0; i < kWriters; ++i) {
        int refilled = 0, refused = 0, invalid = 0;
        for (int t = 0; t < cases; ++t) {
            const Args a = random_args(i);
            if (i == kReserve && !reserve_ok(static_cast<u32>(a.r[1]), static_cast<u32>(a.r[2]))) ++invalid;
            random_buffer(room_for(i, a), static_cast<u32>(written(i, a)));
            g_saved = g_mem;
            g_calls.clear();
            g_dead = 0;
            const u64 game_ret = call(i, true, a);
            g_game = g_mem;
            const std::vector<RefillCall> game_calls = g_calls;
            if (!game_calls.empty()) ++(g_refill_gives ? refilled : refused);
            for (int how = 0; how < 2; ++how) {
                g_mem = g_saved;
                g_calls.clear();
                const u64 ours = how == 0 ? call(i, false, a) : write_as_game(i, a);
                if (ours != game_ret) fail(i, t, "the return value");
                if (std::memcmp(&g_mem, &g_game, sizeof g_mem) != 0) fail(i, t, "memory");
                if (g_calls != game_calls) fail(i, t, "the refill's calls");
            }
            // Not written: nothing touched, no refill.
            g_mem = g_saved;
            g_calls.clear();
            g_dead = 1;
            const u64 dumps = g_dumps_while_dead.load();
            const u64 r = call(i, false, a);
            if (i == kDump) {
                if (r != game_ret || std::memcmp(&g_mem, &g_game, sizeof g_mem) != 0 || g_dumps_while_dead.load() != dumps + 1)
                    fail(i, t, "the dump not written or not counted");
            } else if (i == kReserve || i == kWaitDiff) {
                if (r != game_ret || std::memcmp(&g_mem, &g_game, sizeof g_mem) != 0 || g_calls != game_calls)
                    fail(i, t, i == kReserve ? "the reserve not written as the game's" : "the wait not written as the game's");
            } else {
                if (std::memcmp(&g_mem, &g_saved, sizeof g_mem) != 0 || !g_calls.empty()) fail(i, t, "written while not written");
                if (r != load<u64>(cb() + 0x10)) fail(i, t, "the value returned while not written");
            }
            g_dead = 0;
        }
        std::printf("  %-32s %d cases against the game's (%d with a refill, %d the refill refused%s)\n", kTable[i].name, cases, refilled,
                    refused, i == kReserve ? (", " + std::to_string(invalid) + " out of range").c_str() : "");
    }
    std::printf("gx_constant_engine: %s\n", g_failed ? "FAILED" : "ok");
    return g_failed ? 1 : 0;
}
