// Gnm's DMA_DATA writer (decomp/gx/dma.cpp) beside the game's own
// (tests/decomp/eboot_kit.h): on generated command buffers and arguments -
// room to spare, exactly enough, short, none, past 32 bits, a refill that
// gives a chunk and one that gives nothing, every select's bits - ours must
// leave everything as the game's leaves it: the return value, the command
// buffer object, both chunks and the refill's calls. Then what bbhost runs it
// with: the two GDS moves writing nothing (and only those two callers), and
// the census's ring of callers. And the entry holds the instructions written
// against. Skips without the 1.09 eboot.
#include "decomp/gx/dma.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0xd3a0d3a);

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

u32 random_u32() {
    switch (g_rng.range(0, 9)) {
        case 0: return 0;
        case 1: return 0xffffffffu;
        case 2: return static_cast<u32>(g_rng.range(0, 15));
        case 3: return 0x80000000u | static_cast<u32>(g_rng.range(0, 15));
        default: return g_rng.u32();
    }
}
u64 high(u32 v) { return (static_cast<u64>(g_rng.u32()) << 32) | v; }
u64 random_u64() { return g_rng.chance(0.2) ? static_cast<u64>(random_u32()) : (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32(); }

struct Args {
    u64 r[6];
    u64 s0;
};

void random_buffer() {
    for (u8& b : g_mem.object) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.chunk) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.next) b = static_cast<u8>(g_rng.u32());
    const u32 need = 7;
    const u64 off = static_cast<u64>(g_rng.range(0, static_cast<int>(sizeof g_mem.chunk - (need + 2) * 4))) & ~3ull;
    const u64 wp = reinterpret_cast<u64>(g_mem.chunk) + off;
    u64 room;
    switch (g_rng.range(0, 6)) {
        case 0: room = 0; break;
        case 1: room = need - 1; break;
        case 2: room = need; break;
        case 3: room = need + 1; break;
        case 4: room = static_cast<u64>(g_rng.range(0, 300)); break;
        case 5: room = static_cast<u64>(g_rng.u32()) + 0x100000000ull * g_rng.range(0, 1); break;
        default: room = 0x100000000ull * static_cast<u64>(g_rng.range(1, 3)) + static_cast<u64>(g_rng.range(0, need - 1)); break;
    }
    store<u64>(cb() + 0x8, wp + room * 4 + static_cast<u64>(g_rng.range(0, 3)));
    store<u64>(cb() + 0x10, wp);
    store<u64>(cb() + 0x18, reinterpret_cast<u64>(&stub_refill));
    store<u64>(cb() + 0x20, high(g_rng.u32()));
    g_refill_gives = g_rng.chance(0.75);
    g_refill_result = g_refill_gives ? ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu) | 1u)
                                     : ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu));
}

using Fn = u64(GUEST_ABI*)(u64, u32, u64, u32, u64, u32, u32);
u64 call(bool game, const Args& a) {
    const Fn f = game ? eboot_kit::fn<Fn>(kEntry) : &gnm_dma_data;
    return f(a.r[0], static_cast<u32>(a.r[1]), a.r[2], static_cast<u32>(a.r[3]), a.r[4], static_cast<u32>(a.r[5]), static_cast<u32>(a.s0));
}

int g_failed = 0;
void fail(int t, const char* what) {
    if (++g_failed <= 40) std::printf("FAIL: case %d: %s\n", t, what);
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_dma_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 20000;
    const u8* at = eboot_kit::file_at("gx_dma_test", kEntry);
    if (!at || std::memcmp(at, kEntryBytes, sizeof kEntryBytes) != 0) fail(-1, "the entry's instructions");
    for (const GdsMove& m : kGdsMoves) {
        const u8* p = eboot_kit::file_at("gx_dma_test", m.call);
        if (!p || std::memcmp(p, m.bytes, 5) != 0) fail(-1, "a GDS move's call");
        // call rel32 to the writer
        s32 rel;
        std::memcpy(&rel, m.bytes + 1, 4);
        if (m.call + 5 + static_cast<u64>(static_cast<s64>(rel)) != kEntry) fail(-1, "a GDS move's call is not to the writer");
    }

    int refilled = 0, refused = 0;
    for (int t = 0; t < cases; ++t) {
        Args a{};
        a.r[0] = cb();
        a.r[1] = high(random_u32());
        a.r[2] = random_u64();
        a.r[3] = high(random_u32());
        a.r[4] = random_u64();
        a.r[5] = high(random_u32());
        a.s0 = high(random_u32());
        random_buffer();
        g_saved = g_mem;
        g_calls.clear();
        const u64 game_ret = call(true, a);
        g_game = g_mem;
        const std::vector<RefillCall> game_calls = g_calls;
        if (!game_calls.empty()) ++(g_refill_gives ? refilled : refused);
        for (int how = 0; how < 2; ++how) {
            g_mem = g_saved;
            g_calls.clear();
            const u64 ours = how == 0 ? call(false, a)
                                      : write_dma(a.r[0], static_cast<u32>(a.r[1]), a.r[2], static_cast<u32>(a.r[3]), a.r[4],
                                                  static_cast<u32>(a.r[5]), static_cast<u32>(a.s0));
            if (ours != game_ret) fail(t, "the return value");
            if (std::memcmp(&g_mem, &g_game, sizeof g_mem) != 0) fail(t, "memory");
            if (g_calls != game_calls) fail(t, "the refill's calls");
        }
    }
    std::printf("  gnm_dma_data  %d cases against the game's (%d with a refill, %d the refill refused)\n", cases, refilled, refused);

    // The two GDS moves write nothing - room asked for none, the write
    // pointer returned - and only for those two return addresses.
    g_gds_dead = 1;
    g_gds_return[0] = 0x1111;
    g_gds_return[1] = 0x2222;
    for (int t = 0; t < 2000; ++t) {
        random_buffer();
        g_saved = g_mem;
        const u64 wp = load<u64>(cb() + 0x10);
        g_calls.clear();
        const u64 r = dma_from(t & 1 ? 0x2222 : 0x1111, cb(), 1, 2, 3, 4, 5, 0);
        if (r != wp || std::memcmp(&g_mem, &g_saved, sizeof g_mem) != 0 || !g_calls.empty()) fail(t, "a GDS move wrote something");
        g_mem = g_saved;
        const u64 other = dma_from(0x3333, cb(), 1, 2, 3, 4, 5, 0);
        g_mem = g_saved;
        g_calls.clear();
        if (other != write_dma(cb(), 1, 2, 3, 4, 5, 0)) fail(t, "another caller's DMA did not write");
    }
    g_gds_dead = 0;

    // The census's ring: each packet's caller, as a Binary Ninja address.
    g_census = 1;
    random_buffer();
    store<u64>(cb() + 0x8, load<u64>(cb() + 0x10) + 4096);  // room
    const u64 packet = dma_from(g_decomp_slide + 0x2ab2400 - 0x400000, cb(), 1, 2, 3, 4, 5, 0);
    if (gnm_dma_caller(packet) != 0x2ab2400) fail(-1, "the census ring's caller");
    if (gnm_dma_caller(packet + 4) != 0) fail(-1, "the census ring at another packet");
    g_census = 0;

    std::printf("gx_dma: %s\n", g_failed ? "FAILED" : "ok");
    return g_failed ? 1 : 0;
}
