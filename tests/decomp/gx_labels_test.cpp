// Gnm's label, wait, inline-data and predication writers
// (decomp/gx/labels.cpp) beside the game's own (tests/decomp/eboot_kit.h). For
// each of the nine, on generated command buffers and arguments - room to
// spare, exactly enough, one dword short, none, past 32 bits, a refill that
// gives a chunk and one that gives nothing, every argument's bits (the
// end-of-shader event 0x2f and its look-alikes for RELEASE_MEM, addresses
// either side of 0xffff for the waits, WRITE_DATA payloads of every size up to
// 200 dwords, apart from the packet or overlapping it from either side, at
// any alignment) - ours must leave everything as the game's leaves it: the
// return value, the command buffer object (its flags too), both chunks' bytes
// and the refill's calls with their arguments. And the entries the decomp
// list places them at hold the instructions written against. Skips without
// the 1.09 eboot.
#include "decomp/gx/labels.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x1abe15);

// A command buffer as Gnm's are: an object {+0x8 the chunk's end, +0x10 the
// write pointer, +0x18 the refill, +0x20 its argument, +0x28 its flags} over a
// chunk, and the chunk the refill hands over.
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
        case 2: return static_cast<u32>(g_rng.range(0, 63));
        case 3: return 0x80000000u | static_cast<u32>(g_rng.range(0, 7));
        default: return g_rng.u32();
    }
}

u64 random_u64() {
    switch (g_rng.range(0, 7)) {
        case 0: return static_cast<u64>(g_rng.range(0, 0x1ffff));  // registers and just past them
        case 1: return 0xffffull;
        case 2: return 0x10000ull;
        case 3: return 0;
        case 4: return ~0ull;
        default: return (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32();
    }
}

// Garbage above the 32 bits a register carries a u32 in.
u64 high(u32 v) { return (static_cast<u64>(g_rng.u32()) << 32) | v; }

alignas(64) u8 g_data[1024];  // where an apart payload comes from

// Where a WRITE_DATA's payload ends up, and how many dwords: decided before
// the buffer, which leaves room for it.
u32 g_payload = 0;

Args random_args(std::size_t i) {
    Args a{};
    a.r[0] = cb();
    for (int k = 1; k < 6; ++k) a.r[k] = high(random_u32());
    for (u64& s : a.s) s = high(random_u32());
    switch (i) {
        case kEndOfPipe:
        case kReleaseMem: {
            static const u32 kEvents[] = {0x2f, 0x12f, 0x8000002f, 0x2e, 0x14, 0x28, 0x3f, 0x40};
            if (g_rng.chance(0.4)) a.r[1] = high(kEvents[g_rng.range(0, static_cast<int>(std::size(kEvents)) - 1)]);
            a.r[3] = random_u64();
            a.r[5] = random_u64();
            break;
        }
        case kEndOfShader: a.r[2] = random_u64(); break;
        case kWait:
        case kComputeWait: a.r[1] = random_u64(); break;
        case kSetPredication:
        case kIndirectBuffer: a.r[1] = random_u64(); break;
        case kWriteData:
        case kWriteDataL2: {
            a.r[1] = random_u64();
            static const u32 kSizes[] = {0, 1, 15, 16, 17, 31, 32, 33, 48, 64, 100, 200};
            g_payload = g_rng.chance(0.6) ? static_cast<u32>(g_rng.range(0, 40))
                                          : kSizes[g_rng.range(0, static_cast<int>(std::size(kSizes)) - 1)];
            a.r[3] = high(g_payload);
            for (u8& b : g_data) b = static_cast<u8>(g_rng.u32());
            a.r[2] = reinterpret_cast<u64>(g_data) + static_cast<u64>(g_rng.range(0, 160));  // unaligned too
            break;
        }
        default: break;
    }
    return a;
}

// The command buffer for one case: random bytes everywhere, the write pointer
// somewhere in the chunk with room for the packet, and the room around the
// packet's size - or none - with up to three bytes over a dword. A WRITE_DATA
// payload may then be moved to lie across the packet's place.
void random_buffer(std::size_t i, Args& a) {
    for (u8& b : g_mem.object) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.chunk) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.next) b = static_cast<u8>(g_rng.u32());
    const u32 need = room_for(i, a);
    const u64 span = (static_cast<u64>(need) + 2) * 4;
    const u64 off = static_cast<u64>(g_rng.range(0, static_cast<int>(sizeof g_mem.chunk - span))) & ~3ull;
    const u64 wp = reinterpret_cast<u64>(g_mem.chunk) + off;
    u64 room;
    switch (g_rng.range(0, 6)) {
        case 0: room = 0; break;
        case 1: room = need - 1; break;
        case 2: room = need; break;
        case 3: room = static_cast<u64>(need) + 1; break;
        case 4: room = static_cast<u64>(g_rng.range(0, 300)); break;
        case 5: room = static_cast<u64>(g_rng.u32()) + 0x100000000ull * g_rng.range(0, 1); break;  // past 32 bits
        default:  // past 32 bits with the low 32 short: the game asks the refill, and passes all 64
            room = 0x100000000ull * static_cast<u64>(g_rng.range(1, 3)) + static_cast<u64>(g_rng.range(0, static_cast<int>(need) - 1));
            break;
    }
    store<u64>(cb() + 0x8, wp + room * 4 + static_cast<u64>(g_rng.range(0, 3)));
    store<u64>(cb() + 0x10, wp);
    store<u64>(cb() + 0x18, reinterpret_cast<u64>(&stub_refill));
    store<u64>(cb() + 0x20, high(g_rng.u32()));
    g_refill_gives = g_rng.chance(0.75);
    g_refill_result = g_refill_gives ? ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu) | 1u)
                                     : ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu));
    if (has_payload(i) && g_payload && g_rng.chance(0.4)) {
        // Across the packet's place: from before its payload to after it, a
        // few bytes off a dword at times, inside the chunk.
        const u64 dst = wp + 0x10;
        const s64 shift = static_cast<s64>(g_rng.range(-static_cast<int>(g_payload) * 4 - 16, static_cast<int>(g_payload) * 4 + 16));
        const s64 lo = static_cast<s64>(reinterpret_cast<u64>(g_mem.chunk));
        const s64 hi = static_cast<s64>(reinterpret_cast<u64>(g_mem.chunk) + sizeof g_mem.chunk) - static_cast<s64>(g_payload) * 4;
        s64 src = static_cast<s64>(dst) + shift;
        if (src < lo) src = lo;
        if (src > hi) src = hi;
        a.r[2] = static_cast<u64>(src);
    }
}

// The game's function (`game`) or ours, with the arguments in their registers.
u64 call(std::size_t i, bool game, const Args& a) {
    const u64 c = a.r[0];
    const auto d = [&](int k) { return static_cast<u32>(a.r[k]); };
    const auto s = [&](int k) { return static_cast<u32>(a.s[k]); };
    const u64 bn = kTable[i].bn;
#define CALL(F, ours, ...) (game ? eboot_kit::fn<u64(GUEST_ABI*) F>(bn)(__VA_ARGS__) : ours(__VA_ARGS__))
    switch (i) {
        case kEndOfPipe:
            return CALL((u64, u32, u32, u64, u32, u64, u32, u32), gnm_write_end_of_pipe_event, c, d(1), d(2), a.r[3], d(4), a.r[5], s(0), s(1));
        case kEndOfShader: return CALL((u64, u32, u64, u32), gnm_write_end_of_shader_event, c, d(1), a.r[2], d(3));
        case kReleaseMem:
            return CALL((u64, u32, u32, u64, u32, u64, u32, u32), gnm_dispatch_write_release_mem, c, d(1), d(2), a.r[3], d(4), a.r[5], s(0),
                        s(1));
        case kWait: return CALL((u64, u64, u32, u32, u32), gnm_wait_reg_mem, c, a.r[1], d(2), d(3), d(4));
        case kComputeWait: return CALL((u64, u64, u32, u32, u32), gnm_dispatch_wait_reg_mem, c, a.r[1], d(2), d(3), d(4));
        case kWriteData: return CALL((u64, u64, u64, u32, u32), gnm_write_data_inline, c, a.r[1], a.r[2], d(3), d(4));
        case kWriteDataL2: return CALL((u64, u64, u64, u32, u32, u32), gnm_write_data_inline_l2, c, a.r[1], a.r[2], d(3), d(4), d(5));
        case kSetPredication: return CALL((u64, u64, u32, u32), gnm_set_predication, c, a.r[1], d(2), d(3));
        case kClearPredication: return CALL((u64), gnm_clear_predication, c);
        case kIndirectBuffer: return CALL((u64, u64, u32), gnm_dispatch_write_indirect_buffer, c, a.r[1], d(2));
        case kIncrementDeCounter: return CALL((u64), gnm_increment_de_counter, c);
        case kWaitOnCeCounter: return CALL((u64), gnm_wait_on_ce_counter, c);
        default: return 0;
    }
#undef CALL
}

int g_failed = 0;
void fail(std::size_t i, int trial, const char* what) {
    if (++g_failed <= 40) std::printf("FAIL: %s, case %d: %s\n", kTable[i].name, trial, what);
}

int run(std::size_t i, int cases) {
    int refilled = 0, refused = 0, overlapping = 0;
    for (int t = 0; t < cases; ++t) {
        Args a = random_args(i);
        random_buffer(i, a);
        if (has_payload(i)) {
            const u64 wp = load<u64>(cb() + 0x10), src = a.r[2], bytes = 4ull * g_payload;
            if (src < wp + 0x10 + bytes && wp + 0x10 < src + bytes) ++overlapping;
        }
        g_saved = g_mem;

        // The game's.
        g_calls.clear();
        const u64 game_ret = call(i, true, a);
        g_game = g_mem;
        const std::vector<RefillCall> game_calls = g_calls;
        if (!game_calls.empty()) ++(g_refill_gives ? refilled : refused);

        // Ours, through its entry on the list - and the same function as the
        // compare runs it, from the registers.
        // The draw engine's counters with the constant engine off: nothing
        // written, no room asked for, the write pointer returned.
        if (i == kIncrementDeCounter || i == kWaitOnCeCounter) {
            g_mem = g_saved;
            g_calls.clear();
            g_ce_counters_dead = 1;
            const u64 r = call(i, false, a);
            g_ce_counters_dead = 0;
            if (r != load<u64>(cb() + 0x10) || std::memcmp(&g_mem, &g_saved, sizeof g_mem) != 0 || !g_calls.empty())
                fail(i, t, "a counter written with the constant engine off");
        }
        for (int how = 0; how < 2; ++how) {
            g_mem = g_saved;
            g_calls.clear();
            const u64 ours_ret = how == 0 ? call(i, false, a) : write_as_game(i, a);
            if (ours_ret != game_ret) fail(i, t, "the return value");
            if (std::memcmp(g_mem.object, g_game.object, sizeof g_mem.object) != 0) fail(i, t, "the command buffer object");
            if (std::memcmp(g_mem.chunk, g_game.chunk, sizeof g_mem.chunk) != 0) fail(i, t, "the chunk's bytes");
            if (std::memcmp(g_mem.next, g_game.next, sizeof g_mem.next) != 0) fail(i, t, "the refill's chunk's bytes");
            if (g_calls != game_calls) fail(i, t, "the refill's calls");
        }
    }
    std::printf("  %-32s %d cases against the game's (%d with a refill, %d the refill refused", kTable[i].name, cases, refilled, refused);
    if (has_payload(i)) std::printf(", %d with the payload across the packet", overlapping);
    std::printf(")\n");
    return cases;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_labels_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 20000;

    // The decomp list places each at its entry only when these bytes are there.
    for (std::size_t i = 0; i < kWriters; ++i) {
        const u8* at = eboot_kit::file_at("gx_labels_test", kTable[i].bn);
        if (!at || std::memcmp(at, kTable[i].bytes, kTable[i].n) != 0) fail(i, -1, "the entry's instructions");
    }

    long total = 0;
    for (std::size_t i = 0; i < kWriters; ++i) total += run(i, cases);
    std::printf("gx_labels: %ld cases over %zu writers: %s\n", total, static_cast<std::size_t>(kWriters), g_failed ? "FAILED" : "ok");
    return g_failed ? 1 : 0;
}
