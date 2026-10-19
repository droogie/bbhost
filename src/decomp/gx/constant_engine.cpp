// Gnm's constant-engine writers as our source (docs/decomp.md): the
// constant command buffer's packets - its RAM written, its counters stepped,
// its RAM dumped into memory for the shaders to read - which Gnmx's constant
// update engine and YEBIS's draws through it write:
//
// - gnm_ce_reserve_const_ram (0x15fd3e0): room for a WRITE_CONST_RAM of
//   `dwords` at `offset` of constant RAM, the payload's address returned for
//   the caller to fill (0 for an offset or size out of range) - Gnmx's
//   constant update engine, three a flip;
// - gnm_ce_wait_on_de_counter_diff (0x15fd230), gnm_ce_increment_ce_counter
//   (0x15fd190): the constant engine's side of the two engines' counters -
//   one a flip, and YEBIS's 75 a session;
// - gnm_ce_dump_const_ram (0x15fd280): DUMP_CONST_RAM - none in a tour.
//
// Nothing dumps constant RAM any more: YEBIS's resource tables come from its
// own state (engine/yebis_bind), and Gnmx's draws are tokens, whose predraw
// flush - the dump - never runs. So with the dead packets on, every dispatch
// and draw a token and YEBIS's binder ours (hle_gx_trace_arm), the counter
// step writes nothing and returns the write pointer.
//
// The wait writes the game's packet always: its callers rewind over it. Each
// of the five - Gnmx's pre-draw (0x1492e80, the flip pass's) and pre-dispatch
// (0x1493180), the Gnm context's pre-draw (0x148cb70), YEBIS's commit of a
// draw (0x15f9720) and of a dispatch (0x15f9d20) - copies the command
// buffer's header right after the wait and, when the draw left the constant
// engine unused, writes a two-dword NOP through the copy 8 bytes below its
// write pointer: over the wait. A wait written as nothing (from PR #149 on)
// put that NOP over whatever lay below it: the last two dwords of a
// reservation's payload, or, in an empty constant command buffer, the heap
// block's header in front of it - the flip pass's two contexts are
// back-to-back blocks, and a live resize's YEBIS shutdown crashed in free
// (0x24859c7). The counters the wait compares are never stepped while so
// (this step, the draw engine's - decomp/gx/labels.cpp - and the only other
// constant-engine step, 0x15fd1e0, which nothing reachable calls), so it
// never holds the constant engine (gnm_exec's ce_step). No other writer that
// writes nothing has a caller that moves the write pointer back, reads what
// the writer returned, keeps the write pointer across the call or writes
// through a copy of the command buffer (decomp/gx/dead_packets.cpp).
//
// The reserve writes the game's packet always: its payload is the guest's
// memory in the constant command buffer, whose address the constant update
// engine keeps per chunk and fills over several calls, and draws read the
// bytes there - a host scratch area in its place (one, for every
// reservation) washed the world out, the fog and the lights reading through
// it. The dump writes the game's packet always too, and a dump while the
// counters write nothing is counted and logged - its table would be stale.
//
// Each is the game's function as source, instruction for instruction (the
// reserve's checks of offset and size, the room check on the low 32 bits of
// the dwords left, the refill, the packet, the write pointer, the return
// value). Leaves: entered straight from the game's code, they touch only the
// game's memory and their own globals; BBHOST_DECOMP_COMPARE=1 runs the
// game's version of each in their place and checks ours against every call
// (hosted); with the writers' census (BBHOST_CP_WRITERS=1) the game's stay.
#include "decomp/gx/constant_engine.h"

#include "core/thunk.h"
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "log.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using namespace decomp;

#ifdef DECOMP_CE_MUTANTS
extern int g_ce_mutant;
#define CE_MUTANT(n) (g_ce_mutant == (n))
#else
#define CE_MUTANT(n) false
#endif

namespace {

enum Writer : std::size_t { kReserve, kWaitDiff, kIncrementCe, kDump, kWriters };

struct CeWriter {
    const char* name;
    u64 bn;
    u8 n;
    u8 bytes[17];
};
const CeWriter kTable[kWriters] = {
    // push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx, rax.
    {"gnm_ce_reserve_const_ram", 0x15fd3e0, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50}},
    // push rbp; mov rbp, rsp; push r14, rbx; mov r14d, esi; mov rbx, rdi; mov rcx, [rbx+8].
    {"gnm_ce_wait_on_de_counter_diff", 0x15fd230, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08}},
    // push rbp; mov rbp, rsp; push rbx, rax; mov rbx, rdi; mov rcx, [rbx+8]; mov rax, [rbx+0x10].
    {"gnm_ce_increment_ce_counter", 0x15fd190, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}},
    // push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov r15d, ecx.
    {"gnm_ce_dump_const_ram", 0x15fd280, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xcf}},
};

// Set by gnm_ce_configure; read by the leaves.
u8 g_dead = 0;
std::atomic<u64> g_steps_not_written{0}, g_waits_while_dead{0}, g_dumps_while_dead{0};

using Refill = GUEST_ABI u64 (*)(u64 cb, u32 dwords, u64 arg, u64 room);

// The room for `need` dwords; false with the refill's result when it gave
// nothing (the low 32 bits of the dwords left against the packet's size).
__attribute__((always_inline)) inline bool make_room(u64 cb, u32 need, u64* wp, u64* failed) {
    u64 p = load<u64>(cb + 0x10);
    const u64 room = (load<u64>(cb + 0x8) - p) >> 2;
    if (static_cast<u32>(room) < need) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u64 r = refill(cb, need, load<u64>(cb + 0x20), room);
        if (static_cast<u8>(r) == 0) {
            *failed = r;
            return false;
        }
        p = load<u64>(cb + 0x10);
    }
    *wp = p;
    return true;
}

// The reserve's checks: a 4-aligned offset at or below 0xbffc, at most
// 0x3000 dwords, ending at or below 0xc000.
inline bool reserve_ok(u32 offset, u32 dwords) {
    return (offset & 0xfffc) == offset && (offset & 0xffff) <= 0xbfff && dwords <= 0x3000 &&
           (offset & 0xffff) + 4 * static_cast<u64>(dwords) <= 0xc000;
}

// The game's functions, written out.

// WRITE_CONST_RAM reserved: its header (the payload's dwords in the count)
// and the offset, the write pointer moved past the payload, the payload's
// address returned; 0 for arguments out of range and when the refill gives
// nothing (not the refill's result).
inline u64 w_reserve(u64 cb, u32 offset, u32 dwords) {
    if (!reserve_ok(offset, dwords)) return 0;
    const u32 need = dwords + 2;
    u64 wp = load<u64>(cb + 0x10);
    const u64 room = (load<u64>(cb + 0x8) - wp) >> 2;
    if (static_cast<u32>(room) < need) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        if (static_cast<u8>(refill(cb, need, load<u64>(cb + 0x20), room)) == 0) return 0;
        wp = load<u64>(cb + 0x10);
    }
    store<u32>(wp, (dwords << 16) | 0xc0008100u);
    store<u32>(wp + 4, offset);
    const u64 p = load<u64>(cb + 0x10);
    store<u64>(cb + 0x10, p + 4 * static_cast<u64>(need));
    return p + 8;
}

inline u64 w_wait_diff(u64 cb, u32 diff) {
    u64 wp, failed;
    if (!make_room(cb, 2, &wp, &failed)) return failed;
    store<u32>(wp, 0xc0008800u);
    store<u32>(wp + 4, diff);
    store<u64>(cb + 0x10, wp + 8);
    return wp;
}

inline u64 w_increment_ce(u64 cb) {
    u64 wp, failed;
    if (!make_room(cb, 2, &wp, &failed)) return failed;
    store<u64>(wp, 0x00000001c0008400ull);
    store<u64>(cb + 0x10, wp + 8);
    return wp;
}

// DUMP_CONST_RAM of `dwords` at `offset` to `dest` (dword aligned): a size
// out of range dumps nothing (0 dwords), the packet written all the same.
inline u64 w_dump(u64 cb, u64 dest, u32 offset, u32 dwords) {
    const u64 bytes = 4 * static_cast<u64>(dwords);
    if ((offset & 0xffff) > 0xbfff || bytes > 0xc000 || bytes + (offset & 0xffff) >= 0xc001) dwords = 0;
    u64 wp, failed;
    if (!make_room(cb, 5, &wp, &failed)) return failed;
    store<u32>(wp, 0xc0038300u);
    store<u32>(wp + 4, offset);
    store<u32>(wp + 8, dwords & 0x7fff);
    store<u32>(wp + 12, static_cast<u32>(dest) & 0xfffffffcu);
    store<u32>(wp + 16, static_cast<u32>(dest >> 32));
    store<u64>(cb + 0x10, wp + 0x14);
    return wp;
}

}  // namespace

DECOMP_LEAF u64 gnm_ce_reserve_const_ram(u64 cb, u32 offset, u32 dwords) {
    return w_reserve(cb, offset, dwords);  // the game's packet whatever the mode (above)
}
DECOMP_LEAF u64 gnm_ce_wait_on_de_counter_diff(u64 cb, u32 diff) {
    if (g_dead) {
        if (CE_MUTANT(1)) return load<u64>(cb + 0x10);  // written as nothing, as PR #149 had it
        g_waits_while_dead.fetch_add(1, std::memory_order_relaxed);
    }
    return w_wait_diff(cb, diff);  // the game's packet whatever the mode: its callers NOP over it (above)
}
DECOMP_LEAF u64 gnm_ce_increment_ce_counter(u64 cb) {
    if (g_dead) {
        g_steps_not_written.fetch_add(1, std::memory_order_relaxed);
        return load<u64>(cb + 0x10);
    }
    return w_increment_ce(cb);
}
DECOMP_LEAF u64 gnm_ce_dump_const_ram(u64 cb, u64 dest, u32 offset, u32 dwords) {
    if (g_dead) g_dumps_while_dead.fetch_add(1, std::memory_order_relaxed);
    return w_dump(cb, dest, offset, dwords);
}

namespace {

struct Args {
    u64 r[6];
};

u64 write_as_game(std::size_t i, const Args& a) {
    const u64 cb = a.r[0];
    const auto d = [&](int k) { return static_cast<u32>(a.r[k]); };
    switch (i) {
        case kReserve: return w_reserve(cb, d(1), d(2));
        case kWaitDiff: return w_wait_diff(cb, d(1));
        case kIncrementCe: return w_increment_ce(cb);
        case kDump: return w_dump(cb, a.r[1], d(2), d(3));
        default: return 0;
    }
}

// The dwords the game's version writes at the write pointer (0: none).
u64 written(std::size_t i, const Args& a) {
    switch (i) {
        case kReserve: return reserve_ok(static_cast<u32>(a.r[1]), static_cast<u32>(a.r[2])) ? 2 : 0;  // the header; the caller fills the rest
        case kWaitDiff: case kIncrementCe: return 2;
        case kDump: return 5;
        default: return 0;
    }
}
u32 room_for(std::size_t i, const Args& a) {
    return i == kReserve ? static_cast<u32>(a.r[2]) + 2 : static_cast<u32>(written(i, a));
}

// BBHOST_DECOMP_COMPARE=1: the game's version on every call, ours beside it
// on the same command buffer - the bytes written, the write pointer and the
// return value compared - then the game's result left in place. A call the
// refill would serve runs the game's alone.
DecompCompare g_counts[kWriters];
void* g_original[kWriters] = {};
std::atomic<u64> g_compare_alone{0};
std::atomic<int> g_compare_logs{0};

u64 compare(std::size_t i, const Args& a) {
    const u64 cb = a.r[0];
    GuestCallRegs g{};
    g.fn = g_original[i];
    for (int k = 0; k < 6; ++k) g.ints[k] = a.r[k];
    const u64 wp = load<u64>(cb + 0x10);
    const std::size_t bytes = static_cast<std::size_t>(written(i, a)) * 4;
    if (static_cast<u32>((load<u64>(cb + 0x8) - wp) >> 2) < room_for(i, a)) {
        g_compare_alone.fetch_add(1, std::memory_order_relaxed);
        hle_call_guest_regs(&g);
        return g.ret;
    }
    u8 before[20], game[20];
    std::memcpy(before, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), bytes);
    hle_call_guest_regs(&g);
    const u64 game_ret = g.ret, game_wp = load<u64>(cb + 0x10);
    std::memcpy(game, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), bytes);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), before, bytes);
    store<u64>(cb + 0x10, wp);
    const u64 our_ret = write_as_game(i, a);
    const u64 our_wp = load<u64>(cb + 0x10);
    const bool same = our_ret == game_ret && our_wp == game_wp &&
                      std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), game, bytes) == 0;
    g_counts[i].calls.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        g_counts[i].differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            host_log("decomp: %s differs from the game's: returned 0x%llx (game 0x%llx), write pointer +%lld (game +%lld)", kTable[i].name,
                     static_cast<ull>(our_ret), static_cast<ull>(game_ret), static_cast<long long>(our_wp - wp),
                     static_cast<long long>(game_wp - wp));
        }
    }
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), game, bytes);
    store<u64>(cb + 0x10, game_wp);
    return game_ret;
}

template <std::size_t I>
GUEST_ABI u64 compare_writer(u64 r0, u64 r1, u64 r2, u64 r3, u64 r4, u64 r5) {
    return compare(I, Args{{r0, r1, r2, r3, r4, r5}});
}
template <std::size_t... I>
std::array<void*, kWriters> compare_table(std::index_sequence<I...>) {
    return {reinterpret_cast<void*>(&compare_writer<I>)...};
}

void report() {
    static bool once = false;
    if (once) return;
    once = true;
    if (decomp_placed(kTable[0].name) == 2) {
        host_log("decomp: the constant engine's writers: compared (BBHOST_DECOMP_COMPARE=1), %llu calls ran the game's alone",
                 static_cast<ull>(g_compare_alone.load()));
        return;
    }
    if (!g_dead) {
        host_log("decomp: the constant engine's writers write the game's packets");
        return;
    }
    host_log("decomp: the constant engine's counter steps not written - nothing dumps its RAM: %llu steps; its waits on the draw engine "
             "(%llu, which the game NOPs over when a draw leaves the engine unused) and its RAM writes are the game's packets; constant RAM "
             "dumps while so: %llu",
             static_cast<ull>(g_steps_not_written.load()), static_cast<ull>(g_waits_while_dead.load()),
             static_cast<ull>(g_dumps_while_dead.load()));
}

}  // namespace

extern u8 g_ce_counters_dead;  // decomp/gx/labels.cpp: the draw engine's side of the counters
void gnm_ce_configure(bool dead) {
    g_dead = dead ? 1 : 0;
    g_ce_counters_dead = g_dead;
}

void decomp_gx_constant_engine_add() {
    if (const char* e = std::getenv("BBHOST_CP_WRITERS"); e && e[0] == '1') return;
    static void* const kOurs[kWriters] = {
        reinterpret_cast<void*>(&gnm_ce_reserve_const_ram),
        reinterpret_cast<void*>(&gnm_ce_wait_on_de_counter_diff),
        reinterpret_cast<void*>(&gnm_ce_increment_ce_counter),
        reinterpret_cast<void*>(&gnm_ce_dump_const_ram),
    };
    static const std::array<void*, kWriters> kCompare = compare_table(std::make_index_sequence<kWriters>{});
    for (std::size_t i = 0; i < kWriters; ++i) {
        const CeWriter& w = kTable[i];
        DecompFunction fn{w.name, "gx", w.bn, w.bytes, w.n, kOurs[i], DecompKind::Leaf};
        fn.original = &g_original[i];
        fn.compare = kCompare[i];
        fn.counts = &g_counts[i];
        fn.report = &report;
        decomp_add(fn);
    }
}
