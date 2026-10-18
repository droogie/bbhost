// Gnm's DMA_DATA writer as our source (docs/decomp.md): 0x1473e60, through
// which GX's buffer copies and fills, the GPU particles' counter moves through
// GDS, Gnmx's and YEBIS's resets and Toolkit's surface copies have the
// command processor move or fill memory - 21 callers. The game's function
// written out, instruction for instruction: the room check on the low 32 bits
// of the dwords left, the refill (cb, 7 dwords, its argument, the 64-bit
// room), the packet - the source and destination selects split across its
// control and size dwords as the game splits them - the write pointer moved
// and the packet's address returned (the refill's result when it gives
// nothing).
//
// With the GPU particles' counters in the dispatch tokens
// (BBHOST_GX_TOKEN_COUNTERS=2, the default with native dispatches and their
// records), the two moves through GDS around a counting dispatch - the
// compute commit's load, whose call returns to 0x2acc314, and the post's
// store (0x2acd091) - write nothing, and return the write pointer: nothing
// reads the packets or the result. That replaces the two five-byte NOPs
// render/cporder wrote over those calls, the last byte patches of its dead
// packets. With the CP census (BBHOST_CP_CENSUS=1) each packet's caller goes
// into a ring the census reads (gnm_dma_caller), as its prologue hook did.
//
// A leaf: entered straight from the game's code, it touches only the game's
// memory and its own globals. BBHOST_DECOMP_COMPARE=1 runs the game's version
// in its place on every call and checks ours against it (hosted); with the
// writers' census (BBHOST_CP_WRITERS=1) the game's version stays.
#include "decomp/gx/dma.h"

#include "core/thunk.h"
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

using namespace decomp;

namespace {

constexpr u64 kEntry = 0x1473e60;
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx, rax (14 bytes).
constexpr u8 kEntryBytes[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
// The two GDS moves: each call's place, its bytes (call rel32 to the writer),
// and where it returns to.
struct GdsMove {
    u64 call;
    u8 bytes[5];
};
constexpr GdsMove kGdsMoves[2] = {{0x2acc30f, {0xe8, 0x4c, 0x7b, 0x9a, 0xfe}}, {0x2acd08c, {0xe8, 0xcf, 0x6d, 0x9a, 0xfe}}};

// Set by gnm_dma_configure before any of the game's code runs; read by the leaf.
u8 g_gds_dead = 0, g_census = 0;
u64 g_gds_return[2] = {~0ull, ~0ull};
std::atomic<u64> g_gds_skipped{0};

// The census's ring: the last packets' places and callers (leaf-safe).
constexpr u32 kRing = 4096;
struct RingSlot {
    std::atomic<u64> va{0}, caller{0};
};
RingSlot g_ring[kRing];
std::atomic<u32> g_ring_next{0};

using Refill = GUEST_ABI u64 (*)(u64 cb, u32 dwords, u64 arg, u64 room);

__attribute__((always_inline)) inline void put(u64 at, u32 v) { store<u32>(at, v); }

// The game's function, written out.
inline u64 write_dma(u64 cb, u32 dst_sel, u64 dst, u32 src_sel, u64 src, u32 bytes, u32 blocking) {
    u64 wp = load<u64>(cb + 0x10);
    const u64 room = (load<u64>(cb + 0x8) - wp) >> 2;
    if (static_cast<u32>(room) < 7) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u64 r = refill(cb, 7, load<u64>(cb + 0x20), room);
        if (static_cast<u8>(r) == 0) return r;
        wp = load<u64>(cb + 0x10);
    }
    put(wp, 0xc0055000u);
    put(wp + 4, ((dst_sel & 1) << 20) | ((src_sel & 3) << 29) | (blocking << 31) | 0x4004000u);
    put(wp + 8, static_cast<u32>(src));
    put(wp + 12, static_cast<u32>(src >> 32));
    put(wp + 16, static_cast<u32>(dst));
    put(wp + 20, static_cast<u32>(dst >> 32));
    put(wp + 24, (bytes & 0x1fffff) | ((src_sel & 4) << 24) | ((dst_sel & 4) << 25) | ((src_sel & 8) << 25) | ((dst_sel & 8) << 26));
    store<u64>(cb + 0x10, wp + 0x1c);
    return wp;
}

// What the leaf does, given whom it returns to.
inline u64 dma_from(u64 ret, u64 cb, u32 dst_sel, u64 dst, u32 src_sel, u64 src, u32 bytes, u32 blocking) {
    if (g_gds_dead && (ret == g_gds_return[0] || ret == g_gds_return[1])) {
        g_gds_skipped.fetch_add(1, std::memory_order_relaxed);
        return load<u64>(cb + 0x10);
    }
    const u64 wp = load<u64>(cb + 0x10);
    const u64 r = write_dma(cb, dst_sel, dst, src_sel, src, bytes, blocking);
    if (g_census && load<u64>(cb + 0x10) != wp) {
        RingSlot& s = g_ring[g_ring_next.fetch_add(1, std::memory_order_relaxed) & (kRing - 1)];
        s.caller.store(ret, std::memory_order_relaxed);
        s.va.store(r, std::memory_order_release);
    }
    return r;
}

}  // namespace

DECOMP_LEAF u64 gnm_dma_data(u64 cb, u32 dst_sel, u64 dst, u32 src_sel, u64 src, u32 bytes, u32 blocking) {
    return dma_from(reinterpret_cast<u64>(__builtin_return_address(0)), cb, dst_sel, dst, src_sel, src, bytes, blocking);
}

namespace {

// BBHOST_DECOMP_COMPARE=1: the game's version on every call, ours beside it on
// the same command buffer - the packet's bytes, the write pointer and the
// return value compared - then the game's result left in place. A call the
// refill would serve runs the game's alone.
DecompCompare g_counts;
void* g_original = nullptr;
std::atomic<u64> g_compare_alone{0};
std::atomic<int> g_compare_logs{0};

GUEST_ABI u64 compare_dma(u64 cb, u64 dst_sel, u64 dst, u64 src_sel, u64 src, u64 bytes, u64 blocking) {
    GuestCallRegs g{};
    g.fn = g_original;
    g.ints[0] = cb;
    g.ints[1] = dst_sel;
    g.ints[2] = dst;
    g.ints[3] = src_sel;
    g.ints[4] = src;
    g.ints[5] = bytes;
    g.stack[0] = blocking;
    const u64 wp = load<u64>(cb + 0x10);
    if (static_cast<u32>((load<u64>(cb + 0x8) - wp) >> 2) < 7) {
        g_compare_alone.fetch_add(1, std::memory_order_relaxed);
        hle_call_guest_regs(&g);
        return g.ret;
    }
    u8 before[28], game[28];
    std::memcpy(before, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), sizeof before);
    hle_call_guest_regs(&g);
    const u64 game_ret = g.ret, game_wp = load<u64>(cb + 0x10);
    std::memcpy(game, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), sizeof game);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), before, sizeof before);
    store<u64>(cb + 0x10, wp);
    const u64 our_ret = write_dma(cb, static_cast<u32>(dst_sel), dst, static_cast<u32>(src_sel), src, static_cast<u32>(bytes),
                                  static_cast<u32>(blocking));
    const u64 our_wp = load<u64>(cb + 0x10);
    const bool same = our_ret == game_ret && our_wp == game_wp &&
                      std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), game, sizeof game) == 0;
    g_counts.calls.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        g_counts.differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            host_log("decomp: gnm_dma_data differs from the game's: returned 0x%llx (game 0x%llx), write pointer +%lld (game +%lld)",
                     static_cast<ull>(our_ret), static_cast<ull>(game_ret), static_cast<long long>(our_wp - wp),
                     static_cast<long long>(game_wp - wp));
        }
    }
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), game, sizeof game);
    store<u64>(cb + 0x10, game_wp);
    return game_ret;
}

void report() {
    if (decomp_placed("gnm_dma_data") != 1) return;
    host_log("decomp: the DMA_DATA writer: the two moves through GDS %s (%llu calls not written)",
             g_gds_dead ? "write nothing: the counters ride in the dispatch tokens" : "write their packets",
             static_cast<ull>(g_gds_skipped.load()));
}

}  // namespace

void gnm_dma_configure(bool gds_moves_dead, bool census) {
    g_census = census ? 1 : 0;
    g_gds_dead = 0;
    if (!gds_moves_dead) return;
    // The two calls still the game's, as written against.
    for (const GdsMove& m : kGdsMoves) {
        if (std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(decomp_guest(m.call))), m.bytes, 5) != 0) {
            host_log("decomp: the DMA_DATA writer's GDS move at 0x%llx is not the call it was written against; both write their packets",
                     static_cast<ull>(m.call));
            return;
        }
    }
    for (int k = 0; k < 2; ++k) g_gds_return[k] = decomp_guest(kGdsMoves[k].call + 5);
    g_gds_dead = 1;
}

bool gnm_dma_placed() { return decomp_placed("gnm_dma_data") == 1; }

std::uint64_t gnm_dma_caller(std::uint64_t packet_va) {
    for (const RingSlot& s : g_ring) {
        if (s.va.load(std::memory_order_acquire) == packet_va) {
            const u64 ret = s.caller.load(std::memory_order_relaxed);
            return ret ? ret - g_decomp_slide + 0x400000 : 0;
        }
    }
    return 0;
}

void decomp_gx_dma_add() {
    if (const char* e = std::getenv("BBHOST_CP_WRITERS"); e && e[0] == '1') return;
    DecompFunction fn{"gnm_dma_data", "gx", kEntry, kEntryBytes, sizeof(kEntryBytes), reinterpret_cast<void*>(&gnm_dma_data), DecompKind::Leaf};
    fn.original = &g_original;
    fn.compare = reinterpret_cast<void*>(&compare_dma);
    fn.counts = &g_counts;
    fn.report = &report;
    decomp_add(fn);
}
