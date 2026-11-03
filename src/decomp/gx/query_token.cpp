// The occlusion query's packets as one token (docs/decomp.md): Gnm's
// writeOcclusionQuery, Binary Ninja 0x1477660 (cb, kind, block), which GX's
// query Begin (sub_2569ec0) and End (sub_256a070) call for an occlusion query
// - and YEBIS for its glare's (sub_15db2b0, sub_15db380). For a begin it
// resets the query's 128-byte block with a DMA_DATA fill, waits for the fill
// (WAIT_REG_MEM on the block's first dword), then has the pixel pipes start
// counting and dump their counters into the block (two EVENT_WRITEs: the
// pixel-pipe stat control, then the dump); for an end, the two EVENT_WRITEs
// with the dump at block + 8. The command processor turned those into the
// renderer's own occlusion queries (host/occlusion.cpp: the draws between a
// begin's dump and its end's are measured, the counts written into the block
// on the GPU's timeline) and the reset into a fill on the GPU's timeline
// behind the barrier before it. That is D3D11's Begin/End (Dark Souls III's
// GX calls ID3D11DeviceContext::Begin/End, 0x141913990 / 0x141913a01), so
// ours writes it as one: a NOP {kGxQueryMagic, block | kind, block >> 32},
// and the command processor resets the block (a begin) and starts or ends
// the query where it walks it - in stream order, on the GPU's timeline,
// whatever came before. 351k begins and 351k ends over the 12-place tour: a
// DMA_DATA, a WAIT_REG_MEM and four EVENT_WRITEs each pair, 26 dwords for 8.
//
// A leaf, as the drain's token (drain_token.cpp): the command buffer {+0x8
// end, +0x10 write pointer, +0x18 refill, +0x20 its argument}, the refill
// asked for room when there is none.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "hle/modules.h"
#include "host/host_lists.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

using namespace decomp;

namespace {

// push rbp; mov rbp, rsp; push r15, r14, rbx, rax; mov r14, rdx; mov r15d, esi (16 bytes).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x41, 0x89, 0xf7};

using Refill = GUEST_ABI u8 (*)(u64 cb, u32 dwords, u64 arg, u32 room);

std::atomic<std::uint64_t> g_begins{0}, g_ends{0}, g_others{0}, g_refused{0}, g_cleared{0};

// BBHOST_OCCLUSION_FRESH (on by default; =0 leaves the block as it was): a
// begin also clears the query's 128-byte block on the CPU, as YEBIS's own
// begin does before it calls here (sub_15db2b0: memset 0x80) and as GX's End
// does for its other query kinds (their first word). Without it the block
// keeps its previous query's sixteen valid words until the GPU reaches this
// query's reset - and Bloodborne's culler asks GetData for the block on the
// frames after it recorded the query (sub_122f190), so with the GPU a frame
// or more behind it took that previous answer as this query's: the same
// object's earlier test, or, on a query entry its pool had given another
// object, that object's - an "occluded" one hid a visible object until the
// object's own answer came. Cleared, the block reads not ready until this
// query's own counts land, as D3D11's GetData does for Dark Souls III's GX on
// PC: the culler waits for its own answer and keeps the object as it was
// meanwhile. A GPU that keeps up changes nothing (the answer is in by the
// next poll). The game begins a block again only once it has taken the
// block's last answer, so no older count is still on its way to it; and only
// a begin whose token went in clears, so a refused one cannot leave a block
// waiting for counts that never come.
const bool g_fresh = [] {
    const char* e = std::getenv("BBHOST_OCCLUSION_FRESH");
    return !(e && e[0] == '0');
}();

}  // namespace

DECOMP_LEAF void gx_query_token(u64 cb, u32 kind, u64 block) {
    // The game's: a begin (0) resets the block and dumps at it, an end (1)
    // dumps at block + 8; anything else dumps at the block without a reset.
    const u32 k = kind == 0 ? 0u : kind == 1 ? 1u : 2u;
    (k == 0 ? g_begins : k == 1 ? g_ends : g_others).fetch_add(1, std::memory_order_relaxed);
    u64 wp = load<u64>(cb + 0x10);
    const u64 end = load<u64>(cb + 0x8);
    if (!load<u64>(cb) || end < wp + 16) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u32 room = end > wp ? static_cast<u32>((end - wp) >> 2) : 0;
        if (!refill || !refill(cb, 4, load<u64>(cb + 0x20), room)) {
            g_refused.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        wp = load<u64>(cb + 0x10);
        if (load<u64>(cb + 0x8) < wp + 16) {
            g_refused.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    store<u32>(wp, 0xc0021000u);  // NOP, three dwords of body
    store<u32>(wp + 4, kGxQueryMagic);
    store<u32>(wp + 8, static_cast<u32>(block & ~7ull) | k);
    store<u32>(wp + 12, static_cast<u32>(block >> 32));
    store<u64>(cb + 0x10, wp + 16);
    host_lists::note(cb, wp, 0x10);
    if (k == 0 && g_fresh) {
        const u64 b = block & ~7ull;
        for (u32 i = 0; i < 16; ++i) store<u64>(b + 8ull * i, 0);
        g_cleared.fetch_add(1, std::memory_order_relaxed);
    }
}

namespace {

void report() {
    if (!g_begins.load() && !g_ends.load() && !g_others.load()) return;
    host_log("decomp: occlusion queries written as query tokens: %llu begins, %llu ends, %llu other dumps; %llu not written (the refill gave "
             "no room); %llu blocks cleared at their begin%s",
             static_cast<ull>(g_begins.load()), static_cast<ull>(g_ends.load()), static_cast<ull>(g_others.load()),
             static_cast<ull>(g_refused.load()), static_cast<ull>(g_cleared.load()), g_fresh ? "" : " (BBHOST_OCCLUSION_FRESH=0)");
}

}  // namespace

void decomp_gx_query_token_add() {
    DecompFunction fn{"gnm_write_occlusion_query", "gx", 0x1477660, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&gx_query_token),
                      DecompKind::Leaf};
    fn.report = &report;
    decomp_add(fn);
}
