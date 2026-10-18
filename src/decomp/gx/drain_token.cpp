// GX's pipeline drain as a barrier token (docs/decomp.md):
// gx_command_buffer_wait_for_eop, Binary Ninja 0x2ab09e0. It is the only
// way a GX context makes the GPU wait for itself, and has one caller,
// gx_context_apply_pending_flags (0x2ab9850), for wait kinds 1, 3 and 7 of
// the context's hazard tracker: after a dispatch wrote through a UAV, before
// a buffer is read as indices or indirect arguments, before a target is
// sampled. It carves an 8-byte label from the end of the command buffer,
// zeroes it, has an end-of-pipe event (bottom of pipe, or a colour/depth
// cache flush) write 1 into it and the command processor wait until it reads
// 1 - on the console a stall of the CP until everything before has run.
//
// Here the renderer orders its own work - a barrier ends every pass, and
// dispatches, transfers and copies carry theirs - so the GPU half of that is
// redundant, and the label's one other effect was on the command processor:
// a wait it passes on a label still to be written puts what it then writes
// to memory itself (a query block's reset, a WRITE_DATA, a DMA_DATA) on the
// GPU's timeline, behind the work before (hle/gnm_exec.cpp,
// cp_write_after_waits). So ours writes a two-dword NOP {kGxBarrierMagic} in
// the drain's place, and the command processor takes it as exactly that
// wait: no label, no end-of-pipe event, no wait, nothing for the GPU. Over
// the 12-place tour: 4.0M drains, each an EOP label (1.4 us of the command
// processor, and a fill recorded at the end of the command buffer) and a
// WAIT_REG_MEM (0.26 us) - 60 bytes of command buffer for 8.
//
// A leaf: the drain's arguments are (command buffer, event, cache action);
// it reads and writes the command buffer {+0x8 end, +0x10 write pointer,
// +0x18 refill, +0x20 its argument} as the game's writers do, and when there
// is no room calls the refill as they do (cb, dwords, argument, room left).
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "hle/modules.h"
#include "host/host_lists.h"
#include "log.h"

#include <atomic>
#include <cstdint>

using namespace decomp;

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov r14d, edx
// (14 bytes; mov r15d, esi follows).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xd6};

using Refill = GUEST_ABI u8 (*)(u64 cb, u32 dwords, u64 arg, u32 room);

// Written from many GX threads; a leaf takes no lock, so the counts are
// relaxed atomics.
std::atomic<std::uint64_t> g_tokens{0}, g_refills{0}, g_refused{0};

}  // namespace

DECOMP_LEAF void gx_drain_token(u64 cb, u32 event, u32 cache) {
    (void)event;
    (void)cache;
    u64 wp = load<u64>(cb + 0x10);
    const u64 end = load<u64>(cb + 0x8);
    if (!load<u64>(cb) || end < wp + 8) {
        // No chunk, or no room for two dwords: a new chunk from the refill,
        // as the game's writers ask for one.
        g_refills.fetch_add(1, std::memory_order_relaxed);
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u32 room = end > wp ? static_cast<u32>((end - wp) >> 2) : 0;
        if (!refill || !refill(cb, 2, load<u64>(cb + 0x20), room)) {
            g_refused.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        wp = load<u64>(cb + 0x10);
        if (load<u64>(cb + 0x8) < wp + 8) {
            g_refused.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    store<u32>(wp, 0xc0001000u);  // NOP, one dword of body
    store<u32>(wp + 4, kGxBarrierMagic);
    store<u64>(cb + 0x10, wp + 8);
    host_lists::note(cb, wp, 0x10);
    g_tokens.fetch_add(1, std::memory_order_relaxed);
}

namespace {

void report() {
    if (!g_tokens.load() && !g_refused.load()) return;
    host_log("decomp: GX's pipeline drains written as barrier tokens: %llu (%llu after a refill, %llu not written: the refill gave no room)",
             static_cast<ull>(g_tokens.load()), static_cast<ull>(g_refills.load()), static_cast<ull>(g_refused.load()));
}

}  // namespace

void decomp_gx_drain_token_add() {
    DecompFunction fn{"gx_command_buffer_wait_for_eop", "gx", 0x2ab09e0, kEntry, sizeof(kEntry),
                      reinterpret_cast<void*>(&gx_drain_token), DecompKind::Leaf};
    fn.report = &report;
    decomp_add(fn);
}
