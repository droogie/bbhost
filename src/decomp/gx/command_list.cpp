// GX's command lists as source (docs/decomp.md): where recorded GPU work
// changes hands. ID3D11DeviceContext's FinishCommandList on a deferred
// context (gx_finish_command_list, 0x2ad5850, and the body it wraps,
// 0x256aca0) takes the spans the context recorded into a command list; the
// immediate context's ExecuteCommandList (0x25679a0 and its body 0x256a7d0)
// hands one list's spans to the GPU, and ExecuteCommandLists (0x25679e0 and
// its body 0x256a980) - the frame's render step, up to 72 lists at a time -
// puts each list's spans into its own stream; a command buffer's segment
// goes to the GPU in gx_command_buffer_close_segment (0x2ab0b00), through
// sceGnmSubmitCommandBuffers - and the frame's last one with the flip
// (gx_command_buffer_flip 0x2aafb30, gx_command_buffer_close_segment_and_flip
// 0x2ab0ce0, gx_submit_segment_and_flip 0x2ab0e30: the room for the system's
// flip packets, the compute ring's waits, sceGnmSubmitAndFlipCommandBuffers
// and sceGnmSubmitDone). The objects they hand on are in
// decomp/gx/command_list.h: the span, the span queue, the command list.
//
// Each is the game's call sequence, its callees at the game's addresses
// (a callee on the decomp list is ours, the rest stay the game's), and the
// memory each writes itself: the targets a list ended with taken as the
// immediate context's bound ones, the list emptied and its pages freed. The
// game's one-line helpers - the pending-flags byte's test and clear
// (0x2ab9a50, 0x2ab9a40), the list's identity (0x2ad5740) and the empty
// command buffer the close builds on its stack (0x1471270) - are folded:
// they touch nothing else.
//
// Leaves: nothing of the host's but memset, which the game's own calls are
// (the import is the host's). No compare stand-in: each hands its work on -
// pages freed, spans submitted - so nothing can run twice; the test runs
// them against the game's code (tests/decomp/gx_command_list_test.cpp).
#include "decomp/gx/command_list.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace decomp;
using decomp_gx::GxCommandList;
using decomp_gx::GxSpan;

#ifdef DECOMP_GX_COMMAND_LIST_MUTANTS
extern int g_list_mutant;
#define MUTANT(n) (g_list_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

// What they call, by Binary Ninja address (symbols/1.09-names.tsv).
constexpr u64 kFinishBody = 0x256aca0;          // gx_finish_command_list_impl(ctx, ctx + 0xb730, list, restore)
constexpr u64 kExecuteBody = 0x256a7d0;         // gx_context_execute_command_list(ctx, list, list, 0)
constexpr u64 kExecuteListsBody = 0x256a980;    // gx_context_execute_command_lists(ctx, lists, lists, count, 0)
constexpr u64 kCommandBufferFlush = 0x2aaf700;  // gx_command_buffer_flush(cb)
constexpr u64 kFinishRecording = 0x2aafd30;     // gx_context_finish_recording(cb, list): 0, or 9 when a page is not had
constexpr u64 kSnapshotOutputs = 0x2ad2f30;     // gx_command_list_snapshot_outputs(om, list + 0x20, list + 0x68)
constexpr u64 kResetInputAssembly = 0x2abba10;  // (ctx + 0xb518): its cache made unknown
constexpr u64 kResetRasterizer = 0x2ad3300;     // (ctx + 0xb528)
constexpr u64 kResetOutputMerger = 0x2ad2d20;   // (ctx + 0xb558)
constexpr u64 kResetCommit = 0x2abc090;         // (ctx + 0x50)
constexpr u64 kRenameResolveB = 0x2ad1090;      // gx_rename_map_resolve_b(ctx + 0xb6f8, cb)
constexpr u64 kRenameResolve = 0x2aadf50;       // gx_rename_map_resolve(ctx + 0xb3c0, ctx)
constexpr u64 kOutputTransition = 0x2ad3070;    // gx_output_transition(om, ctx, the list's first targets)
constexpr u64 kApplyPendingFlags = 0x2ab9850;   // gx_context_apply_pending_flags(ctx + 0xb478, cb)
constexpr u64 kCloseCurrentSegment = 0x2aafa00;  // (cb): the segment so far to the GPU
constexpr u64 kSubmitListSpans = 0x2abb730;     // (list): its spans to the GPU in one submission
constexpr u64 kSpanStreamBegin = 0x2ab0110;     // (cb): flushed, and its refill one that keeps spans
constexpr u64 kAppendSpan = 0x2ab03f0;          // (cb, span): 0, or 9 when a page is not had
constexpr u64 kRetireChunkList = 0x2ab0630;     // gx_context_retire_chunk_list(cb): the spans to the GPU, the chunks retired
constexpr u64 kWaitRegMem = 0x1477b30;          // gnm_wait_reg_mem(cb, address, mask, function, reference)
constexpr u64 kSubmitCommandBuffers = 0x1489a10;  // sceGnmSubmitCommandBuffers' import entry
constexpr u64 kSubmitAndFlip = 0x1489a20;         // sceGnmSubmitAndFlipCommandBuffers' import entry
constexpr u64 kSubmitDone = 0x1489b50;            // sceGnmSubmitDone's import entry
constexpr u64 kReserveFlipPackets = 0x14784a0;    // gnm_reserve_flip_packets(cb): the system's room, a 64-dword NOP
constexpr u64 kCloseSegmentAndFlip = 0x2ab0ce0;   // gx_command_buffer_close_segment_and_flip(cb, fresh, header, flip)
constexpr u64 kSubmitSegmentAndFlip = 0x2ab0e30;  // gx_submit_segment_and_flip(&flip, header)

using Cb = GUEST_ABI void (*)(u64);
using Ctx2 = GUEST_ABI void (*)(u64, u64);
using Ctx3 = GUEST_ABI void (*)(u64, u64, u64);
using Ret2 = GUEST_ABI u32 (*)(u64, u64);

void call1(u64 bn, u64 a) { game_function<Cb>(bn)(a); }
void call2(u64 bn, u64 a, u64 b) { game_function<Ctx2>(bn)(a, b); }

// The pending-flags byte (+0x24 of ctx + 0xb478): cleared where set.
void clear_pending_byte(u64 ctx) {
    if (load<u8>(ctx + 0xb478 + 0x24)) store<u8>(ctx + 0xb478 + 0x24, MUTANT(1) ? 1 : 0);
}

// The input-assembly, rasterizer, output-merger and commit caches made
// unknown, so the next draw writes all of its state.
void reset_caches(u64 ctx) {
    call1(kResetInputAssembly, ctx + 0xb518);
    call1(kResetRasterizer, ctx + 0xb528);
    call1(kResetOutputMerger, ctx + 0xb558);
    call1(kResetCommit, ctx + 0x50);
}

// The targets the list ended with, as the immediate context's bound ones:
// its eight colour targets into the output merger's colour record set
// (+0x188 bit 0 selects it), its depth target into the depth one (bit 1).
void take_last_targets(u64 ctx, u64 list) {
    const u64 colour = ctx + 0xb5b8 + (load<u32>(ctx + 0xb6e0) & 1) * 0x48;
    u64 targets[8];
    std::memcpy(targets, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(list + offsetof(GxCommandList, last_colour))), sizeof targets);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(colour)), targets, sizeof targets);
    const u64 depth = load<u64>(list + offsetof(GxCommandList, last_depth));
    store<u64>(ctx + 0xb5f8 + ((load<u32>(ctx + 0xb6e0) >> (MUTANT(2) ? 0 : 1)) & 1) * 0x48, depth);
}

// The list's pages back to their allocator (its +0x70; a page's link read
// before the page goes), the queue emptied.
void free_pages(u64 list) {
    using Free = GUEST_ABI void (*)(u64 self, u64 page);
    const auto free_page = [](u64 allocator, u64 page) {
        reinterpret_cast<Free>(static_cast<std::uintptr_t>(load<u64>(load<u64>(allocator) + 0x70)))(allocator, page);
    };
    if (const u64 allocator = load<u64>(list)) {
        const u64 link = static_cast<u64>(load<u32>(list + 0x18)) << 6;
        if (u64 page = load<u64>(list + 8)) {
            u64 next = load<u64>(page + link);
            free_page(allocator, page);
            while (next) {
                page = next;
                next = load<u64>(page + link);
                free_page(load<u64>(list), page);
            }
        }
    }
}

}  // namespace

// gx_finish_command_list(deferred ctx, list, restore): the recorded work into
// the list; 8 without a list. The context's compute state (+0xb730) cleared
// unless restore asks to keep it (D3D11's RestoreDeferredContextState).
DECOMP_LEAF u64 gx_finish_command_list(u64 ctx, u64 list, u64 restore) {
    if (!list) return 8;
    const u64 keep = restore & 0xff;
    using Body = GUEST_ABI u32 (*)(u64, u64, u64, u64);
    const u32 result = game_function<Body>(kFinishBody)(ctx, ctx + 0xb730, list, keep);
    if (!keep) std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ctx + 0xb730)), 0, 0x23a8);
    return result;
}

// gx_finish_command_list_impl(ctx, -, list, -): the command buffer flushed,
// its spans taken into the list (gx_context_finish_recording: 9 if a page
// could not be had, returned), the output merger's records of the targets
// around them noted in the list, the context's caches made unknown.
DECOMP_LEAF u64 gx_finish_command_list_impl(u64 ctx, u64 state, u64 list, u64 restore) {
    if (!list) return 8;
    const u64 cb = ctx + 0x10;
    clear_pending_byte(ctx);
    call1(kCommandBufferFlush, cb);
    const u32 result = game_function<Ret2>(kFinishRecording)(cb, list);
    if (result && !MUTANT(3)) return result;
    game_function<Ctx3>(kSnapshotOutputs)(ctx + 0xb558, list + offsetof(GxCommandList, first_records),
                                           list + offsetof(GxCommandList, last_colour));
    reset_caches(ctx);
    return 0;
}

// gx_context_execute_command_list(immediate ctx, -, list): the renamed
// buffers resolved, the deferred passes the list's first targets owe the
// immediate context's run (gx_output_transition), the pending flags applied,
// the segment so far closed, the list's spans submitted, the caches made
// unknown, the targets the list ended with taken as bound, the list emptied.
// Returns what the game's returns: where its last clear began.
DECOMP_LEAF u64 gx_context_execute_command_list(u64 ctx, u64 unused, u64 list) {
    const u64 cb = ctx + 0x10;
    clear_pending_byte(ctx);
    call2(kRenameResolveB, ctx + 0xb6f8, cb);
    call2(kRenameResolve, ctx + 0xb3c0, ctx);
    game_function<Ctx3>(kOutputTransition)(ctx + 0xb558, ctx, list + offsetof(GxCommandList, first_records));
    call2(kApplyPendingFlags, ctx + 0xb478, cb);
    call1(kCloseCurrentSegment, cb);
    call1(kSubmitListSpans, list);
    reset_caches(ctx);
    take_last_targets(ctx, list);
    free_pages(list);
    store<u64>(list + 8, 0);
    store<u64>(list + 0x10, 0);
    std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(list + 0x1c)), 0, 0x94);
    return list + 0x1c;
}

// gx_context_execute_command_lists(immediate ctx, -, lists, count): as
// above for each list that is there, except that each list's spans go into
// the context's own span stream (appended one by one; a span the stream
// could not take a page for ends that page's spans) and the stream goes to
// the GPU, the chunks retired, after the last (gx_context_retire_chunk_list).
DECOMP_LEAF void gx_context_execute_command_lists(u64 ctx, u64 unused, u64 lists, u64 count) {
    const u32 n = static_cast<u32>(count);
    if (!lists || !n) return;
    const u64 cb = ctx + 0x10;
    clear_pending_byte(ctx);
    call2(kApplyPendingFlags, ctx + 0xb478, cb);
    call1(kCloseCurrentSegment, cb);
    call1(kSpanStreamBegin, cb);
    for (u64 i = 0;;) {
        if (const u64 list = load<u64>(lists + 8 * i)) {
            call2(kRenameResolveB, ctx + 0xb6f8, cb);
            call2(kRenameResolve, ctx + 0xb3c0, ctx);
            game_function<Ctx3>(kOutputTransition)(ctx + 0xb558, ctx, list + offsetof(GxCommandList, first_records));
            call2(kApplyPendingFlags, ctx + 0xb478, cb);
            if (u64 page = load<u64>(list + 8)) {
                const u64 per = load<u32>(list + 0x18);
                do {
                    const u64 link_at = page + (per << 6);
                    u64 next;
                    for (u64 k = 0;;) {
                        next = load<u64>(link_at);
                        const u64 spans = next ? per : load<u32>(list + 0x1c);
                        if (k >= spans) break;
                        const u32 taken = game_function<Ret2>(kAppendSpan)(cb, page + (k << 6));
                        k = static_cast<u32>(k + 1);
                        if (taken && !MUTANT(4)) {
                            next = load<u64>(link_at);
                            break;
                        }
                    }
                    page = next;
                } while (page);
            }
            reset_caches(ctx);
            take_last_targets(ctx, list);
            free_pages(list);
            store<u64>(list + 0x10, 0);
            store<u64>(list + 8, 0);
            std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(list + 0x1c)), 0, 0x94);
        }
        ++i;
        if (static_cast<u32>(i) == n) break;
    }
    call1(kRetireChunkList, cb);
}

// gx_execute_command_list(immediate ctx, list, restore): ExecuteCommandList -
// the body above, then the compute state (+0xb730) cleared unless restore
// asks to keep it.
DECOMP_LEAF void gx_execute_command_list(u64 ctx, u64 list, u64 restore) {
    using Body = GUEST_ABI void (*)(u64, u64, u64, u64);
    game_function<Body>(kExecuteBody)(ctx, list, list, 0);
    if (!(restore & 0xff)) std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ctx + 0xb730)), 0, 0x23a8);
}

// gx_execute_command_lists(immediate ctx, lists, count, restore): the same for
// ExecuteCommandLists.
DECOMP_LEAF void gx_execute_command_lists(u64 ctx, u64 lists, u64 count, u64 restore) {
    using Body = GUEST_ABI void (*)(u64, u64, u64, u64, u64);
    game_function<Body>(kExecuteListsBody)(ctx, lists, lists, static_cast<u32>(count), 0);
    if (!(restore & 0xff)) std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(ctx + 0xb730)), 0, 0x23a8);
}

// gx_command_buffer_close_segment(cb, fresh, header, -): the segment the
// header describes, [begin, write), to the GPU. With a fresh start (the
// buffer itself, its write pointer where the next segment begins), what the
// compute ring's counter asks first: when the counter the buffer last waited
// for (ring +0x20) is not the ring's last (+0x18 less one), WAIT_REG_MEMs on
// the ring's word (+0x10) - for the wrap first when the old one was negative
// and the new is not - written at the fresh start, which goes ahead of the
// segment in the same submission; the counter noted. Without one, the
// segment alone. Nothing goes for no bytes.
DECOMP_LEAF void gx_command_buffer_close_segment(u64 cb, u64 fresh, u64 header, u64 kind) {
    u64 first, second;
    u32 first_bytes, second_bytes;
    if (fresh) {
        const u64 ring = load<u64>(load<u64>(cb + 0xb490) + 0xb728);
        const u32 last = load<u32>(load<u64>(ring + 0x18)) - 1;
        const u32 waited = load<u32>(load<u64>(ring + 0x20));
        if (waited != last) {
            using Wait = GUEST_ABI u64 (*)(u64 cb, u64 address, u64 mask, u64 function, u64 reference);
            const auto wait = game_function<Wait>(kWaitRegMem);
            if (static_cast<s32>(waited) < 0 && static_cast<s32>(last) >= 0 && !MUTANT(5)) wait(fresh, load<u64>(ring + 0x10), 0xffffffffu, 1, waited);
            wait(fresh, load<u64>(ring + 0x10), 0xffffffffu, 5, last);
            store<u32>(load<u64>(ring + 0x20), last);
        }
        first = load<u64>(fresh);
        first_bytes = load<u32>(fresh + 0x10) - static_cast<u32>(first);
        second = load<u64>(header);
        second_bytes = load<u32>(header + 0x10) - static_cast<u32>(second);
    } else {
        first = load<u64>(header);
        first_bytes = load<u32>(header + 0x10) - static_cast<u32>(first);
        second = 0;  // the empty command buffer the game's builds on its stack
        second_bytes = 0;
    }
    first_bytes &= ~3u;
    second_bytes &= ~3u;
    u64 addresses[2];
    u32 sizes[2];
    u32 n = 0;
    if (first_bytes && second_bytes) {
        addresses[0] = first;
        sizes[0] = first_bytes;
        addresses[1] = second;
        sizes[1] = MUTANT(6) ? first_bytes : second_bytes;
        n = 2;
    } else if (first_bytes) {
        addresses[0] = first;
        sizes[0] = first_bytes;
        n = 1;
    } else if (second_bytes) {
        addresses[0] = second;
        sizes[0] = second_bytes;
        n = 1;
    }
    if (!n) return;
    using Submit = GUEST_ABI s32 (*)(u32 count, const u64* dcb, const u32* dcb_bytes, const u64* ccb, const u32* ccb_bytes);
    game_function<Submit>(kSubmitCommandBuffers)(n, addresses, sizes, nullptr, nullptr);
}

// What the flip's submit is told, as the game builds it on its stack and
// passes it by value: the video out handle (low 32 bits) with the display
// buffer's index (high), the flip mode (low 32 bits), the flip's argument,
// and the command buffer's kind byte (not read). The rest is what the stack
// held.
struct GxFlip {
    u64 target, mode, arg, kind;
};
static_assert(sizeof(GxFlip) == 32, "passed in memory, four qwords");

namespace {
using SubmitFlip = GUEST_ABI s32 (*)(u32 count, const u64* dcb, const u32* dcb_bytes, const u64* ccb, const u32* ccb_bytes, u64 handle, u32 index,
                                     u32 mode, u64 arg);
// The one or two buffers a submission takes, from their sizes: both, the
// first, the second, or none.
u32 pick_buffers(u64 first, u32 first_bytes, u64 second, u32 second_bytes, u64* addresses, u32* sizes) {
    if (first_bytes && second_bytes) {
        addresses[0] = first;
        sizes[0] = first_bytes;
        addresses[1] = second;
        sizes[1] = second_bytes;
        return 2;
    }
    if (first_bytes) {
        addresses[0] = first;
        sizes[0] = first_bytes;
        return 1;
    }
    if (second_bytes) {
        addresses[0] = second;
        sizes[0] = second_bytes;
        return 1;
    }
    return 0;
}
}  // namespace

// gx_submit_segment_and_flip(&flip, header): the segment the header describes
// to the GPU with the flip (beside the empty command buffer the game builds
// on its stack), then the submission done. Nothing goes for no bytes.
DECOMP_LEAF void gx_submit_segment_and_flip(u64 flip, u64 header) {
    const u64 first = load<u64>(header);
    const u32 first_bytes = (load<u32>(header + 0x10) - static_cast<u32>(first)) & ~3u;
    u64 addresses[2];
    u32 sizes[2];
    const u32 n = pick_buffers(first, first_bytes, 0, 0, addresses, sizes);
    if (!n) return;
    game_function<SubmitFlip>(kSubmitAndFlip)(n, addresses, sizes, nullptr, nullptr, load<u32>(flip), load<u32>(flip + 4), load<u32>(flip + 8),
                                              load<u64>(flip + 0x10));
    game_function<GUEST_ABI void (*)()>(kSubmitDone)();
}

// gx_command_buffer_close_segment_and_flip(cb, fresh, header, flip): as
// gx_command_buffer_close_segment - the compute ring's waits at a fresh start
// ahead of the segment - but submitted with the flip, then the submission
// done; without a fresh start, gx_submit_segment_and_flip.
DECOMP_LEAF void gx_command_buffer_close_segment_and_flip(u64 cb, u64 fresh, u64 header, GxFlip flip) {
    const u64 flip_at = reinterpret_cast<u64>(&flip);
    if (!fresh) {
        game_function<GUEST_ABI void (*)(u64, u64)>(kSubmitSegmentAndFlip)(flip_at, header);
        return;
    }
    const u64 ring = load<u64>(load<u64>(cb + 0xb490) + 0xb728);
    const u32 last = load<u32>(load<u64>(ring + 0x18)) - 1;
    const u32 waited = load<u32>(load<u64>(ring + 0x20));
    if (waited != last) {
        using Wait = GUEST_ABI u64 (*)(u64 cb, u64 address, u64 mask, u64 function, u64 reference);
        const auto wait = game_function<Wait>(kWaitRegMem);
        if (static_cast<s32>(waited) < 0 && static_cast<s32>(last) >= 0) wait(fresh, load<u64>(ring + 0x10), 0xffffffffu, 1, waited);
        wait(fresh, load<u64>(ring + 0x10), 0xffffffffu, 5, last);
        store<u32>(load<u64>(ring + 0x20), last);
    }
    const u64 first = load<u64>(fresh), second = load<u64>(header);
    const u32 first_bytes = (load<u32>(fresh + 0x10) - static_cast<u32>(first)) & ~3u;
    const u32 second_bytes = (load<u32>(header + 0x10) - static_cast<u32>(second)) & ~3u;
    u64 addresses[2];
    u32 sizes[2];
    const u32 n = pick_buffers(first, first_bytes, second, second_bytes, addresses, sizes);
    if (!n) return;
    // The handle's register carries the index in its high half, as the
    // game's loads it.
    game_function<SubmitFlip>(kSubmitAndFlip)(n, addresses, sizes, nullptr, nullptr, MUTANT(7) ? static_cast<u32>(flip.target) : flip.target,
                                              static_cast<u32>(flip.target >> 32), static_cast<u32>(flip.mode), flip.arg);
    game_function<GUEST_ABI void (*)()>(kSubmitDone)();
}

// gx_command_buffer_flip(cb, video out handle, display buffer, flip mode,
// flip argument): the frame's last segment, with the flip. Room for 64
// dwords first (the refill asked once; 9 if it gave less), the system's flip
// packets reserved, then the segment closed with the flip: a fresh start
// after it unless the compute ring's counter is on its last value (none) or
// fewer than seven dwords are left (then the refill is asked for none and
// nothing is submitted). The write pointer noted as flushed. 0 otherwise.
DECOMP_LEAF u64 gx_command_buffer_flip(u64 cb, u64 handle, u64 index, u64 mode, u64 arg) {
    using Refill = GUEST_ABI u64 (*)(u64 cb, u64 dwords, u64 arg3, u64 arg4);
    const auto refill = [cb](u64 arg3, u64 arg4) {
        reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)))(cb, 0, arg3, arg4);
    };
    const auto room = [cb] { return (load<u64>(cb + 0x8) - load<u64>(cb + 0x10)) >> 2; };
    if (static_cast<u32>(room()) <= 0x3f) {
        // The refill with the buffer itself for its argument, and the flip
        // mode's register as it came, as the game's call leaves them.
        refill(cb, MUTANT(8) ? 0 : mode);
        if (static_cast<u32>(room()) < 0x40) return 9;
    }
    call1(kReserveFlipPackets, cb);
    GxFlip flip{};
    flip.target = static_cast<u32>(handle) | static_cast<u64>(static_cast<u32>(index)) << 32;
    flip.mode = static_cast<u32>(mode);
    flip.arg = arg;
    flip.kind = load<u8>(cb + 0xb4f0);
    u64 header[8];
    u64 fresh = 0;
    const u64 ring = load<u64>(load<u64>(cb + 0xb490) + 0xb728);
    if (load<u32>(load<u64>(ring + 0x20)) == load<u32>(load<u64>(ring + 0x18)) - 1) {
        for (int k = 0; k < 8; ++k) header[k] = load<u64>(cb + 8ull * k);
    } else {
        const u64 left = room();
        if (static_cast<u32>(left) < (MUTANT(9) ? 8u : 7u)) {
            refill(load<u64>(cb + 0x20), left);
            return 0;
        }
        for (int k = 0; k < 8; ++k) header[k] = load<u64>(cb + 8ull * k);
        store<u64>(cb, load<u64>(cb + 0x10));
        fresh = cb;
    }
    game_function<GUEST_ABI void (*)(u64, u64, u64, GxFlip)>(kCloseSegmentAndFlip)(cb, fresh, reinterpret_cast<u64>(header), flip);
    const u64 wp = load<u64>(cb + 0x10);
    store<u64>(cb, wp);
    store<u64>(cb + 0xb4b8, wp);
    return 0;
}

namespace {

// The entries: the prologues the decomp list places ours at.
#define PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov rax, rsi
constexpr u8 kEntryFinish[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x48, 0x89, 0xf0};
// push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; mov rbx, rdx; mov r14, rdi
constexpr u8 kEntryFinishBody[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xd3, 0x49, 0x89, 0xfe};
// PUSH6; push rax; mov r14, rdx
constexpr u8 kEntryExecute[] = {PUSH6, 0x50, 0x49, 0x89, 0xd6};
// PUSH6; sub rsp, 0x68; mov r14d, ecx
constexpr u8 kEntryExecuteLists[] = {PUSH6, 0x48, 0x83, 0xec, 0x68, 0x41, 0x89, 0xce};
// push rbp; mov rbp, rsp; push r14; push rbx; mov r14d, edx; mov rbx, rdi
constexpr u8 kEntryExecuteWrapper[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xd6, 0x48, 0x89, 0xfb};
// push rbp; mov rbp, rsp; push r14; push rbx; mov r14d, ecx; mov eax, edx
constexpr u8 kEntryExecuteListsWrapper[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xce, 0x89, 0xd0};
// PUSH6; sub rsp, 0x68; mov r14, rdx; mov r15, rsi
constexpr u8 kEntryCloseSegment[] = {PUSH6, 0x48, 0x83, 0xec, 0x68, 0x49, 0x89, 0xd6, 0x49, 0x89, 0xf7};
// PUSH6; sub rsp, 0x98; mov [rbp-0xa0], r8; mov r15d, ecx
constexpr u8 kEntryFlip[] = {PUSH6, 0x48, 0x81, 0xec, 0x98, 0x00, 0x00, 0x00, 0x4c, 0x89, 0x85, 0x60, 0xff, 0xff, 0xff, 0x41, 0x89, 0xcf};
// PUSH6; sub rsp, 0x38; mov r15, rdx; mov r12, rsi; lea rdx, [rbp+0x10]
constexpr u8 kEntryCloseSegmentAndFlip[] = {PUSH6, 0x48, 0x83, 0xec, 0x38, 0x49, 0x89, 0xd7, 0x49, 0x89, 0xf4, 0x48, 0x8d, 0x55, 0x10};
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; sub rsp, 0x78; mov rbx, rsi; mov r14, rdi
constexpr u8 kEntrySubmitSegmentAndFlip[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x78, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe};
#undef PUSH6

}  // namespace

void decomp_gx_command_list_add() {
    const DecompFunction fns[] = {
        {"gx_finish_command_list", "gx", 0x2ad5850, kEntryFinish, sizeof kEntryFinish, reinterpret_cast<void*>(&gx_finish_command_list),
         DecompKind::Leaf},
        {"gx_finish_command_list_impl", "gx", kFinishBody, kEntryFinishBody, sizeof kEntryFinishBody,
         reinterpret_cast<void*>(&gx_finish_command_list_impl), DecompKind::Leaf},
        {"gx_execute_command_list", "gx", 0x25679a0, kEntryExecuteWrapper, sizeof kEntryExecuteWrapper,
         reinterpret_cast<void*>(&gx_execute_command_list), DecompKind::Leaf},
        {"gx_context_execute_command_list", "gx", kExecuteBody, kEntryExecute, sizeof kEntryExecute,
         reinterpret_cast<void*>(&gx_context_execute_command_list), DecompKind::Leaf},
        {"gx_execute_command_lists", "gx", 0x25679e0, kEntryExecuteListsWrapper, sizeof kEntryExecuteListsWrapper,
         reinterpret_cast<void*>(&gx_execute_command_lists), DecompKind::Leaf},
        {"gx_context_execute_command_lists", "gx", kExecuteListsBody, kEntryExecuteLists, sizeof kEntryExecuteLists,
         reinterpret_cast<void*>(&gx_context_execute_command_lists), DecompKind::Leaf},
        {"gx_command_buffer_close_segment", "gx", 0x2ab0b00, kEntryCloseSegment, sizeof kEntryCloseSegment,
         reinterpret_cast<void*>(&gx_command_buffer_close_segment), DecompKind::Leaf},
        {"gx_command_buffer_flip", "gx", 0x2aafb30, kEntryFlip, sizeof kEntryFlip, reinterpret_cast<void*>(&gx_command_buffer_flip), DecompKind::Leaf},
        {"gx_command_buffer_close_segment_and_flip", "gx", kCloseSegmentAndFlip, kEntryCloseSegmentAndFlip, sizeof kEntryCloseSegmentAndFlip,
         reinterpret_cast<void*>(&gx_command_buffer_close_segment_and_flip), DecompKind::Leaf},
        {"gx_submit_segment_and_flip", "gx", kSubmitSegmentAndFlip, kEntrySubmitSegmentAndFlip, sizeof kEntrySubmitSegmentAndFlip,
         reinterpret_cast<void*>(&gx_submit_segment_and_flip), DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
