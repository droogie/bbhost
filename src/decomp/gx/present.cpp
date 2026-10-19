// GX's present as source (docs/decomp.md): where the engine ends a frame and
// paces the next. gx_swap_chain_present (0x2ad5dd0) takes a sync interval
// (0-4, as DXGI's Present does; 8 back otherwise) and the swap chain's frame
// state (+0x33d0); gx_swap_chain_present_frame (0x2ad6e50) is the frame's
// end on the render thread:
//
//   - the pacing: the swap chain counts its flips (+0x2e0) and keeps as many
//     display buffers as it lets flips be in flight (+0x328, two here); it
//     asks VideoOut for the flip status and, while the flip that many back
//     has not completed (the status's flip argument is the count of the last
//     one done), waits on its flip event queue (+0x2f8). That is why the game
//     runs at most two frames ahead of the display (hle/video.cpp holds a
//     flip to the tick for the same wait);
//   - the display buffer's address noted (+0x350, from the buffer's Gnm
//     render target, the swap chain's buffers at +0, 0x2c bytes each);
//   - the output merger's end of frame and the copy to the display buffer
//     (0x2ad1eb0, 0x2ab6cb0);
//   - the frame's last segment submitted with the flip
//     (gx_command_buffer_flip, decomp/gx/command_list.cpp) - flip mode 2
//     (immediate) for a sync interval of 0, 1 (on the vblank) otherwise, the
//     count as the flip argument - and the count stepped;
//   - the block pools' releases taken back and the context's state reset
//     (0x2aaa860, 0x2568530), and the command buffer flushed.
//
// Another thread calling it only gets 0 back. Returns VideoOut's error
// class 5 when the status or the wait fails, what the flip returns when it
// fails, 0 otherwise.
//
// That is the frame-pacing surface: how many frames the game runs ahead of
// the display, what a frame waits for before it is handed over, and the
// point a frame ends (docs/modding.md). Leaves: the game's calls - VideoOut's
// and the kernel's imports among them, which switch to the host's FS
// themselves - in the game's order.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <cstdint>

using namespace decomp;

#ifdef DECOMP_GX_PRESENT_MUTANTS
extern int g_present_mutant;
#define MUTANT(n) (g_present_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

// The swap chain.
constexpr u64 kContext = 0x2d8;     // GX's immediate context
constexpr u64 kFlips = 0x2e0;       // flips so far (s64)
constexpr u64 kVideoOut = 0x2f0;    // u32: the VideoOut handle
constexpr u64 kFlipQueue = 0x2f8;   // the flip event queue
constexpr u64 kInFlight = 0x328;    // u32: display buffers, flips in flight at most
constexpr u64 kCopyState = 0x348;   // what the copy to the display buffer keeps
constexpr u64 kDisplayAddress = 0x350;
constexpr u64 kCopyTarget = 0x380;
constexpr u64 kFrameState = 0x33d0;
constexpr u64 kRenderThread = 0xb718;  // of the context: the thread it records on
// VideoOut's flip status (SceVideoOutFlipStatus, 0x40 bytes): the last completed flip's argument at +0x18.
constexpr u64 kStatusFlipArg = 0x18;

constexpr u64 kPthreadSelf = 0x248ba90;       // j_scePthreadSelf
constexpr u64 kGetFlipStatus = 0x2fbf5e8;     // sceVideoOutGetFlipStatus(handle, status*)
constexpr u64 kWaitEqueue = 0x2fbf5f8;        // sceKernelWaitEqueue(queue, events*, count, out*, timeout*)
constexpr u64 kEndOfFrame = 0x2ad1eb0;        // the output merger's end of frame (om, context, frame state)
constexpr u64 kCopyToDisplay = 0x2ab6cb0;     // (context, copy state, frame state + 0x18, copy target)
constexpr u64 kFlip = 0x2aafb30;              // gx_command_buffer_flip(cb, handle, buffer, mode, argument)
constexpr u64 kReleaseBlocks = 0x2aaa860;     // the block pools' releases taken back (pools, &cb)
constexpr u64 kResetState = 0x2568530;        // the context's state reset after the flip (context, ...)
constexpr u64 kFlush = 0x2aaf700;             // gx_command_buffer_flush(cb)
constexpr u64 kPresentFrame = 0x2ad6e50;

using One = GUEST_ABI u64 (*)(u64);
using Two = GUEST_ABI u64 (*)(u64, u64);
using Three = GUEST_ABI u64 (*)(u64, u64, u64);
using Four = GUEST_ABI u64 (*)(u64, u64, u64, u64);
using Five = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
using Self = GUEST_ABI u64 (*)();

u64 self() { return game_function<Self>(kPthreadSelf)(); }

}  // namespace

// gx_swap_chain_present_frame(swap chain, frame state, sync interval).
DECOMP_LEAF u32 gx_swap_chain_present_frame(u64 sc, u64 frame, u32 sync) {
    const u64 buffers = load<u32>(sc + kInFlight);
    const u64 ctx = load<u64>(sc + kContext);
    const u64 buffer = static_cast<u64>(load<s64>(sc + kFlips) % static_cast<s64>(buffers));
    if (self() == load<u64>(ctx + kRenderThread)) {
        // At most `buffers` flips in flight: wait while the one that far
        // back has not completed.
        alignas(8) u8 status[0x40];  // SceVideoOutFlipStatus
        alignas(8) u8 event[0x20];   // SceKernelEvent
        s32 got = 0;
        s64 flips;
        for (;;) {
            if (static_cast<s32>(game_function<Two>(kGetFlipStatus)(load<u32>(sc + kVideoOut), reinterpret_cast<u64>(status))) < 0) return 5;
            flips = load<s64>(sc + kFlips);
            if (flips < static_cast<s64>(buffers) ||
                load<s64>(reinterpret_cast<u64>(status) + kStatusFlipArg) >= static_cast<s64>((MUTANT(1) ? 2 : 1) - buffers) + flips)
                break;
            if (static_cast<s32>(game_function<Five>(kWaitEqueue)(load<u64>(sc + kFlipQueue), reinterpret_cast<u64>(event), 1,
                                                                    reinterpret_cast<u64>(&got), 0)) < 0)
                return 5;
        }
        // The display buffer: its Gnm render target's base (the first register, in 256-byte blocks).
        const u64 shown = static_cast<u64>(flips % static_cast<s64>(buffers));
        store<u64>(sc + kDisplayAddress, static_cast<u64>(load<u32>(sc + shown * 0x2c)) << 8);
        game_function<Three>(kEndOfFrame)(ctx + 0xb558, ctx, frame);
        game_function<Four>(kCopyToDisplay)(ctx, sc + kCopyState, frame + 0x18, sc + kCopyTarget);
        const u64 mode = MUTANT(2) ? 1 : 1 + (sync < 1 ? 1 : 0);
        const u32 r = static_cast<u32>(game_function<Five>(kFlip)(ctx + 0x10, load<u32>(sc + kVideoOut), buffer, mode, load<u64>(sc + kFlips)));
        if (r) return r;
        store<s64>(sc + kFlips, load<s64>(sc + kFlips) + 1);
        u64 cb = MUTANT(3) ? ctx : ctx + 0x10;  // the release takes the command buffer by pointer
        game_function<Two>(kReleaseBlocks)(load<u64>(load<u64>(ctx + 0x8) + 0x8), reinterpret_cast<u64>(&cb));
        game_function<One>(kResetState)(ctx);
    }
    // The command buffer flushed, on the render thread (another thread gets 0 and nothing done).
    const u64 again = load<u64>(sc + kContext);
    if (self() == load<u64>(again + kRenderThread)) game_function<One>(kFlush)(again + 0x10);
    return 0;
}

// gx_swap_chain_present(swap chain, sync interval): 8 for an interval past 4.
DECOMP_LEAF u32 gx_swap_chain_present(u64 sc, u32 sync) {
    if (sync > 4) return 8;
    return static_cast<u32>(game_function<Three>(kPresentFrame)(sc, load<u64>(sc + kFrameState), sync));
}

namespace {
// The entries the decomp list places them at.
// mov eax, esi; cmp eax, 4; jbe +6 (the jump's target is past the five bytes)
constexpr u8 kEntryPresent[] = {0x89, 0xf0, 0x83, 0xf8, 0x04};
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x98
constexpr u8 kEntryPresentFrame[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                     0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x98, 0x00, 0x00, 0x00};
}  // namespace

void decomp_gx_present_add() {
    const DecompFunction fns[] = {
        {"gx_swap_chain_present", "gx", 0x2ad5dd0, kEntryPresent, sizeof kEntryPresent, reinterpret_cast<void*>(&gx_swap_chain_present),
         DecompKind::Leaf},
        {"gx_swap_chain_present_frame", "gx", kPresentFrame, kEntryPresentFrame, sizeof kEntryPresentFrame,
         reinterpret_cast<void*>(&gx_swap_chain_present_frame), DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
