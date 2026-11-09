// The movie player's frame draw and its close, on GX's immediate context
// only once GXRenderThread is idle (docs/decomp.md). The movie player
// (MVMoviePlayer, as Dark Souls III's RTTI names the class) has them in its
// slots 14 and 6:
//
//   - sub_2cfe7c0, the frame draw: the AvPlayer's current frame drawn into
//     the movie's render target. It saves the context's pending state (the
//     shaders at +0xb730 / +0xc1b8 / +0xd598, the constant buffer slots, the
//     blend, depth and target records), puts its own in, updates its constant
//     buffer with UpdateSubresource (a rename into the context's map,
//     +0xb3c0), copies the frame's planes into its textures, draws, and puts
//     the saved state back. CSMovieIns::STEP_Wait_Play (0x23ce3b0), the
//     movie's step, calls it every frame on the main thread with GX's
//     immediate context - the one at +8 of the wrapper at +0x10 of GX's
//     manager (0x59406c8);
//   - sub_2cfe2e0, the close: everything it made released - its constant
//     buffers (each release takes the buffer's renames out of every context's
//     map, the immediate one's among them), textures, views and shaders - and
//     the AvPlayer closed. CSMovieIns::STEP_Finish_Play calls it on the main
//     thread when the movie ends or is skipped.
//
// GXRenderThread works on the immediate context at the same time: the
// previous frame's batch (sub_2599af0: the frame's command lists executed,
// the context's pending state cleared after them, the full-screen passes
// drawn from it, its rename map resolved) and the present. Nothing locks the
// immediate context. The game's other main-thread code that touches what the
// render thread works on waits for the render thread's idle event first (+0x68
// of the GXRenderThread object at +0x348 of GX's manager: sub_25d9df0,
// sub_26c53e0, the "GAME REND" teardown 0x1971340), and only the main thread
// hands the render thread work (sub_2598fd0, sub_25b6aa0), so the render
// thread stays idle until it does. The movie's calls do not wait. On the
// console the render thread's batch is long done when the movie's step comes;
// here, where GX's work costs more and the GPU can be the bottleneck, the two
// overlap - on a Steam Deck at 60 fps all but one of the attract movie's
// 9,789 draws began while the render thread was busy - and the title's
// attract movie could crash the game a minute or two into playing (most
// often with the F10 menu open, whose drawing makes each frame slower), in
// either of two ways:
//
//   - the draw's restore landed between a full-screen pass's shader bind
//     (sub_2591c30) and its draw, which then committed a pixel shader the
//     restore had put back as none: a read of +0xd2 of null in the commit
//     (0x2ac01c3);
//   - its constant buffer's rename went into the map while the render thread
//     was resolving it (and freeing its entries' chunks), and the resolve
//     wrote through a stale entry's rename-slot table, null (0x2aae211).
//
// So ours - for a draw on the immediate context, and for the close, from any
// thread but the render thread (scePthreadSelf against the render thread's,
// as sub_25c1b90 asks) - waits for the render thread's idle event, as those
// callers do, then runs the game's function. The wait is the game's own (the
// event's wait, slot 3) in steps of 100 ms, and bounded: after five seconds
// the call goes ahead as before, which is logged. Everything else is the
// game's. BBHOST_MOVIE_DRAW_WAIT=0: the game's functions at once, as before;
// the exit report counts the calls that found the render thread busy either
// way.
//
// Hosted: it calls the game's functions and the host's clock and log.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "core/thunk.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

using namespace decomp;

// tests/decomp/gx_movie_draw_test.cpp builds this file with mutants: small
// wrong turns it switches on one at a time and must catch.
#ifdef DECOMP_GX_MOVIE_DRAW_MUTANTS
extern int g_movie_mutant;
#define MUTANT(n) (g_movie_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; and rsp, -32
// (20 bytes; the stack-guard load and the frame follow).
constexpr std::uint8_t kEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                   0x41, 0x54, 0x53, 0x48, 0x81, 0xe4, 0xe0, 0xff, 0xff, 0xff};
// push rbp; mov rbp, rsp; push r15, r14, rbx, rax; mov rbx, rdi (13 bytes;
// the call of its slot 9 follows).
constexpr std::uint8_t kEntryClose[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x48, 0x89, 0xfb};

constexpr u64 kGx = 0x59406c8;              // GX's manager (a pointer to it)
constexpr u64 kGxImmediate = 0x10;          // its immediate context's wrapper; the context at +8
constexpr u64 kGxRenderThread = 0x348;      // the GXRenderThread object
constexpr u64 kRenderThreadHandle = 0x8;    // of that object: its thread, whose pthread is at +0x18
constexpr u64 kThreadPthread = 0x18;
constexpr u64 kRenderThreadBatchDone = 0x48;  // its batch event: set once a batch is done (the flip may still be running)
constexpr u64 kRenderThreadIdle = 0x68;     // its idle event: set once a batch and a flip are done
constexpr u64 kEventFlags = 0x18;           // an event's word: bit 1 set while it is signalled
constexpr u32 kEventSet = 2;
constexpr u64 kEventWait = 0x18;            // the event's wait(event, microseconds; 0xffffffff for ever), vtable slot 3
constexpr u64 kPthreadSelf = 0x248ba90;     // j_scePthreadSelf
constexpr u32 kWaitStepUs = 100000;

bool g_wait = [] {
    const char* e = std::getenv("BBHOST_MOVIE_DRAW_WAIT");
    return !(e && e[0] == '0');
}();
std::chrono::milliseconds g_wait_bound{5000};

void* g_game = nullptr;        // the game's draw
void* g_game_close = nullptr;  // the game's close
std::atomic<std::uint64_t> g_draws{0}, g_immediate{0}, g_busy{0}, g_closes{0}, g_closes_busy{0};
std::atomic<std::uint64_t> g_bounded{0}, g_waited_us{0}, g_longest_us{0};

bool idle(u64 event) { return (load<u32>(event + kEventFlags) & kEventSet) != 0; }

// The render thread's idle event waited for the way the game's own callers
// wait for it: true once it is set, false when the bound ran out first.
bool wait_idle(u64 event) {
    void* const wait = reinterpret_cast<void*>(static_cast<std::uintptr_t>(load<u64>(load<u64>(event) + kEventWait)));
    const auto t0 = std::chrono::steady_clock::now();
    while (!idle(event) && !MUTANT(1)) {
        if (std::chrono::steady_clock::now() - t0 > g_wait_bound) return false;
        hle_call_guest<std::uint64_t>(wait, event, kWaitStepUs);
    }
    return true;
}

// Whether the render thread is busy with the context and the caller is not
// it - and, with the wait on, the wait for it to go idle. `what` and `n`
// name the call for the log; `busy` counts it.
void before_render_thread(u64 rt, const char* what, std::uint64_t n, std::atomic<std::uint64_t>& busy) {
    const u64 event = rt + (MUTANT(2) ? kRenderThreadBatchDone : kRenderThreadIdle);
    if (idle(event)) return;
    const u64 thread = load<u64>(rt + kRenderThreadHandle);
    if (!MUTANT(4) && thread &&
        load<u64>(thread + kThreadPthread) == hle_call_guest<std::uint64_t>(reinterpret_cast<void*>(game_address(kPthreadSelf))))
        return;
    const std::uint64_t k = busy.fetch_add(1, std::memory_order_relaxed);
    if (!g_wait) {
        if (k == 0)
            host_log("decomp: the movie's %s %llu began while GXRenderThread was busy with the immediate context "
                     "(BBHOST_MOVIE_DRAW_WAIT=0: no wait)", what, static_cast<ull>(n));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool done = wait_idle(event);
    const auto us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    g_waited_us.fetch_add(us, std::memory_order_relaxed);
    std::uint64_t longest = g_longest_us.load(std::memory_order_relaxed);
    while (us > longest && !g_longest_us.compare_exchange_weak(longest, us, std::memory_order_relaxed)) {
    }
    if (!done) {
        if (g_bounded.fetch_add(1, std::memory_order_relaxed) < 8)
            host_log("decomp: the movie's %s %llu waited %llu ms for GXRenderThread to go idle and goes ahead without it", what,
                     static_cast<ull>(n), static_cast<ull>(us / 1000));
    } else if (k == 0) {
        host_log("decomp: the movie's %s %llu waited %llu us for GXRenderThread to finish its frame first", what, static_cast<ull>(n),
                 static_cast<ull>(us));
    }
}

// GX's manager and its render thread, or 0 before they are made.
u64 render_thread(u64* immediate) {
    const u64 gx = load<u64>(game_address(kGx));
    const u64 wrapper = gx ? load<u64>(gx + kGxImmediate) : 0;
    *immediate = wrapper ? load<u64>(wrapper + 8) : 0;
    return gx && wrapper ? load<u64>(gx + kGxRenderThread) : 0;
}

// sub_2cfe7c0(movie player, GX context, out: the movie's texture view):
// true when it drew a frame.
GUEST_ABI std::uint64_t movie_draw(std::uint64_t player, std::uint64_t ctx, std::uint64_t out) {
    const std::uint64_t n = g_draws.fetch_add(1, std::memory_order_relaxed);
    u64 immediate = 0;
    const u64 rt = render_thread(&immediate);
    if (rt && ctx && (ctx == immediate || MUTANT(3))) {
        g_immediate.fetch_add(1, std::memory_order_relaxed);
        before_render_thread(rt, "draw", n, g_busy);
    }
    return hle_call_guest<std::uint64_t>(g_game, player, ctx, out);
}

// sub_2cfe2e0(movie player): true when it was open.
GUEST_ABI std::uint64_t movie_close(std::uint64_t player) {
    const std::uint64_t n = g_closes.fetch_add(1, std::memory_order_relaxed);
    u64 immediate = 0;
    if (const u64 rt = render_thread(&immediate)) before_render_thread(rt, "close", n, g_closes_busy);
    return hle_call_guest<std::uint64_t>(g_game_close, player);
}

void report() {
    if (!g_draws.load() && !g_closes.load()) return;
    if (g_wait) {
        host_log("decomp: the movie player's frame draws: %llu, %llu on the immediate context, %llu of them found GXRenderThread busy "
                 "with it; closes: %llu, %llu found it busy; each waited for it to go idle, %llu ms in all, the longest %llu ms; "
                 "%llu went ahead after the bound",
                 static_cast<ull>(g_draws.load()), static_cast<ull>(g_immediate.load()), static_cast<ull>(g_busy.load()),
                 static_cast<ull>(g_closes.load()), static_cast<ull>(g_closes_busy.load()), static_cast<ull>(g_waited_us.load() / 1000),
                 static_cast<ull>(g_longest_us.load() / 1000), static_cast<ull>(g_bounded.load()));
    } else {
        host_log("decomp: the movie player's frame draws: %llu, %llu on the immediate context, %llu of them began while GXRenderThread "
                 "was busy with it; closes: %llu, %llu began while it was busy (BBHOST_MOVIE_DRAW_WAIT=0: no wait)",
                 static_cast<ull>(g_draws.load()), static_cast<ull>(g_immediate.load()), static_cast<ull>(g_busy.load()),
                 static_cast<ull>(g_closes.load()), static_cast<ull>(g_closes_busy.load()));
    }
}

}  // namespace

void decomp_gx_movie_draw_add() {
    DecompFunction draw{"sub_2cfe7c0", "gx", 0x2cfe7c0, kEntry, sizeof(kEntry), reinterpret_cast<void*>(&movie_draw), DecompKind::Hosted};
    draw.original = &g_game;
    draw.report = &report;
    decomp_add(draw);
    DecompFunction close{"sub_2cfe2e0", "gx", 0x2cfe2e0, kEntryClose, sizeof(kEntryClose), reinterpret_cast<void*>(&movie_close),
                         DecompKind::Hosted};
    close.original = &g_game_close;
    decomp_add(close);
}
