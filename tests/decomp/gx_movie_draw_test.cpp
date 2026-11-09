// The movie player's frame draw and close (decomp/gx/movie_draw.cpp) with the game's
// own GXRenderThread events (tests/decomp/eboot_kit.h). GX's manager, its
// immediate context's wrapper and the GXRenderThread object are the test's;
// the render thread's batch and idle events are built as the game builds them
// (its event vtable, a recursive mutex and a condition variable), and the
// game's own set (0x247e780) and wait (the vtable's slot 3, 0x247e8f0) run on
// them, the pthread imports they call bound to the host's. The game's draw
// and close are recorders standing for them: whether the render thread was
// idle when they were reached, and their arguments. Cases:
//
//   - idle: the game's draw at once, its arguments and result passed through;
//   - busy, the render thread going idle 300 ms later: the draw reached only
//     then, and only once its idle event (not just the batch event) is set;
//   - busy, on the render thread itself, on another context, with the wait
//     switched off, or with no GX manager yet: at once;
//   - the close: busy, after the idle event; idle, on the render thread or
//     with the wait off, at once;
//   - busy for good: the draw after the bound, counted.
//
// Four deliberate mistakes must be caught, and the entries hold the
// instructions written against. Skips without the 1.09 eboot.
#define DECOMP_GX_MOVIE_DRAW_MUTANTS
int g_movie_mutant = 0;
#include "decomp/gx/movie_draw.cpp"

#include "eboot_kit.h"

#include <pthread.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>

// What the decomp file links against in the game: the game's functions run on
// this thread, which the kit made a guest thread.
extern "C" GUEST_ABI std::int64_t hle_call_guest6(void* fn, std::int64_t a0, std::int64_t a1, std::int64_t a2, std::int64_t a3,
                                                  std::int64_t a4, std::int64_t a5) {
    return reinterpret_cast<GUEST_ABI std::int64_t (*)(std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                                                       std::int64_t)>(fn)(a0, a1, a2, a3, a4, a5);
}

namespace {

const char* const kTest = "gx_movie_draw_test";
int g_failed = 0;
bool g_quiet = false;

constexpr u64 kEventVtable = 0x56b97a0;  // the game's event class
constexpr u64 kEventSetFn = 0x247e780;   // set(event, state)
constexpr u64 kRenderPthread = 0x7e57a11e;
constexpr u64 kOtherPthread = 0x7e570001;
constexpr u32 kSceTimedOut = 0x8002003c;

void fail(const char* what) {
    ++g_failed;
    if (!g_quiet) std::printf("FAIL: %s\n", what);
}

// ---- The pthread imports the game's events call, over host objects -------
// The game passes the address of its handle slot: the slot holds a pointer
// to the host's object.
pthread_mutex_t* mutex_of(u64 slot) { return reinterpret_cast<pthread_mutex_t*>(static_cast<std::uintptr_t>(load<u64>(slot))); }
pthread_cond_t* cond_of(u64 slot) { return reinterpret_cast<pthread_cond_t*>(static_cast<std::uintptr_t>(load<u64>(slot))); }
GUEST_ABI u64 fake_lock(u64 slot) { return static_cast<u64>(pthread_mutex_lock(mutex_of(slot))); }
GUEST_ABI u64 fake_unlock(u64 slot) { return static_cast<u64>(pthread_mutex_unlock(mutex_of(slot))); }
GUEST_ABI u64 fake_broadcast(u64 slot) { return static_cast<u64>(pthread_cond_broadcast(cond_of(slot))); }
GUEST_ABI u64 fake_cond_wait(u64 cond, u64 mutex) { return static_cast<u64>(pthread_cond_wait(cond_of(cond), mutex_of(mutex))); }
GUEST_ABI u64 fake_cond_timedwait(u64 cond, u64 mutex, u64 usec) {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const std::uint64_t ns = static_cast<std::uint64_t>(ts.tv_nsec) + (usec & 0xffffffffu) * 1000ull;
    ts.tv_sec += static_cast<time_t>(ns / 1000000000ull);
    ts.tv_nsec = static_cast<long>(ns % 1000000000ull);
    const int r = pthread_cond_timedwait(cond_of(cond), mutex_of(mutex), &ts);
    return r == ETIMEDOUT ? kSceTimedOut : static_cast<u64>(r);
}
u64 g_self = kOtherPthread;
GUEST_ABI u64 fake_self() { return g_self; }

// ---- GX's manager, the immediate context, the render thread ----------------
struct alignas(16) Event {
    u64 vtable, mutex, cond;
    u32 flags, pad;
};
struct Gx {
    alignas(16) u8 manager[0x400];
    alignas(16) u8 wrapper[0x40];
    alignas(16) u8 ctx[0x40], other_ctx[0x40];
    alignas(16) u8 render_thread[0x100];
    alignas(16) u8 thread[0x40];
    pthread_mutex_t mutex[2];
    pthread_cond_t cond[2];
} g;

u64 addr(const void* p) { return reinterpret_cast<u64>(p); }
u64 event_at(u64 off) { return addr(g.render_thread) + off; }

void build() {
    std::memset(&g, 0, offsetof(Gx, mutex));
    store<u64>(addr(g.manager) + kGxImmediate, addr(g.wrapper));
    store<u64>(addr(g.wrapper) + 8, addr(g.ctx));
    store<u64>(addr(g.manager) + kGxRenderThread, addr(g.render_thread));
    store<u64>(addr(g.render_thread) + kRenderThreadHandle, addr(g.thread));
    store<u64>(addr(g.thread) + kThreadPthread, kRenderPthread);
    const u64 offs[2] = {kRenderThreadBatchDone, kRenderThreadIdle};
    for (int i = 0; i < 2; ++i) {
        Event e{kEventVtable, addr(&g.mutex[i]), addr(&g.cond[i]), 1u, 0};  // created, not set
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(event_at(offs[i]))), &e, sizeof e);
    }
    const u64 manager = addr(g.manager);
    std::memcpy(eboot_kit::at(kGx), &manager, 8);
}
void set_event(u64 event, bool on) { eboot_kit::fn<GUEST_ABI void (*)(u64, u64)>(kEventSetFn)(event, on ? 1 : 0); }
bool is_set(u64 event) { return (load<u32>(event + kEventFlags) & kEventSet) != 0; }

// ---- The game's draw, recorded --------------------------------------------
struct Seen {
    int calls;
    u64 args[3];
    bool idle_at_call;
    double ms;
} g_seen;
std::chrono::steady_clock::time_point g_t0;
GUEST_ABI u64 recorded_draw(u64 player, u64 ctx, u64 out) {
    ++g_seen.calls;
    g_seen.args[0] = player;
    g_seen.args[1] = ctx;
    g_seen.args[2] = out;
    g_seen.idle_at_call = is_set(event_at(kRenderThreadIdle));
    g_seen.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
    return 0x5eed0001;
}
GUEST_ABI u64 recorded_close(u64 player) {
    ++g_seen.calls;
    g_seen.args[0] = player;
    g_seen.idle_at_call = is_set(event_at(kRenderThreadIdle));
    g_seen.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
    return 0x5eed0002;
}

// One draw (or, with ctx 0, one close): the events as given, the render thread setting them later (after
// `batch_ms` the batch event, after `idle_ms` the idle one; < 0: never).
u64 draw(u64 ctx, bool batch_set, bool idle_set, int batch_ms, int idle_ms) {
    set_event(event_at(kRenderThreadBatchDone), batch_set);
    set_event(event_at(kRenderThreadIdle), idle_set);
    g_seen = {};
    g_t0 = std::chrono::steady_clock::now();
    std::thread render([&] {
        const auto start = std::chrono::steady_clock::now();
        bool batch_done = batch_ms < 0, idle_done = idle_ms < 0;
        while (!batch_done || !idle_done) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (!batch_done && ms >= batch_ms) set_event(event_at(kRenderThreadBatchDone), true), batch_done = true;
            if (!idle_done && ms >= idle_ms) set_event(event_at(kRenderThreadIdle), true), idle_done = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    const u64 r = ctx ? movie_draw(0x9a7e5, ctx, 0x0071) : movie_close(0x9a7e5);
    render.join();
    return r;
}

void check(bool ok, const char* what) {
    if (!ok) fail(what);
}

void run() {
    g_wait = true;
    g_wait_bound = std::chrono::milliseconds(400);
    g_self = kOtherPthread;
    const u64 imm = addr(g.ctx);

    // Idle: at once, everything passed through.
    std::uint64_t waits = g_busy.load();
    u64 r = draw(imm, true, true, -1, -1);
    check(g_seen.calls == 1 && r == 0x5eed0001 && g_seen.args[0] == 0x9a7e5 && g_seen.args[1] == imm && g_seen.args[2] == 0x0071,
          "idle: the game's draw once, its arguments and result passed through");
    check(g_busy.load() == waits && g_seen.ms < 150, "idle: no wait");

    // Busy, idle 300 ms later: the draw only then.
    waits = g_busy.load();
    draw(imm, false, false, 100, 300);
    check(g_seen.calls == 1 && g_seen.idle_at_call && g_seen.ms >= 250, "busy: the draw once the render thread is idle");
    check(g_busy.load() == waits + 1, "busy: the wait counted");

    // The batch done but the flip still running: still waits for idle.
    draw(imm, true, false, -1, 300);
    check(g_seen.calls == 1 && g_seen.idle_at_call && g_seen.ms >= 250, "batch done, flip running: the draw once idle");

    // On the render thread itself: at once.
    g_self = kRenderPthread;
    draw(imm, false, false, -1, 500);
    check(g_seen.calls == 1 && !g_seen.idle_at_call && g_seen.ms < 200, "on the render thread: at once");
    g_self = kOtherPthread;

    // Another context: at once.
    draw(addr(g.other_ctx), false, false, -1, 500);
    check(g_seen.calls == 1 && !g_seen.idle_at_call && g_seen.ms < 200 && g_seen.args[1] == addr(g.other_ctx),
          "another context: at once");

    // The wait switched off: at once.
    g_wait = false;
    draw(imm, false, false, -1, 500);
    check(g_seen.calls == 1 && !g_seen.idle_at_call && g_seen.ms < 200, "BBHOST_MOVIE_DRAW_WAIT=0: at once");
    g_wait = true;

    // No GX manager yet: at once.
    const u64 none = 0;
    std::memcpy(eboot_kit::at(kGx), &none, 8);
    draw(imm, false, false, -1, 500);
    check(g_seen.calls == 1 && g_seen.ms < 200, "no GX manager: at once");
    const u64 manager = addr(g.manager);
    std::memcpy(eboot_kit::at(kGx), &manager, 8);

    // The close: busy, it waits as the draw does; on the render thread, or
    // with the wait off, at once.
    std::uint64_t closes = g_closes_busy.load();
    r = draw(0, false, false, 100, 300);
    check(g_seen.calls == 1 && r == 0x5eed0002 && g_seen.args[0] == 0x9a7e5 && g_seen.idle_at_call && g_seen.ms >= 250,
          "close, busy: the game's close once the render thread is idle");
    check(g_closes_busy.load() == closes + 1, "close, busy: counted");
    draw(0, true, true, -1, -1);
    check(g_seen.calls == 1 && g_seen.ms < 150, "close, idle: at once");
    g_self = kRenderPthread;
    draw(0, false, false, -1, 500);
    check(g_seen.calls == 1 && !g_seen.idle_at_call && g_seen.ms < 200, "close on the render thread: at once");
    g_self = kOtherPthread;
    g_wait = false;
    draw(0, false, false, -1, 500);
    check(g_seen.calls == 1 && !g_seen.idle_at_call && g_seen.ms < 200, "close, BBHOST_MOVIE_DRAW_WAIT=0: at once");
    g_wait = true;

    // Busy for good: the draw after the bound, counted.
    const std::uint64_t bounded = g_bounded.load();
    draw(imm, false, false, -1, -1);
    check(g_seen.calls == 1 && !g_seen.idle_at_call && g_seen.ms >= 390 && g_seen.ms < 2000, "busy for good: the draw after the bound");
    check(g_bounded.load() == bounded + 1, "busy for good: counted");
}

bool caught(int mutant) {
    const int before = g_failed;
    g_movie_mutant = mutant;
    g_quiet = true;
    run();
    g_quiet = false;
    g_movie_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main() {
    eboot_kit::load(kTest);
    bool bound = eboot_kit::bind("scePthreadSelf", reinterpret_cast<const void*>(&fake_self));
    bound = eboot_kit::bind("scePthreadMutexLock", reinterpret_cast<const void*>(&fake_lock)) && bound;
    bound = eboot_kit::bind("scePthreadMutexUnlock", reinterpret_cast<const void*>(&fake_unlock)) && bound;
    bound = eboot_kit::bind("scePthreadCondWait", reinterpret_cast<const void*>(&fake_cond_wait)) && bound;
    bound = eboot_kit::bind("scePthreadCondTimedwait", reinterpret_cast<const void*>(&fake_cond_timedwait)) && bound;
    bound = eboot_kit::bind("pthread_cond_broadcast", reinterpret_cast<const void*>(&fake_broadcast)) && bound;
    if (!bound) {
        std::printf("FAIL: an import not bound\n");
        return 1;
    }
    for (int i = 0; i < 2; ++i) {
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);  // the game's events take a recursive mutex
        pthread_mutex_init(&g.mutex[i], &a);
        pthread_cond_init(&g.cond[i], nullptr);
    }
    build();
    g_game = reinterpret_cast<void*>(&recorded_draw);
    g_game_close = reinterpret_cast<void*>(&recorded_close);
    const u8* at = eboot_kit::file_at(kTest, 0x2cfe7c0);
    if (!at || std::memcmp(at, kEntry, sizeof kEntry) != 0) fail("the draw's entry's instructions");
    at = eboot_kit::file_at(kTest, 0x2cfe2e0);
    if (!at || std::memcmp(at, kEntryClose, sizeof kEntryClose) != 0) fail("the close's entry's instructions");
    run();
    for (const int mutant : {1, 2, 3, 4}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("%s: %d failures\n", kTest, g_failed);
        return 1;
    }
    std::printf("%s: ok - 12 cases, 4 mutants caught\n", kTest);
    return 0;
}
