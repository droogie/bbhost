// DLAdaptiveMutex::Lock (decomp/sync/adaptive_mutex.cpp) beside the game's own
// (tests/decomp/eboot_kit.h): threads take one lock word with the game's Lock,
// ours, or either at random per call, and give it back with the game's Unlock
// (sub_2436a40). A counter only the holder may touch must come out exact, no
// two holders may overlap, and the word must be free at the end. The sleeps
// are real - a host semaphore stands in for sceKernelWaitSema/SignalSema - at
// spin budgets from none (every contender sleeps) to long. Single-thread
// cases check the word's transitions: a free lock taken, the owner's own
// re-lock, the sleepers' count kept. Skips without the 1.09 eboot.
#include "decomp/sync/adaptive_mutex.cpp"

#include "eboot_kit.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr u64 kUnlock = 0x2436a40;
// push rbp; mov rbp, rsp; lea rdx, [rdi+0x18]
constexpr std::uint8_t kUnlockEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x48, 0x8d, 0x57, 0x18};

thread_local u64 t_self = 0;
std::atomic<u64> g_next_self{0x1001};
GUEST_ABI u64 stub_self() { return t_self; }

struct Sema {
    std::mutex m;
    std::condition_variable cv;
    s64 count = 0;
};
std::mutex g_semas_mu;
std::map<u64, Sema*> g_semas;
std::atomic<u64> g_sleeps{0};
Sema& sema(u64 h) {
    std::lock_guard<std::mutex> l(g_semas_mu);
    Sema*& s = g_semas[h & 0xffffffffull];
    if (!s) s = new Sema;
    return *s;
}
GUEST_ABI s32 stub_wait(u64 h, s32 need, void*) {
    g_sleeps.fetch_add(1, std::memory_order_relaxed);
    Sema& s = sema(h);
    std::unique_lock<std::mutex> l(s.m);
    s.cv.wait(l, [&] { return s.count >= need; });
    s.count -= need;
    return 0;
}
GUEST_ABI s32 stub_signal(u64 h, s32 n) {
    Sema& s = sema(h);
    {
        std::lock_guard<std::mutex> l(s.m);
        s.count += n;
    }
    s.cv.notify_all();
    return 0;
}

using LockFn = GUEST_ABI s32 (*)(u64);
using UnlockFn = GUEST_ABI s32 (*)(u64);

struct alignas(64) GameMutex {
    u64 vtable;
    s32 spin;  // +0x8
    s32 pad;
    u64 sema;  // +0x10
    u64 word;  // +0x18
};
static_assert(offsetof(GameMutex, spin) == 0x8 && offsetof(GameMutex, sema) == 0x10 && offsetof(GameMutex, word) == 0x18);

LockFn g_game_lock;
UnlockFn g_game_unlock;
const LockFn g_our_lock = &adaptive_mutex_lock;

// mode 0: the game's Lock, 1: ours, 2: either at random per call.
int stress(const char* what, int mode, s32 spin, int threads, int iters, double* ms) {
    static u64 next_sema = 100;
    GameMutex mu{};
    mu.spin = spin;
    mu.sema = next_sema++;
    const u64 m = reinterpret_cast<u64>(&mu);
    volatile u64 counter = 0;
    std::atomic<int> inside{0}, overlaps{0};
    std::vector<std::thread> ts;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&, t] {
            t_self = g_next_self.fetch_add(1);
            std::mt19937 rng(static_cast<unsigned>(t) * 7919u + 1);
            for (int i = 0; i < iters; ++i) {
                const bool ours = mode == 1 || (mode == 2 && (rng() & 1));
                if ((ours ? g_our_lock : g_game_lock)(m) != 0) overlaps.fetch_add(1000000);
                if (inside.fetch_add(1) != 0) overlaps.fetch_add(1);
                const u64 c = counter;
                for (int k = 0; k < static_cast<int>(rng() % 8); ++k) _mm_pause();
                counter = c + 1;
                inside.fetch_sub(1);
                g_game_unlock(m);
            }
        });
    }
    for (std::thread& th : ts) th.join();
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const u64 want = static_cast<u64>(threads) * static_cast<u64>(iters);
    const bool ok = counter == want && overlaps.load() == 0 && mu.word == 0;
    std::printf("  %-30s spin %6d, %d threads x %d: counter %llu of %llu, overlaps %d, word %016llx, %.0f ms - %s\n", what, spin,
                threads, iters, static_cast<ull>(counter), static_cast<ull>(want), overlaps.load(), static_cast<ull>(mu.word), *ms,
                ok ? "ok" : "WRONG");
    return ok ? 0 : 1;
}

// One thread: the word after each call, ours against the game's.
int transitions() {
    int bad = 0;
    t_self = 0x777;
    const u64 cases[] = {0, 0x777, 0x10000000000ull, 0x30000000000ull | 0x777};  // free, ours, free with sleepers, ours with sleepers
    for (const u64 start : cases) {
        GameMutex a{}, b{};
        a.spin = b.spin = 1000;
        a.word = b.word = start;
        const s32 ra = g_game_lock(reinterpret_cast<u64>(&a));
        const s32 rb = g_our_lock(reinterpret_cast<u64>(&b));
        if (ra != rb || a.word != b.word) {
            std::printf("  from %016llx: the game's gives %d, word %016llx; ours %d, word %016llx\n", static_cast<ull>(start), ra,
                        static_cast<ull>(a.word), rb, static_cast<ull>(b.word));
            ++bad;
        }
    }
    std::printf("  single-thread transitions from 4 words: %d differ\n", bad);
    return bad;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_adaptive_mutex");
    if (std::memcmp(eboot_kit::at(kLock), kEntry, sizeof kEntry) != 0 || std::memcmp(eboot_kit::at(kUnlock), kUnlockEntry, sizeof kUnlockEntry) != 0) {
        std::printf("decomp_adaptive_mutex: Lock or Unlock is not where the decomp expects it\n");
        return 1;
    }
    if (!eboot_kit::bind("scePthreadSelf", reinterpret_cast<const void*>(&stub_self)) ||
        !eboot_kit::bind("sceKernelWaitSema", reinterpret_cast<const void*>(&stub_wait)) ||
        !eboot_kit::bind("sceKernelSignalSema", reinterpret_cast<const void*>(&stub_signal))) {
        std::printf("decomp_adaptive_mutex: an import to stand in for is missing\n");
        return 1;
    }
    g_game_lock = eboot_kit::fn<LockFn>(kLock);
    g_game_unlock = eboot_kit::fn<UnlockFn>(kUnlock);
    int bad = transitions();
    const int threads = static_cast<int>(std::min(8u, std::max(2u, std::thread::hardware_concurrency())));
    double game_ms = 0, ours_ms = 0, ms = 0;
    for (const s32 spin : {0, 2000, 200000}) {
        g_sleeps = 0;
        bad += stress("the game's Lock", 0, spin, threads, 20000, &game_ms);
        const u64 game_sleeps = g_sleeps.exchange(0);
        bad += stress("ours", 1, spin, threads, 20000, &ours_ms);
        const u64 our_sleeps = g_sleeps.exchange(0);
        bad += stress("either, at random per call", 2, spin, threads, 20000, &ms);
        std::printf("  spin %d: sleeps %llu (the game's) against %llu (ours)\n", spin, static_cast<ull>(game_sleeps),
                    static_cast<ull>(our_sleeps));
    }
    std::printf("decomp_adaptive_mutex: %s\n", bad ? "FAILED" : "every lock exclusive, every count exact, every word free at the end");
    return bad ? 1 : 0;
}
