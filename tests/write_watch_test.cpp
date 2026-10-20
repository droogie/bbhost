// core/write_watch.cpp: arming, faults, dirty checks, releases and remaps on
// plain host memory, and the property the texture cache relies on - a watch
// that reads clean means the memory is what was read after arming it.
#include "core/write_watch.h"
#include "core/host_ucontext.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
int g_fail = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

constexpr std::size_t kPage = 4096;

void on_segv(int sig, siginfo_t* info, void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    if (write_watch_on_fault(reinterpret_cast<std::uint64_t>(info->si_addr), (UC_REG(uc, ERR) & 2) != 0)) {
        return;
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

std::uint64_t faults() { return write_watch_stats().faults; }
}  // namespace

int main() {
    if (!write_watch_enabled()) {
        std::printf("write watch disabled (BBHOST_WRITE_WATCH=0); nothing to test\n");
        return 0;
    }
    struct sigaction sa{};
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
#if defined(__APPLE__)
    sigaction(SIGBUS, &sa, nullptr);  // macOS's signal for a write to a read-only page
#endif

    // Below 2^40, where guest memory lives and the watch's table reaches.
    void* m = MAP_FAILED;
    for (std::uint64_t hint = 0x7000000000ull; m == MAP_FAILED && hint < 0x8000000000ull; hint += 0x10000000ull) {
#if defined(__APPLE__)
        // No MAP_FIXED_NOREPLACE on macOS: the hint or nothing.
        m = mmap(reinterpret_cast<void*>(hint), 16 * kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED && m != reinterpret_cast<void*>(hint)) {
            munmap(m, 16 * kPage);
            m = MAP_FAILED;
        }
#else
        m = mmap(reinterpret_cast<void*>(hint), 16 * kPage, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
#endif
    }
    CHECK(m != MAP_FAILED);
    if (m == MAP_FAILED) return 1;
    auto* base = static_cast<volatile std::uint8_t*>(m);
    const auto va = reinterpret_cast<std::uint64_t>(m);
    auto page = [&](int i) { return base + i * kPage; };

    // Arm, write, see it.
    WriteWatch a;
    CHECK(write_watch_arm_locked(va, 3 * kPage, a));
    CHECK(write_watch_dirty(va, 3 * kPage, a) == 0);
    std::uint64_t f0 = faults();
    page(1)[10] = 7;
    CHECK(faults() == f0 + 1);
    CHECK(page(1)[10] == 7);  // the write went through
    page(1)[11] = 8;          // writable now: no second fault
    CHECK(faults() == f0 + 1);
    CHECK(write_watch_dirty(va, 3 * kPage, a) == 1);

    // Re-arming clears it; two pages written count two.
    CHECK(write_watch_arm_locked(va, 3 * kPage, a));
    CHECK(write_watch_dirty(va, 3 * kPage, a) == 0);
    page(0)[0] = 1;
    page(2)[0] = 1;
    CHECK(write_watch_dirty(va, 3 * kPage, a) == 2);

    // Two watches sharing a page: re-arming one must not hide the write from
    // the other.
    WriteWatch x, y;
    CHECK(write_watch_arm_locked(va + 4 * kPage, 2 * kPage, x));  // pages 4-5
    CHECK(write_watch_arm_locked(va + 5 * kPage, 2 * kPage, y));  // pages 5-6
    page(5)[1] = 3;
    CHECK(write_watch_arm_locked(va + 4 * kPage, 2 * kPage, x));
    CHECK(write_watch_dirty(va + 4 * kPage, 2 * kPage, x) == 0);
    CHECK(write_watch_dirty(va + 5 * kPage, 2 * kPage, y) == 1);

    // The kernel writing a protected page fails rather than faulting, until
    // the range is released - which counts as a write.
    WriteWatch r;
    CHECK(write_watch_arm_locked(va + 8 * kPage, kPage, r));
    int fds[2];
    CHECK(pipe(fds) == 0);
    CHECK(write(fds[1], "abcd", 4) == 4);
    errno = 0;
    CHECK(read(fds[0], const_cast<std::uint8_t*>(page(8)), 4) == -1 && errno == EFAULT);
    CHECK(write_watch_release(const_cast<std::uint8_t*>(page(8)), 4));
    CHECK(read(fds[0], const_cast<std::uint8_t*>(page(8)), 4) == 4);
    CHECK(std::memcmp(const_cast<std::uint8_t*>(page(8)), "abcd", 4) == 0);
    CHECK(write_watch_dirty(va + 8 * kPage, kPage, r) == 1);
    CHECK(!write_watch_release(const_cast<std::uint8_t*>(page(12)), kPage));  // never watched: nothing to do
    close(fds[0]);
    close(fds[1]);

    // A remap counts as a write.
    CHECK(write_watch_arm_locked(va + 8 * kPage, kPage, r));
    write_watch_forget(va + 8 * kPage, kPage);
    CHECK(write_watch_dirty(va + 8 * kPage, kPage, r) == 1);

    // Faults that are not the watch's are left alone.
    CHECK(!write_watch_on_fault(va + 12 * kPage, true));  // never armed
    CHECK(!write_watch_on_fault(va + kPage, false));       // a read

    // The property: a writer hammers one page while the checker arms, reads,
    // and checks. Whenever the check says clean, the page still holds what
    // was read after arming - and no increment is lost to a retried fault.
    auto* counter = reinterpret_cast<volatile std::uint64_t*>(page(14));
    *counter = 0;
    std::atomic<bool> stop{false};
    std::uint64_t written = 0;
    std::thread writer([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            *counter = *counter + 1;
            ++written;
            for (int k = 0; k < 200; ++k) asm volatile("pause");
        }
    });
    int clean = 0, dirty = 0, wrong = 0;
    WriteWatch s;
    for (int i = 0; i < 20000; ++i) {
        CHECK(write_watch_arm_locked(va + 14 * kPage, kPage, s));
        const std::uint64_t v = *counter;
        for (int k = 0; k < 100; ++k) asm volatile("pause");
        const std::uint64_t v2 = *counter;
        if (write_watch_dirty(va + 14 * kPage, kPage, s)) {
            ++dirty;
        } else {
            ++clean;
            if (v2 != v) ++wrong;
        }
    }
    stop.store(true);
    writer.join();
    CHECK(wrong == 0);
    CHECK(dirty > 0);
    CHECK(*counter == written);
    std::printf("write_watch_test: %d clean, %d dirty checks, %llu writes, %llu faults\n", clean, dirty,
                static_cast<unsigned long long>(written), static_cast<unsigned long long>(faults()));
    if (g_fail) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("write_watch_test: ok\n");
    return 0;
}
