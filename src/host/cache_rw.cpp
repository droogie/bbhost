// Stage 4b.1a's lock (host/cache_rw.h): a reader-writer lock that prefers its
// one writer, the switch that turns the readers on, and its counts.
#include "host/cache_rw.h"

#include "host/gxvk_lists.h"

#include <atomic>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cache_rw {

const bool g_readers = gxvk_lists::pool_threads() > 0;
thread_local constinit bool t_reader = false;
thread_local constinit int t_write_depth = 0;
std::atomic<std::uint64_t> g_gen[kKinds] = {};

namespace {

// Bit 31: the writer holds the lock or waits for its readers to leave; bits
// 0-30: the readers holding it.
constexpr std::uint32_t kWriter = 1u << 31;
std::atomic<std::uint32_t> g_state{0};
std::atomic<std::uint64_t> g_writes{0}, g_writes_waited{0}, g_reads{0}, g_reads_waited{0};

void backoff(unsigned& spins) {
    if (++spins < 64) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
        return;
    }
    std::this_thread::yield();
}

}  // namespace

void lock_slow() {
    unsigned spins = 0;
    bool waited = false;
    for (;;) {
        std::uint32_t s = g_state.load(std::memory_order_relaxed);
        if (!(s & kWriter) && g_state.compare_exchange_weak(s, s | kWriter, std::memory_order_acquire, std::memory_order_relaxed)) break;
        waited = true;
        backoff(spins);
    }
    // The readers under way finish their draw's lookups; new ones wait for us.
    while (g_state.load(std::memory_order_acquire) & ~kWriter) {
        waited = true;
        backoff(spins);
    }
    g_writes.fetch_add(1, std::memory_order_relaxed);
    if (waited) g_writes_waited.fetch_add(1, std::memory_order_relaxed);
}

void unlock_slow() { g_state.fetch_and(~kWriter, std::memory_order_release); }

void lock_shared_slow() {
    unsigned spins = 0;
    bool waited = false;
    for (;;) {
        std::uint32_t s = g_state.load(std::memory_order_relaxed);
        if (!(s & kWriter)) {
            if (g_state.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) break;
            continue;
        }
        waited = true;
        backoff(spins);
    }
    g_reads.fetch_add(1, std::memory_order_relaxed);
    if (waited) g_reads_waited.fetch_add(1, std::memory_order_relaxed);
}

void unlock_shared_slow() { g_state.fetch_sub(1, std::memory_order_release); }

Counts counts() {
    Counts c;
    c.writes = g_writes.load(std::memory_order_relaxed);
    c.writes_waited = g_writes_waited.load(std::memory_order_relaxed);
    c.reads = g_reads.load(std::memory_order_relaxed);
    c.reads_waited = g_reads_waited.load(std::memory_order_relaxed);
    return c;
}

}  // namespace cache_rw
