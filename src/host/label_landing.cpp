// Labels written by the host once their submission has run (label_landing.h).
#include "host/label_landing.h"

#include "core/portable.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <thread>

namespace label_landing {
namespace {

// BBHOST_LABEL_LANDING: 1 lands, 0 keeps the GPU fills. By default it lands,
// but on Windows, where the thread polls (below) and no Windows machine has
// measured that yet.
const bool g_on = [] {
    const char* e = std::getenv("BBHOST_LABEL_LANDING");
    if (e && e[0]) return e[0] == '1';
#if defined(_WIN32)
    return false;
#else
    return true;
#endif
}();

// BBHOST_LABEL_WAIT=block|poll: how the thread waits for the GPU. Block (the
// driver's wait on the timeline, also for a submission not yet in the queue:
// a timeline semaphore may be waited on before its signal is submitted) but
// on Windows, where NVIDIA's driver holds its lock through that wait and every
// other thread's call - the submission thread's - queues behind it (gpu.cpp,
// present_reads_wait).
const bool g_block = [] {
    const char* e = std::getenv("BBHOST_LABEL_WAIT");
    if (e && std::strcmp(e, "poll") == 0) return false;
    if (e && std::strcmp(e, "block") == 0) return true;
#if defined(_WIN32)
    return false;
#else
    return true;
#endif
}();
const std::uint64_t g_poll_us = [] {
    const char* e = std::getenv("BBHOST_LABEL_POLL_US");
    const long v = e ? std::strtol(e, nullptr, 10) : 50;
    return static_cast<std::uint64_t>(v > 0 ? v : 50);
}();

struct Batch {
    std::uint64_t serial;
    std::uint32_t count;  // its entries, in g_writes
    std::uint64_t added_ns;
};
// A label (its bytes inline) or a copy (add_copy: bytes from a host buffer
// the submission's GPU work wrote), in the order they land.
struct Entry {
    std::uint64_t va;
    std::uint32_t bytes;
    bool copy;
    std::uint8_t data[8];
    const void* src;
};

// A queue in a ring that doubles when full: no allocation a batch (a deque's
// block came and went every few submissions, on the command processor that
// adds and on the thread that lands).
template <typename T>
class Ring {
public:
    explicit Ring(std::size_t cap) : buf_(cap) {}
    bool empty() const { return n_ == 0; }
    std::size_t size() const { return n_; }
    T& front() { return buf_[head_]; }
    T& back() { return buf_[(head_ + n_ - 1) & (buf_.size() - 1)]; }
    void push_back(const T& v) {
        if (n_ == buf_.size()) grow();
        buf_[(head_ + n_) & (buf_.size() - 1)] = v;
        ++n_;
    }
    void pop_front() {
        head_ = (head_ + 1) & (buf_.size() - 1);
        --n_;
    }

private:
    void grow() {
        std::vector<T> bigger(buf_.size() * 2);
        for (std::size_t k = 0; k < n_; ++k) bigger[k] = buf_[(head_ + k) & (buf_.size() - 1)];
        buf_.swap(bigger);
        head_ = 0;
    }
    std::vector<T> buf_;  // a power of two
    std::size_t head_ = 0, n_ = 0;
};

Hooks g_hooks;
// Never destroyed: the thread waits on them past exit's static destructors (a
// condition variable destroyed under a waiter blocks the exit).
struct State {
    std::mutex mu;
    std::condition_variable work_cv;    // the thread: a batch to land
    std::condition_variable landed_cv;  // wait_landed_past: something landed
    Ring<Batch> batches{1024};          // in serial order, under mu
    Ring<Entry> writes{16384};          // the batches' entries, in order, under mu
};
State& g_st = *new State;
std::mutex& g_mu = g_st.mu;
std::condition_variable& g_work_cv = g_st.work_cv;
std::condition_variable& g_landed_cv = g_st.landed_cv;
Ring<Batch>& g_batches = g_st.batches;
Ring<Entry>& g_writes = g_st.writes;
int g_landed_waiters = 0;  // under g_mu: wait_landed_past's (the arena acquire's) threads
bool g_idle = false;       // under g_mu: the thread waits for a batch (only then does add() wake it)
std::atomic<std::uint64_t> g_landed{0};
std::atomic<bool> g_started{false};
bool g_on_thread = false;  // under g_mu: the thread is landing (for the counts)

// Counts, under g_mu unless atomic.
std::uint64_t g_n_writes = 0, g_n_batches = 0, g_by_thread = 0, g_by_others = 0, g_unmapped = 0, g_most_batches = 0, g_most_writes = 0;
std::uint64_t g_n_copies = 0, g_copy_bytes = 0;
std::atomic<std::uint64_t> g_waits{0}, g_polls{0}, g_unsubmitted{0};
// Submit to landed, in 0.1 ms buckets up to 256 ms (the last bucket: more).
constexpr int kBuckets = 2561;
constexpr std::uint64_t kBucketNs = 100000;
std::uint64_t g_ages[kBuckets] = {};

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void store(const Entry& w) {
    if (g_hooks.mapped && !g_hooks.mapped(w.va, w.bytes)) {
        ++g_unmapped;
        return;
    }
    void* const p = reinterpret_cast<void*>(static_cast<std::uintptr_t>(w.va));
    if (w.copy) {
        std::memcpy(p, w.src, w.bytes);  // the labels after it store with release
        ++g_n_copies;
        g_copy_bytes += w.bytes;
        return;
    }
    if (w.bytes == 4 && !(w.va & 3)) {
        std::uint32_t v;
        std::memcpy(&v, w.data, 4);
        __atomic_store_n(static_cast<std::uint32_t*>(p), v, __ATOMIC_RELEASE);
    } else if (w.bytes == 8 && !(w.va & 7)) {
        std::uint64_t v;
        std::memcpy(&v, w.data, 8);
        __atomic_store_n(static_cast<std::uint64_t*>(p), v, __ATOMIC_RELEASE);
    } else {
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(p, w.data, w.bytes);
    }
}

// Lands every batch of a submission before `completed`, in order.
void land_locked(std::uint64_t completed) {
    bool any = false;
    const std::uint64_t t = g_batches.empty() ? 0 : now_ns();
    while (!g_batches.empty() && g_batches.front().serial < completed) {
        const Batch b = g_batches.front();
        g_batches.pop_front();
        for (std::uint32_t k = 0; k < b.count; ++k) {
            store(g_writes.front());
            g_writes.pop_front();
        }
        g_n_writes += b.count;  // (copies among them counted apart in store)
        ++g_n_batches;
        ++(g_on_thread ? g_by_thread : g_by_others);
        const std::uint64_t age = t > b.added_ns ? (t - b.added_ns) / kBucketNs : 0;
        ++g_ages[std::min<std::uint64_t>(age, kBuckets - 1)];
        any = true;
    }
    if (completed > g_landed.load(std::memory_order_relaxed)) {
        g_landed.store(completed, std::memory_order_release);
        any = true;
    }
    if (any && g_landed_waiters) g_landed_cv.notify_all();
}

void thread_main() {
    if (g_hooks.thread_start) g_hooks.thread_start();
    for (;;) {
        std::uint64_t front;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_idle = true;
            g_work_cv.wait(lk, [] { return !g_batches.empty(); });
            g_idle = false;
            front = g_batches.front().serial;
        }
        const std::uint64_t done = g_hooks.completed ? g_hooks.completed() : ~0ull;
        if (done > front) {
            std::lock_guard<std::mutex> lk(g_mu);
            g_on_thread = true;
            land_locked(done);
            g_on_thread = false;
            continue;
        }
        if (g_block && g_hooks.wait) {
            // In the driver until it has run, queued yet or not: one wake a
            // submission. (Polling until bb-submit had queued it was half of
            // this thread's wakes, ~25,000 a minute at 60 fps.)
            g_waits.fetch_add(1, std::memory_order_relaxed);
            if (g_hooks.submitted && g_hooks.submitted() <= front) g_unsubmitted.fetch_add(1, std::memory_order_relaxed);
            g_hooks.wait(front + 1, 50'000'000);  // returns when it has run; 50 ms at most, then look again
        } else {
            g_polls.fetch_add(1, std::memory_order_relaxed);
            host_sleep_us(g_poll_us);
        }
    }
}

}  // namespace

bool on() { return g_on; }

void start(const Hooks& hooks) {
    if (g_started.exchange(true)) return;
    g_hooks = hooks;
    std::thread(thread_main).detach();
}

namespace {
// The batch of submission `serial`: the newest one when it is that
// submission's (add_copy made it while the submission was recorded), else a
// new one after it.
Batch& batch_locked(std::uint64_t serial) {
    if (g_batches.empty() || g_batches.back().serial != serial) g_batches.push_back({serial, 0, now_ns()});
    return g_batches.back();
}
void added_locked(std::uint64_t serial) {
    g_most_batches = std::max<std::uint64_t>(g_most_batches, g_batches.size());
    g_most_writes = std::max<std::uint64_t>(g_most_writes, g_writes.size());
    // A submission already counted as run (no GPU any more: every count is ~0).
    if (serial < g_landed.load(std::memory_order_relaxed)) {
        land_locked(g_landed.load(std::memory_order_relaxed));
        return;
    }
    if (g_idle) g_work_cv.notify_one();
}
}  // namespace

void add(std::uint64_t serial, const Write* writes, std::size_t n) {
    if (!n) return;
    std::lock_guard<std::mutex> lk(g_mu);
    Batch& b = batch_locked(serial);  // (added_locked wakes the thread only when it waits for work)
    for (std::size_t k = 0; k < n; ++k) {
        Entry e{writes[k].va, writes[k].bytes, false, {}, nullptr};
        std::memcpy(e.data, writes[k].data, sizeof(e.data));
        g_writes.push_back(e);
    }
    b.count += static_cast<std::uint32_t>(n);
    added_locked(serial);
}

void add_copy(std::uint64_t serial, std::uint64_t va, const void* src, std::uint32_t bytes) {
    if (!bytes) return;
    std::lock_guard<std::mutex> lk(g_mu);
    Batch& b = batch_locked(serial);
    g_writes.push_back({va, bytes, true, {}, src});
    b.count += 1;
    added_locked(serial);
}

std::uint64_t land_through(std::uint64_t completed) {
    const std::uint64_t now = g_landed.load(std::memory_order_acquire);
    if (completed <= now) return now;
    std::lock_guard<std::mutex> lk(g_mu);
    land_locked(completed);
    return g_landed.load(std::memory_order_relaxed);
}

std::uint64_t landed() { return g_landed.load(std::memory_order_acquire); }

bool wait_landed_past(std::uint64_t seen, std::chrono::microseconds timeout) {
    std::unique_lock<std::mutex> lk(g_mu);
    ++g_landed_waiters;
    const bool past = g_landed_cv.wait_for(lk, timeout, [seen] { return g_landed.load(std::memory_order_relaxed) > seen; });
    --g_landed_waiters;
    return past;
}

std::string report() {
    if (!g_on) return "labels written by the GPU as fills at the end of their command buffer (BBHOST_LABEL_LANDING=1: by the host once it has run)";
    std::lock_guard<std::mutex> lk(g_mu);
    std::uint64_t total = 0;
    for (const std::uint64_t n : g_ages) total += n;
    const auto pct = [&](double q) {
        if (!total) return 0.0;
        const std::uint64_t want = static_cast<std::uint64_t>(q * static_cast<double>(total - 1)) + 1;
        std::uint64_t seen = 0;
        for (int k = 0; k < kBuckets; ++k) {
            seen += g_ages[k];
            if (seen >= want) return (k + 0.5) * kBucketNs / 1e6;
        }
        return kBuckets * kBucketNs / 1e6;
    };
    char b[720];
    std::snprintf(b, sizeof(b),
                  "labels written by the host once their submission had run (BBHOST_LABEL_LANDING=1, %s): %llu writes from %llu submissions "
                  "(%llu of them copies ahead of the labels, %llu KiB), "
                  "%llu landed by the landing thread and %llu by a wait for the GPU (a slot's retirement, a flush, a completion asked); "
                  "%llu polls, %llu waits in the driver (%llu begun before the submission was in the queue); %llu writes to unmapped memory "
                  "skipped; most waiting %llu submissions / %llu writes; submit to landed p50 %.2f ms, p99 %.2f ms; %llu still waiting",
                  g_block ? "the thread waits in the driver" : "the thread polls", static_cast<unsigned long long>(g_n_writes),
                  static_cast<unsigned long long>(g_n_batches), static_cast<unsigned long long>(g_n_copies),
                  static_cast<unsigned long long>(g_copy_bytes >> 10), static_cast<unsigned long long>(g_by_thread),
                  static_cast<unsigned long long>(g_by_others), static_cast<unsigned long long>(g_polls.load()),
                  static_cast<unsigned long long>(g_waits.load()), static_cast<unsigned long long>(g_unsubmitted.load()),
                  static_cast<unsigned long long>(g_unmapped), static_cast<unsigned long long>(g_most_batches),
                  static_cast<unsigned long long>(g_most_writes), pct(0.5), pct(0.99), static_cast<unsigned long long>(g_batches.size()));
    return b;
}

}  // namespace label_landing
