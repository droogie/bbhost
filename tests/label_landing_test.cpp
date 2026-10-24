// Labels written by the host once their submission has run
// (host/label_landing.cpp), without the game or a GPU: a counter stands for
// the timeline. Nothing lands before its submission has run; a submission's
// writes land in the order they were made and before any later submission's;
// a wait for the GPU (land_through) lands what the thread has not yet, and the
// thread then lands nothing twice; unmapped memory is skipped; a copy
// added while a submission is recorded lands ahead of its labels; with no GPU any
// more everything lands, and so does what is added after; and with a producer,
// a GPU and two landers racing, every label only ever moves forward and holds,
// once its submission counts as landed, at least that submission's value, and
// never a value whose submission has not run. landing() is past a submission
// before any of its writes is stored, and landed() only after all of them.
// The last landings (recent_into) are each store's address, size, first
// dword and submission, newest first, inside the range asked for; a skipped
// write is not among them; and no more than kRecent are kept.
// Runs with BBHOST_LABEL_LANDING=1 (CMakeLists.txt).
#include "host/label_landing.cpp"

#include <cstdio>
#include <random>
#include <thread>
#include <vector>

namespace {

int g_failed = 0;
void check_that(bool ok, const char* what, int line) {
    if (ok) return;
    if (++g_failed <= 40) std::printf("FAIL (line %d): %s\n", line, what);
}
#define CHECK(x) check_that((x), #x, __LINE__)

std::atomic<std::uint64_t> g_gpu{0};  // submissions run
bool g_dead = false;                  // no GPU any more
std::uint64_t completed() { return g_dead ? ~0ull : g_gpu.load(std::memory_order_acquire); }
std::uint64_t submitted() { return ~0ull >> 1; }
bool wait(std::uint64_t need, std::uint64_t timeout_ns) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
    while (completed() < need) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::yield();
    }
    return true;
}
// Unmapped: [g_hole_lo, g_hi); the order stores were made in, by address.
std::uint64_t g_hole_lo = 0, g_hole_hi = 0;
std::mutex g_order_mu;
std::vector<std::uint64_t> g_order;
bool g_keep_order = false;
// The counts as each write of the watched address is stored.
std::uint64_t g_watch = 0, g_watch_landing = 0, g_watch_landed = 0;
bool mapped(std::uint64_t va, std::size_t bytes) {
    if (va == g_watch) {
        g_watch_landing = label_landing::landing();
        g_watch_landed = label_landing::landed();
    }
    if (va < g_hole_hi && va + bytes > g_hole_lo) return false;
    if (g_keep_order) {
        std::lock_guard<std::mutex> lk(g_order_mu);
        g_order.push_back(va);
    }
    return true;
}

std::uint64_t va(const void* p) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p)); }

label_landing::Write w32(const void* p, std::uint32_t v) {
    label_landing::Write w{va(p), 4, {}};
    std::memcpy(w.data, &v, 4);
    return w;
}
label_landing::Write w64(const void* p, std::uint64_t v) {
    label_landing::Write w{va(p), 8, {}};
    std::memcpy(w.data, &v, 8);
    return w;
}

// Until the thread has landed through `n` (it should, the GPU having run them).
bool landed_through(std::uint64_t n) {
    for (int k = 0; k < 2000 && label_landing::landed() < n; ++k) label_landing::wait_landed_past(label_landing::landed(), std::chrono::milliseconds(1));
    return label_landing::landed() >= n;
}

}  // namespace

int main() {
    using namespace label_landing;
    CHECK(on());
    Hooks h;
    h.completed = completed;
    h.submitted = submitted;
    h.wait = wait;
    h.mapped = mapped;
    start(h);

    alignas(8) static std::uint32_t labels[64] = {};
    alignas(8) static std::uint64_t markers[8] = {};

    // Nothing lands before its submission has run. landing() is past it
    // before its writes are stored, landed() only after: the game may read a
    // label the moment it is in memory.
    {
        const Write a[] = {w32(&labels[0], 1), w32(&labels[1], 2)};
        add(0, a, 2);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(labels[0] == 0 && labels[1] == 0);
        CHECK(landed() == 0 && landing() == 0);
        g_watch = va(&labels[1]);
        g_gpu.store(1);
        CHECK(landed_through(1));
        CHECK(labels[0] == 1 && labels[1] == 2);
        CHECK(g_watch_landing == 1);  // counted before its labels were stored
        CHECK(g_watch_landed == 0);   // landed() only after
        CHECK(landing() == 1);
        g_watch = 0;
    }

    // In order: within a submission (the last write to an address wins; a
    // chunk's retirement after the label written into the chunk before it),
    // and across submissions.
    {
        g_keep_order = true;
        const Write a[] = {w32(&labels[2], 7), w64(&markers[0], 2), w32(&labels[2], 8), w64(&markers[0], 0)};
        add(1, a, 4);
        const Write b[] = {w32(&labels[2], 9)};
        add(2, b, 1);
        const Write c[] = {w32(&labels[3], 10)};
        add(3, c, 1);
        g_gpu.store(3);
        CHECK(landed_through(3));
        CHECK(labels[2] == 9 && markers[0] == 0);
        CHECK(labels[3] == 0);  // submission 3 has not run
        {
            std::lock_guard<std::mutex> lk(g_order_mu);
            const std::vector<std::uint64_t> want = {va(&labels[2]), va(&markers[0]), va(&labels[2]), va(&markers[0]), va(&labels[2])};
            CHECK(g_order == want);
            g_order.clear();
        }
        g_keep_order = false;
        g_gpu.store(4);
        CHECK(landed_through(4));
        CHECK(labels[3] == 10);
    }

    // A wait for the GPU lands what the thread has not reached; nothing twice.
    {
        const Write a[] = {w32(&labels[4], 11)};
        add(4, a, 1);
        const Write b[] = {w32(&labels[4], 12)};
        add(5, b, 1);
        // The GPU has run both, the thread may not have looked yet: the waiter
        // lands them itself, in order.
        g_gpu.store(6);
        CHECK(land_through(6) >= 6);
        CHECK(labels[4] == 12);
        labels[4] = 99;  // the game rewrites it once it has seen 12
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(labels[4] == 99);  // the thread did not land 11 or 12 again
        CHECK(land_through(3) >= 6);  // an older count changes nothing
    }

    // An unaligned eight-byte label (four-aligned) and unmapped memory.
    {
        alignas(8) static std::uint32_t pair[4] = {};
        const Write a[] = {w64(&pair[1], 0x1122334455667788ull), w32(&labels[5], 13), w32(&labels[6], 14)};
        g_hole_lo = va(&labels[5]);
        g_hole_hi = g_hole_lo + 4;
        add(6, a, 3);
        g_gpu.store(7);
        CHECK(landed_through(7));
        std::uint64_t got = 0;
        std::memcpy(&got, &pair[1], 8);
        CHECK(got == 0x1122334455667788ull);
        CHECK(labels[5] == 0);  // skipped
        CHECK(labels[6] == 14);
        // The last landings: the newest first, inside the range asked for,
        // the skipped one not among them.
        Landed got_l[4] = {};
        CHECK(recent_into(va(&labels[5]), va(&labels[7]), got_l, 4) == 1);
        CHECK(got_l[0].va == va(&labels[6]) && got_l[0].bytes == 4 && got_l[0].value == 14 && got_l[0].serial == 6);
        CHECK(recent_into(va(&pair[0]), va(&pair[4]), got_l, 4) == 1);
        CHECK(got_l[0].va == va(&pair[1]) && got_l[0].bytes == 8 && got_l[0].value == 0x55667788u);
        CHECK(recent_into(va(&pair[2]), va(&pair[3]), got_l, 4) == 1);  // overlapping its second half
        CHECK(recent_into(va(&labels[0]), va(&labels[7]), got_l, 4) == 4);
        CHECK(got_l[0].va == va(&labels[6]) && got_l[1].va == va(&labels[4]) && got_l[1].value == 12 && got_l[2].value == 11 &&
              got_l[2].serial == 4 && got_l[3].va == va(&labels[3]) && got_l[3].value == 10);
        CHECK(recent_into(va(&labels[60]), va(&labels[64]), got_l, 4) == 0);
        CHECK(recent_into(va(&labels[4]), va(&labels[6]), got_l, 4) == 2);  // not the one at the range's end
        CHECK(recent_into(va(&labels[7]), va(&labels[8]), got_l, 4) == 0);  // nor the one ending at its start
        g_hole_lo = g_hole_hi = 0;
    }

    // Racing: one producer adds a submission a step (each write's value is its
    // submission + 1, to a few labels), a GPU runs them at random intervals,
    // the thread and a second lander (a slot's retirement) land them. Each
    // label only moves forward, and once a submission counts as landed, every
    // label it wrote holds at least its value.
    // A copy ahead of the labels (add_copy, while the submission is
    // recorded): it lands before them, and on its own in a submission with no
    // labels; not before its submission has run.
    {
        alignas(8) static std::uint8_t dump[64], guest[64], guest2[16];
        for (int k = 0; k < 64; ++k) dump[k] = static_cast<std::uint8_t>(k * 3 + 1);
        g_keep_order = true;
        add_copy(7, va(guest), dump, 64);
        const Write a[] = {w32(&labels[9], 17)};
        add(7, a, 1);
        add_copy(8, va(guest2), dump, 16);  // no labels in submission 8
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(guest[0] == 0 && labels[9] == 0);
        g_gpu.store(8);
        CHECK(landed_through(8));
        CHECK(std::memcmp(guest, dump, 64) == 0);
        CHECK(labels[9] == 17);
        Landed got_c[2] = {};
        CHECK(recent_into(va(guest), va(guest) + 64, got_c, 2) == 1);  // a copy: its first dword
        CHECK(got_c[0].bytes == 64 && got_c[0].value == 0x0a070401u && got_c[0].serial == 7);
        CHECK(guest2[0] == 0);  // submission 8 has not run
        {
            std::lock_guard<std::mutex> lk(g_order_mu);
            const std::vector<std::uint64_t> want = {va(guest), va(&labels[9])};
            CHECK(g_order == want);
            g_order.clear();
        }
        g_keep_order = false;
        g_gpu.store(9);
        CHECK(landed_through(9));
        CHECK(std::memcmp(guest2, dump, 16) == 0);
    }

    constexpr std::uint64_t kFirst = 9, kLast = 20000;
    {
        std::atomic<std::uint64_t> made{kFirst};
        std::atomic<bool> stop{false};
        std::atomic<int> backwards{0}, short_of{0}, early{0};
        std::uint32_t* const race = &labels[16];  // 16 labels
        std::thread producer([&] {
            std::mt19937 rng(1);
            for (std::uint64_t s = kFirst; s < kLast; ++s) {
                Write ws[4];
                const int n = 1 + static_cast<int>(rng() % 4);
                for (int k = 0; k < n; ++k) ws[k] = w32(&race[(s + k * 5) % 16], static_cast<std::uint32_t>(s + 1));
                add(s, ws, static_cast<std::size_t>(n));
                made.store(s + 1, std::memory_order_release);
                if ((s & 63) == 0) std::this_thread::yield();
            }
        });
        std::thread gpu([&] {
            std::mt19937 rng(2);
            while (g_gpu.load() < kLast) {
                const std::uint64_t m = made.load(std::memory_order_acquire);
                const std::uint64_t now = g_gpu.load();
                if (now < m) g_gpu.store(now + 1 + rng() % std::min<std::uint64_t>(m - now, 8));
                if (rng() % 4 == 0) std::this_thread::yield();
            }
        });
        std::thread retire([&] {
            std::mt19937 rng(3);
            while (!stop.load()) {
                land_through(g_gpu.load());
                if (rng() % 2) std::this_thread::yield();
            }
        });
        std::thread watcher([&] {
            std::uint32_t last[16] = {};
            while (!stop.load()) {
                const std::uint64_t l = landed();
                for (int k = 0; k < 16; ++k) {
                    const std::uint32_t v = __atomic_load_n(&race[k], __ATOMIC_ACQUIRE);
                    if (v < last[k]) backwards.fetch_add(1);
                    if (v > g_gpu.load(std::memory_order_acquire)) early.fetch_add(1);  // landed before its submission ran
                    last[k] = v;
                }
                // Every submission before l landed: the label it wrote last
                // (at the latest submission before l that wrote it) holds at
                // least that value. Checked loosely: each label is written by
                // some submission among the last 16 before l.
                if (l > kFirst + 32 && l < kLast) {
                    for (int k = 0; k < 16; ++k) {
                        const std::uint32_t v = __atomic_load_n(&race[k], __ATOMIC_ACQUIRE);
                        if (v + 32 < l) short_of.fetch_add(1);
                    }
                }
            }
        });
        producer.join();
        gpu.join();
        CHECK(landed_through(kLast));
        stop.store(true);
        retire.join();
        watcher.join();
        CHECK(backwards.load() == 0);
        CHECK(short_of.load() == 0);
        CHECK(early.load() == 0);
        // The last writes: each label holds the value of the last submission that wrote it.
        std::uint32_t want[16] = {};
        {
            std::mt19937 rng(1);
            for (std::uint64_t s = kFirst; s < kLast; ++s) {
                const int n = 1 + static_cast<int>(rng() % 4);
                for (int k = 0; k < n; ++k) want[(s + k * 5) % 16] = static_cast<std::uint32_t>(s + 1);
            }
        }
        for (int k = 0; k < 16; ++k) CHECK(race[k] == want[k]);
        // Only the last kRecent landings are kept: all of them the race's
        // now, the newest its last submission's.
        static Landed kept[kRecent + 8];
        CHECK(recent_into(va(&race[0]), va(&race[16]), kept, kRecent + 8) == static_cast<int>(kRecent));
        CHECK(kept[0].serial == kLast - 1 && kept[kRecent - 1].serial <= kept[0].serial);
        CHECK(recent_into(va(&labels[0]), va(&labels[8]), kept, 8) == 0);  // older than the last kRecent
    }

    // No GPU any more: everything lands, and what is added after it too.
    {
        const Write a[] = {w32(&labels[7], 15)};
        add(kLast, a, 1);
        g_dead = true;
        CHECK(landed_through(~0ull));
        CHECK(labels[7] == 15);
        const Write b[] = {w32(&labels[8], 16)};
        add(kLast + 1, b, 1);
        CHECK(labels[8] == 16);
    }

    const std::string r = report();
    std::printf("%s\n", r.c_str());
    if (g_failed) {
        std::printf("label_landing_test: %d checks failed\n", g_failed);
        return 1;
    }
    std::printf("label_landing_test: ok\n");
    return 0;
}
