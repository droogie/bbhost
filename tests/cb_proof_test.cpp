// The constant-buffer proof's copies (host/cb_proof.cpp), without the game:
// buffers of our own copied as a GX call copies them, compared as the command
// processor compares a binding - whole, in part, past either end, after a
// write, after the ring came round over the copy, with copies of every size
// at the ring's end, and with another thread copying into its ring while
// this one compares (never a change where nothing was written, however the
// two meet). Runs with BBHOST_CB_RECORD_PROOF=1 (CMakeLists.txt), which only
// the report reads.
#include "host/cb_proof.cpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace cb_proof;

namespace {

int g_failed = 0;
void check_that(bool ok, const char* what, int line) {
    if (ok) return;
    if (++g_failed <= 40) std::printf("FAIL (line %d): %s\n", line, what);
}
#define CHECK(x) check_that((x), #x, __LINE__)

u64 va(const void* p) { return static_cast<u64>(reinterpret_cast<std::uintptr_t>(p)); }

}  // namespace

int main() {
    // A small ring, so that copies come round over each other quickly.
    set_ring_bytes(1u << 20);
    const u64 ring = g_ring_bytes;

    std::vector<u8> buf(8192);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<u8>(i * 7 + 3);
    const u64 a = va(buf.data()) + 64;

    // Whole, in part, past either end.
    const u64 h = record(a, 256, kStatic, 7, 5, 0x1234);
    CHECK(h != 0);
    Difference d;
    d = Difference{};
    CHECK(check(h, a, 256, kPathSnapshot, &d) == kSame);
    CHECK(d.completed == 5 && d.object == 0x1234 && d.flip == 7 && d.addr == a);  // a same one's header too
    CHECK(check(h, a + 16, 64, kPathSnapshot, &d) == kSame);
    CHECK(check(h, a + 240, 16, kPathSnapshot, &d) == kSame);
    CHECK(check(h, a, 272, kPathSnapshot, &d) == kNotCovered);
    CHECK(check(h, a + 250, 16, kPathSnapshot, &d) == kNotCovered);
    CHECK(check(h, a - 16, 16, kPathSnapshot, &d) == kNotCovered);

    // A write after the copy: the dword it is in, and the parts it is not in.
    buf[64 + 40] ^= 1;  // dword 10 of the copy
    d = Difference{};
    CHECK(check(h, a, 256, kPathSnapshot, &d) == kChanged);
    CHECK(d.first_dword == 10 && d.dwords == 1 && d.flip == 7 && d.kind == kStatic && d.addr == a && d.copied == 256);
    CHECK(d.completed == 5 && d.object == 0x1234);
    CHECK(d.now[0] == (d.was[0] ^ 1u));
    d = Difference{};
    CHECK(check(h, a + 32, 16, kPathRealigned, &d) == kChanged && d.first_dword == 2);
    CHECK(check(h, a + 48, 16, kPathSnapshot, &d) == kSame);
    CHECK(check(h, a, 40, kPathSnapshot, &d) == kSame);
    buf[64 + 40] ^= 1;
    CHECK(check(h, a, 256, kPathSnapshot, &d) == kSame);
    // A change in a tail shorter than a dword.
    buf[64 + 254] ^= 0x80;
    d = Difference{};
    CHECK(check(h, a + 252, 3, kPathSnapshot, &d) == kChanged);
    buf[64 + 254] ^= 0x80;

    // Longer than a copy takes: cut at kMaxCopy.
    std::vector<u8> big(kMaxCopy + 4096, 0x5a);
    const u64 hb = record(va(big.data()), kMaxCopy + 4096, kRenamed, 1, 0, 0);
    CHECK(check(hb, va(big.data()), kMaxCopy, kPathSnapshot, &d) == kSame);
    CHECK(check(hb, va(big.data()), kMaxCopy + 16, kPathSnapshot, &d) == kNotCovered);

    // Copies of every size up to the ring's end: none runs past it, each
    // compares the same right after it is taken, and its header is its own.
    for (int i = 0; i < 4000; ++i) {
        const u32 bytes = 16u * (1u + static_cast<u32>(i) % 300u) + (i % 3 ? 0u : 4u);
        const u64 hh = record(va(buf.data()), std::min<u32>(bytes, 8192), kDynamicOwn, static_cast<u64>(i), 0, 0);
        CHECK(hh != 0);
        const u64 pos = hh & ((u64{1} << 56) - 1);
        CHECK(pos % ring + sizeof(Header) + std::min<u32>(bytes, 8192) <= ring);
        CHECK(check(hh, va(buf.data()), std::min<u32>(bytes, 8192), kPathSnapshot, &d) == kSame);
    }
    // ...and the first copies are long gone.
    CHECK(check(h, a, 256, kPathSnapshot, &d) == kLapped);
    CHECK(check(hb, va(big.data()), 64, kPathSnapshot, &d) == kLapped);

    // Handles that name no ring.
    CHECK(check(0, a, 16, kPathSnapshot, &d) == kNoCopy);
    CHECK(check(u64{63} << 56, a, 16, kPathSnapshot, &d) == kNoCopy);
    CHECK(check(u64{200} << 56, a, 16, kPathSnapshot, &d) == kNoCopy);

    // Another thread copying into its own ring (laps every few dozen copies)
    // while this one compares its newest copies, read from a buffer nothing
    // writes: same or lapped, never changed, whatever the timing. The phase
    // is counted in progress, not in rounds: it waits for the writer's first
    // copy, then compares until the writer has made kCopies more and this
    // thread has compared kCompares times, so a loaded box makes it slower
    // but never empty (a fixed 400,000 rounds were once over before the
    // writer had run at all). Only a writer that makes no copy for kStall
    // ends it early, as a failure.
    {
        std::vector<u8> src(kMaxCopy);
        for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<u8>(i ^ (i >> 8));
        constexpr int kSlots = 64;
        constexpr u64 kCopies = 8192, kCompares = 400000;
        constexpr auto kStall = std::chrono::seconds(30);
        std::atomic<u64> handles[kSlots];
        std::atomic<u32> sizes[kSlots];
        for (int i = 0; i < kSlots; ++i) {
            handles[i] = 0;
            sizes[i] = 0;
        }
        std::atomic<bool> stop{false};
        std::atomic<u64> made{0};
        std::thread writer([&] {
            u64 n = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const u32 bytes = 16u * (1u + static_cast<u32>((n * 2654435761u) % 4096u));
                const u64 hh = record(va(src.data()), bytes, kStatic, n, n, 0);
                sizes[n % kSlots].store(bytes, std::memory_order_relaxed);
                handles[n % kSlots].store(hh, std::memory_order_release);
                ++n;
                made.store(n, std::memory_order_relaxed);
            }
        });
        // Whether the writer still moves: a copy within the last kStall.
        u64 seen = 0;
        auto moved = std::chrono::steady_clock::now();
        const auto moving = [&] {
            const u64 m = made.load(std::memory_order_relaxed);
            if (m != seen) {
                seen = m;
                moved = std::chrono::steady_clock::now();
                return true;
            }
            return std::chrono::steady_clock::now() - moved < kStall;
        };
        // The writer's first copy: on a loaded box a new thread can wait long for a CPU.
        bool stalled = false;
        while (!stalled && made.load(std::memory_order_relaxed) == 0) {
            stalled = !moving();
            std::this_thread::yield();
        }
        const u64 first = made.load(std::memory_order_relaxed);
        u64 compares = 0, same = 0, lapped = 0, other = 0;
        for (u64 round = 0; !stalled; ++round) {
            if (round % 1024 == 0) {
                if (made.load(std::memory_order_relaxed) - first >= kCopies && compares >= kCompares) break;
                stalled = !moving();
            }
            const int slot = static_cast<int>(round % kSlots);
            const u64 hh = handles[slot].load(std::memory_order_acquire);
            const u32 bytes = sizes[slot].load(std::memory_order_relaxed);
            if (!hh || !bytes) continue;
            const Outcome o = check(hh, va(src.data()), std::min<u32>(bytes, 4096), kPathSnapshot, &d);
            ++compares;
            same += o == kSame;
            lapped += o == kLapped;
            other += o != kSame && o != kLapped && o != kNotCovered;
        }
        stop = true;
        writer.join();
        const u64 copies = made.load() - first;
        std::printf("concurrent: %llu copies made while %llu compares ran: %llu the same, %llu lapped, %llu otherwise%s\n",
                    static_cast<unsigned long long>(copies), static_cast<unsigned long long>(compares),
                    static_cast<unsigned long long>(same), static_cast<unsigned long long>(lapped),
                    static_cast<unsigned long long>(other), stalled ? "; the writer stalled" : "");
        CHECK(!stalled);
        CHECK(copies >= kCopies && compares >= kCompares);
        CHECK(other == 0);
        CHECK(same > 0);
        CHECK(lapped > 0);
    }

    // The counts went where they were sent.
    CHECK(g_outcomes[kPathRealigned][kChanged].load() == 1);
    CHECK(g_outcomes[kPathSnapshot][kNotCovered].load() >= 4);
    Difference made_up;
    made_up.kind = kStatic;
    made_up.flip = 2;
    note_result(made_up, kChanged, kPathSnapshot, "test-pipeline", 9, false, false, true, a, 16);
    CHECK(g_changed[kPathSnapshot][kStatic][3].load() == 1 && g_changed_gpu_any[kPathSnapshot].load() == 1);
    CHECK(g_rule[kPathSnapshot][1][0].load() == 1);  // changed, no GPU write since the call: the rule misses it
    note_result(made_up, kChanged, kPathSnapshot, "test-pipeline", 2, true, false, true, a, 16);
    CHECK(g_rule[kPathSnapshot][1][1].load() == 1 && g_changed[kPathSnapshot][kStatic][0].load() == 1);
    note_result(made_up, kChanged, kPathSnapshot, "test-pipeline", 2, false, true, false, a, 16);  // left for the tables alone
    CHECK(g_rule[kPathSnapshot][1][1].load() == 2 && g_rule_dropped[kPathSnapshot][1].load() == 1);
    CHECK(g_rule[kPathSnapshot][1][0].load() == 1);
    note_result(made_up, kSame, kPathRealigned, "test-pipeline", 2, true, false, true, a, 16);
    CHECK(g_rule[kPathRealigned][0][1].load() == 1 && g_rule_dropped[kPathRealigned][0].load() == 0);
    note_result(made_up, kLapped, kPathRealigned, "test-pipeline", 2, true, false, true, a, 16);  // not counted
    CHECK(g_rule[kPathRealigned][0][1].load() == 1);
    note_uncopied(kPathGpuPending, false, true);
    CHECK(g_uncopied[kPathGpuPending][2].load() == 1);
    report();

    if (g_failed) {
        std::printf("%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("cb_proof_test: all checks passed\n");
    return 0;
}
