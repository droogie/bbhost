// The copy-back captures at the GX resolve (host/cb_capture.cpp), without the
// game: resolves' batches of copy-backs over sources of our own, captured as
// native_copy captures them, and the pieces' laps as the command processor
// binds and releases - nothing captured before the ring is made, the
// captures' bytes and places, copies too long and batches past a piece left
// out, unmapped sources skipped, a piece not written again while a token holds
// it or while the submission that read it runs, a piece held by a token never
// walked taken back (its late release then does nothing, its handle no longer
// live), a segment per GX thread. Runs with BBHOST_GXVK_HOMES=1.
#include "host/cb_capture.cpp"

#include <cstdio>
#include <thread>
#include <vector>

// What cb_capture.cpp asks the rest of bbhost.
namespace {
std::uint64_t g_completed = 0;
std::uint64_t g_unmapped_lo = 0, g_unmapped_hi = 0;
}  // namespace
std::uint64_t host_gpu_completed_submits_seen() { return g_completed; }
bool hle_kernel_va_mapped_in(std::uint64_t va, std::size_t len, std::uint64_t* lo, std::uint64_t* hi) {
    if (va < g_unmapped_hi && va + len > g_unmapped_lo) return false;
    *lo = va;
    *hi = va + len;
    return true;
}
namespace gxvk {
std::uint64_t arena_handout_seq() { return 0; }
std::uint32_t arena_thread_tag() { return 1; }
int capture_class(std::uint64_t, std::uint64_t, std::uint64_t, std::uint32_t, bool* landed) {
    *landed = false;
    return kCapNoRecord;
}
}  // namespace gxvk

using namespace cb_capture;

namespace {

int g_failed = 0;
void check_that(bool ok, const char* what, int line) {
    if (ok) return;
    if (++g_failed <= 40) std::printf("FAIL (line %d): %s\n", line, what);
}
#define CHECK(x) check_that((x), #x, __LINE__)

std::uint64_t va(const void* p) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p)); }

// One resolve's batch: n sources of `bytes` each, filled with `seed`.
struct Batch {
    std::vector<std::vector<std::uint8_t>> srcs;
    std::vector<std::uint64_t> dst, src, bytes;
    std::vector<std::uint32_t> at;
    Batch(std::uint32_t n, std::uint64_t each, std::uint8_t seed) {
        srcs.resize(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            srcs[i].resize(each);
            for (std::uint64_t k = 0; k < each; ++k) srcs[i][k] = static_cast<std::uint8_t>(seed + i * 7 + k);
            dst.push_back(0x100000 + i * 0x1000);
            src.push_back(va(srcs[i].data()));
            bytes.push_back(each);
        }
        at.assign(n, 0);
    }
    std::uint32_t capture_now(std::uint64_t flip) {
        return capture(0x5000, dst.data(), src.data(), bytes.data(), static_cast<std::uint32_t>(dst.size()), at.data(), flip);
    }
    bool same(std::uint32_t i) const { return at[i] != kNone && std::memcmp(bytes_at(at[i]), srcs[i].data(), bytes[i]) == 0; }
};

}  // namespace

int main() {
    static std::vector<std::uint8_t> ring(kRingBytes);

    // Nothing before the ring is made.
    {
        Batch b(2, 64, 1);
        CHECK(b.capture_now(1) == kNone);
        CHECK(b.at[0] == kNone && b.at[1] == kNone);
    }
    set_ring(ring.data());
    CHECK(ring_made());

    // The bytes and their places: whole, 256-byte aligned, in one piece, apart.
    {
        Batch b(3, 100, 2);
        const std::uint32_t piece = b.capture_now(1);
        CHECK(piece != kNone);
        CHECK(piece_live(piece));
        for (std::uint32_t i = 0; i < 3; ++i) CHECK(b.same(i));
        CHECK(b.at[0] != b.at[1] && b.at[1] != b.at[2]);
        CHECK(offset(b.at[0]) / kPieceBytes == offset(b.at[2]) / kPieceBytes);
        release(piece);
    }

    // Too long, and unmapped: not captured; the rest are.
    {
        Batch b(3, 64, 3);
        std::vector<std::uint8_t> big(kMaxCopy + 4);
        b.src[1] = va(big.data());
        b.bytes[1] = big.size();
        g_unmapped_lo = b.src[2];
        g_unmapped_hi = b.src[2] + 1;
        const std::uint32_t piece = b.capture_now(2);
        CHECK(piece != kNone);
        CHECK(b.same(0));
        CHECK(b.at[1] == kNone);
        CHECK(b.at[2] == kNone);
        g_unmapped_lo = g_unmapped_hi = 0;
        release(piece);
    }

    // A batch past a piece is left out whole.
    {
        Batch b(1, 1, 4);
        std::vector<std::uint8_t> v(kMaxCopy);
        b.srcs.clear();
        b.dst.clear(), b.src.clear(), b.bytes.clear();
        for (std::uint64_t k = 0; k <= kPieceBytes / kMaxCopy; ++k) {
            b.dst.push_back(0x200000 + k * kMaxCopy);
            b.src.push_back(va(v.data()));
            b.bytes.push_back(kMaxCopy);
        }
        b.at.assign(b.dst.size(), 0);
        CHECK(b.capture_now(3) == kNone);
        for (std::uint32_t a : b.at) CHECK(a == kNone);
    }

    // A piece is written again only once its tokens are walked and the newest
    // submission that read one of its captures has completed.
    {
        std::vector<std::uint32_t> pieces;
        std::vector<Batch> held;
        const std::uint64_t each = kPieceBytes / 2;  // two batches a piece
        for (std::uint32_t k = 0; k < 2 * kPieces + 2; ++k) {
            held.emplace_back(1, std::min<std::uint64_t>(each, kMaxCopy), static_cast<std::uint8_t>(10 + k));
        }
        // Fill every piece of the segment, holding the tokens.
        std::uint32_t first = kNone, last_ok = 0, busy = 0;
        for (std::uint32_t k = 0; k < held.size(); ++k) {
            const std::uint32_t p = held[k].capture_now(10);
            if (p == kNone) {
                ++busy;
                continue;
            }
            if (first == kNone) first = p;
            last_ok = k;
            pieces.push_back(p);
        }
        CHECK(first != kNone);
        CHECK(last_ok > 0);
        // Every piece of the segment is held now: one more is refused.
        Batch more(1, kMaxCopy, 50);
        for (int k = 0; k < static_cast<int>(kPieces) * 8 && more.capture_now(10) != kNone; ++k) {
        }
        CHECK(more.capture_now(10) == kNone);
        // Bound by submission 7, released: still not written while 7 runs.
        for (std::uint32_t p : pieces) bound(p, 7);
        for (std::uint32_t p : pieces) release(p);
        g_completed = 7;
        CHECK(more.capture_now(11) == kNone);
        g_completed = 8;
        const std::uint32_t p2 = more.capture_now(11);
        CHECK(p2 != kNone);
        CHECK(more.same(0));
        CHECK(!piece_live(first) || (first & 0xffff) != (p2 & 0xffff));  // its piece's lap moved on
        release(p2);
        // The captures held before stay as they were until their piece is written again.
        CHECK(held[last_ok].same(0));
    }

    // A piece held by a token never walked is taken back after 600 flips; its
    // handle is no longer live and its late release does nothing.
    {
        std::uint32_t stuck = kNone;
        std::vector<Batch> bs;
        for (std::uint32_t k = 0; k < kPieces * 4; ++k) bs.emplace_back(1, kMaxCopy, static_cast<std::uint8_t>(k));
        for (std::uint32_t k = 0; k < bs.size(); ++k) {
            const std::uint32_t p = bs[k].capture_now(1000);
            if (p == kNone) break;
            if (stuck == kNone) stuck = p;
            else release(p);
        }
        CHECK(stuck != kNone);
        CHECK(piece_live(stuck));
        std::uint32_t retaken = kNone;
        for (std::uint32_t k = 0; k < kPieces * 8 && retaken == kNone; ++k) {
            Batch b(1, kMaxCopy, 99);
            const std::uint32_t p = b.capture_now(1000 + 601);
            if (p != kNone && (p & 0xffff) == (stuck & 0xffff)) retaken = p;
            else if (p != kNone) release(p);
        }
        CHECK(retaken != kNone);
        CHECK(!piece_live(stuck));
        const std::uint64_t before = g_stale_releases.load();
        release(stuck);
        CHECK(g_stale_releases.load() == before + 1);
        CHECK(piece_live(retaken));
        release(retaken);
    }

    // A segment per GX thread: another thread's captures land in another segment.
    {
        std::uint32_t mine = kNone, theirs = kNone;
        Batch a(1, 64, 7);
        mine = a.capture_now(5000);
        std::thread t([&] {
            Batch b(1, 64, 8);
            theirs = b.capture_now(5000);
            CHECK(b.same(0));
            release(theirs);
        });
        t.join();
        CHECK(mine != kNone && theirs != kNone);
        CHECK((mine & 0xffff) / kPieces != (theirs & 0xffff) / kPieces);
        release(mine);
    }

    if (g_failed) {
        std::printf("cb_capture_test: %d failed\n", g_failed);
        return 1;
    }
    std::printf("cb_capture_test: ok\n");
    return 0;
}
