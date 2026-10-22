// The buffer shadow's in-place choices (host/shadow_place.h): which pages an
// integrated GPU reads where they lie instead of from a mirror's copy, when a
// range binds the import, and when a page rewritten again and again goes to
// host memory for good - as a mirror page copied too often did.
#include "host/shadow_place.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using shadow_place::PageFacts;
using shadow_place::Range;
using shadow_place::RewriteWindow;

constexpr std::uint32_t kLimit = 4;       // buffer_shadow.cpp's kMaxUploads
constexpr std::uint64_t kWindow = 8000;   // kUploadWindowMs

PageFacts told() {
    PageFacts f;
    f.told_cover = true;
    return f;
}

void test_eligible() {
    CHECK(shadow_place::eligible(told()));
    PageFacts watched;  // no GX buffer covers it whole: only a write watch would see its writes
    CHECK(!shadow_place::eligible(watched));
    PageFacts handed = told();  // renamed copies over it: rewritten by design
    handed.handed = true;
    CHECK(!shadow_place::eligible(handed));
    PageFacts dynamic = told();
    dynamic.dynamic = true;
    CHECK(!shadow_place::eligible(dynamic));
}

void test_choose() {
    std::vector<PageFacts> mesh(5, told());
    CHECK(shadow_place::choose(mesh.data(), 5) == Range::kInPlace);
    // One page under renamed copies: the whole range to the mirrors - never split.
    std::vector<PageFacts> mixed = mesh;
    mixed[3].handed = true;
    CHECK(shadow_place::choose(mixed.data(), 5) == Range::kMirror);
    // One the game keeps rewriting: host memory, whatever the others are.
    std::vector<PageFacts> ring = mixed;
    ring[1].dynamic = true;
    CHECK(shadow_place::choose(ring.data(), 5) == Range::kHost);
    std::vector<PageFacts> ring_first = mesh;
    ring_first[4].dynamic = true;
    CHECK(shadow_place::choose(ring_first.data(), 5) == Range::kHost);
    // A watched page before the rewritten one: host memory still (the
    // mirrors would read it there anyway, and copy nothing).
    std::vector<PageFacts> watched_then_ring = mesh;
    watched_then_ring[0].told_cover = false;
    watched_then_ring[2].dynamic = true;
    CHECK(shadow_place::choose(watched_then_ring.data(), 5) == Range::kHost);
    CHECK(shadow_place::choose(mesh.data(), 0) == Range::kMirror);  // no pages: nothing to bind in place
}

void test_rewrites() {
    // A placement counts as the first; four in the window are fine, a fifth is one too many.
    RewriteWindow w;
    std::uint64_t t = 100000;
    for (std::uint32_t i = 0; i < kLimit; ++i) CHECK(!shadow_place::rewrite_too_many(w, t + i * 1000, kLimit, kWindow));
    CHECK(shadow_place::rewrite_too_many(w, t + 4000, kLimit, kWindow));
    // Spread out, a mesh made over now and then never goes to host memory.
    RewriteWindow slow;
    for (std::uint32_t i = 0; i < 20; ++i) CHECK(!shadow_place::rewrite_too_many(slow, t + i * 3000, kLimit, kWindow));
    // The window starts again at the first rewrite after it ends, not at its old start plus the window.
    RewriteWindow again;
    CHECK(!shadow_place::rewrite_too_many(again, t, kLimit, kWindow));
    CHECK(!shadow_place::rewrite_too_many(again, t + kWindow + 500, kLimit, kWindow));  // a new window from here
    for (std::uint32_t i = 1; i < kLimit; ++i) CHECK(!shadow_place::rewrite_too_many(again, t + kWindow + 500 + i, kLimit, kWindow));
    CHECK(shadow_place::rewrite_too_many(again, t + kWindow + 600, kLimit, kWindow));
    // Exactly at the window's end it is a new window.
    RewriteWindow edge;
    for (std::uint32_t i = 0; i < kLimit; ++i) CHECK(!shadow_place::rewrite_too_many(edge, t, kLimit, kWindow));
    CHECK(!shadow_place::rewrite_too_many(edge, t + kWindow, kLimit, kWindow));
}

void test_rewritten() {
    CHECK(!shadow_place::rewritten(7, 7));  // the write it was placed after
    CHECK(!shadow_place::rewritten(7, 3));
    CHECK(shadow_place::rewritten(7, 8));   // told after its placement
    CHECK(!shadow_place::rewritten(1, 0));  // never written
}

}  // namespace

int main() {
    test_eligible();
    test_choose();
    test_rewrites();
    test_rewritten();
    if (g_fail) {
        std::printf("shadow_place_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("shadow_place_test: ok\n");
    return 0;
}
