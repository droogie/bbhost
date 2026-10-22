#pragma once
// The buffer shadow's choices for a page read in place (buffer_shadow.cpp,
// BBHOST_SHADOW_IN_PLACE), kept apart from Vulkan so a test can drive them.
//
// On an integrated GPU the mirrors' video memory is the same RAM the imports
// are, so a copy of a page buys no bandwidth. What it does buy is a stable
// version for the draws already queued when the CPU writes the page again,
// and the page tracking that finds the vertex data the game keeps rewriting
// (those ranges are copied as their draws record instead). A page whose every
// write GX tells the shadow of - GX buffers cover it whole and GX never handed
// it out to be rewritten - needs neither a copy nor a write watch to be
// tracked: it is read where it is, and a told write counts as a rewrite, as a
// copy did.
#include <cstdint>

namespace shadow_place {

// A rewrite is counted as a copy was: past `limit` of them inside the window,
// the page holds data the game keeps rewriting and goes to host memory for
// good (its ranges copied as their draws record). The window starts again at
// the first rewrite after it ends.
struct RewriteWindow {
    std::uint64_t start_ms = 0;
    std::uint32_t count = 0;
};
// One more rewrite at `now_ms`: true when it is one too many.
inline bool rewrite_too_many(RewriteWindow& w, std::uint64_t now_ms, std::uint32_t limit, std::uint64_t window_ms) {
    if (now_ms - w.start_ms >= window_ms) {
        w.start_ms = now_ms;
        w.count = 0;
    }
    return ++w.count > limit;
}

// What a page tells about itself for the choice.
struct PageFacts {
    bool told_cover = false;  // GX buffers cover the page whole: GX's calls tell its writes
    bool handed = false;      // GX handed (part of) it out to be rewritten (renamed copies)
    bool dynamic = false;     // already sent to host memory for good
};

// A page may be read in place when GX tells every write to it.
inline bool eligible(const PageFacts& f) { return f.told_cover && !f.handed && !f.dynamic; }

// A range's binding: in place when every page is eligible, host memory when
// any is dynamic (as the mirrors already do), else the mirror path - which
// copies the range's pages, the eligible ones too, so a range is never split.
enum class Range { kInPlace, kHost, kMirror };
template <typename Facts>
Range choose(const Facts* pages, std::uint32_t n) {
    bool all = n != 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (pages[i].dynamic) return Range::kHost;
        all = all && eligible(pages[i]);
    }
    return all ? Range::kInPlace : Range::kMirror;
}

// A page placed in place is written again when GX told a write over it after
// its placement (sequence numbers of the shadow's drained GX writes).
inline bool rewritten(std::uint64_t placed_seq, std::uint64_t newest_write_seq) { return newest_write_seq > placed_seq; }

}  // namespace shadow_place
