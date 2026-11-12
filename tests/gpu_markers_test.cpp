// The markers between the draws (host/gpu_markers.h), without a GPU: the ring
// of op notes hands back the note of a number it still holds and nothing for
// one it has moved past (or 0, which the marker buffer holds before any op),
// across its wrap and the counter's; and the op word read against the draw
// the GPU had started as the report reads it - the ops the front end reached
// after that draw, and the barrier among them past which the draw had
// finished; other threads' notes in between left out, and nothing claimed
// when the ring has moved past them.
#include "host/gpu_markers.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;
void check(bool ok, const char* what, unsigned long long at = 0) {
    if (ok) return;
    if (++g_failures <= 24) std::printf("FAIL: %s (%llu)\n", what, at);
}

std::uint32_t mark(unsigned kind, std::uint32_t index) { return (kind & 3u) << 30 | (index & 0x3fffffffu); }

// The kinds of this test's notes (the real ones are gpu::stream::Op values).
enum Kind : std::uint16_t { kCopy = 1, kEndPassWithBarrier, kTransferIn, kTransferOut, kBeginPass, kPresent };
int classify(const gpu_markers::Note& n) {
    switch (n.kind) {
    case kEndPassWithBarrier:
    case kTransferIn: return gpu_markers::kHolds;
    case kPresent: return gpu_markers::kOtherThread;
    default: return gpu_markers::kPlain;
    }
}

}  // namespace

int main() {
    using namespace gpu_markers;

    // The slots: distinct words of the 256-byte marker buffer, the old two where they were.
    check(kStarted == 0 && kFinishedBefore == 4, "the draw markers' words stay at 0 and 4");
    check(kOpReached % 4 == 0 && kOpReached >= 8 && kOpReached + 4 <= 256, "the op word is a dword of its own inside it");

    // The ring: numbers from 1, each note found while the ring holds it.
    static Ring<16> ring;
    check(ring.find(0) == nullptr, "0 is no op");
    check(ring.find(1) == nullptr, "nothing noted yet");
    std::uint32_t first = 0, last = 0;
    for (std::uint32_t k = 0; k < 40; ++k) {
        const std::uint32_t seq = ring.note(static_cast<std::uint16_t>(k % 7), mark(1, 1000 + k), k, k * 2, k * 3);
        if (!first) first = seq;
        check(seq == last + 1, "numbers count up", seq);
        last = seq;
        const Note* n = ring.find(seq);
        check(n && n->kind == k % 7 && n->after == mark(1, 1000 + k) && n->a == k && n->b == k * 2 && n->c == k * 3, "its own note", seq);
    }
    check(first == 1, "the first number is 1");
    for (std::uint32_t seq = last - 15; seq <= last; ++seq) check(ring.find(seq) != nullptr, "the last 16 are held", seq);
    for (std::uint32_t seq = 1; seq + 16 <= last; ++seq) check(ring.find(seq) == nullptr, "older ones are gone", seq);
    check(ring.find(last + 1) == nullptr, "a number not given yet");

    // The counter's wrap: never 0, and the notes after it found.
    static Ring<16> wrap;
    wrap.last = 0xfffffffeu;
    const std::uint32_t a = wrap.note(1, 0, 1, 0, 0), b = wrap.note(2, 0, 2, 0, 0), c = wrap.note(3, 0, 3, 0, 0);
    check(a == 0xffffffffu && b == 1 && c == 2, "the counter skips 0 as it wraps", b);
    check(wrap.find(a) && wrap.find(a)->a == 1 && wrap.find(b) && wrap.find(b)->a == 2 && wrap.find(c) && wrap.find(c)->a == 3,
          "the notes across the wrap");

    // The op word against the draw started: draw 4's ops, then the composite (draw 5) and the window after it - its pass
    // end with the barrier, the presenter's note between (another thread's), the transfer barrier in, four copies, the
    // transfer barrier out and the next pass's begin, all before draw 6.
    static Ring<64> w;
    const std::uint32_t d4 = mark(1, 4), d5 = mark(1, 5), d6 = mark(1, 6);
    const std::uint32_t before = w.note(kCopy, d4, 0, 0, 0);
    const std::uint32_t end_pass = w.note(kEndPassWithBarrier, d5, 1, 0, 0);
    const std::uint32_t present = w.note(kPresent, 0, 0, 0, 0);
    const std::uint32_t in = w.note(kTransferIn, d5, 1, 0, 0);
    std::uint32_t copies[4];
    for (std::uint32_t& k : copies) k = w.note(kCopy, d5, 0, 0, 0);
    const std::uint32_t out = w.note(kTransferOut, d5, 0, 0, 0);
    const std::uint32_t begin = w.note(kBeginPass, d5, 0, 0, 0);

    // The front end stopped at the pass end's barrier: no barrier past - the composite may not have finished.
    Reach r = reach_after(w, d5, end_pass, classify);
    check(r.known && r.ops == 1 && r.first == end_pass && r.holds == 0, "stopped at the barrier after the draw: nothing past it");
    // Past the pass end's barrier, into the transfer barrier: the composite had finished (the presenter's note skipped).
    r = reach_after(w, d5, in, classify);
    check(r.known && r.ops == 2 && r.first == end_pass && r.holds == end_pass, "past the pass end: the draw had finished");
    // Into the copies and on to the barrier out: the last barrier past is the transfer barrier in.
    r = reach_after(w, d5, out, classify);
    check(r.known && r.ops == 7 && r.first == end_pass && r.holds == in, "the copies reached: past the transfer barrier in");
    r = reach_after(w, d5, begin, classify);
    check(r.known && r.ops == 8 && r.holds == in, "the next pass's begin reached");
    // An op word from before the draw: the front end reached no op after it.
    r = reach_after(w, d5, before, classify);
    check(r.known && r.ops == 0 && r.holds == 0, "an op before the draw: none after it");
    // The word against another draw: the ops after draw 5 are not draw 6's.
    r = reach_after(w, d6, begin, classify);
    check(r.known && r.ops == 0, "draw 6 started: the ops after draw 5 are not after it");
    // The presenter's own note as the word: its place among the stream's ops is not known - nothing inferred.
    r = reach_after(w, d5, present, classify);
    check(r.other && r.ops == 0 && r.holds == 0, "the presenter's note: said as such, nothing inferred");
    r = reach_after(w, d5, in, classify);
    check(!r.other, "a stream op is not another thread's");
    (void)copies;

    // The ring moved past the window's start: said, not guessed.
    static Ring<4> small;
    small.note(kCopy, d4, 0, 0, 0);
    small.note(kEndPassWithBarrier, d5, 1, 0, 0);
    for (int k = 0; k < 4; ++k) small.note(kCopy, d5, 0, 0, 0);
    r = reach_after(small, d5, small.last, classify);
    check(!r.known && r.ops == 4 && r.holds == 0, "the ring has moved past the pass end");
    // Nothing reached yet.
    r = reach_after(w, d5, 0, classify);
    check(r.known && r.ops == 0, "no op word yet");

    if (g_failures) {
        std::printf("gpu_markers_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("gpu_markers_test: ok\n");
    return 0;
}
