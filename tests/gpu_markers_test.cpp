// The markers between the draws (host/gpu_markers.h), without a GPU: the ring
// of op notes hands back the note of a number it still holds and nothing for
// one it has moved past (or 0, which the marker buffer holds before any op),
// across its wrap and the counter's; and the op word read against the draw
// the GPU had started as the report reads it - the ops the front end reached
// after that draw, and the barrier among them past which the draw had
// finished; other threads' notes in between left out, and nothing claimed
// when the ring has moved past them.
//
// And when they are written: what BBHOST_GPU_CHECKPOINTS asks on devices with
// and without buffer markers (by default only while a loading screen is up,
// nothing in the world); the window's steps where recordings begin - opened
// at the first after a loading screen came up, held for its grace after it
// went and opened again only by the next, never in the world, and once for
// good with =amd; its marks, which no draw's, dispatch's or submission's is
// and never 0; and the words read back as the report reads them - nothing
// reached, a window begun, inside one, past its end, and whether what came
// before had finished.
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

    // ---- When they are written ----
    {
        // Unset (null or empty): while a loading screen is up, where the device has buffer markers; nothing otherwise.
        Arming a = arming(nullptr, true, true);
        check(a.mode == Mode::kLoading && !a.dry && !a.nv && !a.after_loss && !a.unknown, "unset: loading screens");
        a = arming("", true, false);
        check(a.mode == Mode::kLoading && !a.unknown, "empty: as unset");
        a = arming(nullptr, false, true);
        check(a.mode == Mode::kOff && !a.nv && !a.after_loss, "unset without buffer markers: none, not NVIDIA's checkpoints either");
        // =0: none at all, not even in loading screens.
        a = arming("0", true, true);
        check(a.mode == Mode::kOff && !a.nv && !a.after_loss && !a.dry, "=0: none");
        // =amd: everywhere where there are buffer markers.
        a = arming("amd", true, true);
        check(a.mode == Mode::kAlways && !a.dry && !a.nv, "=amd: everywhere");
        a = arming("amd", false, true);
        check(a.mode == Mode::kOff && !a.nv, "=amd without buffer markers: none");
        // =1: NVIDIA's checkpoints, no buffer markers.
        a = arming("1", true, true);
        check(a.nv && a.mode == Mode::kOff, "=1: NVIDIA's checkpoints instead");
        a = arming("1", true, false);
        check(!a.nv && a.mode == Mode::kOff, "=1 without them: none");
        // The dry ones: recorded where the device has none.
        a = arming("amd-dry", false, false);
        check(a.mode == Mode::kAlways && a.dry, "=amd-dry on lavapipe: everywhere, recorded only");
        a = arming("amd-dry", true, false);
        check(a.mode == Mode::kAlways && !a.dry, "=amd-dry with buffer markers: written");
        a = arming("loading-dry", false, false);
        check(a.mode == Mode::kLoading && a.dry, "=loading-dry on lavapipe: loading screens, recorded only");
        a = arming("loading-dry", true, false);
        check(a.mode == Mode::kLoading && !a.dry, "=loading-dry with buffer markers: written");
        // =after-loss: loading screens, and the starts after a loss everything - by buffer markers first.
        a = arming("after-loss", true, true);
        check(a.mode == Mode::kLoading && a.after_loss && !a.nv, "=after-loss: loading screens, and armed");
        after_loss_start(a, true);
        check(a.mode == Mode::kAlways && !a.nv, "a start after a loss: everywhere, by buffer markers");
        a = arming("after-loss", false, true);
        check(a.mode == Mode::kOff && a.after_loss, "=after-loss with NVIDIA's checkpoints alone: armed");
        after_loss_start(a, false);
        check(a.nv && a.mode == Mode::kOff, "a start after a loss: NVIDIA's checkpoints");
        a = arming("after-loss", false, false);
        check(!a.after_loss && a.mode == Mode::kOff, "=after-loss with neither: nothing to arm");
        // Not understood: as unset, and said. A prefix of a value is not the value.
        a = arming("amdx", true, false);
        check(a.unknown && a.mode == Mode::kLoading, "a value not understood: as unset");
        a = arming("am", true, false);
        check(a.unknown && a.mode == Mode::kLoading, "a prefix of amd is not amd");
        a = arming("after", true, false);
        check(a.unknown && !a.after_loss, "a prefix of after-loss is not after-loss");
        a = arming("amd-dryer", false, false);
        check(a.unknown && !a.dry && a.mode == Mode::kOff, "amd-dry with more after it is not amd-dry");
    }
    {
        // The marks: kind 0, never 0, the beginning's low bit set, the number above it; apart from every draw's,
        // dispatch's and submission's.
        static const std::uint32_t kNumbers[] = {1u, 2u, 3u, 1000u, 0x1fffffffu};
        for (std::uint32_t n : kNumbers) {
            const std::uint32_t b = window_value(n, true), e = window_value(n, false);
            check(b && e && b != e, "a window's marks are not 0 and not each other", n);
            check((b >> 30) == 0 && (e >> 30) == 0, "kind 0: no draw's, dispatch's or submission's", n);
            check((b & 1) && !(e & 1), "the beginning's low bit", n);
            check(read_words(b, b).window == n && read_words(e, e).window == n, "the number read back", n);
            for (unsigned kind = 1; kind < 4; ++kind) check(window_value(n, true) != mark(kind, n), "not a draw's, dispatch's or submission's", n);
        }
    }
    {
        // =0 (or a device without buffer markers): the window never opens, loading screen or not.
        Window w;
        bool any = false;
        for (int k = 0; k < 50; ++k) any = any || w.step(Mode::kOff, k % 3 == 0, 1000u * k);
        check(!any && !w.open && w.number == 0, "never opens with the markers off");
        // =amd: open at the first recording, for good.
        Window al;
        check(al.step(Mode::kAlways, false, 0) == window_value(1, true) && al.open, "=amd: opens at the first recording");
        any = false;
        for (int k = 0; k < 50; ++k) any = any || al.step(Mode::kAlways, k % 2 == 0, 1000u * k);
        check(!any && al.open && al.number == 1, "=amd: never ends, never begins again");
        check(!al.needs_clock(Mode::kAlways, false), "=amd: no clock read");
    }
    {
        // By default: the world (nothing), a loading screen (opens at once), its end (held for the grace), the world.
        Window w;
        check(w.step(Mode::kLoading, false, 100) == 0 && !w.open, "the world: closed");
        check(!w.needs_clock(Mode::kLoading, false), "the world: no clock read while closed");
        check(w.step(Mode::kLoading, true, 200) == window_value(1, true) && w.open, "a loading screen: opens at the first recording");
        check(w.step(Mode::kLoading, true, 300) == 0 && w.open, "open while it is up");
        check(!w.needs_clock(Mode::kLoading, true), "no clock read while it is up");
        check(w.needs_clock(Mode::kLoading, false), "the clock read once it has gone");
        check(w.step(Mode::kLoading, false, 1000) == 0 && w.open, "gone: still open (the grace)");
        check(w.step(Mode::kLoading, false, 1000 + kWindowGraceMs - 1) == 0 && w.open, "open to the grace's last ms");
        check(w.step(Mode::kLoading, false, 1000 + kWindowGraceMs) == window_value(1, false) && !w.open, "closed at the grace's end");
        check(w.step(Mode::kLoading, false, 5000) == 0 && !w.open, "the world after it: closed");
        // The next loading screen: window 2; back during its grace: still open, the grace counted afresh from the next end.
        check(w.step(Mode::kLoading, true, 6000) == window_value(2, true), "the next loading screen: window 2");
        check(w.step(Mode::kLoading, false, 7000) == 0, "gone");
        check(w.step(Mode::kLoading, true, 7100) == 0 && w.open, "up again within the grace: the same window");
        check(w.step(Mode::kLoading, false, 8000) == 0 && w.open, "gone again: the grace from now");
        check(w.step(Mode::kLoading, false, 8000 + kWindowGraceMs - 1) == 0 && w.open, "not the first gone's grace");
        check(w.step(Mode::kLoading, false, 8000 + kWindowGraceMs) == window_value(2, false) && !w.open, "closed at the second grace's end");
        check(w.number == 2, "two windows");
        // The count's wrap: never a 0 mark.
        Window wrap;
        wrap.number = 0x1fffffffu;
        const std::uint32_t v = wrap.step(Mode::kLoading, true, 0);
        check(v && v == window_value(1, true), "the number skips 0 as it wraps");
    }
    {
        // The words as the report reads them.
        Reading r = read_words(0, 0);
        check(r.where == Where::kNothing, "nothing written: nothing reached");
        r = read_words(mark(1, 77), mark(1, 76));
        check(r.where == Where::kInside, "a draw's: inside a window");
        r = read_words(mark(2, 5), mark(1, 76));
        check(r.where == Where::kInside, "a dispatch's: inside");
        r = read_words(mark(3, 4 * 9 + 1), mark(1, 76));
        check(r.where == Where::kInside, "a submission's: inside");
        r = read_words(window_value(5, true), window_value(5, true));
        check(r.where == Where::kBegun && r.window == 5 && r.finished_all, "a beginning reached, all before it finished");
        r = read_words(window_value(5, true), window_value(4, false));
        check(r.where == Where::kBegun && !r.finished_all, "a beginning reached, the work before it not all finished");
        r = read_words(window_value(5, false), window_value(5, false));
        check(r.where == Where::kEnded && r.window == 5 && r.finished_all, "past an end, all before it finished");
        r = read_words(window_value(5, false), mark(1, 900));
        check(r.where == Where::kEnded && r.window == 5 && !r.finished_all, "past an end, its last work not all finished");
        // A window's marks from its steps, read back.
        Window w;
        const std::uint32_t begun = w.step(Mode::kLoading, true, 0);
        check(read_words(begun, begun).where == Where::kBegun && read_words(begun, begun).window == 1, "the step's beginning reads as one");
        w.step(Mode::kLoading, false, 10);
        const std::uint32_t ended = w.step(Mode::kLoading, false, 10 + kWindowGraceMs);
        check(read_words(ended, ended).where == Where::kEnded && read_words(ended, ended).window == 1, "the step's end reads as one");
    }

    if (g_failures) {
        std::printf("gpu_markers_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("gpu_markers_test: ok\n");
    return 0;
}
