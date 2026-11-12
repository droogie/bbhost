// The GPU's progress markers between the draws, for a device-loss report
// (VK_AMD_buffer_marker: on a start after a lost device, or with
// BBHOST_GPU_CHECKPOINTS=amd; gpu.cpp gpu_write_markers).
//
// Each draw and dispatch has two marker writes before it: the top of the pipe
// (the GPU reached it) and the bottom (everything before it had finished). On
// a loss they name the draw the GPU had started, and say that everything
// before it had finished - but not whether that draw itself finished, nor
// what came after it: the GPU can stop anywhere up to the next draw's marker,
// in the draw or in the ops recorded between the two (the end of its pass and
// the barrier with it, copies, fills, clears, queries, the next pass's
// begin). An RTX 2060 SUPER under Windows lost the device with the marker on
// the light composite (8747a367+32e47788), a draw with no loop, no branch and
// no read its buffers do not hold: the window is what the report has to name.
//
// So the recorder also notes, in its replay, each op between draws - its
// kind, a few words (buffers, images, sizes, stages) and the draw or dispatch
// replayed before it - in a ring the report reads back by number, and writes
// a top-of-pipe marker with the op's number (kOpReached: the last op the GPU's
// front end reached) before the ops where the front end stops or the pass
// changes: barriers, pass ends and begins, the host's own dispatches,
// secondaries executed, the presenter's command buffer (recorder.cpp
// word_on). The front end runs on past the other ops until one of these, so
// that is where its word is left at a loss, and the report lists the ops
// after it that had none. What that gives up is where the front end was held
// between two such ops (by a full queue ahead of it): BBHOST_GPU_MARKER_OPS=all
// writes the word before every op for that. Top of the pipe only, and no word
// on every op by default: a bottom-of-pipe word per op and one after each
// draw made the ops wait for one another and for the draw before them, and on
// NVIDIA's GPUs a word between two copies or fills held them apart too (an RTX
// 3070 Ti Laptop GPU, Central Yharnam uncapped, against the two words a draw
// alone: 10.7% slower with those, 4.2% with a word before every op, 1.5% as
// it is) - and the three starts after a loss run with the markers on.
//
// What the report infers from the op word rests on barriers holding the
// front end: the GPU does not go past a pipeline barrier until the work its
// source stages name has finished, so an op reached past such a barrier after
// the draw says the draw had finished (and the GPU stopped in the ops after
// that barrier), and an op word that stops at that barrier says it had not.
// NVIDIA's and AMD's GPUs hold the front end so; the specification does not
// promise it.
#pragma once

#include <cstdint>

namespace gpu_markers {

// The marker buffer's words (byte offsets into Gpu::marker_buf).
constexpr std::uint32_t kStarted = 0;         // top of the pipe before a draw or dispatch: its marker value
constexpr std::uint32_t kFinishedBefore = 4;  // bottom of the pipe there: everything before it had finished
constexpr std::uint32_t kOpReached = 8;       // top of the pipe before an op between draws: its sequence number

// An op the recorder replayed between draws: its kind (gpu::stream::Op), the
// draw or dispatch marker value replayed before it, and up to three words.
struct Note {
    std::uint32_t seq = 0;  // its sequence number; 0: none
    std::uint16_t kind = 0;
    std::uint32_t after = 0;
    std::uint64_t a = 0, b = 0, c = 0;
};

// The last N notes by sequence number. One writer at a time (the recorder's
// thread, the presenter's under the same lock); the report reads it after the
// device is lost, when the writers have stopped.
template <std::uint32_t N> struct Ring {
    static_assert(N && (N & (N - 1)) == 0, "a power of two");
    Note notes[N];
    std::uint32_t last = 0;
    std::uint32_t note(std::uint16_t kind, std::uint32_t after, std::uint64_t a, std::uint64_t b, std::uint64_t c) {
        if (++last == 0) last = 1;  // never 0, which the marker buffer holds before any op
        Note& n = notes[last & (N - 1)];
        n.seq = last;
        n.kind = kind;
        n.after = after;
        n.a = a;
        n.b = b;
        n.c = c;
        return last;
    }
    // The note of `seq`, or null when it is 0 or the ring has moved past it.
    const Note* find(std::uint32_t seq) const {
        if (!seq) return nullptr;
        const Note& n = notes[seq & (N - 1)];
        return n.seq == seq ? &n : nullptr;
    }
};

// An op's note against the draw or dispatch before it: a plain op; a barrier
// whose source stages hold the front end until that draw or dispatch has
// finished; or another thread's (the presenter's command buffer, numbered
// from the same ring but not between these draws in the stream).
enum OpClass : int { kPlain, kHolds, kOtherThread };

// What the GPU's front end had reached after the draw or dispatch it had
// started (`started`: its marker value), the op word being `reached`: the ops
// noted after that draw up to and with `reached` (`ops`, the first of them
// `first`), and the last of them before `reached` that holds the front end
// (`holds`, 0: none) - past it, the draw had finished. `ops` 0: the front end
// had reached no op after the draw. `known` false: the ring had moved past
// some of them. `other`: the word is another thread's note, whose place among
// the stream's ops the numbers do not give - nothing is inferred from it.
struct Reach {
    std::uint32_t ops = 0, first = 0, holds = 0;
    bool known = true, other = false;
};
template <std::uint32_t N, typename Classify>
Reach reach_after(const Ring<N>& ring, std::uint32_t started, std::uint32_t reached, Classify classify) {
    Reach r;
    if (const Note* n = ring.find(reached); n && classify(*n) == kOtherThread) {
        r.other = true;
        return r;
    }
    std::uint32_t steps = 0;
    for (std::uint32_t seq = reached; seq; --seq, ++steps) {  // (0: before the first op)
        const Note* n = steps < N ? ring.find(seq) : nullptr;
        if (!n) {
            r.known = false;
            break;
        }
        const int c = classify(*n);
        if (c == kOtherThread) continue;
        if (n->after != started) break;  // recorded before the draw
        ++r.ops;
        r.first = seq;
        if (seq != reached && c == kHolds && !r.holds) r.holds = seq;
    }
    return r;
}

}  // namespace gpu_markers
