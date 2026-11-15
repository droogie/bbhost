// The GPU's progress markers between the draws, for a device-loss report
// (VK_AMD_buffer_marker, which NVIDIA's drivers have too: while a loading
// screen is up by default, everywhere with BBHOST_GPU_CHECKPOINTS=amd;
// gpu.cpp gpu_write_markers - the loading screens' window is at the end).
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
// it is).
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

// ---- When they are written: the loading screens' window ----
//
// By default the markers are written only while a loading screen is up, and
// nothing of them is written or done in the world. Losses come in loading
// screens: four of five on one RTX 2060 under Windows were in or just after
// one, most of them a Chalice Dungeon's - where the fast loading screens run
// the GPU flat out, several hundred frames a second - and none of them named
// where the GPU was, the markers being off. In the world they cost frames
// (the 1.5% above); a loading screen's length is set by the work the game's
// loaders do, not by how many frames show. BBHOST_GPU_CHECKPOINTS=amd writes
// them everywhere, =0 never.
enum class Mode : int { kOff, kLoading, kAlways };

// What BBHOST_GPU_CHECKPOINTS asks (`env`, null when unset) on a device with
// AMD's buffer markers (`buffer_markers`) or NVIDIA's checkpoints:
//   unset        the buffer markers while a loading screen is up
//   0            none, not even in loading screens
//   amd          the buffer markers everywhere
//   1            NVIDIA's checkpoints before every draw and dispatch instead
//   after-loss   as unset, and a start after a lost device marks everything,
//                for the three starts its note gives (gpu.cpp take_marked_start)
//   amd-dry, loading-dry
//                everywhere / in loading screens, recorded as they would be and
//                the GPU's writes left out where the device has no buffer
//                markers (lavapipe: a check of the recording)
// Anything else is taken as unset, and said.
struct Arming {
    Mode mode = Mode::kOff;   // the buffer markers: when
    bool dry = false;         // recorded only: the device has none to write
    bool nv = false;          // NVIDIA's checkpoints instead
    bool after_loss = false;  // a start after a lost device marks everything
    bool unknown = false;     // a value not understood
};
inline Arming arming(const char* env, bool buffer_markers, bool nv_checkpoints) {
    Arming a;
    const Mode loading = buffer_markers ? Mode::kLoading : Mode::kOff;
    const auto is = [env](const char* v) {
        std::size_t i = 0;
        for (; v[i]; ++i) {
            if (env[i] != v[i]) return false;
        }
        return env[i] == 0;
    };
    if (!env || !*env) {
        a.mode = loading;
    } else if (env[0] == '0') {
        // none
    } else if (env[0] == '1') {
        a.nv = nv_checkpoints;
    } else if (is("amd")) {
        a.mode = buffer_markers ? Mode::kAlways : Mode::kOff;
    } else if (is("amd-dry")) {
        a.mode = Mode::kAlways;
        a.dry = !buffer_markers;
    } else if (is("loading-dry")) {
        a.mode = Mode::kLoading;
        a.dry = !buffer_markers;
    } else if (is("after-loss")) {
        a.mode = loading;
        a.after_loss = buffer_markers || nv_checkpoints;
    } else {
        a.mode = loading;
        a.unknown = true;
    }
    return a;
}
// A start after a lost device with after-loss: everything marked, by the
// buffer markers where there are any, NVIDIA's checkpoints otherwise.
inline void after_loss_start(Arming& a, bool buffer_markers) {
    if (buffer_markers) {
        a.mode = Mode::kAlways;
    } else {
        a.nv = true;
    }
}

// A window's own mark in the started and finished words: kind 0, which no
// draw, dispatch or submission has, with its number and whether it begins or
// ends there - never 0, what the buffer holds before anything is written.
constexpr std::uint32_t window_value(std::uint32_t number, bool begins) {
    return (number & 0x1fffffffu) << 1 | (begins ? 1u : 0u);
}

// How long a window stays open after the loading screen has gone: the command
// processor records the game's frames a frame or two after the game made
// them, so its last frames - and the world's first, where a loss right after
// a load falls - come after the game has left the loading screen.
constexpr std::uint64_t kWindowGraceMs = 200;

// The command processor's window, stepped where each recording (a
// submission) begins - where no draw, dispatch or run is open, so every one
// is wholly in or out of it: it begins at the first recording after the
// loading screen came up and ends at the first one kWindowGraceMs after it
// went. step() returns the window's mark to record there (0: no change),
// which the recorder meets in the stream's order: from a beginning it notes
// the ops and writes their words, at an end it stops (recorder.cpp
// record_marker_window), so the words always come from one side of it.
struct Window {
    bool open = false;
    std::uint32_t number = 0;   // windows begun so far
    bool going = false;         // open, the loading screen gone: since
    std::uint64_t gone_ms = 0;  // (the clock is read only then)
    std::uint32_t step(Mode mode, bool loading_screen, std::uint64_t now_ms) {
        bool want = mode == Mode::kAlways || (mode == Mode::kLoading && loading_screen);
        if (open && !want && mode == Mode::kLoading) {
            if (!going) {
                going = true;
                gone_ms = now_ms;
            }
            if (now_ms - gone_ms < kWindowGraceMs) return 0;
        }
        going = false;
        if (want == open) return 0;
        open = want;
        if (want && ((++number & 0x1fffffffu) == 0)) number = 1;
        return window_value(number, want);
    }
    // Whether step() needs the clock: open, with the loading screen gone.
    bool needs_clock(Mode mode, bool loading_screen) const { return open && mode == Mode::kLoading && !loading_screen; }
};

// What the started and finished words say of the window: nothing written
// yet; the front end had reached a window's beginning and no draw or
// dispatch of it; inside one (a draw's, a dispatch's or a submission's mark);
// past a window's end, into work recorded without markers. `finished_all`:
// everything recorded before that beginning or end had finished.
enum class Where : int { kNothing, kBegun, kInside, kEnded };
struct Reading {
    Where where = Where::kNothing;
    std::uint32_t window = 0;
    bool finished_all = false;
};
constexpr Reading read_words(std::uint32_t started, std::uint32_t finished) {
    Reading r;
    if (!started) return r;
    if (started >> 30) {
        r.where = Where::kInside;
        return r;
    }
    r.where = (started & 1) ? Where::kBegun : Where::kEnded;
    r.window = (started & 0x3fffffffu) >> 1;
    r.finished_all = finished == started;
    return r;
}

}  // namespace gpu_markers
