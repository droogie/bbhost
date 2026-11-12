// The recorder's rest (host/recorder.cpp, BBHOST_RECORDER_REST): the decision
// alone, a function of half-second windows of the main loop's frame limiter and
// the GPU's busy share, so tests/recorder_rest_test.cpp drives it without the
// game.
//
// The recorder (bb-record) replays the command processor's blocks - about one
// a draw, ~1,000 a frame - into Vulkan as they are published. Between two
// blocks it spins (BBHOST_RECORDER_SPIN: 20,000 pauses with 16 hardware
// threads or more, 4,000 under) before it sleeps, and the command processor
// wakes a sleeping recorder at its next block: a wake is a system call on the
// command processor, and uncapped, where the command processor limits the
// frame, the spin pays (a Radeon 8060S CP-bound at Central Yharnam ran 2.7%
// faster at 20,000 pauses than at 4,000). Where the frame waits for something
// else the spin is the recorder waiting for a command processor busy a third
// of the frame: under Windows on an RTX 3070 Ti Laptop GPU at a true 60
// (1080p, the default path) the recorder spun 28-38% of a core at Old Yharnam,
// the Nightmare of Mensis and Central Yharnam and replayed 9-20%; on an RTX
// 4070 at a 120 cap it spun 50-64% of a core where the cap held the frame.
//
// So the recorder rests - it spins only kRestSpin pauses before it sleeps, and
// the command processor wakes a sleeping recorder only at a submission's block
// (a kick) or every kRestBatch blocks, so it replays its blocks in batches
// between its sleeps - where something else holds the frame with room to spare:
// a frame cap (two windows in a row with the limiter waiting 15% of the window
// or more, as the draw pool's rest at a cap, host/gxvk_rest.h) with the GPU
// busy under 85% of them, or the GPU under the cap (busy 90% of two windows in
// a row, the limiter waiting under 15%) with the command processor busy under
// 85% - bb-cp0's own CPU time; its jobs' time would take in its waits for the
// GPU, 98-101% of the window on a Radeon 680M at 32-37 fps with the thread busy
// 29-44%. Without that room the rest's delays land on the frame: the waits it
// adds to the command processor, the drains above all - Old Yharnam uncapped on
// that laptop (the GPU 93-96% busy, the command processor 91-95%) ran 2-4%
// slower, a Steam Deck's 60 fps profile at the Nightmare of Mensis (55 fps, the
// command processor 94-95%) 0.8% - and a submission it delays at a cap with the
// GPU near full: the same profile at Central Yharnam (the GPU 88-91% busy) had
// its 99th percentile frame go from 17 to 20 ms in one run of two, twice.
// Nothing waits on it longer than that: every wait of the command processor's
// on the recorder (a drain, a full ring, a free packet, a submission) wakes it
// at its first check, and a submission's block always does. A cap's rest ends
// once a window's limiter waits under 8% (the frame meets its cap narrowly, or
// misses it) or the GPU is busy 92%, a GPU's once the GPU is busy under 80%
// (the median of six windows), the command processor 92% or a cap holds the
// frame, and both at a loading screen. BBHOST_RECORDER_REST=0: it never rests;
// =always: it always does.
#pragma once

#include <algorithm>
#include <cstdint>

namespace recorder_rest {

constexpr double kRestSlack = 0.15;  // the limiter's wait, as a share of a window, that counts toward a rest at a cap
constexpr double kWakeSlack = 0.08;  // under it the cap no longer holds the frame
constexpr double kGpuBound = 0.90;   // the GPU busy this share of a window: the frame is the GPU's
constexpr double kGpuFree = 0.80;    // a GPU's rest ends with the GPU busy under this (the median of kGpuWindows)
constexpr double kCpRoom = 0.85;     // a GPU's rest begins only with bb-cp0 busy (its CPU time) under this share
constexpr double kCpFull = 0.92;     // ... and ends with it busy this share or more (the rest's waits would hold the frame)
constexpr double kGpuRoom = 0.85;    // a cap's rest begins only with the GPU busy under this share (time to spare in the frame)
constexpr double kGpuFull = 0.92;    // ... and ends with it busy this share or more (a delayed submission would miss the cap)
constexpr int kQuietWindows = 2;     // windows in a row with the cap holding the frame, or GPU-bound
constexpr int kGpuWindows = 6;       // resting for the GPU: its busy share is the median of this many windows
constexpr int kRestSpin = 100;       // resting: the pauses before the recorder sleeps
constexpr std::uint32_t kRestBatch = 64;  // resting: a sleeping recorder is woken every this many blocks (or at a kick)

struct State {
    bool cap = false;   // resting because a cap holds the frame
    bool gpu = false;   // resting because the frame is the GPU's
    int quiet = 0;      // windows in a row with the cap holding the frame (not resting for it)
    int bound = 0;      // windows in a row GPU-bound (not resting for it)
    double busy[kGpuWindows] = {};  // the last windows' GPU busy shares (a ring)
    int busy_n = 0;
    bool resting() const { return cap || gpu; }
};

namespace detail {
inline double median(const double* v, int n) {
    double t[kGpuWindows];
    n = std::min(n, kGpuWindows);
    std::copy(v, v + n, t);
    std::sort(t, t + n);
    return n ? (n & 1 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2])) : 0.0;
}
}  // namespace detail

// One window: the limiter's wait as a share of it, the GPU's busy share (`gpu`
// negative: not known - the GPU busy meter off), the command processor's busy
// share (`cp` negative: not known), whether a loading screen was up in it. True
// when the recorder rests after it.
inline bool step(State& s, double slack, double gpu, double cp, bool loading) {
    if (loading) {
        s = State{};
        return false;
    }
    // A cap, with the GPU's time to spare in the frame (`gpu` not known: the cap alone). At a cap with the GPU near
    // full, a submission the rest delays makes a frame miss it now and then.
    const bool gpu_known = gpu >= 0;
    if (s.cap) {
        if (slack < kWakeSlack || (gpu_known && gpu >= kGpuFull)) s.cap = false;
    } else {
        s.quiet = slack >= kRestSlack && (!gpu_known || gpu < kGpuRoom) ? s.quiet + 1 : 0;
        if (s.quiet >= kQuietWindows) {
            s.cap = true;
            s.quiet = 0;
        }
    }
    // The GPU, holding the frame under its cap (the limiter waiting under kRestSlack), with the command processor's room.
    if (gpu_known) {
        s.busy[s.busy_n % kGpuWindows] = gpu;
        ++s.busy_n;
    }
    if (s.gpu) {
        if (!gpu_known || cp >= kCpFull || slack >= kRestSlack || (s.busy_n >= kGpuWindows && detail::median(s.busy, kGpuWindows) < kGpuFree))
            s.gpu = false;
    } else {
        s.bound = gpu >= kGpuBound && cp < kCpRoom && slack < kRestSlack ? s.bound + 1 : 0;
        if (s.bound >= kQuietWindows) {
            s.gpu = true;
            s.bound = 0;
        }
    }
    return s.resting();
}

// The command processor at a block it published (not a kick) with the recorder
// asleep: whether it wakes it now. `unwoken`: the blocks published since its
// last wake (counted here).
inline bool wake_now(bool resting, std::uint32_t& unwoken) {
    if (!resting) return true;
    if (++unwoken < kRestBatch) return false;
    unwoken = 0;
    return true;
}

}  // namespace recorder_rest
