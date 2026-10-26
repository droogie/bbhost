// The bottleneck meter's verdict: which stage holds the frame rate in a
// window, from that window's counters. host/perf_meter.cpp collects the
// counters and logs the verdicts (`perf:` each second with
// BBHOST_FRAME_STATS=1, `perf summary:` about once a minute always);
// tools/perf_verdict.py reads both and keeps a copy of this function for old
// logs' `frames:` lines. A pure function, so tests/bottleneck_test.cpp feeds
// it the windows the meter met.
//
// The stages, in the order a frame goes through them:
//   main     the game's main loop on the main thread: its own work (the
//            game's code and ours) and its waits on its job threads;
//   gx       the GX workers the main loop waits for: the main thread spinning
//            in the GX job barrier (sub_25b8710, a guest loop without `pause`,
//            so it reads as main-thread work to a CPU counter);
//   render   the game's render thread (GXRenderThread): the draw lists into
//            command buffers, submitted to the command processor;
//   cp       the command processor (bb-cp0): the command buffers walked and
//            drawn;
//   record   the recorder (bb-record): the draws replayed into Vulkan;
//   gpu      the GPU;
//   present  the presenter (bb-present).
// And three states that are not stages: capped (the limiter sleeps: the
// frame cap or V-Sync holds the rate), stall (long frames: a load, pipeline
// builds, imports) and none (nothing is near full: no verdict).
//
// How it decides, in order:
//   1. Under 2 frames a second, or a loading screen up half the window, or
//      long frames (over twice the median and 25 ms) taking a fifth of it
//      past the median: stall, with the largest cause.
//   2. The rate at the cap (98.5% of it) with the limiter waiting 2% of the
//      window: capped.
//   3. The GPU busy 90% of the window: gpu.
//   4. The game waited for the command processor's backlog (or for GPU memory
//      to retire) 5% of the window: the command processor or a stage after
//      it - the recorder if it is the busier and 85% busy, the GPU if the
//      command processor waits for it a tenth of the time, else cp.
//   5. The render thread 85% on a core: render.
//   6. The main loop busy (not waiting in the limiter) 90% of the window:
//      the main loop is the limit - gx when the barrier's spin is a third of
//      the main loop's time or more than the main thread's own work, else main.
//   7. The presenter 80% busy with the display showing under 90% of the
//      flips: present.
//   8. The limiter waiting a tenth of the window under the cap (uneven
//      frames): capped. Else none.
// The thresholds are what the RTX 4070, the Steam Deck and the Radeon 680M
// showed between their limited and unlimited stages (docs/performance.md).
#pragma once

#include <cstddef>

namespace bottleneck {

enum class Stage : int { kNone, kCapped, kStall, kMain, kGx, kRender, kCp, kRecord, kGpu, kPresent, kCount };

inline const char* stage_name(Stage s) {
    switch (s) {
    case Stage::kNone: return "none";
    case Stage::kCapped: return "capped";
    case Stage::kStall: return "stall";
    case Stage::kMain: return "main";
    case Stage::kGx: return "gx";
    case Stage::kRender: return "render";
    case Stage::kCp: return "cp";
    case Stage::kRecord: return "record";
    case Stage::kGpu: return "gpu";
    case Stage::kPresent: return "present";
    case Stage::kCount: break;
    }
    return "?";
}

constexpr int kStages = static_cast<int>(Stage::kCount);

// One window's counters, in milliseconds of the window unless said; -1 is
// "not measured" (a platform or a build without it).
struct Window {
    double secs = 0;            // the window's length
    double flips = 0;           // flips the game submitted
    double displayed = -1;      // flips the display completed
    double cap_fps = 0;         // the limiter's target, 0 for none (a loading screen, the game's own 30)
    double median_ms = 0;       // the median flip interval
    double long_ms = 0;         // what long frames took past the median
    double loading_ms = 0;      // a loading screen was up
    double limiter_ms = -1;     // the main thread waiting in the frame limiter (sleep and spin)
    double main_cpu_ms = -1;    // the main thread on a core
    double gx_spin_ms = -1;     // of that, spinning in the GX job barrier
    double main_wait_ms = 0;    // the main thread blocked on other threads
    double main_file_ms = 0;    // the main thread in file I/O
    double backlog_ms = 0;      // any thread waiting for the command processor's backlog or GPU memory to retire
    double render_cpu_ms = -1;  // GXRenderThread on a core
    double cp_cpu_ms = -1;      // bb-cp0 on a core
    double cp_gpu_wait_ms = 0;  // the command processor waiting for the GPU (fences, labels)
    double rec_busy_ms = -1;    // bb-record replaying draws
    double gpu_busy_ms = -1;    // the GPU busy with our work (host/gpu_busy.cpp)
    double present_cpu_ms = -1; // bb-present on a core
    double import_ms = 0;       // the command processor waiting for guest memory imports
    double pipeline_ms = 0;     // the command processor building pipelines on its thread
};

struct Verdict {
    Stage limit = Stage::kNone;
    Stage next = Stage::kNone;  // the busiest stage but the limit (for capped and stall: the busiest)
    double limit_load = -1;     // the share of the window the limit was busy, 0..1 (-1: a state, not a stage)
    double next_load = -1;
    const char* why = "";       // a few words: what decided it
    double fps = 0, frame_ms = 0;
    double load[kStages];       // by Stage, 0..1, -1 not measured (capped/stall/none: -1)
    double main_loop = -1;      // the main loop's busy share (not in the limiter), -1 not measured
    double backlog = 0;         // the share of the window the game waited for the command processor
};

inline double share(double ms, double win_ms) { return ms < 0 || win_ms <= 0 ? -1 : ms / win_ms; }

inline Verdict classify(const Window& w) {
    Verdict v;
    for (double& l : v.load) l = -1;
    if (w.secs <= 0) return v;
    const double win = w.secs * 1000.0;
    v.fps = w.flips / w.secs;
    v.frame_ms = w.flips > 0 ? win / w.flips : 0;
    // The main thread's own work: on a core and not in the barrier's spin.
    double own = w.main_cpu_ms;
    if (own >= 0 && w.gx_spin_ms >= 0) own = own > w.gx_spin_ms ? own - w.gx_spin_ms : 0;
    v.load[static_cast<int>(Stage::kMain)] = share(own, win);
    v.load[static_cast<int>(Stage::kGx)] = share(w.gx_spin_ms, win);
    v.load[static_cast<int>(Stage::kRender)] = share(w.render_cpu_ms, win);
    v.load[static_cast<int>(Stage::kCp)] = share(w.cp_cpu_ms, win);
    v.load[static_cast<int>(Stage::kRecord)] = share(w.rec_busy_ms, win);
    v.load[static_cast<int>(Stage::kGpu)] = share(w.gpu_busy_ms, win);
    v.load[static_cast<int>(Stage::kPresent)] = share(w.present_cpu_ms, win);
    v.main_loop = w.limiter_ms < 0 ? -1 : (win - w.limiter_ms) / win;
    if (v.main_loop > 1) v.main_loop = 1;
    if (w.limiter_ms >= 0 && v.main_loop < 0) v.main_loop = 0;
    v.backlog = w.backlog_ms / win;
    const auto load = [&](Stage s) { return v.load[static_cast<int>(s)]; };
    const auto main_loop_stage = [](Stage s) { return s == Stage::kMain || s == Stage::kGx; };
    // The main loop as one stage: its busy share (its own work, the barrier's
    // spin and its waits together), named for the larger lever - the GX
    // workers when the spin is a third of its time or more than its own work.
    const double busy_ms = w.limiter_ms >= 0 ? win - w.limiter_ms : win;
    const bool gx_leads = w.gx_spin_ms >= 0 && own >= 0 && (w.gx_spin_ms >= own || w.gx_spin_ms * 3 >= busy_ms);
    const auto decide = [&](Stage s, const char* why) {
        v.limit = s;
        v.why = why;
        v.limit_load = main_loop_stage(s) && v.main_loop >= 0 ? v.main_loop : s >= Stage::kMain ? load(s) : -1;
        // What limits next: the busiest stage but the limit. Under the cap
        // (and in stalls) the main loop counts by its busy share: its waits
        // are on its job threads then. Behind a stage after it, by its parts:
        // its busy share holds the back-pressure.
        const bool by_share = (s == Stage::kCapped || s == Stage::kStall || s == Stage::kNone) && v.main_loop >= 0;
        for (int k = static_cast<int>(Stage::kMain); k < kStages; ++k) {
            const Stage t = static_cast<Stage>(k);
            if (t == s || (main_loop_stage(s) && main_loop_stage(t))) continue;
            double l = v.load[k];
            if (main_loop_stage(t) && by_share) {
                if (t != (gx_leads ? Stage::kGx : Stage::kMain)) continue;
                l = v.main_loop;
            }
            if (l <= v.next_load) continue;
            v.next_load = l;
            v.next = t;
        }
        return v;
    };

    // 1. Stalls.
    if (w.flips < 2.0 * w.secs) return decide(Stage::kStall, w.loading_ms * 2 >= win ? "loading screen" : "no frames");
    if (w.loading_ms * 2 >= win) return decide(Stage::kStall, "loading screen");
    if (w.long_ms >= 0.2 * win) {
        const char* why = "long frames";
        double most = 0.25 * w.long_ms;  // a cause names the stall only when it holds a quarter of it
        if (w.main_file_ms > most) {
            most = w.main_file_ms;
            why = "long frames: file I/O";
        }
        if (w.pipeline_ms > most) {
            most = w.pipeline_ms;
            why = "long frames: pipeline builds";
        }
        if (w.import_ms > most) why = "long frames: imports";
        return decide(Stage::kStall, why);
    }
    // 2. The limiter holds the rate. The rate has to be at the cap: under it
    //    the limiter still sleeps after the frames that came in short (the
    //    Deck at Mensis: 58.2 at 60, GPU-bound, the limiter 6% of the time).
    const double lim = w.limiter_ms < 0 ? -1 : w.limiter_ms / win;
    const bool at_cap = w.cap_fps > 0 && v.fps >= 0.985 * w.cap_fps;
    if (lim >= 0.02 && at_cap) return decide(Stage::kCapped, "at the frame cap");
    // 3. The GPU.
    if (load(Stage::kGpu) >= 0.90) return decide(Stage::kGpu, "the GPU is busy");
    // 4. Back-pressure: the game waits for the command processor. A
    //    twentieth of the second: under it the command processor only caught
    //    up now and then (the Cainhurst courtyard uncapped: 0-4% in the
    //    seconds the main loop held, 6-19% in those the command processor did).
    if (v.backlog >= 0.05) {
        const double cp = load(Stage::kCp), rec = load(Stage::kRecord), gpu = load(Stage::kGpu);
        if (rec >= 0.85 && rec >= cp) return decide(Stage::kRecord, "the game waits for the command processor, the recorder busiest");
        if (gpu >= 0.80 && w.cp_gpu_wait_ms >= 0.10 * win && gpu > cp)
            return decide(Stage::kGpu, "the game waits for the command processor, which waits for the GPU");
        return decide(Stage::kCp, "the game waits for the command processor");
    }
    // 5. The render thread.
    if (load(Stage::kRender) >= 0.85) return decide(Stage::kRender, "the render thread is busy");
    // 6. The main loop. The GX workers when the barrier's spin is a third of
    //    the main loop's time or more than its own work: the spin is a pure
    //    wait (the Cainhurst courtyard: 4.7 ms of an 11.4 ms frame, the own
    //    work 5.0; Central Yharnam 2.3 of 10.3).
    if (v.main_loop < 0 || v.main_loop >= 0.90) {
        if (gx_leads) return decide(Stage::kGx, "the main loop waits for the GX workers");
        return decide(Stage::kMain, w.gx_spin_ms < 0 ? "the main loop is busy (the GX barrier not measured)" : "the main loop is busy");
    }
    // 7. The presenter.
    if (w.displayed >= 0 && w.displayed < 0.9 * w.flips && load(Stage::kPresent) >= 0.80)
        return decide(Stage::kPresent, "the display shows fewer flips than the game makes");
    // 8. Under the cap with the limiter waiting a tenth of the time: frames
    //    of uneven length, the short ones waited out.
    if (lim >= 0.10) return decide(Stage::kCapped, "the limiter waits, the frames uneven");
    return decide(Stage::kNone, "no stage is near full");
}

}  // namespace bottleneck
