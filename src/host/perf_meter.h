#pragma once

// The bottleneck meter: once a second, which stage holds the frame rate - the
// GPU, the main loop (its own work, or the GX workers it waits for), the
// render thread, the command processor, the recorder, the presenter - or
// whether the frame cap holds it or long frames (loads, pipeline builds,
// imports) do (host/bottleneck.h decides; docs/performance.md reads it).
//
// It runs on a thread of its own (bb-meter) from counters the stages keep
// anyway: the flips' intervals, the limiter's wait, the main thread's waits
// (frame_stats.h), the threads' CPU time, the GPU busy meter, the recorder's
// replay time and the waits for the command processor's backlog. The one
// thing it adds is the share of the main thread's time spent spinning in the
// GX job barrier (sub_25b8710, a guest loop that waits for the GX workers):
// on Linux a perf_events task clock on the main thread samples where it is
// every 2 ms of its CPU time (nothing is delivered to the thread); on
// Windows the meter suspends it 25 times a second (100 with
// BBHOST_FRAME_STATS=1; BBHOST_PERF_METER_HZ sets it) to read its
// instruction pointer, ~40 us each. The 1.09 eboot only (the loop's
// addresses).
//
// What it logs:
//   `perf summary:` about once a minute and when the player's map changes,
//   always (with tooling on): the window's verdicts by seconds, the stages'
//   loads and the main loop's frame;
//   `perf:` each second, with BBHOST_FRAME_STATS=1: the second's verdict;
//   a run summary at exit.
// BBHOST_PERF_METER=0 turns it off; =1 turns it on with tooling off.
// BBHOST_PERF_SUMMARY_S sets the summary's period (60).

#include <cstdint>

// main(), once the configuration is final: the guest image's range and
// whether it is the 1.09 eboot, whose GX barrier's addresses the meter knows.
void perf_meter_start(std::uint64_t guest_lo, std::uint64_t guest_hi, bool eboot_109);
// On the game's main loop thread, once (main.cpp's guest_main_thread).
void perf_meter_note_main_thread();
// Each flip the game submits (frame_stats_on_flip).
void perf_meter_flip();
// The frame limiter, each frame (engine/frame_rate.cpp): how long it waited,
// how much of that it spun on a core, and the rate it held the frame to (0:
// none).
void perf_meter_limiter(std::uint64_t wait_ns, std::uint64_t spin_ns, float target_fps);
// Any thread held back by the command processor: waiting for its backlog to
// drain, or for GPU memory to retire (hle/gnm_exec.cpp).
void perf_meter_backlog_wait(std::uint64_t ns);
// The exit reports: the last summary and the run's.
void perf_meter_exit();
