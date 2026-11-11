#pragma once

// A wall-clock sampling profiler for host threads, for finding where the
// command processor's frame time goes when a good part of it is not CPU.
//
// BBHOST_SAMPLE=bb-cp0,bb-present   threads to sample, by name prefix
// BBHOST_SAMPLE_HZ=997              samples a second per thread
// BBHOST_SAMPLE_OUT=build/samples   writes <out>.txt (one stack a line:
//                                   flip, tid, return addresses) and
//                                   <out>.maps (/proc/self/maps)
//
// Each tick signals the thread (SIGPROF) whether it is running or blocked,
// and the handler unwinds its stack, so a futex or a driver wait shows up
// as the frames that called it. tools/wallprof.py symbolizes and sums the
// stacks.
//
// Guest threads can be sampled too, by their guest names, or "main" for the
// game's main loop (every unnamed guest thread is "bbhost"): the handler moves to the thread's host FS before it
// touches anything, and puts the guest's back. Interrupted in guest code it
// walks the guest's frame pointers (the eboot keeps them) and marks those
// addresses with a leading 'g' - they are Binary Ninja addresses; in host
// code (an HLE call) it unwinds the host frames.

#include <cstdint>

// [guest_lo, guest_hi): the eboot's image.
void sampler_start(std::uint64_t guest_lo, std::uint64_t guest_hi);

// Called on the thread the game's main loop runs on, before it enters the
// guest: BBHOST_SAMPLE's "main".
void sampler_note_main_thread();

// From the SIGSEGV handler, on host FS: a fault inside the sampler's unwind
// on this thread abandons that sample and does not return. Otherwise a no-op.
void sampler_recover();

// When no flip comes for BBHOST_WATCHDOG_S seconds (15; 0 off), every
// thread's stack at the stall's start and 30 s on (named, host frames as
// module+offset, guest ones as Binary Ninja addresses; tools/hang_stacks.py
// names ours). For runs on machines the log is all we get from: one that
// stops without a crash report then says whether it hung and where.
// Windows: from the start, with a line every 5 s, the threads suspended to
// read them. Linux: once the game has flipped 60 times, each thread's stack
// taken by its own SIGPROF handler (a signal can cut a blocking call short,
// so not during the start).
void sampler_watchdog_start(std::uint64_t guest_lo, std::uint64_t guest_hi);

// The window's thread pumped its messages (main.cpp, every 4 ms). Windows
// calls a window whose thread has not looked at its messages for 5 s "Not
// Responding"; with none for 4 s the Windows watchdog above logs every
// thread's stack (once a stall, within its six dumps), flips or not.
void sampler_note_pump();
// The run is ending and is at `step` now (main.cpp exit_step: each of the
// exit's reports, its waits). While exiting, a window not pumped for 4 s gets
// one line - which step, for how long - and every thread's stack only once a
// step has stood still for 30 s: a slow report is not a hang, a hung exit
// still says where.
void sampler_note_exit(const char* step);
