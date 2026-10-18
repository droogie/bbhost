# Where the CPU goes: the ETW harness (Windows)

Kyo's KyoPS4x method, carried over to bbhost: a Windows Performance Recorder
trace of the machine while the game runs (a sample a millisecond on every core,
with stacks, and every context switch with its wait reason), reduced to one
process by thread, module and function. It answers "which thread limits the
frame rate, and what is it doing" where the frame statistics only say how busy
each thread is.

## Running it

With the perf pass (tools/perf_pass_win.sh), a trace of PERF_ETW seconds from
PERF_ETW_AT (70) seconds into the run:

    PERF_FULLSCREEN=1 PERF_ETW=30 PERF_ETW_AT=80 tools/perf_pass_win.sh NAME

`tools/etw/record.ps1` starts WPR, which needs an administrator: one UAC prompt
a trace (what the elevated half did is in `NAME.etl.record.log`). Afterwards
`NAME.etw.txt` holds the report. By hand:

    powershell -File tools\etw\record.ps1 -Seconds 30 -Out build\perf\t.etl [-Delay 60] [-Profiles CPU,GPU]
    python tools/etw/profile.py build/perf/t.etl build/perf/t.pickle [--process bbhost.exe]
    python tools/etw/report.py build/perf/t.pickle --exe build-win/bbhost.exe --eboot <eboot> --log <run log>
    python tools/etw/callees.py build/perf/t.pickle bb-cp0 draw_impl --exe build-win/bbhost.exe
    python tools/etw/thread_cpu.py bbhost.exe 10       (no trace, no administrator: a quick look)

The report names bbhost's functions from the build with symbols
(`build-win/bbhost.exe`; the package is stripped), the game's code as
`eboot!sub_<address>` (Binary Ninja's addresses: the run's "guest slide" line
places the image, a prologue scan of the eboot finds the functions), and system
DLL frames by their exports. A game thread's stack walk ends at its first game
frame: game code has no unwind information.

## What it found (2026-10-12, Radeon 8060S, Kyo's area, uncapped)

~112 fps with the GPU 67% busy - neither the GPU nor any thread saturated. The
command processor (bb-cp0) ran 71% of a core but was switched out 23,400 times
a second: 25% of its time was the contended slow path of the renderer's mutex
(a winpthreads `std::mutex`, a kernel wait and wake each time) and most of its
31% kernel time those waits. The other side was bb-flip-wait, which polled
`host_gpu_submissions_completed()` every 500 us - taking that mutex and asking
up to 16 fences in AMD's driver under it - and so also completed every flip up
to a poll late. Fixed: the mutex is a `HostLock` (core/futex.h), the game's
submissions signal a timeline semaphore that is read without the lock, and
bb-flip-wait sleeps on it until its frame is done. Afterwards the same spot ran
134-145 fps with the GPU 97-98% busy.
