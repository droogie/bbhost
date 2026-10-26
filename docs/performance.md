# What limits the frame rate: the bottleneck meter

bbhost runs a frame through a chain of stages, and the frame rate is the
slowest stage's. The bottleneck meter names that stage once a second, with the
numbers behind it, so a log says whether a machine is held by its GPU or its
CPU, and on the CPU by which thread.

## The stages

| Name in the log | What it is |
|---|---|
| `main` | The game's main loop on its main thread: the game's own code, ours on that thread, and its waits on the game's job threads (physics, characters). |
| `gx` | The GX workers, the threads that record the frame's draws while the main loop waits for them in the GX job barrier. |
| `render` | The game's render thread (`GXRenderThread`), which turns the frame's draw lists into command buffers. |
| `cp` | The command processor (`bb-cp0`), which walks those command buffers and makes the draws. |
| `record` | The recorder (`bb-record`), which replays the draws into Vulkan. |
| `gpu` | The GPU. |
| `present` | The presenter (`bb-present`), which puts the frame on the screen. |

Three verdicts are not stages:

- `capped`: the frame cap (or V-Sync's refresh rate) holds the rate, and the
  main loop sleeps in the limiter. The next stage tells the headroom.
- `stall`: long frames: a loading screen, or frames of more than twice the
  usual length taking a fifth of the second (file reads, pipeline builds,
  memory imports, named when one of them holds the time).
- `none`: no stage is near full.

The `gx` verdict matters most on a fast GPU. The GX job barrier waits for the
workers in a loop that never sleeps, so to a CPU counter, or Task Manager, the
GX workers' slowness looks like a busy main thread. The meter samples where the
main thread is (every 2 ms of its CPU time with Linux's perf events, 25 times
a second on Windows) and counts the time in that loop as the GX workers'.

## Reading it

With **Tooling and logging** on (the default), every log carries a line about
once a minute, and one each time the player's map changes:

    perf summary: 41 s at m23_00: 108.0 fps (1% low 60.8); limit cp 95%, next main 56% (the game waits for the command processor);
    seconds by limit: cp 38 s, stall 3 s; loads main 56% gx 21% render 42% cp 95% record 24% gpu 46% present -;
    main loop a frame 9.26 ms: own 5.17 gx 1.96 waits 2.20 file 0.00 limiter 0.00 ms; held back by the command processor 49.7%;
    long frames 151 ms; the meter 0.10% of a core

(Old Yharnam on an RTX 4070 at 1080p, uncapped: the command processor holds
it, and the game waited for it half the time.)

- `limit` is the window's verdict, with how busy that stage was, and `next`
  is the stage that would limit after it.
- `seconds by limit` counts each second's own verdict: a place where two
  stages are close splits its seconds between them.
- `loads` is each stage's busy share of the time: a thread's time on a core
  (`main` is the main thread's own work, `gx` its time in the barrier), the
  recorder's time replaying, the GPU's busy time. `-` is a stage the meter
  could not see (no presenter without a window).
- `main loop a frame` splits the main loop's frame: its own work, the GX
  barrier, its waits on other threads and file reads, and the limiter's sleep.
- `held back by the command processor` is the share of the time the game
  waited for the command processor to catch up: above 5% the command
  processor, or a stage after it, is the limit.

At exit, `perf run:` gives the whole session's seconds by verdict.

On screen, F10's **Performance meter** (off by default) shows the last
second's verdict under the FPS counter, in two lines:

    Limit: GPU 99% | next: command processor 42%
    main 8.1 ms  GX 0.0  waits 27.8 | CP 42%  GPU 99%

(a Radeon 680M at 1080p). The first line is the verdict and what would limit
next; the second is the main loop's frame (its own work, the GX barrier, its
waits) and the command processor's and the GPU's loads. With **Tooling and
logging** off it starts the meter, and the GPU's load shows when the
Performance meter was already on as the game started (`GPU -` otherwise: the
GPU busy meter starts with the game).

`BBHOST_FRAME_STATS=1` adds a `perf:` line every second, with the second's
verdict and the same fields (the test harnesses turn it on).
`BBHOST_PERF_METER=0` turns the meter off; `=1` turns it on with Tooling off.
`BBHOST_PERF_SUMMARY_S` sets the summary's period (60 s).

`tools/perf_verdict.py` reads a log, a tester's or a harness's, and prints the
verdicts by place - a harness's measure windows, or the maps the log names -
and over the run:

    tools/perf_verdict.py logs/bbhost-2026-10-26.log
    tools/perf_verdict.py build/area-check/perf/run.log --json

It also reads logs from before the meter, through their `frames:` lines
(`BBHOST_FRAME_STATS=1`), as far as those lines go: without the GX barrier
and without the waits for the command processor, so `gx` never shows and `cp`
needs the command processor at 92% of a core.

## What to do with a verdict

- `gpu`: lower the resolution, Model detail or the effects in the options.
  `BBHOST_GPU_PROFILE=3` times the frame's passes on the GPU (it records
  without the recorder thread, so judge its pass times, not its frame rate).
- `main`, `gx`, `render`, `cp`, `record`: the CPU. A faster CPU, or one
  with more cores free (close what else runs), helps; the frame cap or V-Sync
  smooths what the CPU cannot hold.
- `capped`: the machine has room above the cap: a higher frame rate holds.
- `stall`: the first visit to a place builds its shaders and reads its files;
  a second visit is smoother. **Compile shaders now** in the setup window builds
  them ahead.

## The meter's cost

The meter is one thread that wakes ten times a second on Linux and a hundred
on Windows, and reads counters the stages keep anyway: 0.1-0.25% of a core.
On Linux the kernel takes the main thread's samples, with nothing delivered
to the thread (a busy loop under the same sampling ran within the
measurement's noise, 0.5%). On Windows each sample stops the main thread for
about 40 microseconds: 25 samples a second (about 1 ms a second), 100 with
`BBHOST_FRAME_STATS=1`; `BBHOST_PERF_METER_HZ` sets the rate.

The GX barrier's share needs the 1.09 eboot (the loop's addresses), and on
Linux `kernel.perf_event_paranoid` at 2 or lower (the default on most
distributions; `sysctl kernel.perf_event_paranoid`). Without it the log says
so once, and `main` then includes the barrier.
