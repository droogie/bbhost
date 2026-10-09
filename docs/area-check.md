# The area check

`tools/area_check.py` plays a build through a fixed list of places in the
game and measures the same things in each, so that a rendering change can be
judged by what it does to the frames, the frame times and the shaders that
draw them, place by place. Two builds can be run against each other.

## The places

| Id | Place | Reached by |
|---|---|---|
| `hunters-dream` | Hunter's Dream | its lamp |
| `central-yharnam` | Central Yharnam | the Central Yharnam lamp |
| `cathedral-ward` | Cathedral Ward | its lamp |
| `old-yharnam` | Old Yharnam | its lamp |
| `forbidden-woods` | Forbidden Woods (tessellated meshes) | its lamp |
| `moonside-lake` | Byrgenwerth, Rom's arena | the first lamp, then the lake's doorway |
| `fishing-hamlet` | Fishing Hamlet, the Orphan of Kos's coast | the first lamp, then the arena's edge |
| `astral-clocktower` | Lady Maria's arena | the Research Hall lamp, then the arena's door |
| `lumenwood-garden` | The Living Failures' garden | the Research Hall lamp, then the balcony's door |
| `hunters-nightmare` | Ludwig's arena | the first lamp, then the arena's edge |

The last four are The Old Hunters' and need it on (bbhost's default). An
arena view stands just outside the boss's fog wall, facing in, where the boss
rush plugin starts its rounds, without walking in. Two arenas are behind
doors, which the run opens with the flags the boss rush uses. The
places are visited in the order given, and what a place changes (a door) stays
changed for the rest of the run.

## What a run does

A run copies a data folder (`--seed`: its saves hold a character in the world,
which Continue loads), plays it headless and offline, pinned to one NUMA node
with `numactl`, with the map-validation probe plugin (`tools/mapval_plugin.c`)
moving the player. At each place the probe travels to a lamp the way the
Hunter's Dream headstones do, moves to the place's view, switches off the
ordinary enemies within 40 m, puts the camera behind the player and holds
still:

1. a settle (15 s): streaming and the first compiles;
2. the window (20 s): what the place costs;
3. two frames of the display, 2 s apart.

That is the perf pass. A second session, the capture pass, visits the same
places and captures the first draw of up to six pixel shaders at each that
earlier places did not capture; `tools/lift_verify.py` replays each with the
lifted pixel shader and with the translated one and compares the outputs byte
for byte. Captures stall the GPU, which is why they get a session of their own.

For each place the report says:

- in the window: fps and the frame-time percentiles from every flip interval,
  the GPU's busy time (bbhost's busy meter, not `BBHOST_GPU_PROFILE`, which
  changes how draws are recorded), the main loop's work, the CPU, the draws by
  how their pixel shader ran - lifted, translated, the page-table fallback's
  translation, none - and the draw failures by reason;
- on arrival (the travel and the settle): the loads, pipeline compiles over
  30 ms, flip stalls over 40 ms, main-loop frames over 33 ms, and the pixel
  shaders met for the first time there, lifted or not and why;
- the frames and their brightness;
- the lift results of the place's captures.

`report.json` has everything and `summary.txt` the same to read.

## Running it

It needs a Linux build (`bbhost`, with `gcnlift` and `drawreplay` beside it),
the game files and the 1.09 eboot, named by a config's `[paths]` or by
`--app0` and `--eboot`, and a C compiler for the probe. Nothing else of the
config is used: each pass writes its own (offline, its own P2P port and its own
config folder), so a run never touches a player's settings or saves.

```sh
tools/area_check.py areas --app0 <game folder>       # the places, checked against the game's data
tools/area_check.py run --dry-run --bbhost build/bbhost --config test.toml --seed <save folder> --out <dir>
tools/area_check.py run --bbhost build/bbhost --config test.toml --seed <save folder> --out <dir>
tools/area_check.py ab --a <build A>/bbhost --b <build B>/bbhost --config test.toml --seed <save folder> --out <dir>
tools/area_check.py compare --a <run>... --b <run>... --out <dir>
tools/area_check.py report <run>                     # the report again from a run's logs
```

`--dry-run` checks the configuration, copies the seed and checks the copy,
checks each place's lamp and view against the game's data, builds the probe
and writes the plan, the tours, the configs and the commands, and starts
nothing. `--areas` picks places, `--settle`, `--measure` and `--captures`
change the timings, `--pipeline-cache off` compiles every shader when first
met instead of from the seed's stage manifest. The dry run says how long each
pass should take: about 12 minutes each for all ten places, then the lift
checks.

## Reading an A/B

`ab` runs the two builds alternately, two rounds each by default. The second
round is the noise floor: a difference in a number counts only when it is
larger than either build's own spread across its rounds, and a frame counts as
different only when it differs from the other build's more than from its own
repeat - lantern flicker, fog and cloth move up to a third of the pixels
between two identical runs. Frames are compared by the share of pixels that
changed, a blurred difference that leaves noise out, and an 8 x 8 table of
brightness changes; a picture of the difference goes in `diffs/`.
`ab.txt` lists what is worse beyond the noise floor, then what is better. A
build older than the check still gives fps, its seconds' worst frames, the GPU
busy meter and the hitches, but no frame intervals (its percentiles are marked
approximate), draws by shader kind, frames or captures; the comparison leaves
out what one side lacks.

## The switches it uses

These are for test runs and off by default.

| Switch | What it does |
|---|---|
| `BBHOST_FRAME_DETAIL=1` | with `BBHOST_FRAME_STATS=1`: after each second's `frames:` line, the second's draws by how their pixel shader ran, its draw failures and every flip interval |
| `BBHOST_TEST_WARP=1` | lets plugins move the player (the probe) |
| `BBHOST_TEST_REQUESTS=<dir>` | the game carries out requests a harness leaves there: a frame dump, a capture budget |
| `BBHOST_CAPTURE_DRAW=*ps` | a draw capture of the first draw of each pixel shader |
| `BBHOST_CAPTURE_ARMED=1` | captures only while a request has allowed them |

The flip log's `pixel shaders:` line, every 300 flips in any run, gives the
same split of draws by pixel shader.
