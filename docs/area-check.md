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
| `nightmare-of-mensis` | Nightmare of Mensis, Mergo's Loft and the brain's window | the Nightmare of Mensis lamp, then the rocks above the north path |
| `fishing-hamlet` | Fishing Hamlet, the Orphan of Kos's coast | the first lamp, then the arena's edge |
| `astral-clocktower` | Lady Maria's arena | the Research Hall lamp, then the arena's door |
| `lumenwood-garden` | The Living Failures' garden | the Research Hall lamp, then the balcony's door |
| `hunters-nightmare` | Ludwig's arena | the first lamp, then the arena's edge |
| `laurence-staircase` | The staircase up to the Nightmare Grand Cathedral, Laurence's arena | the first lamp, then the staircase's upper flight |

The last five are The Old Hunters' and need it on (bbhost's default). An
arena view stands just outside the boss's fog wall, facing in, where the boss
rush plugin starts its rounds, without walking in. Two arenas are behind
doors, which the run opens with the flags the boss rush uses.

Mensis and the staircase are views rather than arenas, two of the heaviest
scenes the game draws, and each has a standpoint of its own: a position in the
map's layout and a facing. Mergo's Loft is seen from the top of a rock slope
off the path, where the brain's gaze frenzies the player; the probe keeps the
HP full. The staircase is seen from its upper flight, facing up, above the
landing where an executioner wakes when the player comes down.

The places are visited in the order given, and what a place changes (a door)
stays changed for the rest of the run.

## What a run does

A run copies a data folder (`--seed`: its saves hold a character in the world,
which Continue loads), plays it headless and offline, pinned to one NUMA node
with `numactl`, with the map-validation probe plugin (`tools/mapval_plugin.c`)
moving the player. It gets through the title by asking the game for the menus'
confirm button every 2 s until the world's first frame (the test request
`menutap confirm`, which bbhost carries out only while one of the game's menus
has the input), so it reaches the title whenever the title comes: a build's
first start compiles its shader caches before it. Each press is held 1.5 s
until the game has read one (`BBHOST_TRACE_PAD`), 200 ms after: a pad the game
finds disconnected it reads only once a second, and a short press then met one
of those reads only by chance - one to three minutes of ignored presses on a
Steam Deck's cold start, before bbhost plugged a harness's pad in from the
start. The report says how many presses it took and at which flip the game
read the first.

At each place the probe travels to a lamp the way the
Hunter's Dream headstones do - done once the player stands still in the
place's map with the loading screen gone -, moves to the place's view,
switches off the ordinary enemies within 40 m, puts the camera behind the
player and holds still:

1. a settle (15 s): streaming and the first compiles;
2. the window (20 s): what the place costs;
3. two frames of the display, 2 s apart.

That is the perf pass. A second session, the capture pass, visits the same
places and captures the first draw of up to six pixel shaders at each that
earlier places did not capture; `tools/lift_verify.py` replays each with the
lifted pixel shader and with the translated one and compares the outputs byte
for byte. Captures stall the GPU, which is why they get a session of their own.

A place counts as measured only where the player was: each of the probe's
marks (the arrival, the window's start and end) must be in the place's map, a
view's within 3 m of its standpoint, and from the arrival on the player must
not die, be placed in a map again (a respawn, or a bounce out of an area the
game would not keep the player in), meet a loading screen in the window or
move more than 3 m across (5 m up or down). A place that fails one of these
says why in its status - `not there: m21_00_00_00 at the measure mark`,
`died in the window`, `moved 40 m in the window` - and is left out of an
A/B's numbers. The settle a place had is counted from the loading screen's
end.

For each place the report says:

- in the window: fps and the frame-time percentiles from every flip interval,
  the GPU's busy time (bbhost's busy meter, not `BBHOST_GPU_PROFILE`, which
  changes how draws are recorded), the main loop's work, the CPU, the draws by
  how their pixel shader ran - lifted, translated, the page-table fallback's
  translation, none - and the draw failures by reason, and the compute
  dispatches by how their shader ran, lifted or translated;
- on arrival (the travel and the settle): the loads, pipeline compiles over
  30 ms, flip stalls over 40 ms, main-loop frames over 33 ms, and the pixel
  shaders met for the first time there, lifted or not and why;
- the frames and their brightness, over the picture: a dump is the whole
  display buffer (5120x2160 on Linux), and the game renders into its top-left
  corner at the run's render size. A band edge - a step in brightness between
  two neighbouring columns or rows that runs all the way across the picture,
  as the 4K world that was black left of three quarters of the width had - is
  flagged, and so is a black picture;
- memory as the place was left, from the 5-second report: the device-local
  heaps' use, the image heap and everything bbhost allocated, and what the
  place added since the place before;
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
pass should take: about 15 minutes each for all twelve places, then the lift
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
`ab.txt` lists what is worse beyond the noise floor, then what is better,
memory at each place included. A
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
| `BBHOST_TEST_REQUESTS=<dir>` | the game carries out requests a harness leaves there: a frame dump, a sequence of consecutive frames (`dumpseq <dir> <n>`: the next n flips, one file each), a capture budget |
| `BBHOST_CAPTURE_DRAW=*ps` | a draw capture of the first draw of each pixel shader |
| `BBHOST_CAPTURE_ARMED=1` | captures only while a request has allowed them |

Switches given with `--env` that write files relative to the game's folder -
`BBHOST_DUMP_FRAME` (`build/frame-N.ppm`), `BBHOST_DUMP_AT_DRAW`
(`build/rtd-*.ppm`) - write into the pass's own `build/`, which the run makes;
the report lists what landed there.

The flip log's `pixel shaders:` line, every 300 flips in any run, gives the
same split of draws by pixel shader.
