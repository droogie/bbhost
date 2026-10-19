# The full tour

`tools/full_tour.py` plays the whole game the way a test harness can: it enters
every map, puts the player down at stops spread over each map's walkable
ground, turns the camera through a full circle at each stop, plays every
cutscene, and records what the renderer had to do for the first time at each
place. It is two things at once:

- **a pipeline list.** Every pipeline the tour's draws create is recorded in
  the pipeline manifest (`<data>/bbhost/pipeline-manifest.bin`, see
  [rendering.md](rendering.md)). Merged over the whole tour and turned into a
  list with `tools/pipeline_list.py`, it is what a first start can compile
  before the player gets anywhere;
- **a full-game regression check.** Crashes, hangs, missing files, draw
  failures, flip stalls and load times, per map and per stop, where the twelve
  places of `tools/area_check.py` see only twelve views.

It is a test harness: it moves the player (the map-validation probe,
`tools/mapval_plugin.c`, with `BBHOST_TEST_WARP=1`), turns on the debug menu's
Player No Dead and Player Hide, and plays a copy of a test save. Never give it
a player's data folder.

## What it visits

`full_tour.py inventory --app0 <game folder>` lists it from the game's files:

- **the 16 map blocks** of the world list, each reached by a lamp travel the
  way the Hunter's Dream headstones do it (`ReturnPointParam`);
- **each block's light states:** `GameMapParam` gives every block a light
  setting per world-time slot (evening, night, late night, Blood Moon). The
  world's time is event flags 9800-9802; the tour sets them before a travel
  to load the other states (Central Yharnam has four);
- **the Hunter's Dream's states:** its event script switches objects and fire
  by the time and two more flags (9462, 12101852);
- **the ten fixed chalice dungeons** (`m29_XX_90_00`), entered the way the
  altars do it: an altar slot holds the dungeon's map id, then the travel to
  the dungeon lamp;
- **every cutscene:** the 42 remo files, played by id the way the event
  scripts' 2002 instructions ask for one, in their own map, skipped with
  Options after a while;
- **the bosses**, through `plugins/boss_rush` (a chunk of its own: the
  rounds run by themselves, each boss woken by a warp next to it and fought
  for a while with Player No Dead before the plugin's test kill);
- **the title's attract movie**, which plays when the title sits idle (the
  chunk holding it waits there before pressing anything). The other movies
  (the opening, the staff rolls) belong to a new character and the endings;
  `RequestPlayMovie` played none of them in the world.

Stops come from each block's navmesh: the face centroids (placed by the
block's collision parts), without the faces nothing in the layout stands near,
picked farthest first from a seeded start until every face is within a radius
of a stop (60 m by default; height counts twice). Chalice layouts have no
navmesh here: their stops are the layout's enemies and player starts.

## Running it

```sh
# the plan: items, stops, chunks (offline)
python3 tools/full_tour.py plan --app0 <game folder> --navmesh-tools <dir with the worldmap module> --out PLAN
# one chunk (a game.lock hold of at most ~20 minutes); the caches chain from chunk to chunk
python3 tools/full_tour.py run --plan PLAN --chunk 0 --bbhost build/bbhost --config test.toml \
    --seed <test save folder> --runs RUNS --live /dev/shm/tour [--glcache DIR]
# what was found, per item and stop
python3 tools/full_tour.py report --plan PLAN --runs RUNS
# the list from every chunk's manifest, in the order a player reaches things
python3 tools/full_tour.py list --runs RUNS [RUNS2 ...] --app0 <game folder> --eboot <eboot> --out pipeline-list.bin
```

- `--navmesh-tools` (default `$MAPVAL_NAVMESH_TOOLS`) is a folder holding a
  `worldmap` package with `navmesh.py` (`faces(data)`: the polygons of a Havok
  2014 navmesh packfile) and `msb.py` (`MSB`, `part_matrix`), the same module
  `tools/mapval_blocks.py` takes. It is not part of bbhost, and only `plan`
  imports it (with numpy), when it runs: without it `plan` stops and names
  what is missing. `run`, `report`, `compare`, `refall` and `list` need
  neither, and no package carries them.
- The config gives only `[paths]` (app0, eboot); each chunk gets its own
  offline config, data folder and `BBHOST_CONFIG_DIR`.
- `--live` should be a RAM disk: the chunk's data, its caches and its frame
  dumps live there while it runs (a network file system stalls the game on
  large writes).
- Each chunk starts from the save as it is, with the caches of the chunk
  before it and, as its pipeline manifest, the union of every chunk's so far
  (`RUNS/union-manifest.bin`). bbhost drops recipes no start has used for 16
  starts; the union keeps them.
- The probe ends the run the way `BBHOST_EXIT_SECONDS` does, so the exit
  reports are written and the caches saved. A chunk that did not end that way
  leaves `failed.json` and does not move the chain on.

Each chunk's folder keeps `run.log`, the tour it ran, its manifest, a
1280x720 JPEG of every stop and `done.json`. `--cpus 0-3,40-43` runs a chunk
on those CPUs only (with `--env BBHOST_PRECOMPILE_BOOT_THREADS=4 --env
BBHOST_PRECOMPILE_THREADS=2`, a Steam Deck's start, CPU-wise).

## The list

`list` merges the chunks' manifests (the last run folder's union), leaves
out records that are the same inputs under a second key (a start re-keys a
recipe whose build now makes another key), names every program by hash and
orders the records by where a player meets each pipeline first: the title
and the start, Central Yharnam and the Hunter's Dream, then the story's maps
with the world's time moving where the story moves it, The Old Hunters, the
chalice dungeons by depth. bbhost builds a list in that order, behind the
player's own manifest. The records keep the fetch shaders' words where they
need them (the engine's, made at run time; `--no-fetch-words` leaves those
records out), so tessellated pipelines are in it too, with the host
tessellator's LS passes: a tessellated recipe records the LS compute
pipeline of every pass its draws ran (recipe file version 3), and the start
builds those as well. Compute pipelines are recorded too (version 4): each
one a dispatch used, with the images and samplers its key was made from, so
the start builds them first - the title's own compute shaders among them,
which the game never creates through the hook bbhost compiles at, so nothing
else builds them before their first dispatch. The GPU particles'
geometry-shader pipelines are recorded too (version 5): the record carries
the export shader and the GS's copy shader beside the GS, and a list names
the copy shader by its GS's container, which holds it after its first footer.

## What the report counts

Per item (a map, a light state, a dungeon), per stop and per cutscene:

- **new pipelines**: recipes the manifest did not have (`BBHOST_LOG_NEW_RECIPES=1`);
  per item, by stop, which shows whether the stops converge;
- the stages the command processor translated itself
  (`BBHOST_LOG_STAGE_TRANSLATIONS=1`) and its pipeline builds over 5 ms
  (`BBHOST_LOG_PIPELINE_BUILDS=5`);
- flip stalls over 40 ms, the travel's time and its loading screen;
- draw failures by reason (the per-second frame detail), the exit report's
  fallbacks;
- crashes, timed-out waits, HLE stubs, deaths, files the game could not open,
  the layouts each load opened (`BBHOST_FS_TRACE=.msb`).

The first chunk visits the twelve places of `tools/area_check.py`; what the
tour records after them is what those places miss.

## As a regression check

Run one plan with each build, each from the same start (both `--cold`, or
both on a fresh `--runs` and `--live` from the same seed), and put the two
side by side:

```sh
python3 tools/full_tour.py compare --plan PLAN --a RUNS_A --b RUNS_B
```

Per item: the command processor's pipeline builds over 5 ms and their time,
its translations, new recipes, flip stalls and first-seen pixel shaders; then
the tessellated draws apart - new tessellated recipes, the host tessellator's
LS passes the command processor built at first sight, and the compute
pipelines' share of the stalls' pipeline time (the LS passes are in it), with
the exit report's count of LS passes built from recipes at the start. The
two reports' crashes, hangs, deaths, draw failures and missing files are the
rest of the check. For a first start with and without a list, `--list` on one
arm and `--cold --no-chain` on both.

## The probe's steps

`tools/mapval_plugin.c` runs a tour file, one step a line. Besides the
travel, warp, mark, hold, camera and dump steps `area_check.py` uses, the full
tour uses:

| step | what it does |
|---|---|
| `god [hide]` | Player No Dead (and Player Hide: the enemies do not notice the player) |
| `turn <n> <s>` | the follow camera through 360 degrees in n headings, s seconds each |
| `flags <id>=<0\|1> ...` | event flags set now (the world's time before a travel) |
| `flaglog <first> <n>` | n flags from first logged as one number |
| `chalice <map id> <slot> <name>` | a fixed chalice dungeon entered through that altar slot |
| `remo <id> <mode> <skip s> <max s>` | a cutscene by id, Options pressed after skip s |
| `movie <id> <skip s> <max s>` | a movie by id (`RequestPlayMovie`; ids 0-5 play nothing in the world) |
| `exit` | the run ends with its exit reports and cache saves |

The host side has two test-only switches for it: `BBHOST_LOG_NEW_RECIPES=1`
and the `exit` request of `BBHOST_TEST_REQUESTS`.

## Shipping the list

bbhost reads a list from `pipeline-list.bin` beside its executable (or the
file `BBHOST_PIPELINE_LIST` names) and queues every pipeline in it that the
player's own manifest lacks, behind the player's own, at each start. The list
names the game's programs by their footer hash and holds none of their code;
the words come from the player's own game files at the start, and a record
whose hash does not match is dropped.

The list is published as an asset of its own release and pinned by tag
and SHA-256 in `tools/fetch_pipeline_list.sh`, as `tools/fetch_dxc.sh` pins
DXC. The Linux and Windows packages (and so the Steam Deck kit) fetch it and
put it beside the binary; `tools/check_release_bundle.sh` fails a release
whose bundles lack the pinned list. The pin starts empty - no list
published, none packed, none asked for - and a list's release turns it on.

```sh
tools/fetch_pipeline_list.sh DIR                              # into DIR, checked against the pin
BBHOST_PACKAGE_PIPELINE_LIST=new.bin tools/package_linux.sh   # a local list instead (a new one under test)
BBHOST_PIPELINE_LIST_FETCH=0 tools/package_linux.sh           # none (a development kit)
```

A new list is a new release under a tag that is not `v*` (the release
workflow leaves those alone), its SHA-256 in the fetch script, and the
first-start check run again. The record format is one thing that makes a
list old (`recipe_file::kVersion` in render.cpp, `VERSION` in
`tools/pipeline_list.py`, `LIST_FORMAT` in the fetch script; a test checks
they agree; bbhost reads the format it writes and the one before). The other
is the draw path: a record holds what its draw was made from - the programs
by hash, the draw state - but also the vertex-input plan the recording build
made for it, and a start builds the pipeline from that plan. A build whose
draws plan their vertex input another way (packing an element's components,
a plan hash over a grown struct) looks every such pipeline up under another
key unless its start plans the recorded input again the way its draws do. So
the first-start check runs on the build that ships the list, and a list
whose records it does not take (its new recipes with the list as many as
without) is made again on that build.
