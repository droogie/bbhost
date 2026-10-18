# Decompilation

bbhost runs the game's own code, so nothing has to be decompiled for the game
to work. Decompilation is how parts of the engine become changeable: a game
function rewritten as C++ in `src/decomp/` runs in the original's place, and
can then be read, fixed and extended like any other source - which is where
room for mods and fixes comes from that byte patches and hooks cannot give.

The rule is that a rewritten function must behave exactly like the
original. It is checked twice: without the game, against the original's own
machine code on generated inputs, and in the game, against the original while
the game runs.

## Where it stands

The 1.09 executable has about 169,000 functions. In a default run today:

| | Functions |
|---|---|
| Rewritten as source, on the decomp list (`src/decomp/`) | 31 |
| Replaced by bbhost code outside the list (written before the list existed) | 4, and the YEBIS resource builder bypassed |
| Hooked at the entry or at call sites (GX methods, the resource registry, live resolution, settings, key prompts, menus) | 117 |
| **Taken over in all** | **152** |
| With any code byte changed (including byte patches and redirected calls) | 173; 255 at 60 fps |
| The game's own code | everything else |

The TLS rewrite also changes one instruction at each of 17,127 sites in about
8,100 functions, to make the game's thread-local accesses work on the host.

Functions are taken over when there is a reason to change them - a mod
surface, a fix, a cost - not for their own sake.

## The list, by area

`src/decomp/` has a folder for each area of the engine, named as the engine
map names them; `src/decomp/areas.cpp` adds every area's functions to the
list. Each area's test, in `tests/decomp/`, runs the game's own code beside
ours (below, "How one is checked").

### Events (`src/decomp/events/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| `IsEventFlag` | `0x17cfc00` | reads one event flag | no calls observed (reached only through a table, if at all) |
| `SetEventFlag` | `0x17cfcc0` | writes one event flag | 81,687 calls compared in a world session: 0 differences |
| `GetEventFlagValue` | `0x17cfd80` | reads a run of flags as a number | 83,467 calls compared: 0 differences |
| `SetEventFlagValue` | `0x17d0060` | writes a run of flags | 692 calls compared: 0 differences |
| `SprjEmkEventIns::Update` | `0x16ecdc0` | the event-script interpreter's loop: an event's conditions updated, then its instructions run while its main condition group holds | 20,237 ticks of 1,500 random scripts against the game's own code: 0 differences; in a world session 202 events ended and 165 restarted with ours in place |
| `emevd_dispatch_instruction` | `0x1bb93a0` | an instruction to its bank's function | every bank from -2 to 2100 and the extremes, 14,777 instructions: 0 differences; 14,052 dispatched in a world session, none the game has no function for |
| `SprjEmkEventIns::EndOrRestart` | `0x16ed100` | an event's end (and its restart record) or restart, and its completion flag | the loop's test; 367 calls checked in a world session: 0 differences |
| `SprjEmkConditionHolder::Update` | `0x16ec380` | every condition group's update, then the latch of satisfied groups into the event's state word | 3,000 random holders; 8,357,660 calls compared in a world session: 0 differences |
| `SprjEmkConditionHolder::LatchGroupBits` | `0x16ec4a0` | the latch alone | 3,000 random holders; 413 calls compared: 0 differences |
| `SprjEmkConditionHolder::QueryGroup` | `0x16ec5a0` | one group's state | 210,000 queries over random holders; 12,885,945 calls compared: 0 differences |
| `SprjEmkConditionHolder::QueryMain` | `0x16ec660` | the main group's state | 3,000 random holders: 0 differences (not called in a world session) |
| `SprjEmkConditionGroup::Query` | `0x16ebcf0` | a group's AND / OR | every group of 3,000 random holders: 0 differences (not called in a world session) |

The four flag functions (`flag_store.cpp`) are every read and write the game
makes through its flag store - event scripts, Lua, talk scripts, the online
session. With them as source every flag change is visible:
`BBHOST_EVENT_FLAG_LOG=1` logs each one, and plugins get a callback for each.

The eight `SprjEmk` functions (`interpreter.cpp`) are the event-script (EMEVD)
interpreter: the loop every map's event scripts run in, the dispatcher that
sends each instruction to its bank's function, and the condition groups the
scripts wait on. An instruction no bank function takes is skipped, as the game
skips it, and counted at exit (the game's own scripts have none). The game ends
an event by writing its completion flag (event id plus slot) straight into the
flag store rather than through `SetEventFlag` - which does nothing more than
that write - so the flag log and plugins never saw an event end; ours writes it
the same way and passes the change on (31 flags in a world session).

### Player data (`src/decomp/player/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| item discovery (`sub_1981830`) | `0x1981830` | the drop chance a kill rolls with: arcane through its curve (1 below 8, 1.7 at 30, 2.1 from 50), plus the effects' `itemDropRate` while an item-discovery effect is on | 200,000 generated cases against the game's own code; 185 calls compared in a world session: 0 differences |
| echoes gained (`PlayerIns::vf122`) | `0x1cfc600` | every echo gain: the player's `soulRate` effects, the 999,999,999 cap, the echoes ever gained, the HUD's counters | 200,000 generated cases; 1 call compared in a world session: 0 differences |
| echoes for a kill (`sub_1cfc860`) | `0x1cfc860` | the victim's `haveSoulRate` effects (scaled by the clear count for `bGameClearBonus` effects: 1, 1.1, 1.25, 1.5, 2, 2.5), halved or x1.2 for a cooperator, rounded up past 5e-6 | 200,000 generated cases: 0 differences; no kill in the test sessions |
| scripts' penalty (`lua_cli_ExcutePenalty`) | `0x1734e40` | the Lua binding that takes a share of the echoes and some insight (not what a death costs) | 200,000 generated cases: 0 differences; not called in the test sessions |
| level-up price check (`sub_1f2f5e0`) | `0x1f2f5e0` | whether an attribute may rise: the planned level's price (`CalcCorrectGraph` row 200) against the echoes not yet committed | 200,000 generated cases, 41,942 of them at the price itself: 0 differences; the level-up screen not opened in a world session |

These are where every echo the player gains, the price of every level and the
drop chance of every kill are decided: a mod that changes those rules changes
one function each, not the sites that call them. They read their constants
where the game does, so a patch to one of those still applies.

### Items (`src/decomp/items/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| item-lot roll (`sub_1bceaf0`) | `0x1bceaf0` | what an enemy, a corpse, a chest or a scripted award gives from `ItemLotParam`: each slot's weight (drop rate, item discovery, the cumulative counters that make a missed drop likelier), one draw from the game's random generator a row, the picks merged by their flags | 200,000 random lots against the game's own code - results, generator and every flag after: 0 differences; 317 rolls compared in a world session: 0 differences |

The roll is the drop rules as source: how discovery scales a slot, how the
cumulative counters raise a missed drop's weight and reset when it drops, which
flags pass a slot over. The rows stay `ItemLotParam`'s, editable by field name
as before. Its debug-menu variant and preview stay the game's. Shops have no
purchase function to take over - the purchase is inline in the shop's menu -
so `ShopLineupParam` stays the shop surface.

### Talk scripts (`src/decomp/talk/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| the EzState evaluator (`sub_2b73910`) | `0x2b73910` | evaluates every condition and command argument of the engine's state machines: the NPCs' talk scripts and the menus | every expression of the 271 talk scripts (42,016, each on six states) and 60,000 random expressions against the game's own code: 0 differences; in three world sessions 222,468 talk-script expressions compared, the environment's answers replayed: 0 differences |

An expression is bytecode - literals, arithmetic in double precision, calls to
the game's talk functions, registers, strings - run on a stack of typed values.
Ours keeps the game's rules to the bit (an int when a result converts back
exactly, else a float; the operand order NaN payloads follow; how the strings'
reference counts move) and gives the game's own handlers the opcodes no script
uses. The same interpreter runs the menus' state machines, so besides every
talk-script expression the game ships, the test runs random ones over every
opcode a script can use. The calls an expression makes pass through it: the
base for talk functions and commands of a mod's own.

### Chalice dungeons (`src/decomp/chalice/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| chalice roll (`dungeon_ritual_config_initialize`) | `0x1eaeec0` | a ritual's dungeon: which of its prebuilt layouts (the open unlock ranges weighted by size) and its rites | 20,000 rolls against the game's own code over generated params, flags and draws; in the game, 1,500 rolls through the real rows, flags and heap on a copy of the game's generator: 0 differences |
| a rite's pick (`sub_2316e70`) | `0x2316e70` | one of a `DungeonSubFeatLotParam` row's ten rates, by weight | 20,000 picks against the game's own code: 0 differences |
| chalice map uid (`sub_231fbc0`) | `0x231fbc0` | the setup's map uid: `m29_AA_BB_CC` from the layout, or the row's own for a fixed chalice | 60,000 against the game's own code; 6,000 compared in the game: 0 differences |
| chalice feature pairs (`sub_1eaf9d0`) | `0x1eaf9d0` | the nine (lot, feature) pairs the dungeon loads with | 40,000 sets against the game's own code; 1,500 compared in the game: 0 differences |

Chalice dungeons are not generated: each root dungeon is one of 100 or 200
layouts built ahead of time, and a ritual rolls which one, and its rites, when
the altar's menu opens - from the game's random generator, seeded from the
clock, so no glyph-like seed reproduces a roll in the game itself (glyphs are
the online server's keywords). With the roll as source, a mod can pick any
layout and any rites, or open every range.

### Camera (`src/decomp/camera/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| the follow camera's step (`NS_SPRJ::ChrExFollowCam::Update`) | `0x183ac60` | the player camera, once a frame: `LockCamParam` easing, a moving floor, the character's frame, reset, lock-on and turn-behind angles, the stick and the fast turn, the wanted position, the wall casts and the wall escape, the chase and the view basis, the camera's collision, the NaN guard, the keep-out spheres, the near clamp | 400,000 generated cameras, 24 million frames, against the game's own code (casts and the debug draw as stubs both versions call; the fov-uncap and 60 fps variants; the auto-rotation patch's nops and the mouse's stubs), every line of its phases run but the panics for a missing singleton and two of the param file's three layouts; 12,586 frames compared in a world session, and 1,494 in the Hunter's Dream with the mouse turning it, auto-rotation held and a controller picked up: 0 differences |

The step is 20,077 bytes of hand-scheduled SIMD - 550 multiplies, 248
shuffles, reciprocal square roots refined by Newton steps, sine and cosine
series summed with horizontal adds - where only the same operations in the
same order give the same bits. Ours is its 23 phases as functions named for
what they do, over a struct of the camera's fields with every offset checked at
compile time. The game's idioms are small helpers that perform its operations
in its order - the length from `rsqrtps` and two Newton steps, its two ways of
summing a row times a matrix, the angle wraps through integer conversions, the
sine, cosine and arcsine series - and the rest is plain C++. Its denormals come
along too: under the game's MXCSR `minss` turns a denormal into zero where a
branch keeps it, and ours chooses each value the way the game's instruction
does. Three places in the game's code are rewritten at load - the fov-uncap
patch and two 60 fps sites - and ours reads them where the game would; any
other change to the function's body, or to a constant it reads, keeps the
game's version in place.

Being source, the step is where two PC additions live, neither in the game's
function. The mouse turns the free camera as Dark Souls III's does - an angle
a count, after the stick and only without it - in the stick's phase, where
hooks on the game's code put it before (`engine/mouse_camera.h`; the hooks
remain for when the game's version runs). And while the keyboard and mouse
were used last, the camera's own turns as the character moves are held. The
community patch "Disable Camera Auto Rotation via Movement" removes four
stores: the yaw read back from where the camera is, the pitch return's weight
ramping down, the turn away from a wall - and the auto turn's step of the
yaw, whose only cause is the lock-on button finding no target (the
turn-behind request has one writer, `sub_1c8e900`), so that one only takes
away the recentre. The mouse holds the first three, each exactly as the
patch's nop leaves the game's code (a value the game goes on using from a
register, ours goes on using too), and leaves the recentre to the player.
The body check accepts the patch's nops at all four, which ours then
mirrors. The test checks all of it against the game's code: the patch's
nops at all four stores, at each alone, and at the three the mouse holds
against ours holding them by its flag; and the mouse camera's own stubs in
the game's code turning it by the same counts as ours.

### GX (`src/decomp/gx/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| flush wait (`sub_15d7030`) | `0x15d7030` | the render thread's wait for the GPU to finish a flush | the same condition, yielding the core instead of spinning on it |
| GX table reclaim (`sub_2aaa860`) | `0x2aaa860` | frees the GX layer's blocks of draw resource tables once the GPU is done with them | the same walk as the game's, plus a wait when a pool runs empty (which used to fail an allocation and black out a frame) |

### Memory (`src/decomp/memory/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| the game's `memcpy` (`sub_2a1a1b0`) | `0x2a1a1b0` | the engine's own memory copy, 1,440 call sites | checked by what it copies |
| parallel resource copy (`sub_23bde30`) | `0x23bde30` | the copy of streamed resource data across the engine's worker pool | copies on the calling thread; removes a ~1 s wait per area tour |

### Sync (`src/decomp/sync/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| `DLAdaptiveMutex::Lock` | `0x24368c0` | Dantelion's spin-then-sleep mutex: take the word when its owner is free, spin for the mutex's budget, then count in as a sleeper on its semaphore until an unlock hands the lock over | threads taking one word with the game's Lock, ours, and either at random per call, released by the game's Unlock, at spin budgets of 0, 2,000 and 200,000 ticks: exclusion, exact counts and a free word at the end (`decomp_adaptive_mutex`) |

Every work pool's queue takes this lock to pop a job (`CSEzWork::ProcessOne`),
and so do the heaps and the GX layer. The game's spin is a `lock cmpxchg` of
the word with its own value and an `rdtsc` on every turn, with no `pause`: in
the world the six character workers spent 60% of their time there, popping
jobs while the main loop waited ~1.4 ms a frame for them, because each turn
pulls the word's cache line from the core that holds it and the holder's
unlock has to win it back. Ours makes the same transitions in the same order
with the same budget, and spins on a plain load with `pause`, swapping only to
take the lock or to count itself in. The game's unlock is unchanged, and
threads using either version share a word.

### Effects (`src/decomp/sfx/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| effect ribbon tail, facing the eye (`sub_2cce7b0`) | `0x2cce7b0` | the last one to three points of an effect ribbon as vertices, the strip turned to the camera | 400,000 random strips against the game's own code; 21 calls compared in a world session: 0 differences |
| effect ribbon tail, along normals (`sub_2cceec0`) | `0x2cceec0` | the same for a ribbon laid along its points' normals | 400,000 random strips; 426 calls compared in a world session: 0 differences |

The two ribbon writers are rewritten for Windows. The game's versions keep
their arguments in the 128 bytes below the stack pointer - the red zone, which
the SysV convention the game is built for promises no signal will touch.
Windows has no red zone: an exception writes its frame there. The texture
write watch takes an exception at the first write to a page it watches, and
the ribbons' vertices often land in such pages, so on Windows the game read
its own pointers back as zero and crashed at `0x2cce9b5` (the crash the
community's "Intel 12th Gen+ SFX workaround" patch avoids by not drawing
those effects). The rewrites keep nothing below the stack pointer. Linux
skips the red zone when it delivers a signal, so only Windows crashed.

### Replaced outside the list

| Function | Address | Where |
|---|---|---|
| `SprjFlipper::Update`, the frame-time manager | `0x2434770` | `src/engine/frame_rate.cpp` |
| the frame loop's pooled-object borrow | `0x1467f60` | `src/engine/frame_pool.cpp` |
| YEBIS's per-stage resource builder | `0x15f7840` | `src/engine/yebis_bind.cpp` |
| the GX command-arena acquire | `0x2ad3b80` | `src/hle/runtime.cpp` |
| `DL_PANIC` | `0x24b55b0` | `src/hle/runtime.cpp` |

### The game's file formats

The same standard - reproduce the game's own bytes before changing anything -
applies to the data formats bbhost edits:

| Format | Where | Checked |
|---|---|---|
| Talk scripts (EzState `.esd`) | `src/engine/esd.h` | all 271 scripts parse and write back byte for byte (`tests/esd_test.cpp`) |
| Map layouts (MSB) | the randomizer, `src/engine/dream_mirror_layout.cpp` | all 2,690 layouts rebuild byte for byte (`tests/randomizer_files_test.cpp`) |
| AI script bundles (`luabnd`, `.luainfo`, `.luagnl`) | the randomizer | all 18 bundles rebuild byte for byte |
| Event scripts (EMEVD) | `src/engine/maiden_events.cpp` | checked by SHA-256 in and out, differences only where intended |
| Param definitions | `src/engine/paramdef.cpp` | the game's own definitions; all 62 tables match their rows |

## How a function takes the original's place

`src/decomp/decomp.h` keeps the list. Each entry gives the function's entry
address, the instruction bytes expected there, and the replacement.

- At load, once the executable's SHA-256 is the 1.09 build's, the first 5 or
  more bytes of the function - whole instructions - become a jump to the
  replacement. A function whose bytes differ from what it was written against
  is refused and the original stays.
- Those instructions go into a trampoline that runs them and jumps back, so
  the original remains callable - for a compare run, and for the replacement
  to fall back on. The trampoline sits within reach of the executable, so an
  instruction that addresses memory relative to itself, or a relative branch,
  is re-aimed at what it reached (`src/decomp/insn.h`); a short branch is
  widened. Every one of the executable's 162,959 functions decodes to its end
  (or to its first jump table, which it keeps in its code), and the entries of
  148,238 of them can be taken this way; the rest are shorter than the 5-byte
  jump.
- A function that replaces a whole body the game patches elsewhere can carry a
  check of that body (`body_ok`): any change beyond the patches it mirrors
  keeps the original in place.
- A **leaf** is entered directly from the game's code, on the game's thread
  and stack. It touches only the game's memory and its own globals - no host
  thread-locals, locks, logging or allocation - and is compiled without the
  stack protector. It costs what the original did.
- A **hosted** function is entered through the same thunk as the system
  library calls and may do anything host code does.
- Either reads the game through `src/decomp/guest.h`: the integer names,
  `game_address()` and `game_function<F>()` for a Binary Ninja address,
  `load<T>()`, `store<T>()` and `ref<T>()` for the game's memory - all
  leaf-safe.

Switches for a run:

| | |
|---|---|
| `BBHOST_DECOMP=0` | the game's code throughout |
| `BBHOST_DECOMP_OFF=name,...` | these functions stay the game's |
| `BBHOST_DECOMP_COMPARE=1` | functions with a compare mode run it instead |

## How one is checked

**Without the game.** `tests/decomp/eboot_kit.h` loads the executable into a
test the way bbhost does (relocated, its thread-local reads moved, the
console's floating-point mode, every system call a trap that names itself), so
the original runs there on generated inputs beside the replacement, bit for
bit. Synthetic objects live in test memory, the game's singletons point at
them, and a function the test cannot set up (a param lookup, a heap search) is
replaced by a stub both versions call. Each area's test is
`tests/decomp/<name>_test.cpp`, the ctest `decomp_<name>`, and skips where
there is no executable. `decomp_sfx_ribbons` is the smallest: 800,000 random
strips through both of the effect ribbons' writers.

**In the game.** A compare mode runs the original and checks the replacement
against it while the game plays. A function that only reads runs both
versions and compares their results; one that writes predicts what the
original will write, lets it run, and checks the memory afterwards - or
snapshots what it touches, runs the original, restores, runs ours and compares.
A function becomes the default once a world session compares with zero
differences, and a session with the replacement in place performs within noise
of one without.

Exactness is measured against the machine code, not a decompiler's reading of
it: shift counts masked to 5 bits, 32-bit arithmetic that wraps, exactly the
argument bits the original tests, the floating-point operations in their order
(no fused multiply-add; `minss` and `maxss` only where the game uses them).

## Adding one

1. Find the function in a disassembler and name it from evidence: the
   binary's strings, Dark Souls III's RTTI (the same engine, with class names),
   the PS3 build's debug symbols, or a plain descriptive name.
   `tools/class_catalog.cpp` lists the 882 classes the executable registers by
   name at start, each with the vtables that hold its `GetRuntimeClass` and the
   functions that store them (constructors and destructors).
2. Read its disassembly as well as a decompile, and list its callers.
3. Write it in `src/decomp/<area>/<name>.cpp` (a new area gets a folder) with
   its entry bytes, `decomp/guest.h` for the game's memory, and a compare mode;
   add the file to `BBHOST_DECOMP_SOURCES` in `CMakeLists.txt`.
4. Add its `decomp_<name>_add()` to `src/decomp/areas.cpp`.
5. Write `tests/decomp/<name>_test.cpp` against the game's code on the eboot
   kit, with `bbhost_decomp_test(<name>)` in `CMakeLists.txt`.
6. Run a compare session (zero differences) and a performance session (within
   noise).

## Next

- **Items in and out**: `EquipGameData::AddItem`, through which every item
  enters the inventory (it creates item instances, so a compare must follow the
  calls it makes instead of running it twice).
- **Event-script and talk-script extensions**: the interpreters are source;
  next, plugins adding event instructions and conditions, and talk functions
  and commands, of their own.
- **Damage and stamina**: the functions that apply attack params and attribute
  scaling, for balance changes params cannot express - combat stays the game's
  code until there is a decision to take it over.
