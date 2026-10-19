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
| Rewritten as source, on the decomp list (`src/decomp/`) | 190 |
| Replaced by bbhost code outside the list (written before the list existed) | 4, and the YEBIS resource builder bypassed |
| Hooked at the entry or at call sites (GX's writers and passes, the resource registry, live resolution, settings, key prompts, menus) | 110 |
| **Taken over in all** | **304** |
| With any code byte changed (including byte patches and redirected calls) | 325; 407 at 60 fps |
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
| pipeline drain (`gx_command_buffer_wait_for_eop`) | `0x2ab09e0` | makes the GPU wait for the work before (a label, an end-of-pipe event and a wait) | written as a barrier token for the command processor instead, which orders its own writes behind it; the same command buffer read and refilled the same way as the game's version, beside it on test buffers (`decomp_gx_drain_token`) |
| occlusion query (`gnm_write_occlusion_query`) | `0x1477660` | an occlusion query's begin or end: the block reset, a wait, the counter dumps | written as one query token instead, taken as the renderer's own query; checked in the game by the queries it makes |
| the Gnm packet writers nothing reads (21, below) | `0x1472ed0`-`0x148b610` | each makes room for one packet, writes it and moves the write pointer | twenty written as nothing by default, `prepareFlip`'s reservation as the game's; writing the game's packet, 420,000 generated cases against the game's own code - every byte of both chunks, the write pointer, the return value and the refill's calls - 0 differences (`decomp_gx_dead_packets`); `BBHOST_DECOMP_COMPARE=1` checks every call the game makes: 7.6 million in a three-place session, 0 differences |
| the Gnm label, wait, inline-data, predication, indirect-buffer and counter writers (12, below) | `0x1473b80`-`0x148b430` | each makes room for one packet - a label when the work before it is done, a wait, data written inline, predication on an occlusion result, the compute ring's call of a command buffer, the draw engine's side of the constant engine's counters - writes it and moves the write pointer | the game's packets, written the same; 240,000 generated cases against the game's own code - every byte of both chunks, the command buffer and its flags, the return value and the refill's calls - 0 differences (`decomp_gx_labels`); `BBHOST_DECOMP_COMPARE=1` checks every call the game makes: 1.3 million in a three-place session (four of the nine called), 0 differences |
| the DMA_DATA writer (`gnm_dma_data`) | `0x1473e60` | the command processor moving or filling memory: GX's buffer copies and fills, the GPU particles' counter moves through GDS, Gnmx's, YEBIS's and Toolkit's | the game's packet, except for the two moves through GDS around a counting dispatch while the counters ride in the dispatch tokens: those write nothing (they were two five-byte NOPs at their calls); 20,000 generated cases against the game's own code, 0 differences, and the two moves' calls checked to be the writer's (`decomp_gx_dma`) |
| the constant engine's writers (`gnm_ce_reserve_const_ram`, `gnm_ce_wait_on_de_counter_diff`, `gnm_ce_increment_ce_counter`, `gnm_ce_dump_const_ram`) | `0x15fd190`-`0x15fd3e0` | the constant command buffer's packets: its RAM written (reserved for the caller to fill), its counters stepped, its RAM dumped into memory | nothing dumps that RAM (YEBIS's tables come from its own state, Gnmx's draws are tokens), so with the dead packets on the counter step writes nothing; the reserve (its payload is guest memory the draws read), the wait and the dump write the game's packets - the wait's five callers NOP over it when a draw leaves the engine unused, and written as nothing it put that NOP below it, over the heap block in front of an empty constant command buffer (the flip pass's: a live resize crashed in free) - and a dump is counted if it ever comes; writing the game's packets, 80,000 generated cases against the game's own code, 0 differences (`decomp_gx_constant_engine`); the pre-draw's NOP on a real constant command buffer, ours and the game's pre-draw, both modes (`decomp_gnmx_cue_ccb`, below) |
| GX's draw, dispatch and commit methods (`gx_immediate_draw`, `gx_immediate_draw_indexed`, `gx_immediate_draw_instanced`, `gx_immediate_draw_indexed_instanced`, `gx_immediate_draw_instanced_indirect`, `gx_immediate_draw_indexed_instanced_indirect`, `gx_context_commit`, `gx_immediate_dispatch`, `gx_immediate_dispatch_indirect`) | `0x25695c0`-`0x2569ba0` | `ID3D11DeviceContext`'s Draw\*, Dispatch\* and a commit without a draw, as GX has them: the output-merger, input-assembly and rasterizer flushes, the command buffer's flush, the commit of the shaders' user data and tables, Gnm's draw or dispatch packets, the post-draw, the flush again | each first hands the call to the GX trace, as its prologue hooks did - a draw it takes whole becomes a host-draw token and nothing more runs; otherwise the game's calls in the game's order. 18,000 generated cases against the game's own code: the calls and their arguments, the frame handed to the trace, and the return address each draw and dispatch writer is called from, which the trace knows a GX method's packets by (`decomp_gx_draw`) |
| the output-merger flush (`gx_flush_output_merger`) | `0x2ad2230` | the deferred passes of targets no longer bound (the fast-clear eliminate, the depth decompress), then the blend, depth-stencil, stencil-reference and depth-bounds state written where it differs from what the context last wrote (the blend constants compared as floats) | 10,000 generated cases - the calls, the floats' bits, the block - and five deliberate mistakes caught; `BBHOST_DECOMP_COMPARE=1` with `BBHOST_GX_NATIVE_FLUSHES=1` (draws taken whole skip the flushes by default): 2,667,388 calls at three places, 0 differences (97,484 left to the game's alone: a deferred clear due, or too little room) |
| the post-draw (`gx_post_draw`) | `0x2acbd50` | the flushes' dirty bits folded into the flags the next commit reads; the multi-VGT parameter and the GS mode put back | 10,000 generated cases; `BBHOST_DECOMP_COMPARE=1`: 2,764,861 calls at three places, 0 differences |
| the command buffer's flush (`gx_command_buffer_flush`) | `0x2aaf700` | what every GX operation ends with: the write pointer noted, and for a context recording into an arena chunk the segment closed - the release nodes handed back, the queued chunks retired, a fresh start | 10,000 generated cases (6,994 closing a segment); `BBHOST_DECOMP_COMPARE=1`: 21,035,589 calls at three places, 0 differences (the 170 that closed a segment left to the game's alone) |
| FinishCommandList (`gx_finish_command_list`, `gx_finish_command_list_impl`) | `0x2ad5850`, `0x256aca0` | a deferred context's recorded spans taken into an `ID3D11CommandList`, with the targets around them; the context's caches made unknown, its compute state cleared unless kept | 40,000 generated cases against the game's own code for the ten - the calls and their arguments, what the submissions hold, the pages freed, the memory, the return values - and nine deliberate mistakes caught (`decomp_gx_command_list`); in the game every frame's lists, segments and flip, at three places: frames right, 60 fps, CPU inside the run-to-run spread |
| ExecuteCommandList(s) (`gx_execute_command_list`, `gx_context_execute_command_list`, `gx_execute_command_lists`, `gx_context_execute_command_lists`) | `0x25679a0`, `0x256a7d0`, `0x25679e0`, `0x256a980` | on the immediate context: the deferred passes a list's first targets owe run, the list's spans submitted (one list) or put into the context's own stream (the frame's batches of up to 72), the targets each list ended with taken as bound, the lists emptied and their pages freed | the same test (`decomp_gx_command_list`); in the game every frame's lists, segments and flip, at three places: frames right, 60 fps, CPU inside the run-to-run spread |
| a segment to the GPU (`gx_command_buffer_close_segment`), and the frame's last with the flip (`gx_command_buffer_flip`, `gx_command_buffer_close_segment_and_flip`, `gx_submit_segment_and_flip`) | `0x2ab0b00`, `0x2aafb30`, `0x2ab0ce0`, `0x2ab0e30` | a command buffer's segment submitted (`sceGnmSubmitCommandBuffers`), after the compute ring's waits when the buffer starts afresh; for the flip, room made for the system's flip packets first, and the segment submitted with the flip (`sceGnmSubmitAndFlipCommandBuffers`, then `sceGnmSubmitDone`) | the same test (`decomp_gx_command_list`); in the game every frame's lists, segments and flip, at three places: frames right, 60 fps, CPU inside the run-to-run spread |
| the swap chain's present (`gx_swap_chain_present`, `gx_swap_chain_present_frame`) | `0x2ad5dd0`, `0x2ad6e50` | the frame's end on the render thread and its pacing: at most as many flips in flight as there are display buffers (two) - VideoOut's flip status asked and, while the flip that many back has not completed, a wait on the flip event queue - then the display buffer's address, the output merger's end of frame and the copy to the display buffer, the last segment submitted with the flip (mode 2 for a sync interval of 0, 1 otherwise), the count stepped, the block pools' releases taken back, the context's state reset | 10,000 generated cases against the game's own code, VideoOut's status, the event wait and the thread query bound to the test and scripted - the calls, the swap chain and the context, the return value - and three deliberate mistakes caught (`decomp_gx_present`); in the game at three places, on against off: frames right, 60 fps, the command processor and CPU inside the run-to-run spread |

The 21 packet writers (`gx/dead_packets.cpp`) are Gnm's own: the four
`ACQUIRE_MEM` cache actions (`gnm_acquire_mem_range`, `dcb_acquire_mem`,
`gnm_compute_acquire_mem_range`, `gnm_compute_acquire_mem`), fifteen
context-register writers (the multi-sample sample locations, centroid priority
and configuration, the depth-stencil disable, the blend constants, alpha to
mask, the scratch ring size, the screen offset, the guard band's two
adjustments, the instance step rates, the pixel shader's input table, the
vertex rounding and the line width), the shader prefetch's `DMA_DATA` and
`prepareFlip`'s room for the console's flip packets. Every draw is a host token
carrying its own state, the renderer orders its work with Vulkan barriers and
the flip is bbhost's, so nothing on the host reads those packets: by default
ours writes nothing for twenty of them and returns the write pointer, the
address no caller reads (`BBHOST_CP_DEAD_PACKETS=2` returns one no caller could
follow instead, a check; `=0` writes the game's packets;
`BBHOST_CP_DEAD_KEEP=<places>` keeps chosen ones writing, by their place in the
table). The context-register writers write theirs while the command processor
keeps a register file (its shadow and packet-path modes). `prepareFlip`'s
reservation writes the game's 64-dword NOP all the same: it changes where a
frame's command buffer ends in its chunk, and without it the flips settled
into a jittery pacing more often and Fishing Hamlet missed a refresh every
3-4 seconds. Over the 12-place tour the game calls the twenty about 31 million
times: 96.9M dwords of cache actions and 95.3M of context registers the
command processor no longer walks. As source, each is also where a host
command can take the packet's place.

The twelve writers in `gx/labels.cpp` carry every completion the game asks the
GPU for and every wait it has the GPU make: the end-of-pipe and end-of-shader
labels (`gnm_write_end_of_pipe_event`, `gnm_write_end_of_shader_event`) that
GX's block pools and command arena, its Map and queries, YEBIS's drains,
Scaleform's fences and the flip are written with, the compute command buffer's
release (`gnm_dispatch_write_release_mem`, the compute ring's retirement), the
two `WAIT_REG_MEM` writers (`gnm_wait_reg_mem`, `gnm_dispatch_wait_reg_mem`,
the same code in the draw and compute command buffers), the two `WRITE_DATA`
writers (`gnm_write_data_inline`, `gnm_write_data_inline_l2`) and
`SET_PREDICATION` set and cleared with the command buffer's predication flag
(`gnm_set_predication`, `gnm_clear_predication`), and the compute ring's call
of each command buffer it runs (`gnm_dispatch_write_indirect_buffer`), and the
draw engine's side of the constant engine's counters
(`gnm_increment_de_counter`, `gnm_wait_on_ce_counter`). They write the game's
packets: the command processor walks them, and records each label as a GPU
write of the label's memory. As source, each is where a label or a wait
becomes a command of the command buffer's own host list - the step that lets
the command processor stop walking the game's command buffers.

GX's methods (`gx/draw.cpp`) are where the game says what it draws, in
Direct3D 11's terms - the binary names the interfaces it emulates on the
console (`ID3D11DeviceContext`, `ID3D11CommandList`, `ID3D11BlendState` and
the rest). Before they were source, the GX trace saw each call through a hook
at the method's entry; now each method hands the call to the trace itself,
with the same frame. The trace tells a GX method's packets from everyone
else's by the return address of the Gnm writer that writes them, so the
methods - and YEBIS's wrappers, below - share a section of their own
(`__start`/`__stop` bounds on ELF, grouped sections on COFF) and a return
address inside it is read as the game's from the same function
(`decomp_gx::game_return`). A deferred context's draws end in
a command list, and the immediate context executes the lists in order
(`gx/command_list.cpp`); the objects that carry the work across - the span
(a segment of a command buffer, as its eight-qword header stood when it was
cut), the span queue and the command list with the targets around it - are
named in `gx/command_list.h`.

**The plugin surface these open (design - the plugin API does not have it
yet).** Four kinds of hooks, all costing nothing until a plugin asks:

- *Per command*: a callback as one of GX's methods is entered, before the
  trace, on the thread that records the call (a GX worker for a deferred
  context, the render thread for the immediate one). It gets the call as GX
  objects - the kind and its arguments (counts, starts, instances, base
  vertex, an indirect draw's argument buffer and offset), the context, and
  the pending state as Direct3D 11 objects: the shaders, the shader-resource,
  render-target, depth-stencil and unordered-access views, the constant
  buffers, the input layout, the vertex and index buffers, the blend,
  depth-stencil and rasterizer states, each with the ids the shader dump and
  the HLSL overrides use. It can let the call run, skip it, change the
  pending state for it (another shader, texture or state object, written
  where the flushes read it, so the same draw goes out with them), or add
  calls of its own on the same context. The output-merger flush, as source,
  is where a state object can be swapped for every draw at once, as it is
  applied.
- *Per command list*: callbacks at FinishCommandList (a deferred context's
  list complete: its spans, the targets it began and ended with) and at
  ExecuteCommandList(s) (the immediate context taking the lists, in order).
  The lists in order, with the targets each wrote, are the frame as a graph of
  passes: a plugin can name passes, capture one pass's targets, insert a pass
  of its own between two lists (an effect after the scene, before the UI),
  or leave one out. The segment's close is where host commands could go to
  the GPU between two of the game's.
- *Per frame*: a callback in the swap chain's present, where the frame ends:
  before its last segment goes with the flip (a pass over the final image, a
  capture) and around the wait that keeps the game at most two frames ahead
  of the display (another pacing).
- *Per post-processing parameter*: a callback in PPFX's parameter setters
  (YEBIS, below), which the flip pass calls with every frame's parameters
  before it runs the chain - the scene's projections and luminance scale,
  auto exposure, tone mapping, chromatic aberration, the colour matrix, the
  glare, light shafts, depth of field and auto focus, motion blur, the
  feedback and antialiasing. It sees each value as the game set it for the
  frame and can change it before YEBIS takes it: an effect left out, the
  bloom scaled, the focus moved.

The callbacks would run on the GX threads, so they must be thread-safe and
must not block; the cost is one call through the host's thunk per command
(about 10 ns measured on Linux) plus what the callback does. YEBIS's
post-process passes are covered at its D3D11-shaped context, its Gnm
wrappers and PPFX's two entry points, and its parameters at PPFX's setters
(below); Scaleform's menus and the flip pass's own draw are not yet.

### YEBIS (`src/decomp/yebis/`)

YEBIS is the post-process library the engine's post-processing goes through
(its `ppfx.cpp` layer): tone mapping, glare, depth of field, motion blur,
colour grading. On the console it keeps a D3D11-shaped device context of its
own - the D3D11 device context of its Gnm shim - whose vtable (`0x570a380`)
holds `ID3D11DeviceContext`'s 115 methods in their order and 18 of YEBIS's
own; YEBIS's effects call it as they would call D3D11, and it turns each call
into Gnm work through three wrappers.

| Function | Address | What it does | Checked |
|---|---|---|---|
| YEBIS's Gnm draws and dispatch (`yebis_gnm_draw_index`, `yebis_gnm_draw_auto`, `yebis_gnm_dispatch`) | `0x15fbc40`, `0x15fbb50`, `0x15fbd10` | a pass's draw - the prep (its last two arguments into two user-data slots), the commit (every stage's shader, resources and user data), Gnm's draw packet, the post-draw - or its compute dispatch the same way | the indexed draw and the dispatch hand the call to the GX trace first, as its hooks did (YEBIS's passes become tokens built from YEBIS's own state); writer callers, as GX's methods are; 12,000 generated cases against the game's own code - the calls, the frame handed to the trace, the writers' return addresses (`decomp_yebis_gnm`) |
| YEBIS's context: OMSetRenderTargets, the draw and dispatch prologues, DrawIndexed, Draw, DrawIndexedInstanced, DrawInstanced, Dispatch, DispatchIndirect, Flush, YEBIS's two flushes of its own, the three waits (`yebis_context_om_set_render_targets`, `yebis_context_prepare_draw`, `yebis_context_prepare_dispatch`, `yebis_context_draw_indexed`, `yebis_context_draw`, `yebis_context_draw_indexed_instanced`, `yebis_context_draw_instanced`, `yebis_context_dispatch`, `yebis_context_dispatch_indirect`, `yebis_context_flush`, `yebis_context_flush_to`, `yebis_context_wait_flush`, `yebis_wait_compute`, `yebis_wait_pixels`, `yebis_wait_dispatch`) | `0x15d74c0`, `0x15d7400`, `0x2e433a0`, `0x2e40880`, `0x2e408e0`, `0x2e40aa0`, `0x2e40b40`, `0x2e410b0`, `0x2e410f0`, `0x15d99e0`, `0x15d9ab0`, `0x15d9b70`, `0x15e8e20`, `0x15ebc20`, `0x15ebc90` | the context's slots 33, 12, 13, 20, 21, 41, 42, 111, 131 and 132 and what they call: the colour and depth targets bound, with YEBIS's own hazard tracking - a target bound again after a draw since the last drain costs a drain (an end-of-pipe label, a wait for it, the caches written back) before anything draws into it; the state a draw needs written in its prologue; a dispatch after a draw waits for the pixel shaders, one after a dispatch and a draw after a dispatch for the compute shaders; a flush submits YEBIS's command buffers and starts the ring's next | 30,000 generated cases against the game's own code, 2,000 a function - the calls and their arguments, every byte of the context, the views, their textures, the command buffer's header and the drain's label, the return value - and nine deliberate mistakes caught (`decomp_yebis_context`); `BBHOST_DECOMP_COMPARE=1` checks OMSetRenderTargets call by call: 81,034 calls at three places, 0 differences (289,977 left to the game's alone: a view's first bind, which puts it on its owner's list, or a drain); in the game at three places, all fifteen on against off: frames right, 60 fps, the command processor and CPU inside the run-to-run spread |
| PPFX's entry points (`ppfxBindFramebuffer`, `ppfxApplyEffects`) | `0xfb68f0`, `0xfb6980` | the engine's whole post-processing chain as two calls: the frame's colour and depth handed to YEBIS's single post effect, then every effect run (tone mapping, glare, depth of field, motion blur, colour grading); a failure printed with YEBIS's debug output on, and a failed ApplyEffects asserts | 10,000 generated cases against the game's own code - the calls and their arguments, the result - and two deliberate mistakes caught (`decomp_yebis_ppfx`); in the game at three places, on against off: frames right, 60 fps, the command processor and CPU inside the run-to-run spread |
| PPFX's parameter setters (69: `ppfx_set_render_scene_luminance_scale`, `ppfx_set_auto_exposure_*` (6), `ppfx_set_tonemap_parameters`, `ppfx_set_chromatic_aberration_enable`, `ppfx_initialize_matrix`, `ppfx_multiply_matrix_*` (4), `ppfx_set_glare_parameters`, `ppfx_set_glare_bloom_num_levels`, `ppfx_set_active_light_shaft`, `ppfx_set_light_shaft_source_enable`, `ppfx_set_auto_focus_*` (6), `ppfx_set_depth_of_field_edge_quality`, `ppfx_set_depth_of_field_depth_remap_lut_parameters`, `ppfx_set_motion_blur_enable`, `ppfx_set_feedback_enable`, `ppfx_set_antialias_enable`, and 40 the binary gives no name for, `ppfx_sub_<address>`; the source file lists each with the fields it sets) | `0xfb2710`-`0xfb45c0` | each frame's post-processing parameters as the flip pass hands them to YEBIS's single post effect, in its order: each entry passes its arguments to the effect's method of the same parameters (five write their one field in place), so every value the game sets for a frame passes through ours, where a hook can read it or change it; the methods stay the game's. The flip pass's first argument to each, its PPFX context, is left unread, as the game's entries leave it | 138,000 generated cases against the game's own entries, 2,000 an entry - the method reached, the effect, each integer argument in the width the method reads, the floats' bits, the result; the five stores' bytes - and six deliberate mistakes caught (`decomp_yebis_ppfx_setters`); in the game on lavapipe at Central Yharnam (189 of 189 functions ours, frame right) and in four runs on the GPU at Central Yharnam and Mensis, uncapped: frames right, 0 draw failures; no run without them - each costs the game's entry a jump, at most 69 times a frame |
| YEBIS's ClearRenderTargetView and its compute clear (`yebis_context_clear_render_target_view`, `yebis_clear_target_compute`, `yebis_compute_clear`) | `0x2e41320`, `0x15e8e90`, `0x15e8930` | slot 50: a target with a fast-clear mask cleared by YEBIS's fast clear and its slot's clear colour (8-bit unorm: floor(c * 256), 255 for exactly 1); any other by a compute pass - the shader for the texture's type, the colour in a 16-byte constant buffer, a dispatch per mip level and array slice, the wait for the compute shaders, the context's shader, texture and constant buffer put back | 9,000 generated cases against the game's own code - the calls and their arguments, the context, the texture - and three deliberate mistakes caught (`decomp_yebis_clear`); in the game at three places, on against off: frames right, 60 fps, the command processor and CPU inside the run-to-run spread |

As source, each of YEBIS's passes is seen as D3D11 calls - which targets it
binds, what it draws, where it flushes - and as a command with its context at
the wrappers: the post-process chain's side of the per-command plugin surface
above, at the granularity Dark Souls III's D3D11 renderer has.

### Gnmx (`src/decomp/gnmx/`)

Gnmx is the console SDK's graphics context over Gnm; the flip pass
(`0x25d5f80`) draws its full-screen pass through one, once a frame, and
submits that context's command buffer itself.

| Function | Address | What it does | Checked |
|---|---|---|---|
| the constant update engine's pre-draw flush and post-draw (`gnmx_cue_pre_draw`, `gnmx_cue_post_draw`) | `0x1492e80`, `0x1493150` | before the draw: the pixel shader's input table when a shader changed, each stage in use (pixel and vertex, the geometry pair or the hull pair when the engine's flags say) prefetched and its dirty resources applied, the global table written, the constant and draw engines' counters (the constant engine's wait undone when nothing used it); after it, the draw engine's counter | 10,000 generated cases against the game's own code - the calls and their arguments, every byte of the engine - and three deliberate mistakes caught (`decomp_gnmx_cue`); with the constant engine's writers ours and the game's NOP writer, runs of draws on one constant command buffer, empty or holding a reservation, with the dead packets on and off: the NOP lands on the wait (or the wait stays, the engine used), nothing below the buffer, in a payload or past the write pointer written, ours as the game's - three mistakes caught, among them the wait written as nothing (`decomp_gnmx_cue_ccb`); in the game at three places, on against off: frames right, 60 fps, the command processor and CPU inside the run-to-run spread |

### Memory (`src/decomp/memory/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| the game's `memcpy` (`sub_2a1a1b0`) | `0x2a1a1b0` | the engine's own memory copy, 1,440 call sites | checked by what it copies |
| parallel resource copy (`sub_23bde30`) | `0x23bde30` | the copy of streamed resource data across the engine's worker pool | copies on the calling thread; removes a ~1 s wait per area tour |

### Sync (`src/decomp/sync/`)

| Function | Address | What it does | Checked |
|---|---|---|---|
| `DLAdaptiveMutex::Lock` | `0x24368c0` | Dantelion's spin-then-sleep mutex: take the word when its owner is free, spin for the mutex's budget, then count in as a sleeper on its semaphore until an unlock hands the lock over | threads taking one word with the game's Lock, ours, and either at random per call, released by the game's Unlock, at spin budgets of 0, 2,000 and 200,000 ticks: exclusion, exact counts and a free word at the end (`decomp_adaptive_mutex`) |
| `NS_SPRJ::CSEzWork::ProcessOne` | `0x2432370` | a work pool's job pop: under the pool's queue lock, unlink the first node of its job list and free it; after the lock, run the job and settle it - its group's countdown and reference count, then by its kind nothing, the finished list and its event, or its own destruction - as the game's does; and the first time it sees a pool whose job list allocates from the engine's general allocator, it gives that list a node cache (below) | 20,000 generated pools against the game's own code - empty, one to five jobs, a count that disagrees with the list, jobs of every kind, groups at every countdown and reference count - the calls and their arguments in order, the memory, the result, six deliberate mistakes caught (`decomp_ez_work`); the cache through the game's own Enqueue, Append, ProcessOne, DeleteCompleted and ~CSEzWork against the same sequences without it - 3,000 generated pools, runs past its cap, lists on another heap - the same jobs run, destroyed and freed in the same order, the same lists after every step, every block back in its heap exactly once, only ever called under the pool's lock, eight deliberate mistakes caught (`decomp_ez_node_cache`); in the game (Central Yharnam and Cathedral Ward, uncapped and at 60 fps, the cache off and on in turn): all but the first thousand nodes from the caches, the queue lock's waits 2.5-3.5 down to 1.35-2.2 ms a frame, the heap's lock taken half as often, the character workers 9-13 points of a core lower at 60 fps, the main loop 0.35-0.55 ms shorter and 4-6% more frames uncapped |

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

The pop itself held that lock longer than it needed: every push of a job
allocates its list node (0x18 bytes, `CSEzWorkList::Append`, `0x2432780`) and
every pop frees one, both under the queue lock and both through the engine's
general allocator, whose heap takes a lock of its own. At Central Yharnam the
character pool pushes some 375 jobs a frame, and with the heap's free inside
the queue lock's hold 46-47% of that lock's takes found it held and its
threads waited 3-3.5 ms a frame. Freeing the node after the unlock only moved
the convoy onto the heap's lock, which then met the pool's whole churn. So
ours gives each pool a cache of its own nodes: the first time it sees a pool
whose job list allocates from the general allocator, it puts a stand-in in the
list's allocator slot - that allocator object copied, its aligned alloc and
its free replaced - which keeps up to 1,024 freed nodes and gives them back to
the list's pushes, passing anything else on. Every alloc and free on a pool's
job list happens under that pool's queue lock, or in its destructor, so the
cache needs no lock of its own; the destructor frees the list's sentinel last,
through the slot, and the cache gives its nodes back to the heap then.
`BBHOST_EZ_NODE_CACHE=0` leaves the pools' lists as they were. The workers
still meet on the queue lock - 41-43% of its takes find it held, against
46-47% - but wait 35-60% less, and spend 43-53% of their time spinning
instead of ~60%.

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

- **The other writers outside GX**: the flip pass itself (`0x25d5f80`, one
  large function: each frame's post-processing parameters handed to PPFX, its
  Gnmx draw, its label and wait and its submit inline) - its two natural hook
  points, PPFX's entry points and the swap chain's present, are source, and
  so are the parameter setters it calls. YEBIS's own submissions (`0x15faff0`,
  `0x15fb220`, `0x15fb2c0` and the list they build, `0x15fb070`) and Gnmx's
  lightweight context's submit (`0x149bc90`, Scaleform's and two engine
  paths') are left as they are: nothing in the sessions checked showed them
  running - YEBIS's flushes, the only callers of its submission, never wrote
  their labels in the writers' census, nor did Scaleform's path into Gnmx's -
  so as source they would open no surface.

- **GX's command lists resolved where they are finished**: with
  FinishCommandList and ExecuteCommandList(s) as source, each list's draws
  could be resolved and recorded into Vulkan secondary command buffers on the
  thread that finished it, the command processor only checking and stitching
  the lists in order. The design note in [rendering](rendering.md#resolving-draws-where-their-command-lists-are-finished-a-design-note)
  says what it could take off the command processor, what must stay serial,
  its risks, and why it waits for a faster main loop.

- **Items in and out**: `EquipGameData::AddItem`, through which every item
  enters the inventory (it creates item instances, so a compare must follow the
  calls it makes instead of running it twice).
- **Event-script and talk-script extensions**: the interpreters are source;
  next, plugins adding event instructions and conditions, and talk functions
  and commands, of their own.
- **Damage and stamina**: the functions that apply attack params and attribute
  scaling, for balance changes params cannot express - combat stays the game's
  code until there is a decision to take it over.
