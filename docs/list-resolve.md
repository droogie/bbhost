# Resolving draws per command list: the plan, measured

The design note in [rendering.md](rendering.md) ("Resolving draws where their
command lists are finished") proposes that the command processor (`bb-cp0`)
stop visiting each GX-native draw: each GX command list's draws resolved where
the list is finished, and the command processor checking each list against
what ran before it and stitching the lists in order. This page is the first
measured step towards it: what the command processor's time is made of today
at a place where it limits the frame rate, which parts of a draw could be
resolved per list and which must stay in the command processor's order, how
its token stream falls into the runs a list's resolution would cover, and the
order to build it in. `BBHOST_LIST_RESOLVE=runs` counts the runs
(`src/host/list_resolve.cpp`), and `BBHOST_RUN_BATCH=1` has a run's draws
share their targets' lookups on the command processor (+3% where it limits);
nothing that resolves or commits a draw off the command processor is built
yet.

## Where it was measured

Kyo's save in Central Yharnam (`m24_01`) on his Radeon 8060S (Strix Halo, 16
cores), build 81cc83bb (the buffer shadow's lite mode on an integrated GPU)
with ThinLTO and `-O3`, headless at 640x360 so that the command processor and
not the GPU is the limit, his options otherwise, 100-second runs of which ~88
are in the world.

- There the command processor limits: 165 fps in the sampled run, `bb-cp0`
  96% busy, the game waiting for it 73% of the time; the main loop's own work
  4.4 ms a frame and its GX barrier 0.5 ms. (The A/B runs half an hour later ran
  at 130-140 fps with `bb-cp0` 98% busy and the main loop's own work 4.7-4.9
  ms: the machine's state drifts, so only alternating runs compare.)
- The game draws ~1,390 GX-native draws a flip, and the command processor
  takes ~4.2 µs of its time a draw, everything it does included.
- `BBHOST_SAMPLE=bb-cp0` at 250 Hz from flip 17,000: 12,371 samples. Each
  sample goes to the part its innermost frame names (inline chains included,
  `addr2line -i`); a sample in `draw_impl`'s own lines goes to the part that
  the `BBHOST_GX_COST` section it lies in serves (the stamps' lines in
  `render.cpp`). A helper several parts share (`copy_version_over`) is passed
  over for its caller.
- `BBHOST_GX_COST=1` times the same sections at every draw. Its timers add
  about a microsecond a draw, so its nanoseconds are weights, not the cost.
- Kyo's save moved at 19:14 the same evening, to a place in the same area
  with ~350 draws a flip, where the main loop limits at ~290 fps headless and
  the command processor is 72-75% busy. The runs after that use a copy of his
  save from 18:11, the place everything above was measured at; where a number
  comes from the later save, it says so.

## Today's draw path, split

Three classes:

- **Per list.** Depends only on the draw's records, written at the GX call,
  and on caches that only grow (programs, fetch shaders, translations,
  pipelines, stage plans); what it allocates (descriptor sets, params slots,
  table and packet space) can come from arenas the list owns. Resolvable as
  soon as the list is finished.
- **Per list, checked.** Depends on state that the command processor's order
  changes now and then: a list resolves it with the state as it stands, and
  the command processor checks, once per run, that nothing it passed since
  touched what the run reads - a GPU write, a copy version, a twin, a window
  over its ranges, a texture's surface changed.
- **In order.** The command processor's own work: the packet walk with its
  labels and waits, copies, render targets and passes, copy versions and GPU
  writes as they are made, dispatches, submission.

| Part | What it is | Share of `bb-cp0` |
|---|---|---|
| **Per list** | | **42.3%** |
| programs, fetch shaders, vertex formats | `program_for`, `fetch_for`, `paths_for`, `plan_vertex_input`'s layout | 8.1% |
| the draw packet | descriptor-write copies, pipeline and set binds, the rasterizer and depth state, dynamic state, the draw, its bookkeeping (`DrawCmds`) | 7.8% |
| user data and tables | `build_gx_user_data`, `place_vertex_table` | 6.3% |
| token, records, lookahead | `hle_gx_native_token`, the records' unpacking and prefetches, `native_draw` | 6.1% |
| pipeline key and lookup | `resolve_key_stage`, `pipeline_key`, `gfx_pipeline` | 5.2% |
| descriptor sets and params | the set cache, the params slots | 4.9% |
| other draw code | the capture and trace checks, bind setup | 3.0% |
| YEBIS and Scaleform tokens | their own draw paths | 0.8% |
| **Per list, checked** | | **31.4%** |
| constant buffers | `resolve_stage_buffers`: the V#s from the records, the heap's and the lookups' answers, the snapshot copies | 13.7% |
| the buffer shadow | vertex and index placement, the lite mode's snapshots and the GX hand-outs it drains | 10.0% |
| texture views | `prefetch_stage_images`, view entries, surface refreshes | 5.0% |
| copy-version lookups | a binding's range against the copy versions | 1.7% |
| GPU-write lookups | a binding's range against the GPU writes still to land, the twins | 1.0% |
| **In order** | | **20.2%** |
| render targets and passes | `target_image` (with `rt_written` and pending clears), `begin_pass`, layouts | 9.3% |
| copy versions made | the copy tokens' copy-backs | 3.3% |
| the packet walk | `execute`, `execute_packet` | 2.7% |
| copies, fills, uploads, clears | their tokens | 1.9% |
| submission | `submit_locked`, the recorder's hand-off | 1.4% |
| dispatches | their tokens | 0.7% |
| labels and waits | | 0.4% |
| GPU writes noted | | 0.3% |
| tessellation | the LS passes | 0.2% |
| **Neither** | | **6.1%** |
| the GPU mutex | its release where another thread waits for it (1.1% at the end of each draw), other locks | 3.4% |
| the log | the harness runs with tooling on; Kyo's game does not | 2.4% |
| other | | 0.3% |

Most of the render-target share - the `target_image` memo's hit, the target
marked written, a pending clear, the pass key compared with the open pass - is
the same for every draw of a run, and becomes per-run work; what stays per
draw is a sampled depth target's layout and the view clears' scan of the
images the draw samples.

`BBHOST_GX_COST`'s weights agree. From two runs at the 18:11 save that agree
within 3% (another session's game shared the machine with both, the second
for 35 of its ~100 s): of ~4.9 µs a draw under its timers, 0.42 of it the
cost mode's own constant-buffer check, the programs and vertex input take
0.96, the constant buffers 0.80 (two stages), the user data 0.51, the sets
0.46, the targets 0.26, the pipeline key 0.23 and the render pass 0.18.

## Runs

`BBHOST_LIST_RESOLVE=runs` counts, as the command processor walks, the runs of
GX-native draws: consecutive draws into the same targets (colour targets'
bases, views, formats and resource generations; the depth target's when the
draw tests or writes depth; a clear the draw asks for) with nothing between
them that the command processor must do in order - a wait, a label, a copy,
fill, upload or clear token, a dispatch, an occlusion query's token or dump,
a YEBIS or Scaleform draw, a CP write, a call, the end of a submission. A GX
pipeline drain's barrier token ends no run: it orders the command processor's
own writes after the work before it, and no draw's resolution reads it.

In the world (eleven 1,000-flip windows from the world's first full one):

| | |
|---|---|
| GX-native draws a flip | 1,387 |
| runs a flip | 67.3, 20.6 draws each |
| stretches between in-order work (a change of targets not counted) | 47.3 a flip, 29.3 draws each |
| GX command lists (a change of GX context between two draws) | 32.5 a flip, 42.7 draws each |
| runs with draws from two GX contexts | none |
| runs ended by | a copy token 55.0% (30.4 draws), other targets 29.7% (12.2), an occlusion query 10.8% (1.0), a dispatch 3.0% (4.3), the end of a submission 1.5% (1.0) |
| draws a run: share of runs (of draws) | 1: 46.5% (2.3%); 2-3: 10.4% (1.0%); 4-7: 1.9% (0.7%); 8-15: 15.7% (9.4%); 16-31: 5.8% (6.3%); 32-63: 8.9% (21.7%); 64-127: 8.2% (35.1%); 128+: 2.5% (23.6%) |
| flips from the GX call to the command processor | 1: 51.3%, 2: 45.9%, 3 or more: 2.6%, the same flip: 0.2% |

Half the runs are one draw - an occlusion query's proxy between its begin and
end, a draw between two copy passes - and hold 2.3% of the draws; 58.7% of
the draws are in runs of 64 or more. The copy tokens that end most runs are
GX's buffer-copy passes, mostly the copy-backs of renamed dynamic buffers
that the renderer makes copy versions of. Those are in the list's own order,
so a list's resolver could take them in and its runs would merge; GX's
pipeline drains (112 barrier tokens a flip) end none.

## What the command processor would keep

At 165 fps the command processor's share of a 6.1-ms flip is ~5.8 ms. The
per-list parts and the checked ones are ~4.3 ms of it. What stays is the
in-order part, the lock, and each run's own work: a check of what the run
reads, its pass (begun only where the targets change), and one record for
`bb-record` - 67 runs a flip at a microsecond or two each, about 0.1 ms. That is ~1.5 ms a flip, and the main loop would be the limit: 4.4-4.9
ms of its own work and 0.5 in the GX barrier, ~200 fps here against 165-140
today (+20-40% by the machine's state). At Kyo's 1080p, where the GPU is
~70% busy at the command processor's ~145 fps, the GPU would be next at about
the same point. The ~4.3 ms a flip goes to the pool: about one core at 200
fps, on a machine whose game threads used 4.6 of 32 hardware threads in these
runs. `bb-record` (15-17% busy) records proportionally more as the frame rate
rises.

## The disagreements to expect

- **Constant buffers.** Of 5.1 million constant buffers hashed at the GX call
  (one draw in 16, `BBHOST_GX_COST`), 12.9% held other bytes when the command
  processor bound them. `BBHOST_CB_RECORD_PROOF=1` (`host/cb_proof.h`) says
  which: none that the command processor copies. At the 18:11 save, of 85.3
  million bindings it copied as recorded (memory the game writes on the CPU),
  every one held the bytes it had at the call (56.7 million of 56.7 at the
  19:14 save). What changed is memory the GPU writes: 4.99 of 9.03 million
  bindings read in place because a GPU write was still to land, and 5.00 of
  7.49 million copy-backs' copies (1.66 of 4.64 and 1.87 of 4.83 million at
  the 19:14 save), every one a dynamic buffer at its own address with a GPU
  write noted on the range, bound one or two flips after the call.
  Those are ordered today by the copy versions and the GPU-write tables, in
  the command processor's order, and with the proof's rule (a GPU write noted
  on the range since the call leaves the binding to today's path) no binding
  would have bound stale bytes at either save.
- **The pass's attachments.** A colour target smaller than the draw's area is
  left out of the pass, by sizes the renderer learns in the command
  processor's order, and the pipeline key holds the pass's formats: the key's
  attachment part is the run's, taken where the run is committed.
- **Vertex and index data.** The lite mode copies the vertex records on pages
  a GX Map handed out in the last 120 flips. GX hands a renamed range out
  again only after the GPU is done with it, so a list's copy at its finish
  holds the bytes the command processor copies today; the rest are read in
  place either way.
- **Texture views.** A texture's image changes in the command processor's
  order only - an upload or copy token, a target drawn, a refresh - and GX's
  view entries carry an epoch that every such change moves
  (`BBHOST_GXVK_IMAGES`): a run checks its views' epochs.
- **Copy versions and GPU writes.** Made in order, by copy and dispatch tokens,
  which end runs. Each binding looks its range up today (2.7% of `bb-cp0`); a
  run would check its ranges once against those made since it was resolved.
- **Time.** The command processor reaches a draw one or two flips after its
  GX call (51% and 46% at ~135 fps; 22% and 70% in a run where it was slower,
  under `BBHOST_GX_COST`), so a pool that resolves a list when the immediate
  context executes it has about a frame; and every disagreement above has
  that long to happen.

## Committed per run where the command processor reaches it

Resolving a list ahead - at GX's ExecuteCommandList, a frame or two before
the command processor gets there - moves each draw away from the state it is
committed against today, so it needs a check per run of everything that may
have changed since, and arenas that live for two flips. A narrower step needs
neither. At the first draw of a run the command processor looks ahead through
the packets for the run's other draws, as the census does: GX-native draw
tokens with nothing between them that it must do in order, into the same
targets. A few worker threads then commit contiguous parts of the run, each
draw whole as `draw_impl` commits it today, while it waits; it hands the
parts to `bb-record` in order and walks on. Nothing moves in time: every draw
is committed with the state the command processor has at that point of its
walk, which the draws of a run share by the run's definition.

- **What the command processor keeps.** The walk, the in-order work, and per
  run a look ahead, a fork, a join and a splice. At the 18:11 save that is
  ~1.2 ms of in-order work a flip and the draws' ~4.3 ms divided among the
  threads that commit them, plus ~67 forks and joins: ~2.5-3 ms a flip with
  three or four threads (a draw on a worker may well take longer than on the
  command processor), under the main loop's 4.4-4.9 ms. That is the ceiling
  resolving ahead reaches there too (~200 fps), because the main loop is
  next either way.
- **The splice.** A draw's commands are a packet in `bb-record`'s ring of
  1,024 (`g_packets`, by number) and a `kDrawState` op in the stream that
  names it. The command processor takes the run's packet numbers at the fork,
  the workers fill those packets where they lie, and after the join it
  appends their ops in order: nothing is copied, and a draw it commits itself
  after a worker stopped takes its reserved number.
- **What a worker must own.** Everything else a draw allocates or appends:
  chunks of the params ring, of the GX ring's tables and of the
  constant-buffer and vertex copy rings, its own snapshot memos, a descriptor
  pool and set cache, and its counters: the draw counters are plain adds with
  one writer under `g.mu`
  (`bump(g.draws)`), and a shared atomic would bounce its line between the
  cores at every draw.
- **What must hold still for the run.** What a draw changes today on its
  way: the lite shadow's drain of GX's Map notices (`drain_gx_writes_locked`,
  inside `lite_rewritten_locked`: drained once before the fork), the imports'
  idle notes (`idle_note_use_locked`: kept per worker, applied at the join),
  lazy fields on shared objects (a program's fetch-shader slot, a pipeline's
  paths check: filled by the command processor), and `g.queued`, whose
  overflow submits in the middle of a frame (the run's slots reserved at the
  fork, or no fork).
- **Where a worker stops.** Before any side effect, at anything off the common
  path: a cache miss (programs, fetch shaders, translations, pipelines, stage
  and vertex-input plans, bindless slots), a pipeline still building, a
  texture entry due for its refresh, a target's pending clear or HTILE fill,
  copy versions to put in place (a pipeline that walks memory, an index or
  indirect range a copy straddles), a view clear's barrier, a draw's pass
  other than the run's, an occlusion query open around the run, a capture,
  the trace or another diagnostic, tessellation, a geometry shader, YEBIS
  windows. The command processor then commits that draw and the rest of its
  part itself, in order.
- **The compare mode.** Each run committed both ways, the workers' parts into
  scratch packets that are dropped, and the two compared packet by packet - a
  ring offset by the bytes behind it, a set by its writes - with
  disagreements counted by field.
- **Its costs.** The fork and join on Windows: workers spin for a few tens of
  microseconds between runs (the in-order work between two runs averages
  ~17 µs here) and park on an address after that, and on an APU every busy
  core takes from the clock the command processor and the main loop boost to,
  which only an A/B shows. The Ryzen AI Max has two core complexes, and a
  line passed between them costs ~600 ns against ~100 within one (the
  recorder's counters, `recorder.cpp`): the workers belong on the command
  processor's. And the tools that assume the command processor commits each
  draw: F12 captures and the diagnostics keep it doing so.

## The order to build it in

Each step is behind `BBHOST_LIST_RESOLVE` (0, the default, is today's path),
with a mode that resolves both ways and counts disagreements by part. The
steps below were the first plan; the measurements since change it (see "The
order now" at the end of this section).

1. **Resolve and commit, apart, on the command processor.** `draw_impl` split
   into a resolve half - from the records and the caches that only grow to a
   compact resolved draw: programs, fetch shader, vertex formats, pipeline
   key and pipeline, user data, stage plans, texture view identities, the
   constant buffers' and vertex records' ranges - and a commit half: targets
   and passes, the copy-version and GPU-write answers, ring allocations,
   snapshots, descriptor sets, the packet. Both on `bb-cp0`; the compare mode
   resolves twice and finds no difference. It must cost nothing: within the
   runs' noise.
2. **A resolve half without the GPU mutex.** The caches it reads made safe for
   readers on other threads: the programs and fetch shaders by id, the
   translations, the pipeline map (entries are never erased), the stage and
   vertex-input plans, the texture key dimensions. The resolve half allocates
   nothing.
3. **Resolved per list, ahead.** At GX's ExecuteCommandList(s), when a list
   is handed to the immediate context, a pool resolves its draws next to their
   tokens' slots; the commit takes them, and the compare mode resolves again at
   the token and counts what differs. This step alone is not expected to make
   the command processor faster - it still visits every draw and reads what
   the pool wrote, as both experiments that moved work to the GX call found -
   but it measures the disagreements at the real distance (two flips for most
   draws).
4. **Committed per run.** The pool also commits what does not depend on the
   command processor's order - descriptor sets, pushed ranges, the snapshot
   copies, into arenas of the list's own that retire with the submission that
   runs it - and builds the run's packets. The command processor checks the
   run, begins its pass, and hands `bb-record` the run's packets as one
   record; a run whose check fails is committed by the command processor as
   today, counted by reason. This is the step that takes the per-draw work
   off the command processor.

Three things the measurements change in the design note:

- **No secondary command buffers.** `bb-record` already records off the
  command processor (15-17% busy here); a list's output can be packets in its
  stream, which keeps one recording path and spares the drivers' cost of
  hundreds of secondaries a frame.
- **A run is a stretch of one list.** No run's draws came from two GX
  contexts, so a list resolves its runs alone.
- **Barrier tokens do not split a list.**

**The order now.** Steps 1 to 3 gain nothing on their own, and their purpose,
a resolve half safe off the command processor, is half of what a worker
committing whole draws needs anyway. Committing per run where the command
processor reaches it (the section above) reaches the same ceiling at Kyo's
save with neither the checks nor the two-flip arenas, so it comes first,
after batching per run on the command processor (the next section), which
needs only the run tracking and is built:

1. **The draw's own context, on the command processor alone** (built: the
   section "The draw's own context" below). The rings a draw copies into,
   taken through an object of its own in chunks, with its own snapshot memo,
   and a census of what it still takes outside one: params slots and packet
   numbers, which the run's fork reserves in step 2.
2. **Workers commit runs**, with the stops above and the compare mode, and
   alternating A/B where the command processor limits. Not built yet; in
   this order, each behind its switch and measured:
   - the run's look ahead (the tokens that follow in the command buffer, a
     token's inputs peeked in its slot for the targets), the pool, and a fork
     and join with nothing in between, to price them;
   - the reservations at the fork - params slots for the run (two a draw, the
     mid-frame submit's check passed for all of them first, the unused tail
     handed back at the join) and the run's packet numbers, whose packets the
     workers fill where they lie and whose `kDrawState` ops the command
     processor appends after the join;
   - a context per worker, with its chunks taken at the fork or under a lock
     of the rings' own, and its counters (`bump` is a plain add under
     `g.mu`) merged at the join;
   - the stops, counted by reason, and the draw and the rest of its part left
     to the command processor with their reserved numbers;
   - the compare mode: each run committed both ways, packet against packet;
   - alternating runs where the command processor limits.
3. **Resolved ahead**, with the checks per run (steps 3 and 4 above), on the
   same contexts - only once the main loop is faster than the per-run
   commit's ceiling, which at Kyo's save it is not.

## Batching per run (`BBHOST_RUN_BATCH`)

Before any draw leaves the command processor, the draws of a run can share
what they ask alike. By line at the 18:11 save (inclusive), `draw_impl`'s
targets section is 3.9% of `bb-cp0` and its pass section 4.1%, but less of
either repeats than those totals say: `target_image` already keeps each
slot's last answer (a hit applies the pending clears and the write stamp),
and `begin_pass` returns at once in the open pass. What a run's later draws
repeat is decoding each slot's registers, the memos' compares and the small
targets' check; in the pass section little is the same from draw to draw (the
sampled images' clears and layouts, the copy-version straddles). The lite
shadow drains GX's notices at each vertex record, reading a line GX's threads
write at every Map.

`BBHOST_RUN_BATCH=1` shares two of those:

- **The targets.** A GX-native draw whose target inputs - the colour targets'
  registers, extents and generations with the masks, the depth target's
  registers and whether the draw tests, writes or clears depth, the scissors
  - are the draw before's takes that draw's answers (`run_targets`, around
  what is now `draw_targets`), when nothing was done in order between them
  (the census's splits move a serial), nobody else took the GPU mutex since
  (its count of takes is one more), no target was made, resized or retired
  (`g_rt_gen`), and no HTILE fill was noted or taken (`g_pending_htile_gen`:
  a fill that meets no target stays pending for good, so the map is seldom
  empty). It applies what each lookup's hit would (`target_hit`) and keeps
  the census's counts. Tessellated and geometry-shader draws look up as
  before.
- **The drain.** Once a draw, at its first vertex record.

`BBHOST_RUN_BATCH=compare` draws as before - every target looked up, the
notices drained at each vertex record - and checks every answer a run would
have shared against the lookup: the targets, their layers and masks, the
scissor, the depth target and the blends. The exit report counts the draws
that shared, why the rest did not, and the disagreements.

At the 18:11 save, 640x360 headless:

- Of 20.79 million draws, 17.07 million (82.1%) took the draw before's
  targets; compared, none of them differed. The rest: something done
  in order since 11.0%, other targets or state 3.0%, draws that do not share
  3.9%, the mutex taken by another thread 1,118 draws in all.
- Alternating runs, in-world fps (88 s each, nothing else running): off 170.2
  and 171.4, on 175.2 and 176.7 - **+3.0%**. With the switch off the build
  runs as 81cc83bb (173.8 against 173.4, 170.0 against 170.3).
- The batched run's frames at 60 and 80 s show what the compare run's do.

Tried and dropped: handing the draw packets to `bb-record` sixteen at a time
instead of one, to save most of the hand-over (1.1% of `bb-cp0`, a store
fence among it, at every draw). Alternating runs with it were 2.2% slower
than off (168.0 and 168.3 against 173.8 and 170.0), so ~5% slower than the
sharing alone, and `bb-record` was 27% busy instead of 15%, replaying sixteen
packets written over some 60 µs rather than each as it was written.

The GPU mutex is left as it is. Holding it for a run would not save its
hand-offs - every time another thread takes it while the command processor
works costs one wake either way - and the vblank's flip waiter, which takes
it some 2,000 times a second, would wait a run each time.

## The draw's own context (`BBHOST_DRAW_CONTEXT`)

The first step of committing per run: what a draw allocates, taken through an
object of its own (`DrawCtx`, `gpu_internal.h`) as a worker committing part of
a run will take it. Three rings go through it - the GX ring's tables
(`gx_ring_place`), the constant-buffer copies' ring with its snapshot memo
(`resolve_stage_buffers`) and the vertex copies' ring (`snap_copy_locked`) -
the two that keep laps and segments being the ones a worker could not share
without a lock. A context takes a chunk of a ring the way the ring's own
allocation takes a copy of that size - round the ring when it runs out, each
segment waited for when its last lap's recording still reads it, the vertex
ring's refusal when the open recording does - and places its draws' copies
in it; its chunks go when the recording they were taken for is submitted. A
context keeps its own constant-buffer memo, so a buffer snapshotted again by
the same context's draws binds its copy as before.

Two allocations stay outside for now, because a run's worth of them is known
at the fork: params slots (two a draw, more for a table rebuild) and draw
packets (one a draw), which step 2 reserves for the run and hands back unused
at the join. Descriptor sets come from the set cache, which a worker can read
while the run lasts; a miss allocates (`set_cache_alloc`, `alloc_set_locked`)
and is a worker's stop.

`BBHOST_DRAW_CONTEXT=chunks` has the command processor take every draw's
three rings through a context of its own, and the exit report counts the
chunks, the bytes placed in them, the copies its memo bound, and - the census
step 2 starts from - what a draw still takes outside one.
`BBHOST_DRAW_CONTEXT=compare` does the same and checks every chunk against the
others its ring gave the same recording (none may overlap another or leave its
ring: with several contexts, the invariant that keeps their copies apart), and
every copy a context's memo binds against the game's bytes. Off (the default)
nothing goes through one.

At the 18:11 save, 640x360 headless, with `chunks`:

- 19.7 million draws went through the context. Per draw it placed 0.9 KiB
  of tables in the GX ring and copied 1.5 KiB of constant buffers and 1.4 KiB
  of vertex records into the copies' rings, in 338,000, 284,000 and 150,000
  chunks over the run; its memo bound 4.0 copies a draw.
- What a draw still took outside it: 2.00 params slots, 1.00 draw packet and
  0.001 descriptor sets - the set cache served the rest.
- `compare`: 138,000 chunks checked against the others their ring gave the
  same recording, none overlapping one or leaving its ring; 10.5 million
  copies bound from a context's memo checked against the game's bytes, none
  changed since its copy (reading the copies back from write-combined memory
  makes that run crawl, 29 fps).
- The frames at 60 and 80 s show what the switch-off run's do. The speed is
  the same within the machine's noise that night: six alternating runs,
  in-world 153.8, 158.3 and 149.7 fps off against 155.9, 157.7 and 143.0
  with chunks (the last with a dip; the minutes' medians 145.9, 149.9, 137.9
  against 147.9, 150.0, 143.9). By the profile it can only be a little
  faster: `gx_ring_place` was 1.6% of `bb-cp0`, half of it its `std::mutex`,
  which a chunk takes once instead of at every table.

## Workers commit runs (`BBHOST_RUN_COMMIT`)

Step 2, built: the design in "Committed per run where the command processor
reaches it", in the sub-steps of "The order now".

**The look ahead and the fork (`=fork`).** At a draw token the command
processor walks the packets after it (`gnm_exec.cpp run_following`): the draw
tokens that follow with nothing it does in order between them
(`list_split_of`, the census's own classification) and into the same targets
(`list_resolve::targets_of` of the inputs peeked in their slots,
`hle_gx_native_peek`). A run of `BBHOST_RUN_COMMIT_MIN` draws or more (8) is
forked to `BBHOST_RUN_COMMIT_THREADS` workers (3, `bb-run0..2`), which spin
`BBHOST_RUN_COMMIT_SPIN_US` (50) after a part and then sleep on the fork's
word (WaitOnAddress). `=fork` prices that with nothing for the workers to do:
at the 18:11 save, 339,272 runs forked holding 95.7% of the draws looked at,
7 µs from fork to join a run, 0.9 µs a look ahead, 89% of forks waking a
sleeping worker - and in alternating pairs 147.3/150.4 fps off against
144.0/146.1 forked (about -2.5%, three spinning cores and the wakes). The
first runs hung at the first fork: a worker that read the fork count once it
was running could find that fork counted and never join it; they start from
the count before the fork that started them.

**The commit (`=on`).** After a run's first draw - drawn by the command
processor, which looks its targets up and begins its pass - the rest of the
run is split into contiguous parts, one on the command processor and one on
each worker, while it holds the GPU mutex (`render.cpp host_gpu_run_commit`).
Each draw goes through `draw_impl` as it would in order:

- prepared from its slot as the walk's take of its token would prepare it
  (`hle_gx_native_copy`): its records unpacked on the committing thread, its
  user data as a token draw has it. A peeked token's objects still carry the
  slot's last records (the take sets them): without this a part's draws
  resolved no constant buffers and 62% of them bound their pipelines'
  fallback variants, which took the GPU from 74% to 97% busy;
- with what it takes from what the draws share reserved at the fork
  (`RunDraw`): three params slots, a packet number in the recorder's ring
  (`recorder_reserve_packets`; `DrawCmds::start_reserved` fills the packet
  where it lies and keeps it out of the stream), a draw record's number;
- its rings' space from its thread's chunks (`DrawCtx`), taken under a lock of
  their own (`g_part_ring_mu`) and only inside the segments their lap already
  holds; its recorded state of its own (`t_part_recorded`: a part's first draw
  binds everything); its counters in its part's sink (`BumpSink`, which
  `bump()` feeds on a part's thread); its targets as the run's first draw
  looked them up (`run_targets_part`, or a stop where its target inputs
  differ);
- the lookups read their memos and maps and write neither (programs,
  translations, pipelines, the set cache's front, stage plans, texture views
  by id and by words, samplers, input layouts); GX's Map notices are drained
  once at the fork; its occlusion state is "unmeasured", forked only where no
  draw would take a query.

Wherever it would change what other draws share, or do something in its
order, it stops first (`RunStop`): a program, translation, plan, pipeline,
set, texture view or sampler to make, a texture to refresh (its upload lands
before the draw's packet), a store, copies to put in place, an owed clear it
samples or its targets owe, a depth layout to change, an import to make, a
page-table walk's windows. At the join the command processor merges the
sinks and the parts' notes (their pipelines' counts, the vertex-shader
census, the lazy barriers' sampled images), appends the parts' packets in
their order up to the run's first stop, and commits every draw from there in
order itself, into the packets reserved for them: a stop's own work - a
store above all - can change what the draws after it are answered.

Measured at the 18:11 save, 640x360 headless, in
alternating pairs (60 in-world seconds' medians): 146.2, 153.0 and 155.5 fps
off against 193.2, 196.0 and 194.5 on (+28%). Off, the command processor is
98-99% busy and the GPU 73-76%; on, 95-97% and 93-94% - the GPU is the limit
with it on, even at 640x360. Of 18.0 million draws looked at, 296,022 forked
runs held 79.8%, 129 µs from fork to join a run (the parts' work included),
half of the forks waking a worker. Fullscreen at 1080p with V-Sync off, the
window kept in front (an unfocused one resolves no occlusion queries on
AMD's driver, and culls nothing), three alternating pairs: 140.4, 138.8 and
140.8 fps off (the command processor 98% busy, the GPU 81-83%) against
156.8, 157.2 and 157.3 on (82-86%, 96-97%: the GPU the limit), +12%.
Occlusion culling the same both ways where the frame rate is: capped at
120, 76.0% of the pairs occluded off and on (about 1,530 visible and 4,850
occluded a 300-flip window); uncapped, on's faster frames see each test's
result later and start fewer (about 18 pairs a flip against 20.5), the
visible pairs as many as off's. At the 120 cap the command processor is 53%
busy on against 77% off - room the APU's shared power budget gives the GPU.

**The compare mode (`=compare`).** Each forked run committed in parts, then
again in order into packets of its own; every draw the parts committed before
the run's first stop is compared with its commit in order (`RunDigest`:
pipeline, sets, dynamic state, indices, draw call, vertex input by guest
address, params blocks with their GX-ring pointers left out, constant buffers
by guest range or, in the GX ring, by their bytes), and the parts' packets
are dropped. 7,484,328 draws compared, none differing in any field. (Its
first versions told the digest's mistakes apart from the commit's: constant
buffers GX placed in its own ring, and a mask that took overlapping dword
pairs for pointers on one side only.)

**Local stops taken up again (d24f963b).** Every stop sent the rest of its
run back to the command processor, though most stops change what no other
draw is answered: a cache filled (programs, translations, plans, input
layouts), a texture refreshed at its first bind in the recording (only draws
that sample it read its upload, and they stop there too), a copy ring's
segment waited for, a params slot past the reservation, and - since the
discrete configuration below - a copy into device memory made before the
recording's draws (`kRunLocalStops`). At such a stop the join commits that
draw in order, noting each stop a part would have met in it (`run_part_stop`
on a draw committed in order records instead of stopping), and, all of it
local, takes the part up again itself from the next draw
(`run_part_resume`); the later parts stand. Anything else still has every
draw from there committed in order. At the 18:11 save: 19.6 million draws
committed in parts and 0.8 million again in order (13.9 and 3.4 million
before), 175,617 parts taken up again, 6 stopped draws that did more in order.

**The first draw's targets by generation (b18f466b).** A run whose first
draw's targets were not taken the way batching shares them - another thread
took the GPU mutex between that draw and the fork, as the vblank's fence
questions often do - was refused (22-32 thousand runs). It now forks when
nothing changed the targets meanwhile: no target made or resized, no HTILE
fill, no pending clear noted or taken (`run_targets_shared`, by the targets'
generations); 17,600 refused there.

**Targets by name (dbce71d5).** A part's draw stopped wherever its target
inputs differed from the run's first draw's, and those include the pixel
shader's export masks: two draws with different pixel shaders into the same
targets stopped (17,000 runs' first stops). A part now compares what names
and sizes the target images (`target_names`: each slot's base, format, pitch,
height, layers, extent and generation, the depth target's with its HTILE base
and clear bits, the scissors' corner) and works the rest out from its own
inputs as `draw_targets` does - channels, blends, viewport scissor, layers,
depth and stencil state, and the targets too small for the draw that its
pass leaves out (`drop_small_targets`, one function for both). No targets
stops are left; draws committed again in order 0.21 million against 0.80.
A run whose first draw took no answer a part could share (a tessellated or
geometry-shader draw ahead of plain ones) says so (-1), and the next draw
starts a run of its own. Headless 640x360: 199.7 fps, the GPU the limit.

**A discrete GPU's configuration.** Until now the fork refused anything but
an integrated GPU's paths: a discrete GPU has the buffer shadow's mirrors
(device-local copies of the vertex and index data), GX buffers bound by
handle (their device-local copies) and the particles' twins on by default -
and a discrete GPU is where the command processor limits most. A part's
draw now asks the mirrors without changing what they hold for anyone else
(`buffer_shadow.cpp part_locate_locked`): a range whose pages the recording
has checked is served from its mirror, one with a page bound from host
memory for good or written by the GPU in the recording stays in host memory,
as in order. A valid page's once-a-recording check - its write watch or its
told writes, the recording's first question about it - is made as in order
under a lock of the parts' own (`g_part_mirror_mu`), the mirror's ok bits
read and set as atomics: without it every recording's first draw over each
mesh would stop. A page to copy, a window or mirror to ask for, a record to
stage stops the draw (`kRunStopMirror`, local); a window or mirror the memory
keeper is making (650ff304: their memory made off the command processor, a
1 GiB mirror's 25-32 ms included) is answered as in order, host memory
meanwhile. GX buffers by handle: a copy
to place stops it (also local: its copy runs before the recording's draws).
The twins: the fork is refused while a drop is owed (a buffer let go or
mapped by the CPU since the last lookup), the miss filter is made current
there, and the parts' lookups only read; a candidate's twin to make, let go
or drop - which may end the pass - stops it, not locally. Still refused:
bindless sets, `BBHOST_GXVK_BUFFERS=2`, `BBHOST_GXVK_ARENA`, the mirrors read
in place (an integrated GPU's with the mirrors forced on), and
`BBHOST_SHADOW_VERIFY` and the GX census as diagnostics.

The compare mode's digest now holds each vertex binding's buffer and offset
too (but in the ring vertex copies go to, whose places differ), so a mirror
served on one side and host memory on the other differs. On the 8060S with
the configuration forced (`BBHOST_SHADOW_LITE=0 BBHOST_SHADOW_IN_PLACE=0`, and
`BBHOST_GXVK_BUFFERS=1 BBHOST_PARTICLE_TWINS=1`): 7,982,725 draws compared with
the mirrors alone and 11,357,321 with all three, none differing; the lite
default 10,115,016 with one differing in its pipeline and vertex input - its
vertex table read from guest memory by its part, and again later by the commit
in order compare mode adds after the parts (the one-off seen before, a race
with the game's own writes that only compare mode's second read can lose). On Windows an integrated GPU's mirror
windows are 1 MiB, so most of its ranges cross one and read host memory
(47 million binds from mirrors against 241 million from host memory there);
a discrete GPU's are 64 MiB, or sparse.

What is next, by what the counts say (640x360 at the 18:11 save, after
dbce71d5): of a run's first stops, a texture to refresh 138,594 and a copy
ring's segment still in use 36,871 - both local, so their parts go on - and
programs, pipelines, images and sets in the hundreds; runs refused because
the first draw's targets were not shared 32,028. With the GPU the limit, the
next frame rate is the GPU's: the G-buffer pass (seven targets and depth at
1920x1080, 14.7 million vertices, 4.9 million primitives and 4.8 million
fragments a frame - a fragment and a bit a triangle) is 2.0-2.6 of the
frame's 5.6 ms (BBHOST_GPU_PROFILE=3), 1.9 ms of it at 720p as well: the
vertices, not the pixels.

**The vertex data in video memory, with the command processor freed.** With
the commit on, the 8060S's frame is the GPU's, and its G-buffer pass reads the
meshes from the game's memory where GX left them - host memory, imported: an
integrated GPU keeps GX buffers on today's path (its video memory being the
host's own) and the buffer shadow's lite path. Bound from their device-local
copies by handle instead (`BBHOST_GXVK_BUFFERS=1`, a discrete GPU's default:
86.5% of the vertex records and 92% of the index ranges, 256 MiB of copies),
the same frames take less of the GPU. Fullscreen 1080p, V-Sync off, the
window in front, the run commit on: 164.9 and 164.8 fps against 157.9 and
157.7 with the lite default; the mirrors on top (the discrete GPU's whole
configuration) 164.1 and 163.9 - nothing more. Capped at 120: the GPU 67%
busy against 70%, occlusion culling the same (77.6% and 77.1% of the pairs
occluded, 52,645 and 52,666 visible). Headless 640x360: 213.9-218.1 against
200.9. By pass (BBHOST_GPU_PROFILE=3, 1080p headless, in order): 7.27-7.41
ms of GPU a flip against 7.88-7.97, the G-buffer pass 3.54 ms against 3.85.
Frames at 60 and 90 s alike (0.6-0.9% of the pixels apart, two runs of one
configuration 0.6%).

**The in-order path, and PGO.** With every switch off the draw path is
today's plus the steps' checks and code, and costs for it: without PGO, at the
18:11 save headless, 158.6-163.1 fps at 81cc83bb against 152.9-155.0 at
df04b282, the loss spread over the steps (the draw context's 160.0, the
fork's 157.6, the commit's 153.6-158.0). Sampled on the command processor
(BBHOST_SAMPLE, df04b282 against 81cc83bb), about 2% of it was the switches'
thread-local reads - bump() no longer inlined everywhere 1.0% of its time,
the TLS wrapper's __tls_init 0.6%, run_part() 0.6% - which 8f8639d5 takes
out of the off path (constinit, and read only with a switch on; one A/B pair
could not tell it from the noise, 151.2 and 155.0), and the rest is where
the inliner now leaves things out of line (draw_targets, 3.7% inclusive). A PGO build loses
far more, unless it is profiled with the switch off alone: kyo/combined with
this merged and profiled with the run commit on, or on and off, measured
161.1-165.6 fps off against the deployed build's 181.4-187.0 (the profile's
runs with the commit on make the in-order path less hot to the compiler);
profiled off only (two runs: 640x360 at the frozen save, 1080p with a walk
on the live save), 182.4 and 177.4 off against 182.0 and 184.5, and 200.9 on.

## What this step built

- `BBHOST_LIST_RESOLVE=runs`: the census above, logged every 1,000 flips with
  tooling on and for the whole run at exit. Off by default; with it off the
  command processor takes one more predictable branch a packet.
- `BBHOST_RUN_BATCH=1|compare`: the batching above, on the census's run
  tracking (`list_resolve::run_serial`, which the per-run commit will start
  from). On with the run commit (so by default since cc9a46d1); `=0` off.
- `BBHOST_DRAW_CONTEXT=chunks|compare`: the draw's own context above, with its
  census and checks. Off by default.
- `BBHOST_RUN_COMMIT=fork|on|compare` (`BBHOST_RUN_COMMIT_THREADS`,
  `_MIN`, `_SPIN_US`): the runs committed in parts above, in an integrated
  GPU's configuration (the lite shadow) and a discrete one's (the mirrors, GX
  buffers by handle, the particles' twins). **On by default since cc9a46d1
  (2026-10-29)**; `BBHOST_RUN_COMMIT=0` (or `off`) commits every draw in order,
  the draw path then taking the same branches as before.
- With it, `BBHOST_GXVK_BUFFERS` defaults to on for an integrated GPU with
  8 GiB of video memory or more (`host_gpu_integrated_with_room`), as for a
  discrete GPU with room: the section above on the vertex data in video
  memory. A smaller integrated GPU keeps today's path.

A/B, 100-s runs, medians of 60 in-world seconds: base (81cc83bb) 130.1 fps;
the switch off 130.1 (its mean dipped while a build ran beside it); the
census on 134.9. The series' second half (on, off, base) ran two hours later
at the 19:14 save, where the main loop limits: in-world means 285.1, 296.7
and 286.3 - the census costs nothing there, which says nothing about the
command processor. A clean alternating series at the 18:11 save is still to
run: another session's test games, which take no GPU lock, overlapped every
run tried that evening.
