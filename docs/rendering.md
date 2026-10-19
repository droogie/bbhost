# Rendering

## The problem

Bloodborne's renderer was written for the PS4's GPU, an AMD GCN part. The game
draws through FromSoftware's graphics layer, GX, which sits on Sony's Gnm
library: GX turns the engine's draws into PM4 command buffers - the packet
format the GPU's command processor reads - and the game ships its shaders as
GCN machine code. A PC has neither a GCN command processor nor a way to run
GCN shaders directly.

An emulator would answer this at the lowest level: decode every packet,
keep the GPU's registers, and reconstruct each draw from them. bbhost started
there and is moving up a level, to where the engine's intent is still visible.
Dark Souls III on PC, built on the same engine, puts Direct3D 11 under the
same GX layer; bbhost does the equivalent with Vulkan.

## How a frame is drawn today

**GX hooks.** bbhost hooks GX's draw, dispatch, resource-creation and state
methods. When the game draws, bbhost records the draw's inputs from the
engine's own objects - its shaders, vertex layout, state descriptions,
textures and constant buffers - and leaves a small token in the command
buffer. When the command processor reaches the token, the draw is built from
those inputs. No packet is decoded and no register file is consulted for a
draw. At Uncapped, 90 and 120, where the command processor limits the frame
rate, the draw's resource records travel packed - each stage's header and
only the slots it sets, a few hundred bytes - and the command processor
unpacks them into a set of its own; at 30 and 60 the GX threads hand over the
whole sparse set, which costs them less CPU, and a paced frame does not wait
for the command processor (`BBHOST_GX_PACKED_RECORDS=1` or `=0` forces
either).

**Compute dispatches** are tokens too. A dispatch through GX's compute
commit - the GPU particles, the tone-map and depth-of-field look-up tables -
takes its compute stage from the engine's objects at the call, as a draw does:
the compute shader, its views, UAVs, samplers and constant buffers, from which
the command processor builds the shader's user data, tables included - and,
for the particles, each UAV's hidden counter: its word and the initial count
the game set, as D3D11's `pUAVInitialCounts` (`BBHOST_GX_TOKEN_COUNTERS`). YEBIS's
passes are built from YEBIS's own state, as its draws are, and need no constant
engine (`BBHOST_GX_YEBIS_DISPATCH`); GX's own resource passes take the user data
their writers set. A dispatch the hooks did not take would still run from its
packet and the register file (`BBHOST_GX_NATIVE_DISPATCH=0` runs them all that
way), and the exit report counts those by reason.

**The command processor.** bbhost still walks the game's command buffers,
and mostly to take the tokens in their order. What else it does is the
completions the game waits on - the labels GX's block pools, its command arena,
Map and its queries' results, YEBIS's flush counter and Scaleform's fences are
written with, which it records on the GPU's timeline - and a few waits and
constant-engine writes outside GX (YEBIS's own drains, the flip's, Gnmx's
constant update engine). GX's pipeline drains are barrier tokens: the CP's own
writes after one go behind the work before it, with no label and no wait.
Occlusion queries are tokens from GX's Begin and End, measured by the
renderer's own queries. The packets nothing on the host reads - the cache
flushes, the context registers, the shader prefetches, the compute stage's
registers (a dispatch token carries them, and a fallback writes them before
its packet), the constant engine's counters - are not written at all
(`BBHOST_CP_DEAD_PACKETS`). Clears and copies come as tokens from GX's own
calls: a clear names its view, and is made as the next pass's load op, or as a
transfer clear when something reads the image first.

**Host command lists.** What the command processor acts on in those buffers -
the tokens, labels and waits, which our versions of Gnm's packet writers write -
can be recorded as it is written, a list per chunk of command buffer, so that a
submitted range reaches the command processor as its list's packets instead
of all its bytes (`BBHOST_HOST_LISTS=1`, the default; a range no list covers
is walked, counted by reason; `=0` walks every range as before, `=2` records
the lists and checks them against the walk, range by range).

**Labels.** A submission carries its labels, and a thread of the host's,
`bb-labels`, writes them into guest memory in the order the command processor
made them once the timeline semaphore says the submission has run: the
queue's order holds without barriers between fills, and a label's memory
needs no import. A submission counts as completed for the rest of bbhost only
once its labels are in memory, and the command arena's acquire, waiting for a
chunk, looks again as each label lands. The thread waits in the driver, or
polls on Windows (`BBHOST_LABEL_WAIT`), where NVIDIA's driver holds its lock
through that wait; there the labels are still GPU fills at the end of their
submission's command buffer unless `BBHOST_LABEL_LANDING=1` (`=0` keeps the
fills anywhere).

**Recording.** The command processor leaves the Vulkan calls to a thread of
its own, `bb-record`. Each draw goes into a packet, and most of its other
commands - barriers, fills, copies and the like - into the same ordered
stream as typed records (`src/host/recorder.cpp`, `src/host/stream_ops.h`).
`bb-record` replays the stream into command buffers, begins and ends each one
and hands it to the thread that submits, while the command processor goes on
to the next draw. With `BBHOST_STREAM_SUBMIT=0` the command processor begins,
ends and submits the command buffers itself, to compare. The submitting
thread, `bb-submit`, gives the queue the game's submissions as they come; when
it falls behind - where the kernel's half of a submission is costly, as on a
Steam Deck, whose driver makes every import of the game's memory resident
again at each one - the submissions queued behind each other go to the queue
as one (`BBHOST_SUBMIT_MERGE=1` keeps them apart, to compare).

**Shaders.** GCN shaders are converted to SPIR-V when the game creates them,
and GX creates every shader of the game's bundles at the start (1,663 vertex
and 1,346 pixel programs; all but the debug bundle's), so their stages and
pipeline libraries are compiled behind the title. What a draw still has to
make itself is a variant creation does not know of - most of them the
pixel shader of a pair that clips by a varying - and the eboot's own
programs (YEBIS's, Scaleform's: the eboot carries some 5,400, a tour draws
about 80). Every pipeline a draw creates is recorded in
`pipeline-manifest.bin` with what built it, and the next start builds them
all again on the precompile workers, its optimized link included, so what
an earlier run has drawn is not compiled again when it comes into view
(`tools/pipeline_list.py` makes a list of them that names the programs
instead of holding them). There are two compilers:

- the *translator* (`src/gcn/translate.cpp`) reproduces the GCN program's
  execution faithfully - one invocation per lane, with the execution mask as
  data - and handles every shader the game uses;
- the *lifter* (`src/gcn/lift.cpp`) decompiles a shader into typed,
  structured SPIR-V, the form a person could read and edit. It is used for
  every pixel and compute shader, and every vertex shader on vertex input, it
  accepts - with the vertex formats read from the shader's parameters, or as
  specialisation constants, the way the translator reads them; the rest keep
  the translation. `tools/lift_verify.py` compares the two compilers' output
  byte for byte.

All 13,394 vertex, pixel and compute shaders in the game's bundles translate
to valid SPIR-V (`gcn2spv --check`). Domain, hull and local shaders run on the
host GPU's own tessellation stages; no geometry or export shader has been seen
in use. The lifter takes those too: the particles' local, hull and domain
shaders, whose one control point travels as stage attributes, and the game's
own hull draws (the Forbidden Woods' meshes, every hull in the bundles), whose
local, hull and domain shaders become one native pipeline - the LDS they pass
patches through turned into stage inputs and outputs, the game's patch list
drawn as it is, with no compute pass in front. `BBHOST_LIFT_TESS=0` keeps every
tessellation stage translated.

A compute shader is lifted against its dispatch's translation and binds
exactly what that binds - the parameter block with the page table, the images
and samplers, the particles' counter words - so a dispatch is set up the same
way whichever runs. Its memory reads and writes go through the same
page-table walk. GCN's loops that run until every lane's execution bit
clears become structured per-invocation loops, a block GCN branches over
where no lane needs it becomes a selection an invocation skips where its own
bit is clear, and the GPU particles' wave-wide counter idiom (`ds_append`/`ds_consume` in GDS, each lane's rank
from `v_mbcnt`) becomes one atomic add per invocation, on the counter word
the dispatch hands over by its device address, so no compute lift needs
subgroup operations or the page table for it. A barrier is accepted only where every invocation
of the workgroup provably reaches it, and LDS, which no game compute shader
uses, is refused. `BBHOST_LIFT_COMPUTE=0` keeps every compute shader
translated. A compute shader reads its constant buffers, tables and typed
loads, and reads and writes the elements of its particle buffers (typed
loads and stores at an index or offset, format loads, atomic adds; GCN's
bounds checks kept), through storage buffers its dispatch pushes, where
every one of them binds, rather than through the page-table walk
(`BBHOST_CS_BUFFERS=0`: the walk; `BBHOST_CS_BOUND_ELEMENTS=0`: the elements
walk): a translated one as that variant of its translation, a lifted one as
the lift of that translation, which reads what the translation binds from the
same buffers and walks what it walks (`BBHOST_LIFT_CS_BUFFERS=0`: a lifted
shader walks). `tools/lift_verify.py --stage cs --lean` replays dispatch
captures in that form, both the translation and its lift against the walking
translation, with their buffers bound from the captured pages.

**Textures and memory.** The game's memory is shared with the GPU, as on the
PS4: its GPU-visible parts (about 5 GB) are imported into Vulkan, because
shaders that cannot be fully resolved read memory through a page table. Over
the 12-place tour the GPU touches about half of that, and only a few hundred
MB in place; the rest it reads only as the source of a copy (texture
detiling, the buffer shadow's mirrors). `BBHOST_IMPORT_WINDOWS=1` imports
only what is used, a 16 MB window at a time when something first needs it or
when the engine makes a resource there (on a thread of its own); an import
over several windows replaces the ones it covers, so no page is imported
twice, and a dispatch imports what its shader reads and stores. It is off by
default (`gpu.import_windows` in the config turns it on too): the work under
way moves each of those uses off guest memory, and windows are then only the
fallback for what is not moved yet. The Steam Deck kit turns it on: there the
memory shared with the GPU stays under the driver's budget and the start is
about 3 s sooner, at the same frame rate (its device-local twins, below, stay
off: on an APU they are the same memory plus copies). One such move,
off while it is measured: `BBHOST_GX_TWINS=1` keeps the GX buffers the GPU
writes - particle state, append buffers and their counters, arguments a
dispatch fills - in device memory, as a copy made at the buffer's first GPU
use that every GPU use reads and writes from then on; nothing copies their
guest bytes on the CPU after that, and anything that would read them from
guest memory (a page-table walk, a texture over one) has the copy written
back first. `BBHOST_GX_TWINS=2` adds the buffers GX writes with GPU copies:
a dynamic buffer's renamed bytes written back to its own address, a constant
buffer's update (a world has about 3,800 of them, 4 MB, kept in shared
device-memory blocks). With window imports a copy's source the CPU wrote is
copied through upload staging, as copy versions' bytes are, rather than read
through an import. The exit report lists what still reads or writes guest
memory in place, by the path that asked.
Textures are detiled on the GPU. A texture's image, a render target's and a
copy's belong to the GX resource they were made for, by that resource's
generation: GX's own calls change them - the resource's creation, upload and
copy tokens, the shaders that write it - and they are retired when the
resource is destroyed, so memory the game gives to a new resource with texels
of its own never hands it the old image. A target, or an image only shaders
wrote, goes instead to a texture the game makes over the same memory without
data, as the console's memory carries it over: the game re-creates its targets
at an area load and may only clear and copy from them. YEBIS's and
Scaleform's own textures are known the same way, from those libraries' own
creation and release calls, and their writes from their own Map, update and
unmap calls. The device-local copies of vertex and index data are told what
changed by GX's calls as well: the renamed copy a Map hands out is copied
again until GX writes it back or frees it, and a creation with data marks the
memory it fills. Only memory no call tells of - the engine's own rings outside
GX, YEBIS's buffers - is still noticed by write-protecting its pages.
A draw names its targets' resource generations, so the image at a target's
base is the drawing resource's from its first draw; copies and texture binds
take a target there only if it is their resource's - drawn since the
resource was made, or the resource made over it without data. A whole
texture copied from one a shader or a copy wrote is copied image to image,
so the copy carries what the GPU wrote (`BBHOST_COPY_BY_RESOURCE=1` copies
the memory under it instead, which the GPU's writes never reach; `=0` goes by
addresses).
Render targets are Vulkan images sized from the engine's own views, and can be
re-created at any resolution while the game runs, the way the PC engine's GX
device resizes.

**Native passes.** Several parts of the frame no longer run the game's GPU
path at all:

- YEBIS, the post-processing library (bloom, glare, depth of field, tone
  mapping): every stage's resources and constants are built by bbhost.
- Scaleform, the menu renderer: its draws are taken from its hardware
  abstraction layer.
- Depth snapshots, the on-screen overlay and FSR 1 upscaling are bbhost's
  own.

## Native or emulated

| Operation | Today |
|---|---|
| Draws | native: built from the engine's objects at a token |
| Tessellated draws | native: host tessellation stages; the game's hull draws one lifted pipeline each |
| Post-processing (YEBIS) | native resource building; the game's shaders |
| Menus (Scaleform) | native draw building from the HAL |
| Fills and copies without a shader | recognised and performed directly |
| Other compute dispatches | native: built at a token from the engine's objects (GX's compute stage, the particles' counters with it; YEBIS's state) or the user data its writers set (GX's own passes); the game's shaders |
| Clears and target copies | native: tokens from GX's calls, in the view's terms |
| Occlusion queries | native: tokens from GX's calls, measured by the renderer's own queries |
| Pipeline drains | native: barrier tokens from GX's calls |
| Labels and fences the game waits on | emulated: recorded by the command processor, written by the host once their submission has run (GPU fills on Windows for now) |
| Shaders | decompiled (lifted) to SPIR-V; a few refused programs translated (see below) |
| Resource identity | GX's, YEBIS's and Scaleform's textures by their objects and generations; vertex and index copies told by GX's Map calls; memory no call tells of (YEBIS's buffers, the engine's own rings) by address, with write tracking |

## Shader lifting

Every shader the game creates goes through the lifter. One it refuses is translated instead, so nothing is lost. The program counts below cover every shader the game ships. The draw shares were measured at the twelve places `tools/area_check.py` visits; elsewhere they are expected, not measured.

| Shader stage | Programs lifted | Draws lifted (12 places) | Left |
|---|---|---|---|
| Pixel | 3,206 of 3,209 | 100% | FXAA3HQ, which needs a whole-wave proof; two debug shaders whose result depends on how the PS4 packs pixels into waves (no exact PC form). None is drawn at the 12 places. |
| Vertex | all 10,116 | 100% | - |
| Tessellation: local, hull, domain | all; the 141 mesh hull sets as native pipelines | 100% | - |
| Compute | all 38 in the bundles, and the 6 of the executable's own (GX's and YEBIS's) that the 12 places dispatch | 100% of dispatches | The executable carries 24 more, lifted when first dispatched; none is dispatched at the 12 places. |

A draw whose setup the lifter hasn't seen still falls back to the translator. The exit report counts those by reason, and `BBHOST_LIFT_TESS` and the other switches turn the lifted paths off for comparison.

| Step | State |
|---|---|
| Pixel and vertex shaders lifted | done |
| Tessellation lifted, mesh hulls native | done |
| Compute shaders lifted | done |
| HLSL: the lifted SPIR-V printed as HLSL and compiled with DXC to SPIR-V for Vulkan on the player's machine | on by default, compiled behind the game (below) |
| Shaders as editable source, generated on the player's machine from their own game files (never shipped) | a first slice: edited HLSL replaces a stage's module (below) |
| D3D12 (Shader Model 6) from the same HLSL, if ever wanted; D3D11 is not planned, since it lacks the quad operations 703 programs use | later, optional |

## HLSL

Every lift is printed as HLSL (`src/gcn/hlsl.cpp`) and compiled by DXC to
SPIR-V for Vulkan on the player's machine, and the stage takes DXC's module
where it declares the lift's interface and keeps its operations; anything
else keeps the lift's SPIR-V, counted by reason in the exit report. This is
the default, and nothing waits on DXC:

- **A start uses what is ready.** A stage takes DXC's module when the
  translation cache has it, and the lift's otherwise. The first start after
  an install or an update is the lift's throughout.
- **DXC compiles behind the game.** The modules a start did not have are
  compiled by `bb-hlsl` threads (two; one on a machine with fewer than eight
  hardware threads; `BBHOST_HLSL_THREADS`) at the lowest priority, after the
  start's own shader compiles, and kept in the translation cache. On an
  integrated GPU, which shares the CPU's chip and power budget (the Steam
  Deck), they compile only before the first in-game frame and on loading
  screens: beside the world they cost the Deck ~2 fps and a long frame almost
  every second (`BBHOST_HLSL_IN_WORLD=1` lets them run in the world on any
  GPU, `=0` holds them there on any). Their order
  follows the pipeline manifest: what this start's draws took, then what the
  manifests replay (what earlier starts drew, the most recent first), then
  what GX created but nothing drew yet, then predictions and the opt-in
  sweep. A stage a draw takes moves to the front.
- **The switch is at the next start.** A pipeline is never rebuilt live to
  swap a module: the frames are the same either way, and a rebuild would be
  a driver compile in the middle of play. The next start takes DXC's
  modules, and its precompile builds their pipelines from the manifests
  before the world is drawn.
- **The off switch.** The Shaders setting (F10, under Graphics; HLSL or
  Lifted; the game's System > PC Graphics > HLSL Shaders, On or Off;
  `video.shaders` in `bbhost.toml`) applies at the next start.
  `BBHOST_HLSL=0` is the lift alone, `=1` compiles each module at its first
  use (the start waits for DXC: for checks), `=2` is the default; the
  variable wins over the setting.
- **Compiling up front.** The setup window's Shaders tab shows whether the
  caches were made by this bbhost's translator and for this GPU and driver,
  what the last compile did, and has **Compile shaders now**: bbhost starts
  itself again with `--compile-shaders`, a run with no game and no window
  that starts the GPU and compiles every shader in the game's bundles (the
  debug menu's aside) and every stage and pipeline the manifests name - the
  player's earlier sessions' and the list a release ships beside bbhost
  (`pipeline-list.bin`, [running.md](running.md)) - translations, lifts, DXC's
  modules and the driver's pipelines - into the same caches, on the game's own
  precompile workers. Progress (N of M, the time left; the manifests' and the
  list's pipelines counted in it and on a line of their own), failures by
  reason and the HLSL counts show as it runs, and it ends when every job has;
  Cancel keeps what is done; the window stays live and Play waits for it.
  When the caches are missing, or were made by another bbhost's translator or
  for another GPU or driver, the window recommends it - on the Shaders tab and
  in a line on the main tab, each with its button and the time it is likely
  to take (the last compile's, else a guess: about 2-3 minutes on a Steam
  Deck); it never starts by itself. The run logs to
  `<data>/bbhost/shader-compile.log` and records its result in
  `<data>/bbhost/shader-compile.txt`. On the RTX 4070 it takes ~60 s (3,281
  shaders, 3,682 DXC modules, 16 workers), on a laptop's RTX 3070 Ti (Ryzen 7
  6800H, Windows) ~90 s and on the Steam Deck ~150 s. What it does not do is
  a first session's first sight: the variants a draw makes (clip-discard pixel
  shaders, vertex layouts, the eboot's own programs, the tessellation LS
  pipelines) come from the pipeline manifest, which only earlier sessions
  fill; a first tour after it built as many pipelines on arrival as a cold
  start (25 vs 27-31). Its gain on a new install is a first session already on
  DXC's modules with no compiling behind it (CPU ~385% against ~470%).
  Skipping it costs nothing: the background path does the same over the first
  sessions.
- **Edited shaders.** `BBHOST_HLSL_DUMP=1` writes each module's HLSL to
  `<data>/bbhost/hlsl-dump/` (or `BBHOST_HLSL_DUMP=<dir>`), and an edited copy
  in `<data>/bbhost/shaders/` replaces that stage's module when it compiles,
  declares the lift's interface and validates ([modding.md](modding.md#shaders)).
  The source is generated from the player's own game files and stays on
  their machine; none is committed or shipped.

- **What the HLSL is.** The lifter's SPIR-V stays the shader's IR of record:
  typed values in structured control flow. The printer reads it back and
  writes one statement an operation, values named by their SPIR-V ids,
  resources and stage variables by the engine's names, and constant-buffer
  loads annotated with the member they read
  (`// cbSceneParam.VC_MatrixViewProj[0].x`). Constant buffers are
  `ByteAddressBuffer`s read a dword at a time, as the lift reads them.
- **Exact by construction.** Every operation is spelled the way DXC lowers
  back to the lift's instruction: floats are `precise` (NoContraction); where
  HLSL's spelling is another instruction - `sign`, `ldexp`, `mad` (which DXC
  makes a fused multiply-add a driver must not split), `fmin`/`fmax`, the
  half packing, bitfield extracts, carries, borrows and wide multiplies - the
  operation is inline SPIR-V (spelled out instead, the wide multiplies made
  modules the NVIDIA driver's compiler refused); the
  helper-lane query is the built-in, which keeps `discard` an `OpKill`; quad
  operations are `QuadReadLaneAt`/`QuadReadAcross*`; an invariant position
  (`BBHOST_VS_INVARIANT=1`) keeps its decoration. Each compiled module is
  checked against its lift: the same descriptor bindings, stage interface
  and specialization constants, and no float, image, quad or discard
  operation the lift does not have.
- **Tessellation and compute.** The LS is a vertex shader with an array
  output, the domain shader takes its control points as an `OutputPatch`,
  and the game's hull - one body each control point runs, a barrier, then the
  patch's factors - becomes HLSL's control-point and patch-constant
  functions from that one body. Compute lifts print too (raw addresses as
  `vk::RawBufferLoad`/`Store`, their counters as `InterlockedAdd` on a
  `vk::BufferPointer`).
- **Offline checks.** `gcn2hlsl` lifts every program in the bundles as the
  renderer does, prints, compiles and validates it: 13,318 of 13,318 vertex
  and pixel lifts, 450 of 450 tessellation stages, and the compute lifts;
  with `--runtime-offsets`, as on a device with maintenance8, 89 pixel
  shaders sample with a run-time offset, and those compile too.
  `hlsl_equiv` runs each lift and its DXC module on the same random inputs on
  lavapipe and compares the outputs bit for bit: 13,308 of the 13,318 vertex
  and pixel lifts identical, 10 differing only in the bits of NaNs (which
  Vulkan leaves to the implementation), none otherwise; the 89 with run-time
  offsets identical; all 450 tessellation stages identical (a domain shader
  behind a generated hull, a hull with its outputs stored from inside it,
  since random constants give factors that cull every patch).
  `tools/hlsl_verify.py` replays captured draws and dispatches with DXC's
  stage against the lift's: identical on lavapipe and on an RTX 4070, and
  `hlsl_equiv --compile` asks a device's compiler to build every compute
  module.
- **In game.** At the title screen, every lift through DXC, the frames equal
  the direct path's except the animated menu highlight, which differs between
  any two runs. The default, on twelve places of a save that reaches the Old
  Hunters areas (an RTX 4070 at 60 fps; two rounds against the lift alone):
  - *A first start* (no caches) draws with the lift's modules throughout,
    and its first-sight hitches are the lift's. Meanwhile two `bb-hlsl`
    threads compile ~3,360 modules: ~435 s of their time, all done ~250 s
    after the first was queued (~60 s of that waiting for the start's own
    compiles). The process's CPU is ~470% against ~385% while they run, and
    its memory up to ~0.6 GiB higher; when the queue empties ~0.4 GiB of
    DXC's heap goes back to the system (glibc's `malloc_trim`).
  - *The next start* takes DXC's module for every stage met before (~3,380;
    a few new ones keep the lift and compile behind). The pipeline manifest
    builds their pipelines at the start, though the driver has never seen
    those modules, and no pipeline is built during play.
  - *From then on* every stage is DXC's: every frame within the runs' own
    noise at all twelve places (24 of 24), the frame rate the same, nothing
    compiled. Against the same build with `BBHOST_HLSL=0` the GPU is busy
    2-5% less at five places (DXC's optimized code) and more at none.
  Compute dispatches take DXC's modules too.
  - *A Windows laptop* (RTX 3070 Ti Laptop GPU): after Compile shaders now,
    every start at 60 fps with no stall after the first in-game frame, the GPU
    busy ~4.5% less than with the lift (409-415 against 429-430 ms/s), the
    frames within the runs' own noise; its driver builds every lift and every
    DXC module of the game (`gcn2hlsl --sweep`: none refused).
  - *The Steam Deck*: after Compile shaders now, 60 fps as with the lift, and
    GPU-bound (uncapped) 66.6-67.1 fps against the lift's 66.5-66.9. A first
    session's compiles beside the world cost it ~2 fps and a long frame almost
    every second, hence the hold on an integrated GPU (above): with it a first
    session's world holds 60.0 fps (p95 16.8 ms, as the lift's own first
    start does), and an install that skips Compile shaders now takes DXC's
    modules a few hundred at a time, compiled over the title, the menus and
    the loading screens of its next starts.
- **Halves on a device that rounds them itself.** GCN's
  `v_cvt_pkrtz_f16_f32` rounds toward zero. On a device that converts that
  way inside a module and keeps f16 denormals (RADV, so the Steam Deck) the
  lift uses the device's conversion, two operations a pair instead of ~6
  integer operations a component. `BBHOST_HALF_NATIVE=0` keeps the integer
  form, which a driver cannot fold (RADV's compiler drops some f32-to-f16-to-f32
  round trips the native form leaves to it); on the Deck it cost ~1.8% of the
  frame rate GPU-bound (65.3-66.0 against 66.6-67.1 fps uncapped), so the
  device's own rounding stays the default.
- **A driver that refuses DXC's module.** A pipeline the driver will not
  build from DXC's modules is built again from the lifts' own: a graphics
  pipeline whole or from its libraries, the native hulls' included, and a
  compute pipeline from the lift's module and then from the translation. No
  draw or dispatch is lost to the switch. A stage refused once takes its
  lift's module at once in the pipelines after. The exit report's HLSL line
  counts the refusals by what was being built; `BBHOST_TEST_HLSL_REFUSE=1`
  makes every such pipeline a refusal, to run the path on a driver that
  refuses none (the RTX 4070 here builds every module since the carries and
  wide multiplies were spelled out).
- **DXC.** bbhost loads libdxcompiler (`dxcompiler.dll` on Windows) at run
  time - the one `BBHOST_HLSL_DXC` names, else the one beside the executable,
  which every package carries (DXC's v1.9.2602 release, pinned and checked by
  `tools/fetch_dxc.sh`), else the system's - and needs nothing else of it (not
  `dxil.dll`). A library that does not compile to SPIR-V is passed over;
  without one the switch keeps every lift, counted as "DXC not found". A
  module costs DXC ~65 ms median (-O3, eight at a time on this machine), once:
  the result is kept in the translation cache. A sample's run-time offset,
  which Vulkan allows with maintenance8, is one thing DXC's own validator
  refuses with no option to allow it: such a module is compiled without that
  validator and validated by bbhost with SPIRV-Tools, with the offset allowed
  (the Windows build has SPIRV-Tools for it too; a build without keeps the
  lift).
- **Drivers.** `gcn2hlsl --sweep --app0 <game folder>` lifts every program as
  bbhost does on the machine's GPU, compiles it through HLSL, and has that
  GPU's driver build both the lift's module and DXC's into the pipelines the
  renderer makes of the stage (its graphics pipeline libraries, then a fast
  and an optimized link; a game hull's three stages as one native hull
  pipeline), printing what it built and refused by stage and every DXC module
  refused where the lift was built - the stages the in-game fallback would
  give back the lift. The pipeline state the modules do not say (target and
  vertex formats, the neighbouring stages) is made from their interfaces.
  On lavapipe with the validation layer and on an RTX 4070 every one of the
  13,858 lifts and its DXC module was built (171 hull sets among them).
  `tools/package_hlsl_sweep.sh` makes a kit that runs it on another machine.

## What comes next

- The completions the game waits on (GX's block pools and command arena, Map,
  query results) from the host's own submission timeline rather than labels,
  after which the PM4 command processor only orders work.
- Guest memory off the GPU, the way the PC engine's resources are API objects
  filled by copies: copy sources through staging, constants and rewritten
  vertex data copied as recorded, compute shaders' buffers bound, the buffers
  the GPU writes held by the host; the whole import then goes.
- Editable source beyond replacing one stage's HLSL: edits read again while
  the game runs.

## Tools

| Tool | What it does |
|---|---|
| `gcn2spv` | translates and validates every shader in the game's bundles |
| `gcn2hlsl` | lifts every shader in the bundles, prints it as HLSL, compiles it with DXC and checks the module against the lift (`--tess` for the tessellation stages; `--spv` for one module); `--sweep` has a GPU's driver build every lift and DXC module as the renderer would |
| `hlsl_equiv` | each lift and its DXC module on the same random inputs (on lavapipe), outputs compared bit for bit: pixel, vertex, LS, hull and domain shaders; `--compile` has a device's compiler build each compute module |
| `tools/hlsl_verify.py` | captured draws and dispatches replayed with DXC's stage against the lift's |
| `gcndis` | disassembles GCN programs |
| F12 in game | dumps the frame, every render target and the draw list into a folder of its own (`tools/f12_check.py` reads it; `BBHOST_F12_PNG=1` for PNG) |
| `BBHOST_CAPTURE_DRAW` + `drawreplay` | captures one draw with its inputs and replays it outside the game; `BBHOST_CAPTURE_DISPATCH` does the same for a compute dispatch |
| `tools/area_check.py` | twelve places in the game measured the same way every time - frame times, GPU time, draws by how their pixel shader ran, compiles, frames and lift checks - and two builds compared against their own noise ([area-check.md](area-check.md)) |
| `BBHOST_VK_VALIDATE=1` | runs with the Vulkan validation layer |
| `BBHOST_GPU_PROFILE=1` | per-pass GPU timings |
| `bbhost --stream-selftest[=N]` | checks the command stream headless against a CPU model, N rounds (16 by default), and exits with 0 when it matched; run it with `BBHOST_VK_VALIDATE=1` too |
