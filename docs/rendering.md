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
its packet), the counter steps between the constant and draw engines - are
not written at all (`BBHOST_CP_DEAD_PACKETS`). The constant engine's wait is:
the game writes a NOP over it when a draw leaves the engine unused, so a
writer whose packet its caller rewinds over or edits cannot write nothing. Clears and copies come as tokens from GX's own
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
fills anywhere). A label meant for a command buffer the game has begun
writing again since the label's submission - YEBIS cycles two command buffers
and reuses one without waiting for the GPU, so a GPU running behind lands the
old round's fences on the new round's packets - is counted and not written:
what it was written for is gone (`BBHOST_LABEL_REUSED_DROP=0` writes it
anyway). As GPU fills, only those found reused when the fill is recorded can
be left out. The chunk rounds this needs come from the host command lists, and
the exit report counts what was found (and the CP what it lost: a wait no
value could pass, a token without its magic).

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
host GPU's own tessellation stages, and the GPU particles' export and
geometry shaders on its vertex and geometry stages. The lifter takes those
too: the particles' local, hull and domain
shaders, whose one control point travels as stage attributes, and the game's
own hull draws (the Forbidden Woods' meshes, every hull in the bundles), whose
local, hull and domain shaders become one native pipeline - the LDS they pass
patches through turned into stage inputs and outputs, the game's patch list
drawn as it is, with no compute pass in front. `BBHOST_LIFT_TESS=0` keeps every
tessellation stage translated.

The GPU particles - the effects' standard particles, their sparks and the
wandering visions - draw a point a particle through an export shader, a
geometry shader and the copy shader the geometry shader's container carries.
On the PS4 the export shader writes each point's attributes to a ring in
memory, the geometry shader reads them there and writes its vertices to a
second ring, and the copy shader exports those to the rasterizer. The lifter
makes the export shader the pipeline's vertex stage, its ring stores its
outputs, and the geometry shader its geometry stage on a point: its ring
loads are the point's inputs, its stores and emits the vertices, and a
vertex's outputs what the copy shader exports of its dwords. Neither ring
exists on the host. A geometry shader that stores at an offset its compiler
counts per lane (the standard particles') keeps that ring as an array in the
invocation. The pipeline is built on the pipeline workers, so a system draws
from the frame its pipeline is ready, and is recorded with its two extra
programs in the stage manifest (format 5) for the next start. A draw the
path refuses - its pipeline still being built, or a geometry shader the lift
does not take - is not drawn: as its export and pixel shaders, as before, it
would draw nothing and only store the export shader's ring through a buffer
its draw's tables name. The exit report counts the refusals by reason.
`BBHOST_GX_GS=0` draws every one as before, the exit report counting those
draws by program. Vicar Amelia's
room draws ~37,500 particles a frame; on an RTX 4070 they cost nothing
measurable (uncapped, 72.2 fps with them against 71.8 without).

On GCN under AMD's own driver (the proprietary one or AMDVLK: a GPU whose
only subgroup size is 64, such as an RX 580), the path is off by default and
those draws go as before. There, the Hunter's Dream's particles came out as
long streaks across the screen, while NVIDIA, RDNA under AMD's driver, RADV
and lavapipe draw them right. The log's start says which way it went, and
`BBHOST_GX_GS=1` draws them with their geometry shader anyway.

`BBHOST_GX_PARTICLES=vs` draws the particles with no geometry stage, on every
device, the way a PC engine draws them: the export shader's and the geometry
shader's lifts are spliced into one vertex stage (`gcn/gs_vertex.h`). Each
output vertex runs its particle's export shader, keeps the ring dwords in the
invocation for the geometry shader's loads, and keeps the one emitted vertex
it stands for - a particle's vertices are the most the geometry shader can
emit, four for a standard particle's quad, forty for a spark's trail of ten.
The draw is indexed over a static list of each particle's triangles in the
order its strips make them, so the provoking vertex of every triangle is the
strip's: one strip a particle where the geometry shader cuts only after its
last emit (the standard particles: up to eight vertices), else strips of four
(the sparks); a vertex past its particle's emits is its last emit, which leaves
every triangle it is in without area. The stage reads the simulation's
indirect arguments itself and leaves out every particle at or past their
count. The draw takes the live count: right after each dispatch - the
simulation's run outside any render pass - one more writes every emitter's
indexed-draw arguments from the game's, its live count times a particle's
indices, and the draw reads those. An emitter's first draws, before such a
pass has written its arguments, run its room, the most records a buffer its
export shader reads by index holds; `BBHOST_GX_PARTICLE_ARGS=0` always runs
the room. `=check` draws with the geometry shader and runs the vertex stage
beside it with nothing rasterized: both write what each vertex gets into a
record, which the host compares word for word once the submission is done,
and the exit report gives the differences by program set, those at a flat
input's provoking vertex apart, and each emitter's room against its live
count. `=gs`, the default, is the geometry shader's path above.

A compute shader is lifted against its dispatch's translation and binds
exactly what that binds - the parameter block with the page table, the images
and samplers, the particles' counter words - so a dispatch is set up the same
way whichever runs. Its memory reads and writes go through the same
page-table walk. GCN's loops that run until every lane's execution bit
clears become structured per-invocation loops, a block GCN branches over
where no lane needs it becomes a selection an invocation skips where its own
bit is clear, and the GPU particles' wave-wide counter idiom (`ds_append`/`ds_consume` in GDS, each lane's rank
from `v_mbcnt`) becomes one atomic add per subgroup, as GCN steps the counter
once a wave and as the translation does: an elected invocation adds the count
of the invocations stepping together and each takes the base plus (append)
or minus (consume) its rank among them, on the counter word the dispatch
hands over by its device address, with no page-table walk
(`BBHOST_GDS_SUBGROUP=0` keeps an atomic an invocation). A barrier is accepted only where every invocation
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
twice, and a dispatch imports what its shader reads and stores (an import a
draw waits for is counted in the log's `frames:` and `stall:` lines, `imports
waited for N in M ms`, and the exit report gives the driver's time by the
thread that waited). It is on by default on an integrated GPU under Windows,
where whole spans kept about 1.5 GB more of the game's memory resident for the
GPU. It is off on a graphics card of its own, and under Linux, where window
imports have not been measured on an integrated GPU yet (AMD's imports go
through udmabuf there). `gpu.import_windows`, true or false, decides it either
way. The work under way moves each of those uses off guest memory, and
windows are then only the fallback for what is not moved yet. The Steam Deck
kit turns it on: there the
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
device-memory blocks). The GPU particles' buffers alone are kept that way by
default on a discrete GPU (`BBHOST_PARTICLE_TWINS`: `0` keeps them in guest
memory, `1` makes them twins on any GPU): a buffer becomes one once a
dispatch that counts in GDS binds it - its records, and whatever else that
dispatch reads and stores - from the next command buffer on; the counter
words, past the records, stay in guest memory. In guest memory the particle
update read and wrote its 176-byte records across the bus at about 0.2 µs a
particle: 7.4 ms of an RTX 4070 a frame in Vicar Amelia's room (73 fps
uncapped at 1080p), against 0.14 ms as twins (161 fps). An integrated GPU's
device memory is the host's own, so there they stay in guest memory, as they
do under a capture. Every buffer binding asks the twins first; one that meets
no twin and no armed buffer - nearly all of a frame's - is answered from a
sorted list of their ranges' union instead of two map walks
(`BBHOST_TWIN_FAST_MISS`: `0` walks the maps for every lookup, `check` does
both and counts any answer that differs in the exit report). With window imports a copy's source the CPU wrote is
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
A colour target a draw writes no channel of - its write mask, the game's cut to
what the pixel shader exports, is empty - gets no blend state either. The game
leaves blending on for targets a shader does not export: the G-buffer's puddle
decals export six of the seven targets bound and blend into the seventh too,
and the effects' soft sprites export one of two. The shader's output for such a
target is undefined and Vulkan writes nothing through an empty mask, so the
state is dead where a driver honours the mask; a driver may blend the undefined
output in anyway, and with the state zeroed there is nothing to blend
(`BBHOST_MASKED_BLEND=0` keeps the game's blend state).
Lavapipe, once a pipeline without colour attachments has been bound in a
command buffer, draws every colour attachment of the later draws in it with
attachment 0's blend state: a target the shader does not export is written
wherever attachment 0 blends, and the puddle decals left NaN in the G-buffer's
seventh target - black holes in the floor. On lavapipe alone, a draw that
follows a depth-only pipeline in the command buffer being recorded, and whose
attachments' blend states differ where that would change what it draws, starts
a command buffer of its own, what was recorded before it submitted without
waiting (`BBHOST_LVP_BLEND_SPLIT=0` records it like any other); other drivers
draw it correctly.
A draw whose rasterizer state comes from the registers rather than a GX
description takes its depth clamp from PA_CL_CLIP_CNTL: with CLIP_DISABLE,
ZCLIP_NEAR_DISABLE or ZCLIP_FAR_DISABLE set z is not clipped, and the depth
block clamps depth to the viewport's range, which is Vulkan's depth clamp.
GX's rasterizer builder sets both z bits from D3D11's DepthClipEnable, so this
is exact for its states; Vulkan has no switch per plane, and a register with one
alone clamps the other plane's geometry onto it instead of clipping it.
DX_RASTERIZATION_KILL is rasterizer discard, a dynamic state only the pipelines
of such draws take (`BBHOST_CLIP_CONTROL=0` keeps depth clamp from CLIP_DISABLE
alone and rasterizes every draw).

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
| Geometry-shader draws (the GPU particles) | native: the host's geometry stage, the PS4's two rings as stage inputs and outputs |
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
| Geometry, with its export shader | the effects' 4 particle geometry shaders of their 7; none of the models' 1,694 | 100% once its pipeline is built (its draws until then are not drawn) | The models' geometry shaders (cube-map depth, a face a layer) and the effects' two array shaders write a layer, which the lift does not take; the effects' point sprite reads its vertices through a fetch shader. None is drawn at the 12 places. |
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

### Window imports and the textures' sources (a design note)

The game's GPU-visible memory can be imported a 16 MB window at a time
(`BBHOST_IMPORT_WINDOWS=1`, or `import_windows = true` under `[gpu]`) instead
of whole. About 1.0-1.3 GB stays imported instead of 5 GB, the driver spends
0.27 s importing instead of 1.27 s (RTX 4070), and the command processor's
time in play is the same. The Steam Deck kit uses it. On the desktop it waits
on the area arrivals, measured at Central Yharnam, Cathedral Ward, the Hunter's
Nightmare, Laurence's staircase, Lumenwood and Astral:

- With windows, three copies the GPU made from the whole import became CPU
  copies into upload staging: the buffer shadow's mirror pages (8,552 pages,
  534 MB at Nightmare of Mensis' arrival), copy tokens' sources (about 2 GB a
  run), and texture uploads' tiled bytes (copied by the prep pool besides being
  hashed).
- The first hitch after a load (130-160 ms) was the mirror pages, copied on
  the command processor. On threads of their own (`bb-pagecopy`,
  `BBHOST_SHADOW_COPY_THREADS`) that first frame came back to whole imports'
  42-62 ms.
- What is left are copies a submission needs at once. The command processor
  waits for the texture preps whose staged sources the next submission reads:
  350-450 waits and 150-180 ms a run, against about 45 and 4 ms with whole
  imports; more prep threads do not shorten it. It also waits for copy
  tokens' sources copied just before a submission. Window mode then counts
  one to three more arrival stalls a place than whole imports, and 32-36 a
  12-place tour against 25.
- The remedy is to let the GPU read those sources again: windows imported
  ahead of the upload, off the command processor, at the GX resource's
  creation (as the import prefetch does for buffers), and retired once idle so
  that the imports stay near a gigabyte, with staging only where no window is
  there yet. It belongs to the step that makes GX textures Vulkan images at
  their creation, which removes most of these imports anyway.

### Resolving draws where their command lists are finished (a design note)

The command processor limits the uncapped frame rate at the heaviest scenes -
at Nightmare of Mensis about 11 ms of its CPU a flip for 1,270 draws, against
8.4-8.75 ms for 913 at Central Yharnam - and the next large step for it would
be to stop visiting draws: each GX command list's draws resolved, and recorded
into Vulkan secondary command buffers, on the thread or pool that finished it
(GX's FinishCommandList and ExecuteCommandList(s) are source,
`src/decomp/gx/command_list.cpp`), the command processor checking each list
against what ran before it and stitching the lists in order. It is not being
built yet; the reasoning, as measured on the development machine:

- **Recording is already off the command processor.** `bb-record` replays its
  packets, ~2.3 ms of work a flip at Mensis. Secondary command buffers as such
  would spread work that is not on the critical path; what costs the command
  processor is resolving each draw.
- **Moving pieces of it to the GX call does not help.** Two experiments did:
  a draw's CPU-written constant buffers copied at the call, and each stage's
  key bindings (its T#s' words, image types and views, its samplers' modes)
  resolved there. Both were correct and neither shortened the command
  processor, while the GX threads paid 0.5-1.5 ms a flip for them. Its cost is
  reading each draw's data - 116-135 last-level cache misses a draw, a quarter
  of them its lookahead's prefetches of what the GX threads wrote - and a piece
  moved to the call is replaced by reading what the call wrote. Only a command
  processor that no longer visits each draw gains.
- **What could leave it.** Of its time at Mensis ~19% does not depend on the
  order of execution (pipeline keys, vertex formats, user data, the token
  lookahead, building packets), ~16% needs only a check per list (the buffer
  shadow's placements, texture views), ~22% must stay in order (copy tokens
  and copy-backs, the packet walk with its labels and waits, render targets
  and passes, the GPU-write tracking and copy versions, tessellation,
  submission), and the draw path's own code, ~42%, goes with whichever of these
  it serves. Keeping the serial work (~1.4 ms) and adding a check, the barriers
  and an execute per list, 7-9 ms a flip could leave it, realistically 4-6.
- **What has to stay serial.** Labels and waits (a list splits at each wait,
  and the draws after it are resolved only once the command processor has
  passed it), the submission and flip order, copies and dispatches between
  lists, the buffer shadow's imports, GPU-write tables and copy versions (each
  list's read ranges checked against the GPU writes and imports since it was
  resolved), and render targets - their surfaces, layouts, clears and
  barriers, which go only between secondary command buffers, so a list splits
  at every target change, barrier, clear, copy or wait.
- **Why it waits.** The main loop's own frame is ~10.1 ms at both places, so
  at Mensis the command processor needs only ~1 ms less to stop limiting it
  (+10-12% there), and every millisecond after that shows nothing until the
  main loop is faster; Central Yharnam is limited by the main loop already. A
  paced frame rate gains nothing, the CPU time moves to other threads rather
  than shrinking, and a four-core machine has no spare thread for it. It is
  the step after the main loop is under ~8-9 ms.
- **Its risks.** State that changes between a list's finish and its
  execution - the command processor runs about a frame behind: streaming,
  area entry, render-to-texture within a frame, resolution changes - which
  needs a mode that resolves both ways and counts disagreements;
  dynamic-rendering inheritance per secondary; descriptor and command pools
  per thread, alive until their submission retires; the pipeline cache and
  its compiles from many threads; resolving on the GX workers would lengthen
  the barrier the main loop waits in for their jobs; the drivers' cost for
  hundreds of secondaries a frame; and the tools that assume the command
  processor resolves each draw (F12 captures, `tools/glitch_hunt.sh`,
  `BBHOST_GX_COST`, `drawreplay`). Its first step would be a compact command
  per draw, built at the GX call in place of the token's sparse records.

### GX's objects as Vulkan objects (a design note)

GX - the engine's Direct3D 11-shaped layer - keeps its buffers and textures in
guest memory, and the command processor finds each draw's bytes there by
address: copy versions, the buffer shadow's mirrors and snapshots, the
import, the texture cache. The direction is to make GX's objects Vulkan
objects when GX creates them, so that a draw carries their handles and the
command processor resolves nothing, in stages: buffers, then textures and
views, then GX's state as pipeline pieces, then command lists recorded into
Vulkan command buffers in parallel and completions on timeline semaphores,
after which guest memory is no longer what the GPU reads for GX's objects.

**Buffers (the first stage).** A census
(`BBHOST_GXVK_CENSUS=1`) measured what it rests on at Central Yharnam, Old
Yharnam, Nightmare of Mensis and Cainhurst: GX makes 53-60 thousand buffers
in such a session, 92% of them constant buffers without data; its immutable
vertex and index buffers - about 1,040-1,340 of them, 1.2-1.6 GB - are never
written again through GX (no Map of them, no UpdateSubresource of a buffer
other than a constant buffer), and hold 75-85% of the vertex records the
draws bind at the three places where the command processor limits the frame
rate (85-94% of those in GX's own buffers) and 74-79% of their index ranges
(99%). A GX holder has spare bytes that no
GX code writes: the dword at +0x14, the upper half of the reference count
(every locked operation on the count is 32-bit), in buffers and textures
alike, which canaries confirmed at every binding and death (1 of 2.2 million
checks failed). The dword at +0x44 is not spare in a constant buffer's
holder: GX counts in its upper half.

Such a buffer - immutable, or default with initial data; no CPU access; a
vertex or index bind and nothing the GPU writes - gets an entry in a host
table when GX's CreateBuffer (on the decomp list) makes it, and the entry's
handle in the holder's +0x14. The GX call
copies the handle into the draw's record beside the holder; the command
processor binds the entry's device-local copy - placed in a pool and filled
from the buffer's memory in the uploads of the first recording that binds it
- instead of resolving the record's address, and plans the vertex input at
the guest addresses, so its bindings and pipeline are today's. A Map, an
UpdateSubresource or a GPU copy into the buffer retires the entry and its
bindings take today's path again. Every GPU copy GX records asks a watch of
the kept default buffers' ranges first, by the 64 KiB granules they cover
(one-byte counters in a 16 KiB table), so a copy far from all of them reads
a counter or two, and one next to them searches the ranges, sorted, in
halves; a buffer that ended (its holder released,
its memory unmapped or given to a new resource) is still bound from its copy
by the draws recorded before its end. Video memory is used down to the
reserve kept for render targets and device buffers; short of room a buffer
keeps today's path - never a copy in host memory on a discrete GPU, where
one placed during a dip stayed for the session and every draw of it read
across the bus (on an integrated GPU, whose host memory is the same RAM,
host-visible memory is taken). `BBHOST_GXVK_BUFFERS=2` keeps today's path
and checks the handles
beside it - the vertex input, the bytes as today's path finds them, the
buffer's bytes against their hash when copied, the device copy read back -
and the exit report counts the bindings by handle, the fallbacks by reason
and any disagreement.

GX's buffers are bound by handle by default (`BBHOST_GXVK_BUFFERS=0` keeps
today's path), except on an integrated GPU or with tight memory, where the
draws read GX's buffers where they are - the buffer shadow in place, or no
shadow - so a copy would hold memory and buy nothing, and with small video
memory (under 7.5 GiB free at the start), where the copies compete with the
targets and textures for it and one dip to the reserve costs far more GPU
time than the command processor saves; there today's path stays unless
`BBHOST_GXVK_BUFFERS=1`. On an RTX 4070, uncapped, the frame rate goes from
83.6 to 94.6 at Old Yharnam and from 82.2 to 95.0 at the Nightmare of Mensis
by handle, the command processor taking 1.4-1.6 ms a flip less; at 60 it
takes up to 1 ms a flip less, with the GPU's time and the frame pacing
unchanged.

GX's texture views bind from their entries by default
(`BBHOST_GXVK_IMAGES=0` keeps today's lookup and refresh). The entry a view
gets at its first binding is one line: its image view, the pipeline key's
dimensions for its bindings (decoded once from its T#, which never changes),
the T#'s base and, bindless, its slot. A GX texture's image that nothing has
written since its upload binds without its surface's refresh while nothing
could have changed it: every place that changes such a surface moves one
epoch, and each entry then refreshes once (and every 200 ms regardless, for
the content hashes' schedule and the eviction's clock). No image, view or
layout changes. `BBHOST_GXVK_IMAGES=2` binds today's way and counts where an
entry would have differed - a refresh it would have skipped that found
work, an image of another generation, dimensions, bases, slots. On an RTX
4070, uncapped, the command processor takes 0.42-0.56 ms a flip less at Old
Yharnam, the Nightmare of Mensis and the Cainhurst courtyard (Old Yharnam
92.8 to 98.3 fps), with the GPU's time, the frames and the frame pacing
unchanged.

**The upload heap (the stage's second part, off by default while it is
measured).** GX's device gives its block pool one region - GX's upload heap,
160 MiB, made by the pool's init, which is on the decomp list - and every
renamed constant buffer, Map rename and ring allocation comes from it. The
pool hands out from both ends of it: the deferred contexts from the heap's
start, the immediate context from its top (its Map discards, among them a
256 KiB copy a frame for the ring its user-primitive draws fill), with one
free list between them, so after a large load the two ends meet. GX
hands a range out again only after an end-of-pipe label behind the work that
used it has landed, and a label lands only once the host's submission has
completed, so a renamed copy holds what its draws read until the GPU has
done with them: in the census, 70% of constant-buffer bindings and 10% of
vertex records lie in the heap. `BBHOST_GXVK_ARENA=1` binds those where they
are, in the heap's import - a vertex record as far as its draw reads -
without the snapshot copy or the lookups. Every run it is used
proves the rule again: each hand-out is a record, each binding stamps the
record it lies in with its submission, and a hand-out over a record's memory
needs that submission completed; and one binding in 64 has its bytes hashed
when it is bound and again when its submission has completed. A violation
sends every binding of the heap back to today's path for the rest of the run,
and the exit report says so. Both ends' hand-outs are records: while only the
start's were, memory the top end took back after the load into Old Yharnam
read as a stale record's, and the proof stopped in two of four runs.
A submission counts as completed for the proof from just before its labels
land, not after: GX hands a range out again the moment it reads the label,
which can come before bbhost has counted the labels in memory, and counted
after, such a reuse read as early in one of four runs at Old Yharnam.
`BBHOST_GXVK_ARENA=compare` keeps today's path
and checks every binding of the heap as in place would bind it.
Moving the heap into video memory the CPU writes was tried and dropped
(`BBHOST_GXVK_ARENA=2`, removed): GX's device is made before bbhost's, so its
pool was given a host mapping of bbhost's, whose bytes were copied into video
memory at the GPU's start and the video memory mapped over it
(`VK_EXT_map_memory_placed`, on Linux), in pieces of 32 MiB - NVIDIA's driver
places that memory at most 124 MiB at a time. Every move worked, and on an
RTX 4070 the frame rate fell by about two thirds - uncapped, Old Yharnam
from 90.8 to 31.7 fps, the Nightmare of Mensis from 90.9 to 34.4 and Central
Yharnam from 105.2 to 31.6, and 31-34 fps at 60 - because the CPU reads the
heap too - bbhost's copies of GX's renamed buffers and the game's main
thread - and the CPU reads video memory at about 30 MB/s. The heap stays in
host memory, where `=1` binds it.

**The heap's constant buffers, snapshotted with no lookups (on by default;
`BBHOST_GXVK_HEAP_SNAP=0` keeps today's path).** On a discrete GPU, binding
the heap in place has the shaders read those constant buffers from host
memory across the bus, where today's copies sit in video memory, so the GPU
took 0.2-0.3 ms a flip more uncapped at Old Yharnam and the Nightmare of
Mensis; and on the Steam Deck, where nothing crosses a bus, binding in place
measured no faster than the snapshots once a snapshotted constant buffer took
no import lookup. What is left to take away is the lookups a snapshot still
asks before its copy: whether a copy-back is still in a copy version over the
range, whether a device-local twin holds it, whether a GPU write is still to
land on it, and whether it is mapped. For the heap the answers are known - it
is a copy-back's source and never its destination, no GX buffer's own memory
lies in it, the GPU writes nothing there, and it is mapped for the session -
so a binding there takes the snapshot with those answers: the same copy of
the same bytes into the same ring, shared within a recording as before. It is
the shape Dark Souls III's PC renderer has on the same engine: its material
constant buffers are updated with `UpdateSubresource` just before they are
bound, so Direct3D 11 copies the bytes at the call into the driver's own copy
in video memory. The answers are watched where they could change, not at each
binding: a GPU write noted over the heap, a copy version made into it, a twin
made over it, its memory unmapped. The first such event is said in the log,
and from it on every binding of the heap asks the lookups again; in the
command processor's order it comes before any binding that could see it.
`BBHOST_GXVK_HEAP_SNAP=compare` keeps today's path and checks each binding's
lookups against the answers. The exit report counts the bindings that took
the answers, the ones that did not by reason (stored to, empty or over 1 MiB,
past the heap's end, a draw with CPU-written windows, snapshots off, a check
that needs the lookups, no ring yet, after a trip), each guard, and with
compare each disagreement. On an RTX 4070, uncapped, the command processor
takes 0.46-0.49 ms a flip less at Old Yharnam, the Nightmare of Mensis and
Central Yharnam, and Old Yharnam goes from 90.6 to 95.1 fps; at 60 it takes
up to 0.55 ms a flip less, with the GPU's time, the frames and the frame
pacing unchanged.

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
