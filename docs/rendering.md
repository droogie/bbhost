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
draw.

**The command processor.** bbhost still walks the game's command buffers, for
what has not moved to tokens: compute dispatches, labels and fences the game
waits on, and some clears and copies recognised from their packets. Compute
dispatches go through an emulated register file, by decision for now: they
are tightly interleaved with the constant engine's memory traffic, and most
of them are fills and copies that never run a shader, which bbhost recognises
and performs directly.

**Recording.** The command processor leaves the Vulkan calls to a thread of
its own, `bb-record`. Each draw goes into a packet, and most of its other
commands - barriers, fills, copies and the like - into the same ordered
stream as typed records (`src/host/recorder.cpp`, `src/host/stream_ops.h`).
`bb-record` replays the stream into command buffers, begins and ends each one
and hands it to the thread that submits, while the command processor goes on
to the next draw. With `BBHOST_STREAM_SUBMIT=0` the command processor begins,
ends and submits the command buffers itself, to compare.

**Shaders.** GCN shaders are converted to SPIR-V when the game creates them
(the few that cannot be, at their first draw), so most pipelines are ready
before they are needed. There are two compilers:

- the *translator* (`src/gcn/translate.cpp`) reproduces the GCN program's
  execution faithfully - one invocation per lane, with the execution mask as
  data - and handles every shader the game uses;
- the *lifter* (`src/gcn/lift.cpp`) decompiles a shader into typed,
  structured SPIR-V, the form a person could read and edit. It is used for
  every pixel shader, and every vertex shader on vertex input, it accepts -
  with the vertex formats read from the shader's parameters, or as
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

**Textures and memory.** The game's memory is shared with the GPU, as on the
PS4: its GPU-visible parts (about 5 GB) are imported into Vulkan, because
shaders that cannot be fully resolved read memory through a page table. Textures are detiled on the GPU, and bbhost notices when the game
rewrites one by write-protecting its pages. Render targets are Vulkan images
sized from the engine's own views, and can be re-created at any resolution
while the game runs, the way the PC engine's GX device resizes.

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
| Other compute dispatches | emulated register state, the game's shaders |
| Clears and target copies | partly recognised from packets |
| Labels and fences | emulated, on the command processor |
| Shaders | decompiled (lifted) to SPIR-V; compute and a few refused programs translated (see below) |
| Resource identity | the game's memory addresses, plus write tracking |

## Shader lifting

Every shader the game creates goes through the lifter. One it refuses is translated instead, so nothing is lost. The program counts below cover every shader the game ships. The draw shares were measured at the twelve places `tools/area_check.py` visits; elsewhere they are expected, not measured.

| Shader stage | Programs lifted | Draws lifted (12 places) | Left |
|---|---|---|---|
| Pixel | 3,206 of 3,209 | 100% | FXAA3HQ, which needs a whole-wave proof; two debug shaders whose result depends on how the PS4 packs pixels into waves (no exact PC form). None is drawn at the 12 places. |
| Vertex | all 10,116 | 100% | - |
| Tessellation: local, hull, domain | all; the 141 mesh hull sets as native pipelines | 100% | - |
| Compute | 0 | - (dispatches) | not started: every compute shader is translated |

A draw whose setup the lifter hasn't seen still falls back to the translator. The exit report counts those by reason, and `BBHOST_LIFT_TESS` and the other switches turn the lifted paths off for comparison.

| Step | State |
|---|---|
| Pixel and vertex shaders lifted | done |
| Tessellation lifted, mesh hulls native | done |
| Compute shaders lifted | next |
| HLSL: the lifted SPIR-V printed as HLSL and compiled with DXC to SPIR-V for Vulkan; checked byte-exact against the direct path before players switch | planned, after compute |
| Shaders as editable source, generated on the player's machine from their own game files (never shipped) | after HLSL |
| D3D12 (Shader Model 6) from the same HLSL, if ever wanted; D3D11 is not planned, since it lacks the quad operations 703 programs use | later, optional |

## MoltenVK defaults

On macOS Vulkan runs on MoltenVK, which turns SPIR-V into Metal shaders and
Vulkan commands into Metal encoders. bbhost detects it by the driver's ID and
does several things differently there; other drivers are unchanged unless a
row says otherwise. Most of them have a switch that goes back.

| What differs on MoltenVK | Why | Switch |
|---|---|---|
| Pixel shaders lose their `NoContraction` decorations, except on what feeds `FragDepth` | The translator and the lifter mark every float add, subtract, multiply, divide and negate `NoContraction`, so that lifts and translations round alike. MoltenVK makes each marked op a call to an unoptimised helper: 204 calls in one lighting shader, which drew in 8.4 ms against 1.0 without them, and a clinic frame went from 127.8 to 37.2 ms | `BBHOST_NO_CONTRACTION=1` keeps every decoration |
| Vertex and tessellation evaluation stages that write `Position` lose them too, and `Position` is decorated `Invariant` instead | The same helpers made the scene pass vertex-bound (about 29 ms a frame in the Hunter's Dream, 9 ms without). MoltenVK compiles `Invariant` as `[[position, invariant]]`, so a surface drawn by two pipelines still lands on the same depths | `BBHOST_VS_CONTRACTION=1` drops them in pixel shaders only |
| A pixel shader that can discard no longer keeps them on its branch conditions | Keeping them made alpha-tested foliage cost most of the scene pass (about 22 ms of 37 at the Byrgenwerth lamp, 12.5 without). They were there so that two pipelines drawing one surface under an equal depth test discard the same pixels; none of the frames logged has such a test | `BBHOST_PS_DISCARD_EXACT=1` keeps them |
| The SFX particles are drawn as one vertex stage | They are patches of one control point with constant factors, and on MoltenVK every tessellated draw is a compute encoder and a render pass restart: some 60 draws a frame in the clinic. The local shader and the domain shader are fused into one vertex stage, and the draw becomes a triangle list. Indexed or indirect draws and other shapes keep the tessellator | `BBHOST_TESS_VERTEX=0` keeps the tessellator; `=1` takes the vertex stage on any device |
| A clear of a render target is held back and becomes the next pass's load op when that pass starts with the whole image as an attachment | MoltenVK makes each `vkCmdClear*Image` a Metal render pass of its own: 85 of 258 render encoders a flip in the clinic were clears, and 4 of 177 after. Anything else recorded first records the held clears before it | `BBHOST_FOLD_CLEARS=0`; `=1` turns it on elsewhere, where it is off |
| A new pipeline is built on a worker thread, and its draws are skipped until it is ready | MoltenVK compiles the Metal shaders when the whole pipeline is created, and that froze the game for seconds on the first sight of an area (the longest frame in one walk went from 6.9 s to 829 ms). Now a new object, light or effect shows a few frames late. bbhost waits up to 16 ms a flip for the build; at the start and on loading screens pipelines are still built in place, and one still building when a loading screen begins is waited for, since a draw made once there would be lost | `BBHOST_PIPELINE_DEFER=0`; `=1` turns it on elsewhere. `BBHOST_PIPELINE_DEFER_WAIT_MS` sets the wait |
| Rect lists are drawn through the tessellation stages | Metal has no geometry stage, which draws them where there is one | none; `BBHOST_RECT_GS=0` takes the tessellation path on a device that has a geometry stage too |
| Each graphics set's descriptor layout names the stages that read it, and the layouts' sampler, image and buffer counts fit the device's limits per stage | Every binding was declared to all stages, and MoltenVK binds 16 samplers and 31 buffers a stage (Metal's limits), which the validation layer reported at device creation. A stage that names more samplers than the device binds is said once in the log and not built | none: MoltenVK's alone, when its limits are below the old layouts; another driver over its limits keeps the old layouts and says so in the log |
| Vertex input past Metal's buffer slots is refused, and the draw runs its fetch shader | Metal gives a stage 31 buffer slots. MoltenVK fills them with its argument buffers and the push set's buffers from one end and the vertex buffers from the other, which leaves room for 6 vertex buffers. One pipeline with 8 failed to build and lost its 60-90 draws a second. The exit report counts the refusals as "no Metal buffer slots left" | none |
| At most 8 command buffers in flight, where others allow 16 | MoltenVK's command pool keeps every temporary buffer its command buffers ever made. With 16 slots the pool took 1.7 GB at Byrgenwerth and a 16 GB Mac swapped; with 8 it is 0.97 GB, still at the 30 fps cap | `BBHOST_GPU_SLOTS=N` (2 to 16) |

## What comes next

- Compute dispatches built from the engine's objects, like the draws, so the
  register file can go.
- Resources identified by the engine's own objects and generations rather than
  by guest address and write tracking.
- Clears and copies from their GX calls rather than from packets, after which
  the PM4 command processor only orders work.
- Compute shaders lifted, then HLSL (the table above).

## Tools

| Tool | What it does |
|---|---|
| `gcn2spv` | translates and validates every shader in the game's bundles |
| `gcndis` | disassembles GCN programs |
| F12 in game | dumps the frame, every render target and the draw list into a folder of its own (`tools/f12_check.py` reads it; `BBHOST_F12_PNG=1` for PNG) |
| `BBHOST_CAPTURE_DRAW` + `drawreplay` | captures one draw with its inputs and replays it outside the game |
| `tools/area_check.py` | ten places in the game measured the same way every time - frame times, GPU time, draws by how their pixel shader ran, compiles, frames and lift checks - and two builds compared against their own noise ([area-check.md](area-check.md)) |
| `BBHOST_VK_VALIDATE=1` | runs with the Vulkan validation layer |
| `BBHOST_GPU_PROFILE=1` | per-pass GPU timings |
| `BBHOST_ENCODER_STATS=1` | per 300 flips, on lines of their own, the Metal encoders the replayed stream makes (render, compute, blit; image clears counted), the barriers by kind and how many sit back to back or between two encoders, and the callers that record the most memory and image barriers. Commands recorded in place are not seen, `BBHOST_GPU_PROFILE` among them |
| `BBHOST_ENCODER_TRACE=N` | the same counting, and logs the encoders of the first 40 command buffers from flip N on, in order |
| `BBHOST_BARRIER_SKIP=<mask>` | leaves the barrier kinds whose bits are set out of the stream (bit 0 pass end, 1 and 2 transfer begin and end, 3 copy order, 4 memory, 5 image): measurement only, may draw wrong |
| `BBHOST_SPIRV_SALT=<text>` | renames every module-scope variable in the modules handed to the driver, so that its pipeline cache and, on macOS, Metal's own shader cache (which a fresh data folder leaves warm) miss: a cold shader build on purpose |
| `bbhost --stream-selftest[=N]` | checks the command stream headless against a CPU model, N rounds (16 by default), and exits with 0 when it matched; run it with `BBHOST_VK_VALIDATE=1` too |
