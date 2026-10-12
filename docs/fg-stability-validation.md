# Frame-generation stability and guide diagnostics

## Hardware and limits

Tests use an RTX 5080, driver 617.14 and one 1920x1080 presentation surface
at 119.88 Hz. Logical reconstruction workloads are approximately 7 and 11
megapixels. Saves and options are copied into isolated test profiles.
No driver FPS limit is installed. The NVIDIA application profile selects
layered DXGI presentation; its inherited V-Sync setting is forced off.
Application V-Sync comparisons must therefore retain this driver override
as part of the test conditions.

A read-only NVAPI query reports monitor VRR support, current-mode capability,
VRR requested/enabled and display-in-VRR-mode all equal to one. DXGI reports
tearing support and hardware-composition capability in fullscreen and windowed
modes. These capability queries do not prove individual scanout timing or the
monitor's low-framerate-compensation threshold. Other refresh rates, display
models and AMD hardware remain untested.

## Display and source-cadence matrix

Sixteen Central Yharnam runs alternate application V-Sync Off/On for each
workload and repeat the sequence twice. DLSS 2x, Reflex On and object motion
are active; the selected game cap is 60. The 30/50 source conditions use a
diagnostic host throttle. The native DLAA workload has no source throttle and
is GPU-bound at approximately eleven megapixels. The windowed condition uses
50 FPS source pacing. Application Off selects IMMEDIATE; On selects MAILBOX.
The driver's forced-off V-Sync setting remains unchanged.

| Condition, four runs each | Display changes/s range | p99 interval range |
| --- | ---: | ---: |
| Source 30, fullscreen | 59.998–60.001 | 18.149–18.493 ms |
| Source 50, fullscreen | 99.985–99.995 | 12.266–12.534 ms |
| Native DLAA, GPU-bound | 69.134–70.860 | 19.912–21.544 ms |
| Source 50, windowed | 99.989–100.000 | 12.000–12.645 ms |

All sixteen measured windows have zero missing display timing and lost ETW
events; all runs exit normally with generation active and zero reported
evaluation failures. Presentation remains Hardware Composed: Independent
Flip with AllowsTearing=1, including application On under this driver override.
This tests the current profile, not driver V-Sync On or VRR-disabled operation.

Windows normally span 75–105 seconds, after the screenshot at 65 seconds.
The two first-round native cases use 90–105 seconds to exclude diagnostic
build activity. The [aggregate CSV](data/fg-stability-validation-2026-10-11.csv)
records the exact windows. Whole-run source estimates and expired-position
samples include warm-up and must not be attributed to those settled windows.

## Long session

A 30-minute Central Yharnam run uses DLSS Performance, DLSS 2x generation,
object motion and 60 FPS simulation. It exits normally after 1,804 seconds.
Observed gameplay continues across borderless-fullscreen to windowed changes,
a smaller-window resize, minimize/restore, return to fullscreen, exit to title
and Continue reload. Generation resumes after loading. The test uses a copied
save; it is not a traversal or multiplayer soak test. This soak and the display
matrix below use the repaired pacing/Reflex build before the instanced-motion
selection change. Candidate gameplay and alternating performance checks are
reported separately; they do not constitute a second 30-minute soak.

Private memory, working set, handles, threads and global GPU memory are sampled
every 30 seconds. GPU memory stays roughly steady; handles and threads do not
grow continuously. CPU private memory does rise: a pre-reload segment measures
about 14 MiB/minute. Loading and settling change the baseline, so a single
whole-session fit is misleading. A matching FG-disabled run measures 13.86
MiB/minute against 13.96 MiB/minute with generation in the same 300–700 second
interval. The growth occurs in both paths; this does not locate the allocation
source or establish that either path is leak-free.

A five-minute settled presentation window contains 36,001 displayed intervals,
120.00 display changes/s and 10.620 ms p99 intervals, with no missing display
timing rows. The mode is Hardware Composed: Independent Flip. Deliberate window,
loading and diagnostic-build intervals are excluded. Display changes do not
prove full scanouts or end-to-end input latency.

Use `tools/win/analyze_stability.py memory.csv --after SECONDS --before SECONDS`
to compare settled intervals, rather than initial cache/loading growth.

## Same-frame FSR guide capture

Set `BBHOST_FG_CAPTURE_DIR` to an existing output directory and optionally
`BBHOST_FG_CAPTURE_FRAME` (default 1500). This explicit diagnostic copies R32
depth and RG16 half-float motion during the selected evaluation recording.
The shared producer images may be recycled later; the per-slot snapshots
remain protected by the output lease through compositor retirement/readback.
Normal runs allocate no snapshot resources. Capture readback and PNG writes
can stall presentation and must be excluded from timing comparisons.

Alongside `input.png`, `real.png`, generated images and FSR HUD masks, the dump
contains little-endian `depth.bin`, `motion.bin` and `guides_manifest.json`.
The manifest records render/guide/Reflex IDs, input/output dimensions, formats,
jitter, frame time and camera parameters. Motion uses previous-minus-current
render pixels. `tools/win/analyze_fg_guides.py DIRECTORY` verifies formats,
exact byte sizes, finite values and the depth range. Numerical validity alone
does not establish correct motion or color/depth alignment.

`BBHOST_FSR_FG_DEBUG_VIEW=1` selects the SDK's embedded interpolation debug
pass through a small, reproducible patch to the pinned provider. Existing
dispatch structures retain their ABI; an additional context setter validates
supported debug bits. The default is zero. The view replaces generated color,
so it is a diagnostic rather than a gameplay setting.

The grid's top row shows the inpainted game motion field, depth priority and
optical-flow field. The bottom row shows two-channel disocclusion factors,
current input and present/distortion data. Its untouched middle band retains
the interpolated output. Motion colors saturate at large displacement; gray
means zero motion. Yellow in the disocclusion view is not by itself proof of
an error: interpret the two factors using the SDK shader's sampling logic.
HUD compositing still runs and can cover parts of the debug view.

Actual stationary and sustained-pan Hunter's Dream captures have matching
snapshots and finite guide values. During the pan, optical-flow detail and
disocclusion changes concentrate around fine branches and animated edges.
Large raw motion outliers require spatial analysis; they are not sufficient
evidence for changing vector sign, scale or clipping. The diagnostic captures
retain validation output, including ordinary shader-output warnings and
latency-marker threading reports; they are not presented as warning-free runs.
No synchronization-hazard text appears in the retained capture logs.

## Instanced foliage motion history

Visible instance order does not identify an object across frames. Culling can
change a batch's instance count and order, while history was indexed by the
instance ordinal. A bounded three-frame trace confirms that foliage batches
vary from 3 to 43 instances; their palette/model buffer addresses are transient
and cannot substitute for persistent object IDs.

Object motion now accepts individually drawn meshes. Instanced draws retain
camera reprojection and the generator's optical-flow fallback until a stable
object identity is available. This preserves animated-mesh motion without
pairing one tree's clip history with another. There is no arbitrary motion
clamp or change to vector sign, scale or near-plane conventions.

Actual sustained-pan captures contain 4,857,600 guide texels:

| Motion input | Maximum magnitude | Pixels above 1,000 px |
| --- | ---: | ---: |
| Original instanced motion, debug pan | 4,161.59 px | 95,364 |
| Original instanced motion, repeat pan | 3,246.32 px | 41,055 |
| Object motion disabled | 578.72 px | 0 |
| Candidate, individually drawn meshes | 578.72 px | 0 |

All four captures have finite depth and motion values and depth in [0,1].
The poses differ; these are separate pan samples, not a pixel-matched quality
score. The candidate still reports successful object-history loads and used
vertices. Eliminating these guide outliers does not establish artifact-free
interpolation of thin branches, transparency or disocclusions.

Reproduce the diagnostic counts with
`tools/win/analyze_fg_guides.py DIRECTORY --motion-threshold 1000`.
`BBHOST_OBJECT_MOTION_UNSAFE_INSTANCES=1` restores the original selection for
diagnostic A/B runs only; leave it unset during normal gameplay.

## Candidate performance

Two alternating original/candidate pairs use the same executable, copied
Central Yharnam save, stationary camera, native DLAA, DLSS 2x, Reflex On,
object motion On and an approximately eleven-megapixel request. The selected
simulation cap remains 60; hardware limits the source cadence to the mid-30s.
The diagnostic override changes only instanced-motion selection.

| Selection / round | Display changes/s | p99 interval |
| --- | ---: | ---: |
| Original / A | 69.838 | 20.918 ms |
| Candidate / A | 72.703 | 19.419 ms |
| Original / B | 70.403 | 21.259 ms |
| Candidate / B | 73.720 | 19.302 ms |

Measured windows are 110–140 seconds, after the capture at 100 seconds.
Neither pair overlaps a local build. All four have zero missing display-timing
rows, lost ETW events and evaluation failures. Sampled expired positions across
the whole active runs are 0/1/0/0 in table order. The two initial menu-only
warm-up attempts are excluded. The original repeats differ by about 0.8%,
while the candidate improves display cadence by 4.1% and 4.7% in the two pairs.
This does not measure input-to-photon latency or moving-scene quality.

The [aggregate CSV](data/fg-stability-validation-2026-10-11.csv) records each
presentation window separately from whole-run sampled source-period and
expired-position telemetry. The latter includes active-game warm-up and is
not a settled-window real-FPS measurement.

## Candidate gameplay and render checks

The candidate completes a three-minute Hunter's Dream run with FSR 3
reconstruction, FSR 2x, object motion and forced synchronization validation.
No synchronization-hazard reports appear. Existing shader-output/stage-interface
diagnostics and latency-marker threading reports remain visible; this is not
a warning-free gameplay run.

A separate packaged Central Yharnam run uses DLSS Quality, DLSS 2x, object
motion and a 30 FPS simulation cap. Literal Alt-Tab leaves the owned window;
refocusing restores normal gameplay and generation. Its fresh F12 checker
reports no clipped targets, lost clears, stale uploads or copy-back data, and
47 float targets with no infinite/NaN texels. Capture stalls are excluded from
timing evidence.

The native Windows smoke equivalent reaches flip 1,800 and exits with code
zero, without a crash report, core dump or called unbound import. Its owned
window is still present at 12 seconds. This runs the packaged executable on
Windows rather than claiming that the Wine-based cross-build smoke target ran.
Registered renderer checks and the GPU provider debug-flag/pixel/resize
contract checks pass; hosted GPU/game-file skips remain separate evidence.
