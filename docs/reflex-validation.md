# Reflex and generated-frame pacing validation — 2026-10-11

The Vulkan integration sleeps before input and simulation, carries immutable
simulation IDs through asynchronous rendering, and attributes real and generated
presentation separately. Setup and F10 expose Off, On and On + Boost as restart
choices. Driver sleep and its CPU semaphore wait do not hold the marker mutex;
holding that mutex reproduced periodic long pauses and was corrected before the
measurements below.

The generated-pair scheduler starts earlier than the former one-subframe delay.
A bounded, slowly decaying budget for late GPU completions preserves space
between generated and real images under varying load. Its deterministic checks
cover variable completions, discontinuities and expired-position selection.

## Method

RTX 5080, driver 617.14, NGX 310.9.1, PresentMon 2.6.0, otherwise idle GPU,
driver FPS limits removed. The output request is approximately seven megapixels;
the presentation surface is 1920×1080 on a 119.88 Hz display. Tests retain the
application's `bbhost.exe` filename and layered DXGI driver profile. Copied saves
and camera sweeps in the Hunter's Dream are used for paired measurements;
Central Yharnam is checked separately. Animated-object motion vectors are on.

Runs last 110 seconds. PresentMon's measured window is 70–109 seconds; startup,
capture and shutdown are excluded. GPU tracking is disabled in these PresentMon
runs; no lost-ETW-event warning occurs in either alternating Reflex A/B round.
Source cadence and evaluation cost come from the renderer's GPU timestamps.
Driver latency reports use the last 39 seconds of completed reports with valid
input timestamps. Their window is consequently distinct from PresentMon's.

**Input-to-present means input sampling to the CPU present call. It excludes
monitor scanout and photon latency.** Display-change counts above refresh do not
establish that many complete scanouts. The measurements are local evidence on
one NVIDIA card and driver, not latency certification across hardware.

## GPU-bound Reflex comparison

Native DLAA plus DLSS 2x, game cap 60; actual real cadence is below that cap.
Both modes retain markers, the adaptive scheduler and the ordinary simulation
limiter. The only A/B choice is Reflex Off versus On.

| Run | Real FPS | Display changes/s | p99 display interval | Median input-to-present | p99 input-to-present |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off / A | 48.63 | 96.87 | 13.74 ms | 83.95 ms | 91.62 ms |
| On / A | 49.06 | 98.00 | 13.58 ms | 44.06 ms | 50.24 ms |
| Off / B | 48.87 | 97.13 | 13.88 ms | 83.96 ms | 92.16 ms |
| On / B | 48.40 | 97.01 | 15.16 ms | 45.06 ms | 54.09 ms |

These are repeats on the deployed executable with the final WSI guard enabled.
All four measured display windows have zero missing timing rows. Sampled
expired-position counts are 0/3/1/1 in table order; no sampled resets suppress
interpolation. The median driver-reported input-to-GPU-end interval changes
from approximately 86 ms to 46–47 ms. Shared
queue attribution has weaker precision than the tested dedicated compositor
queue, so these GPU reports must not be generalized to that fallback.

An earlier fixed 2 ms launch budget produced a 19.02 ms p99 display interval
under the earlier workload. Adapting the budget brought an exploratory repeat
to 13.57 ms. Earlier pre-guard A/B rounds are retained separately in the CSV;
their absolute cadence differs from the final repeats and must not be used as
an isolated estimate of the guard's performance cost.

## Source-rate and changing-load checks

DLSS Quality plus 2x uses the same adaptive scheduler and Reflex On. The 40/50
cases select 60 FPS simulation and add a diagnostic source throttle; these are
not new supported game-speed presets. The changing-load case cycles source
targets 60, 40, 30 and 50 every ten seconds. Its aggregate FPS mixes those rates
and must not be interpreted as a fixed-rate target.

| Source target | Measured real FPS | Display changes/s | p99 display interval | Missing timing rows | Expired positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 | 30.00 | 60.00 | 18.45 ms | 0 | 0 |
| 40 | 40.00 | 80.00 | 14.47 ms | 0 | 0 |
| 50 | 50.00 | 100.00 | 12.96 ms | 0 | 0 |
| 60 | 59.52 | 119.13 | 11.56 ms | 0 | 2 |
| Changing targets | 44.46 | 90.31 | 18.06 ms | 0 | 0 |

The expired-position counts use the final sampled GPU-source window; they are
not a claim that startup and every transition throughout the run never skip a
position. The source and PresentMon windows differ slightly. No sampled reset
suppression occurs. Cadence remains approximately twice real FPS below the
selected cap rather than collapsing when input falls below 60.

A separate native-DLAA workload requests approximately eleven megapixels with
the selected game cap still at 60 and no diagnostic source throttle. The real
GPU-bound cadence is 38.06 FPS; 2x reaches 76.37 display changes/s, with a
16.62 ms p99 display interval and zero missing timing rows. One sampled generated
position expires, with no sampled reset suppression. This run includes the
same narrow WSI guard as the final A/B table.

At fixed 30 real FPS, 3x reaches 90.01 display changes/s with
14.62 ms p99 intervals; 4x reaches 120.00 with 10.82 ms p99 intervals. Both have
zero missing timing rows and sampled expired/suppressed positions. These are
cadence checks on a 120 Hz display, not certification of interpolation quality
or a reason to expose higher factors as ordinary supported settings.

The final deployed build also repeats the eleven-megapixel workload with the
higher factors. At a selected cap of 60 without a diagnostic throttle, 3x
measures 35.54 real FPS / 105.37 display changes/s and a 15.89 ms p99 display
interval; 18 sampled generated positions expire. With a diagnostic 30 FPS
source throttle, 4x measures 30.01 / 119.96, p99 12.14 ms, and one expired
position. Neither run has missing timing rows or sampled reset suppression.
Median input-to-present is 62.45/66.62 ms respectively. Higher generated output
does not recover input responsiveness, and GPU contention makes intervals less
uniform than the fixed-source, lighter-workload checks. These remain measurements of one display and GPU.

[Aggregate measurements](data/reflex-validation-2026-10-11.csv) retain both A/B
rounds before and after the WSI guard, heavy-load higher-factor checks,
submission/display percentiles, driver intervals and failure counters.

## FSR pixel investigation

The GPU fixture now uses textured foreground/background planes, perspective
depth, per-pixel camera motion and an analytic midpoint. It requires centroid
error ≤2 output pixels, foreground silhouette IoU ≥0.93, area error ≤7%, and
mean normalized background error ≤0.0125. Motion-vector scale is one for
pixel-valued vectors, matching the host adapter.

Camera motion at 16, 32 and 64 render pixels per source frame passes with
synchronization validation: midpoint error rounds to 0.00 pixels, IoU is 1.0000,
area error is 0.0000, and background error is 0.0006/0.0004/0.0004. Outputs are
finite and the resize/resource-retirement fixture passes at all three speeds.
The 64-pixel fixture uses fewer history frames to keep its endpoints in view;
this is a controlled planar test, not a gameplay quality score.

An owned-output gameplay readback captures HUD-free input, the real endpoint,
the generated image, HUD color and its alpha mask after compositor retirement.
The inspected fast-pan mask is confined to HUD pixels. The generated world image
still shows breakup around fine branches and disoccluded detail. Those artifacts
remain a limitation of the current analytical provider/integration; the passing
fixture and mask do not prove every gameplay guide or occlusion case correct.
No speculative vector sign/scale or HUD threshold adjustment is shipped.

A separate audit checks previous-minus-current pixel motion against the SDK's
previous-color sampling, finite non-inverted near/far reconstruction against
the captured projection, and the snapshot/color frame IDs through prepare and
dispatch. No mismatch is found in those paths. SDK disocclusion rejection and
optical-flow/inpainting fallback are plausible contributors at thin foliage
edges; the planar fixture cannot establish that as the exclusive cause or rule
out every guide defect in gameplay.

Final deployed-build readbacks during a sustained right-stick pan complete
with object motion both Off and On. Fine foliage edges still show breakup in
both, so disabling animated-mesh vectors does not eliminate that symptom.
The captured poses differ slightly; this is an observational check rather than
a matched-pixel quality score. Readback stalls and those runs are excluded from
the performance CSV and tables.

## NGX transfer-dependency repair

The unmodified NGX 310.9.1 dispatch reports private image-transition/clear/copy
hazards and overlapping buffer fills inside its evaluation call. Barriers
before or after the call cannot order those internal operations. A separate
prototype established that NGX records them through the documented Vulkan
function-table callbacks supplied at initialization.

The bridge now extends the runtime's image barriers with transfer stages and
access masks, preserving layouts, ranges, queue ownership, pNext chains and
existing dependencies. A transfer-to-transfer buffer barrier orders fills over
the actual destination range. Thread-local command-buffer scope restricts the
repair to FG evaluation. There are no additional all-commands image barriers,
no vendor DLL modifications and no validation filtering. SR and other commands
keep their original dependencies. `BBHOST_NGX_TRANSFER_SYNC=0` restores the
unmodified dispatch for diagnosis.

With the same forced validation DLL/manifest and synchronization checking
explicitly enabled, the standalone GPU probe reports 20 hazards with the repair
disabled (ten write-after-write and ten read-after-write before the message
limit) and zero with it enabled. Both runs pass reconstruction, distinct
2x/3x/4x generated images, reset/history recovery and preservation of real
endpoints. Zero hazards in this fixture is not proof of wholly warning-free
gameplay or every GPU/runtime combination. The related vendor report is
[NVIDIA Streamline issue 84](https://github.com/NVIDIA-RTX/Streamline/issues/84).

Two alternating gameplay pairs isolate the repair on the eleven-megapixel,
GPU-bound native-DLAA/2x workload, cap 60, object motion and Reflex On. The same
executable, save, driver profile and stationary camera are used in all four
runs; only the bridge override changes. Timing windows follow the method above.

| Repair | Real FPS | Display changes/s | p99 display interval | Mean FG evaluation |
| --- | ---: | ---: | ---: | ---: |
| Off / A | 37.20 | 74.23 | 17.45 ms | 2.580 ms |
| On / A | 36.95 | 73.78 | 17.61 ms | 2.602 ms |
| Off / B | 37.24 | 74.48 | 17.01 ms | 2.593 ms |
| On / B | 36.93 | 73.83 | 17.59 ms | 2.605 ms |

Both repaired runs cost approximately 0.7–0.8% real FPS in this workload;
evaluation increases by 0.012–0.022 ms. No missing display rows or sampled reset
suppression occur. Sampled expired positions are 1/0/1/0 in table order. These
runs compare synchronization repair, not Reflex modes, and do not collect
per-frame driver latency reports. The earlier Reflex/source-rate tables above
precede this bridge repair; the new comparison is retained in the aggregate CSV.

FSR's standalone fast-pan fixture is synchronization-clean. Existing gameplay
renderer diagnostics remain visible and are compared against a paired run.
The first paired run also exposed swapchain host-thread warnings that a summary
of VUID strings alone missed. Sleep calls and timing queries now serialize with
acquire/present through a separate WSI guard; the CPU sleep wait remains outside
that guard. Marker overlap warnings are retained and investigated against the
explicit [marker concurrency exception](https://docs.vulkan.org/refpages/latest/refpages/source/vkSetLatencyMarkerNV.html),
which permits concurrent swapchain use except destruction. A clean GPU hazard
check must not be described as a wholly warning-free validation run.
The repeated DLSS 2x run with the WSI guard has no sleep/query threading
conflicts and no `SYNC-HAZARD` reports. Eleven marker-overlap reports remain;
they are retained rather than filtered. Earlier standalone NGX results without
the repair reproduced the private
hazards described above. Passing temporal pixel checks alone did not establish
that their missing synchronization was harmless on every driver.

## Build and menu checks

All registered local renderer checks pass, including the FSR GPU fixture and
the new transfer-dispatch contract test. The contract test also passes in the
separate MSVC bridge build and needs no GPU or NVIDIA binaries. The FSR
fixture also passes with the validation DLL/manifest explicitly selected at
16/32/64-pixel camera speeds. Hosted fixture skips are not treated as GPU proof.

Setup changes Off to On + Boost and saves both the options choice and
`[dlss] reflex = "boost"`. A restart reports successful driver mode/boost values
of one. F10 renders in output space, identifies Reflex as restart-required,
cycles Boost to Off and saves the options choice; the active driver's Boost
mode remains unchanged until restart, as intended.
The next Central Yharnam restart reads that saved Off choice and reports
`lowLatencyMode=0`, `boost=0`, with generation active and a normal gameplay
picture. Both test areas use separate copied saves.

The new build's Hunter's Dream F12 check reports no clipped target-size
problems, lost clears, stale uploads or stale copy-back data, and 46 float
targets with no infinite or NaN texels. Diagnostic captures are excluded from
the performance tables.
The Central Yharnam F12 check also passes the same resource/finite-value checks.

On the final WSI-guard build, FSR 3 Quality plus FSR 2x at 30 real FPS measures
60.00 display changes/s and 18.48 ms p99 intervals. FSR 4 source-v07 Quality plus
the same generation backend measures 60.00 and 18.40 ms. Both have zero missing
timing rows and no sampled expired positions/reset suppression. One malformed
FSR 4 driver timing report is rejected by the latency analyzer and retained in
the aggregate's invalid-report count.

## Scope

DLSS Off/2x/3x/4x are selectable in both menus, without an opt-in switch.
The retained measurements cover one 120 Hz display; backpressure and artifact
limits still need broader hardware testing. FSR supports 2x only; its optional source-v07 FSR 4 model is reconstruction, not ML frame
generation. Multiplayer should retain 60 FPS simulation.

See [Reflex configuration and diagnostics](reflex.md),
[DLSS development](dlss-development.md) and [FSR validation](fsr-validation.md).
