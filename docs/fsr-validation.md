# FSR validation — 2026-10-10

Independent bbhost host adapters use the pinned MIT-licensed FSR-Vulkan
provider at `c64f093404125e960813f95f1996c14780f0dd69`. The reference is
Supermedo's Windows bbport, including its optional source-v07 INT8/DOT4
FSR 4 model approach. No emulator integration code is included.

## Pixel and resource checks

The registered `fsr_provider_check` fixture runs FSR 3 reconstruction and
2x generation on a moving rectangle. RGB centroid checks require the
generated content to lie between both real endpoints, within six output
pixels of the expected midpoint, with bounded background error. On the
RTX 5080, centroids are 485.50 / 491.65 / 509.50 pixels (previous / generated /
current), with zero measured background error. All readback values are finite.
The resize cycle also passes, with imported views and descriptor pools
retired only after their GPU submission completes.

Vulkan synchronization validation is clean for that fixture. A narrow patch
to the pinned provider orders successive initial/reset transfer clears even
when their image layout stays unchanged. The host uses FP16 reconstruction
output, initializes its shared image layouts, allows provider transfer clears,
and enables the optical-flow shader's linear compute derivatives explicitly.

Paired in-game validation runs with FSR reconstruction plus generation and
ordinary rendering report the same three diagnostic IDs:

- `Undefined-Value-ShaderOutputNotConsumed-DynamicRendering`
- `VUID-vkCmdDraw-None-06479`
- `VUID-RuntimeSpirv-OpEntryPoint-08743`

These existing game-renderer warnings/errors are retained, not suppressed.
The new FSR format, transfer-usage, initial-layout and missing-derivative
errors found during development do not recur after correction.

## Gameplay method

RTX 5080, driver 617.14, PresentMon 2.6.0, otherwise idle GPU and no driver FPS
limit. Copied saves in the Hunter's Dream and Central Yharnam; animated-object
motion is enabled. Each run lasts 110 seconds, with camera sweeps; the measured
window is 70–109 seconds, excluding startup, screenshots and shutdown.
Foreground status is checked throughout. Reconstruction requests about
7 megapixels; the presentation surface is 1920×1080 on a 119.88 Hz display.
The tested per-application NVIDIA present method is layered on DXGI swapchain.

FSR 3 Quality off/on is repeated in two alternating rounds. The generated
frame GPU fixture and validation-layer runs are separate from these timing
measurements. Presentation counts include real and interpolated images;
source cadence is measured from the rendering/GPU timeline. Display-change
rates alone do not establish complete scanouts, perceived smoothness or latency.

| Workload | Selected game FPS | Source FPS | Display changes/s | p99 display interval |
| --- | ---: | ---: | ---: | ---: |
| FSR 3 Quality / off / A | 30 | 30.00 | 30.00 | 35.44 ms |
| FSR 3 Quality / 2x / A | 30 | 30.00 | 60.00 | 24.77 ms |
| FSR 3 Quality / off / B | 30 | 30.00 | 30.00 | 35.44 ms |
| FSR 3 Quality / 2x / B | 30 | 30.00 | 60.00 | 24.43 ms |
| FSR 3 Quality / 2x | 60 | 58.32 | 118.49 | 11.25 ms |
| Reconstruction off / FSR 2x | 30 | 30.00 | 60.00 | 18.64 ms |
| FSR 4 Quality / FSR 2x | 30 | 30.00 | 60.00 | 24.71 ms |
| DLSS Quality / DLSS 2x regression | 30 | 30.00 | 60.00 | 24.60 ms |

All measured windows have no missing display intervals. Generation windows
have no evaluation failures, expired generated positions or reset suppression.
The initial history reset is outside the measured window. The 60 FPS selected
cap is a simulation limit, not a guarantee that every real frame finishes at
60 Hz; source and display statistics use their respective sampled timelines.

[Aggregate measurements](data/fsr-validation-2026-10-10.csv) retain submission
and display percentiles, both alternating rounds and failure counters.

## Image-quality limits

Rapid camera sweeps at 30 source FPS show visible warping around geometry and
fine detail with analytical FSR frame generation. The fixtures establish real
interpolated content and safe resources; they do not certify perceptual quality
or latency. This provider remains experimental. FSR 4 is the optional
source-v07 reconstruction model, not FSR 4 ML frame generation.

## Packaged build checks

The packaged executable loads FSR 4 from its own asset directory and renders in
both test areas. With that directory overridden to a missing location, it logs
one fallback message, initializes FSR 3 and continues 2x generation.
FSR 4 plus generation reports the same three gameplay validation IDs listed
above; no new FSR diagnostics appear.

Setup's Start-up and display section exposes both providers, reconstruction
presets and object vectors. Its FSR factor dropdown contains Off and 2x only.
F10 presents those settings at output size; clicking the FSR generation arrow
from 2x selects Off and saves correctly. The FSR cycle stays limited to those
two states even with experimental DLSS MFG enabled. Temporal settings remain
restart choices. Setup closes normally.

The F12 capture checker passes: no targets clipped by size, no lost clears,
no stale uploads or copy-back textures, and 46 float targets with no infinite
or NaN texels. Seven registered reconstruction, pointer, timing, history and
motion checks pass on Windows. Hosted GPU fixture skips are not used as local
GPU validation evidence.
