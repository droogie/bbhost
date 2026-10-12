# Upstream integration validation, 2026-10-12

The renderer is rebased onto upstream master `566184a74f071166299bb656c4af479e5bbe2771`. Upstream frame timing, native shader lifting/tessellation, particle counters, indirect guards and GPU retirement are retained. Individually instrumented object-motion meshes retain translated shaders because the native lifter does not yet emit their history outputs; other draws retain upstream lifting.

DLSS frame generation offers Off, 2x, 3x and 4x in Setup and F10 without an experimental environment opt-in. Saved higher factors survive loading. SDK capability negotiation still applies: unsupported factors fall back to the supported path. FSR generation offers Off and 2x. Temporal setting changes require a restart. The optional FSR 4 reconstruction provider retains its existing experimental status.

## Source and executable boundaries

- Renderer code: `a3a7fa7b9deee041fe2509744d05bdd5e9bed73d`.
- Integrated Windows build: `9f330ae273591445a89109165dc4448cf5953948`. It additionally includes the independent chromatic-field, Windows certificate-trust and dispatch-diagnostic fixes. Those changes are outside the renderer branch.
- Initial rebased executable (v1): SHA256 `E7757DAB04E2ED61C881FED8D5CC6D6093CCC2A2B95FF8DBE7AA39E343209B95`; predates the FSR transient-retirement repair.
- Repaired executable (v2): SHA256 `FD2231F8E38CB726E3213CBC99B5043CA2CFDE1299517C555C5A4952C17DFF98`; same functional source as the final build, compiled before commit metadata was regenerated.
- Final executable: SHA256 `EA1769F935BD65AB5A1C00E470174F533748623C942513AB575EB10C6B1C4E19`.
- Unchanged NGX bridge: SHA256 `17A58F924837C6763395EF098B430E73C05F9BA0FFBA8642F75A6738FB67732A`.

## FSR transient allocation repair

A longer FSR generation run exposed descriptor pools and constant allocations retained until context destruction. The provider did not call `BridgeBeginFrame(frameId)` before generation preparation or `BridgeRetireFrame(completedFrameId)` after completion. Consequently its transient allocations were not tagged with the application's frame IDs. This is a distinct provider lifetime issue; the tests do not establish that the upstream rebase introduced it.

The pinned provider patch tags work before allocation and retires it only through the GPU-completed watermark. Context rebuilds drain outstanding submissions before destroying resources. Failed/unsubmitted recordings never advance the cumulative completion watermark. A diagnostic count API permits the provider fixture to verify retirement without destroying the context, without changing existing dispatch-structure ABI.

Negative control with the two bridge calls disabled retains **539 descriptor allocations and 207 constant allocations** and fails the fixture. Restoring the calls retires **68 descriptor allocations and 26 constant allocations to zero**, with pixel/reset/resize checks passing. Normal gameplay does not enable guide captures or allocation profiling.

The initial FSR soak was stopped deliberately after private bytes reached approximately 20,424 MiB at 195 seconds and global VRAM approached 15,619 MiB. It is a failed soak, despite a normal shutdown. The repaired run exits normally after 579 seconds: its nine settled samples at seconds 315–555 span four minutes, range from **8,231.840 to 8,239.184 MiB**, and have slope **−1.540 MiB/minute**. Global VRAM remains approximately 6.9 GB. The last report contains 31,800 real frames, 31,777 generated frames and zero evaluation failures. This finite test does not establish that all allocation paths are leak-free.

## Gameplay and presentation

RTX 5080, driver 617.14, PresentMon 2.6.0, one 119.88 Hz display and layered DXGI presentation. The GPU was otherwise idle; the driver FPS limit was removed. The inherited driver V-Sync-off policy was unchanged. Isolated copies of the profile and saves were used. Tests cover Central Yharnam and Hunter's Dream, DLSS Performance with object motion enabled, and FSR 3 reconstruction/generation.

Timing windows exclude startup and deliberate captures/menu interactions. Values describe actual display changes when available, rather than counting every submitted generated frame. Missing display-timing rows are retained in the table and CSV; they are not treated as zero-duration intervals.

| Build / case | Source cap | Timing window (seconds) | Display changes/s | p99 display interval (ms) | Missing display-timing rows |
| --- | ---: | --- | ---: | ---: | ---: |
| v1 DLSS 3x, Central Yharnam | 30 | 110–140 | 89.77 | 13.536 | 8 |
| v1 DLSS 4x, Central Yharnam | 30 | 110–140 | 119.79 | 10.610 | 6 |
| v1 FSR 2x, Hunter's Dream | 30 | 110–140 | 59.80 | 18.964 | 0 |
| v2 FSR 2x repaired soak | 60 | 300–550 | 118.24 | 13.362 | 0 |
| Final saved DLSS 4x | 30 | 120–170 | 119.69 | 10.784 | 9 |

The final saved-4x test has neither `BBHOST_DLSS_FG` nor an experimental opt-in flag. It exits normally after 189 seconds and reports 3,900 real frames, 11,691 generated frames and zero evaluation failures. The v1 DLSS 2x test also exits normally with 6,599 generated frames and zero evaluation failures. These measurements support factor loading and functioning generation on this device, not universal hardware compatibility or perfect pacing. The repaired FSR soak still contains uneven intervals. Display changes do not measure input-to-photon latency or visual interpolation quality.

## Checks and limits

All 13 registered native Windows checks pass, including the Vulkan provider lifetime/pixel/reset/resize fixture, motion reprojection, timing/history/layout, shader instrumentation, NGX transfer contract and chromatic stub checks. Final native startup smoke reaches flip 1800 and exits normally after 39 seconds without core dumps or unbound imports.

Final FSR Vulkan-validation gameplay completes normally in Hunter's Dream. A later F12 capture at flip 8874 covers flips 7743–8873: 602 textures, 32,768 events, zero size/clipping findings, lost clears, stale uploads or CP-stale findings, and all 46 float targets finite. An earlier loading-containing capture had 237 stale-upload-pattern findings and is not used as a clean settled-scene result. No synchronization hazards or descriptor-pool/free-memory-in-use errors were found in the final validation logs. Existing unused-fragment-output, shadow-depth comparison-format and latency-marker/swapchain thread-layer messages remain.

Linux build/CTest succeeds for [renderer code](https://github.com/FIocker/bbhost/actions/runs/38229564356) (62 registered cases) and the [integrated source](https://github.com/FIocker/bbhost/actions/runs/38229811433) (63 registered cases). GPU/game-file cases skip in hosted CI; these runs do not establish Linux gameplay behavior.

Aggregate timing and memory values are retained in [the numeric evidence](data/upstream-rebase-validation-2026-10-12.csv). Earlier source-version measurements remain in the existing validation documents. Other GPUs, refresh rates and displays, driver V-Sync-on behavior, perceptual artifacts and monitor scanout latency remain unvalidated by this integration pass. Keep 60 FPS simulation for multiplayer.
