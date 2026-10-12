# NVIDIA Reflex and frame-generation diagnostics

Setup and F10 offer **Off**, **On** and **On + Boost** under NVIDIA Reflex.
The choice is saved and takes effect after restart. On is the default; Boost
also asks the driver to keep GPU clocks high. Unsupported drivers retain the
ordinary simulation limiter and generated-frame scheduler.

The native Vulkan path requires `VK_NV_low_latency2` revision 3, timeline
semaphores and `VK_KHR_present_id`. It calls driver sleep once per real frame
before input and simulation callbacks, then waits on the returned timeline
value. Sleep and its CPU wait hold no renderer/marker lock: submissions and
presentation must progress while the simulation thread waits.
Acquire/present calls serialize with the driver sleep call and timing queries
using a separate short-lived WSI guard. That guard is released before the CPU
semaphore wait. Markers retain the specification's concurrency exception.

Simulation IDs are captured in command-processor jobs and carried through
recording, submission, the upscaler guides and presentation. Real frames use
simulation/input/render/present markers; generated images use out-of-band
markers. The separate compositor queue is marked out-of-band present so its
blits do not count as game rendering. Submission IDs obey each queue's
monotonic rule; stale asynchronous submissions clear explicit attribution.
The shared-queue fallback can consequently have less precise driver GPU
attribution than the dedicated compositor path.

Generated pairs now start near GPU readiness, with a small compositor budget,
instead of holding their first interpolated image for another subframe
interval. Observed late GPU completions increase that budget, with a slowly
decaying envelope bounded by one output interval. This avoids compressing the
generated-to-real spacing when a GPU-bound frame arrives after its predicted
launch time. The existing bounded cadence filter still follows actual source
timing and skips expired positions under backpressure. This reduces the
present-call delay without changing simulation speed or interpolation factors.

## Measurements

Set `BBHOST_REFLEX_TIMING=1` to log complete per-frame driver timestamp reports.
Use `python tools/win/analyze_reflex.py path/to/stderr.log` for a summary of the
final 39 seconds. Input samples missing during startup are excluded from the
input metric. These timestamps measure application/driver stages, **not
monitor scanout or photon latency**. Pair them with PresentMon display timing;
an FPS counter alone cannot establish pacing, latency or interpolation quality.

Diagnostic overrides:

- `BBHOST_REFLEX=off|on|boost` overrides the saved mode.
- `BBHOST_FG_LOW_LATENCY=0` restores the older generated-pair phase for A/B
  comparisons; the earlier phase is the default.
- `BBHOST_REFLEX_SLEEP=0` disables only the driver sleep call for diagnosis;
  use the normal saved Off mode for ordinary comparisons.
- `BBHOST_REFLEX_INTERVAL_US` overrides the driver's minimum interval for
  diagnosis; the default is zero, with simulation cadence handled separately.
- `BBHOST_TEST_SOURCE_FPS=40` adds a diagnostic source throttle. A comma-separated
  list such as `60,40,30,50` cycles targets every ten seconds. Values are clamped
  to 30–60. This is a test throttle, not an additional supported game-speed mode.
- `BBHOST_FG_CAPTURE_DIR` requests one owned-output readback after the frame's
  GPU work retires. `BBHOST_FG_CAPTURE_FRAME` selects its render serial. Create
  the directory first and exclude readback from timing measurements. Captures
  include the HUD-free input, real endpoint, generated images and FSR HUD mask.
- `BBHOST_NGX_TRANSFER_SYNC=0` disables the NGX FG transfer-dependency repair
  for diagnosis. Leave it unset for normal use. This switch affects the bridge,
  not Reflex or FSR; it takes effect at NGX initialization.

The implementation follows the Vulkan [sleep](https://docs.vulkan.org/refpages/latest/refpages/source/vkLatencySleepNV.html),
[marker concurrency](https://docs.vulkan.org/refpages/latest/refpages/source/vkSetLatencyMarkerNV.html)
and [frame-attribution](https://docs.vulkan.org/spec/latest/chapters/VK_KHR_surface/wsi.html#low-latency-frame-attribution)
contracts. Runtime tests and remaining limits are recorded in
[renderer validation](reflex-validation.md).
