# DLAA (NVIDIA DLSS at native resolution)

Optional, Windows + NVIDIA RTX only. bbhost runs NVIDIA DLAA on the lit HDR
scene, before YEBIS's post-processing (depth of field, motion blur, bloom,
tone mapping). No game files change. Without the two DLLs below, on another
GPU or on Linux, it stays off and the game renders exactly as before.

## Using it

1. Put `ngx_bridge.dll` (built from `tools/ngx_bridge`, below) and NVIDIA's
   `nvngx_dlss.dll` (from the DLSS SDK, or any recent DLSS game) beside
   `bbhost.exe`.
2. Turn the game's own anti-aliasing off (F10, graphics).
3. The log says `dlaa: NGX init ok` and `dlaa: ready at 1920x1080 ...`.

| | |
|---|---|
| `bbhost.toml` `[dlaa] enabled = false` / `BBHOST_DLAA=0` | off |
| `[dlaa] preset = "K"` / `BBHOST_DLAA_PRESET=K` | a DLSS render preset (default: the driver's) |
| Ctrl+F1 | DLAA on/off (A/B) |
| Ctrl+F2 | jitter on / off / test (DLSS skipped, jitter x8: the scene visibly shakes when the jitter reaches it) |
| Ctrl+F3 | the jitter's sign as given to DLSS (diagnosis) |
| Ctrl+F4 | zero motion vectors (diagnosis: moving edges alias again) |
| Ctrl+F5-F8 | the same as Ctrl+F1-F4, for keyboards where another program holds Ctrl+F2/F3 |

## How it works (src/host/dlaa.cpp)

- **Where:** YEBIS's first post-process draw, `0b0acf50+ccbf44a6`, copies the
  lit scene (RGBA16F, the target the main depth was drawn with) into its
  working image; the scene is read there and nowhere else. At that draw,
  before its pass, the scene goes through DLSS and the output is copied back
  into it. `BBHOST_DLAA_ANCHOR=<pipeline name prefix>` picks another draw.
  Not the depth-of-field composite after it: its pixel-shader hash changes
  with the settings, and it reads the copy whose alpha (the circle of
  confusion) a draw in between writes; DLSS does not keep alpha.
- **Frames:** a new frame is "depth-writing draws happened since the last
  anchor", not the flip count: the command processor can lag the guest's
  flips.
- **Depth:** the 1920x1080 depth target with the most depth-writing draws in
  the frame, copied to R32F with the existing depth-copy compute pass.
  Standard Z (cleared to 1).
- **Jitter:** the scene's geometry - draws that test against the main depth
  (DB_DEPTH_CONTROL Z_ENABLE), except screen-space quads (the deferred
  lights: 4-vertex strips / rect lists) - gets a sub-pixel shift of its
  Vulkan viewport, Halton(2, 3) over 16 frames. That jitters every draw
  whatever matrices its shader uses, with no constant-buffer rewrite.
  Shadows, the half-res pass and post-processing are not jittered.
- **Camera:** the 864-byte per-view constant buffer the scene's vertex
  shaders bind, found by its signature (+0x010 the target size, +0x0D0 a
  perspective projection). Rows, used as M*v: +0x0D0 projection, +0x2D0
  inverse view (camera position in column 3), +0x320 ViewProj without the
  translation (geometry is camera-relative). Checked at run time
  (`dlaa: camera found ... ViewProj check ok`).
- **Motion vectors:** `shaders/dlaa_mv.comp`, camera only: previous ViewProj
  x the camera's move x inverse of this frame's ViewProj, built on the CPU in
  double; each pixel unjittered, reprojected, previous minus current in
  pixels. A camera cut (a jump or a sharp turn) resets DLSS's history.
  Optional object motion vectors add animated-mesh movement; see
  [DLSS development](dlss-development.md) for the current SR, frame-generation
  and object-motion settings and validation limits.
- **F12** also writes one frame of constant buffers (`cbs-<flip>.txt/.bin`)
  beside the dump; `tools/dlaa_cbscan.py` searches them for projection
  matrices. That is how the camera buffer was found.

## ngx_bridge (tools/ngx_bridge)

NVIDIA's NGX library (`nvsdk_ngx_s.lib`) is an MSVC static library; bbhost's
Windows build is Clang + mingw-w64, so they cannot link. `ngx_bridge.dll` is
a small MSVC build that links the SDK and exports plain C functions
(`ngx_bridge.h`). bbhost loads it with `LoadLibrary` from the exe's folder,
asks it for the Vulkan extensions NGX needs before creating the instance and
device (added only when available, device extensions only on NVIDIA), and
otherwise talks to it through the C API. NGX logs go to `logs/ngx/`.

```
cmake -S tools/ngx_bridge -B build/ngx_bridge -G "Visual Studio 17 2022" -A x64 ^
      -DDLSS_SDK_DIR=<DLSS SDK> -DVULKAN_HEADERS_DIR=<Vulkan-Headers>
cmake --build build/ngx_bridge --config Release
```

The DLSS SDK (https://github.com/NVIDIA/DLSS) is not in this repository; it
is under NVIDIA's RTX SDK licence.
