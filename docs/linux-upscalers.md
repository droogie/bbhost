# DLSS, FSR and frame generation on Linux - handoff

For droogie (Linux, NVIDIA, a screen above 1080p). Written 2026-11-07 on a Windows box with an AMD
iGPU and no Linux, so **none of this has run on Linux yet**. What is in the tree, how to build and
test it, and what is left: DLSS should work as it stands; FSR 3 and frame generation need one more
piece, a Linux build of AMD's FidelityFX library, whose blocker is a shader permutation generator.

## State per piece

| Piece | Windows | Linux, after this branch |
|---|---|---|
| Upscaler bridge (Numity1504, MIT) | `upscaler_bridge.dll`, MSVC | source in `third_party/upscaler_bridge/src`, built as `upscaler_bridge.so` by the portable build |
| DLSS (NVIDIA) | `nvngx_dlss.dll` 310.9.16 | NVIDIA ships Linux files: `libnvsdk_ngx.a` linked into the bridge, `libnvidia-ngx-dlss.so.310.9.16` shipped beside bbhost. **Should work: please test** |
| FSR 3 upscaling (AMD) | AMD's signed `amd_fidelityfx_*_vk.dll` | AMD ships **no** Linux files. Needs `libamd_fidelityfx_vk.so` built from the FidelityFX SDK 2.4.0's MIT source - **not done** |
| FSR 4 | AMD's ML DLL | Windows only, closed. Not possible |
| Frame generation (AMD) | `amd_fidelityfx_framegeneration_vk.dll` | the same `libamd_fidelityfx_vk.so` - **not done** |
| TAA | bbhost's own | unchanged, works everywhere |

Without a file bbhost logs why and keeps the method off; Graphics > Upscaling still offers TAA.

## What the branch changed

- `third_party/upscaler_bridge/src/upscaler_bridge.cpp` - the bridge's source with a POSIX side:
  `dlopen`/`dladdr`/`realpath`, UTF-8 to `wchar_t` (NGX's paths are `wchar_t` on Linux too), the DLSS
  runtime found by name in the bridge's folder (`libnvidia-ngx-dlss.so.<version>`, the newest; the
  version from the name), FSR as `libamd_fidelityfx_vk.so`, `ub_get_api` the one visible symbol. No
  SEH on Linux, so a crash inside NVIDIA's or AMD's code is not caught there.
  `third_party/upscaler_bridge/ffx` has the FFX headers the bridge compiles against.
- `src/host/upscaler.cpp` - the real path on `_WIN32 || __linux__`; loads `upscaler_bridge.so` beside
  bbhost (`load_bridge`). The BBsuite layer check stays Windows-only.
- `src/host/framegen.cpp` - on Linux loads `libamd_fidelityfx_vk.so` beside bbhost (the same five
  entry points: ffxCreateContext/DestroyContext/Configure/Query/Dispatch).
- `tools/build_upscaler_bridge_linux.sh` - `--fetch` gets NVIDIA's five NGX headers,
  `libnvsdk_ngx.a` and `libnvidia-ngx-dlss.so.310.9.16` from `github.com/NVIDIA/DLSS/raw/v310.9.16`
  at pinned SHA-256 (cache `build/upscalers-linux`); `DIR` builds the bridge into DIR with the NGX
  runtime and licences beside it. `FFX_LINUX_SO=path` adds a FidelityFX library.
- `tools/linux_portable_build.sh` fetches outside the chroot and builds into
  `build-linux-portable/build/upscalers` inside (`UPSCALERS=0` skips it).
  `tools/package_linux.sh` copies them beside `bbhost` with `licenses/`, and the README gets an
  "Upscalers" section.

Checked on Windows only: the ported bridge still builds with MSVC and the DLSS SDK, and the POSIX code
compiles against stub headers (C++ errors, not glibc ones). Not checked: a real Linux link
(`libnvsdk_ngx.a`'s relocations are PIC - PLT32/GOTPCREL/TLSGD - so it should link into a `.so`), and
DLSS at run time.

## Build and test DLSS

```bash
tools/linux_portable_build.sh
```

```bash
tools/package_linux.sh
```

Then from the package (or with `upscaler_bridge.so` and `libnvidia-ngx-dlss.so.310.9.16` beside a
local `build/bbhost`):

1. The start log should have `upscaler: .../upscaler_bridge.so loaded` and a line
   `upscaler: bridge 1.1.0; DLSS available 310.9.16.0; FSR not available (...)`. If DLSS is not
   available, the reason is in brackets (driver too old, not an RTX card, NGX's own lines start
   `upscaler: NGX:`).
2. Graphics > Upscaling = DLSS, at each quality and at native (DLAA). The window above 1080p
   (1440p / 4K) is the case nobody has tried with DLSS: the render size is the Resolution over the
   quality, the output the window's. Look for a black or smeared picture, a wrong-sized image, or
   `upscaler: evaluate failed` (then the plain picture stays for the session).
3. DLSS presets J/K/L/M (Graphics > DLSS preset) - the bridge maps them to NGX's render presets.
4. Turn the camera fast and watch thin things (fences, hair) for ghosting: motion and jitter are the
   same code as on Windows, where they were checked on FSR 4.
5. `BBHOST_UPSCALE=dlss` forces it over the setting; `BBHOST_UPSCALER=0` loads no bridge.

## Left to do: libamd_fidelityfx_vk.so (FSR 3 + frame generation)

Source: FidelityFX SDK **v2.4.0** (`github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK`, tag
`v2.4.0`), `Kits/FidelityFX/{api,backend/vk,upscalers/fsr3,framegeneration}` - all on the MIT list in
`Kits/FidelityFX/docs/license.md` (the text is in `third_party/fidelityfx/LICENSE-source.md`). Use
2.4.0, not 1.1.4: 1.1.4's ffx-api has other structure type ids and no PrepareV2, and bbhost's
`third_party/fidelityfx` headers are 2.4.0's.

### 1. A shader permutation generator (the blocker)

`upscalers/fsr3/internal/*_shaderblobs.cpp` and `framegeneration/fsr3/internal/*_shaderblobs.cpp`
include `<pass>_permutations.h`, `<pass>_wave64_permutations.h`, `<pass>_16bit_permutations.h` and
`<pass>_wave64_16bit_permutations.h` for each HLSL pass in `internal/shaders/`. AMD generates them with
its FidelityFX_SC tool, which is in neither 2.4.0 nor any Linux form (1.1.4 has a Windows exe and its
source in `sdk/tools/ffx_shader_compiler/src`). Proposal: `tools/ffx_permutations.py`, run in the
portable build's chroot with DXC (`tools/fetch_dxc.sh linux` already gives `libdxcompiler.so`). What a
header must hold - see `ffx_sc.cpp` (`WritePermutationHeader` and the HLSL compiler's reflection
writers) in 1.1.4's tool for the exact text:

- one `g_<pass>_<hash>_size` / `g_<pass>_<hash>_data[]` per unique SPIR-V blob (identical blobs
  shared);
- `typedef union <pass>_PermutationKey { struct { uint32_t <OPTION> : <bits>; ... }; uint32_t index; }`,
  with the options in the order the `POPULATE_PERMUTATION_KEY` macro of that `_shaderblobs.cpp` names
  them, so `key.index` is the bitfield;
- `<pass>_PermutationInfo` with `blobSize`, `blobData`, then the reflection members the
  `POPULATE_SHADER_BLOB_FFX` macro (`api/internal/ffx_internal_types.h`) reads: `entryName`,
  `numConstantBuffers`, `numSRVTextures`, `numUAVTextures`, `numSRVBuffers`, `numUAVBuffers`,
  `numSamplers`, `numRTAccelerationStructures`, and for each kind `<kind>Names`, `<kind>Bindings`,
  `<kind>Counts`, `<kind>Spaces`;
- `g_<pass>_IndirectionTable[2^bits]` - key index to row (0 for keys never built) - and
  `g_<pass>_PermutationInfo[]`.

The option lists (`-DNAME={0,1}`) and fixed defines are in 1.1.4's
`sdk/include/FidelityFX/gpu/<effect>/CMakeCompile*Shaders.txt`; check each against 2.4.0's
`POPULATE_PERMUTATION_KEY`, which wins. DXC flags for the four variants: `-spirv -fspv-target-env=vulkan1.2
-T cs_6_2 -E CS -DFFX_GPU=1 -DFFX_HLSL=1`, `-DFFX_PREFER_WAVE64` for wave64 (the VK backend picks those
when the subgroup size allows, `backend/vk/ffx_vk.cpp`), `-enable-16bit-types -DFFX_HALF=1` for
16bit (both names are used in `api/internal/gpu`; check how FFX_PREFER_WAVE64 is spelled for SPIR-V
there). Bindings come from SPIR-V (`OpDecorate ... Binding` /
`DescriptorSet`, `OpName`), names as the resource identifiers in the HLSL, which `backend/vk/ffx_vk.cpp`
matches to the effect's resource tables. The DX12 path (FidelityFX_SC's `-reflection`) is the
reference for what each array should hold.

### 2. Win32 in the Vulkan frame generation swapchain

`framegeneration/fsr3/vk/FrameInterpolationSwapchainVK.cpp`: CRITICAL_SECTION (~18 sites),
CreateEvent/SetEvent/WaitForSingleObject, CreateThread, QueryPerformanceCounter, SetThreadPriority,
SetThreadDescription, `VkSurfaceFullScreenExclusiveWin32InfoEXT`. A small shim header
(std::mutex / condition_variable / std::thread / steady_clock) keeps the diff small. Also check
`backend/vk/ffx_vk.cpp` and `api/internal` for `wchar_t` messages, `_countof`, `swprintf`, `__declspec`.

### 3. One shared library

`api/internal/ffx_api.cpp` (the five exports) + the providers fsr3upscale, fsr3framegeneration and
framegenerationswapchain_vk + `backend/vk`, `-fvisibility=hidden` except the five, static libstdc++.
Then `FFX_LINUX_SO=/src/<path>/libamd_fidelityfx_vk.so tools/linux_portable_build.sh` packs it.

### 4. The bridge's FSR path

With no FSR 4 provider the bridge's `newest_is_4` stays false and it takes the 3.x path. Check what
`ffxQuery` for versions returns from a source-built provider list, and that the bridge's 4.1.2
version-override structure does no harm there.

### 5. Frame generation's queues

`src/host/gpu.cpp` asks the graphics family for 5 queues; frame generation needs `family_queues >= 5`
(queues 2-4: present, image acquire, async compute). NVIDIA's Linux driver has 16 in family 0, so
your machine is fine. Mesa's RADV has **1** graphics queue (several compute queues), so on AMD Linux
frame generation stays off until FG's async-compute and image-acquire queues can come from the compute
family (AMD's swapchain takes a family per queue; mind ownership of the shared images).

## Frame generation test plan (once the library exists)

Frame generation must interpolate the 3D only: the HUD and menus are taken out against the HUD-less
copy (display buffer before the HUD's first draw) and composited over every shown frame. Check: menus
and HUD never smear; the FPS counter shows about double; `framegen: N frames shown for the last 600`
lines in the log about 1200; a Resolution above the window (render size above display) still
generates (`framegen: AMD's swapchain ... render sizes up to WxH`). It needs a visible window.

## Licences

- NVIDIA DLSS: the RTX SDK licence in `third_party/dlss/LICENSE.txt`, the same terms as the Windows
  files - only inside an application, not offered alone; NVIDIA asks to be notified before a commercial
  release, naming the platform.
- The bridge: MIT, Numity1504 (`third_party/upscaler_bridge/LICENSE.txt`); the Linux side is ours.
- FidelityFX source: MIT (`third_party/fidelityfx/LICENSE-source.md`); AMD's signed Windows DLLs stay
  under their binary licence (`LICENSE-signedbin.md`).
