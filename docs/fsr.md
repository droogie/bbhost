# AMD FidelityFX Super Resolution (FSR) Upscaling & FG

bbhost includes independent Vulkan adapters for AMD FidelityFX Super Resolution:
- **FSR 3.1.5 upscaler**: analytical temporal upscaling (Native AA, Quality, Balanced, Performance, Ultra Performance).
- **FSR 3.1.6 frame generation**: analytical optical flow and frame interpolation.
- **FSR 4 (source-v07)**: experimental INT8/DOT4 neural upscaling model.

## Requirements

### FSR 3.1.5 & FSR 3.1.6
- Vulkan 1.3 capable GPU.
- Support for 16-bit storage / float16.
- Frame generation additionally requires `VK_KHR_compute_shader_derivatives`
  with linear derivative groups for this provider's embedded optical-flow shaders.
  Unsupported devices retain ordinary presentation.

### FSR 4 (source-v07)
- Vulkan 1.3 with:
  - `shaderFloat16` and `shaderInt8` (Vulkan 1.2)
  - `shaderInt16` (Vulkan 1.0)
  - `shaderIntegerDotProduct` (Vulkan 1.3)
  - `shaderStorageImageExtendedFormats` (Vulkan 1.0)
- Neural network model shaders & weights installed in `fsr4_shaders/` (or specified by `BBHOST_FSR4_ASSETS` / `BB_FSR4_DIR`).
- If hardware features or asset files are unavailable, FSR 4 automatically logs a message once and falls back cleanly to FSR 3.1.5.

## Selecting providers

Setup's Start-up and display section and the F10 menu offer separate
reconstruction and frame-generation providers. Both take effect after restart.
Reconstruction offers Off, Native AA, Quality, Balanced, Performance and Ultra
Performance. FSR frame generation offers Off or 2x; it also works with
reconstruction off. Object motion vectors feed either provider when enabled.

The existing `[dlss]` configuration section remains compatible:

```toml
[dlss]
upscaler_backend = "fsr3" # "dlss", "fsr3", or "fsr4"
mode = "quality"
frame_generation_backend = "fsr3" # "dlss" or "fsr3"
frame_generation = true
frame_generation_factor = 2
object_motion = true
```

`BBHOST_UPSCALER` and `BBHOST_FG_BACKEND` override those providers for a run.
This integration uses analytical FSR 3 frame generation, including when FSR 4
reconstruction is selected; it does not provide FSR 4 ML frame generation.
The optional source-v07 FSR 4 model is experimental and is not the newer FSR
4.1.1 runtime.

## Installing FSR 4 Assets on Windows

Run the installation script to download and verify the model bundle:

```powershell
.\tools\win\Install-FSR4.ps1
```

or via the command prompt helper:

```cmd
tools\win\Install-FSR4.bat
```

The script downloads the MIT-licensed FSR 4 v07 INT8/DOT4 shader assets from the upstream Q2RTX repository and verifies file sizes and SHA-256 hashes against the manifest.

## Validation and limits

See [FSR validation](fsr-validation.md) for the local GPU pixel checks,
paired validation captures and gameplay measurements. Rapid camera sweeps at
30 game FPS can show visible interpolation warping. Passing pixel and cadence
checks does not establish artifact-free image quality or certify input latency.
FSR generation remains an experimental provider pending wider hardware and
long-session testing. Multiplayer should keep the game's simulation at 60 FPS;
generated frames change presentation only.
