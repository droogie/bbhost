# Building

bbhost builds with CMake on x86-64 Linux. The Windows executable is
cross-compiled from Linux with Clang and the mingw-w64 runtime; MSVC is not
supported, because the host needs `__attribute__((sysv_abi))` to call into
the game's code.

## Linux

Requirements:

- Clang or GCC with C++20, CMake 3.20+, Ninja or Make
- libstdc++ 14 or newer
- recent Vulkan headers (Ubuntu 24.04's 1.3.275 are too old; CI uses 1.4.350)
  and the Vulkan loader
- zlib (required); libcurl, SDL3, ffmpeg (libavcodec, libavformat, libavutil,
  libswscale, libswresample) and SPIRV-Tools are optional but needed for a
  usable build: networking, windowing, audio, movies and shader optimisation
  are compiled out without them

On Arch Linux the packages are `vulkan-headers vulkan-icd-loader sdl3 ffmpeg
curl zlib spirv-tools`. On Ubuntu 24.04 the distribution's Vulkan headers and
SDL are too old; `tools/ci_build.sh` builds them into a prefix and is the
reference for a from-scratch build there.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

The result is `build/bbhost`, plus the official plugins in `build/plugins/`
and the developer tools (`gcn2spv`, `gcndis`, `drawreplay`, ...).

`-DBBHOST_RELEASE=ON` makes a release build, which by default checks at start
whether a newer release is out and offers to open its page in the browser.

## Windows (cross-compiled)

Requirements on the Linux machine: Clang, lld and llvm tools, the mingw-w64
sysroot (`x86_64-w64-mingw32`), and wine for running the tests and the smoke
run.

```sh
tools/win_deps.sh       # once: zlib, SDL3, libcurl (schannel) and ffmpeg for the target
cmake -S . -B build-win -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-toolchain.cmake \
      -DCMAKE_PREFIX_PATH=$PWD/build/win-deps/prefix -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-win
```

The result is `build-win/bbhost.exe`, a single static executable that needs
only the system DLLs and the GPU driver's `vulkan-1.dll`. It is linked at a
fixed base (`0x600000000`) because the game's memory windows are reserved at
fixed addresses before anything else runs.

`tools/win_syntax_check.sh <file>` checks one source file for the Windows
target while editing; `tools/win_tests.sh` builds and runs the low-level host
tests (calling-convention thunks, TLS, `setjmp`/`longjmp`, `va_list`) under
wine.

## macOS (in progress)

`tools/build_macos.sh` builds an x86_64 bbhost, which Rosetta 2 runs on Apple
Silicon, against the LunarG Vulkan SDK (`~/VulkanSDK/<version>/macOS`, or
`VULKAN_SDK`): Vulkan goes through MoltenVK, the SDK's KosmicKrisp being
arm64-only. The build tree runs as it is: bbhost finds the SDK's loader by its
rpath, and the loader finds MoltenVK and the validation layer through the
manifests CMake writes into `vulkan/icd.d` and `vulkan/explicit_layer.d` beside
the executable. A bbhost started anywhere else needs
`VK_DRIVER_FILES=<sdk>/share/vulkan/icd.d/MoltenVK_icd.json`, and
`VK_LAYER_PATH=<sdk>/share/vulkan/explicit_layer.d` for
`BBHOST_VK_VALIDATE=1`, unless the SDK was installed system-wide.

## Tests

```sh
cd build && ctest --output-on-failure
```

Tests that read game files skip unless `BBHOST_APP0` names the game folder
(the one that contains `dvdroot_ps4`). No game file is ever part of the
repository.

`cmake --build build --target smoke` boots the game to the title screen and
checks the log for crashes and unimplemented system calls; it needs a
configured `bbhost.toml` with the game paths (see [running.md](running.md)).
The Windows build has the same target under wine (`build-win`, target
`smoke`).

`build/gcn2spv --check <dvdroot_ps4>/shader/*.dcx` translates and validates
every shader in the game's shader bundles.

## Packages

| Target or script | Produces |
|---|---|
| `cmake --build build-win --target package-win` | `build/win/bbhost-win-<rev>.zip`, and the unstripped `.debug.exe` for symbolising crash reports |
| `tools/package_linux.sh <bbhost> <outdir>` | a Linux tarball (built with `tools/linux_portable_build.sh` for a portable binary) |
| `tools/package_steamdeck.sh` | the Linux package with the Steam Deck's settings |
| `tools/package_plugins.sh` | the official plugins with their signatures |
| `tools/package_playtest.sh` | tester kits for a playtest on a dev server, built by hand |

The release workflow (`.github/workflows/release.yml`) builds the Windows,
Linux and Steam Deck packages from a `v*` tag, signs `SHA256SUMS` and the
official plugins with the release key, and publishes the release that bbhost's
update check looks for. The bare executables it attaches are named the same in
every release (`bbhost.exe`, and `bbhost` for Linux), so a shortcut or a Steam
entry pointing at one keeps working when it is replaced. The packages pick no
server, so they play on the live server (`https://thehuntersdream.com`),
bbhost's default; `tools/check_release_bundle.sh` fails a release whose
packages would go anywhere else, or that carries a playtest kit.

## Continuous integration

`.github/workflows/build.yml` runs `tools/ci_build.sh` (a Linux build from
scratch on Ubuntu 24.04, with the tests) and the Windows cross build on every
push and pull request.
