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
- for the HLSL shader path, the default, DXC's `libdxcompiler.so`, which bbhost
  loads at run time: the one `BBHOST_HLSL_DXC` names, else the one beside the
  executable, else the system's. `tools/fetch_dxc.sh linux build` puts the
  packages' pinned library beside `build/bbhost`. Without one, every shader
  keeps the lifter's SPIR-V

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

`-DBBHOST_LTO=ON` builds every target with ThinLTO (`-flto=thin`), so the
command processor's calls across files (the render path into the buffer
shadow and the GPU layer) can be inlined: about 4-5% more frame rate where the
command processor is the limit (Kyo's Central Yharnam save on a Radeon 8060S,
~137 to ~143-145 fps). Off by default. It applies to the first configure of a
build directory, as the other `-D` options do.

`-DBBHOST_PGO=generate` builds an instrumented bbhost (Clang's IR profiles;
MSYS2 needs `mingw-w64-ucrt-x86_64-compiler-rt`, the version of its clang).
Run it through the places that matter with
`LLVM_PROFILE_FILE=<dir>/run-%p.profraw`; a timed exit (`BBHOST_EXIT_SECONDS`)
or closing the window writes the profile - bbhost writes it itself, since it
ends the process without the runtime's exit handler. Then
`llvm-profdata merge -o bbhost.profdata <dir>/*.profraw`, and
`-DBBHOST_PGO=<absolute path>/bbhost.profdata` (with `-DBBHOST_LTO=ON`) builds
the optimized one in a build directory of its own. With two headless runs at
Kyo's Central Yharnam save (640x360 and 1080p) as the profile, where the
command processor is the limit on a Radeon 8060S: 148 and 136 fps against 134
and 124 for the same build without it, alternating (+10%). A profile outlives
small changes: a function it no longer matches is built as without one.

## Windows (cross-compiled)

Requirements on the Linux machine: Clang, lld and llvm tools, the mingw-w64
sysroot (`x86_64-w64-mingw32`), and wine for running the tests and the smoke
run.

```sh
tools/win_deps.sh       # once: zlib, SDL3, libcurl (schannel), ffmpeg and SPIRV-Tools for the target
cmake -S . -B build-win -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-toolchain.cmake \
      -DCMAKE_PREFIX_PATH=$PWD/build/win-deps/prefix -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-win
```

The result is `build-win/bbhost.exe`, a single static executable that needs
only the system DLLs and the GPU driver's `vulkan-1.dll` (and, for the HLSL
shader path, `dxcompiler.dll` beside it, which the package carries).
SPIRV-Tools gives it the SPIR-V validator the HLSL path uses; shader liveness,
which the Linux build also takes from SPIRV-Tools, is left out of the Windows
build until it has been run there. It is linked at a
fixed base (`0x600000000`) because the game's memory windows are reserved at
fixed addresses before anything else runs.

`tools/win_syntax_check.sh <file>` checks one source file for the Windows
target while editing; `tools/win_tests.sh` builds and runs the low-level host
tests (calling-convention thunks, TLS, `setjmp`/`longjmp`, `va_list`) under
wine.

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
| `tools/package_hlsl_sweep.sh` | a kit that runs the HLSL path's driver-acceptance sweep (`gcn2hlsl --sweep`) on another machine's GPU, built by hand ([rendering.md](rendering.md)) |

Every package carries the DirectX Shader Compiler for the HLSL shader path
beside the executable: `libdxcompiler.so` (Linux, Steam Deck) or the x64
`dxcompiler.dll` (Windows) from DXC's own GitHub release, unmodified, with its
licences in `licenses/dxc` (`third_party/dxc`). `tools/fetch_dxc.sh` pins the
release by tag and checks both the archive and the library by SHA-256; the
CI tests compile the HLSL printer's tests with the same library, and
`tools/check_release_bundle.sh` fails a release without it or with DXC's
DXIL validator (`dxil.dll`, which SPIR-V does not need and whose licence is
not open). A new DXC version is a new pin and the HLSL path's checks run
again with it.

The release workflow (`.github/workflows/release.yml`) builds the Windows,
Linux and Steam Deck packages from a `v*` tag, signs `SHA256SUMS` and the
official plugins with the release key, and publishes the release that bbhost's
update check looks for. The bare executables it attaches are named the same in
every release (`bbhost.exe`, and `bbhost` for Linux), so a shortcut or a Steam
entry pointing at one keeps working when it is replaced. The packages pick no
server, so they play on the live server (`https://thehuntersdream.com`),
bbhost's default; `tools/check_release_bundle.sh` fails a release whose
packages would go anywhere else, or that carries a playtest kit.

The Windows and Linux packages (and so the Steam Deck kit) carry the pipeline
list beside the executable, `pipeline-list.bin`: every pipeline a tour of the
whole game drew, which a first start builds before the player reaches it
([full-tour.md](full-tour.md)). It is an asset of its own release, and
`tools/fetch_pipeline_list.sh` downloads it pinned by tag and SHA-256 (kept in
`build/pipeline-list`). `BBHOST_PACKAGE_PIPELINE_LIST=<file>` packs a local
list instead, `BBHOST_PIPELINE_LIST_FETCH=0` leaves it out (a development
kit), and `tools/check_release_bundle.sh` fails a release whose packages lack
the pinned list. The pin is `pipeline-list-v3`, a pre-release of its own with
1,199 pipelines (8 of them the GPU particles' geometry-shader pipelines) and
10 compute records; with the pin left empty, the packages carry no list and
the check asks for none.

## Continuous integration

`.github/workflows/build.yml` runs `tools/ci_build.sh` (a Linux build from
scratch on Ubuntu 24.04, with the tests) and the Windows cross build on every
push and pull request.
