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
- for DLSS, NVIDIA's NGX SDK (its headers and `libnvsdk_ngx.a`), which the
  configure step fetches from NVIDIA's DLSS repository with
  `tools/fetch_ngx.sh`, each file pinned by SHA-256 (kept in `build/ngx`,
  `NGX_CACHE` elsewhere). It builds `build/libbbhost_ngx_bridge.so`, bbhost's
  NGX bridge (`src/ngx_bridge`, MIT, not GPL: the only thing NVIDIA's library
  is linked into), which bbhost loads at run time from beside itself.
  `-DBBHOST_NGX=OFF`, or no network, builds without it: DLSS is then off and
  everything else is the same. `tools/fetch_ngx.sh runtime build` puts
  NVIDIA's DLSS runtime beside `build/bbhost`, for DLSS on an RTX card

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
shadow and the GPU layer) can be inlined: about 3-5% more frame rate where the
command processor is the limit (Kyo's Central Yharnam save on a Radeon 8060S,
~137 to ~143-145 fps; on an RTX 4070 with two Xeon Gold 6230 at 1080p, Old
Yharnam ~100 to ~104 fps and the Cainhurst courtyard ~107 to ~110). Off by
default; the release packages (Windows, Linux and the Steam Deck kit) are
built with it. It applies to the first configure of a build directory, as the
other `-D` options do.

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

The release's Windows build is built with the repository's profile,
`pgo/bbhost.proftext` (`CI_PGO=1 tools/ci_windows.sh`, the release workflow):
LLVM's text form, which any later `llvm-profdata` reads, with the source tree
it was recorded in written `@SRC@` - a file-local function's name carries its
source's absolute path (about 40% of them), so a profile used from another
checkout would miss those. `tools/pgo_store.sh <profdata> <tree it was built
from>` stores a new one; `tools/pgo_use.sh <out.profdata> [tree]` makes the
`.profdata` for a tree. Record it again before a release (an instrumented
build from a clean tree, headless runs in the places that limit the frame -
Kyo's Central Yharnam house, the clinic, the burning Hunter's Dream - with
`BBHOST_RUN_COMMIT=0 BBHOST_GXVK_BUFFERS=0`: a profile with the run commit's
parts in it was no faster) and check it against the build without it. The
Linux builds stay without: Ubuntu 24.04's Clang 18 is not the profile's.

A ThreadSanitizer build (`-DCMAKE_CXX_FLAGS=-fsanitize=thread`, the same in
`CMAKE_C_FLAGS` and the linker flags) runs with the write watch off unless
`BBHOST_WRITE_WATCH=1`: a write fault inside one of ThreadSanitizer's atomics on
guest memory makes it wait on its own lock, which stopped such builds at the
first loading screen.

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

`bbhost --compile-selftest DIR` (ctest's `shader_compile_e2e`) runs the setup
window's **Compile shaders now** end to end: bbhost started again as the
window starts it, with children that pretend the compile (no GPU, no game
files; the folder, eboot and configuration they use are made in `DIR`), whose
lines arrive, end without a newline, are lost after the first ones, or never
come. The Windows build runs it under wine the same way; started with no
file or pipe for its output, it opens a console as a double-clicked
`bbhost.exe` does.

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

The Windows package carries the upscalers' runtime beside `bbhost.exe`
(`tools/fetch_upscalers.sh`): AMD's FidelityFX DLLs and NVIDIA's
`nvngx_dlss.dll`, each from its maker's repository at a pinned tag and
SHA-256 with its licence, and Numity1504's `upscaler_bridge.dll`. The Linux
package carries DLSS the same way: the NGX bridge the build made and NVIDIA's
`libnvidia-ngx-dlss.so.310.9.16` (`tools/fetch_ngx.sh runtime`), with their
licences in `licenses/dlss`, `licenses/ngx_bridge` and
`licenses/upscaler_bridge`; the Steam Deck kit leaves them out. None of
NVIDIA's or AMD's files is in the repository, and bbhost itself links none of
their code. `tools/check_release_bundle.sh` fails a release whose Windows or
Linux package lacks them, or whose Steam Deck kit has DLSS's.

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
