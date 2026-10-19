# Building

bbhost builds with CMake on x86-64 Linux. The Windows executable is
cross-compiled from Linux with Clang and the mingw-w64 runtime; MSVC is not
supported, because the host needs `__attribute__((sysv_abi))` to call into
the game's code. The macOS build (in progress) is an x86_64 program built
on the Mac itself.

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

The macOS build compiles, links and passes its tests, but the game has not
yet been played on it: treat it as work in progress.

bbhost on macOS is an x86_64 program. The game's code is x86-64 and runs in
bbhost's own process, so on Apple Silicon Rosetta 2 runs the whole of it; an
Intel Mac runs it directly. It needs **macOS 15 or later**: the game uses AVX
and AVX2, which Rosetta runs from macOS 15 on. bbhost checks the instructions
it needs at start and stops with a message when one is missing.

Requirements:

- macOS 15+, with the Xcode Command Line Tools (`xcode-select --install`) for
  Clang, the SDK, `strip`, `dsymutil` and `codesign`; a full Xcode works too
- on Apple Silicon, Rosetta 2 (`softwareupdate --install-rosetta
  --agree-to-license`), for the build's tests and for running it
- the LunarG Vulkan SDK for macOS (https://vulkan.lunarg.com/sdk/home#mac),
  installed in `~/VulkanSDK/<version>` or named by `VULKAN_SDK=<sdk>/macOS`:
  the Vulkan loader, MoltenVK (Vulkan on Metal), the validation layer and
  SPIRV-Tools, all universal binaries. Its KosmicKrisp driver is arm64-only,
  so an x86_64 bbhost uses MoltenVK.

```sh
tools/build_macos.sh       # everything, in build-macos/; the result is build-macos/build/bbhost
CI_CMAKE_ARGS=-DBBHOST_RELEASE=ON tools/build_macos.sh   # a release build
```

The script fetches CMake and Ninja when there are none, builds the pinned
Vulkan headers, SDL3 and ffmpeg's movie decoders static for x86_64 into
`build-macos/prefix`, then bbhost, its tools and its tests, and runs ctest
(`CI_TESTS=0` skips it). It does not use Homebrew: an Apple Silicon Mac's
Homebrew installs arm64 libraries, which an x86_64 program cannot link; a
second, x86_64 Homebrew under `/usr/local` is a setup of its own; and Homebrew
refuses to build from source on a macOS newer than its Xcode (macOS 27 with
Xcode 16.2 here). Every dependency is pinned in the script instead, as
`tools/ci_build.sh` does on Linux.

The build tree runs as it is: bbhost finds the SDK's loader by its rpath, and
the loader finds MoltenVK and the validation layer through the manifests CMake
writes into `vulkan/icd.d` and `vulkan/explicit_layer.d` beside the
executable. A bbhost started anywhere else needs
`VK_DRIVER_FILES=<sdk>/share/vulkan/icd.d/MoltenVK_icd.json`, and
`VK_LAYER_PATH=<sdk>/share/vulkan/explicit_layer.d` for
`BBHOST_VK_VALIDATE=1`, unless the SDK was installed system-wide. The package
(below) carries its own loader and MoltenVK and needs neither.

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
| `tools/package_macos.sh <bbhost> <outdir>` | `bbhost-mac-<rev>.zip`, a folder with the Vulkan loader and MoltenVK beside bbhost, and the `.dSYM` for symbolising crash reports |
| `tools/package_steamdeck.sh` | the Linux package with the Steam Deck's settings |
| `tools/package_plugins.sh` | the official plugins with their signatures |
| `tools/package_playtest.sh` | tester kits for a playtest on a dev server, built by hand |

The release workflow (`.github/workflows/release.yml`) builds the Windows,
Linux, Steam Deck and macOS packages from a `v*` tag, signs `SHA256SUMS` and
the official plugins with the release key, and publishes the release that
bbhost's update check looks for. The bare executables it attaches are named
the same in every release (`bbhost.exe`, and `bbhost` for Linux), so a
shortcut or a Steam entry pointing at one keeps working when it is replaced;
macOS has none, since its bbhost runs only from its package's folder, beside
the Vulkan loader and MoltenVK. The packages pick no server, so they play on
the live server (`https://thehuntersdream.com`), bbhost's default;
`tools/check_release_bundle.sh` fails a release whose packages would go
anywhere else, or that carries a playtest kit.

## Continuous integration

`.github/workflows/build.yml` runs `tools/ci_build.sh` (a Linux build from
scratch on Ubuntu 24.04, with the tests) and `tools/build_macos.sh` (the
x86_64 build on an Apple Silicon runner, its tests run by Rosetta, after
`tools/ci_macos_setup.sh` installs Rosetta and the Vulkan SDK) on every push
and pull request; the release workflow builds the Windows cross build and
every package for each pull request too. The macOS jobs (`tests-macos`,
`package-macos`) are not yet among the checks master's ruleset requires.
