#!/bin/bash
# The Windows release build (.github/workflows/release.yml), and a
# from-scratch one on an Arch Linux machine or container: the packages
# (CI_PACMAN=1), the target's dependencies (tools/win_deps.sh), bbhost.exe
# cross-built with Clang + the mingw-w64 sysroot (cmake/mingw-toolchain.cmake),
# and the package (tools/win_package.sh, with DXC's pinned release library) in
# build/win.
#
#   tools/ci_windows.sh                       deps in build/win-deps, build in build-win
#   WIN_DEPS_ROOT=... WIN_BUILD=... tools/ci_windows.sh
set -eu
cd "$(dirname "$0")/.."
SRC=$PWD
if [ "${CI_PACMAN:-0}" = 1 ]; then
    pacman -Syu --noconfirm --needed base-devel git clang lld llvm cmake ninja nasm curl zip unzip python \
        mingw-w64-gcc mingw-w64-winpthreads vulkan-headers wine > /tmp/pacman.log 2>&1 \
        || { tail -20 /tmp/pacman.log; exit 1; }
fi
export WIN_DEPS_ROOT=${WIN_DEPS_ROOT:-$SRC/build/win-deps}
# DXC's release archive (tools/fetch_dxc.sh, which the package step runs) in
# the dependency prefix the workflow caches.
export DXC_CACHE=${DXC_CACHE:-$WIN_DEPS_ROOT/prefix/dxc-cache}
B=${WIN_BUILD:-$SRC/build-win}
tools/win_deps.sh
cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$SRC/cmake/mingw-toolchain.cmake" \
    -DCMAKE_PREFIX_PATH="$WIN_DEPS_ROOT/prefix" -DCMAKE_BUILD_TYPE=RelWithDebInfo ${CI_CMAKE_ARGS:-} > "$B.configure.log" 2>&1 \
    || { mkdir -p "$(dirname "$B")"; tail -30 "$B.configure.log"; exit 1; }
grep -E "^-- bbhost" "$B.configure.log" || true
cmake --build "$B" --target package-win 2>&1 | tee "$B.build.log" | grep -E "error|FAILED|package:|plugins" || true
[ "${PIPESTATUS[0]}" = 0 ] || { echo "build failed (log: $B.build.log)"; exit 1; }
ls -la build/win/
