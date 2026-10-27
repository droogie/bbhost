#!/bin/bash
# Configure and build bbhost.exe on Windows in MSYS2 UCRT64 (after
# tools/win_deps_msys2.sh): build-win/bbhost.exe, static, the release's
# configuration. CMAKE_BUILD_TYPE from BUILD_TYPE (RelWithDebInfo).
#   tools/win_build_msys2.sh [extra cmake -D args]
set -eu
export PATH="$PATH:/c/Program Files/Git/cmd"  # git for cmake/version.cmake
export MSYSTEM=UCRT64
case ":$PATH:" in *:/ucrt64/bin:*) ;; *) export PATH=/ucrt64/bin:$PATH ;; esac
cd "$(dirname "$0")/.."
src=$(cygpath -m "$PWD")
b=${WIN_BUILD:-$src/build-win}
prefix=$(cygpath -m "${WIN_DEPS_ROOT:-$PWD/build/win-deps}/prefix")
if [ ! -f "$b/build.ninja" ]; then
  cmake -S "$src" -B "$b" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$src/cmake/msys2-ucrt64.cmake" \
    -DCMAKE_PREFIX_PATH="$prefix" -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}" "$@" > "$b.configure.log" 2>&1 \
    || { tail -40 "$b.configure.log"; exit 1; }
  grep -E "^-- bbhost" "$b.configure.log" || true
fi
cmake --build "$b" --target bbhost bb_plugin_region_switch bb_plugin_debug_menu bb_plugin_mutators bb_plugin_randomizer bb_plugin_boss_rush -j"$(nproc)" 2>&1 | tee "$b.build.log" | grep -E "error|FAILED|warning: unused" | head -60
[ "${PIPESTATUS[0]}" = 0 ] || { echo "build failed (log: $b.build.log)"; exit 1; }
ls -la "$b/bbhost.exe"
