#!/bin/bash
# The continuous-integration build (.github/workflows/build.yml), and a
# from-scratch Linux build on a stock Ubuntu 24.04: the packages, the Vulkan
# headers (24.04's 1.3.275 are older than the renderer's), SDL3 and the movie
# decoders of ffmpeg built static into a prefix as tools/linux_portable_build.sh
# does, then bbhost, its tools and its tests, and ctest - the HLSL printer's
# test compiling with the DXC the packages carry (tools/fetch_dxc.sh, its
# release checked against the pinned SHA-256 on every run). No game files: the
# tests that read them skip (tests/test_app0.h).
#
#   tools/ci_build.sh                 everything, in build-ci/
#   CI_WORK=/somewhere tools/ci_build.sh
#   CI_APT=0                          the packages are already there
#   CI_CMAKE_ARGS="-DBBHOST_RELEASE=ON"   more configure arguments (a release build)
#   CI_PORTABLE=1                     link libstdc++ and libgcc statically, as
#                                     tools/linux_portable_build.sh does: the
#                                     binary tools/package_linux.sh packages
#
# The prefix (CI_WORK/prefix) is what the workflow caches between runs.
set -eu
cd "$(dirname "$0")/.."
SRC=$PWD
W=${CI_WORK:-$SRC/build-ci}
P=$W/prefix
VK_TAG=${VK_TAG:-v1.4.350}
SDL_TAG=${SDL_TAG:-release-3.4.16}
FFMPEG_VER=${FFMPEG_VER:-7.1.1}
JOBS=$(nproc)
# A prefix restored from a cache made in another directory (the repository's
# name is in the runner's path, and it can change) names
# that directory in its pkg-config files and CMake packages, and CMake refuses
# the paths that do not exist here. Such a prefix is built again.
if [ -f "$P/lib/pkgconfig/libavcodec.pc" ] && ! grep -q "^prefix=$P\$" "$P/lib/pkgconfig/libavcodec.pc"; then
    echo "the cached prefix was made in another directory ($(sed -n 's/^prefix=//p' "$P/lib/pkgconfig/libavcodec.pc")); building it again"
    rm -rf "$P"
fi
mkdir -p "$W/src" "$P"

if [ "${CI_APT:-1}" = 1 ]; then
    SUDO=$(command -v sudo || true)
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -q > "$W/apt-update.log" 2>&1
    $SUDO apt-get install -y -q --no-install-recommends clang lld llvm cmake ninja-build pkg-config make nasm curl \
        git libstdc++-14-dev libcurl4-openssl-dev libvulkan-dev zlib1g-dev spirv-tools python3 xz-utils \
        ca-certificates libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev libxss-dev \
        libxtst-dev libxkbcommon-dev libwayland-dev wayland-protocols libegl-dev libdecor-0-dev libpulse-dev \
        libasound2-dev libpipewire-0.3-dev libdbus-1-dev libudev-dev libdrm-dev libgbm-dev \
        > "$W/apt-install.log" 2>&1 || { tail -20 "$W/apt-install.log"; exit 1; }
fi
# libstdc++ 14: the HLE libc calls std::acosf and friends, which 13 lacks.
export CC=clang CXX=clang++

fetch() {  # fetch NAME URL DIR-GLOB
    if ! ls -d "$W/src"/$3 > /dev/null 2>&1; then
        echo "fetching $1"
        curl -sSL -m 600 -o "$W/src/$1.tar" "$2"
        tar xf "$W/src/$1.tar" -C "$W/src" --no-same-owner
        rm -f "$W/src/$1.tar"
    fi
}

# Vulkan headers, newer than the distribution's.
if [ ! -f "$P/include/vulkan/vulkan_core.h" ]; then
    fetch vulkan-headers "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$VK_TAG.tar.gz" 'Vulkan-Headers-*'
    v=$(ls -d "$W"/src/Vulkan-Headers-* | head -1)
    cmake -S "$v" -B "$W/vk-headers" -G Ninja -DCMAKE_INSTALL_PREFIX="$P" > "$W/vk-headers.log"
    cmake --install "$W/vk-headers" > /dev/null
fi

# SDL3, static (its window and audio backends are loaded at run time).
if [ ! -f "$P/lib/libSDL3.a" ]; then
    fetch sdl3 "https://github.com/libsdl-org/SDL/archive/refs/tags/$SDL_TAG.tar.gz" 'SDL-*'
    s=$(ls -d "$W"/src/SDL-* | head -1)
    cmake -S "$s" -B "$W/sdl3" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
        -DCMAKE_INSTALL_LIBDIR=lib -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TEST_LIBRARY=OFF -DSDL_TESTS=OFF \
        -DSDL_EXAMPLES=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$W/sdl3-configure.log"
    cmake --build "$W/sdl3" > "$W/sdl3-build.log"
    cmake --install "$W/sdl3" > /dev/null
fi

# ffmpeg, only the movie decoders (tools/win_deps.sh has the reasons), static.
if [ ! -f "$P/lib/libavcodec.a" ]; then
    fetch ffmpeg "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz" 'ffmpeg-*'
    f=$(ls -d "$W"/src/ffmpeg-* | head -1)
    mkdir -p "$W/ffmpeg"
    (cd "$W/ffmpeg" && "$f/configure" --prefix="$P" --cc=clang --cxx=clang++ --enable-static --disable-shared \
        --enable-pic --disable-programs --disable-doc --disable-network --disable-everything \
        --enable-decoder=h264,hevc,aac,mp3 --enable-demuxer=mov,mp3,aac --enable-parser=h264,hevc,aac \
        --enable-protocol=file --enable-swscale --enable-swresample --disable-avdevice --disable-avfilter \
        --disable-postproc --disable-debug --disable-vaapi --disable-vdpau --disable-xlib --disable-libdrm \
        --disable-iconv --disable-zlib --disable-bzlib --disable-lzma --extra-cflags=-O2 \
        > "$W/ffmpeg-configure.log" 2>&1 \
        && make -j"$JOBS" > "$W/ffmpeg-build.log" 2>&1 && make install > "$W/ffmpeg-install.log" 2>&1) \
        || { tail -20 "$W/ffmpeg-configure.log" "$W/ffmpeg-build.log"; exit 1; }
    sed -i 's/ -latomic/ -l:libatomic.a/' "$P"/lib/pkgconfig/*.pc
fi

# DXC's release library, as the packages carry it: kept in the prefix, checked
# against its pins every run.
DXC_CACHE=$P/dxc-cache tools/fetch_dxc.sh linux "$P/dxc" > "$W/dxc.log" 2>&1 || { cat "$W/dxc.log"; exit 1; }
tail -1 "$W/dxc.log"

# bbhost, its tools and tests.
export PKG_CONFIG_PATH=$P/lib/pkgconfig
static=""
[ "${CI_PORTABLE:-0}" = 1 ] && static="-static-libstdc++ -static-libgcc"
cmake -S "$SRC" -B "$W/build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="$P" \
    -DVulkan_INCLUDE_DIR="$P/include" -DCMAKE_EXE_LINKER_FLAGS="$static -fuse-ld=lld" \
    -DCMAKE_MODULE_LINKER_FLAGS="${static#-static-libstdc++ } -fuse-ld=lld" ${CI_CMAKE_ARGS:-} > "$W/bbhost-configure.log" \
    || { tail -30 "$W/bbhost-configure.log"; exit 1; }
grep -E "^-- bbhost" "$W/bbhost-configure.log" || true
cmake --build "$W/build" -j"$JOBS" 2>&1 | tee "$W/bbhost-build.log" | grep -E "error|warning: unused|FAILED" | head -40 || true
[ "${PIPESTATUS[0]}" = 0 ] || { echo "build failed (log: $W/bbhost-build.log)"; exit 1; }
# BBHOST_HLSL_DXC: tests/gcn_hlsl_test.cpp compiles every lift with that DXC
# and fails when it does not load.
(cd "$W/build" && BBHOST_HLSL_DXC="$P/dxc/libdxcompiler.so" ctest --output-on-failure)
echo "ci build: $W/build/bbhost"
