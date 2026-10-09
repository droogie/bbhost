#!/bin/bash
# The macOS build: bbhost as an x86_64 program, which Rosetta 2 runs on Apple
# Silicon (the game's code is x86-64 and runs as it is, as on a PC). The
# counterpart of tools/ci_build.sh: cmake and ninja when they are missing,
# then the Vulkan headers, SDL3 and the movie decoders of ffmpeg built static
# for x86_64 into a prefix, then bbhost, its tools and its tests, and ctest.
#
# The Vulkan loader and MoltenVK come from the LunarG Vulkan SDK (universal
# binaries): VULKAN_SDK, or the newest ~/VulkanSDK/<version>/macOS. The SDK's
# KosmicKrisp driver is arm64 only, so an x86_64 bbhost uses MoltenVK.
#
#   tools/build_macos.sh              everything, in build-macos/
#   CI_WORK=/somewhere tools/build_macos.sh
#   CI_CMAKE_ARGS="-DBBHOST_RELEASE=ON"   more configure arguments
#   CI_TESTS=0                        no ctest
#
# Needs macOS 15+: Rosetta runs AVX/AVX2 from 15 on, and the game uses them.
set -eu
cd "$(dirname "$0")/.."
SRC=$PWD
W=${CI_WORK:-$SRC/build-macos}
P=$W/prefix
T=$W/tools
VK_TAG=${VK_TAG:-v1.4.350}
SDL_TAG=${SDL_TAG:-release-3.4.16}
FFMPEG_VER=${FFMPEG_VER:-7.1.1}
CMAKE_VER=${CMAKE_VER:-4.4.4}
NINJA_VER=${NINJA_VER:-1.13.2}
JOBS=$(sysctl -n hw.ncpu)
ARCH=x86_64
mkdir -p "$W/src" "$P" "$T/bin"
export PATH=$T/bin:$PATH

fetch() {  # fetch NAME URL DIR-GLOB
    if ! ls -d "$W/src"/$3 > /dev/null 2>&1; then
        echo "fetching $1"
        curl -sSL -m 600 -o "$W/src/$1.tar" "$2"
        tar xf "$W/src/$1.tar" -C "$W/src"
        rm -f "$W/src/$1.tar"
    fi
}

if ! command -v cmake > /dev/null; then
    fetch cmake "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VER/cmake-$CMAKE_VER-macos-universal.tar.gz" 'cmake-*-macos-universal'
    ln -sf "$(ls -d "$W"/src/cmake-*-macos-universal | head -1)/CMake.app/Contents/bin/cmake" "$T/bin/cmake"
    ln -sf "$(ls -d "$W"/src/cmake-*-macos-universal | head -1)/CMake.app/Contents/bin/ctest" "$T/bin/ctest"
fi
if ! command -v ninja > /dev/null; then
    echo "fetching ninja"
    curl -sSL -m 300 -o "$W/src/ninja.zip" "https://github.com/ninja-build/ninja/releases/download/v$NINJA_VER/ninja-mac.zip"
    unzip -qo "$W/src/ninja.zip" -d "$T/bin"
    rm -f "$W/src/ninja.zip"
fi

# The SDK's headers are searched last, after the prefix's pinned ones.
if [ -z "${VULKAN_SDK:-}" ]; then
    VULKAN_SDK=$(ls -d "$HOME"/VulkanSDK/*/macOS 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "$VULKAN_SDK" ] || [ ! -f "$VULKAN_SDK/lib/libvulkan.1.dylib" ]; then
    echo "no Vulkan SDK: install it from https://vulkan.lunarg.com/sdk/home#mac (or set VULKAN_SDK=<sdk>/macOS)"
    exit 1
fi
echo "Vulkan SDK: $VULKAN_SDK"

if [ ! -f "$P/include/vulkan/vulkan_core.h" ]; then
    fetch vulkan-headers "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$VK_TAG.tar.gz" 'Vulkan-Headers-*'
    v=$(ls -d "$W"/src/Vulkan-Headers-* | head -1)
    cmake -S "$v" -B "$W/vk-headers" -G Ninja -DCMAKE_INSTALL_PREFIX="$P" > "$W/vk-headers.log"
    cmake --install "$W/vk-headers" > /dev/null
fi

if [ ! -f "$P/lib/libSDL3.a" ]; then
    fetch sdl3 "https://github.com/libsdl-org/SDL/archive/refs/tags/$SDL_TAG.tar.gz" 'SDL-*'
    s=$(ls -d "$W"/src/SDL-* | head -1)
    cmake -S "$s" -B "$W/sdl3" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
        -DCMAKE_OSX_ARCHITECTURES=$ARCH -DCMAKE_INSTALL_LIBDIR=lib -DSDL_SHARED=OFF -DSDL_STATIC=ON \
        -DSDL_TEST_LIBRARY=OFF -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        > "$W/sdl3-configure.log"
    cmake --build "$W/sdl3" > "$W/sdl3-build.log"
    cmake --install "$W/sdl3" > /dev/null
fi

# ffmpeg, only the movie decoders (tools/win_deps.sh has the reasons), static,
# cross-compiled for x86_64, without X11 (configure finds a Homebrew one through
# pkg-config and links bbhost to it). Without nasm the decoders' x86 assembly is left
# out (slower movies, not wrong ones).
if [ ! -f "$P/lib/libavcodec.a" ]; then
    fetch ffmpeg "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz" 'ffmpeg-*'
    f=$(ls -d "$W"/src/ffmpeg-* | head -1)
    asm="--disable-x86asm"
    command -v nasm > /dev/null && asm=""
    mkdir -p "$W/ffmpeg"
    (cd "$W/ffmpeg" && "$f/configure" --prefix="$P" --enable-cross-compile --arch=$ARCH --target-os=darwin \
        --cc="clang -arch $ARCH" --cxx="clang++ -arch $ARCH" --enable-static --disable-shared $asm \
        --enable-pic --disable-programs --disable-doc --disable-network --disable-everything \
        --enable-decoder=h264,hevc,aac,mp3 --enable-demuxer=mov,mp3,aac --enable-parser=h264,hevc,aac \
        --enable-protocol=file --enable-swscale --enable-swresample --disable-avdevice --disable-avfilter \
        --disable-postproc --disable-debug --disable-videotoolbox --disable-audiotoolbox \
        --disable-securetransport --disable-xlib --disable-libxcb --disable-iconv --disable-zlib --disable-bzlib --disable-lzma \
        --extra-cflags=-O2 > "$W/ffmpeg-configure.log" 2>&1 \
        && make -j"$JOBS" > "$W/ffmpeg-build.log" 2>&1 && make install > "$W/ffmpeg-install.log" 2>&1) \
        || { tail -20 "$W/ffmpeg-configure.log" "$W/ffmpeg-build.log"; exit 1; }
fi

# bbhost, its tools and tests.
export PKG_CONFIG_PATH=$P/lib/pkgconfig
cmake -S "$SRC" -B "$W/build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_OSX_ARCHITECTURES=$ARCH \
    -DCMAKE_PREFIX_PATH="$P;$VULKAN_SDK" -DCMAKE_CXX_FLAGS="-idirafter $VULKAN_SDK/include" -DVulkan_INCLUDE_DIR="$P/include" -DVulkan_LIBRARY="$VULKAN_SDK/lib/libvulkan.1.dylib" \
    ${CI_CMAKE_ARGS:-} > "$W/bbhost-configure.log" \
    || { tail -30 "$W/bbhost-configure.log"; exit 1; }
grep -E "^-- bbhost" "$W/bbhost-configure.log" || true
set +e
cmake --build "$W/build" -j"$JOBS" > "$W/bbhost-build.log" 2>&1
st=$?
set -e
grep -E "error|FAILED" "$W/bbhost-build.log" | head -40 || true
[ $st = 0 ] || { echo "build failed (log: $W/bbhost-build.log)"; exit 1; }
# bbhost looks in <exe>/../patches, which here is $W/patches, not the source's.
[ -e "$W/patches" ] || ln -s "$SRC/patches" "$W/patches"
if [ "${CI_TESTS:-1}" = 1 ]; then
    (cd "$W/build" && DYLD_LIBRARY_PATH="$VULKAN_SDK/lib" VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json" \
        ctest --output-on-failure)
fi
echo "macos build: $W/build/bbhost"
