#!/bin/bash
# Builds what bbhost.exe links against, for the Windows target, from Linux:
# zlib, SDL3, libcurl, ffmpeg's movie decoders and SPIRV-Tools as static
# libraries with cmake/mingw-toolchain.cmake, the Vulkan headers from this
# machine (platform-independent) and wine's import library for vulkan-1.dll
# as the loader. Everything lands in
# build/win-deps/prefix, which the host's configure takes as CMAKE_PREFIX_PATH:
#   tools/win_deps.sh                 (WIN_DEPS_ROOT: elsewhere than build/win-deps)
#   cmake -S . -B build-win -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-toolchain.cmake \
#         -DCMAKE_PREFIX_PATH=$PWD/build/win-deps/prefix -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build build-win
# SPIRV-Tools: its validator is the HLSL path's for the modules DXC compiles
# without its own (CMakeLists.txt); the optimizer is built too, for shader
# liveness, which the Windows build does not take yet.
set -eu
cd "$(dirname "$0")/.."
root=${WIN_DEPS_ROOT:-$PWD/build/win-deps}
prefix=$root/prefix
# The mingw-w64 GCC's library directory (libgcc, libstdc++), whatever its version.
mingw_gcc_dir=$(ls -d /usr/lib/gcc/x86_64-w64-mingw32/*/ | sort -V | tail -1)
mingw_gcc_dir=${mingw_gcc_dir%/}
tc=$PWD/cmake/mingw-toolchain.cmake
SDL_TAG=${SDL_TAG:-release-3.4.16}
ZLIB_TAG=${ZLIB_TAG:-v1.3.1}
mkdir -p "$root/src" "$prefix/include" "$prefix/lib"

# A source is downloaded only for a library the prefix does not have yet: a CI
# run whose prefix comes from its cache downloads nothing (a slow mirror once
# failed such a run while it fetched sources it would not use). A download that
# fails or stalls is tried again, a few times, before the build gives up.
download() {  # download URL FILE
  curl -fsSL --retry 4 --retry-delay 10 --retry-all-errors --connect-timeout 30 -m 600 -o "$2" "$1"
}

# Its own failures said and returned: it runs in a command substitution, where
# set -e does not stop it, so a failed download would otherwise surface only as
# the build's error about a missing source folder.
fetch() {  # fetch NAME URL DIR-GLOB
  local name=$1 url=$2 dir=$3 d
  if ! ls -d "$root/src"/$dir > /dev/null 2>&1; then
    echo "fetching $name" >&2
    download "$url" "$root/src/$name.tar.gz" || { echo "downloading $name from $url failed" >&2; return 1; }
    tar xzf "$root/src/$name.tar.gz" -C "$root/src" || { echo "unpacking $name failed" >&2; return 1; }
  fi
  d=$(ls -d "$root/src"/$dir 2> /dev/null | head -1)
  [ -n "$d" ] || { echo "no $dir in $root/src after fetching $name" >&2; return 1; }
  echo "$d"
}

if [ ! -f "$prefix/lib/libzlibstatic.a" ]; then
  zsrc=$(fetch zlib "https://github.com/madler/zlib/archive/refs/tags/$ZLIB_TAG.tar.gz" 'zlib-*')
  cmake -S "$zsrc" -B "$root/zlib" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" -DZLIB_BUILD_SHARED=OFF -DZLIB_BUILD_STATIC=ON -DZLIB_BUILD_EXAMPLES=OFF > "$root/zlib-configure.log"
  cmake --build "$root/zlib" > "$root/zlib-build.log"
  cmake --install "$root/zlib" > /dev/null
  rm -f "$prefix/lib/libzlib.dll.a" "$prefix/bin"/*zlib*.dll  # static only, or FindZLIB picks the import library
fi

if [ ! -f "$prefix/lib/libSDL3.a" ]; then
  ssrc=$(fetch sdl3 "https://github.com/libsdl-org/SDL/archive/refs/tags/$SDL_TAG.tar.gz" 'SDL-*')
  cmake -S "$ssrc" -B "$root/sdl3" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TEST_LIBRARY=OFF -DSDL_TESTS=OFF \
    -DSDL_EXAMPLES=OFF > "$root/sdl3-configure.log"
  cmake --build "$root/sdl3" > "$root/sdl3-build.log"
  cmake --install "$root/sdl3" > /dev/null
fi

# libcurl on schannel (Windows' own TLS), static, for sceHttp and the session service.
CURL_TAG=${CURL_TAG:-curl-8_22_0}
if [ ! -f "$prefix/lib/libcurl.a" ]; then
  csrc=$(fetch curl "https://github.com/curl/curl/releases/download/$CURL_TAG/curl-$(echo ${CURL_TAG#curl-} | tr _ .).tar.gz" 'curl-*')
  cmake -S "$csrc" -B "$root/curl" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release     -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$prefix" -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON     -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF -DBUILD_LIBCURL_DOCS=OFF -DBUILD_MISC_DOCS=OFF -DENABLE_CURL_MANUAL=OFF     -DCURL_USE_SCHANNEL=ON -DCURL_USE_LIBPSL=OFF -DUSE_LIBIDN2=OFF -DCURL_USE_LIBSSH2=OFF -DCURL_BROTLI=OFF     -DCURL_ZSTD=OFF -DCURL_DISABLE_LDAP=ON -DUSE_NGHTTP2=OFF -DCURL_ZLIB=ON > "$root/curl-configure.log"
  cmake --build "$root/curl" > "$root/curl-build.log"
  cmake --install "$root/curl" > /dev/null
fi

# ffmpeg, only what the movies need (H.264 + AAC in MP4; HEVC comes along
# because 7.1 puts the film-grain object H.264 references under it), static; its .pc
# files are what the host's configure finds through the toolchain's
# PKG_CONFIG_LIBDIR.
FFMPEG_VER=${FFMPEG_VER:-7.1.1}
if [ ! -f "$prefix/lib/libavcodec.a" ]; then
  # FFmpeg's own mirror on GitHub first - the release's tag: the release
  # tarball's tree but for git's own files and the VERSION file, whose version
  # the tree's RELEASE file gives the same - from the host every other source
  # here comes from; ffmpeg.org's tarball if that fails (its downloads failed
  # two CI runs in a row, a stall and then a reset connection).
  fsrc=$(ls -d "$root/src"/ffmpeg-* "$root/src"/FFmpeg-n* 2> /dev/null | head -1)
  if [ -z "$fsrc" ]; then
    echo "fetching ffmpeg" >&2
    if download "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n$FFMPEG_VER.tar.gz" "$root/src/ffmpeg.tar.gz"; then
      tar xzf "$root/src/ffmpeg.tar.gz" -C "$root/src"
    else
      echo "FFmpeg's mirror on GitHub failed; trying ffmpeg.org" >&2
      download "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz" "$root/src/ffmpeg.tar.xz"
      tar xJf "$root/src/ffmpeg.tar.xz" -C "$root/src"
    fi
    fsrc=$(ls -d "$root/src"/ffmpeg-* "$root/src"/FFmpeg-n* 2> /dev/null | head -1)
  fi
  # The tarball's VERSION file, which the tag's tree lacks: without it ffmpeg's
  # version.sh asks git, which finds the repository the sources were unpacked
  # in and names bbhost's own describe as ffmpeg's version.
  [ -f "$fsrc/VERSION" ] || echo "$FFMPEG_VER" > "$fsrc/VERSION"
  mkdir -p "$root/ffmpeg"
  (cd "$root/ffmpeg" && "$fsrc/configure" --prefix="$prefix" --target-os=mingw64 --arch=x86_64     --cc="clang --target=x86_64-w64-mingw32 --sysroot=/usr/x86_64-w64-mingw32"     --cxx="clang++ --target=x86_64-w64-mingw32 --sysroot=/usr/x86_64-w64-mingw32"     --ar=llvm-ar --ranlib=llvm-ranlib --nm=llvm-nm --windres=llvm-windres     --ld="clang --target=x86_64-w64-mingw32 --sysroot=/usr/x86_64-w64-mingw32 -fuse-ld=lld -L$mingw_gcc_dir"     --enable-cross-compile --enable-static --disable-shared --disable-programs --disable-doc --disable-network     --disable-everything --enable-decoder=h264,hevc,aac,mp3 --enable-demuxer=mov,mp3,aac --enable-parser=h264,hevc,aac     --enable-protocol=file --enable-swscale --enable-swresample --disable-avdevice --disable-avfilter     --disable-postproc --disable-debug --extra-cflags="-O2" --pkg-config=false > "$root/ffmpeg-configure.log" 2>&1     && make -j"$(nproc)" > "$root/ffmpeg-build.log" 2>&1 && make install > "$root/ffmpeg-install.log" 2>&1)
fi

# SPIRV-Tools and the SPIR-V headers it is built from, at the Vulkan SDK's tag
# (Arch's spirv-tools, which the Linux builds here use): the static validator
# and optimizer and the headers, nothing else (no executables or tests).
SPIRV_TAG=${SPIRV_TAG:-vulkan-sdk-1.4.350.0}
if [ ! -f "$prefix/lib/libSPIRV-Tools.a" ] || [ ! -f "$prefix/lib/libSPIRV-Tools-opt.a" ]; then
  hsrc=$(fetch spirv-headers "https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/$SPIRV_TAG.tar.gz" "SPIRV-Headers-$SPIRV_TAG")
  tsrc=$(fetch spirv-tools "https://github.com/KhronosGroup/SPIRV-Tools/archive/refs/tags/$SPIRV_TAG.tar.gz" "SPIRV-Tools-$SPIRV_TAG")
  cmake -S "$tsrc" -B "$root/spirv-tools" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release \
    -DSPIRV-Headers_SOURCE_DIR="$hsrc" -DSPIRV_SKIP_EXECUTABLES=ON -DSPIRV_SKIP_TESTS=ON -DSPIRV_WERROR=OFF \
    -DSPIRV_TOOLS_BUILD_STATIC=ON > "$root/spirv-tools-configure.log"
  cmake --build "$root/spirv-tools" --target SPIRV-Tools-static SPIRV-Tools-opt > "$root/spirv-tools-build.log"
  mkdir -p "$prefix/include/spirv-tools"
  cp "$tsrc"/include/spirv-tools/*.h "$tsrc"/include/spirv-tools/*.hpp "$prefix/include/spirv-tools/"
  cp "$root/spirv-tools/source/libSPIRV-Tools.a" "$root/spirv-tools/source/opt/libSPIRV-Tools-opt.a" "$prefix/lib/"
fi

# Vulkan: the headers are the same on every platform; the loader is wine's
# import library for vulkan-1.dll (the real SDK's vulkan-1.lib on Windows).
ln -sfn /usr/include/vulkan "$prefix/include/vulkan"
ln -sfn /usr/include/vk_video "$prefix/include/vk_video"
cp /usr/lib/wine/x86_64-windows/libvulkan-1.a "$prefix/lib/libvulkan-1.a"
echo "win-deps ready in $prefix"
