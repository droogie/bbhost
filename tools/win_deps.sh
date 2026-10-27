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
mingw_gcc_dir=$(dirname "$(x86_64-w64-mingw32-gcc -print-libgcc-file-name)")
mingw_sysroot=$(x86_64-w64-mingw32-gcc -print-sysroot)
mingw_sysroot=${mingw_sysroot:-/usr/x86_64-w64-mingw32}
mingw_target=$(realpath "$mingw_gcc_dir/../../../../x86_64-w64-mingw32")
mingw_unwind=""
if [ ! -f "$mingw_gcc_dir/libgcc_eh.a" ]; then
  mkdir -p "$root/mingw-runtime"
  cp "$mingw_gcc_dir/libgcc.a" "$root/mingw-runtime/libgcc_eh.a"
  mingw_unwind="-L$root/mingw-runtime"
fi
tc=$PWD/cmake/mingw-toolchain.cmake
SDL_TAG=${SDL_TAG:-release-3.4.16}
ZLIB_TAG=${ZLIB_TAG:-v1.3.1}
mkdir -p "$root/src" "$prefix/include" "$prefix/lib"

fetch() {  # fetch NAME URL DIR-GLOB
  local name=$1 url=$2 dir=$3
  if ! ls -d "$root/src"/$dir > /dev/null 2>&1; then
    echo "fetching $name" >&2
    curl -fsSL -m 300 -o "$root/src/$name.tar.gz" "$url"
    tar xzf "$root/src/$name.tar.gz" -C "$root/src"
  fi
  ls -d "$root/src"/$dir | head -1
}

zsrc=$(fetch zlib "https://github.com/madler/zlib/archive/refs/tags/$ZLIB_TAG.tar.gz" 'zlib-*')
if [ ! -f "$prefix/lib/libzlibstatic.a" ]; then
  cmake -S "$zsrc" -B "$root/zlib" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" -DZLIB_BUILD_SHARED=OFF -DZLIB_BUILD_STATIC=ON -DZLIB_BUILD_EXAMPLES=OFF > "$root/zlib-configure.log"
  cmake --build "$root/zlib" > "$root/zlib-build.log"
  cmake --install "$root/zlib" > /dev/null
  rm -f "$prefix/lib/libzlib.dll.a" "$prefix/bin"/*zlib*.dll  # static only, or FindZLIB picks the import library
fi

ssrc=$(fetch sdl3 "https://github.com/libsdl-org/SDL/archive/refs/tags/$SDL_TAG.tar.gz" 'SDL-*')
if [ ! -f "$prefix/lib/libSDL3.a" ]; then
  cmake -S "$ssrc" -B "$root/sdl3" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TEST_LIBRARY=OFF -DSDL_TESTS=OFF \
    -DSDL_EXAMPLES=OFF > "$root/sdl3-configure.log"
  cmake --build "$root/sdl3" > "$root/sdl3-build.log"
  cmake --install "$root/sdl3" > /dev/null
fi

# libcurl on schannel (Windows' own TLS), static, for sceHttp and the session service.
CURL_TAG=${CURL_TAG:-curl-8_22_0}
csrc=$(fetch curl "https://github.com/curl/curl/releases/download/$CURL_TAG/curl-$(echo ${CURL_TAG#curl-} | tr _ .).tar.gz" 'curl-*')
if [ ! -f "$prefix/lib/libcurl.a" ]; then
  cmake -S "$csrc" -B "$root/curl" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$tc" -DCMAKE_BUILD_TYPE=Release     -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$prefix" -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON     -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF -DBUILD_LIBCURL_DOCS=OFF -DBUILD_MISC_DOCS=OFF -DENABLE_CURL_MANUAL=OFF     -DCURL_USE_SCHANNEL=ON -DCURL_USE_LIBPSL=OFF -DUSE_LIBIDN2=OFF -DCURL_USE_LIBSSH2=OFF -DCURL_BROTLI=OFF     -DCURL_ZSTD=OFF -DCURL_DISABLE_LDAP=ON -DUSE_NGHTTP2=OFF -DCURL_ZLIB=ON > "$root/curl-configure.log"
  cmake --build "$root/curl" > "$root/curl-build.log"
  cmake --install "$root/curl" > /dev/null
fi

# ffmpeg, only what the movies need (H.264 + AAC in MP4; HEVC comes along
# because 7.1 puts the film-grain object H.264 references under it), static; its .pc
# files are what the host's configure finds through the toolchain's
# PKG_CONFIG_LIBDIR.
FFMPEG_VER=${FFMPEG_VER:-7.1.1}
if ! ls -d "$root/src"/ffmpeg-* > /dev/null 2>&1; then
  echo "fetching ffmpeg"
  curl -fsSL -m 600 -o "$root/src/ffmpeg.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz"
  tar xJf "$root/src/ffmpeg.tar.xz" -C "$root/src"
fi
fsrc=$(ls -d "$root/src"/ffmpeg-* | head -1)
if [ ! -f "$prefix/lib/libavcodec.a" ]; then
  mkdir -p "$root/ffmpeg"
  (cd "$root/ffmpeg" && "$fsrc/configure" --prefix="$prefix" --target-os=mingw64 --arch=x86_64     --cc="clang --target=x86_64-w64-mingw32 --sysroot=$mingw_sysroot"     --cxx="clang++ --target=x86_64-w64-mingw32 --sysroot=$mingw_sysroot"     --ar=llvm-ar --ranlib=llvm-ranlib --nm=llvm-nm --windres=x86_64-w64-mingw32-windres     --ld="clang --target=x86_64-w64-mingw32 --sysroot=$mingw_sysroot $mingw_unwind -fuse-ld=lld -L$mingw_gcc_dir -L$mingw_target/lib"     --enable-cross-compile --enable-static --disable-shared --disable-programs --disable-doc --disable-network     --disable-everything --enable-decoder=h264,hevc,aac,mp3 --enable-demuxer=mov,mp3,aac --enable-parser=h264,hevc,aac     --enable-protocol=file --enable-swscale --enable-swresample --disable-avdevice --disable-avfilter     --disable-postproc --disable-debug --extra-cflags="-O2" --pkg-config=false > "$root/ffmpeg-configure.log" 2>&1     && make -j"$(nproc)" > "$root/ffmpeg-build.log" 2>&1 && make install > "$root/ffmpeg-install.log" 2>&1)
fi

# SPIRV-Tools and the SPIR-V headers it is built from, at the Vulkan SDK's tag
# (Arch's spirv-tools, which the Linux builds here use): the static validator
# and optimizer and the headers, nothing else (no executables or tests).
SPIRV_TAG=${SPIRV_TAG:-vulkan-sdk-1.4.350.0}
hsrc=$(fetch spirv-headers "https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/$SPIRV_TAG.tar.gz" "SPIRV-Headers-$SPIRV_TAG")
tsrc=$(fetch spirv-tools "https://github.com/KhronosGroup/SPIRV-Tools/archive/refs/tags/$SPIRV_TAG.tar.gz" "SPIRV-Tools-$SPIRV_TAG")
if [ ! -f "$prefix/lib/libSPIRV-Tools.a" ] || [ ! -f "$prefix/lib/libSPIRV-Tools-opt.a" ]; then
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
