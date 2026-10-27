#!/bin/bash
# Builds a bbhost that runs on other Linux machines, not just this one.
#
# A plain `cmake --build build` here links against this machine's glibc
# (2.43 on Arch today: acosf@GLIBC_2.43, __isoc23_strtol@GLIBC_2.38), its
# libstdc++ (GLIBCXX_3.4.35) and its ffmpeg 8 / SDL3 shared libraries, so the
# binary starts on an up-to-date Arch box and nowhere else. This builds inside
# an Ubuntu 24.04 root filesystem instead (glibc 2.39: Ubuntu 24.04+, Mint 22+,
# Debian 13, Fedora 40+, SteamOS, Arch), rootless (a user + mount namespace and
# chroot; no docker, no sudo), with:
#   - libstdc++ 14 and libgcc linked in (-static-libstdc++ -static-libgcc);
#   - SDL3 static (it opens X11/Wayland/PulseAudio/PipeWire itself at run time);
#   - ffmpeg static, only the movie decoders (the same cut as tools/win_deps.sh);
#   - SPIRV-Tools static (Ubuntu's .a);
#   - the Vulkan headers of this machine (headers only; the loader is the
#     player's libvulkan.so.1);
# leaving libvulkan.so.1, libcurl.so.4 and libz.so.1 (plus glibc) as the only
# shared libraries it needs, which every desktop distribution ships.
#
#   tools/linux_portable_build.sh            -> build-linux-portable/build/bbhost
#                                               (and plugins/, the official ones
#                                               tools/package_linux.sh packs)
#   PORTABLE_TARGETS="gcn2hlsl" tools/linux_portable_build.sh
#                                            more targets built the same way:
#                                            gcn2hlsl, the HLSL sweep a Steam
#                                            Deck or another machine runs
#                                            (tools/package_hlsl_sweep.sh)
#
# LINUX_ROOTFS is where the Ubuntu tree goes (default
# build-linux-portable/rootfs). It must be on a local filesystem that keeps
# ownership (ext4, xfs, btrfs; not NFS, where dpkg's chown fails).
set -eu
cd "$(dirname "$0")/.."
SRC=$PWD
W=$SRC/build-linux-portable
ROOTFS=${LINUX_ROOTFS:-$W/rootfs}
UBUNTU_BASE=${UBUNTU_BASE:-https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/ubuntu-base-24.04.5-base-amd64.tar.gz}
SDL_TAG=${SDL_TAG:-release-3.4.16}
FFMPEG_VER=${FFMPEG_VER:-7.1.1}

if [ "${1:-}" != --inside ]; then
    mkdir -p "$W/src" "$W/prefix/include"
    # The root filesystem, once.
    if [ ! -x "$ROOTFS/usr/bin/apt-get" ]; then
        mkdir -p "$ROOTFS"
        curl -sL -m 600 -o "$W/ubuntu-base.tar.gz" "$UBUNTU_BASE"
        # A few files belong to groups the one-uid namespace cannot map; they do not matter.
        unshare -rm tar xzf "$W/ubuntu-base.tar.gz" -C "$ROOTFS" --exclude='dev/*' 2>/dev/null || true
        rm -f "$W/ubuntu-base.tar.gz"
        cp /etc/resolv.conf "$ROOTFS/etc/resolv.conf"
        echo 'APT::Sandbox::User "root";' > "$ROOTFS/etc/apt/apt.conf.d/99sandbox"
    fi
    # Sources, fetched here (the chroot has no copy of this tree's network setup).
    fetch() {  # fetch NAME URL DIR-GLOB
        if ! ls -d "$W/src"/$3 > /dev/null 2>&1; then
            echo "fetching $1"
            curl -sL -m 600 -o "$W/src/$1.tar" "$2"
            tar xf "$W/src/$1.tar" -C "$W/src"
            rm -f "$W/src/$1.tar"
        fi
    }
    fetch sdl3 "https://github.com/libsdl-org/SDL/archive/refs/tags/$SDL_TAG.tar.gz" 'SDL-*'
    fetch ffmpeg "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz" 'ffmpeg-*'
    # The Vulkan headers of this machine (Ubuntu 24.04's are 1.3.275, older than
    # the renderer's), copied: the chroot cannot follow a symlink out of itself.
    rm -rf "$W/prefix/include/vulkan" "$W/prefix/include/vk_video"
    cp -r /usr/include/vulkan /usr/include/vk_video "$W/prefix/include/"
    # The revision the binary logs on its first line (cmake/version.cmake):
    # the chroot has the tree but not its git directory.
    rev=$(git -c core.excludesFile=/dev/null rev-parse --short HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git -c core.excludesFile=/dev/null status --porcelain --untracked-files=no 2>/dev/null)" ]; then
        rev="$rev+"
    fi
    export BBHOST_GIT_REV=$rev
    exec unshare -rm --fork sh -c '
        R=$1; SRC=$2; shift 2
        mount --rbind /dev "$R/dev"
        mount -t proc proc "$R/proc" 2>/dev/null || mount --rbind /proc "$R/proc"
        mount --rbind /sys "$R/sys"
        mkdir -p "$R/src" && mount --bind "$SRC" "$R/src"
        exec chroot "$R" /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
            HOME=/root DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 BBHOST_GIT_REV="$BBHOST_GIT_REV" PORTABLE_TARGETS="${PORTABLE_TARGETS:-}" \
            bash /src/tools/linux_portable_build.sh --inside
    ' sh "$ROOTFS" "$SRC"
fi

# --- inside the Ubuntu 24.04 root ---------------------------------------------
W=/src/build-linux-portable
P=$W/prefix
# libstdc++ 14, not 24.04's default 13: the HLE libc calls std::acosf and friends,
# which libstdc++ has only had since 14. Clang takes the newest GCC it finds.
if [ ! -x /usr/bin/clang ] || [ ! -f /usr/lib/x86_64-linux-gnu/libSPIRV-Tools-opt.a ] || [ ! -d /usr/include/c++/14 ]; then
    apt-get update -q > /tmp/apt-update.log 2>&1
    apt-get install -y -q --no-install-recommends clang lld llvm cmake ninja-build pkg-config make nasm \
        libstdc++-14-dev libcurl4-openssl-dev libvulkan-dev zlib1g-dev spirv-tools \
        libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev libxss-dev libxtst-dev \
        libxkbcommon-dev libwayland-dev wayland-protocols libegl-dev libdecor-0-dev libpulse-dev \
        libasound2-dev libpipewire-0.3-dev libdbus-1-dev libudev-dev libdrm-dev libgbm-dev \
        ca-certificates file python3 xz-utils > /tmp/apt-install.log 2>&1 || { tail -20 /tmp/apt-install.log; exit 1; }
fi
export CC=clang CXX=clang++

# SDL3, static. Its video/audio backends are opened with dlopen at run time,
# so the player's X11 or Wayland and PulseAudio or PipeWire are used as found.
if [ ! -f "$P/lib/libSDL3.a" ]; then
    s=$(ls -d $W/src/SDL-* | head -1)
    cmake -S "$s" -B "$W/sdl3" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
        -DCMAKE_INSTALL_LIBDIR=lib -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TEST_LIBRARY=OFF -DSDL_TESTS=OFF \
        -DSDL_EXAMPLES=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$W/sdl3-configure.log"
    cmake --build "$W/sdl3" > "$W/sdl3-build.log"
    cmake --install "$W/sdl3" > /dev/null
fi

# ffmpeg, only what the movies need (tools/win_deps.sh has the reasons), static.
if [ ! -f "$P/lib/libavcodec.a" ]; then
    f=$(ls -d $W/src/ffmpeg-* | head -1)
    mkdir -p "$W/ffmpeg"
    (cd "$W/ffmpeg" && "$f/configure" --prefix="$P" --cc=clang --cxx=clang++ --enable-static --disable-shared \
        --enable-pic --disable-programs --disable-doc --disable-network --disable-everything \
        --enable-decoder=h264,hevc,aac,mp3 --enable-demuxer=mov,mp3,aac --enable-parser=h264,hevc,aac \
        --enable-protocol=file --enable-swscale --enable-swresample --disable-avdevice --disable-avfilter \
        --disable-postproc --disable-debug --disable-vaapi --disable-vdpau --disable-xlib --disable-libdrm \
        --disable-iconv --disable-zlib --disable-bzlib --disable-lzma --extra-cflags=-O2 \
        > "$W/ffmpeg-configure.log" 2>&1 \
        && make -j"$(nproc)" > "$W/ffmpeg-build.log" 2>&1 && make install > "$W/ffmpeg-install.log" 2>&1) \
        || { tail -20 "$W/ffmpeg-configure.log" "$W/ffmpeg-build.log"; exit 1; }
fi

# ffmpeg's .pc files ask for -latomic, a shared library that is not on every
# machine (Fedora keeps it in its own package): the static archive instead.
sed -i 's/ -latomic/ -l:libatomic.a/' "$P"/lib/pkgconfig/*.pc

# bbhost. pkg-config sees only the prefix's ffmpeg; the static archives'
# private libraries (-lm, -lpthread) come with libstdc++'s.
export PKG_CONFIG_PATH=$P/lib/pkgconfig
cmake -S /src -B "$W/build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_PREFIX_PATH="$P" -DVulkan_INCLUDE_DIR="$P/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -fuse-ld=lld" \
    -DCMAKE_MODULE_LINKER_FLAGS="-static-libgcc -fuse-ld=lld" > "$W/bbhost-configure.log"
grep -E "^-- bbhost" "$W/bbhost-configure.log" || true
cmake --build "$W/build" --target bbhost bb_plugin_region_switch bb_plugin_hello bb_plugin_randomizer bb_plugin_boss_rush bb_plugin_mutators bb_plugin_debug_menu ${PORTABLE_TARGETS:-} -j"$(nproc)" > "$W/bbhost-build.log" 2>&1 \
    || { grep -E "error|undefined" "$W/bbhost-build.log" | head -30; exit 1; }
echo "portable build: $W/build/bbhost"
