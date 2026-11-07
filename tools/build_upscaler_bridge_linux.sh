#!/bin/bash
# The upscalers' runtime the Linux package carries beside bbhost
# (host/upscaler.cpp, host/framegen.cpp), the counterpart of
# tools/fetch_upscalers.sh: Numity1504's upscaler bridge (MIT, its source in
# third_party/upscaler_bridge/src) built as upscaler_bridge.so with NVIDIA's
# NGX library linked in, and NVIDIA's DLSS runtime for Linux beside it. NVIDIA's
# files come from NVIDIA's DLSS repository at a pinned tag and SHA-256,
# unmodified, under NVIDIA's licence (licenses/dlss - never bbhost's GPL); the
# SDK's headers and libnvsdk_ngx.a are only used to build the bridge and are
# not shipped. FSR on Linux is libamd_fidelityfx_vk.so (AMD ships no Linux
# files; build/HANDOFF-linux-upscalers.md): copied in when FFX_LINUX_SO names
# one. Without these files DLSS, FSR and frame generation stay off and TAA
# (bbhost's own) still runs.
#
#   tools/build_upscaler_bridge_linux.sh --fetch     NVIDIA's files into the cache (needs the network; outside the
#                                                    portable build's chroot)
#   tools/build_upscaler_bridge_linux.sh DIR         the bridge built into DIR with NVIDIA's runtime beside it, their
#                                                    licences into DIR/licenses/ (needs only the cache: the chroot)
#   UPSCALERS_LINUX_CACHE=DIR (default build/upscalers-linux)   where the downloads are kept
#   FFX_LINUX_SO=FILE   a libamd_fidelityfx_vk.so to put beside the bridge (FSR 3 and frame generation)
#   CXX (default clang++)
#
# A download that does not match its pin is deleted and the run fails.
set -eu
cd "$(dirname "$0")/.."
DLSS_TAG=v310.9.16
DLSS_RAW=https://github.com/NVIDIA/DLSS/raw/$DLSS_TAG
DLSS_SO=libnvidia-ngx-dlss.so.310.9.16

# NAME PATH-IN-REPO SHA-256
files="
nvsdk_ngx_vk.h include/nvsdk_ngx_vk.h 2d364ce7132881eb669e9498fd570d74cba563b1fbebc82c235f8e1ad2dd8b6d
nvsdk_ngx_helpers_vk.h include/nvsdk_ngx_helpers_vk.h c192bff72138f12f770db48e12c1d8f712dfc81ea0e8e055ebb27d8dd1b31623
nvsdk_ngx_defs.h include/nvsdk_ngx_defs.h ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e
nvsdk_ngx_defs_vk.h include/nvsdk_ngx_defs_vk.h 0a24d0861ace7d6b9362a67b7f08bea1b33ea123c5ce730b19133de8a891d031
nvsdk_ngx_params.h include/nvsdk_ngx_params.h 943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc
libnvsdk_ngx.a lib/Linux_x86_64/libnvsdk_ngx.a d3c50bb0d9624fc35e502a8e929b8fc1e2b505017ebdb84c0d5b627e11a235df
$DLSS_SO lib/Linux_x86_64/rel/$DLSS_SO 30982c937afed85c96b8d313323de9a650a2eec6de1bb0f714b21750f4ce98f6
"

fail() { echo "build_upscaler_bridge_linux: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
cache=${UPSCALERS_LINUX_CACHE:-build/upscalers-linux}

if [ "${1:-}" = --fetch ]; then
    mkdir -p "$cache"
    echo "$files" | while read -r name path want; do
        [ -n "$name" ] || continue
        kept=$cache/$name
        [ -f "$kept" ] && [ "$(sha "$kept")" = "$want" ] && continue
        rm -f "$kept"
        echo "build_upscaler_bridge_linux: $DLSS_RAW/$path"
        curl -sSL --fail --retry 3 -m 600 -o "$kept.part" "$DLSS_RAW/$path" || { rm -f "$kept.part"; fail "download failed: $DLSS_RAW/$path"; }
        got=$(sha "$kept.part")
        if [ "$got" != "$want" ]; then
            rm -f "$kept.part"
            fail "$name: SHA-256 $got, not the pinned $want"
        fi
        mv "$kept.part" "$kept"
    done
    exit 0
fi

dest=${1:-}
[ -n "$dest" ] || fail "usage: tools/build_upscaler_bridge_linux.sh --fetch | DIR"
echo "$files" | while read -r name path want; do
    [ -n "$name" ] || continue
    [ -f "$cache/$name" ] && [ "$(sha "$cache/$name")" = "$want" ] || fail "$cache/$name missing or not the pinned file (run --fetch first)"
done
mkdir -p "$dest/licenses/dlss" "$dest/licenses/upscaler_bridge"
# Static C++ runtime and hidden symbols, as bbhost's own (tools/linux_portable_build.sh):
# one export, ub_get_api. NVIDIA's library calls dlopen and pthreads.
${CXX:-clang++} -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -DNDEBUG -DUB_WITH_DLSS \
    -Ithird_party/upscaler_bridge -Ithird_party/upscaler_bridge/ffx -I"$cache" \
    third_party/upscaler_bridge/src/upscaler_bridge.cpp "$cache/libnvsdk_ngx.a" \
    -static-libstdc++ -static-libgcc -Wl,--no-undefined -Wl,-soname,upscaler_bridge.so -ldl -lpthread \
    -o "$dest/upscaler_bridge.so"
cp "$cache/$DLSS_SO" "$dest/$DLSS_SO"
cp third_party/dlss/LICENSE.txt "$dest/licenses/dlss/LICENSE.txt"
cp third_party/upscaler_bridge/LICENSE.txt "$dest/licenses/upscaler_bridge/LICENSE.txt"
if [ -n "${FFX_LINUX_SO:-}" ]; then
    [ -f "$FFX_LINUX_SO" ] || fail "FFX_LINUX_SO=$FFX_LINUX_SO is not a file"
    mkdir -p "$dest/licenses/fidelityfx"
    cp "$FFX_LINUX_SO" "$dest/libamd_fidelityfx_vk.so"
    cp third_party/fidelityfx/LICENSE-source.md "$dest/licenses/fidelityfx/LICENSE.md" 2>/dev/null ||
        fail "third_party/fidelityfx/LICENSE-source.md (the SDK's MIT licence) is missing"
fi
echo "upscalers: the bridge (with NGX) and DLSS $DLSS_TAG's Linux runtime into $dest${FFX_LINUX_SO:+, with libamd_fidelityfx_vk.so}"
