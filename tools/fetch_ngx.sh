#!/bin/bash
# NVIDIA's files for DLSS on Linux, from NVIDIA's DLSS repository at a pinned
# tag, each pinned by SHA-256 and used unmodified, under NVIDIA's licence
# (third_party/dlss/LICENSE.txt; licenses/dlss in the package - never bbhost's
# GPL). None of them is in this repository:
#   - the NGX SDK's headers and static library, libnvsdk_ngx.a, which build the
#     NGX bridge (src/ngx_bridge, MIT) and nothing else: bbhost itself never
#     links them, and loads the bridge at run time (host/upscaler.cpp);
#   - the DLSS runtime, libnvidia-ngx-dlss.so.310.9.16, which the Linux package
#     carries beside bbhost and NVIDIA's NGX (in the NVIDIA driver) loads.
# Without them the bridge is not built, or the package carries no runtime, and
# DLSS stays off on Linux; everything else builds and runs as before.
#
#   tools/fetch_ngx.sh sdk DIR        the headers into DIR/include, libnvsdk_ngx.a into DIR/lib
#   tools/fetch_ngx.sh runtime DIR    the runtime into DIR, NVIDIA's licence into DIR/licenses/dlss
#   tools/fetch_ngx.sh --sha256 FILE  the pinned SHA-256 of one of them (tools/check_release_bundle.sh)
#   tools/fetch_ngx.sh --files        the runtime's file names (what a package carries beside bbhost)
#   tools/fetch_ngx.sh --tag          NVIDIA's tag the files are pinned at
#   NGX_CACHE=DIR (default build/ngx)   where the downloads are kept
#
# A download that does not match its pin is deleted and the run fails; a kept
# file is checked again at every run. A new version is a new tag and SHA-256s
# here, and the licence text from that tag in third_party/dlss.
set -eu
cd "$(dirname "$0")/.."
TAG=v310.9.16
URL=https://github.com/NVIDIA/DLSS/raw/$TAG
RUNTIME=libnvidia-ngx-dlss.so.310.9.16

headers="nvsdk_ngx_defs.h nvsdk_ngx_defs_vk.h nvsdk_ngx_params.h nvsdk_ngx_vk.h nvsdk_ngx_helpers_vk.h"
pin() {
    case "$1" in
        nvsdk_ngx_defs.h) echo ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e ;;
        nvsdk_ngx_defs_vk.h) echo 0a24d0861ace7d6b9362a67b7f08bea1b33ea123c5ce730b19133de8a891d031 ;;
        nvsdk_ngx_params.h) echo 943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc ;;
        nvsdk_ngx_vk.h) echo 2d364ce7132881eb669e9498fd570d74cba563b1fbebc82c235f8e1ad2dd8b6d ;;
        nvsdk_ngx_helpers_vk.h) echo c192bff72138f12f770db48e12c1d8f712dfc81ea0e8e055ebb27d8dd1b31623 ;;
        libnvsdk_ngx.a) echo d3c50bb0d9624fc35e502a8e929b8fc1e2b505017ebdb84c0d5b627e11a235df ;;
        "$RUNTIME") echo 30982c937afed85c96b8d313323de9a650a2eec6de1bb0f714b21750f4ce98f6 ;;
        *) return 1 ;;
    esac
}
where() {  # the file's path in NVIDIA's repository
    case "$1" in
        *.h) echo include/$1 ;;
        libnvsdk_ngx.a) echo lib/Linux_x86_64/$1 ;;
        *) echo lib/Linux_x86_64/rel/$1 ;;
    esac
}

fail() { echo "fetch_ngx: $*" >&2; exit 1; }
case "${1:-}" in
    --sha256) pin "${2:-}" || fail "usage: tools/fetch_ngx.sh --sha256 FILE (one of: $headers libnvsdk_ngx.a $RUNTIME)"; exit 0 ;;
    --files) echo "$RUNTIME"; exit 0 ;;
    --tag) echo "$TAG"; exit 0 ;;
    sdk | runtime) ;;
    *) fail "usage: tools/fetch_ngx.sh sdk|runtime DIR" ;;
esac
what=$1
dest=${2:-}
[ -n "$dest" ] || fail "usage: tools/fetch_ngx.sh $what DIR"
sha() { sha256sum "$1" | cut -d' ' -f1; }

cache=${NGX_CACHE:-build/ngx}
mkdir -p "$cache"
# One file into the cache, checked against its pin (a kept one too).
get() {
    local f=$1 want kept url got
    want=$(pin "$f")
    kept=$cache/$f
    if [ -f "$kept" ] && [ "$(sha "$kept")" = "$want" ]; then return 0; fi
    rm -f "$kept"
    url=$URL/$(where "$f")
    echo "fetch_ngx: $url"
    curl -sSL --fail --retry 2 --connect-timeout 15 -m 900 -o "$kept.part" "$url" || { rm -f "$kept.part"; fail "download failed: $url"; }
    got=$(sha "$kept.part")
    if [ "$got" != "$want" ]; then
        rm -f "$kept.part"
        fail "$f: SHA-256 $got, not the pinned $want"
    fi
    mv "$kept.part" "$kept"
}

if [ "$what" = sdk ]; then
    mkdir -p "$dest/include" "$dest/lib"
    for f in $headers; do get "$f"; cp "$cache/$f" "$dest/include/$f"; done
    get libnvsdk_ngx.a
    cp "$cache/libnvsdk_ngx.a" "$dest/lib/libnvsdk_ngx.a"
    echo "ngx: NVIDIA's NGX SDK $TAG (headers, libnvsdk_ngx.a) in $dest, under NVIDIA's licence (third_party/dlss/LICENSE.txt)"
else
    mkdir -p "$dest/licenses/dlss"
    get "$RUNTIME"
    cp "$cache/$RUNTIME" "$dest/$RUNTIME"
    cp third_party/dlss/LICENSE.txt "$dest/licenses/dlss/LICENSE.txt"
    echo "ngx: NVIDIA's DLSS runtime $RUNTIME ($TAG) in $dest, with its licence in $dest/licenses/dlss"
fi
