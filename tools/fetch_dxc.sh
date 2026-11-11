#!/bin/bash
# The DirectX Shader Compiler the packages carry for the HLSL shader path
# (BBHOST_HLSL, host/dxc.h): DXC's own GitHub release, pinned below by tag and
# SHA-256, of which only the compiler library is taken - libdxcompiler.so, or
# the x64 dxcompiler.dll - unmodified, with its licences (third_party/dxc).
# Never dxil.dll or libdxil.so: SPIR-V needs no DXIL validator, and theirs are
# Microsoft's own licence terms, not an open one. bbhost loads the library at
# run time from beside its executable; without it every shader keeps the
# lifter's SPIR-V, as with the switch off.
#
#   tools/fetch_dxc.sh linux|windows DIR   the library into DIR, the licences into DIR/licenses/dxc
#   tools/fetch_dxc.sh linux-cli DIR       the Linux release's command-line compiler, DIR/bin/dxc with the
#                                          library in DIR/lib, for the build's FidelityFX shaders
#                                          (tools/ffx_permutations.py); a build tool, never packaged
#   tools/fetch_dxc.sh --sha256 linux|windows   the pinned SHA-256 of the library (tools/check_release_bundle.sh)
#   DXC_CACHE=DIR (default build/dxc)       where the release archives are kept
#
# The archive and the library taken out of it are both checked against the
# pins; a download that does not match is deleted and the run fails. A new
# DXC is a new tag, both archives' and both libraries' SHA-256 here, and
# third_party/dxc's licence texts from that tag (its VERSION file): the HLSL
# path's checks (docs/rendering.md) are run again with it first, since the
# translation cache keys modules on DXC's version and every player then
# compiles them anew.
set -eu
cd "$(dirname "$0")/.."
DXC_TAG=v1.9.2602
LINUX_ASSET=linux_dxc_2026_02_20.x86_64.tar.gz
LINUX_ASSET_SHA256=a1d3e3b5e1c5685b3eb27d5e8890e41d87df45def05112a2d6f1a63a931f7d60
LINUX_LIB_SHA256=f64b6a9cf32ecb6b456e3f492d736ce41c250845e2dba282f4fc1c4cbeaa0f04
LINUX_CLI_SHA256=8d40f5eb5359ea0ea68ffdd9955a796a940d034ae3ff86629bc23eb2b1b56d6e
WINDOWS_ASSET=dxc_2026_02_20.zip
WINDOWS_ASSET_SHA256=a1e89031421cf3c1fca6627766ab3020ca4f962ac7e2caa7fab2b33a8436151e
WINDOWS_LIB_SHA256=b86a738ece4c05dbe2d9bbb29668a2ccb28a0740773c9b027cdc61e8d07b4d75

fail() { echo "fetch_dxc: $*" >&2; exit 1; }
if [ "${1:-}" = --sha256 ]; then
    case "${2:-}" in
        linux) echo "$LINUX_LIB_SHA256" ;;
        windows) echo "$WINDOWS_LIB_SHA256" ;;
        *) fail "usage: tools/fetch_dxc.sh --sha256 linux|windows" ;;
    esac
    exit 0
fi
target=${1:-}; dest=${2:-}
[ -n "$dest" ] || fail "usage: tools/fetch_dxc.sh linux|windows DIR"
case "$target" in
    linux | linux-cli) asset=$LINUX_ASSET asset_sha=$LINUX_ASSET_SHA256 lib=libdxcompiler.so lib_sha=$LINUX_LIB_SHA256 ;;
    windows) asset=$WINDOWS_ASSET asset_sha=$WINDOWS_ASSET_SHA256 lib=dxcompiler.dll lib_sha=$WINDOWS_LIB_SHA256 ;;
    *) fail "the target is linux, linux-cli or windows, not '$target'" ;;
esac
[ "$(cat third_party/dxc/VERSION)" = "$DXC_TAG" ] || fail "third_party/dxc holds the licences of $(cat third_party/dxc/VERSION), not $DXC_TAG"
sha() { sha256sum "$1" | cut -d' ' -f1; }
# The release asset from its download address, then through GitHub's API (the
# release by its tag, the asset's API address from it, that address asked for
# the file itself): on 2026-11-08 github.com's download addresses answered 504
# for minutes on end while the API served the releases.
download() {  # OUT
    local url at
    url=https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DXC_TAG/$asset
    echo "fetch_dxc: $url"
    curl -sSL --fail --retry 3 -m 600 -o "$1" "$url" && return 0
    at=$(curl -sSL --fail --retry 3 -m 60 "https://api.github.com/repos/microsoft/DirectXShaderCompiler/releases/tags/$DXC_TAG" |
        python3 -I -c 'import json, sys; d = json.loads(sys.stdin.read() or "{}"); print(next((a["url"] for a in d.get("assets", []) if a["name"] == sys.argv[1]), ""))' "$asset") && [ -n "$at" ] &&
        echo "fetch_dxc: $at" &&
        curl -sSL --fail --retry 3 -m 600 -H 'Accept: application/octet-stream' -o "$1" "$at" && return 0
    rm -f "$1"
    return 1
}

cache=${DXC_CACHE:-build/dxc}
mkdir -p "$cache"
archive=$cache/$DXC_TAG-$asset
if [ ! -f "$archive" ] || [ "$(sha "$archive")" != "$asset_sha" ]; then
    rm -f "$archive"
    download "$archive.part" || fail "download failed: DXC $DXC_TAG $asset"
    got=$(sha "$archive.part")
    if [ "$got" != "$asset_sha" ]; then
        rm -f "$archive.part"
        fail "$asset: SHA-256 $got, not the pinned $asset_sha"
    fi
    mv "$archive.part" "$archive"
fi

if [ "$target" = linux-cli ]; then
    # The compiler and the library it loads from ../lib (its RUNPATH), each checked.
    mkdir -p "$dest/bin" "$dest/lib"
    tar xzf "$archive" -O ./bin/dxc > "$dest/bin/dxc.part"
    tar xzf "$archive" -O ./lib/libdxcompiler.so > "$dest/lib/libdxcompiler.so.part"
    [ "$(sha "$dest/bin/dxc.part")" = "$LINUX_CLI_SHA256" ] || { rm -f "$dest/bin/dxc.part"; fail "bin/dxc out of $asset is not the pinned one"; }
    [ "$(sha "$dest/lib/libdxcompiler.so.part")" = "$lib_sha" ] || { rm -f "$dest/lib/libdxcompiler.so.part"; fail "libdxcompiler.so out of $asset is not the pinned one"; }
    mv "$dest/bin/dxc.part" "$dest/bin/dxc"
    mv "$dest/lib/libdxcompiler.so.part" "$dest/lib/libdxcompiler.so"
    chmod 755 "$dest/bin/dxc"
    chmod 644 "$dest/lib/libdxcompiler.so"
    echo "fetch_dxc: DXC $DXC_TAG's command-line compiler in $dest/bin/dxc"
    exit 0
fi
mkdir -p "$dest/licenses/dxc"
case "$target" in
    linux) tar xzf "$archive" -O ./lib/libdxcompiler.so > "$dest/$lib.part" ;;
    windows) unzip -p "$archive" bin/x64/dxcompiler.dll > "$dest/$lib.part" ;;
esac
got=$(sha "$dest/$lib.part")
if [ "$got" != "$lib_sha" ]; then
    rm -f "$dest/$lib.part"
    fail "$lib out of $asset: SHA-256 $got, not the pinned $lib_sha"
fi
mv "$dest/$lib.part" "$dest/$lib"
chmod 644 "$dest/$lib"
for f in third_party/dxc/*; do
    [ "$(basename "$f")" = VERSION ] || install -m 644 "$f" "$dest/licenses/dxc/"
done
echo "fetch_dxc: DXC $DXC_TAG $lib ($(( $(stat -c %s "$dest/$lib") / 1048576 )) MiB, SHA-256 $lib_sha) and its licences in $dest"
