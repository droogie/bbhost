#!/bin/bash
# The upscalers' and frame generation's runtime the Windows package carries
# beside bbhost.exe (host/upscaler.cpp, host/framegen.cpp): AMD's signed
# FidelityFX binaries from the FidelityFX SDK's repository and NVIDIA's DLSS
# runtime from NVIDIA's DLSS repository, each at a pinned tag and SHA-256,
# unmodified, under its own licence (licenses/fidelityfx, licenses/dlss - never
# bbhost's GPL); and Numity1504's upscaler bridge (MIT), kept in
# third_party/upscaler_bridge/bin. Without them FSR 3, FSR 4, DLSS and frame
# generation stay off and TAA (bbhost's own) still runs.
#
#   tools/fetch_upscalers.sh DIR        the five DLLs into DIR, their licences into DIR/licenses/
#   tools/fetch_upscalers.sh --sha256 FILE   the pinned SHA-256 of one of them (tools/check_release_bundle.sh)
#   tools/fetch_upscalers.sh --files    their names
#   UPSCALERS_CACHE=DIR (default build/upscalers)   where the downloads are kept
#
# A download that does not match its pin is deleted and the run fails. A new
# version is a new tag and SHA-256 here and the licence text from that tag.
set -eu
cd "$(dirname "$0")/.."
FFX_TAG=v2.4.0
FFX_REPO=GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
FFX_DIR=Kits/FidelityFX/signedbin
DLSS_TAG=v310.9.16
DLSS_REPO=NVIDIA/DLSS
DLSS_DIR=lib/Windows_x86_64/rel
BRIDGE=third_party/upscaler_bridge/bin/upscaler_bridge.dll

files="amd_fidelityfx_loader_vk.dll amd_fidelityfx_upscaler_vk.dll amd_fidelityfx_framegeneration_vk.dll nvngx_dlss.dll upscaler_bridge.dll"
pin() {
    case "$1" in
        amd_fidelityfx_loader_vk.dll) echo 38a56a01fb3565c0757d9f5c1e4e69de4b65e5f2e5769b323b200dc1f142bab9 ;;
        amd_fidelityfx_upscaler_vk.dll) echo 4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7 ;;
        amd_fidelityfx_framegeneration_vk.dll) echo d002dabff8c5e71eaa715fc99438cd4dc81230262a7897b16bc402efec3498b4 ;;
        nvngx_dlss.dll) echo d00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e ;;
        upscaler_bridge.dll) echo 5bfc95dcd4c650ffe6f1030e9df5d152772b948f288d8df6e6ea9044586cd496 ;;
        *) return 1 ;;
    esac
}

fail() { echo "fetch_upscalers: $*" >&2; exit 1; }
if [ "${1:-}" = --sha256 ]; then
    pin "${2:-}" || fail "usage: tools/fetch_upscalers.sh --sha256 FILE (one of: $files)"
    exit 0
fi
if [ "${1:-}" = --files ]; then
    echo "$files"
    exit 0
fi
dest=${1:-}
[ -n "$dest" ] || fail "usage: tools/fetch_upscalers.sh DIR"
sha() { sha256sum "$1" | cut -d' ' -f1; }
# One file of a repository at a tag: from raw.githubusercontent.com, where
# github.com's /raw/ address redirects, then from github.com itself (on
# 2026-11-08 github.com answered 504 for these files while
# raw.githubusercontent.com served them). The pin checks either.
download() {  # REPO TAG PATH OUT
    local url
    for url in "https://raw.githubusercontent.com/$1/$2/$3" "https://github.com/$1/raw/$2/$3"; do
        echo "fetch_upscalers: $url"
        curl -sSL --fail --retry 3 -m 600 -o "$4" "$url" && return 0
    done
    rm -f "$4"
    return 1
}

cache=${UPSCALERS_CACHE:-build/upscalers}
mkdir -p "$cache" "$dest/licenses/fidelityfx" "$dest/licenses/dlss" "$dest/licenses/upscaler_bridge"
for f in $files; do
    want=$(pin "$f")
    if [ "$f" = upscaler_bridge.dll ]; then
        [ "$(sha "$BRIDGE")" = "$want" ] || fail "$BRIDGE is not the pinned bridge"
        cp "$BRIDGE" "$dest/$f"
        continue
    fi
    case "$f" in
        nvngx_*) repo=$DLSS_REPO tag=$DLSS_TAG path=$DLSS_DIR/$f ;;
        *) repo=$FFX_REPO tag=$FFX_TAG path=$FFX_DIR/$f ;;
    esac
    kept=$cache/$f
    if [ ! -f "$kept" ] || [ "$(sha "$kept")" != "$want" ]; then
        rm -f "$kept"
        download "$repo" "$tag" "$path" "$kept.part" || fail "download failed: $repo $tag $path"
        got=$(sha "$kept.part")
        if [ "$got" != "$want" ]; then
            rm -f "$kept.part"
            fail "$f: SHA-256 $got, not the pinned $want"
        fi
        mv "$kept.part" "$kept"
    fi
    cp "$kept" "$dest/$f"
done
cp third_party/fidelityfx/LICENSE-signedbin.md "$dest/licenses/fidelityfx/LICENSE.md"
cp third_party/dlss/LICENSE.txt "$dest/licenses/dlss/LICENSE.txt"
cp third_party/upscaler_bridge/LICENSE.txt "$dest/licenses/upscaler_bridge/LICENSE.txt"
echo "upscalers: FidelityFX $FFX_TAG (loader, upscaler, frame generation), DLSS $DLSS_TAG and the upscaler bridge into $dest, with their licences"
