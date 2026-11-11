#!/bin/bash
# AMD FidelityFX SDK's source for bbhost's Linux FSR 3 library
# (libbbhost_fidelityfx_vk.so, src/ffx_vk): the FidelityFX API, its Vulkan
# backend and the FSR 3 upscaler, from AMD's FidelityFX SDK repository at a
# pinned tag - the files third_party/fidelityfx/source.sha256 lists, each
# pinned by SHA-256 and all on the MIT list of the SDK's own licence
# (Kits/FidelityFX/docs/license.md, the first of them). None of them is in this
# repository. AMD's sources are used as they are, but for the changes in
# third_party/fidelityfx/patches (each says why), applied here after the pins
# are checked. Without them the library is not built and FSR stays off on
# Linux; everything else builds and runs as before. The same tag gives the
# Windows package AMD's signed DLLs (tools/fetch_upscalers.sh), so both run
# FSR 3.1.6.
#
#   tools/fetch_fidelityfx.sh sdk DIR   the files into DIR/Kits/FidelityFX/..., patched
#   tools/fetch_fidelityfx.sh --tag     AMD's tag the files are pinned at
#   FFX_CACHE=DIR (default build/fidelityfx)   where the downloads are kept
#
# A download that does not match its pin is deleted and the run fails; a kept
# file is checked again at every run. A new version is a new tag, the files'
# SHA-256s in source.sha256 and the patches made again against it.
set -eu
cd "$(dirname "$0")/.."
TAG=v2.4.0
REPO=GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
LIST=third_party/fidelityfx/source.sha256
PATCHES=third_party/fidelityfx/patches

fail() { echo "fetch_fidelityfx: $*" >&2; exit 1; }
case "${1:-}" in
    --tag) echo "$TAG"; exit 0 ;;
    sdk) ;;
    *) fail "usage: tools/fetch_fidelityfx.sh sdk DIR | --tag" ;;
esac
dest=${2:-}
[ -n "$dest" ] || fail "usage: tools/fetch_fidelityfx.sh sdk DIR"
command -v patch > /dev/null || fail "patch is needed to apply $PATCHES"
sha() { sha256sum "$1" | cut -d' ' -f1; }

cache=${FFX_CACHE:-build/fidelityfx}/$TAG
mkdir -p "$cache"
# One file of AMD's repository: from raw.githubusercontent.com, then from
# github.com itself (as tools/fetch_ngx.sh). The pin checks either.
download() {  # PATH OUT
    local url
    for url in "https://raw.githubusercontent.com/$REPO/$TAG/$1" "https://github.com/$REPO/raw/$TAG/$1"; do
        curl -sSL --fail --retry 2 --connect-timeout 15 -m 300 -o "$2" "$url" && return 0
    done
    rm -f "$2"
    return 1
}
fetched=0
while read -r want path; do
    case "$want" in '' | '#'*) continue ;; esac
    kept=$cache/$path
    if [ -f "$kept" ] && [ "$(sha "$kept")" = "$want" ]; then continue; fi
    mkdir -p "$(dirname "$kept")"
    rm -f "$kept"
    download "$path" "$kept.part" || fail "download failed: $REPO $TAG $path"
    got=$(sha "$kept.part")
    if [ "$got" != "$want" ]; then
        rm -f "$kept.part"
        fail "$path: SHA-256 $got, not the pinned $want"
    fi
    mv "$kept.part" "$kept"
    fetched=$((fetched + 1))
done < "$LIST"
[ "$fetched" = 0 ] || echo "fetch_fidelityfx: $fetched files of $REPO $TAG fetched"

# A fresh tree each run: the pinned files, then the patches.
rm -rf "$dest.part"
mkdir -p "$dest.part"
while read -r want path; do
    case "$want" in '' | '#'*) continue ;; esac
    mkdir -p "$dest.part/$(dirname "$path")"
    cp "$cache/$path" "$dest.part/$path"
done < "$LIST"
for p in "$PATCHES"/*.patch; do
    [ -f "$p" ] || continue
    patch -d "$dest.part" -p1 --forward --batch --quiet < "$p" || fail "$p does not apply"
done
rm -rf "$dest"
mv "$dest.part" "$dest"
echo "fidelityfx: AMD FidelityFX SDK $TAG's sources for the Linux FSR 3 library in $dest ($(grep -c '  ' "$LIST") files, $(ls "$PATCHES"/*.patch 2> /dev/null | wc -l) patches), under the SDK's MIT licence"
