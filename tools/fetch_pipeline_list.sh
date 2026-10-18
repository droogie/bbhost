#!/bin/bash
# The pipeline list the packages carry (docs/full-tour.md): tools/full_tour.py's
# list of every pipeline a tour of the whole game drew, published as an asset
# of its own GitHub release and pinned below by tag and SHA-256. bbhost reads
# pipeline-list.bin from beside its executable at each start and builds every
# pipeline in it that the player's own manifest lacks, in the order a player
# reaches them (render.cpp pipeline_manifest_load). It names the game's
# programs by hash and holds none of their code; the fetch shaders' words it
# carries are the engine's, made at run time.
#
#   tools/fetch_pipeline_list.sh DIR          pipeline-list.bin into DIR
#   tools/fetch_pipeline_list.sh --sha256     the pinned SHA-256 (tools/check_release_bundle.sh)
#   tools/fetch_pipeline_list.sh --url        where it is published
#   PIPELINE_LIST_CACHE=DIR (default build/pipeline-list)   where the download is kept
#
# The download is checked against the pin and its record format against the
# one tools/pipeline_list.py writes (bbhost reads that one and the one before);
# a file that does not match is deleted and the run fails. A new list is a
# new release (a tag that is not v*, so the release workflow leaves it alone),
# its SHA-256 here, and its first-start check run again (full_tour.py compare).
# Until a list is published LIST_SHA256 is empty: nothing is fetched, the
# packages carry no list and the bundle check asks for none.
set -eu
cd "$(dirname "$0")/.."
LIST_REPO=droogie/bbhost
LIST_TAG=pipeline-list-v1
LIST_ASSET=pipeline-list.bin
LIST_SHA256=
LIST_FORMAT=3

fail() { echo "fetch_pipeline_list: $*" >&2; exit 1; }
url=https://github.com/$LIST_REPO/releases/download/$LIST_TAG/$LIST_ASSET
case "${1:-}" in
    --sha256) echo "$LIST_SHA256"; exit 0 ;;
    --url) echo "$url"; exit 0 ;;
esac
dest=${1:-}
[ -n "$dest" ] || fail "usage: tools/fetch_pipeline_list.sh DIR"
if [ -z "$LIST_SHA256" ]; then
    echo "fetch_pipeline_list: no list published yet (LIST_SHA256 is empty); none packed"
    exit 0
fi
sha() { sha256sum "$1" | cut -d' ' -f1; }
format() { od -An -tu4 -j4 -N4 "$1" | tr -d ' '; }

cache=${PIPELINE_LIST_CACHE:-build/pipeline-list}
mkdir -p "$cache"
file=$cache/$LIST_TAG-$LIST_ASSET
if [ ! -f "$file" ] || [ "$(sha "$file")" != "$LIST_SHA256" ]; then
    rm -f "$file"
    echo "fetch_pipeline_list: $url"
    curl -sSL --fail --retry 3 -m 600 -o "$file.part" "$url" || { rm -f "$file.part"; fail "download failed: $url (is the release published?)"; }
    got=$(sha "$file.part")
    if [ "$got" != "$LIST_SHA256" ]; then
        rm -f "$file.part"
        fail "$LIST_ASSET: SHA-256 $got, not the pinned $LIST_SHA256"
    fi
    mv "$file.part" "$file"
fi
[ "$(od -An -tx4 -N4 "$file" | tr -d ' ')" = 4d504242 ] || fail "$LIST_ASSET is not a pipeline list"
[ "$(format "$file")" = "$LIST_FORMAT" ] || fail "$LIST_ASSET is format $(format "$file"), not $LIST_FORMAT"
mkdir -p "$dest"
install -m 644 "$file" "$dest/pipeline-list.bin"
echo "fetch_pipeline_list: $LIST_TAG $LIST_ASSET -> $dest/pipeline-list.bin"
