#!/bin/bash
# Packs the Windows cross build into the zip a Windows machine runs from
# (docs/building.md), the release's Windows bundle: the exe without its DWARF,
# DXC's release dxcompiler.dll for the HLSL shader path with its licences
# (tools/fetch_dxc.sh), a launcher that keeps a log (run-bbhost.bat), a config
# template with Windows paths, the player's README, and the empty data/,
# build/ and logs/ directories the run writes into. Nothing in it picks a server: bbhost's
# defaults are the live server's (https://thehuntersdream.com). The
# unstripped exe is kept beside the zip under the same revision name, for
# symbolizing a crash report from that machine.
#   tools/win_package.sh [EXE] [OUTDIR]     (cmake --build build-win --target package-win)
set -eu
cd "$(dirname "$0")/.."
exe=${1:-build-win/bbhost.exe}
out=${2:-build/win}
rev=$(git -c core.excludesFile=/dev/null rev-parse --short HEAD 2>/dev/null || echo unknown)
# A release (HEAD at a v* tag) is named for its version, anything else for its revision.
if tag=$(git -c core.excludesFile=/dev/null describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null); then
    rev=$tag
fi
if [ -n "$(git -c core.excludesFile=/dev/null status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    rev="$rev+"
fi
name=bbhost-win-$rev
dir=$out/$name
rm -rf "$dir"
mkdir -p "$dir/data" "$dir/build"
cp -r patches "$dir/patches"
mkdir -p "$dir/plugins"
cp include/bbhost_plugin.h "$dir/plugins/"
cp LICENSE "$dir/LICENSE.txt"
tools/package_plugins.sh "$(dirname "$exe")/plugins" "$dir/plugins"
llvm-objcopy --strip-debug "$exe" "$dir/bbhost.exe"
cp "$exe" "$out/$name.debug.exe"
# The HLSL path's compiler beside the exe, where bbhost looks first.
tools/fetch_dxc.sh windows "$dir"
dxc_tag=$(cat third_party/dxc/VERSION)
# The pipeline list beside the binary, where bbhost reads it at each start and
# builds every pipeline it names (render.cpp pipeline_manifest_load): the
# pinned release asset (tools/fetch_pipeline_list.sh), or the local list
# BBHOST_PACKAGE_PIPELINE_LIST names (a new one under test);
# BBHOST_PIPELINE_LIST_FETCH=0 leaves it out (a development kit). It names the
# game's programs by hash and holds none of their code.
if [ -n "${BBHOST_PACKAGE_PIPELINE_LIST:-}" ]; then
    python3 tools/pipeline_list.py stats "$BBHOST_PACKAGE_PIPELINE_LIST" > /dev/null
    cp "$BBHOST_PACKAGE_PIPELINE_LIST" "$dir/pipeline-list.bin"
elif [ "${BBHOST_PIPELINE_LIST_FETCH:-1}" != 0 ]; then
    tools/fetch_pipeline_list.sh "$dir"
fi
sed -e 's#^app0 = .*#app0 = "C:/Games/CUSA03014"#' \
    -e 's#^eboot = .*#eboot = "./eboot-109-decrypted.bin"#' \
    -e 's#^skip_intro = false#skip_intro = true#' bbhost.example.toml > "$dir/bbhost.example.toml"
tools/package_japanese.sh "$dir"
cp docs/japanese-release.md "$dir/README.txt"

# The launcher that keeps a log (tools/win/run-bbhost.bat); CRLF for cmd and
# Notepad.
sed 's/\r*$/\r/' tools/win/run-bbhost.bat > "$dir/run-bbhost.bat"
sed -i 's/\r*$/\r/' "$dir/README.txt"
mkdir -p "$dir/logs"
(cd "$out" && rm -f "$name.zip" && zip -qr "$name.zip" "$name")
echo "package: $out/$name.zip ($(( $(stat -c %s "$out/$name.zip") / 1048576 )) MB); symbols: $out/$name.debug.exe"
