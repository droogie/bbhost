#!/bin/bash
# A kit for running the HLSL path's driver-acceptance sweep (gcn2hlsl --sweep,
# tools/pipeline_probe.h) on another machine - a Windows PC, a Steam Deck, a
# tester's GPU: gcn2hlsl built for it, the release DXC beside it
# (tools/fetch_dxc.sh) and a script that runs the sweep on that machine's own
# game folder and keeps the summary and the per-program list. No game file
# goes in: the sweep reads the game where it is installed, and nothing it
# makes of the game's shaders is written out (no --out).
#   tools/package_hlsl_sweep.sh windows <gcn2hlsl.exe> OUTDIR   the Windows cross build's (tools/ci_windows.sh's build tree)
#   tools/package_hlsl_sweep.sh linux <gcn2hlsl> OUTDIR         a portable one: PORTABLE_TARGETS=gcn2hlsl tools/linux_portable_build.sh
# On the machine:
#   run-sweep.bat <game folder>      (Windows)     ./run-sweep.sh <game folder>      (Linux, Steam Deck)
# where <game folder> holds dvdroot_ps4 (CUSA00900); sweep-summary.txt and
# sweep-list.txt are written beside the script (SWEEP_NAME=<name>: <name>-summary.txt
# and <name>-list.txt, to keep one run's beside another's).
set -eu
cd "$(dirname "$0")/.."
target=${1:-}; bin=${2:-}; out=${3:-}
[ -n "$out" ] && [ -f "$bin" ] || { echo "usage: tools/package_hlsl_sweep.sh linux|windows GCN2HLSL OUTDIR" >&2; exit 2; }
out=$(realpath -m "$out")
rev=$(git -c core.excludesFile=/dev/null rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git -c core.excludesFile=/dev/null status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    rev="$rev+"
fi
name=hlsl-sweep-$target-$rev
dir=$out/$name
rm -rf "$dir"
mkdir -p "$dir"
cp LICENSE "$dir/LICENSE.txt"
case "$target" in
windows)
    llvm-objcopy --strip-debug "$bin" "$dir/gcn2hlsl.exe"
    tools/fetch_dxc.sh windows "$dir"
    sed 's/\r*$/\r/' > "$dir/run-sweep.bat" <<'EOF'
@echo off
rem The HLSL path's driver-acceptance sweep on this PC's GPU (gcn2hlsl --sweep):
rem every shader of the game lifted as bbhost lifts it here, compiled by DXC,
rem and both modules built into the pipelines bbhost makes, by the driver.
rem   run-sweep.bat <game folder>      the folder that holds dvdroot_ps4
rem   run-sweep.bat <game folder> --gpu 1    another GPU (bbhost's log numbers them)
rem It takes 5 to 15 minutes. Send sweep-summary.txt and sweep-list.txt
rem (set SWEEP_NAME=other first: other-summary.txt and other-list.txt).
setlocal
cd /d "%~dp0"
rem The kit's folder, kept before shift moves the arguments (and the script's own name) along.
set here=%~dp0
if "%~1"=="" (
  echo usage: run-sweep.bat ^<game folder^>
  exit /b 2
)
if "%SWEEP_NAME%"=="" set SWEEP_NAME=sweep
set game=%~1
shift
gcn2hlsl.exe --sweep --app0 "%game%" --list %SWEEP_NAME%-list.txt %1 %2 %3 %4 > %SWEEP_NAME%-summary.txt 2>&1
set rc=%errorlevel%
type %SWEEP_NAME%-summary.txt
echo.
echo exit %rc%: the summary is %here%%SWEEP_NAME%-summary.txt, every program in %here%%SWEEP_NAME%-list.txt
exit /b %rc%
EOF
    (cd "$out" && rm -f "$name.zip" && zip -qr "$name.zip" "$name")
    echo "kit: $out/$name.zip ($(( $(stat -c %s "$out/$name.zip") / 1048576 )) MB)"
    ;;
linux)
    objcopy --strip-debug "$bin" "$dir/gcn2hlsl" 2>/dev/null || llvm-objcopy --strip-debug "$bin" "$dir/gcn2hlsl"
    tools/fetch_dxc.sh linux "$dir"
    cat > "$dir/run-sweep.sh" <<'EOF'
#!/bin/sh
# The HLSL path's driver-acceptance sweep on this machine's GPU (gcn2hlsl --sweep):
# every shader of the game lifted as bbhost lifts it here, compiled by DXC,
# and both modules built into the pipelines bbhost makes, by the driver.
#   ./run-sweep.sh <game folder>             the folder that holds dvdroot_ps4
#   ./run-sweep.sh <game folder> --gpu 1     another GPU (bbhost's log numbers them)
# It takes 5 to 15 minutes. Send sweep-summary.txt and sweep-list.txt
# (SWEEP_NAME=other ./run-sweep.sh ...: other-summary.txt and other-list.txt).
cd "$(dirname "$0")" || exit 1
if [ -z "${1:-}" ]; then
    echo "usage: ./run-sweep.sh <game folder>"
    exit 2
fi
game=$1
shift
name=${SWEEP_NAME:-sweep}
./gcn2hlsl --sweep --app0 "$game" --list "$name-list.txt" "$@" > "$name-summary.txt" 2>&1
rc=$?
cat "$name-summary.txt"
echo
echo "exit $rc: the summary is $(pwd)/$name-summary.txt, every program in $(pwd)/$name-list.txt"
exit $rc
EOF
    stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
    cp -r "$dir" "$stage/"
    find "$stage" -type d -exec chmod 755 {} + && find "$stage" -type f -exec chmod 644 {} +
    chmod 755 "$stage/$name/gcn2hlsl" "$stage/$name/run-sweep.sh"
    rm -f "$out/$name.tar.gz"
    tar czf "$out/$name.tar.gz" -C "$stage" --owner=0 --group=0 "$name"
    echo "kit: $out/$name.tar.gz ($(( $(stat -c %s "$out/$name.tar.gz") / 1048576 )) MB)"
    ;;
*)
    echo "the target is linux or windows" >&2
    exit 2
    ;;
esac
