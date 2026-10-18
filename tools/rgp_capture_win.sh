#!/bin/bash
# An RGP (Radeon GPU Profiler) capture of a perf pass on an AMD GPU: the
# Radeon Developer Panel's command line connects to bbhost as it starts, and
# at CAPTURE_AT seconds into the run asks for a frame (RGP's frame is one
# vkQueuePresentKHR to the next, so the run must present: fullscreen by
# default, where AMD's driver also resolves the occlusion queries).
#
#   tools/rgp_capture_win.sh NAME [VAR=value ...]   (the VARs go to bbhost)
#
# RDTS: the Radeon Developer Tool Suite folder (default ~/Tools/RadeonDeveloper
# ToolSuite-*); CAPTURE_AT (100) seconds; RGP_EXTRA: more panel options (say
# --rgp-counter-collection or --rgp-instruction-tracing). The perf pass's own
# variables apply (PERF_SECONDS, default 150 here; PERF_MENUTAP=1, the default
# here, so a press never reaches the world). Writes build/perf/NAME.rgp next to
# the run's log.
set -u
cd "$(dirname "$0")/.."
NAME=${1:?NAME}
shift
root=$(pwd -W 2>/dev/null || pwd)
rdts=${RDTS:-$(ls -d "$HOME"/Tools/RadeonDeveloperToolSuite-* 2>/dev/null | tail -1)}
[ -x "$rdts/RadeonDeveloperPanelCLI.exe" ] || { echo "no RadeonDeveloperPanelCLI.exe under ${rdts:-~/Tools} (RDTS)"; exit 1; }
at=${CAPTURE_AT:-100}
secs=${PERF_SECONDS:-150}
mkdir -p build/perf
rm -f "build/perf/$NAME.rgp"
(
    sleep "$at"
    echo c
    sleep 25
    echo q
) | "$rdts/RadeonDeveloperPanelCLI.exe" -m profiling -p bbhost -o "$root/build/perf/$NAME.rgp" ${RGP_EXTRA:-} 2>&1 | tr '\r' '\n' | grep -v 'MB *$' > "build/perf/$NAME.rdp.txt" &
panel=$!
sleep 3
PERF_SECONDS=$secs PERF_FULLSCREEN=${PERF_FULLSCREEN-1} PERF_MENUTAP=${PERF_MENUTAP:-1} tools/perf_pass_win.sh "$NAME" "$@"
wait $panel 2>/dev/null
tail -5 "build/perf/$NAME.rdp.txt"
ls -la "build/perf/$NAME"*.rgp 2>/dev/null || echo "no capture: build/perf/$NAME.rdp.txt"
