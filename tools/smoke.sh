#!/bin/sh
# 60 s headless run: must end by timeout (124) with no core dump.
bin="$1"; log="$2"; src="$3"
# A per-user config folder of its own: a test never writes the player's
# (%APPDATA%/bbhost, ~/.local/share/bbhost).
export BBHOST_CONFIG_DIR=${BBHOST_CONFIG_DIR:-$(dirname "$log")/smoke-cfg}
BBHOST_SMOKE=1 timeout -k 5 60 "$bin" > "$log" 2>&1; rc=$?
"$src/tools/coverage.py" "$log" || exit 1
test "$rc" -eq 124 || exit 1
# A timeout before graphics initialized used to pass (zero stubs, zero core
# dumps). Require actual game-loop progress, not just a process surviving.
if ! grep -q 'sceGnmSubmitAndFlip #' "$log"; then
    echo "smoke failed: no submitted flips (check device access/startup)"
    exit 1
fi
