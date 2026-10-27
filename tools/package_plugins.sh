#!/bin/bash
# Copies the official plugins of a build into a package's plugins/ folder,
# without their DWARF, and signs each one (docs/plugins.md, "Official
# plugins"): <file>.sig is the release key's Ed25519 signature over the file
# as shipped, which the host checks before it loads the plugin.
#   tools/package_plugins.sh BUILD_PLUGIN_DIR PACKAGE_PLUGIN_DIR
# The key comes from RELEASE_SIGNING_KEY (the PEM text: the release workflow's
# secret) or RELEASE_SIGNING_KEY_FILE; without either the plugins ship unsigned.
set -eu
from=$1; to=$2
plugins="region_switch randomizer boss_rush mutators debug_menu"
mkdir -p "$to"
key=""
if [ -n "${RELEASE_SIGNING_KEY:-}" ]; then
    key=$(mktemp); chmod 600 "$key"
    printf '%s\n' "$RELEASE_SIGNING_KEY" > "$key"
elif [ -n "${RELEASE_SIGNING_KEY_FILE:-}" ]; then
    key=$RELEASE_SIGNING_KEY_FILE
fi
n=0
for p in $plugins; do
    for f in "$from/$p.so" "$from/$p.dll"; do
        [ -f "$f" ] || continue
        out="$to/$(basename "$f")"
        objcopy --strip-debug "$f" "$out" 2>/dev/null || llvm-objcopy --strip-debug "$f" "$out" 2>/dev/null || cp "$f" "$out"
        if [ -n "$key" ]; then
            openssl pkeyutl -sign -inkey "$key" -rawin -in "$out" -out "$out.sig"
        fi
        n=$((n + 1))
    done
done
[ -n "${RELEASE_SIGNING_KEY:-}" ] && rm -f "$key"
echo "plugins: $n packed into $to$( [ -n "$key" ] && echo ', signed' || echo ', unsigned')"
# Every package carries every official plugin: a missing one is a build that
# did not make it, not something to ship around.
want=$(echo $plugins | wc -w)
if [ "$n" -lt "$want" ]; then
    echo "package_plugins: $n of $want official plugins found in $from - build their targets first" >&2
    exit 1
fi
