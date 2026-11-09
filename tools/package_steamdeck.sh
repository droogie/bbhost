#!/bin/bash
# The Steam Deck kit: the Linux package (tools/package_linux.sh's) with the
# Deck's settings - tools/steamdeck/bbhost-steamdeck.toml: its screen's
# 1280x800, full screen, 30 fps - STEAMDECK.md, and the launcher pointed at
# them. The game's paths stay in the per-user config, as for every package.
# Without DLSS: the Deck's GPU is AMD's, so the NGX bridge, NVIDIA's runtime,
# their licences and the README's Upscaling section are left out.
#   tools/package_steamdeck.sh LINUX_TARBALL OUTDIR
set -eu
cd "$(dirname "$0")/.."
lin_tgz=$(realpath "$1"); out=$(realpath -m "$2")
sd=$PWD/tools/steamdeck
mkdir -p "$out"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
(cd "$work" && tar xzf "$lin_tgz")
ldir=$(ls -d "$work"/bbhost-linux-* | head -1)
name=$(basename "$ldir" | sed 's/^bbhost-linux-/bbhost-steamdeck-/')
cp "$sd/bbhost-steamdeck.toml" "$sd/STEAMDECK.md" "$ldir/"
sed -i 's/^cfg=\${BBHOST_CONFIG:-bbhost.toml}/cfg=${BBHOST_CONFIG:-bbhost-steamdeck.toml}/' "$ldir/run-bbhost.sh"
grep -q 'bbhost-steamdeck.toml' "$ldir/run-bbhost.sh" || { echo "run-bbhost.sh: config line not found"; exit 1; }
sed -i 's/^  bbhost.example.toml lists every setting./  bbhost.example.toml lists every setting; STEAMDECK.md has the Steam Deck steps./' "$ldir/README-linux.txt"
grep -q 'STEAMDECK.md' "$ldir/README-linux.txt" || { echo "README-linux.txt: settings line not found"; exit 1; }
rm -f "$ldir/libbbhost_ngx_bridge.so" "$ldir"/libnvidia-ngx-dlss.so.*
rm -rf "$ldir/licenses/dlss" "$ldir/licenses/ngx_bridge" "$ldir/licenses/upscaler_bridge"
sed -i '/^Upscaling$/,/^Setup$/{/^Setup$/!d}' "$ldir/README-linux.txt"
if grep -q 'libnvidia-ngx-dlss' "$ldir/README-linux.txt"; then echo "README-linux.txt: the Upscaling section is still there"; exit 1; fi
find "$ldir" -type d -exec chmod 755 {} + && find "$ldir" -type f -exec chmod 644 {} +
chmod 755 "$ldir/bbhost" "$ldir/run-bbhost.sh"
mv "$ldir" "$work/$name"
(cd "$work" && rm -f "$out/$name.tar.gz" && tar czf "$out/$name.tar.gz" --owner=0 --group=0 "$name")
ls -la "$out/$name.tar.gz"
