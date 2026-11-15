#!/bin/bash
# The AppImage: the Linux package (tools/package_linux.sh's) as one executable
# file, for players who would rather not keep a folder - the same files,
# under bbhost/ in the image, run read-only from where the image mounts
# itself. Nothing bbhost writes goes beside its executable: the settings,
# saves and caches are the per-user folders' (~/.local/share/bbhost) as for
# every package, and the image's AppRun keeps each session's log, and F12's
# frame dumps, in ~/.local/share/bbhost/logs (the newest ten logs are kept)
# where the tarball's run-bbhost.sh keeps them in its logs/. A newer
# official plugin fetched from the setup window goes to the per-user plugins
# folder, as from any package.
#
# The image is AppImage's type-2 runtime, pinned below by release and
# SHA-256, followed by the folder packed by mksquashfs (squashfs-tools), what
# appimagetool itself does. That runtime is static and mounts the image with
# its own FUSE 3: the player needs no libfuse2, and on a machine without FUSE
# (some containers) `bbhost-....AppImage --appimage-extract-and-run` runs it.
# It needs what the tarball needs (README-linux.txt: glibc, the Vulkan loader
# and a driver, libcurl, zlib), which it does not carry either. The runtime's
# licences and where its source is (libfuse, under the LGPL, is linked into
# it) go into the image's licenses/appimage-runtime, from
# third_party/appimage-runtime; tools/check_release_bundle.sh fails an image
# without them.
#   tools/package_appimage.sh LINUX_TARBALL OUTDIR
#   APPIMAGE_CACHE=DIR (default build/appimage-cache)   where the runtime is kept
set -eu
cd "$(dirname "$0")/.."
RUNTIME_TAG=20251108
RUNTIME_SHA256=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d
fail() { echo "package_appimage: $*" >&2; exit 1; }
[ $# -eq 2 ] || fail "usage: tools/package_appimage.sh LINUX_TARBALL OUTDIR"
lin_tgz=$(realpath "$1"); out=$(realpath -m "$2")
command -v mksquashfs > /dev/null || fail "no mksquashfs (the squashfs-tools package)"
sha() { sha256sum "$1" | cut -d' ' -f1; }

cache=$(realpath -m "${APPIMAGE_CACHE:-build/appimage-cache}")
runtime=$cache/runtime-x86_64-$RUNTIME_TAG
mkdir -p "$cache"
if [ ! -f "$runtime" ] || [ "$(sha "$runtime")" != "$RUNTIME_SHA256" ]; then
    curl -fsSL --retry 3 -o "$runtime.part" \
        "https://github.com/AppImage/type2-runtime/releases/download/$RUNTIME_TAG/runtime-x86_64" ||
        fail "the runtime $RUNTIME_TAG could not be fetched"
    got=$(sha "$runtime.part")
    [ "$got" = "$RUNTIME_SHA256" ] || { rm -f "$runtime.part"; fail "the runtime's SHA-256 is $got, not the pinned $RUNTIME_SHA256"; }
    mv "$runtime.part" "$runtime"
fi

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
(cd "$work" && tar xzf "$lin_tgz")
ldir=$(ls -d "$work"/bbhost-linux-* | head -1)
[ -x "$ldir/bbhost" ] || fail "$1 holds no bbhost-linux-*/bbhost"
name=$(basename "$ldir")-x86_64.AppImage
app=$work/AppDir
mkdir -p "$app/usr/share/applications" "$app/usr/share/icons/hicolor/256x256/apps"
mv "$ldir" "$app/bbhost"
# The folder's launcher and the empty folders a run writes into have no use
# in an image nobody can write to; AppRun is the launcher.
rm -f "$app/bbhost/run-bbhost.sh"
rm -rf "$app/bbhost/logs" "$app/bbhost/build"
sed -i 's|^  Untar this folder somewhere of its own, for example ~/Games/bbhost, and run$|  (In the AppImage, run the .AppImage itself wherever ./run-bbhost.sh is\n  named below; its logs go to ~/.local/share/bbhost/logs.)\n  Untar this folder somewhere of its own, for example ~/Games/bbhost, and run|' "$app/bbhost/README-linux.txt"
grep -q 'In the AppImage' "$app/bbhost/README-linux.txt" || fail "README-linux.txt: the Setup line not found"
mkdir -p "$app/bbhost/licenses/appimage-runtime"
cp third_party/appimage-runtime/* "$app/bbhost/licenses/appimage-runtime/"

cat > "$app/AppRun" << 'EOF'
#!/bin/bash
# Starts bbhost from the image and keeps its log in
# ~/.local/share/bbhost/logs (the newest ten are kept), the folder F12's frame
# dumps go to as well (BBHOST_CAPTURE_DIR names another). The arguments go to
# bbhost: --setup opens the setup window, --config FILE adds FILE's settings.
here=${APPDIR:-$(dirname "$(readlink -f "$0")")}
logs=${XDG_DATA_HOME:-$HOME/.local/share}/bbhost/logs
if ! mkdir -p "$logs" 2> /dev/null; then
    # No log then; F12's dumps go to the temporary folder, not the image.
    export BBHOST_CAPTURE_DIR=${BBHOST_CAPTURE_DIR:-${TMPDIR:-/tmp}}
    exec "$here/bbhost/bbhost" "$@"
fi
export BBHOST_CAPTURE_DIR=${BBHOST_CAPTURE_DIR:-$logs}
log=$logs/bbhost-$(date +%Y%m%d-%H%M%S).log
(cd "$logs" && ls -1t bbhost-*.log 2> /dev/null | tail -n +10 | xargs -r rm -f)
echo "bbhost: log $log"
# tee -i: Ctrl-C stops the game, not the log, so its last lines are kept.
"$here/bbhost/bbhost" "$@" 2>&1 | tee -i "$log"
exit "${PIPESTATUS[0]}"
EOF

cat > "$app/usr/share/applications/bbhost.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=bbhost
Comment=Bloodborne on PC from your own game dump
Exec=bbhost
Icon=bbhost
Categories=Game;
Terminal=false
EOF
ln -s usr/share/applications/bbhost.desktop "$app/bbhost.desktop"
# The icon: res/bbhost.ico's 256 pixel image, which the file holds as a PNG.
python3 -I - res/bbhost.ico "$app/usr/share/icons/hicolor/256x256/apps/bbhost.png" << 'EOF'
import struct, sys
d = open(sys.argv[1], 'rb').read()
for i in range(struct.unpack('<H', d[4:6])[0]):
    w, h, _, _, _, _, size, off = struct.unpack('<BBBBHHII', d[6 + 16 * i:22 + 16 * i])
    if w == 0 and h == 0 and d[off:off + 8] == b'\x89PNG\r\n\x1a\n':
        open(sys.argv[2], 'wb').write(d[off:off + size])
        break
else:
    sys.exit('res/bbhost.ico: no 256 pixel PNG image')
EOF
ln -s usr/share/icons/hicolor/256x256/apps/bbhost.png "$app/bbhost.png"
ln -s bbhost.png "$app/.DirIcon"

# Plain modes, as the tarball's, whatever the filesystem the work was on.
find "$app" -type d -exec chmod 755 {} + && find "$app" -type f -exec chmod 644 {} +
chmod 755 "$app/AppRun" "$app/bbhost/bbhost"
for f in "$app"/bbhost/*.so "$app"/bbhost/*.so.*; do [ ! -f "$f" ] || chmod 755 "$f"; done

mkdir -p "$out"
mksquashfs "$app" "$work/image.sqfs" -all-root -noappend -comp zstd -Xcompression-level 19 -b 1M -quiet > /dev/null
rm -f "$out/$name"
cat "$runtime" "$work/image.sqfs" > "$out/$name"
chmod 755 "$out/$name"
echo "appimage: $out/$name ($(( $(stat -c %s "$out/$name") / 1048576 )) MB)"
