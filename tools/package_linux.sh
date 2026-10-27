#!/bin/bash
# Packs a Linux build into the tarball another Linux machine runs from, the
# counterpart of tools/win_package.sh: the binary without its DWARF, the patch
# manifests it applies (patches/, looked for next to the binary), the plugin
# header, the example config, DXC's release library for the HLSL shader path
# with its licences (tools/fetch_dxc.sh), a README with what the machine must
# have, a launcher that keeps a log file, and the empty data/, logs/ and
# build/ (frame dumps) the run writes into: the release's Linux bundle. Nothing in it picks
# a server: bbhost's defaults are the live server's (https://thehuntersdream.com).
# The unstripped binary is kept beside the tarball under the same revision
# name, for symbolizing a crash report from that machine.
#   tools/package_linux.sh [BIN] [OUTDIR]
# BIN defaults to build-linux-portable/build/bbhost (tools/linux_portable_build.sh),
# the build that runs on other distributions; build/bbhost only runs on a
# machine as new as this one. Never packs game files (eboot, dump, keys).
set -eu
cd "$(dirname "$0")/.."
bin=${1:-build-linux-portable/build/bbhost}
out=${2:-build/linux}
[ -x "$bin" ] || { echo "no binary at $bin (tools/linux_portable_build.sh builds it)"; exit 1; }
rev=$(git -c core.excludesFile=/dev/null rev-parse --short HEAD 2>/dev/null || echo unknown)
# A release (HEAD at a v* tag) is named for its version, anything else for its revision.
if tag=$(git -c core.excludesFile=/dev/null describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null); then
    rev=$tag
fi
if [ -n "$(git -c core.excludesFile=/dev/null status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    rev="$rev+"
fi
name=bbhost-linux-$rev
dir=$out/$name
rm -rf "$dir"
mkdir -p "$dir/data" "$dir/logs" "$dir/plugins" "$dir/build"
cp -r patches "$dir/patches"
cp include/bbhost_plugin.h "$dir/plugins/"
cp LICENSE "$dir/LICENSE"
tools/package_plugins.sh "$(dirname "$bin")/plugins" "$dir/plugins"
objcopy --strip-debug "$bin" "$dir/bbhost" 2>/dev/null || llvm-objcopy --strip-debug "$bin" "$dir/bbhost"
chmod +x "$dir/bbhost"
cp "$bin" "$out/$name.debug"
cp bbhost.example.toml "$dir/bbhost.example.toml"
# The HLSL path's compiler beside the binary, where bbhost looks first.
tools/fetch_dxc.sh linux "$dir"
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

# What the binary still loads from the system, from its own dynamic section,
# so the README cannot drift from the build.
needed=$(readelf -d "$bin" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | tr '\n' ' ')
glibc=$(objdump -T "$bin" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
glibcxx=$(objdump -T "$bin" | grep -o 'GLIBCXX_[0-9.]*' | sort -uV | tail -1)

cat > "$dir/run-bbhost.sh" <<'EOF'
#!/bin/bash
# Starts bbhost and keeps its log in logs/ (the newest ten are kept).
#   ./run-bbhost.sh                       the per-user config (~/.config/bbhost/bbhost.toml)
#   BBHOST_CONFIG=other.toml ./run-bbhost.sh   that file on top of it
cd "$(dirname "$0")" || exit 1
cfg=${BBHOST_CONFIG:-bbhost.toml}
mkdir -p logs
log=logs/bbhost-$(date +%Y%m%d-%H%M%S).log
ls -1t logs/bbhost-*.log 2>/dev/null | tail -n +10 | xargs -r rm -f
args=()
[ -f "$cfg" ] && args=(--config "$cfg")
echo "bbhost: 設定 ${args[1]:-ユーザー設定}、ログ $(pwd)/$log"
# tee -i: Ctrl-C stops the game, not the log, so its last lines are kept.
./bbhost "${args[@]}" "$@" 2>&1 | tee -i "$log"
exit "${PIPESTATUS[0]}"
EOF
chmod +x "$dir/run-bbhost.sh"

tools/package_japanese.sh "$dir"
cp docs/japanese-release.md "$dir/README-linux.txt"
cat >> "$dir/README-linux.txt" <<EOF

この Linux ビルドの実行時依存関係:
  $needed
必要なバージョンの下限: $glibc $glibcxx
EOF


# Packed from a local copy with plain modes: on a filesystem without them
# (NFS here) every file would go out executable.
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
cp -r "$dir" "$stage/"
find "$stage" -type d -exec chmod 755 {} + && find "$stage" -type f -exec chmod 644 {} +
chmod 755 "$stage/$name/bbhost" "$stage/$name/run-bbhost.sh"
rm -f "$out/$name.tar.gz"
tar czf "$out/$name.tar.gz" -C "$stage" --owner=0 --group=0 "$name"
echo "package: $out/$name.tar.gz ($(( $(stat -c %s "$out/$name.tar.gz") / 1048576 )) MB); symbols: $out/$name.debug"
