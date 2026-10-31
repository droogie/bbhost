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
echo "bbhost: config ${args[1]:-per-user}, log $(pwd)/$log"
# tee -i: Ctrl-C stops the game, not the log, so its last lines are kept.
./bbhost "${args[@]}" "$@" 2>&1 | tee -i "$log"
exit "${PIPESTATUS[0]}"
EOF
chmod +x "$dir/run-bbhost.sh"

cat > "$dir/README-linux.txt" <<EOF
bbhost for Linux, build $name
==================================

bbhost runs Bloodborne on PC: the game's own 1.09 executable, its calls into
the PS4's system libraries answered by bbhost and its graphics recompiled for
Vulkan. Free software under the GNU GPL, version 3 or later (LICENSE); the
source is at https://github.com/droogie/bbhost.

You need
  - x86-64 Linux with glibc ${glibc#GLIBC_} or newer (Ubuntu 24.04, Linux Mint 22, Debian 13,
    Fedora 39 or anything newer; Arch; a current SteamOS). Check with: ldd --version
  - A Vulkan 1.3 GPU driver and the Vulkan loader. Check with: vulkaninfo --summary
  - These system libraries (every desktop has most of them already):
        $needed
    Packages that provide them:
        Ubuntu / Debian / Mint : libvulkan1 mesa-vulkan-drivers libcurl4t64 zlib1g
        Fedora                 : vulkan-loader mesa-vulkan-drivers libcurl zlib
        Arch / SteamOS         : vulkan-icd-loader vulkan-radeon|vulkan-intel|nvidia-utils curl zlib
    NVIDIA: the proprietary driver brings its own Vulkan driver.
  - X11 or Wayland, and PulseAudio, PipeWire or ALSA for sound (SDL3 is built
    in and uses what it finds).
  - The eboot.bin of the game's 1.09 update, decrypted to an ELF (SHA-256
    941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7; bbhost
    stops at start and says why when it is another version), and the game
    dump with the 1.09 update's files copied over it: the CUSA00900 folder
    that contains dvdroot_ps4, whose sce_sys/param.sfo says APP_VER 01.09
    (or the update kept beside it in CUSA00900-UPDATE, the way shadPS4 keeps
    a game library). Neither is in this package.

Built static: libstdc++ ($glibcxx), SDL3, ffmpeg (movie decoders only),
SPIRV-Tools.

libdxcompiler.so beside it is the DirectX Shader Compiler $dxc_tag, Microsoft's
release unmodified (https://github.com/microsoft/DirectXShaderCompiler,
licences in licenses/dxc): the HLSL shader path, on by default, compiles with
it, and it loads the system's libstdc++.so.6 (GCC 11 or newer). The game's
shaders are compiled in the background after a start's own shader builds (on
an integrated GPU such as the Steam Deck's, only while the title and loading
screens are up), and the translator's own are drawn until then; the setup
window's Shaders tab compiles them all at once (Compile shaders now), and its
Shaders setting picks Lifted to keep the translator's own. Without it that
path keeps every shader as it was; nothing else uses it.

Setup
  Untar this folder somewhere of its own, for example ~/Games/bbhost, and run
  ./run-bbhost.sh from a terminal there. The first time, the setup window
  opens: pick the game folder and the decrypted eboot with the Browse buttons
  (each turns green when it is right), then Play. ./run-bbhost.sh --setup
  opens it again.

  The settings are kept in ~/.config/bbhost/bbhost.toml, which every copy of
  bbhost reads, so a new download or an update needs nothing set again. The
  F10 settings and your account sign-in are kept beside it, saves and caches
  in ~/.local/share/bbhost/data.
  bbhost.example.toml lists every setting.

Run
  ./run-bbhost.sh          (the log goes to logs/ and the terminal)
  The first visit to an area stutters for a moment while its graphics are
  prepared for your GPU; later visits do not. F10 in the game window opens
  the PC settings: resolution, window mode, V-Sync, frame cap, mouse, key
  bindings and your account.

Online
  bbhost plays online on the community server, https://thehuntersdream.com,
  with other bbhost players (not with PS4s). The game stays offline until this
  PC is linked to an account, and the account's name is your name online.
  Link it once, in the setup window's Account section or with F10 > ACCOUNT
  at the title screen:
    - Log in with Discord (Link with Discord in F10): a code appears and your
      browser opens https://thehuntersdream.com/link. Sign in there with
      Discord, check that the page names this PC, and approve.
    - Create account on this PC: type a name (3 to 16 letters, digits, _ or
      -). A recovery code appears once. Write it down: it is the only way to
      get the account back on another PC or after reinstalling (Recover
      account takes the name and that code).
  Then choose Play Online at the title.

  Co-op: the host rings the Beckoning Bell and the helper the Small Resonant
  Bell, in the same area. Invasions: the Sinister Resonant Bell. The website
  has the live map of deaths and messages, the leaderboards and your account
  page (Public profile puts your name on them).

  Other players reach you on UDP port 9307: a firewall on this PC must let it
  in. If summons keep failing with one person, forwarding that port to this
  PC on your router often fixes it. Two PCs in one house: set p2p_addr under
  [online] in ~/.config/bbhost/bbhost.toml to each PC's local address.

Steam Deck
  The release's bbhost-steamdeck tarball is this package with the Deck's
  settings (its screen's 1280x800, full screen, 30 fps) and STEAMDECK.md,
  the steps for adding it to Steam and playing in Game Mode.

Plugins
  plugins/ holds the official ones, signed by the bbhost release key
  (the .sig beside each). Turn them on and set them up in the setup
  window's Plugins tab (bbhost --setup), which can also fetch the latest
  official plugins; in the game, F9 opens the plugin menu (start the boss
  rush there). Or in the per-user bbhost.toml:
      [plugins]
      randomizer = true    # every pickup and shop shuffled from a seed
      boss_rush = true     # the game's bosses back to back, timed
      mutators = true      # speed, bullet time, one-hit, chaos and more
      quake = true         # first person with Quake 3's movement and guns
      lohit = true         # the map's collision drawn in place of the map
      debug_menu = true    # the developers' debug menu: \` opens it
  Each one's settings go in a section of its own name ([randomizer],
  [boss_rush], [mutators], [quake], [lohit]); docs/plugins.md in the repository lists them.
  Use the boss rush on a save of its own. Cheating against other players
  with the debug menu gets an account banned; trying it out online with
  friends is fine.

Reporting a problem
  Send the log of that session from logs/ (privately: it contains your
  internet address), with what happened, when, your account name and your
  GPU. If the game crashed, the log ends with a block starting "SIGSEGV pc="
  with the registers and the frames; the release's $name.debug symbolizes
  it.

  A wrong picture (wrong colours, a green or black screen, missing parts):
  press F12 in the game while it shows. bbhost writes that frame, the images
  the game drew it from and its list of draws into a folder of its own in
  logs/ (logs/f12-<date>-<time>; the log names it). Pack that folder
  (tar czf f12.tar.gz logs/f12-<date>-<time>) and send it with the log.
  BBHOST_F12_PNG=1 ./run-bbhost.sh writes PNG files instead of PPM (any
  image viewer opens them, a fifth of the size); BBHOST_F12_TEXTURES=1 adds
  every texture the frame used (the game pauses while it writes them).
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
