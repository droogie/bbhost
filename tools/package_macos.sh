#!/bin/bash
# Packs a macOS build into a zip, the counterpart of tools/package_linux.sh,
# with the debug information beside it as <name>.dSYM.zip.
#   tools/package_macos.sh [BIN] [OUTDIR]
# The SDK is VULKAN_SDK=<sdk>/macOS, or the binary's rpath's, or the newest
# ~/VulkanSDK/<version>/macOS. Never packs game files.
#
# The loader and MoltenVK are bundled with relative paths (rpath, vulkan/icd.d)
# so the folder runs anywhere with no SDK or VK_* variable. Signed ad hoc, which
# keeps the files consistent after strip and install_name_tool; not notarized.
set -eu
cd "$(dirname "$0")/.."
bin=${1:-build-macos/build/bbhost}
out=${2:-build/mac}
[ -x "$bin" ] || { echo "no binary at $bin (tools/build_macos.sh builds it)"; exit 1; }
[ "$(uname)" = Darwin ] || { echo "package_macos.sh runs on macOS (codesign, dsymutil)"; exit 1; }
# Plugin signatures are Ed25519, which LibreSSL can't make: fail early.
if [ -n "${RELEASE_SIGNING_KEY:-}${RELEASE_SIGNING_KEY_FILE:-}" ] \
    && ! openssl version 2>/dev/null | grep -qE '^OpenSSL [3-9]'; then
    echo "signing the plugins needs OpenSSL 3, and openssl here is $(openssl version 2>/dev/null || echo missing):"
    echo "  brew install openssl@3; export PATH=\"\$(brew --prefix openssl@3)/bin:\$PATH\""
    exit 1
fi
rev=$(git -c core.excludesFile=/dev/null rev-parse --short HEAD 2>/dev/null || echo unknown)
if tag=$(git -c core.excludesFile=/dev/null describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null); then
    rev=$tag
fi
if [ -n "$(git -c core.excludesFile=/dev/null status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    rev="$rev+"
fi

# The SDK whose loader the binary was linked against.
sdk=${VULKAN_SDK:-}
if [ -z "$sdk" ]; then
    for r in $(otool -l "$bin" | awk '/LC_RPATH/ { getline; getline; print $2 }'); do
        [ -f "$r/libvulkan.1.dylib" ] && { sdk=$(dirname "$r"); break; }
    done
fi
if [ -z "$sdk" ]; then
    sdk=$(ls -d "$HOME"/VulkanSDK/*/macOS 2>/dev/null | sort -V | tail -1 || true)
fi
[ -n "$sdk" ] && [ -f "$sdk/lib/libvulkan.1.dylib" ] && [ -f "$sdk/lib/libMoltenVK.dylib" ] \
    || { echo "no Vulkan SDK with the loader and MoltenVK (set VULKAN_SDK=<sdk>/macOS)"; exit 1; }
sdkver=$(basename "$(dirname "$sdk")")

name=bbhost-mac-$rev
dir=$out/$name
rm -rf "$dir" "$out/$name.zip" "$out/$name.dSYM" "$out/$name.dSYM.zip"
mkdir -p "$dir/data" "$dir/logs" "$dir/plugins" "$dir/build" "$dir/lib" "$dir/vulkan/icd.d" "$dir/licenses"
cp -r patches "$dir/patches"
cp include/bbhost_plugin.h "$dir/plugins/"
cp LICENSE "$dir/LICENSE"
cp bbhost.example.toml "$dir/bbhost.example.toml"
cp tools/macos/Apache-2.0.txt "$dir/licenses/Apache-2.0.txt"
cp tools/macos/cereal-LICENSE.txt "$dir/licenses/cereal-LICENSE.txt"
cp third_party/imgui/LICENSE.txt "$dir/licenses/imgui-LICENSE.txt"
cp third_party/libatrac9/LICENSE "$dir/licenses/LibAtrac9-LICENSE.txt"
cp third_party/monocypher/LICENCE.md "$dir/licenses/Monocypher-LICENCE.txt"
tools/package_plugins.sh "$(dirname "$bin")/plugins" "$dir/plugins"

# dsymutil before strip: it follows the debug map to the object files.
dsymutil "$bin" -o "$out/$name.dSYM"
strip -S -o "$dir/bbhost" "$bin"
chmod 755 "$dir/bbhost"

# x86_64 only: an x86_64 process can't load the arm64 half.
lipo -thin x86_64 "$sdk/lib/libvulkan.1.dylib" -output "$dir/lib/libvulkan.1.dylib" 2>/dev/null \
    || cp "$sdk/lib/libvulkan.1.dylib" "$dir/lib/libvulkan.1.dylib"
lipo -thin x86_64 "$sdk/lib/libMoltenVK.dylib" -output "$dir/lib/libMoltenVK.dylib" 2>/dev/null \
    || cp "$sdk/lib/libMoltenVK.dylib" "$dir/lib/libMoltenVK.dylib"
python3 - "$sdk/share/vulkan/icd.d/MoltenVK_icd.json" "$dir/vulkan/icd.d/MoltenVK_icd.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
m["ICD"]["library_path"] = "../../lib/libMoltenVK.dylib"  # from vulkan/icd.d
json.dump(m, open(sys.argv[2], "w"), indent=4)
EOF

for r in $(otool -l "$dir/bbhost" | awk '/LC_RPATH/ { getline; getline; print $2 }'); do
    install_name_tool -delete_rpath "$r" "$dir/bbhost" 2>/dev/null
done
install_name_tool -add_rpath @executable_path/lib "$dir/bbhost" 2>/dev/null
otool -L "$dir/bbhost" | grep -q '@rpath/libvulkan.1.dylib' || { echo "bbhost does not load the loader by @rpath"; exit 1; }
if otool -L "$dir/bbhost" | awk 'NR > 1 { print $1 }' | grep -vE '^(/usr/lib/|/System/|@rpath/libvulkan\.1\.dylib$)'; then
    echo "bbhost links a library the package does not carry (above)"; exit 1
fi

# A Mach-O built for a newer macOS than the minimum won't load on it.
min_macos=15.0
for f in "$dir/bbhost" "$dir/lib/"*.dylib "$dir/plugins/"*.so; do
    [ -f "$f" ] || continue
    v=$(otool -arch x86_64 -l "$f" | awk '/LC_BUILD_VERSION/ { b = 1 } b && $1 == "minos" { print $2; exit }
                                          /LC_VERSION_MIN_MACOSX/ { m = 1 } m && $1 == "version" { print $2; exit }')
    [ -n "$v" ] || { echo "$f: no minimum macOS version in it"; exit 1; }
    if [ "$(printf '%s\n%s\n' "$v" "$min_macos" | sort -V | tail -1)" != "$min_macos" ]; then
        echo "$f needs macOS $v, newer than the $min_macos the package supports (MACOSX_DEPLOYMENT_TARGET)"; exit 1
    fi
    echo "  minos $v  ${f#"$dir"/}"
done

for f in "$dir/lib/"*.dylib; do codesign --force --sign - "$f" 2>/dev/null; done
codesign --force --sign - --entitlements tools/macos/bbhost.entitlements "$dir/bbhost"
codesign --verify --strict "$dir/bbhost"

# The licenses of what the package carries that is not bbhost's.
# The SDK's VERSIONS.txt names the revision each was built from; the versions
# of SDL3 and ffmpeg are tools/build_macos.sh's.
rev_of() { grep -m1 "/$1.git" "$sdk/../VERSIONS.txt" 2>/dev/null | cut -d' ' -f1 || true; }
mvk=$(rev_of MoltenVK); ldr=$(rev_of Vulkan-Loader); spvc=$(rev_of SPIRV-Cross)
dep_ver() { sed -n "s/^$1=.*:-\(.*\)}\$/\1/p" tools/build_macos.sh; }
sdl=$(dep_ver SDL_TAG); ffv=$(dep_ver FFMPEG_VER)
[ -n "$sdl" ] && [ -n "$ffv" ] || { echo "no SDL_TAG or FFMPEG_VER in tools/build_macos.sh"; exit 1; }
cat > "$dir/licenses/README.txt" <<EOF
bbhost is under the GNU GPL, version 3 or later (../LICENSE). This package also
carries, from the LunarG Vulkan SDK $sdkver:

  lib/libMoltenVK.dylib    MoltenVK, Vulkan on Metal.
                           Copyright The Brenwill Workshop Ltd.
                           Apache License 2.0 (Apache-2.0.txt).
                           Source: https://github.com/KhronosGroup/MoltenVK${mvk:+ at $mvk}
                           It has built in:
                           - SPIRV-Cross, copyright Arm Limited and others,
                             Apache License 2.0 (Apache-2.0.txt).
                             Source: https://github.com/KhronosGroup/SPIRV-Cross${spvc:+ at $spvc}
                           - cereal, copyright Randolph Voorhies and Shane
                             Grant, BSD 3-Clause (cereal-LICENSE.txt).
                             Source: https://github.com/USCiLab/cereal
  lib/libvulkan.1.dylib    The Vulkan loader.
                           Copyright The Khronos Group Inc., Valve Corporation
                           and LunarG, Inc.
                           Apache License 2.0 (Apache-2.0.txt).
                           Source: https://github.com/KhronosGroup/Vulkan-Loader${ldr:+ at $ldr}

MoltenVK's, SPIRV-Cross's and the loader's LICENSE files are each the text
of Apache-2.0.txt.

bbhost itself has linked in:

  SDL3                     zlib license.
                           Source: https://github.com/libsdl-org/SDL, tag $sdl
  ffmpeg $ffv             Its movie decoders only (h264, hevc, aac, mp3),
                           built as static libraries in ffmpeg's LGPL
                           configuration: GNU LGPL, version 2.1 or later.
                           Source: https://ffmpeg.org/releases/ffmpeg-$ffv.tar.xz
                           (the configure options are in bbhost's
                           tools/build_macos.sh).
  LibAtrac9                MIT (LibAtrac9-LICENSE.txt).
  Dear ImGui               MIT (imgui-LICENSE.txt).
  Monocypher               BSD-2-Clause or CC0 (Monocypher-LICENCE.txt).

bbhost's source, with those of LibAtrac9, Dear ImGui and Monocypher, is at
https://github.com/droogie/bbhost.
EOF

cat > "$dir/run-bbhost.command" <<'EOF'
#!/bin/bash
# Starts bbhost and keeps its log in logs/ (the newest ten are kept). A
# .command file opens in Terminal when double-clicked in the Finder.
#   ./run-bbhost.command                     the per-user config
#                                            (~/Library/Application Support/bbhost/bbhost.toml)
#   BBHOST_CONFIG=other.toml ./run-bbhost.command   that file on top of it
cd "$(dirname "$0")" || exit 1
cfg=${BBHOST_CONFIG:-bbhost.toml}
mkdir -p logs
log=logs/bbhost-$(date +%Y%m%d-%H%M%S).log
ls -1t logs/bbhost-*.log 2>/dev/null | tail -n +10 | while IFS= read -r f; do rm -f "$f"; done
args=()
[ -f "$cfg" ] && args=(--config "$cfg")
# Keeps out a system-wide driver.
export VK_DRIVER_FILES="$PWD/vulkan/icd.d/MoltenVK_icd.json"
# On Apple Silicon bbhost needs Rosetta 2, which macOS no longer always has.
if [ "$(uname -m)" = arm64 ] && ! /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
    echo "bbhost: Rosetta 2 is not installed. Install it with"
    echo "  softwareupdate --install-rosetta --agree-to-license"
    exit 1
fi
echo "bbhost: config ${args[1]:-per-user}, log $(pwd)/$log"
# tee -i: Ctrl-C stops the game, not the log, so its last lines are kept.
./bbhost "${args[@]}" "$@" 2>&1 | tee -i "$log"
exit "${PIPESTATUS[0]}"
EOF
chmod 755 "$dir/run-bbhost.command"

cat > "$dir/README-macos.txt" <<EOF
bbhost for macOS, build $name
====================================

bbhost runs Bloodborne on PC: the game's own 1.09 executable, its calls into
the PS4's system libraries answered by bbhost and its graphics recompiled for
Vulkan. Free software under the GNU GPL, version 3 or later (LICENSE); the
source is at https://github.com/droogie/bbhost.

The macOS port is new and experimental: it builds and passes its tests, but
the game has not been played on a Mac yet. Expect it not to get far, and
please report what happens.

You need
  - macOS 15 or later. On Apple Silicon, Rosetta 2 runs bbhost (an x86_64
    program) and the game's code with it; macOS 15 is the first whose Rosetta
    runs the AVX instructions the game uses. If Rosetta is not installed:
        softwareupdate --install-rosetta --agree-to-license
    An Intel Mac runs it directly.
  - Nothing else to install: Vulkan runs on Metal through MoltenVK, which is
    in lib/ with the Vulkan loader.
  - The eboot.bin of the game's 1.09 update, decrypted to an ELF (SHA-256
    941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7; bbhost
    stops at start and says why when it is another version), and the game
    dump with the 1.09 update's files copied over it: the CUSA00900 folder
    that contains dvdroot_ps4 (or the update kept beside it in
    CUSA00900-UPDATE). Neither is in this package.

Setup
  Unzip this folder somewhere of its own, for example ~/Games. This package
  is signed ad hoc, not by an Apple Developer ID, and is not notarized, so
  macOS refuses to open it as downloaded. Clear the download's quarantine
  flag once, in Terminal:
      xattr -dr com.apple.quarantine ~/Games/$name
  Then double-click run-bbhost.command (or run ./run-bbhost.command in
  Terminal). The first time, the setup window opens: pick the game folder
  and the decrypted eboot with the Browse buttons (each turns green when it
  is right), then Play. ./run-bbhost.command --setup opens it again.

  The settings, the F10 settings and your account sign-in are kept in
  ~/Library/Application Support/bbhost, saves and caches in its data/
  folder, so a new download needs nothing set again.
  bbhost.example.toml lists every setting.

Online and plugins
  As on Linux and Windows: docs/online.md and docs/plugins.md in the
  repository. plugins/ holds the official plugins, signed by the bbhost
  release key (the .sig beside each).

Reporting a problem
  Send the log of that session from logs/ (privately: it contains your
  internet address), with what happened, your account name and your Mac's
  model and macOS version. If the game crashed, the log ends with a block
  starting "SIGSEGV pc=" (or SIGBUS) with the registers and the frames; the
  release's $name.dSYM.zip symbolizes it. A wrong picture: press F12 in the
  game while it shows, then zip the logs/f12-<date>-<time> folder the log
  names and send it with the log.

Licenses: LICENSE (bbhost), licenses/ (what bbhost is built with, and
MoltenVK and the Vulkan loader; licenses/README.txt says which is which).
EOF

# ditto keeps modes and signatures intact.
ditto -c -k --norsrc --keepParent "$dir" "$out/$name.zip"
ditto -c -k --norsrc --keepParent "$out/$name.dSYM" "$out/$name.dSYM.zip"
echo "package: $out/$name.zip ($(( $(stat -f %z "$out/$name.zip") / 1048576 )) MB); symbols: $out/$name.dSYM.zip"
