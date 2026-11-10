#!/bin/bash
# The release's bundles go to the live server by default: no playtest kit
# among the assets, no config in a bundle that names a server, the example
# config showing the live server's settings, launchers that pass no config,
# and the binaries' own first-start template naming the live server over
# https. Each bundle carries the pinned DXC library beside its executable
# with DXC's licences, and no DXIL validator (tools/fetch_dxc.sh), and
# bbhost's licence's additional permission (LICENSE-EXCEPTION). The publish
# job runs it on what it is about to sign.
#   tools/check_release_bundle.sh DIR     (dist/release)
set -eu
d=${1:?usage: tools/check_release_bundle.sh DIR}
fail() { echo "release bundle: $*" >&2; exit 1; }
if ls "$d" | grep -i playtest; then fail "a playtest kit is among the assets"; fi
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
n=0
for z in "$d"/bbhost-win-*.zip; do [ -f "$z" ] || continue; mkdir -p "$work/w$n"; unzip -q "$z" -d "$work/w$n"; n=$((n + 1)); done
for t in "$d"/bbhost-linux-*.tar.gz "$d"/bbhost-steamdeck-*.tar.gz; do [ -f "$t" ] || continue; mkdir -p "$work/l$n"; tar xzf "$t" -C "$work/l$n"; n=$((n + 1)); done
[ "$n" -gt 0 ] || fail "no bundles in $d"
# Every bundle carries the pinned pipeline list (tools/fetch_pipeline_list.sh),
# once one is published (an empty pin: none yet, and none in any bundle).
list_sha=$("$(dirname "$0")/fetch_pipeline_list.sh" --sha256)
for b in "$work"/*/; do
    f=$(find "$b" -name pipeline-list.bin | head -1)
    if [ -z "$list_sha" ]; then
        [ -z "$f" ] || fail "${f#"$work"/}: a pipeline list, but none is pinned (tools/fetch_pipeline_list.sh)"
        continue
    fi
    [ -n "$f" ] || fail "${b#"$work"/}: no pipeline-list.bin (tools/fetch_pipeline_list.sh)"
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$list_sha" ] || fail "${f#"$work"/}: not the pinned pipeline list (tools/fetch_pipeline_list.sh)"
done
# Only the example config may name a server (it shows the defaults).
if find "$work" -name '*.toml' ! -name bbhost.example.toml -exec grep -lE '^[[:space:]]*(host|np_server|auth_server)[[:space:]]*=' {} + | grep .; then
    fail "a bundle carries a config that names a server"
fi
while IFS= read -r f; do
    for want in 'host = "thehuntersdream.com"' 'scheme = "https"' 'verify_tls = true'; do
        tr -d '\r' < "$f" | grep -qxF "$want" || fail "${f#"$work"/}: lacks $want"
    done
done < <(find "$work" -name bbhost.example.toml)
while IFS= read -r f; do
    if tr -d '\r' < "$f" | grep -E '^[^r#].*--config|^bbhost\.exe .*--config'; then fail "${f#"$work"/} passes a config"; fi
done < <(find "$work" -name 'run-bbhost.bat')
# DXC: the pinned library beside each executable, its licences, never dxil.
fetch_dxc=$(dirname "$0")/fetch_dxc.sh
win_dxc=$("$fetch_dxc" --sha256 windows); lin_dxc=$("$fetch_dxc" --sha256 linux)
while IFS= read -r exe; do
    b=$(dirname "$exe")
    case "$exe" in *.exe) lib=$b/dxcompiler.dll want=$win_dxc ;; *) lib=$b/libdxcompiler.so want=$lin_dxc ;; esac
    [ -f "$lib" ] || fail "${b#"$work"/}: no $(basename "$lib") beside the executable"
    [ "$(sha256sum "$lib" | cut -d' ' -f1)" = "$want" ] || fail "${lib#"$work"/}: not the pinned DXC (tools/fetch_dxc.sh)"
    for l in LICENSE.TXT LICENSE-Apache-2.0-LLVM.txt ThirdPartyNotices.txt LICENSE-SPIRV-Tools.txt; do
        [ -f "$b/licenses/dxc/$l" ] || fail "${b#"$work"/}: no licenses/dxc/$l"
    done
done < <(find "$work" -mindepth 3 -maxdepth 3 -type f \( -name bbhost.exe -o -name bbhost \))
if find "$work" -iname 'dxil.dll' -o -iname 'libdxil.so*' | grep .; then fail "a bundle carries DXC's DXIL validator"; fi
# bbhost's licence's additional permission (LICENSE-EXCEPTION: combining bbhost
# with GPU vendors' upscaling and frame generation runtimes) beside each
# executable, as the repository has it: LICENSE-EXCEPTION.txt on Windows.
exception=$(dirname "$0")/../LICENSE-EXCEPTION
while IFS= read -r exe; do
    b=$(dirname "$exe")
    case "$exe" in *.exe) f=$b/LICENSE-EXCEPTION.txt ;; *) f=$b/LICENSE-EXCEPTION ;; esac
    [ -f "$f" ] || fail "${b#"$work"/}: no $(basename "$f") beside the executable"
    cmp -s "$f" "$exception" || fail "${f#"$work"/}: not the repository's LICENSE-EXCEPTION"
done < <(find "$work" -mindepth 3 -maxdepth 3 -type f \( -name bbhost.exe -o -name bbhost \))
# The upscalers' runtime beside each Windows executable, the pinned files with
# their licences (tools/fetch_upscalers.sh): without them a release has no FSR,
# DLSS or frame generation.
fetch_up=$(dirname "$0")/fetch_upscalers.sh
while IFS= read -r exe; do
    b=$(dirname "$exe")
    for f in $("$fetch_up" --files); do
        [ -f "$b/$f" ] || fail "${b#"$work"/}: no $f beside bbhost.exe (tools/fetch_upscalers.sh)"
        [ "$(sha256sum "$b/$f" | cut -d' ' -f1)" = "$("$fetch_up" --sha256 "$f")" ] || fail "${b#"$work"/}/$f: not the pinned file"
    done
    for l in fidelityfx/LICENSE.md dlss/LICENSE.txt upscaler_bridge/LICENSE.txt; do
        [ -f "$b/licenses/$l" ] || fail "${b#"$work"/}: no licenses/$l"
    done
done < <(find "$work" -mindepth 3 -maxdepth 3 -type f -name bbhost.exe)
# Each Windows executable runs in UTF-8 (res/bbhost.manifest): without it a
# user or folder name outside ASCII stops bbhost at its start.
while IFS= read -r exe; do
    grep -aqF '<activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>' "$exe" \
        || fail "${exe#"$work"/}: no UTF-8 code page in its manifest (res/bbhost.manifest)"
done < <(find "$work" -mindepth 3 -maxdepth 3 -type f -name bbhost.exe)
# DLSS on Linux beside each Linux executable: bbhost's NGX bridge and NVIDIA's
# pinned runtime, with their licences (tools/fetch_ngx.sh; without them the
# release has no DLSS on Linux) - and none of it in a Steam Deck kit, whose GPU
# is AMD's. NVIDIA's NGX code is in the bridge alone, never in bbhost.
fetch_ngx=$(dirname "$0")/fetch_ngx.sh
while IFS= read -r exe; do
    b=$(dirname "$exe")
    if grep -aq 'NVSDK_NGX_VULKAN_Init' "$exe"; then fail "${exe#"$work"/}: NVIDIA's NGX code in bbhost itself"; fi
    case "$b" in
        */bbhost-steamdeck-*)
            if ls "$b" | grep -E '^libbbhost_ngx_bridge\.so$|^libnvidia-ngx-dlss'; then fail "${b#"$work"/}: DLSS's files in a Steam Deck kit"; fi
            for l in dlss ngx_bridge upscaler_bridge; do
                [ ! -e "$b/licenses/$l" ] || fail "${b#"$work"/}: licenses/$l in a Steam Deck kit"
            done
            ;;
        *)
            [ -f "$b/libbbhost_ngx_bridge.so" ] || fail "${b#"$work"/}: no libbbhost_ngx_bridge.so beside bbhost (the NGX bridge: BBHOST_NGX, tools/fetch_ngx.sh)"
            for f in $("$fetch_ngx" --files); do
                [ -f "$b/$f" ] || fail "${b#"$work"/}: no $f beside bbhost (tools/fetch_ngx.sh)"
                [ "$(sha256sum "$b/$f" | cut -d' ' -f1)" = "$("$fetch_ngx" --sha256 "$f")" ] || fail "${b#"$work"/}/$f: not the pinned file"
            done
            for l in dlss/LICENSE.txt ngx_bridge/LICENSE upscaler_bridge/LICENSE.txt; do
                [ -f "$b/licenses/$l" ] || fail "${b#"$work"/}: no licenses/$l"
            done
            ;;
    esac
done < <(find "$work" -mindepth 3 -maxdepth 3 -type f -name bbhost)
# What a first start writes (core/config.cpp's template), in each executable.
for b in "$d"/bbhost "$d"/bbhost.exe; do
    [ -f "$b" ] || continue
    s=$(strings -a "$b")
    for want in 'host = "thehuntersdream.com"' 'scheme = "https"' 'verify_tls = true' 'require_account = true' \
                'auth_server = "https://thehuntersdream.com"'; do
        grep -qxF "$want" <<<"$s" || fail "$(basename "$b"): its first-start config lacks $want"
    done
done
if [ -n "$list_sha" ]; then listmsg="and the pinned pipeline list (${list_sha:0:12}) beside each executable"; else listmsg="beside each executable (no pipeline list pinned yet)"; fi
echo "release bundles: $n checked, the live server over https by default, no playtest kit, DXC $(cat "$(dirname "$0")/../third_party/dxc/VERSION") $listmsg; the Windows ones in UTF-8 with FidelityFX, DLSS and the upscaler bridge; the Linux ones with DLSS and the NGX bridge, the Steam Deck's without"
