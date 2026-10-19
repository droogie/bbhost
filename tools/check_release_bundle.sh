#!/bin/bash
# The release's bundles go to the live server by default: no playtest kit
# among the assets, no config in a bundle that names a server, the example
# config showing the live server's settings, launchers that pass no config,
# and the binaries' own first-start template naming the live server over
# https. The publish job runs it on what it is about to sign.
#   tools/check_release_bundle.sh DIR     (dist/release)
set -eu
d=${1:?usage: tools/check_release_bundle.sh DIR}
fail() { echo "release bundle: $*" >&2; exit 1; }
if ls "$d" | grep -i playtest; then fail "a playtest kit is among the assets"; fi
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
n=0
for z in "$d"/bbhost-win-*.zip; do [ -f "$z" ] || continue; mkdir -p "$work/w$n"; unzip -q "$z" -d "$work/w$n"; n=$((n + 1)); done
for t in "$d"/bbhost-linux-*.tar.gz "$d"/bbhost-steamdeck-*.tar.gz; do [ -f "$t" ] || continue; mkdir -p "$work/l$n"; tar xzf "$t" -C "$work/l$n"; n=$((n + 1)); done
for z in "$d"/bbhost-mac-*.zip; do
    case $z in *.dSYM.zip) continue ;; esac   # the debug information, not a bundle
    [ -f "$z" ] || continue; mkdir -p "$work/m$n"; unzip -q "$z" -d "$work/m$n"; n=$((n + 1))
done
[ "$n" -gt 0 ] || fail "no bundles in $d"
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
# What a first start writes (core/config.cpp's template), in each executable
# (macOS's only in its package).
for b in "$d"/bbhost "$d"/bbhost.exe "$work"/m*/bbhost-mac-*/bbhost; do
    [ -f "$b" ] || continue
    s=$(strings -a "$b")
    for want in 'host = "thehuntersdream.com"' 'scheme = "https"' 'verify_tls = true' 'require_account = true' \
                'auth_server = "https://thehuntersdream.com"'; do
        grep -qxF "$want" <<<"$s" || fail "$(basename "$b"): its first-start config lacks $want"
    done
done
echo "release bundles: $n checked, the live server over https by default, no playtest kit"
