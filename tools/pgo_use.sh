#!/bin/bash
# tools/pgo_use.sh OUT.profdata [SRCDIR]: the repository's PGO profile
# (pgo/bbhost.proftext, tools/pgo_store.sh) for this tree: @SRC@ written as
# SRCDIR (this checkout by default) and merged into OUT for -DBBHOST_PGO.
# llvm-profdata reads the text form whatever its version.
set -eu
cd "$(dirname "$0")/.."
out=${1:?output .profdata}
src=${2:-$PWD}
command -v cygpath > /dev/null && src=$(cygpath -m "$src")
src=${src%/}
[ -f pgo/bbhost.proftext ] || { echo "pgo_use: no pgo/bbhost.proftext"; exit 1; }
sed "s|^@SRC@/|${src}/|" pgo/bbhost.proftext > "$out.txt"
llvm-profdata merge -o "$out" "$out.txt"
rm -f "$out.txt"
echo "pgo_use: $out for $src"
