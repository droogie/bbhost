#!/bin/bash
# tools/pgo_store.sh PROFDATA SRCDIR: a PGO profile (llvm-profdata merge of a
# BBHOST_PGO=generate build's runs, docs/building.md) into the repository as
# pgo/bbhost.proftext - LLVM's text form, which any later llvm-profdata reads,
# with the tree it was built from (SRCDIR) written @SRC@: a file-local
# function's name in a profile carries its source's absolute path (about 40%
# of them), which differs on every machine that builds. tools/pgo_use.sh
# writes it back for the tree building.
set -eu
cd "$(dirname "$0")/.."
in=${1:?profdata}
src=${2:?the source tree the instrumented build was made from}
command -v cygpath > /dev/null && src=$(cygpath -m "$src")
src=${src%/}
mkdir -p pgo
llvm-profdata merge --text -o pgo/bbhost.proftext.tmp "$in"
sed "s|^${src}/|@SRC@/|" pgo/bbhost.proftext.tmp > pgo/bbhost.proftext
rm -f pgo/bbhost.proftext.tmp
echo "pgo/bbhost.proftext: $(grep -c '^# Func Hash' pgo/bbhost.proftext) functions, $(grep -c '^@SRC@/' pgo/bbhost.proftext) named by path, $(grep -c '^[A-Za-z]:/\|^/' pgo/bbhost.proftext || true) paths outside $src"
