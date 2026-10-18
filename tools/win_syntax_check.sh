#!/bin/bash
f="$1"
out=$(clang++ --target=x86_64-w64-mingw32 --sysroot=/usr/x86_64-w64-mingw32 -I/usr/x86_64-w64-mingw32/include/c++/16.1.0 -I/usr/x86_64-w64-mingw32/include/c++/16.1.0/x86_64-w64-mingw32 -std=c++20 -fsyntax-only -Wno-invalid-constexpr -idirafter /usr/include -ferror-limit=200 -DBBHOST_HAVE_CURL=1 -DBBHOST_HAVE_FFMPEG=1 -DBBHOST_HAVE_SDL3=1 -DBBHOST_HAVE_SPIRV_TOOLS=1 -DBBHOST_HAVE_SPIRV_OPT=1 -I"$(dirname "$0")/../src" -I"$(dirname "$0")/../third_party/libatrac9/C/src" -Isrc "$f" 2>&1)
n=$(echo "$out" | grep -c "error:")
echo "$n $f"
echo "$out" | grep "error:" | sed -E 's/^[^:]+:[0-9]+:[0-9]+: error: //' | sort | uniq -c | sort -rn | head -6 | sed 's/^/     /'
