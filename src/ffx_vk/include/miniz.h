/* The two names of miniz that the FidelityFX SDK's ffx_blob_decompress.h uses, through zlib, whose uncompress() takes
 * the same zlib stream: the SDK's build links miniz, which is not in its repository. A blob stored deflated
 * (tools/ffx_permutations.py --compress) is inflated by this; one stored as it is is never handed here.
 * Copyright (c) 2026 the bbhost authors. MIT licence: see LICENSE in src/ffx_vk. */
#pragma once
#include <zlib.h>

typedef unsigned char mz_uint8;
typedef unsigned long mz_ulong;
#define MZ_OK Z_OK

static inline int mz_uncompress(unsigned char* dest, mz_ulong* dest_len, const unsigned char* source, mz_ulong source_len) {
    uLongf n = *dest_len;
    const int r = uncompress(dest, &n, source, source_len);
    *dest_len = n;
    return r;
}
