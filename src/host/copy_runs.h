#pragma once
// Copies gathered into runs for one vkCmdCopyBuffer (buffer_shadow.cpp's
// mirror pages): a copy grows the last run only where its source and its
// destination both follow on from that run's ends. From an import the one
// implies the other; from upload staging it does not - the copies follow one
// another in staging across the pages skipped between them - and a run grown
// on its source alone put a page's bytes on the page after the last one copied.
#include <cstdint>

// `Region` is VkBufferCopy, or anything with srcOffset, dstOffset and size.
// True: [src, src + bytes) -> [dst, dst + bytes) is now part of `last`.
template <typename Region>
bool copy_run_extends(Region& last, std::uint64_t src, std::uint64_t dst, std::uint64_t bytes) {
    if (last.srcOffset + last.size != src || last.dstOffset + last.size != dst) return false;
    last.size += bytes;
    return true;
}
