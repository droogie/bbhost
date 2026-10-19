// GX's command lists as source (decomp/gx/command_list.cpp): the objects a
// deferred context hands the immediate one, and the immediate context hands
// the GPU. Each is the game's own memory, laid out as the game's code reads
// it; these name its fields.
#pragma once

#include <cstddef>
#include <cstdint>

namespace decomp_gx {

// A span: one segment of a command buffer as its header stood when it was
// cut - where it begins, where its chunk ends, where it was written to, the
// refill and its argument, and the rest of the eight qwords as they were.
// GPU work is [begin, write).
struct GxSpan {
    std::uint64_t begin;       // +0x00
    std::uint64_t end;         // +0x08
    std::uint64_t write;       // +0x10
    std::uint64_t refill;      // +0x18: called for more room
    std::uint64_t refill_arg;  // +0x20
    std::uint64_t rest[3];     // +0x28: as the buffer had them
};
static_assert(sizeof(GxSpan) == 0x40, "a span is a command buffer's header");

// A queue of spans: pages of `per_page` spans from an allocator (its +0x58
// allocates, +0x70 frees), the slot after a page's last span linking the
// next page, the last page holding `in_last`. A deferred context's command
// buffer keeps one (+0xb498 of the buffer) as it records; a command list
// takes it over whole.
struct GxSpanQueue {
    std::uint64_t allocator;  // +0x00
    std::uint64_t first;      // +0x08: the first page
    std::uint64_t last;       // +0x10: the page being filled
    std::uint32_t per_page;   // +0x18
    std::uint32_t in_last;    // +0x1c
};
static_assert(sizeof(GxSpanQueue) == 0x20, "the queue's layout");

// ID3D11CommandList as GX has it (the binary names the interface): the
// spans a deferred context recorded, and the output merger's records of the
// targets around them - as the list began (the colour targets and the depth
// target of the first record set) and as it ended (the eight colour targets
// and the depth target bound last). FinishCommandList fills it; the
// immediate context's ExecuteCommandList(s) puts its spans in its own
// stream, takes the targets it ended with as bound, and empties it.
struct GxCommandList {
    GxSpanQueue spans;             // +0x00
    std::uint64_t first_records[9];  // +0x20: colour 0..7, depth
    std::uint64_t last_colour[8];    // +0x68
    std::uint64_t last_depth;        // +0xa8
};
static_assert(sizeof(GxCommandList) == 0xb0, "the list's layout");
static_assert(offsetof(GxCommandList, first_records) == 0x20 && offsetof(GxCommandList, last_colour) == 0x68 &&
                  offsetof(GxCommandList, last_depth) == 0xa8,
              "the list's fields where the game reads them");

}  // namespace decomp_gx
