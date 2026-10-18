// Host command lists: where each packet the command processor acts on begins
// in the game's command buffers, recorded as it is written - the first step of
// retiring the walk of those buffers (docs/rendering.md, "The command
// processor"). A list belongs to a chunk of command buffer (what a Gnm command
// buffer object writes into: GX's arena chunks, YEBIS's, Gnmx's and the
// compute ring's buffers), found by the chunk's end - the object's +0x8,
// which stays put while +0x0 moves to each segment's start - and holds, in
// the order written, each such packet's place in the chunk and
// its opcode: our tokens, the labels, the waits, predication, the compute
// ring's calls, data written inline. Data the shaders read in place, markers
// and register packets nothing reads are not in it.
//
// BBHOST_HOST_LISTS: 1 (the default) is the handoff: a range goes to the
// command processor as the packets its list holds, instead of all its bytes
// (hand_over); a range no list covers is walked, counted by reason. 0 records
// nothing and walks every range. 2 (shadow) records the lists, takes each
// submitted range's part of its list on the submitting thread, and has the
// command processor compare it with what its walk of the range meets - packet
// for packet, in order - counting agreements and every disagreement by reason
// (the exit report). 3 records only (the recording's own cost).
//
// The recording side is leaf-safe (decomp leaves call it on the game's
// threads, with the game's FS): no locks, no allocation, no host
// thread-locals. One thread writes a chunk at a time - the command buffer
// that holds it - and a range is taken from its list when it is submitted,
// after it was written and before the chunk can be written again.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace host_lists {

// BBHOST_HOST_LISTS, read before any of the game's code runs.
extern const int g_mode;

void note_slow(std::uint64_t cb, std::uint64_t va, std::uint32_t op);

// A packet with opcode `op` written at `va` through the Gnm command buffer
// object `cb` ({+0x0 the chunk's start, +0x8 its end, +0x10 the write
// pointer}). Leaf-safe.
inline void note(std::uint64_t cb, std::uint64_t va, std::uint32_t op) {
    if (g_mode) note_slow(cb, va, op);
}

// Hosted: a GX command arena the arena's acquire met - `chunks` chunks of
// `chunk_bytes` from `base` (hle_gx_arena_acquire, note_arena_extent).
void note_arena(std::uint64_t base, std::uint64_t chunk_bytes, std::uint64_t chunks);

// Hosted: GX's command arena handed out the chunk [chunk_va, + chunk_bytes)
// (hle_gx_arena_acquire); what its list held is gone.
void chunk_handed_out(std::uint64_t chunk_va, std::size_t chunk_bytes);

// A submitted range's part of its list: each entry the packet's place from
// the range's start in dwords << 8 | its opcode.
struct Taken {
    enum State : std::uint8_t { kNoList, kTaken, kOverflowed };
    State state = kNoList;
    std::vector<std::uint32_t> entries;
};
// Hosted, on the submitting thread: [va, va + bytes)'s part of its list.
void take(std::uint64_t va, std::size_t bytes, Taken* out);

// The handoff (BBHOST_HOST_LISTS=1), on the submitting thread: the range as
// the packets its list holds, each copied whole from its header, in order,
// into `out` - what the command processor walks in place of the range. False,
// counted by reason, when the range is to be walked as before: no list, a
// list that filled up, a packet past the range's end, or (the proof,
// BBHOST_HOST_LISTS_PROVE=1, a check: every header of the range read and
// matched against the list) a packet the list does not account for.
bool hand_over(std::uint64_t va, std::size_t bytes, const Taken& taken, std::vector<std::uint8_t>* out);

// Hosted, on a command-processor thread (shadow): a top-level range's walk
// begins - `walk_va` where the walker reads it (its copy), `guest_va` where
// the game wrote it (for the logs), `taken` what the list holds for it - and
// ends; an indirect buffer's walk inside it begins and ends (taken from the
// callee's list there and then); each packet the walk meets in between, at
// `va` where the walker reads it.
void shadow_begin(const Taken* taken, std::uint64_t walk_va, std::size_t bytes, std::uint64_t guest_va = 0);
void shadow_push(std::uint64_t va, std::size_t bytes);
void shadow_packet(std::uint64_t va, std::uint32_t op, const std::uint32_t* body, std::size_t n);
void shadow_pop();
void shadow_end();

void report();

}  // namespace host_lists
