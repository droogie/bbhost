// GX on Vulkan, stage 1: the table of GX buffers the host keeps
// (host/gxvk.h), shared by the side that GX's threads call
// (gxvk_buffers.cpp) and the command processor's (gxvk_vk.cpp).
//
// An entry is made on the thread that made the buffer, and its handle - the
// tag, the entry's generation, its index - goes into the holder's +0x14. The
// command processor places it (a range of a device-local pool, the bytes
// copied in from the buffer's memory) at the first draw that binds it. Any
// thread may retire it: a write the host hears of, the holder's death, its
// memory's unmap. Only the command processor takes a retired entry back -
// its range freed once the work recorded with it has run, the entry free
// again under a new generation - so it never meets an entry reused under a
// draw it is recording, and a handle of an older generation no longer
// matches.
#pragma once

#include "host/gxvk.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace gxvk {

enum EntryState : std::uint32_t { kEntryFree = 0, kEntryMade = 1, kEntryPlaced = 2, kEntryRetired = 3 };

struct BufferEntry {
    // The generation (bits 8-15) and the state (bits 0-7). Published with
    // release once the fields above it are written; read with acquire.
    std::atomic<std::uint32_t> word{0};
    // Written by the thread that makes the entry, before its word.
    std::uint64_t holder = 0, memory = 0;
    std::uint32_t bytes = 0;
    std::uint8_t usage = 0, bind = 0;
    std::atomic<std::uint8_t> retired_why{0};
    std::int8_t watch_slot = -1;  // a default buffer's slot in the copy watch (gxvk_buffers.cpp), under the table's lock
    // The command processor's alone (under its lock): the placed copy.
    std::uint64_t vk_buffer = 0, vk_offset = 0;
    std::int32_t block = -1;
    std::uint64_t placed_serial = 0;  // the recording whose uploads made the copy
    std::uint64_t hash = 0;           // compare mode: the buffer's bytes when copied
    std::uint32_t binds = 0;          // compare mode: bindings so far, for the sampled checks
};

constexpr std::uint32_t kIndexBits = 20;
constexpr std::uint32_t kIndexMask = (1u << kIndexBits) - 1;
constexpr std::uint32_t kChunkBits = 12;
constexpr std::uint32_t kChunks = 1u << (kIndexBits - kChunkBits);

inline std::uint32_t entry_state(std::uint32_t word) { return word & 0xff; }
inline std::uint32_t entry_gen(std::uint32_t word) { return (word >> 8) & 0xff; }
inline std::uint32_t handle_of(std::uint32_t index, std::uint32_t gen) { return kHandleTag | ((gen & 0xff) << kIndexBits) | index; }
inline std::uint32_t handle_index(std::uint32_t handle) { return handle & kIndexMask; }
inline std::uint32_t handle_gen(std::uint32_t handle) { return (handle >> kIndexBits) & 0xff; }

// The entry at an index, null past the entries made so far. Any thread.
BufferEntry* entry_at(std::uint32_t index);
// The command processor's: retired entries' indices not yet taken back, at
// most `cap` (each index once), and an entry taken back (free again, a new
// generation, its index reusable).
std::size_t take_retired(std::uint32_t* out, std::size_t cap);
void recycle(std::uint32_t index);
// An entry retired by its buffer's end - the holder released, the memory
// unmapped or given to a new resource - rather than by a write: its copy
// still holds what every draw recorded before the end reads, and no later
// draw carries its handle (a new buffer in that holder or memory has a new
// one). The command processor binds it until it takes it back, some flips on.
inline bool retired_by_end(std::uint32_t why) { return why == kRetireGone || why == kRetireUnmap || why == kRetireReplaced; }
// Retires an entry from any thread (from made or placed; a no-op for one
// retired or free already, or of another generation): counted by `why`, and
// queued for the command processor to take back.
bool retire(std::uint32_t index, std::uint32_t gen, Retire why);

// Counters the two sides share, for the exit report.
enum Fallback : int {
    kFallNoObject,     // no GX object behind the binding (Scaleform, YEBIS, a table the game binds itself)
    kFallNoHandle,     // a GX buffer the host does not keep (dynamic, a constant buffer, written by the GPU, made before the switch)
    kFallStale,        // a handle that names no entry of that holder (a clobbered dword)
    kFallRecycled,     // its entry taken back since the call (its buffer ended long before the draw ran)
    kFallRetired,      // its entry retired by a write since (UpdateSubresource, a Map, a GPU copy), or ended before it was copied
    kFallRange,        // the range bound runs past the buffer's end
    kFallNoRoom,       // no device memory and no host memory for the copy
    kFallNoSource,     // the buffer's bytes could not be reached for the copy
    kFallIndirect,     // an indirect draw's indices (their count unknown at the binding)
    kFallMixed,        // a vertex binding the plan shares between this buffer and another's
    kFalls,
};
extern std::atomic<std::uint64_t> g_bound[kUses], g_fallback[kUses][kFalls];
extern std::atomic<std::uint64_t> g_bound_after_end[kUses];  // of g_bound: by an entry retired by its end
extern std::atomic<std::uint64_t> g_made[2], g_not_kept[9], g_retired[kRetires];

}  // namespace gxvk
