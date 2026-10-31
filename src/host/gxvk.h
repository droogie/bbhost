// GX on Vulkan, stage 1: the engine's GX buffers as Vulkan buffers.
//
// GX, the engine's Direct3D 11-shaped graphics layer, keeps its buffers in
// guest memory, and a draw reaches a buffer's bytes through the address its
// records carry: the copy versions, the buffer shadow's mirrors and
// snapshots, the constant-buffer snapshots, the import (render.cpp
// plan_vertex_input, resolve_stage_buffers). Stage 1 makes a GX buffer a
// Vulkan buffer when GX creates it, and the draw carries that buffer's handle
// instead of an address to resolve.
//
// Its first part is the census (BBHOST_GXVK_CENSUS=1, or BBHOST_GXVK_BUFFERS=
// census), which changes nothing the game or the renderer does and measures
// what the stage rests on:
// - GX's buffers by usage, bind flags, CPU access and initial data, the Maps
//   and updates each kind takes, and how they die;
// - every vertex record, index buffer and constant buffer a draw binds, by
//   what holds the bytes: an immutable GX buffer, a dynamic one's own memory,
//   a renamed copy in GX's upload heap, a GPU-written one, or no GX object;
// - GX's upload heap (the one region the device gives its block pool at
//   creation, 0x256b0f0 -> 0x2aa9b50): its size, its peak use, and whether
//   every range in it is handed out again only after the GPU finished with
//   the work that bound it (the release proof, below);
// - the spare bytes of GX's objects, which their creators never write: a
//   canary written there at creation must be found intact wherever the
//   object is used and when it dies.
//
// The release proof has two halves. (i) When the heap hands a range out, no
// submission that bound any of it may still be running (the command
// processor stamps each 4 KiB page with the submission it binds the page in;
// the hand-out compares those stamps with the completed submissions). (ii)
// When the command processor binds a range, the range must not have been
// handed out again since the GX call that recorded the binding (each page's
// hand-out sequence against the draw's stamp at its call). Both counts must
// be 0 for a renamed copy to be bound in place without a snapshot.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gxvk {

// BBHOST_GXVK_BUFFERS: unset, GX buffers bound by their handles (1) on a
// discrete GPU with room in its video memory, and on an integrated GPU with
// 8 GiB of it or more (a Strix Halo's carve-out): read from the carve-out
// rather than from the imports, the Radeon 8060S's frame took ~7.5% less GPU
// time (164.9 against 157.9 fps at 1080p, the GPU the limit, the run commit on;
// docs/list-resolve.md) for 256 MiB of copies. Today's path (0) on a smaller
// integrated GPU or with tight memory, where the draws read GX's buffers
// where they are (the buffer shadow in place there, or no shadow) and a
// buffer's copy would hold memory the system needs, and with small video
// memory, where the copies compete with the targets and textures for it and
// one dip to the reserve costs far more GPU time than the command processor
// saves. 0 today's path; 1 by handle on any GPU; 2 both, compared; "census"
// the census alone (as BBHOST_GXVK_CENSUS=1, which can join the others).
// Read once, as the GX hooks are installed: the device's kind is asked before
// the device is made (host_gpu_integrated, host_gpu_memory_tight,
// host_gpu_small_vram).
enum class Buffers : int { kOff = 0, kOn = 1, kCompare = 2 };
Buffers buffers_mode();
bool census();
// The device as the default sees it.
enum class BuffersDevice : int { kRoomy, kIntegrated, kTight, kSmallVram, kIntegratedRoomy };
// The mode for the switch's value (null or empty: unset) on that device.
inline Buffers buffers_mode_for(const char* env, BuffersDevice device) {
    if (!env || !*env) return device == BuffersDevice::kRoomy || device == BuffersDevice::kIntegratedRoomy ? Buffers::kOn : Buffers::kOff;
    return env[0] == '1' ? Buffers::kOn : env[0] == '2' ? Buffers::kCompare : Buffers::kOff;
}

// A GX buffer holder's spare dwords: +0x14, the reference count's upper half
// (every locked access GX makes to the count is to the dword at +0x10), and
// +0x44, which the buffer creator (0x2565c10) copies with the resource struct
// from a stack slot it never initialises. A texture holder has the first
// only (its resource struct is written whole); a shader-resource view has
// +0x48..+0x4f, an unordered-access view +0x44..+0x4f (their record builders,
// 0x257a5d0 and 0x257a8b0, write the T# and the description and nothing
// after them).
constexpr std::uint64_t kHolderSpareA = 0x14, kHolderSpareB = 0x44;
constexpr std::uint64_t kSrvSpare = 0x48, kUavSpare = 0x44;

// --- The census's hooks. Each is a no-op unless census() is on. ---

// The registry's (engine/gx_resources.cpp), on the thread that called GX.
void census_buffer_made(std::uint64_t holder, bool initial_data);
void census_buffer_mapped(std::uint64_t holder, std::uint32_t type, std::uint64_t mapped, std::uint32_t bytes, bool ok, bool own);
void census_buffer_updated(std::uint64_t holder);
void census_buffer_gone(std::uint64_t holder);
void census_texture_made(std::uint64_t holder);
void census_texture_gone(std::uint64_t holder);
// kind: 3 shader-resource view, 5 unordered-access view.
void census_view_made(std::uint64_t view, int kind);
// The block pool's region (0x2aa9b50's base and size).
void census_heap(std::uint64_t pool, std::uint64_t base, std::uint64_t bytes);
// A hand-out by the block pool (0x2aa9d50 / 0x2aa9fa0 returned `node`; the
// range is [node+8, +bytes)).
void census_handout(std::uint64_t pool, std::uint64_t node, std::uint64_t bytes);
// The ring allocator's hand-outs (0x2adb1b0).
void census_ring(std::uint64_t va, std::uint64_t bytes);

// The GX threads' (hle/gx_trace.cpp), at a draw's call.
// What a draw's call is stamped with for half (ii): the hand-outs so far.
std::uint64_t handout_seq();
// A constant buffer the call's records name: its holder and the address the
// records carry (its own memory, or the context's renamed copy).
void census_constant(std::uint64_t holder, std::uint64_t va, std::uint32_t bytes);
// A view a draw or dispatch binds (kind 3 shader-resource, 5 unordered-access):
// its spare bytes; a shader-resource view's texture holder's; an
// unordered-access view's buffer, as a binding of kUseUav.
void census_view_bound(std::uint64_t view, int kind);

// The command processor's (host/render.cpp), as it binds.
enum Use : int { kUseVertex, kUseIndex, kUseConstant, kUseUav, kUses };
// A vertex record or index buffer: the holder the GX call found it in (0:
// none - Scaleform, YEBIS, a table the game binds itself), the bytes bound,
// the draw's call stamp (0: none), the submission being recorded.
void census_binding(Use use, std::uint64_t holder, std::uint64_t va, std::uint64_t bytes, std::uint64_t call_seq,
                    std::uint64_t serial);
// A constant buffer as the command processor binds it (the GX side has
// classed it): only the heap's two checks, and the bindings of draws without
// GX records.
void census_constant_bound(std::uint64_t va, std::uint64_t bytes, std::uint64_t call_seq, std::uint64_t serial, bool gx_records);

// The exit report (gx_resources_report calls it).
void census_report();

// --- GX buffers by handle (BBHOST_GXVK_BUFFERS: the default, except on an integrated GPU or with
// tight memory; 2 compares) ---
//
// A GX buffer the host can keep - immutable, or default with initial data;
// no CPU access; bound as vertex or index data, and as nothing the GPU
// writes - gets an entry in the host's table when GX makes it, and the
// entry's handle in the holder's +0x14 (kHolderSpareA). The GX thread copies
// the handle into the draw's record beside the holder, and the command
// processor binds the entry's device-local copy - made from the buffer's
// memory in the uploads of the first recording that binds it - instead of
// resolving the record's address. A write the host hears of after the
// creation (UpdateSubresource, a Map, a GPU copy into its memory), the
// holder's death or its memory's unmap retires the entry: its bindings take
// today's path again. With 2 every binding takes today's path, and the
// handle's is checked against it (gxvk_vk.cpp).
bool buffers_on();
// Whether GX's writes of buffers are watched for kept ones - its
// UpdateSubresource hooked, its GPU copies' destinations looked up: with 2,
// as a check (a kept buffer is immutable, which nothing may write). With 1
// nothing on GX's threads looks: the hook alone, on every constant buffer's
// update, cost the GX workers about a core.
bool watch_writes();
// The handle in a holder's +0x14: the tag in the top four bits, the entry's
// generation, its index (host/gxvk_table.h). Any other value is no handle.
constexpr std::uint32_t kHandleTag = 0xa0000000u, kHandleTagMask = 0xf0000000u;
inline bool is_handle(std::uint32_t word) { return (word & kHandleTagMask) == kHandleTag; }
// Why an entry was retired (the exit report counts them).
enum Retire : int {
    kRetireUpdate,    // UpdateSubresource of it
    kRetireMap,       // a Map of it
    kRetireCopy,      // a GPU copy into its memory
    kRetireGone,      // its holder's release
    kRetireUnmap,     // its memory unmapped
    kRetireReplaced,  // its holder or memory given to a new buffer before the host heard of its end
    kRetireCompare,   // a compare run's check disagreed
    kRetireNoRoom,    // no memory for its copy
    kRetireNoSource,  // its bytes could not be reached for the copy
    kRetires,
};
// The registry's (engine/gx_resources.cpp), on the thread that made or
// changed the buffer: GX made one (its holder, memory, size and description
// as the holder holds them); its holder was released; something wrote it;
// a GPU copy wrote [va, va + bytes); the kernel unmapped [va, va + bytes).
void buffer_made(std::uint64_t holder, std::uint64_t memory, std::uint32_t bytes, std::uint8_t usage, std::uint8_t cpu, std::uint8_t bind,
                 std::uint32_t misc, bool initial_data);
void buffer_gone(std::uint64_t holder);
// The registry found the buffer at `holder` ended unseen (its memory given to
// a new resource): its entry retired, whatever the holder's bytes are now.
void buffer_replaced(std::uint64_t holder);
void buffer_written(std::uint64_t holder, Retire why);
void memory_written(std::uint64_t va, std::uint64_t bytes, Retire why);
// A GPU copy's destination with the handles on (=1): a kept default buffer it
// writes is retired - asked of the copy watch without a lock
// (host/gxvk_watch.h): a copy far from every kept default buffer reads a
// counter or two, one next to them searches their ranges in halves.
void copy_written(std::uint64_t va, std::uint64_t bytes);
// BBHOST_GXVK_BUFFERS=2: the copy watch's granules checked against its slots
// for a copy's destination (the exit report counts a miss, which may not be).
void copy_watch_check(std::uint64_t va, std::uint64_t bytes);
void memory_released(std::uint64_t va, std::uint64_t bytes);
// The exit report's lines for the table (gx_resources_report calls it; the
// command processor's side reports with the GPU's).
void buffers_report();

// --- Stage 1c: GX's upload heap bound in place (host/gxvk_arena.cpp) ---
//
// The heap - the one region GX's device gives its block pool, where every
// renamed constant buffer, Map rename and ring allocation lies - is handed
// out again only after the GPU has finished the work that used it, so a
// range of it the command processor binds can be bound where it is: no
// snapshot, no lookup. The release order is proven for every binding and
// the bytes for a sample of them, every run; a violation sends every
// binding of the heap back to today's path.
// BBHOST_GXVK_ARENA: 0 (the default) today's path; 1 in place, where the
// heap is; compare: today's path, with every binding of the heap checked as
// 1 would bind it. (2, the heap moved into video memory the CPU writes, was
// measured slower and is gone: docs/rendering.md.)
enum class Arena : int { kOff = 0, kInPlace = 1, kCompare = 2 };
Arena arena_mode();
inline bool arena_wanted() { return arena_mode() != Arena::kOff; }
// The pool's init (decomp/gx/upload_heap.cpp's entry): the pool and the
// region it is made over, which stays the game's.
void arena_region(std::uint64_t pool, std::uint64_t region, std::uint64_t bytes);
// The heap's bounds, once its pool is made.
bool arena_heap(std::uint64_t* lo, std::uint64_t* hi);
// A hand-out by the block pool (the registry's pool hooks, as they return):
// the pool and the node it returned; `top`: from the pool's top end (the
// immediate context's).
void arena_handout(std::uint64_t pool, std::uint64_t node, bool top = false);
// Whether the arena still records hand-outs: it is on and its proof has held.
// Once the proof fails it records nothing more, and the pool hooks need not
// stop for it.
bool arena_recording();
// A hand-out the pool hooks let pass because the arena no longer records (counted for its report).
void arena_unrecorded();
// The command processor, as it binds [va, +bytes) for `use` in submission
// `serial`: true when the range lies in one of the heap's hand-outs and is
// to be bound in place (its submission is stamped on the hand-out); false
// for today's path (counted by reason when the range lies in the heap).
// `stored`: the shader stores to it.
bool arena_bind(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial, Use use, bool stored);
// The binding found no one buffer over the heap to bind it in.
void arena_no_backing();
// Today's path over the heap, counted where it runs: the work stage 1c takes
// away, and what the command processor reads of the heap on the CPU (the
// descriptor tables in it).
enum ArenaNote : int { kArenaSnapshotCb, kArenaSnapshotVertex, kArenaGpuPending, kArenaTableRead, kArenaNotes };
void arena_note(ArenaNote what, std::uint64_t bytes);
// The exit report (gx_resources_report calls it).
void arena_report();
// --- Stage 1d.1: the heap's constant buffers snapshotted with no lookups (host/gxvk_heapsnap.cpp) ---
//
// A constant buffer the game writes on the CPU is copied as its draw records
// (render.cpp resolve_stage_buffers, BBHOST_CB_SNAPSHOT), and before the copy
// the command processor asks four things of its range: whether a copy-back is
// still in a copy version over it, whether a device-local twin holds it,
// whether a GPU write is still to land on it, and whether it is mapped. For a
// range in GX's upload heap the answers are known. The heap holds the renamed
// copies the game writes on the CPU and nothing else: a copy-back's source,
// never its destination; no GX buffer's own memory, so no twin; nothing the
// GPU writes; mapped for the session. So such a binding takes the snapshot
// with those answers instead of the lookups - the same copy of the same bytes
// into the same ring as today's path. What could make an answer wrong is
// watched where it would happen, not at each binding: a GPU write noted over
// the heap, a copy version made into it, a twin made over it, its memory
// unmapped (heap_written). The first such event is said once and counted, and
// from it on every binding of the heap asks again - in the command processor's
// order, so before any binding that could see what it did.
// BBHOST_GXVK_HEAP_SNAP: unset or 1 (the default) the answers known; 0
// today's path; compare: today's path, with each binding's lookups checked
// against the known answers (counted by kind, the first few said).
enum class HeapSnap : int { kOff = 0, kOn = 1, kCompare = 2 };
HeapSnap heap_snap_mode();
inline bool heap_snap_wanted() { return heap_snap_mode() != HeapSnap::kOff; }
// A constant-buffer binding as the command processor has it, for the choice.
struct HeapSnapCase {
    std::uint64_t lo = 0, hi = 0;       // the heap's bounds (lo 0: not known yet)
    std::uint64_t base = 0, bytes = 0;  // the binding's range
    bool stored = false;                // the shader stores to it
    bool windows = false;               // its draw has CPU-written windows (YEBIS)
    bool snapshots = true;              // BBHOST_CB_SNAPSHOT is on
    bool check = false;                 // a check that needs the lookups is on (the CB proof, the glitch watch)
    bool ring = true;                   // the constant-buffer copies' ring is made
    bool tripped = false;               // a guard has tripped this run
};
// The choice: kHeapSnapFast (the snapshot, the answers known), kHeapSnapNotHeap
// (not the heap's: nothing to count), or why the binding takes the lookups.
enum HeapSnapWhy : int {
    kHeapSnapStored, kHeapSnapSize, kHeapSnapPastEnd, kHeapSnapWindows, kHeapSnapOff, kHeapSnapCheck, kHeapSnapNoRing, kHeapSnapTripped,
    kHeapSnapWhys,
    kHeapSnapFast = -1, kHeapSnapNotHeap = -2
};
int heap_snap_class(const HeapSnapCase& c);
// At a constant buffer's binding (resolve_stage_buffers, the renderer's lock
// held), with BBHOST_GXVK_HEAP_SNAP set: the choice for [base, +bytes), counted.
// True: the binding takes the snapshot with the answers known (=1), or has its
// lookups checked against them (compare).
bool heap_snap_binding(std::uint64_t base, std::uint64_t bytes, bool stored, bool windows, bool snapshots, bool check, bool ring);
// The writers': what would make an answer wrong for the heap's bindings. A
// range that meets the heap trips the guard; anything else returns at once.
enum HeapGuard : int { kHeapGuardGpuWrite, kHeapGuardCopyVersion, kHeapGuardTwin, kHeapGuardUnmap, kHeapGuards };
void heap_written(HeapGuard what, std::uint64_t va, std::uint64_t bytes);
// Compare: a lookup's answer, for a binding the choice took, that is not the known one.
enum HeapSnapDiff : int { kHeapDiffCopyVersion, kHeapDiffTwin, kHeapDiffGpuWrite, kHeapDiffUnmapped, kHeapDiffs };
void heap_snap_disagree(HeapSnapDiff what, std::uint64_t va, std::uint64_t bytes);
// The heap's bounds are known (the pool's init, through arena_region): its first line.
void heap_snap_region(std::uint64_t lo, std::uint64_t hi);
// The exit report (gx_resources_report calls it).
void heap_snap_report();
#ifdef GXVK_HEAPSNAP_TESTING
// tests/gxvk_heapsnap_test.cpp: everything forgotten, the mode and the heap set.
void heap_snap_test_reset(HeapSnap mode, std::uint64_t lo, std::uint64_t hi);
struct HeapSnapTestCounts {
    std::uint64_t fast = 0, fell[kHeapSnapWhys] = {}, guards[kHeapGuards] = {}, diffs[kHeapDiffs] = {};
    bool tripped = false;
};
HeapSnapTestCounts heap_snap_test_counts();
#endif

#ifdef GXVK_ARENA_TESTING
// tests/gxvk_arena_test.cpp: everything forgotten, and the mode set.
void arena_test_reset(Arena mode);
struct ArenaTestCounts {
    std::uint64_t violations_order = 0, violations_bytes = 0, samples_checked = 0, fell = 0;
    std::uint64_t proof_off = 0, unrecorded = 0, live = 0;  // after a failure: bindings and hand-outs it left alone; records kept
    bool in_place = true;
};
ArenaTestCounts arena_test_counts();
#endif

}  // namespace gxvk
