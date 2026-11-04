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
// discrete GPU with room in its video memory; on one with small video memory
// within a budget for their copies (host/gxvk_budget.h), the least recently
// bound dropped first and the pool giving room back before the buffer
// shadow's mirrors are refused; and on an integrated GPU with 8 GiB of it or
// more (a Strix Halo's carve-out): read from the carve-out rather than from
// the imports, the Radeon 8060S's frame took ~7.5% less GPU time (164.9
// against 157.9 fps at 1080p, the GPU the limit, the run commit on;
// docs/list-resolve.md) for 256 MiB of copies. Today's path (0) on a smaller
// integrated GPU or with tight memory, where the draws read GX's buffers
// where they are (the buffer shadow in place there, or no shadow) and a
// buffer's copy would hold memory the system needs. 0 today's path; 1 by
// handle on any GPU; 2 both, compared; "census" the census alone (as
// BBHOST_GXVK_CENSUS=1, which can join the others).
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
    if (!env || !*env)
        return device == BuffersDevice::kRoomy || device == BuffersDevice::kSmallVram || device == BuffersDevice::kIntegratedRoomy ? Buffers::kOn
                                                                                                                                 : Buffers::kOff;
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
// The import census's (gpu.cpp, BBHOST_IMPORT_CENSUS=1): whether a kept buffer's live entry holds any of
// [va, va + bytes) - a binding of it that reached guest memory took today's path where its handle's was there.
bool kept_at(std::uint64_t va, std::uint64_t bytes);

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
// BBHOST_GXVK_COPYBACK_CHECK=1 (a diagnostic, off by default; counts only): today's copy-backs read their source when
// the command processor reaches the copy token, and GX retires a context's blocks when UpdateSubresource kicks its
// command buffer, before the copy-backs that read renames in them are resolved. So a copy-back's source can be in a
// block whose retire label has landed - GX may take it back any time - or one GX has already handed out again. The
// arena keeps its hand-out records for this (both of the pool's ends), each with its node, the hand-out's number and,
// once the command processor writes the node's status (the retire's end-of-pipe label), the submission that label
// lands with; at each copy-back token its source's record is classed and counted for the exit report.
bool copyback_check_wanted();
// The hand-outs so far (lock-free): what the GX thread stamps a copy-back batch with as it records it.
std::uint64_t arena_handout_seq();
// The command processor writes `bytes` at `va` in the submission being recorded, `serial` (host_gpu_mem_write_serial):
// a hand-out's node, its block retired.
void arena_label(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial);
// A copy-back from [src, +bytes) as the command processor reaches its token; `seq`: the hand-outs so far when the GX
// thread recorded it.
void copyback_check(std::uint64_t src, std::uint64_t bytes, std::uint64_t seq);
// This thread's tag (a hand-out records the tag of the thread GX handed it out on, with the check on).
std::uint32_t arena_thread_tag();
// Stage 4b.3a(i) with the check: the GX thread that captures a copy-back (host/cb_capture.h) classes its source's block
// against what UpdateSubresource noted when it wrote the rename - `seq` the hand-outs so far then, `tid` its thread's tag:
// the hand-out that held the source then (seq <= `seq`: the update's block, its bytes GX's), a hand-out since on the same
// thread (the update's own new block, or that block handed out again to the same thread), one since on another thread
// (handed out again: a miss), none. *landed: the block's retire label had landed (GX may take it back any time).
// A hand-out since on the same thread whose first bytes are the rename is the update's own new block (the pool carves
// a fresh block from its start); one the rename lies deeper in was handed out again since.
enum CaptureClass : int { kCapUpdateBlock, kCapNewBlock, kCapSinceSameThread, kCapSinceOtherThread, kCapNoRecord, kCapNotHeap, kCapClasses };
int capture_class(std::uint64_t src, std::uint64_t bytes, std::uint64_t seq, std::uint32_t tid, bool* landed);
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
// The arena's records and its proof are wanted: by the heap's bindings in place (BBHOST_GXVK_ARENA) or by
// copy-backs from the heap (BBHOST_GXVK_HEAP_COPY, stage 1d.2a). The pool's hand-out hooks and the region's
// bounds are installed for either.
bool arena_proof_wanted();
// A copy-back's source in the heap (stage 1d.2a): the batch's GPU copy and the draws that bind its version read
// [va, +bytes), all in submission `serial`. True: one hand-out holds it whole and the proof holds - the record is
// stamped with `serial`, and one hold in 64 has its bytes hashed (every one with BBHOST_GXVK_HEAP_COPY=compare) - so
// the range keeps those bytes until that submission has completed: a hand-out over it before then waits for the GPU
// first (arena_handout; sixteen such waits within a minute stop the proof). False: why is counted (no hand-out holds
// it, it runs past its hand-out, the proof failed earlier).
bool arena_hold(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial);
// The binding found no one buffer over the heap to bind it in.
void arena_no_backing();
// Today's path over the heap, counted where it runs: the work stage 1c takes
// away, and what the command processor reads of the heap on the CPU (the
// descriptor tables in it).
enum ArenaNote : int { kArenaSnapshotCb, kArenaSnapshotVertex, kArenaGpuPending, kArenaTableRead, kArenaNotes };
void arena_note(ArenaNote what, std::uint64_t bytes);
// The exit report (gx_resources_report calls it).
void arena_report();
// GX's upload heap's bounds once its pool is made, whatever BBHOST_GXVK_ARENA says (the import census's owners:
// gpu.cpp); false before.
bool upload_heap(std::uint64_t* lo, std::uint64_t* hi);
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

// --- Stage 1d.2a: copy-backs from the heap with no CPU copy (host/gxvk_heapsnap.cpp; gpu.cpp copy_version_make_locked) ---
//
// GX's map resolves copy each renamed buffer back to its own memory, its home: on the console a GPU copy in command
// order. bbhost makes each such copy-back a copy version on the CPU - the source's bytes copied into staging as the
// command processor reaches it, bound for the home's draws until the batch's GPU copy puts them in place - so a
// draw of the home sees the bytes the copy-back will write. Every source lies in GX's upload heap, which GX hands
// out again only after the work that read it has completed (the arena's release order, proven per run), so the
// version's bytes can be the source range itself: no staging, no CPU copy. The batch's GPU copy reads the heap; a
// draw that binds the home as a shader's buffer takes the source through stage 1d.1's snapshot (a copy in the
// ring, so no shader reads a renamed constant buffer in place), ahead of a YEBIS window as today's version is - one
// over the snapshots' 1 MiB puts the copy-backs in place first, and one the shader stores to does so today; a
// vertex or index binding reads the heap range, under the same proof as the copy. A source the proof does not hold, one an earlier version covers, or any while
// 1d.1's snapshots are off or its guards have tripped takes today's version, counted by reason.
// BBHOST_GXVK_HEAP_COPY: 0 (the default) today's versions; 1 the heap's sources taken as they are; compare: today's
// versions, with the proof run over the sources as 1 would, and each version's bytes compared with its source's at
// its batch (counted, the first few said).
enum class HeapCopy : int { kOff = 0, kOn = 1, kCompare = 2 };
HeapCopy heap_copy_mode();
inline bool heap_copy_wanted() { return heap_copy_mode() != HeapCopy::kOff; }
// A copy-back as copy_version_make_locked has it, for the choice.
struct HeapCopyCase {
    std::uint64_t lo = 0, hi = 0;      // the heap's bounds (lo 0: not known yet)
    std::uint64_t src = 0, bytes = 0;  // the copy-back's source
    bool snapshots = true;             // stage 1d.1 takes the heap's answers (its mode on)
    bool tripped = false;              // a 1d.1 guard has tripped this run
    bool ring = true;                  // the constant-buffer copies' ring is made
    bool earlier = false;              // an earlier version covers the source (the copy reads that version)
    bool located = true;               // the source lies in one import
};
// The choice: kHeapCopyFast (the version is the heap range, if the arena holds it), kHeapCopyNotHeap (not the
// heap's: nothing to count), or why the version is a CPU copy as today.
enum HeapCopyWhy : int {
    kHeapCopySnapOff, kHeapCopyTripped, kHeapCopyPastEnd, kHeapCopyNoRing, kHeapCopyEarlier, kHeapCopyNotLocated,
    kHeapCopyNotHeld,
    kHeapCopyWhys,
    kHeapCopyFast = -1, kHeapCopyNotHeap = -2
};
int heap_copy_class(const HeapCopyCase& c);
// copy_version_make_locked's choice for a copy-back from [src, +bytes) (the renderer's lock held), counted:
// kHeapCopyFast, kHeapCopyNotHeap or a reason. `ring`, `earlier`, `located`: as HeapCopyCase.
int heap_copy_decide(std::uint64_t src, std::uint64_t bytes, bool ring, bool earlier, bool located);
// The arena's answer for a source the choice took (arena_hold): counted; with false the version is today's.
void heap_copy_held(bool held, std::uint64_t bytes);
// Compare: at its batch, a version's bytes and its source's differ (the source changed while the version was pending).
void heap_copy_changed(std::uint64_t src, std::uint64_t bytes);
// A shader's-buffer binding of a home whose version is its source in the heap (render.cpp): snapshotted from the
// source (`snapshot`), or over 1 MiB, the copy-backs put in place first. With =compare, what 1 would do.
void heap_copy_bound(bool snapshot);
// The constant-buffer copies' ring is made (render.cpp, the first snapshot): a heap-sourced version's constant-buffer
// bindings need it.
void heap_copy_ring_made();
bool heap_copy_ring();
// The exit report (gx_resources_report calls it).
void heap_copy_report();
#ifdef GXVK_HEAPSNAP_TESTING
void heap_copy_test_reset(HeapCopy mode);
struct HeapCopyTestCounts {
    std::uint64_t fast = 0, held = 0, fell[kHeapCopyWhys] = {}, changed = 0, bound = 0, bound_wide = 0;
};
HeapCopyTestCounts heap_copy_test_counts();
#endif

#ifdef GXVK_ARENA_TESTING
// tests/gxvk_arena_test.cpp: everything forgotten, and the mode set.
void arena_test_reset(Arena mode);
struct ArenaTestCounts {
    std::uint64_t violations_order = 0, violations_bytes = 0, samples_checked = 0, fell = 0;
    std::uint64_t proof_off = 0, unrecorded = 0, live = 0;  // after a failure: bindings and hand-outs it left alone; records kept
    std::uint64_t holds = 0, hold_fell = 0;                  // copy-backs' sources held (arena_hold), and refused
    std::uint64_t hold_waits = 0;                            // hand-outs that waited for a held source's copy
    bool in_place = true;
};
ArenaTestCounts arena_test_counts();
#endif

}  // namespace gxvk
