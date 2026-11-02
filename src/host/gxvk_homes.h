// Stage 4b.3c, BBHOST_GXVK_HOMES=1: GX's constant buffers in device-local
// homes made when GX creates them, bound by handle.
//
// A constant buffer's own memory - its home - is written by GX alone, and only
// on the GPU: UpdateSubresource of a constant buffer always renames it (BN
// 0x2568a20: bind flag 4 -> 0x2aadb90, a copy carved from the upload heap),
// a Map of a dynamic one hands out a rename too (types 3 and 4), and the map
// resolves copy the renames back to the home in command order (0x2aadf50,
// 0x2ad1090 -> 0x2ab2810 / 0x2ab2670). Direct3D 11, which GX follows, lets no
// CPU write a default or dynamic buffer's own memory. Today the command
// processor turns each copy-back into a copy version at its token, and every
// later binding of the home asks the versions, the twins and the pending GPU
// writes what the home holds at that point of the walk: an answer only the
// command processor's order gives (stage 4's census: 66-83% of GX-native draws
// read a version, 14-44% a GPU write to land).
//
// Here each such buffer gets its home when GX makes it: a range of a
// device-local pool block, its place fixed for the buffer's life, and a handle
// (a tag, the entry's generation and its index) in the holder's spare dword at
// +0x14 (host/gxvk.h kHolderSpareA, which stage 1b uses for the vertex and
// index buffers it keeps and leaves 0 for a constant buffer). The GX thread
// copies the handle into a draw's records beside the constant buffer's record
// when the record names the home (no rename of it in the draw's context), and
// the command processor binds the home by the handle: an index into the table
// and a generation check. Nothing on that path asks the copy versions, the
// twins, the pending GPU writes or the snapshot memo: the home is the buffer,
// and the GPU's queue order is what puts each copy-back in it before the draws
// after the copy-back read it.
//
// The copy-backs keep stage 4b.3a(i)'s capture at the GX resolve
// (host/cb_capture.h) and land in the home as GPU copies from the captures'
// ring in stream order - at the token outside a render pass, else when the
// next pass starts (a draw in the same pass that binds the home binds the
// capture: counted, as an answer of the order). A copy into a home the capture
// missed (a single copy, one over 64 KiB, a ring that was busy) is staged on the
// CPU at its token and copied in the same way. A home is filled from the
// buffer's memory - its initial data, which nothing writes after the creation -
// in the uploads of the first recording that uses it, unless that use is a
// copy-back over the whole of it.
//
// What reads a home's memory without a handle - a dispatch's constant buffer,
// a stage without GX records, a copy, a fill, a device address - finds it by
// address (the command processor's index of the live homes, a page filter in
// front of it): locate() and guest_device_address() hand out the home, and
// every CPU copy of its guest bytes, which are stale, is refused, as for a
// twin. A page-table walk or a range across a home's edge reads guest memory:
// the home is copied back there first (a GPU copy, in stream order) and stays
// the home - no state of it changes, so the next binding's answer is the same.
// A home ends with its buffer (its holder released, its memory unmapped or
// given to another resource): its entry stays bindable for the draws recorded
// before the end (8 flips) and its range is freed once the work that read it
// has run. A Map that hands out a home's own memory (none in any census: D3D11
// forbids it for these usages) copies the home back and retires it - the
// buffer then takes today's path.
//
// BBHOST_GXVK_HOMES: unset or 0 today's path (the default until its gate
// passes); 1 homes by handle (with the captures; the GPU particles' twins stay
// as they are); compare: today's path is bound, and the homes are kept as 1
// keeps them - made, filled, written by the copy-backs on the GPU, with a CPU
// mirror of what each holds - and each binding today's path makes of a home is
// checked against its mirror, a sample of homes read back from the GPU after
// their copies; twins: stage 4b.3a(i) as it was (the homes as level-2 twins).
// Off on an integrated GPU (its video memory is the system's); on a discrete
// GPU the pool has a budget (128 MiB, 64 MiB with small video memory,
// BBHOST_GXVK_HOMES_BUDGET_MB): a buffer made past it gets no home and takes
// today's path for its life - decided at its creation, so no home is ever
// taken from a buffer that has one.
#pragma once

#include <atomic>
#include <cstdint>

namespace gxvk_homes {

enum class Mode : int { kOff = 0, kHandle = 1, kCompare = 2, kTwins = 3 };
// BBHOST_GXVK_HOMES as given (the device's decision aside).
Mode env_mode();

// The handle in a holder's +0x14: the tag, the entry's generation (8 bits) and
// its index (20 bits). Stage 1b's handles have tag 0xa.
constexpr std::uint32_t kTag = 0xb0000000u, kTagMask = 0xf0000000u;
constexpr std::uint32_t kIndexBits = 20, kIndexMask = (1u << kIndexBits) - 1;
inline bool is_handle(std::uint32_t w) { return (w & kTagMask) == kTag; }
inline std::uint32_t handle_of(std::uint32_t index, std::uint32_t gen) { return kTag | ((gen & 0xff) << kIndexBits) | index; }
inline std::uint32_t handle_index(std::uint32_t h) { return h & kIndexMask; }
inline std::uint32_t handle_gen(std::uint32_t h) { return (h >> kIndexBits) & 0xff; }

// The pool's blocks: a home is a range of one. A binding by handle binds its
// block whole, the home's place in the dwords' bias (one descriptor for every
// home in a block, as the constant-buffer snapshots bind their ring whole).
constexpr std::uint64_t kBlockBytes = 16ull << 20;

// Whether homes are made at creation (1 or compare, on a device that allows
// them): what GX's threads ask before they look for a handle. False until the
// device is chosen.
extern std::atomic<bool> g_made;
inline bool made_on() { return g_made.load(std::memory_order_relaxed); }
// Whether bindings take the homes (1), rather than only check them (compare).
extern std::atomic<bool> g_bind;
inline bool bind_on() { return g_bind.load(std::memory_order_relaxed); }

// Why a buffer got no home, and why a binding with a handle took today's path.
enum NoHome : int {
    kNoHomeNotConstant,  // not a constant buffer, or usage immutable or staging (nothing GX's copies write)
    kNoHomeOtherBind,    // a constant buffer bound as something else too (vertex data: read by address, past its end)
    kNoHomeCpu,          // CPU access other than a dynamic buffer's write (staging-like)
    kNoHomeSize,         // empty, or over 64 KiB
    kNoHomeBudget,       // the pool at its budget
    kNoHomeNoRoom,       // no device memory for a block
    kNoHomeTable,        // the table full
    kNoHomeNoDevice,     // made before the device (or homes off on it)
    kNoHomes,
};
enum BindFall : int {
    kFallStale,      // the handle names no live entry of that buffer (taken back, a clobbered dword)
    kFallRetired,    // its entry retired by a CPU map of its own memory
    kFallRange,      // the binding runs outside the home
    kFallPart,       // a run part's draw (BBHOST_RUN_COMMIT) with a fill to do: in order
    kFallFill,       // its bytes could not be reached for its fill
    kFalls,
};
// How a home ends. The last two hand the buffer to today's path (the home copied back to its memory first): a Map
// that hands out its own memory, and a GPU access of guest memory across the home's edge (a copy, a fill, a binding
// past its end) or a page-table walk's stores over it - none expected, each counted.
enum End : int { kEndGone, kEndReplaced, kEndUnmapped, kEndCpuMapped, kEndEdge, kEnds };

}  // namespace gxvk_homes
