// BBHOST_CB_RECORD_PROOF=1: the proof that a GX draw's CPU-written constant
// buffers can be copied when the draw is recorded instead of when the command
// processor resolves it (cpretire phase C; the copy the renderer takes today
// is BBHOST_CB_SNAPSHOT, render.cpp resolve_stage_buffers).
//
// At the GX call (gx_trace.cpp snapshot_records, on the recording thread)
// every constant buffer the draw's records name is copied, whole up to
// kMaxCopy, into that thread's ring, with the submissions the GPU had
// completed by then. Where the command processor binds the buffer, the bytes
// it would copy or read then are compared with that copy, and counted by the
// path the binding takes there (copied as a snapshot, copied to realign it,
// read in place because a GPU write is to land on it, a copy-back's copy, a
// YEBIS window, a buffer the shader stores to) and by what the buffer is (a
// static buffer, a dynamic one's renamed copy, a dynamic one at its own
// address). For the copied paths it also says whether a GPU write not yet
// complete at the call was noted on the range since - the rule a copy taken at
// the call would need to leave such a binding to today's copy - and whether
// that rule catches every binding that changed. Nothing a draw binds changes.
//
// Hosted code only (the GX hooks and the renderer), never a decomp leaf.
#pragma once

#include <cstdint>

namespace cb_proof {

// BBHOST_CB_RECORD_PROOF=1.
extern const bool g_on;

// What a buffer is, as its GX object says at the call.
enum Kind : std::uint8_t {
    kStatic,        // a buffer (type 1) GX updates with GPU copies in command order
    kRenamed,       // a dynamic buffer's copy renamed in the calling context (Map)
    kDynamicOwn,    // a dynamic buffer read at its own address
    kOtherObject,   // anything else
    kKinds
};

// The path the binding takes where the command processor resolves it.
enum Path : std::uint8_t {
    kPathSnapshot,    // copied as recorded: memory the game writes on the CPU
    kPathRealigned,   // copied to put its base on a vec4 (or a Scaleform draw's)
    kPathGpuPending,  // read in place: a GPU write to it is still to land
    kPathVersion,     // a copy-back's copy
    kPathWindow,      // a YEBIS window's copy
    kPathStored,      // the shader stores to it: the guest memory itself
    kPathInPlace,     // read in place otherwise
    kPaths
};
// The paths that copy the bytes when the command processor binds them: what
// a copy taken at the call would replace.
inline bool copied(Path p) { return p == kPathSnapshot || p == kPathRealigned; }

// The largest copy taken at the call; a longer buffer is copied this far.
constexpr std::uint32_t kMaxCopy = 64u << 10;

// The recording thread: copies [addr, addr + bytes) (at most kMaxCopy) into
// this thread's ring, with the flip counter, the submissions completed
// (host_gpu_completed_submits_seen) and the buffer's GX object. The caller has
// checked that the range is mapped. Returns the copy's handle, 0 when there is
// none (more recording threads than rings).
std::uint64_t record(std::uint64_t addr, std::uint32_t bytes, Kind kind, std::uint64_t flip, std::uint64_t completed,
                     std::uint64_t object);

enum Outcome : std::uint8_t {
    kSame,        // the bytes as copied at the call
    kChanged,     // written since the call
    kNotCovered,  // the binding reads past the copy (or before it)
    kLapped,      // the ring came round over the copy before the binding
    kNoCopy,      // a handle that names no ring
    kOutcomes
};

// What the compare found: the copy's header, and for a change the dwords.
struct Difference {
    Kind kind = kOtherObject;
    std::uint64_t addr = 0;        // the copy's start, the buffer's address at the call
    std::uint32_t copied = 0;      // bytes copied at the call
    std::uint64_t flip = 0;        // the flip counter at the call
    std::uint64_t completed = 0;   // submissions completed at the call
    std::uint64_t object = 0;      // the buffer's GX object
    std::uint32_t first_dword = 0; // the first differing dword from the binding's base
    std::uint32_t dwords = 0;      // how many dwords of the binding differ
    std::uint32_t was[4] = {}, now[4] = {};  // from the first differing dword on
};

// The command processor: the binding of [base, base + bytes) whose draw's
// records hold the copy `handle`, taking `path`. Counts the outcome; for
// kSame and kChanged `diff` gets the copy's header (and a change's dwords).
Outcome check(std::uint64_t handle, std::uint64_t base, std::uint64_t bytes, Path path, Difference* diff);

// The same binding with what the caller knows of it: the draw's pipeline,
// the flip counter now, whether a GPU write was noted on the range since the
// call's completed submissions (`gpu_since_call`), whether the write tables
// had let go of writes that recent so they cannot say (`dropped_since`), and
// whether a GPU write was noted on it at all (`gpu_any`). Called for kSame and
// kChanged. The rule a copy at the call needs leaves the binding to today's
// copy for either of the first two.
void note_result(const Difference& diff, Outcome outcome, Path path, const char* pipeline, std::uint64_t flip_now,
                 bool gpu_since_call, bool dropped_since, bool gpu_any, std::uint64_t base, std::uint64_t bytes);

// A binding the proof has no copy for: no GX records (`records` false), or
// the records hold none for its slot (not a constant-buffer slot, or the slot
// empty at the call).
void note_uncopied(Path path, bool records, bool scaleform);

// The exit report's lines.
void report();

// Tests: the ring size, before the first record.
void set_ring_bytes(std::uint64_t bytes);

}  // namespace cb_proof
