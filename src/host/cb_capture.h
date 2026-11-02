// Stage 4b.3a(i), BBHOST_GXVK_HOMES: a copy-back's bytes captured where GX
// hands it over, on the GX thread, and copied into its home on the GPU.
//
// GX's map resolves (0x2aadf50, 0x2ad1090) copy each renamed buffer back to its
// own memory - its home - through 0x2ab2810 {destination, source, size} on the
// GX thread that resolves. The source is a rename in GX's upload heap, in a
// block GX retires when UpdateSubresource kicks the context's command buffer
// (sub_2aaf700) - before the copy-back is resolved, in a later submission - and
// takes back once the retire's end-of-pipe label has landed. Today the command
// processor reads the rename when it reaches the copy token; that read and a
// GPU copy of the rename (stage 1d.2a) can come after GX has taken the block
// back. Here the bytes are copied at the resolve, before the block's label can
// have landed behind work the command processor has not yet walked, into a
// ring the GPU reads: the command processor never reads the rename.
//
// The ring is cut into a segment per GX thread, each into pieces; a resolve's
// copies all go in one piece. A piece is written again only when every copy
// token written into it has been walked by the command processor and the
// newest submission that read one of its copies has completed; until then the
// thread captures nothing and the copy-backs go as today. A piece held 600
// flips by a token never walked is taken back, and that token's copies then go
// as today (the lap in the piece's handle tells).
//
// BBHOST_GXVK_HOMES=1 captures and the command processor puts the captures in
// place as GPU copies (gpu.cpp: a home kept as a device-local twin takes the
// copy in stream order, any other copy-back a copy version whose bytes are the
// capture); =compare captures, keeps today's path, and compares each capture
// with the bytes today's path copies at the token. 0 (the default): nothing.
//
// BBHOST_GXVK_COPYBACK_CHECK=1 with it: each dynamic constant buffer's
// UpdateSubresource notes a hash of the bytes the rename gets and the upload
// heap's hand-outs so far; the capture compares its bytes with that hash and
// classes the source's block (host/gxvk.h capture_class).
//
// Hosted code only (the GX hooks and the renderer), never a decomp leaf.
#pragma once

#include <cstdint>

namespace cb_capture {

enum class Mode : int { kOff, kOn, kCompare };
extern const Mode g_mode;  // BBHOST_GXVK_HOMES, read at start
inline bool on() { return g_mode != Mode::kOff; }

constexpr std::uint32_t kSegments = 16;                 // GX threads with a segment
constexpr std::uint64_t kSegmentBytes = 4ull << 20;     // each
constexpr std::uint64_t kPieceBytes = 512ull << 10;
constexpr std::uint32_t kPieces = static_cast<std::uint32_t>(kSegmentBytes / kPieceBytes);
constexpr std::uint64_t kRingBytes = kSegments * kSegmentBytes;
constexpr std::uint64_t kMaxCopy = 64u << 10;           // a longer copy-back is not captured
constexpr std::uint32_t kNone = 0xffffffffu;

// The renderer, under its lock: the ring's mapped memory, once it is made.
void set_ring(void* map);
bool ring_made();

// The GX thread, a resolve's copy-backs {dst[i], src[i], bytes[i]} (whole
// dwords) on the command buffer `cb`: copies each source into this thread's
// segment, at[i] its place in 256-byte units (kNone: not captured), and
// returns the piece (lap << 16 | index), kNone when nothing was captured. The
// piece counts one token until release(). `flip`: the flip counter now.
std::uint32_t capture(std::uint64_t cb, const std::uint64_t* dst, const std::uint64_t* src, const std::uint64_t* bytes, std::uint32_t n,
                      std::uint32_t* at, std::uint64_t flip);
inline std::uint64_t offset(std::uint32_t at) { return static_cast<std::uint64_t>(at) * 256; }
// The capture's bytes (the ring's mapped memory at its place).
const void* bytes_at(std::uint32_t at);

// The command processor: whether `piece` still holds the lap it was written in
// (a piece taken back from a token never walked does not).
bool piece_live(std::uint32_t piece);
// ...a capture in `piece` read by submission `serial` (it is written again
// only once that submission has completed).
void bound(std::uint32_t piece, std::uint64_t serial);
// ...the token whose copies are in `piece` walked.
void release(std::uint32_t piece);

// BBHOST_GXVK_COPYBACK_CHECK=1 with the captures on.
bool check_on();
// A GX thread's UpdateSubresource of a dynamic constant buffer: the context's
// command buffer `cb`, the buffer's memory, the bytes the rename gets.
void note_update(std::uint64_t cb, std::uint64_t dst, const void* data, std::uint64_t bytes);

// The command processor's use of the captures, counted for the report.
enum Use : int {
    kUseHomeNow,      // a twin home's GPU copy, the token outside a pass
    kUseHomePending,  // ...inside a pass
    kUseVersion,      // a copy version whose bytes are the capture
    kUseRefused,      // the version refused (a transfer batch open, ...): the copy goes on the GPU
    kUseGuestCopy,    // ...from the capture (host_gpu_copy_guest)
    kUseStale,        // the piece was taken back: as today
    kUseNotCaptured,  // not captured (no ring, no segment, busy, too long): as today
    kUses
};
void count(Use u, std::uint64_t n = 1);
// =compare: a capture against the bytes today's path took at the token.
void compared(bool same, std::uint64_t dst, std::uint64_t src, std::uint64_t bytes);

void report();

}  // namespace cb_capture
