// BBHOST_COPY_CENSUS=1: the frame's image copies by place - what the host
// copies on the GPU for the game (CopyResource of a render target, the copy
// tokens of whole textures and regions, a target's region copied at a bind)
// and the game's own copy passes (PointSampleColor1, PointSampleColorClamp1:
// a full-screen draw that samples one texture into one target) - with, for
// each copy, what happened to its source and its destination until the
// destination was next written: whether the destination was read, and
// whether the source was written or bound before the destination's last
// read. A copy whose destination is read only while its source is unchanged
// could be read from the source instead (copy_elide.h). Nothing here changes
// what is drawn; every call is under g.mu, on the command processor (a run
// part's draw is left out and counted).
#pragma once

#include <cstdint>

namespace gpu {
struct RtImage;
namespace copy_census {

extern const bool g_on;
inline bool on() { return g_on; }

enum Kind : int {
    kTargetColour,        // a colour target copied whole into a snapshot or a target (record_target_copy_locked)
    kTargetDepthCompute,  // a depth target's plane into an R32F snapshot by the compute pass
    kTargetDepthBuffer,   // the same through a buffer
    kRegionBind,          // a target's top-left region copied for a T# at a bind (rt_copied_view)
    kImageRegion,         // a copy-image token (a region of one texture into another)
    kResource,            // a whole texture image to image by resource (textures_copy_resource_locked)
    kPointCopy,           // the game's copy pass: a full-screen point-sampled draw
    kKinds
};
enum Touch : int {
    kTouchDraw,      // drawn into (an attachment with a write mask, depth or stencil writes)
    kTouchDrawFull,  // drawn into by a draw that may write all of it (every channel, no blending, viewport and scissor cover it)
    kTouchDrawProven,  // drawn into by a draw proven to write every pixel of it (host/full_write.h)
    kTouchBound,     // bound as an attachment of a pass, not written by the draw
    kTouchClear,     // a clear of all of it (a clear token's, DB_RENDER_CONTROL's, HTILE's)
    kTouchClearPart, // a clear of some of its layers
    kTouchFill,      // a fill over its memory
    kTouchCopyInto,  // a copy into it
    kTouchDispatch,  // a dispatch's storage binding at its base
    kTouchRetire,    // its image destroyed or replaced
    kTouchOther,     // any other write stamp (rt_written)
    kTouches
};
// Flags of a point copy (and of a copy whose shape matters).
enum Flag : std::uint32_t {
    kSameSize = 1,     // source and destination the same width and height
    kSameFormat = 2,   // the same VkFormat (or T# format of the target's)
    kBlendOff = 4,     // no blending on the destination
    kMaskFull = 8,     // every channel of the destination written
    kCovers = 16,      // the viewport and scissor cover the destination whole
    kSrcImage = 32,    // the source is a target or a snapshot (an image the host draws or copies)
    kWhole = 64,       // the copy covers its destination whole
};

struct Side {
    std::uint64_t base = 0, bytes = 0;  // the guest range it stands for (reads inside it count)
    std::uint32_t w = 0, h = 0;
    int format = 0;  // VkFormat, or 1000 + T# dfmt for a texture with no host format named
};

// rt_written calls inside a Quiet scope are not touches (a draw's target
// lookups - the draw hook says what it wrote - and a copy's own stamps); As
// names what the stamps inside it are.
struct Quiet {
    Quiet();
    ~Quiet();
};
struct As {
    explicit As(int touch);
    ~As();
    int prev;
};

void copy(int kind, const Side& src, const Side& dst, const char* who, std::uint32_t flags = 0);
void region_bind(const RtImage& src, std::uint32_t w, std::uint32_t h, bool fresh);
void rt_written(const RtImage& r);
bool touch(std::uint64_t base, int touch, const char* who = nullptr);  // whether it reached a copy's source or destination
void read(std::uint64_t base, const char* who);
void retire(std::uint64_t base);
void part_skipped();
// Phase 2's question for a draw into a target: why it is not proven to write all of it (full_write::Fail), by writer.
void proof(const char* writer, int fail);
// The 5-second report (every 300 flips): each copy site's line since the last.
void report_locked();

}  // namespace copy_census
}  // namespace gpu
