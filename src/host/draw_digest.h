// The draw digest (BBHOST_DRAW_DIGEST=path): what the command processor
// draws, in the order it draws it, for a proof that a change leaves a frame's
// draws as they were. Each draw it makes from a GX draw's token becomes one
// 64-bit hash of what the draw is - the call site, the shaders and the input
// layout, the blend, depth-stencil and rasterizer state, the targets and the
// viewport, and the counts; not its constants, its user data or the
// addresses of per-frame buffers - and each frame ends where the flip's
// commands are done. tools/draw_digest.py compares the frames of a run, on
// and off a switch that flips during it (BBHOST_RQ_SPLIT_EVERY), or of two
// runs. One hash a draw on the command processor's thread while on, a flag
// test while off.
//
//   BBHOST_DRAW_DIGEST=path         on; the records written to path as they fill and at exit
//   BBHOST_DRAW_DIGEST_EVERY=N      only every Nth frame (1)
//   BBHOST_DRAW_DIGEST_FRAMES=M     at most M frames recorded (20000)
#pragma once

#include <cstdint>

struct GpuDrawInputs;
struct GxDrawObjects;

extern bool g_draw_digest;
inline bool draw_digest_on() { return g_draw_digest; }

// A draw the command processor is about to make (its thread).
void draw_digest_note(const GpuDrawInputs& in, const GxDrawObjects& o);
// A frame's end: the flip's commands done (the command processor's thread).
void draw_digest_frame();
