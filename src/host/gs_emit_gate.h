#pragma once

// The GPU particles' geometry shaders as one vertex stage (render.cpp,
// gcn/gs_vertex.h): which of a strip GS's emits a draw can leave out. The
// renderer's choice, not the translation's, so it lives outside src/gcn and
// the translator's fingerprint (the shader caches) stays as it was.

#include "gcn/isa.h"

#include <cstdint>

// A GS whose longest path's emits are partly made under a condition every
// lane shares, read from its constant buffer: the standard particle emits a
// trail's four vertices before its quad's four only when dword 13 of the
// constant buffer in its user SGPRs 4-7 is above 0 (and the particle moved
// far enough), so with that dword at 0 or less no particle emits more than
// four. gs_emit_gate finds that shape in the GS's program:
//   - an s_and_saveexec_b64 whose region, up to the s_mov_b64 that gives
//     EXEC back, holds some of the GS's emits, with more emits after it;
//   - the VCC it takes the AND (s_and_b64) of a lane mask and v_cmp_gt_i32
//     vcc, sN, 0, in straight-line code before it;
//   - sN an s_buffer_load_dword* at a constant offset through the V# in user
//     SGPRs that nothing writes before it, in straight-line code before the
//     compare.
// With that dword at 0 or less the region is never entered: a particle makes
// at most `vertices` emits (those outside the region). Anything else - a cut
// emit, an offset in an SGPR, a branch on the way - finds no gate.
struct GsEmitGate {
    std::uint32_t vertices = 0;     // the most emits with the gate shut; 0: no gate
    std::uint32_t vsharp_sgpr = 0;  // the constant buffer's V#: user SGPRs vsharp_sgpr..+3
    std::uint32_t dword = 0;        // the dword of it whose value, as a signed integer above 0, opens the gate
    std::uint32_t gated = 0;        // the emits in the region
};
GsEmitGate gs_emit_gate(const gcn::Program& gs, std::uint32_t user_sgprs);
