#pragma once
// The vertex clamp (render.cpp plan_vertex_input; BBHOST_VERTEX_CLAMP=0 turns
// it off): a vertex record whose V# counts on past the end of the GX buffer it
// starts in, and which the buffer shadow did not take as it was, is stopped at
// that end - gx_buffer_end, a lookup under the GX registry's lock. What the
// stopped extent decides, and nothing else:
//   - the buffer shadow: its mirrors, asked again with it, and its snapshots
//     of the pages the game rewrites (buffer_shadow_on);
//   - vertex staging, the record copied into device memory for its draw
//     (vertex_stage_on, on by default on a discrete GPU);
//   - the race census, the bytes it watches (BBHOST_VB_RACE).
// With none of them - an integrated GPU without room for the mirrors, a Steam
// Deck or a Radeon 680M - the record is read in place whatever its extent, and
// the lookup bought nothing: at the Nightmare of Mensis on a Steam Deck it was
// 7.7% of the command processor, and skipped there the frame rate went from
// ~53 to ~58 at the same 15 W. Pure; tests/vertex_clamp_test.cpp checks it.

#ifdef VERTEX_CLAMP_MUTANTS
extern int g_vertex_clamp_mutant;
#define VERTEX_CLAMP_MUTANT(n) (g_vertex_clamp_mutant == (n))
#else
#define VERTEX_CLAMP_MUTANT(n) false
#endif

namespace vertex_clamp {

// Whether a record kept in host memory looks its GX buffer up: the clamp on,
// and something on that takes the extent.
inline bool lookup_wanted(bool clamp_on, bool shadow_on, bool stage_on, bool race_on) {
    if (VERTEX_CLAMP_MUTANT(1)) return clamp_on;                            // everywhere, as before
    if (VERTEX_CLAMP_MUTANT(2)) return clamp_on && (stage_on || race_on);   // the shadow left out
    if (VERTEX_CLAMP_MUTANT(3)) return clamp_on && (shadow_on || race_on);  // staging left out
    if (VERTEX_CLAMP_MUTANT(4)) return clamp_on && (shadow_on || stage_on); // the census left out
    if (VERTEX_CLAMP_MUTANT(5)) return shadow_on || stage_on || race_on;    // the switch ignored
    return clamp_on && (shadow_on || stage_on || race_on);
}

}  // namespace vertex_clamp
