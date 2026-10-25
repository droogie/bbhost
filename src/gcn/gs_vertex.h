#pragma once

// A geometry-shader draw as one vertex stage (render.cpp's GS draws with
// BBHOST_GX_PARTICLES=vs).
//
// The GPU particles are drawn by an export shader (ES) that fetches a
// particle and a geometry shader (GS) that expands it into quads: one quad a
// particle for the standard particles, ten for the sparks' trails. Geometry
// shaders are slow everywhere and poor on AMD, so the draw goes as a vertex
// shader instead, the way a PC engine draws particles: each output vertex
// runs its particle's ES and GS and keeps the vertex it stands for.
//
// gs_vertex_stage takes the two lifts as they are (lift_es_shader,
// lift_gs_shader: exact, and through the same caches and checks) and splices
// them into one module:
//   - vertex v of the draw is emitted vertex k = v % vertices of particle
//     p = v / vertices (`vertices`: the GS's most emits on one path, four a
//     quad);
//   - the draw's indirect arguments - {count, instances, first vertex, first
//     instance}, the simulation's - are read at the device address in the GS's
//     StageParams::lds_address: a particle at or past the count is left out
//     (all of them with no instance; the first instance alone is drawn), and
//     the ES runs for vertex `first + p`, as the GS draw ran it;
//   - the ES's ring dwords stay in the invocation (a Private array) for the
//     GS's ring loads;
//   - each GS emit up to emit k keeps its outputs: vertex k is emit k, or past
//     the particle's emits its last one, which makes every triangle it is in
//     with the vertices before it degenerate (no area, nothing drawn);
//   - a particle's emits are one strip when every cut of the GS comes after
//     its last emit (`strip`; the standard particles'), else strips of four
//     (the sparks' trails of quads); one that breaks its shape, or emits more
//     than its `vertices`, is left out (the check mode counts it), as is one
//     that emits nothing;
//   - the outputs are the GS's own (the copy shader's map), so the pixel
//     shader's linkage is unchanged.
// The host draws a particle's triangles in Vulkan's strip order, so the
// provoking vertex of each (flat inputs) is the strip's: one strip as triangle
// i = (i, i+1+i%2, i+2-i%2) of its vertices; quads as 4j+0, 4j+1, 4j+2,
// 4j+1, 4j+3, 4j+2 for quad j.
//
// The check mode (BBHOST_GX_PARTICLES=check) draws with the GS and runs the
// vertex stage beside it with the rasterizer off, both writing what each
// vertex gets into the draw's check record for the host to compare bit for
// bit: gs_vertex_stage with GsVertexOptions::check, and gs_check_stage for
// the GS. Then StageParams::lds_address is the record (below), whose header
// holds the arguments' address.

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

// One output of the GS in a check record's vertex: `dwords` words at word
// `offset` of the vertex, a builtin (0 the position) or the output at
// `location` (~0u for the other).
struct GsOutputSlot {
    std::uint32_t builtin = ~0u, location = ~0u;
    std::uint32_t offset = 0, dwords = 0;
};

// The check record (host-visible, at StageParams::lds_address):
//   header, kGsCheckHeaderDwords words: [0..1] the indirect arguments' device
//     address (the host's), [2] the particles it has room for (the host's), [3]
//     a particle's vertices, [4] a vertex's words (both the host's), [5..7] the
//     arguments the vertex stage read (count, instances, first vertex) and [8]
//     1 once it wrote them (its vertex 0 does);
//   then the vertex stage's particles and the GS's, room-for each, a particle
//     [0] its emits, [1] 1 | 2 (its strips not the shape its draw has) | 4
//     (more emits than its vertices), then its first `vertices` emits, a
//     vertex's words each, as the outputs list them.
constexpr std::uint32_t kGsCheckHeaderDwords = 16;
constexpr std::uint32_t kGsCheckParticleHead = 2;
inline std::uint32_t gs_check_particle_dwords(std::uint32_t vertices, std::uint32_t vertex_dwords) {
    return kGsCheckParticleHead + vertices * vertex_dwords;
}

struct GsVertexOptions {
    bool check = false;          // the check mode's vertex stage (above): its every vertex outside the clip volume
    std::uint32_t vertices = 0;  // a particle's vertices; 0: the GS's most emits on one path
};

struct GsVertexStage {
    std::vector<std::uint32_t> spirv;  // empty when refused
    // A particle's vertices, four a quad: the most emits an invocation can make,
    // from the emits on one path of the GS (path_emits) less those that exclude
    // each other by the conditions they are made under.
    std::uint32_t vertices = 0;
    std::uint32_t path_emits = 0;
    bool strip = false;                // a particle's vertices one strip (else quads, `vertices` a multiple of four)
    std::uint32_t vertex_dwords = 0;   // a check record's vertex
    std::vector<GsOutputSlot> outputs; // a check record's vertex, in order
    std::string why;                   // the refusal
    bool ok() const { return !spirv.empty(); }
};

GsVertexStage gs_vertex_stage(const std::vector<std::uint32_t>& es, const std::vector<std::uint32_t>& gs, const GsVertexOptions& options = {});

// The check mode's geometry stage: the GS lift as it is, each invocation's
// first `vertices` emits and its emit count also stored into the draw's check
// record (its primitive the particle).
GsVertexStage gs_check_stage(const std::vector<std::uint32_t>& gs, std::uint32_t vertices);

}  // namespace gcn
