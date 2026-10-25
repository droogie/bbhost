#pragma once

// A geometry-shader draw as one vertex stage (render.cpp's GS draws; with
// BBHOST_GX_PARTICLES=gs, a debug switch, the renderer draws them with the
// geometry shader instead).
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
//     than its `vertices`, is left out, as is one that emits nothing;
//   - the outputs are the GS's own (the copy shader's map), so the pixel
//     shader's linkage is unchanged.
// The host draws a particle's triangles in Vulkan's strip order, so the
// provoking vertex of each (flat inputs) is the strip's: one strip as triangle
// i = (i, i+1+i%2, i+2-i%2) of its vertices; quads as 4j+0, 4j+1, 4j+2,
// 4j+1, 4j+3, 4j+2 for quad j.

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

struct GsVertexOptions {
    std::uint32_t vertices = 0;  // a particle's vertices; 0: the GS's most emits on one path
};

struct GsVertexStage {
    std::vector<std::uint32_t> spirv;  // empty when refused
    // A particle's vertices, four a quad: the most emits an invocation can make,
    // from the emits on one path of the GS (path_emits) less those that exclude
    // each other by the conditions they are made under.
    std::uint32_t vertices = 0;
    std::uint32_t path_emits = 0;
    bool strip = false;  // a particle's vertices one strip (else quads, `vertices` a multiple of four)
    std::string why;     // the refusal
    bool ok() const { return !spirv.empty(); }
};

GsVertexStage gs_vertex_stage(const std::vector<std::uint32_t>& es, const std::vector<std::uint32_t>& gs, const GsVertexOptions& options = {});

}  // namespace gcn
