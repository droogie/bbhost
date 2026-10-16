// One-control-point quad patches with constant factors (the SFX particles)
// drawn as a plain vertex-stage triangle list: MoltenVK runs tessellation
// through a compute encoder that restarts the render pass on every such draw,
// and here the tessellator decides nothing the draw does not already know.
// The LS and domain shader must be translated with tess_lds_attributes and
// params_vertex_formats so their params blocks merge.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

// Must match integer-spaced tessellation of a quad (outer `level`, inner 1,
// origin upper left): for L > 1 the inner 1 rounds to 2, so the patch is a fan
// of 4L triangles from (0.5, 0.5). `cw` as VertexOrderCw.
constexpr std::uint32_t kTessVertexMaxLevel = 16;
std::uint32_t tess_vertex_points(std::uint32_t level);  // vertices a patch: 6, or 12 * level
void tess_vertex_domain_point(std::uint32_t level, bool cw, std::uint32_t k, float& u, float& v);

struct TessVertexFuse {
    std::vector<std::uint32_t> spirv;
    std::string error;  // why the modules were not fused; empty on success
    bool ok() const { return error.empty() && !spirv.empty(); }
};
// `ls` writes its control point at output location 0, `ds` reads it at input
// location 0. Draw the result non-indexed from 0, tess_vertex_points(level)
// vertices a patch. Fails on anything the merge does not know how to carry.
TessVertexFuse fuse_tess_vertex(const std::vector<std::uint32_t>& ls, const std::vector<std::uint32_t>& ds, std::uint32_t level, bool cw);

}  // namespace gcn
