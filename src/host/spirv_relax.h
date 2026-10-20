// Applied on the way to the driver (render.cpp make_module), not in the
// translator: the translation cache and the lifter's byte-identical checks
// see the modules as the translator wrote them.
#pragma once

#include <cstdint>
#include <vector>

namespace spv {

// How a vertex-pipeline stage's Position stays the same across pipelines:
// Graphics keeps its arithmetic exact; GraphicsInvariant decorates it
// Invariant instead (MSL [[invariant]]) and leaves a stage writing none alone.
enum class RelaxStages { Fragment, Graphics, GraphicsInvariant };
struct RelaxStats {
    std::uint32_t dropped = 0, kept = 0;  // NoContraction decorations removed / left
};
// SPIRV-Cross turns each NoContraction op into a call to an optnone helper,
// 4-20x slower pixel shaders on MoltenVK. Kept on the backward slice of what
// pipelines drawing the same surface must agree on (EQUAL / LEQUAL depth):
// Position, Clip/CullDistance and FragDepth writes and every branch
// condition, and with `exact_discards` every branch of a shader that can
// discard. A module with an OpFunctionCall comes back unchanged.
std::vector<std::uint32_t> relax_contraction(const std::vector<std::uint32_t>& words, RelaxStages stages, RelaxStats* stats = nullptr,
                                             bool exact_discards = true);

}  // namespace spv
