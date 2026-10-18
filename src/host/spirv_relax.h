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
// A copy of an assembled module with the NoContraction decorations removed
// where fusing a multiply and an add cannot change what other pipelines must
// agree on. Every Module::emit float op carries one (bit-exact lifts); SPIR-V
// to MSL conversion (MoltenVK's SPIRV-Cross) turns each into a call to a
// [[clang::optnone]] helper, which made the pixel shaders 4-20x slower there.
// Kept, on the backward slice of the module (through values, and through
// every store, copy or atomic into a pointer's variable): in a vertex,
// tessellation or geometry stage (RelaxStages::Graphics) whatever writes the
// Position, ClipDistance or CullDistance outputs, plus every branch and
// switch condition, so positions stay the same in every pipeline that draws a
// surface (EQUAL / LEQUAL depth passes); in a fragment stage that writes
// FragDepth, the FragDepth writes and every branch and switch condition, so
// depth decisions are the same too - and with `exact_discards`, in one that
// can discard, every branch and switch condition, so discard decisions are
// (a surface drawn again by another pipeline under an EQUAL depth test keeps
// its holes). A module with an OpFunctionCall, or of another stage, comes
// back unchanged.
std::vector<std::uint32_t> relax_contraction(const std::vector<std::uint32_t>& words, RelaxStages stages, RelaxStats* stats = nullptr,
                                             bool exact_discards = true);

}  // namespace spv
