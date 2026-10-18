#pragma once

// pipeline_probe: a shader module built into the Vulkan pipelines the
// renderer would make of its stage (host/render.cpp), on a real device, for
// the driver's verdict - the HLSL path's driver-acceptance sweep
// (tools/gcn2hlsl.cpp --sweep). The device is the one bbhost would run on
// (BBHOST_GPU_DEVICE, else a discrete GPU over an integrated one, then the
// most memory), created with the features and extensions of bbhost's that a
// shader's compile can depend on (host/gpu.cpp), and the lifter is set up for
// it as bbhost sets it up: run-time sample offsets with maintenance8, native
// half rounding where the device rounds f16 toward zero.
//
// A graphics stage needs pipeline state its module does not give. A probe
// takes, from the module's own interface:
//   - descriptor set layouts from its bindings (set, binding, type, count),
//     for every stage, as the renderer's per-stage layouts; a push constant
//     range when it has push constants;
//   - where the device fast-links VK_EXT_graphics_pipeline_library, as the
//     renderer: the stage's own library - a vertex shader's
//     pre-rasterization library (dynamic viewport, scissor, cull mode, front
//     face and depth bias, depth clamp where it is dynamic), a pixel
//     shader's fragment library (one sample, depth and stencil dynamic, the
//     subgroup size the renderer asks for) - kept for link-time optimization,
//     then a fast link and an optimized link of it with the other three
//     libraries; elsewhere, or with `monolithic`, one whole pipeline of the
//     same state;
//   - generated neighbours: for a pixel shader, a vertex shader writing each
//     of its inputs (location, type and interpolation as it reads them) and
//     colour targets by output type (R16G16B16A16_SFLOAT for floats,
//     R32G32B32A32 UINT or SINT for integers) over D32_SFLOAT_S8_UINT; for a
//     vertex shader or an LS on its own, its inputs as 32-bit vertex
//     attributes of their type and no pixel shader; for a domain shader with
//     no game hull, a generated vertex and hull shader giving its control
//     points and patch constants; a game hull's LS, hull and domain shader
//     together on a patch list, as a native hull pipeline.
// What that does not cover: the game's real target formats (UNORM, sRGB,
// R11G11B10, its depth formats), blending, multisampling, its real vertex
// formats (on AMD an optimized pipeline compiles the vertex fetch into the
// vertex shader), the specialization constants of the renderer's optimized
// relinks (vertex formats, indexed loads' words), the real vertex and pixel
// shader pairs (an optimized link drops outputs nobody reads), the rect
// lists' geometry shader, the particles' generated control stage, and push
// descriptors.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu {
class Dxc;
}

namespace probe {

struct DeviceInfo {
    int index = -1;            // among the physical devices, as bbhost's log numbers them ("gpu: device N")
    std::string name;          // the device's name
    std::string driver;        // the driver's name and version
    std::uint32_t vendor = 0, api = 0;
    bool libraries = false;    // pipelines from graphics pipeline libraries, as the renderer builds them here
    bool runtime_offsets = false, native_half_rtz = false, dynamic_depth_clamp = false, depth_bounds = false;
    bool validation = false;   // the Khronos validation layer is on
    int ps_wave = 0;           // 0: the device's subgroup size; 1: the driver's choice; 32: required (host/gpu.cpp)
    std::uint32_t subgroup_size = 0, subgroup_min = 0, subgroup_max = 0;
    bool fragment_sizes = false;  // the fragment stage takes a required subgroup size
    std::string describe() const;  // what the sweep prints of the device and the lifter's switches
};

// One probe's outcome: built, refused by the driver at the first creation
// that failed, or not tried (a module the probe cannot place, or a generated
// stage the driver refused - the probe's own failing, not the module's).
struct Verdict {
    bool ok = true;
    bool skipped = false;
    std::string what;           // the creation refused, or why nothing was tried
    int result = 0;             // its VkResult
    std::string message;        // what the driver or the validation layer said while creating it
    std::string warning;        // the first validation error on a creation the driver took
    std::string note;           // a retry the renderer would make too (a subgroup size refused)
    std::uint32_t created = 0;  // pipelines and libraries created
    bool refused() const { return !ok && !skipped; }
    std::string str() const;    // "built", "not tried: ...", "refused: <creation> VkResult <n> (<message>)"
    std::string token() const;  // one word for list files: built, untried, refused:<creation>:<n>
};

class Prober {
public:
    explicit Prober(gpu::Dxc& dxc);  // DXC compiles the generated stages
    ~Prober();
    Prober(const Prober&) = delete;
    Prober& operator=(const Prober&) = delete;

    // `select`: a device index or part of its name, as BBHOST_GPU_DEVICE
    // (empty: that variable, else bbhost's own choice). `validate`: the
    // Khronos validation layer on, its messages in the verdicts.
    // `monolithic`: whole pipelines even where the device has libraries, as
    // BBHOST_PIPELINE_LIBRARY=0. Also sets the lifter's device switches
    // (gcn::set_runtime_sample_offsets, gcn::set_native_half_rtz), honouring
    // BBHOST_RUNTIME_OFFSETS=0 and BBHOST_HALF_NATIVE=0 as bbhost does.
    bool init(const std::string& select, bool validate, bool monolithic, std::string& error);
    const DeviceInfo& info() const;

    // Thread-safe: each probe creates and destroys its own objects.
    Verdict pixel(const std::vector<std::uint32_t>& spirv, std::uint32_t wave64_needs);
    Verdict vertex(const std::vector<std::uint32_t>& spirv);  // a vertex shader, or an LS on its own
    // A domain shader behind a generated vertex and hull shader. `modes`: the
    // module whose tessellation modes (domain, spacing, winding) the hull
    // declares - the lift's, since DXC's domain shader declares the domain only.
    Verdict domain(const std::vector<std::uint32_t>& spirv, const std::vector<std::uint32_t>& modes);
    // A game hull's native pipeline: its LS, hull and domain shader on a patch
    // list of `control_points`.
    Verdict hull_set(const std::vector<std::uint32_t>& ls, const std::vector<std::uint32_t>& hs, const std::vector<std::uint32_t>& ds,
                     std::uint32_t control_points);
    // A compute pipeline against a layout from the module's bindings.
    Verdict compute(const std::vector<std::uint32_t>& spirv);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace probe
