// BBHOST_CAPTURE_DRAW: a self-contained capture of one draw for
// tools/drawreplay: the bound
// shader modules, fixed and dynamic state, the bytes behind every descriptor,
// the guest pages its shaders can reach through the page table, and its colour
// and depth/stencil targets before and after the draw. Debug runs only:
// capturing settles the GPU before the draw and flushes after it.
//
//   BBHOST_CAPTURE_DRAW=<pipeline prefix>[,<prefix>...][%<min count>]   e.g. 8747a367+a22c7f71
//   BBHOST_CAPTURE_DRAW=*                                 the first draw of each pipeline whose no-fallback variant is
//                                                         ready, up to BBHOST_CAPTURE_COUNT (tools/lift_verify.py)
//   BBHOST_CAPTURE_DRAW=*ps                               the same, one draw for each pixel shader
//   BBHOST_CAPTURE_ARMED=1                                only while a test harness has armed it, for as many draws as
//                                                         it asked (BBHOST_TEST_REQUESTS "capture <n> [<dir>]",
//                                                         hle/video.cpp; tools/area_check.py)
//   BBHOST_CAPTURE_MIN_FLIP=<flip>                        first flip considered
//   BBHOST_CAPTURE_DIR=<dir>                              parent directory (default tmp/captures)
//   BBHOST_CAPTURE_COUNT=<n>                              draws to capture (default 1)
//   BBHOST_CAPTURE_EVERY=<n>                              one of every n matching draws (default 1)
#pragma once

#include "host/gpu_internal.h"
#include "replay/json.h"

#include <memory>
#include <string>
#include <vector>

namespace gpu {

struct CaptureStage {
    const gcn::TranslateResult* meta = nullptr;            // the bindings the descriptor set was written from
    const std::vector<std::uint32_t>* spirv = nullptr;     // the bound variant's module; null for no stage
    const std::vector<std::uint32_t>* gcn_words = nullptr;
    gcn::StageParams params{};                             // as written to the params ring
    const StageImages* images = nullptr;                   // what bind_stage_images wrote
    std::vector<VkDescriptorBufferInfo> buffers;           // parallel to meta->buffers
};

// A draw's Vulkan vertex input (BBHOST_VERTEX_INPUT): the elements its vertex
// shader converts, and the bindings and attributes over guest memory.
struct CaptureVertexInput {
    std::vector<gcn::VertexElement> elements;
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<std::uint64_t> binding_va;  // guest address of each binding's first byte
};

struct CaptureDraw {
    std::string pipeline;
    bool lean = false;  // the no-fallback variant is bound
    std::uint64_t flip = 0, draw_index = 0;
    const char* source = "packets";  // "token": drawn from a host-draw token (BBHOST_GX_NATIVE=2)
    CaptureStage stages[2];  // vertex, pixel
    const std::vector<std::uint32_t>* gs_spirv = nullptr;  // RECTLIST expansion, if any
    const std::vector<std::uint32_t>* fetch_words = nullptr;
    bool vertex_input = false;  // the vertex shader takes `vinput` instead of running the fetch shader
    CaptureVertexInput vinput;
    GfxFixedState fixed;
    RtImage* color[8] = {};
    std::uint32_t color_layer[8] = {};  // the layer of each colour target the draw renders into
    RtImage* depth = nullptr;
    std::uint32_t depth_layer = 0;
    std::uint32_t count = 0, instances = 1;
    std::int32_t base_vertex = 0;
    std::uint64_t index_va = 0;
    std::uint32_t index_type = 0;  // 0: 16-bit, 1: 32-bit
    std::uint64_t indirect_va = 0;
    json::Value registers;  // the decoded draw state, for provenance
    json::Value translate;  // translator options the bound modules were built with
};

struct CaptureDynamic {
    VkViewport viewport{};
    VkRect2D scissor{};
    VkExtent2D render_area{};
    bool depth_bounds_dynamic = false;
    float depth_bounds[2] = {0.0f, 1.0f};
    std::uint32_t stencil[6] = {};  // reference front/back, compare mask front/back, write mask front/back
    float blend_constants[4] = {};
};

class CaptureSession;
struct CaptureSessionDeleter {
    void operator()(CaptureSession* session) const;  // an unfinished session is logged as abandoned
};
using CapturePtr = std::unique_ptr<CaptureSession, CaptureSessionDeleter>;

// True when BBHOST_CAPTURE_DRAW selects this draw and captures remain.
// `lean_ready`: the pipeline's no-fallback variant is built, so the draw will most likely bind it (the `*` mode needs it).
bool capture_candidate(const std::string& pipeline, std::uint32_t count, std::uint64_t flip, bool lean_ready);
// For a candidate, before its descriptor sets exist: runs everything recorded
// so far, so guest memory and images hold what this draw will consume.
void capture_settle_locked();
// After the descriptor sets are written and before the render pass: checks the
// draw is capturable, copies its host-side inputs and records readbacks of its
// images and targets. Null when rejected; the reason is logged.
CapturePtr capture_begin_locked(CaptureDraw&& draw);
// After the draw command: reads the targets back, flushes and writes the
// manifest and blobs.
void capture_finish_locked(CapturePtr session, const CaptureDynamic& dynamic);

}  // namespace gpu
