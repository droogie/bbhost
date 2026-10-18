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
//
// BBHOST_CAPTURE_DISPATCH=<program prefix>[,<prefix>...]: a compute dispatch
// the same way, for drawreplay's dispatch mode - its module, params block
// (the GDS counters' words included), images (a storage image's contents also
// after it ran), and the guest pages its program reaches before and after it
// ran. BBHOST_CAPTURE_DISPATCH_COUNT (default 1),
// BBHOST_CAPTURE_DISPATCH_EVERY (default 1), BBHOST_CAPTURE_DISPATCH_EACH (at
// most that many of one program; default no limit of its own), BBHOST_CAPTURE_MIN_FLIP and
// BBHOST_CAPTURE_DIR as for draws. An indirect dispatch whose arguments ask
// for no work, read after the settle, is passed over.
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

// A draw on the host's tessellator: the stages around the domain shader (which
// is stages[0], bound as the evaluation stage) and what they read.
//   attributes: the LS is the pipeline's vertex stage and its control point
//     travels as stage attributes; the control stage is the one made for the
//     hull shader's constant factors.
//   hull: the game's own hull shader is the control stage; the LS ran as a
//     compute pass into the LDS ring before the draw, which then draws one
//     patch a window (pass-through vertex stage, no index buffer).
struct CaptureTess {
    bool present = false;
    bool hull = false;
    bool attrs = false;
    std::uint32_t control_points = 1, attr_vec4s = 0, window = 0;
    bool quads = true, cw = true;
    int spacing = 0;
    float outer[4] = {}, inner[2] = {};
    std::vector<std::uint32_t> vertex_spirv, control_spirv;  // the modules the pipeline was built with
    const std::vector<std::uint32_t>* ls_words = nullptr;
    const std::vector<std::uint32_t>* ls_fetch_words = nullptr;
    const std::vector<std::uint32_t>* hs_words = nullptr;
    std::uint32_t ls_rsrc1 = 0, ls_rsrc2 = 0, hs_rsrc1 = 0, hs_rsrc2 = 0;
    std::uint32_t ls_user[16] = {}, hs_user[16] = {};
    // hull: the ring the LS pass wrote, read back before the draw.
    VkBuffer lds_buffer = VK_NULL_HANDLE;
    VkDeviceSize lds_offset = 0;
    std::uint32_t lds_bytes = 0;
    std::uint32_t tess_constants[12] = {};
    // The draw as the game made it, which a pipeline without the ring draws.
    std::uint64_t index_va = 0;
    std::uint32_t index_type = 0, count = 0, instances = 1, first_instance = 0, first_point = 0;
    std::int32_t base_vertex = 0;
    // The LS's elements as Vulkan vertex input (its fetch shader and the draw's
    // vertex table), for an LS lifted on vertex input; empty when the plan
    // does not cover them.
    bool ls_vertex_input = false;
    CaptureVertexInput ls_vinput;
};

struct CaptureDraw {
    std::string pipeline;
    bool lean = false;  // the no-fallback variant is bound
    CaptureTess tess;
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

// A compute dispatch (BBHOST_CAPTURE_DISPATCH).
struct CaptureDispatch {
    std::string pipeline;  // the program's name
    std::uint64_t flip = 0, dispatch_index = 0;
    const gcn::TranslateResult* meta = nullptr;
    const std::vector<std::uint32_t>* spirv = nullptr;
    const std::vector<std::uint32_t>* gcn_words = nullptr;
    gcn::StageParams params{};  // as written to the params ring, GDS counters included
    const StageImages* images = nullptr;
    std::uint32_t dim[3] = {1, 1, 1};
    std::uint32_t threads[3] = {1, 1, 1};
    std::uint64_t indirect_va = 0;  // read after the settle and captured as the direct dispatch it makes
    std::vector<std::uint64_t> counters;  // the GDS counter words bound (StageParams::gds_counter)
    json::Value translate;
};

class CaptureSession;
struct CaptureSessionDeleter {
    void operator()(CaptureSession* session) const;  // an unfinished session is logged as abandoned
};
using CapturePtr = std::unique_ptr<CaptureSession, CaptureSessionDeleter>;

// True while BBHOST_CAPTURE_DRAW is set and captures remain: the game's hull
// draws then take the LS pass, which a capture describes, instead of native
// pipelines (render.cpp).
bool capture_wanted();
// True when BBHOST_CAPTURE_DRAW selects this draw and captures remain.
// `lean_ready`: the pipeline's no-fallback variant is built, so the draw will most likely bind it (the `*` mode needs it).
// `indirect_va`: an indirect draw's arguments; one whose arguments count nothing now is not a candidate.
bool capture_candidate(const std::string& pipeline, std::uint32_t count, std::uint64_t flip, bool lean_ready, std::uint64_t indirect_va = 0);
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

// True when BBHOST_CAPTURE_DISPATCH selects this program and captures remain.
bool capture_dispatch_candidate(const std::string& program, std::uint64_t flip);
// After the descriptor set is written and before the dispatch (settled first
// with capture_settle_locked, so guest memory holds its inputs): copies its
// inputs and records its images' readbacks. Null when rejected.
CapturePtr capture_dispatch_begin_locked(CaptureDispatch&& dispatch);
// After the dispatch: runs it, reads the same guest pages again and writes the
// manifest and blobs.
void capture_dispatch_finish_locked(CapturePtr session);

}  // namespace gpu
