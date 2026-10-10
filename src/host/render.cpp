// Graphics side of the host GPU executor: PM4 draw state -> Vulkan dynamic
// rendering. Render targets and depth buffers become Vulkan images keyed by
// their guest base address; vertex and pixel shaders go through the GCN
// translator (the vertex shader with its fetch shader inlined); index buffers
// come straight from the imported guest memory.
#include "core/image_file.h"
#include "core/portable.h"
#include "engine/gx_resources.h"
#include "engine/gx_state.h"
#include "host/gpu_internal.h"
#include "host/shader_patch.h"
#include "host/tess_lds.h"
#include "host/translation_cache.h"

#include "host/draw_capture.h"
#include "gcn/container.h"
#include "gcn/lift.h"
#include "gcn/wave.h"
#include "gcn/isa.h"
#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <deque>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <set>
#include <tuple>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace gpu {
bool dump_rt_locked(RtImage& r, const char* path);
bool rt_float_max_locked(RtImage& r, float* out3);
void apply_pending_clear(RtImage& r);
std::atomic<bool> g_dump_request{false};
// The folder the pending F12 capture writes into (host_gpu_request_dump).
std::mutex g_capture_mu;
std::string g_capture_dir;
namespace {

// A file of the pending F12 capture: <its folder>/f12-<flip>-<what>.<format>.
std::string capture_file(std::uint64_t flip, const char* what) {
    std::lock_guard<std::mutex> lk(g_capture_mu);
    return (g_capture_dir.empty() ? std::string("build") : g_capture_dir) + "/f12-" + std::to_string(flip) + "-" + what +
           host_gpu_capture_ext();
}

// ---- register decoding ------------------------------------------------------
struct CbInfo {
    std::uint32_t format, number_type, comp_swap;
    static CbInfo from(std::uint32_t info) { return {(info >> 2) & 0x1f, (info >> 8) & 7, (info >> 11) & 3}; }
};

VkFormat cb_format(const CbInfo& i) {
    const bool srgb = i.number_type == 6;
    const bool flt = i.number_type == 7;
    const bool uint_ = i.number_type == 4;
    const bool sint = i.number_type == 5;
    const bool snorm = i.number_type == 1;
    switch (i.format) {
    case 1: return uint_ ? VK_FORMAT_R8_UINT : sint ? VK_FORMAT_R8_SINT : snorm ? VK_FORMAT_R8_SNORM : VK_FORMAT_R8_UNORM;
    case 2: return flt ? VK_FORMAT_R16_SFLOAT : uint_ ? VK_FORMAT_R16_UINT : sint ? VK_FORMAT_R16_SINT : snorm ? VK_FORMAT_R16_SNORM : VK_FORMAT_R16_UNORM;
    case 3: return uint_ ? VK_FORMAT_R8G8_UINT : sint ? VK_FORMAT_R8G8_SINT : snorm ? VK_FORMAT_R8G8_SNORM : VK_FORMAT_R8G8_UNORM;
    case 4: return flt ? VK_FORMAT_R32_SFLOAT : sint ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
    case 5: return flt ? VK_FORMAT_R16G16_SFLOAT : uint_ ? VK_FORMAT_R16G16_UINT : sint ? VK_FORMAT_R16G16_SINT : snorm ? VK_FORMAT_R16G16_SNORM : VK_FORMAT_R16G16_UNORM;
    // 6 = 10_11_11 (R in the low 11 bits: Vulkan B10G11R11); 7 = 11_11_10
    // has no Vulkan layout and is treated the same (the game uses 6).
    case 6:
    case 7: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 8: return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case 9: return i.comp_swap == 1 ? VK_FORMAT_A2R10G10B10_UNORM_PACK32 : VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 10:
        if (i.comp_swap == 1 || i.comp_swap == 3) {
            return srgb ? VK_FORMAT_B8G8R8A8_SRGB : uint_ ? VK_FORMAT_B8G8R8A8_UINT : snorm ? VK_FORMAT_B8G8R8A8_SNORM : VK_FORMAT_B8G8R8A8_UNORM;
        }
        return srgb ? VK_FORMAT_R8G8B8A8_SRGB : uint_ ? VK_FORMAT_R8G8B8A8_UINT : sint ? VK_FORMAT_R8G8B8A8_SINT : snorm ? VK_FORMAT_R8G8B8A8_SNORM : VK_FORMAT_R8G8B8A8_UNORM;
    case 11: return flt ? VK_FORMAT_R32G32_SFLOAT : sint ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT;
    case 12: return flt ? VK_FORMAT_R16G16B16A16_SFLOAT : uint_ ? VK_FORMAT_R16G16B16A16_UINT : sint ? VK_FORMAT_R16G16B16A16_SINT : snorm ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R16G16B16A16_UNORM;
    case 14: return flt ? VK_FORMAT_R32G32B32A32_SFLOAT : sint ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
    case 16: return VK_FORMAT_R5G6B5_UNORM_PACK16;
    default: return VK_FORMAT_UNDEFINED;
    }
}

VkFormat db_format(std::uint32_t z_info, std::uint32_t stencil_info) {
    const std::uint32_t zf = z_info & 3;
    const bool stencil = (stencil_info & 1) != 0;
    if (zf == 0) return stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_UNDEFINED;
    if (zf == 1) return stencil ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D16_UNORM;
    return stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
}

VkCompareOp compare_op(std::uint32_t f) {
    static const VkCompareOp ops[8] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS, VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL,
                                       VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS};
    return ops[f & 7];
}
VkStencilOp stencil_op(std::uint32_t o) {
    switch (o & 0xf) {
    case 0: return VK_STENCIL_OP_KEEP;
    case 1: return VK_STENCIL_OP_ZERO;
    case 2: return VK_STENCIL_OP_KEEP;         // ONES: handled as replace with 0xff would need ref; keep
    case 3: return VK_STENCIL_OP_REPLACE;      // REPLACE_TEST
    case 4: return VK_STENCIL_OP_REPLACE;      // REPLACE_OP
    case 5: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 6: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 7: return VK_STENCIL_OP_INVERT;
    case 8: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 9: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
    }
}
VkBlendFactor blend_factor(std::uint32_t f) {
    switch (f & 0x1f) {
    case 0: return VK_BLEND_FACTOR_ZERO;
    case 1: return VK_BLEND_FACTOR_ONE;
    case 2: return VK_BLEND_FACTOR_SRC_COLOR;
    case 3: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 5: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_DST_ALPHA;
    case 7: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_DST_COLOR;
    case 9: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 13: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 14: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 15: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 16: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 17: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 18: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 19: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 20: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp blend_op(std::uint32_t f) {
    switch (f & 7) {
    case 0: return VK_BLEND_OP_ADD;
    case 1: return VK_BLEND_OP_SUBTRACT;
    case 2: return VK_BLEND_OP_MIN;
    case 3: return VK_BLEND_OP_MAX;
    case 4: return VK_BLEND_OP_REVERSE_SUBTRACT;
    default: return VK_BLEND_OP_ADD;
    }
}
bool primitive_topology(std::uint32_t prim, VkPrimitiveTopology& topo, bool& rect) {
    rect = false;
    switch (prim) {
    case 1: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; return true;
    case 2: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; return true;
    case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; return true;
    case 4: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; return true;
    case 5: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; return true;
    case 6: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; return true;
    // Gnm DI_PT_PATCH. Without LS/HS we still run the VS/PS so the game's
    // GPU-side buffers get written; drawing control points as a triangle list
    // is wrong-looking but dropping the draw left those buffers untouched.
    case 9: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; return true;
    case 10: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; return true;
    case 11: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY; return true;
    case 12: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; return true;
    case 13: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; return true;
    case 17: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; rect = true; return true;
    default: return false;
    }
}

// ---- images -----------------------------------------------------------------
std::map<std::uint64_t, RtImage> g_rts;   // by base VA

struct ShaderStage {
    VkShaderModule module = VK_NULL_HANDLE;
    // The translation (bindings; spirv is the module actually created): the
    // stage cache's when the stage came from it, read in place - copying its
    // SPIR-V and maps into every pipeline that took it was ~10% of the command
    // processor in the frame a fight's new pipelines arrive in - else this
    // pipeline's own (fresh()).
    std::shared_ptr<const gcn::TranslateResult> shared;
    gcn::TranslateResult own;
    bool lifted = false;  // BBHOST_DECOMP: spirv is the typed lift (gcn/lift.h)
    const gcn::TranslateResult& meta() const { return shared ? *shared : own; }
    gcn::TranslateResult& fresh() {  // a translation made for this stage
        shared.reset();
        return own;
    }
    template <typename Cached>
    void take(const std::shared_ptr<const Cached>& hit) {  // the stage cache's entry, kept alive by the pointer
        shared = std::shared_ptr<const gcn::TranslateResult>(hit, &hit->meta);
        own = gcn::TranslateResult{};
        lifted = hit->lifted;
    }
};
struct GfxPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    ShaderStage vs, ps, gs;
    // A tessellated draw's generated vertex and control stages. The
    // domain shader stays in `vs` - it binds what a vertex stage binds, so
    // every descriptor, prefetch and user-data path is the one already there -
    // and is only given the evaluation stage's bit when the pipeline is built.
    VkShaderModule tess_vs = VK_NULL_HANDLE, tess_tcs = VK_NULL_HANDLE;
    // With the control point carried as attributes the LS is the vertex stage
    // itself, and `tess_vs` above goes unused.
    ShaderStage tess_ls;
    bool tess_hw = false, tess_attrs = false;
    std::uint32_t tess_control_points = 1, tess_attr_vec4s = 8;
    // The VS and PS sets' layouts and the pipeline layout (stage_set_layout);
    // null with BBHOST_SET_LAYOUTS=0, which keeps the shared layout.
    VkDescriptorSetLayout set_layouts[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    // The no-fallback variant, once a draw whose constant buffers all bound
    // created it: bind setup and the table rebuild read it here instead of
    // looking it up by key on every draw.
    GfxPipeline* lean_variant = nullptr;
    bool lean_building = false;  // a worker is building its no-fallback variant (queue_lean_variant)
    bool building = false;       // a worker is building this pipeline; nothing reads it until handed over
    // Per stage: whether the translation lists the images and samplers
    // paths_for does, so the prefetch takes the words the pipeline key
    // resolved (key_stage_matches, checked on the first draw).
    bool paths_checked = false;
    bool paths_match[2] = {false, false};
    bool failed = false;
    bool bound_once = false;
    bool lean = false;         // translated without the page-table fallback (cb_no_fallback)
    bool layout_only = false;  // a fallback pipeline whose stages are its no-fallback ones, until a draw creates it
    bool library = false;      // linked from pipeline libraries: cull, front face, depth and stencil state are set per draw
    bool vs_formats_from_params = false;  // the vertex shader reads its vertex formats from its params (step 6c); captures record it
    std::uint32_t table_reads[2] = {~0u, ~0u};  // user-data slots the VS / PS read, or whose tables they read (table_slots_read)
    std::uint32_t user_reads[2] = {~0u, ~0u};   // user-data dwords the VS / PS touch at all, as data too (table_slots_read's `any`)
    std::uint64_t draws = 0;   // draws bound to it (under g.mu)
    std::uint64_t gx_draws = 0, token_draws = 0;  // of them: rendered from GX records, from host-draw tokens
    std::string name;
    // 1 for d3ca03f3+111fce32, 2 for 7d668276+e0305cef (the YEBIS probes in
    // draw_impl, which log once), 0 for the rest; -1 until a draw asks.
    std::int8_t yebis_probe = -1;
    // The optimized relink, queued at the pipeline's first draw
    // (queue_library_relink): its libraries, and the vertex library's
    // specialization - the elements' formats, the state it was made with.
    bool relink_pending = false;
    VkPipeline relink_libs[4] = {};
    bool relink_formats = false;
    std::uint32_t relink_vertex_formats[16] = {};
    VkPolygonMode relink_polygon = VK_POLYGON_MODE_FILL;
    bool relink_clamp = false;
    // The indexed loads' V# words 3 (StageParams::cb_w3, by stage and buffer)
    // the relink was specialized with, and once it is adopted the fast-linked
    // pipeline, which a draw whose words differ binds instead.
    std::uint32_t spec_w3[2][16] = {};
    std::uint32_t spec_stride[2][16] = {};  // and their strides (gcn::kCbStrideSpecId)
    std::uint16_t spec_w3_mask[2] = {};
    VkPipeline generic = VK_NULL_HANDLE;
    // The GPU profile's name (BBHOST_GPU_PROFILE) of the draws bound to it:
    // `name`, "~fallback" for the variant with the page-table paths, "~walks"
    // for one that still walks.
    std::string profile_name;
};
std::unordered_map<std::uint64_t, GfxPipeline> g_gfx;  // never erased, so references to pipelines stay valid
std::map<std::uint64_t, std::vector<std::uint32_t>> g_program_cache;  // code va -> words (validated by hash on use)

struct Pass {
    bool active = false;
    std::uint64_t key = 0;
    VkExtent2D extent{};
    std::uint64_t targets[9] = {};  // its attachments' bases (lazy pass barriers)
    int ntargets = 0;
    bool profiled = false;  // BBHOST_GPU_PROFILE=3: a timestamp pair is open around it
};
Pass g_pass;
const std::string* g_pass_profile_name = nullptr;  // the profile name of the draw beginning the next pass (BBHOST_GPU_PROFILE=3)

// ---- Lazy pass barriers (BBHOST_PASS_BARRIERS=lazy; off by default) ----
// Every pass end put a full barrier behind it, attachment writes made visible
// to everything after: ~45 a frame, ~2.5% of the Steam Deck's GPU frame and up
// to 7% of the RTX 4070's (priced by leaving them all out). Ended between two
// draws, a pass now owes its barrier and pays it when something needs it: a
// draw that samples a target written since the last barrier, or a pass whose
// targets were written or sampled since then. Everything other than a draw
// ends passes through render_end_pass_locked, which pays it first, as before.
// RTX 4070: 5% less GPU at the seed's spawn; the Deck: no change. The
// validation layer's synchronization checks find the same hazards either way
// (3000 flips, the world included; none from passes), 59% of the barriers left out.
// Off by default: with the targets compressed (BBHOST_RT_DCC) the Deck's world
// went white in blocks within a minute, and with either alone it did not - a
// read of a target this tracking does not see (a T# whose base is inside the
// target rather than at it, or an image bound another way), which the
// validation layer cannot see through the bindless set either.
struct PassHazards {
    bool owed = false;
    std::unordered_set<std::uint64_t> written;  // attachment bases written since the last barrier
    std::unordered_set<std::uint64_t> sampled;  // texture bases sampled since the last barrier
};
PassHazards g_hazards;
const bool g_lazy_barriers = [] {
    const char* e = std::getenv("BBHOST_PASS_BARRIERS");
    return e && std::strcmp(e, "lazy") == 0;
}();
std::atomic<std::uint64_t> g_barriers_paid{0}, g_barriers_skipped{0}, g_barriers_mid_pass{0};

std::atomic<std::uint64_t> g_rt_created{0}, g_untraced_draws{0}, g_clears{0}, g_rt_copies{0};
// Targets re-created in another format of the same size and texel size, their
// pixels carried over (rt_image), and re-created with their pixels lost; of
// those, the ones whose old image had last taken a fill that covers the new
// one, which starts with it.
std::atomic<std::uint64_t> g_rt_carried{0}, g_rt_lost{0}, g_rt_refilled{0}, g_rt_zeroed{0};

// A fill landed on a colour target's image (RtImage::fill_last): until a pass
// or a copy writes it, a target re-created over that memory starts with it.
void note_fill_last(RtImage& r, std::uint64_t va, std::size_t bytes, const float rgba[4]) {
    if (r.depth || !std::isfinite(rgba[0])) return;
    r.fill_last = true;
    std::memcpy(r.fill_rgba, rgba, sizeof(r.fill_rgba));
    r.fill_va = va;
    r.fill_bytes = bytes;
}
// The LS's user data built from its GX records, against the draws
// that had to keep what the command stream held.
std::atomic<std::uint64_t> g_tess_ls_user_gx{0}, g_tess_ls_user_fallback{0};
std::atomic<std::uint64_t> g_tess_ls_slot[16] = {}, g_tess_ls_slot_bad[16] = {}, g_tess_ls_table{0};
std::atomic<std::uint64_t> g_token_draws_seen{0}, g_token_with_gx{0}, g_token_tess{0};
// BBHOST_VPORT_SCISSOR=0 leaves viewport scissor 0 out of the draw scissor
// (for frame comparisons); g_vport_narrowed counts draws it narrows.
const bool g_vport_scissor = [] {
    const char* e = std::getenv("BBHOST_VPORT_SCISSOR");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_vport_narrowed{0};
// BBHOST_VERTEX_INPUT (on unless 0): GX draws in render mode take
// their vertex data through Vulkan vertex input, built from the fetch shader
// and the vertex table made at the call, instead of running the fetch shader:
// one binding per buffer and stride, as IASetVertexBuffers binds streams. `2`:
// one binding per element (for comparison). Captures record the vertex input.
// The vertex shaders then need no dispatcher loop, which halves cold pipeline
// compile time.
const int g_vertex_input = [] {
    const char* e = std::getenv("BBHOST_VERTEX_INPUT");
    if (!e) return 1;
    return e[0] == '1' ? 1 : e[0] == '2' ? 2 : 0;
}();
std::atomic<std::uint64_t> g_vertex_input_draws{0}, g_vertex_input_kept{0}, g_vertex_tables_read{0};
std::atomic<std::uint64_t> g_vertex_tables_with_input{0};  // vertex-input draws whose vertex shader still reads the table
// BBHOST_GX_SKIP_TABLES (on unless 0): GX draws leave out of the
// host ring the user-data tables and blocks their no-fallback variant does not
// read (table_slots_read), once that variant exists. A draw that ends up on the
// fallback variant gets them built and its params and storage buffers
// rewritten in place.
const bool g_skip_tables = [] {
    const char* e = std::getenv("BBHOST_GX_SKIP_TABLES");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_skip_draws{0}, g_tables_skipped{0}, g_skip_rebuilds{0}, g_skip_rebuilds_lean{0};
// BBHOST_GX_TOKEN_USER_DATA=register (checks, as in gx_trace.cpp): token draws
// keep the register file's unnamed user data, every pipeline gets a read mask
// (which roughly doubles pipeline creation time while loading), and the
// report counts token stages whose drawing variant reads such a slot.
const bool g_token_user_data_check = [] {
    const char* e = std::getenv("BBHOST_GX_TOKEN_USER_DATA");
    return e && std::strcmp(e, "register") == 0;
}();
// Draw stages whose prefetch took the pipeline key's binding words, and those
// that resolved them again (the translation lists other bindings).
std::atomic<std::uint64_t> g_prefetch_key_words{0}, g_prefetch_resolved{0};
// Tables and blocks GX draws still place in the host ring (their pipelines read
// them), in all and for draws from host-draw tokens, the latter by user-data slot.
std::atomic<std::uint64_t> g_gx_tables_placed{0}, g_token_tables_placed{0};
std::atomic<std::uint64_t> g_token_tables_by_slot[16] = {};
// Stages of draws from host-draw tokens where the no-fallback variant reads a
// user-data slot no descriptor names and the register file's value there is
// nonzero: a value from another draw that can reach the shader. By slot too.
std::atomic<std::uint64_t> g_token_stages_checked{0}, g_token_stale_stages{0};
std::atomic<std::uint64_t> g_token_stale_by_slot[16] = {};
// Token draw stages with any unnamed user-data slot not 0 in the register
// file, whatever the variant reads (fallback variants have no read mask).
std::atomic<std::uint64_t> g_token_unnamed_set_stages{0};
std::atomic<std::uint64_t> g_token_unnamed_set_by_slot[16] = {};
// Token draw stages checked against the variant that drew them (read masks for
// fallback variants too), those reading an unnamed slot not 0, and of those on
// fallback variants.
std::atomic<std::uint64_t> g_token_bound_checked{0}, g_token_bound_stale{0}, g_token_bound_stale_fallback{0};
// HTILE clears a draw from a token ran (to the values captured at the fill),
// those whose depth or stencil clear value was not 0, and clears by any draw
// where the captured values differ from the register file's at the draw (a
// check on the capture: 0 means the fill's values are the draw's).
std::atomic<std::uint64_t> g_token_htile_clears{0}, g_token_htile_clear_values{0}, g_htile_clear_value_mismatch{0};
// BBHOST_VS_DISPATCHER=1 (diagnostics): vertex shaders taking vertex input keep
// the dispatcher loop they had with the fetch shader inlined, to tell their
// code shape apart from the vertex fetch when timing.
const bool g_vs_dispatcher = [] {
    const char* e = std::getenv("BBHOST_VS_DISPATCHER");
    return e && e[0] == '1';
}();
// BBHOST_VS_INVARIANT=1: every vertex shader declares its position Invariant.
const bool g_vs_invariant = [] {
    const char* e = std::getenv("BBHOST_VS_INVARIANT");
    return e && e[0] == '1';
}();
// Rendering from GX: the tables a draw's user data points at
// (the fetch shader's vertex table, resource tables, extended blocks) live in
// a host-owned ring the GPU page table reaches, instead of the command
// stream's memory.
struct GxRing {
    std::mutex mu;
    std::uint64_t base = 0, size = 0, cursor = 0;
    bool failed = false;
};
GxRing g_gx_ring;
std::atomic<std::uint64_t> g_vertex_tables{0}, g_gx_user_stages{0}, g_gx_user_fallbacks{0};
// BBHOST_GX_COST=1: the renderer's GX work timed per
// draw, and sampled constant buffers checked against their contents at the
// call (gx_trace.cpp hashes them there).
const bool g_render_cost_enabled = [] {
    const char* e = std::getenv("BBHOST_GX_COST");
    return e && e[0] == '1';
}();
// The rest split host_gpu_draw into consecutive sections (RenderCostStamp):
// the register inputs; targets and state; programs, fetch shader and vertex
// input; the sampled constant-buffer check (cost mode only); the pipeline key;
// the pipeline lookup; the prefetch; capture and trace checks; bind setup; per
// stage the user data, constant buffers and params slot, set allocation and
// writes, and images; the descriptor update; the no-fallback variant, table
// rebuilds and capture; recording: the render pass, the pipeline and set
// binds, dynamic state, index and vertex buffers, and the draw.
enum RenderCost { kRenderCostCompare, kRenderCostUserData, kRenderCostVertexTable, kRenderCostRegisters, kRenderCostTargets,
                  kRenderCostPrograms, kRenderCostCbCheck, kRenderCostKey, kRenderCostLookup, kRenderCostPrefetch, kRenderCostChecks,
                  kRenderCostSetup, kRenderCostStageUser, kRenderCostBuffers, kRenderCostSets, kRenderCostImages, kRenderCostUpdate,
                  kRenderCostVariant, kRenderCostPass, kRenderCostBindCmds, kRenderCostDynamic, kRenderCostGeometry, kRenderCostRecord,
                  kRenderCosts };
std::atomic<std::uint64_t> g_render_cost_ns[kRenderCosts] = {}, g_render_cost_n[kRenderCosts] = {};
std::atomic<std::uint64_t> g_cb_check_stages{0}, g_cb_check_buffers{0}, g_cb_check_bytes{0}, g_cb_check_changed{0};
struct RenderCostTimer {
    RenderCost what;
    std::chrono::steady_clock::time_point start;
    explicit RenderCostTimer(RenderCost w)
        : what(w), start(g_render_cost_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~RenderCostTimer() {
        if (!g_render_cost_enabled) return;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        g_render_cost_ns[what].fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
        g_render_cost_n[what].fetch_add(1, std::memory_order_relaxed);
    }
};
// Consecutive sections of one function: to(k) books the time since the last
// stamp (or construction) to cost k.
struct RenderCostStamp {
    std::chrono::steady_clock::time_point t = g_render_cost_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    void to(RenderCost what) {
        if (!g_render_cost_enabled) return;
        const auto now = std::chrono::steady_clock::now();
        g_render_cost_ns[what].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count()),
                                         std::memory_order_relaxed);
        g_render_cost_n[what].fetch_add(1, std::memory_order_relaxed);
        t = now;
    }
    // Leaves the time since the last stamp out of every section.
    void skip() {
        if (g_render_cost_enabled) t = std::chrono::steady_clock::now();
    }
};
// How often a stage's descriptor set repeats, apart from its params slot (a new
// uniform-buffer offset every draw): the same contents as the stage's previous
// draw, or as a set the stage wrote earlier while the same command buffer
// recorded, which a cached set could serve.
std::atomic<std::uint64_t> g_set_stages{0}, g_set_same_as_last{0}, g_set_seen_in_cb{0};
bool note_stage_set(int st, const VkWriteDescriptorSet* w, std::size_t n) {
    std::uint64_t h = fnv1a(&st, sizeof(st));
    for (std::size_t i = 0; i < n; ++i) {
        h = fnv1a(&w[i].dstBinding, sizeof(w[i].dstBinding), h);
        h = fnv1a(&w[i].descriptorType, sizeof(w[i].descriptorType), h);
        if (w[i].dstBinding == gcn::kBindingParams) continue;
        if (w[i].pBufferInfo) h = fnv1a(w[i].pBufferInfo, sizeof(VkDescriptorBufferInfo), h);
        if (w[i].pImageInfo) {
            h = fnv1a(&w[i].pImageInfo->sampler, sizeof(VkSampler), h);
            h = fnv1a(&w[i].pImageInfo->imageView, sizeof(VkImageView), h);
            h = fnv1a(&w[i].pImageInfo->imageLayout, sizeof(VkImageLayout), h);
        }
    }
    static std::uint64_t last[2] = {};
    static std::uint64_t serial = ~0ull;
    static std::unordered_set<std::uint64_t> seen;
    if (serial != g.record_serial) {
        seen.clear();
        serial = g.record_serial;
    }
    g_set_stages.fetch_add(1, std::memory_order_relaxed);
    if (h == last[st]) g_set_same_as_last.fetch_add(1, std::memory_order_relaxed);
    const bool in_cb = !seen.insert(h).second;
    if (in_cb) g_set_seen_in_cb.fetch_add(1, std::memory_order_relaxed);
    last[st] = h;
    return in_cb;
}
// A stage's descriptor set, kept for the next draw that would write the
// same one. Measured (note_stage_params below): 25.7% of stages repeat both
// their params and their set contents inside one command buffer, and 86% of the
// sets that repeat have repeating params - so the params slot is deduplicated by
// content, rather than making the binding a dynamic uniform buffer, which would
// reach 29.7% and rebuild every pipeline layout.
//
// The identity is the bytes the set is built from: the params block, the
// resolved constant buffers, and the resolved images and samplers, under the
// same layout, the same stage metadata and the same command buffer. A reused
// set is only ever bound again, never rewritten - except by the fallback
// variant's table rebuild, which takes a fresh set instead (see below).
const bool g_set_reuse = [] {
    const char* e = std::getenv("BBHOST_SET_REUSE");
    return !e || std::atoi(e) != 0;
}();
std::atomic<std::uint64_t> g_set_reused{0}, g_set_built{0}, g_set_reuse_undone{0};

}  // namespace

// ---- descriptor sets cached by content (BBHOST_SET_CACHE; gpu_internal.h) ----
const bool g_set_cache = [] {
    const char* e = std::getenv("BBHOST_SET_CACHE");
    return !(e && e[0] == '0');
}();
bool set_cache_on() { return g_set_cache; }
bool cb_push_on() { return g_set_cache && g.has_push_descriptor; }
VkDescriptorType params_descriptor_type() {
    return g_set_cache ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}
VkDescriptorBufferInfo g_params_desc{};

namespace {
struct CachedSet {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::vector<std::uint64_t> material;  // what the key hashes, compared on a hit
};
std::unordered_map<std::uint64_t, CachedSet> g_cached_sets;                     // under g.mu
std::unordered_map<VkImageView, std::vector<std::uint64_t>> g_cached_set_views;  // view -> keys of the sets holding it
std::vector<VkDescriptorPool> g_set_cache_pools;
std::uint64_t g_sc_hits = 0, g_sc_misses = 0, g_sc_dropped = 0, g_sc_flushes = 0, g_sc_no_room = 0, g_sc_front_hits = 0;
constexpr std::size_t kCachedSetsMax = 24576;
constexpr std::uint32_t kCachePoolSets = 4096;
// A direct-mapped front for g_cached_sets, by the key's low bits: the map's
// bucket (a 64-bit division) and node were two cold misses for each of a
// draw's stages, 3.3% of a Steam Deck's command processor. The node a slot
// names stays put until it is erased (set_cache_drop clears the slot), and
// the material is still compared. BBHOST_SET_FRONT=0: the map alone.
struct SetFront {
    std::uint64_t key = 0;
    const CachedSet* set = nullptr;
};
constexpr std::size_t kSetFront = 4096;
SetFront g_set_front[kSetFront];
const bool g_set_front_on = [] {
    const char* e = std::getenv("BBHOST_SET_FRONT");
    return !(e && e[0] == '0');
}();

// A set leaves the cache; draws recorded with it may still be in flight, so
// it is freed when the current slot retires.
void set_cache_drop(std::unordered_map<std::uint64_t, CachedSet>::iterator it) {
    if (SetFront& f = g_set_front[it->first & (kSetFront - 1)]; f.set == &it->second) f = SetFront{};
    g.slots[g.slot].dead_sets.emplace_back(it->second.pool, it->second.set);
    g_cached_sets.erase(it);
    ++g_sc_dropped;
}

void set_cache_flush() {
    for (auto it = g_cached_sets.begin(); it != g_cached_sets.end(); it = g_cached_sets.begin()) set_cache_drop(it);
    g_cached_set_views.clear();
    ++g_sc_flushes;
}

VkDescriptorSet set_cache_alloc(VkDescriptorSetLayout layout, VkDescriptorPool& pool_out) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!g_set_cache_pools.empty()) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = g_set_cache_pools.back();
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(g.device, &ai, &set) == VK_SUCCESS) {
                pool_out = ai.descriptorPool;
                return set;
            }
            // Full (or fragmented): older pools get their room back as sets
            // retire; try them before making another.
            for (std::size_t k = 0; k + 1 < g_set_cache_pools.size(); ++k) {
                ai.descriptorPool = g_set_cache_pools[k];
                if (vkAllocateDescriptorSets(g.device, &ai, &set) == VK_SUCCESS) {
                    pool_out = ai.descriptorPool;
                    return set;
                }
            }
        }
        if (attempt || g_set_cache_pools.size() >= 8) break;
        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kCachePoolSets},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCachePoolSets * kMaxImages},
            {VK_DESCRIPTOR_TYPE_SAMPLER, kCachePoolSets * kMaxImages},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCachePoolSets * kMaxImages},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCachePoolSets * gcn::kMaxBuffers},
        };
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = kCachePoolSets;
        dpci.poolSizeCount = 5;
        dpci.pPoolSizes = sizes;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(g.device, &dpci, nullptr, &pool) != VK_SUCCESS) break;
        g_set_cache_pools.push_back(pool);
    }
    ++g_sc_no_room;
    return VK_NULL_HANDLE;
}

// The cached set with exactly this content, or null.
VkDescriptorSet set_cache_find(std::uint64_t key, const std::vector<std::uint64_t>& material) {
    SetFront& f = g_set_front[key & (kSetFront - 1)];
    if (f.set && f.key == key && f.set->material == material) {
        ++g_sc_hits;
        ++g_sc_front_hits;
        return f.set->set;
    }
    auto it = g_cached_sets.find(key);
    if (it == g_cached_sets.end() || it->second.material != material) return VK_NULL_HANDLE;
    ++g_sc_hits;
    if (g_set_front_on) f = SetFront{key, &it->second};
    return it->second.set;
}

void set_cache_insert(std::uint64_t key, const std::vector<std::uint64_t>& material, VkDescriptorSet set, VkDescriptorPool pool,
                      const std::vector<VkImageView>& views) {
    if (g_cached_sets.size() >= kCachedSetsMax) set_cache_flush();
    if (auto old = g_cached_sets.find(key); old != g_cached_sets.end()) set_cache_drop(old);  // a hash collision: the newer wins
    CachedSet& c = g_cached_sets[key];
    c.set = set;
    c.pool = pool;
    c.material = material;
    for (VkImageView v : views) {
        if (v) g_cached_set_views[v].push_back(key);
    }
    ++g_sc_misses;
}
}  // namespace

void set_cache_forget_view_locked(VkImageView view) {
    auto it = g_cached_set_views.find(view);
    if (it == g_cached_set_views.end()) return;
    for (const std::uint64_t key : it->second) {
        if (auto c = g_cached_sets.find(key); c != g_cached_sets.end()) set_cache_drop(c);
    }
    g_cached_set_views.erase(it);
}

std::string set_cache_report() {
    char buf[280];
    std::snprintf(buf, sizeof(buf), "set cache: %zu sets in %zu pools; %llu draws' stages hit (%llu in the front), %llu made; %llu dropped with a view, %llu flushes, %llu without room",
                  g_cached_sets.size(), g_set_cache_pools.size(), static_cast<unsigned long long>(g_sc_hits),
                  static_cast<unsigned long long>(g_sc_front_hits), static_cast<unsigned long long>(g_sc_misses),
                  static_cast<unsigned long long>(g_sc_dropped),
                  static_cast<unsigned long long>(g_sc_flushes), static_cast<unsigned long long>(g_sc_no_room));
    return buf;
}

namespace {
struct StageSetCache {
    std::uint64_t serial = ~0ull;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    const gcn::TranslateResult* meta = nullptr;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorBufferInfo ubi{};
    gcn::StageParams params{};
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<StageImages::Image> images;
    std::vector<VkSampler> samplers;
    bool valid = false;
};
// Sets repeat further back than the previous draw: 23.5% of stage sets have the
// contents of the stage's previous draw but 29.7% those of one written earlier
// in the command buffer, so the cache keeps a few. A miss costs almost nothing
// - the memcmp stops at the first differing byte of the params block - which is
// why this is a short linear scan rather than a hash and a table.
constexpr int kSetCacheWays = 8;
bool stage_set_matches(const StageSetCache& c, VkDescriptorSetLayout layout, const gcn::TranslateResult* meta,
                       const gcn::StageParams& params, const VkDescriptorBufferInfo* bufs, std::size_t nbufs,
                       const StageImages& imgs) {
    if (!c.valid || c.serial != g.record_serial || c.layout != layout || c.meta != meta) return false;
    if (c.buffers.size() != nbufs || c.images.size() != imgs.images.size() || c.samplers.size() != imgs.samplers.size()) return false;
    if (std::memcmp(&c.params, &params, sizeof(params)) != 0) return false;
    if (nbufs && std::memcmp(c.buffers.data(), bufs, nbufs * sizeof(VkDescriptorBufferInfo)) != 0) return false;
    if (!c.images.empty() && std::memcmp(c.images.data(), imgs.images.data(), c.images.size() * sizeof(StageImages::Image)) != 0) {
        return false;
    }
    if (!c.samplers.empty() && std::memcmp(c.samplers.data(), imgs.samplers.data(), c.samplers.size() * sizeof(VkSampler)) != 0) {
        return false;
    }
    return true;
}
void stage_set_remember(StageSetCache& c, VkDescriptorSetLayout layout, const gcn::TranslateResult* meta,
                        const gcn::StageParams& params, const VkDescriptorBufferInfo* bufs, std::size_t nbufs,
                        const StageImages& imgs, VkDescriptorSet set, const VkDescriptorBufferInfo& ubi) {
    c.serial = g.record_serial;
    c.layout = layout;
    c.meta = meta;
    c.set = set;
    c.ubi = ubi;
    c.params = params;
    c.buffers.assign(bufs, bufs + nbufs);
    c.images = imgs.images;
    c.samplers = imgs.samplers;
    c.valid = true;
}
// A cached descriptor set needs its params binding to stop moving.
// The set hash above excludes the params slot because every draw gets a fresh
// one; this asks whether the params *contents* repeat as well, which decides
// whether a slot can be deduplicated by content or the binding has to become a
// dynamic uniform buffer with a per-draw offset.
std::atomic<std::uint64_t> g_par_stages{0}, g_par_same_as_last{0}, g_par_seen_in_cb{0}, g_par_both{0};
// Tessellated draws emulated (host_gpu_draw), their patches, and draws refused.
std::atomic<bool> g_dump_patches{false};
std::uint64_t g_frames_watched = 0, g_frames_bright = 0;  // BBHOST_DUMP_ON_BRIGHT
std::atomic<std::uint64_t> g_tess_draws{0}, g_tess_patches{0}, g_tess_refused{0}, g_tess_vertices{0};
std::atomic<std::uint32_t> g_tess_max_patches{0}, g_tess_max_level{0};
// The draws the game's own hull shader ran for (tess_draw_one), and their patches.
std::atomic<std::uint64_t> g_tess_hull_draws{0}, g_tess_hull_patches{0};
// TranslateOptions::tess_lds_bound for every stage that reads or writes the
// buffer standing in for LDS: each access checked against its patch's window
// or the region the LS pass gave the draw, 0 read and the write dropped past
// it. An AMD card lost the device on the first hull draw of the Forbidden
// Woods without it (an RX 7600 XT, AMD's Windows driver: a read 4 GiB-odd
// past the ring). BBHOST_TESS_LDS_BOUND=0 leaves the accesses unchecked, as
// before 2026-10-08.
bool tess_lds_bound() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_TESS_LDS_BOUND");
        return !(e && e[0] == '0');
    }();
    return on;
}
void note_stage_params(int st, const gcn::StageParams& params, bool set_repeated) {
    const std::uint64_t h = fnv1a(&params, sizeof(params), fnv1a(&st, sizeof(st)));
    static std::uint64_t last[2] = {};
    static std::uint64_t serial = ~0ull;
    static std::unordered_set<std::uint64_t> seen;
    if (serial != g.record_serial) {
        seen.clear();
        serial = g.record_serial;
    }
    g_par_stages.fetch_add(1, std::memory_order_relaxed);
    if (h == last[st]) g_par_same_as_last.fetch_add(1, std::memory_order_relaxed);
    const bool in_cb = !seen.insert(h).second;
    if (in_cb) g_par_seen_in_cb.fetch_add(1, std::memory_order_relaxed);
    if (in_cb && set_repeated) g_par_both.fetch_add(1, std::memory_order_relaxed);
    last[st] = h;
}
// host_gpu_draw's sections in ns per draw (per stage for the stage sections):
// every 200,000 recorded draws for the draws since the last report, so loading
// screens' uploads stay out of in-game figures, and at exit for the whole run.
void report_render_split(bool whole_run) {
    static std::uint64_t last_ns[kRenderCosts] = {}, last_n[kRenderCosts] = {};
    std::uint64_t ns[kRenderCosts], n[kRenderCosts];
    for (int k = 0; k < kRenderCosts; ++k) {
        ns[k] = g_render_cost_ns[k].load(std::memory_order_relaxed);
        n[k] = g_render_cost_n[k].load(std::memory_order_relaxed);
    }
    if (!whole_run && n[kRenderCostRecord] - last_n[kRenderCostRecord] < 200000) return;
    const std::uint64_t sets[3] = {g_set_stages.load(std::memory_order_relaxed), g_set_same_as_last.load(std::memory_order_relaxed),
                                   g_set_seen_in_cb.load(std::memory_order_relaxed)};
    static std::uint64_t last_sets[3] = {};
    const auto sets_count = [&](int k) { return static_cast<unsigned long long>(whole_run ? sets[k] : sets[k] - last_sets[k]); };
    const auto count = [&](int k) { return static_cast<unsigned long long>(whole_run ? n[k] : n[k] - last_n[k]); };
    const auto per = [&](int k) {
        const std::uint64_t c = count(k);
        return static_cast<unsigned long long>(c ? (whole_run ? ns[k] : ns[k] - last_ns[k]) / c : 0);
    };
    host_log("render: draw split (%s), ns each: registers %llu (%llu), targets and state %llu, programs and vertex input %llu, "
             "constant-buffer check %llu, pipeline key %llu, pipeline lookup %llu, prefetch %llu, capture checks %llu, bind setup %llu; "
             "per stage: user data %llu, buffers %llu, sets %llu, images %llu (%llu stages); descriptor update %llu, "
             "variant and capture %llu; recording: render pass %llu, pipeline and sets %llu, dynamic state %llu, index and vertex "
             "buffers %llu, draw %llu (%llu draws); stage sets %llu, with the contents of the stage's previous draw %llu, of a set "
             "earlier in the command buffer %llu",
             whole_run ? "whole run" : "recent draws", per(kRenderCostRegisters), count(kRenderCostRegisters), per(kRenderCostTargets),
             per(kRenderCostPrograms), per(kRenderCostCbCheck), per(kRenderCostKey), per(kRenderCostLookup), per(kRenderCostPrefetch),
             per(kRenderCostChecks), per(kRenderCostSetup), per(kRenderCostStageUser), per(kRenderCostBuffers), per(kRenderCostSets),
             per(kRenderCostImages), count(kRenderCostImages), per(kRenderCostUpdate), per(kRenderCostVariant), per(kRenderCostPass),
             per(kRenderCostBindCmds), per(kRenderCostDynamic), per(kRenderCostGeometry), per(kRenderCostRecord),
             count(kRenderCostRecord), sets_count(0), sets_count(1), sets_count(2));
    if (!whole_run) {
        std::memcpy(last_ns, ns, sizeof(ns));
        std::memcpy(last_n, n, sizeof(n));
        std::memcpy(last_sets, sets, sizeof(sets));
    }
}
std::uint64_t render_cb_fnv(std::uint64_t va, std::uint64_t bytes) {
    std::uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va));
    for (std::uint64_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
const std::uint64_t g_order_target = [] { const char* e = std::getenv("BBHOST_ORDER"); return e ? std::strtoull(e, nullptr, 0) : 0ull; }();
std::atomic<int> g_order_logs{0};

}  // namespace
// The ring of recent draws (gpu_internal.h).
DrawRec g_draw_recs[kDrawRecs];
std::uint64_t g_draw_rec_next = 0;
std::uint32_t g_draw_dummies = 0;
thread_local const char* t_draw_pipeline = nullptr;  // the draw being resolved, for the glitch hunt's lines
// The draw being resolved came from a Scaleform (or wrapper) token: its
// constant buffers are bound from a copy taken now (resolve_stage_buffers).
thread_local bool t_snapshot_cbs = false;
std::atomic<std::uint64_t> g_sf_cb_snapshots{0};
thread_local char t_draw_origin[160] = "";             // and where it came from
void clear_image_locked(RtImage& r, const float rgba[4], std::uint32_t first_layer, std::uint32_t layer_count);  // below, outside the unnamed namespace

// RADV (GFX8 to GFX10.3, the Deck's VanGogh among them) compresses a colour
// target with DCC only when every format it may be viewed in has the
// target's channel count, order, size and float-ness; mutable with no list
// it never does. A target's views are the attachment's format and, for a T#
// that reads it through an integer format, that format (render_target_view).
std::uint32_t rt_view_formats(VkFormat format, VkFormat* out) {
    static const VkFormat kFamilies[][5] = {
        {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SNORM},
        {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UINT, VK_FORMAT_B8G8R8A8_SINT, VK_FORMAT_B8G8R8A8_SNORM},
        {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SRGB, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8_SNORM},
        {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SRGB, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R8_SNORM},
        {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SNORM},
        {VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SNORM},
        {VK_FORMAT_R16_UNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R16_SNORM},
        {VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT},
        {VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT},
        {VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT},
    };
    for (const auto& fam : kFamilies) {
        bool in = false;
        for (VkFormat f : fam) in |= f == format;
        if (!in) continue;
        std::uint32_t n = 0;
        out[n++] = format;
        for (VkFormat f : fam) {
            if (f != VK_FORMAT_UNDEFINED && f != format) out[n++] = f;
        }
        return n;
    }
    out[0] = format;
    return 1;
}

namespace {
// BBHOST_KEEP_INPUT=<pipeline prefix>:<image index>: the image that pass reads,
// copied *before* the pass runs, kept per flip. The watcher's ring keeps the
// display target, but by the time a flip comes round a pass's input has been
// cleared for the next frame, so a frame that went wrong cannot be traced back
// through it. Copying at the pass is the only moment the input is what the
// pass saw.
struct KeptInput {
    DevBuffer buf;
    std::uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::size_t bpp = 0;
    std::uint64_t flip = ~0ull, landed_at = 0;
    // Where in the draw ring this copy was taken. The flip counter advances on
    // the flip thread while draws are recorded on the command-processor
    // thread, so "the last such draw of flip N" is not reliably the last one
    // before flip N's display blit. The draw index is, and the blit's index is
    // in the same list the trigger writes.
    std::uint64_t at = 0;
    std::uint64_t base = 0;
    char pass[40] = {};
    bool filled = false;
};
// Up to four passes can be named at once, comma separated. A bisect over the
// ninety draws that write a target costs one run a step at six minutes a run;
// four probes in one run turns that from an afternoon into two runs.
constexpr int kKeepSpecs = 4;
constexpr int kKeptInputs = 2;  // a flip each, which is all the one-flip detection lag needs
KeptInput g_kept_inputs[kKeepSpecs][kKeptInputs];
struct KeepSpec {
    std::string name;
    std::size_t idx = 0;
    bool rt = false;
    // "#<n>": the nth draw of that pass in the flip, one-based. Most passes
    // that write a target run many times a frame and first run early, so
    // without this a probe cannot be placed in the late part of a frame at
    // all - and copying on every match is what the driver will not take.
    unsigned occurrence = 1;
    // A trailing "+" captures at the first draw into the same target *after*
    // that pass has run, which is the only way to see a single suspect draw's
    // effect: everything else here copies before a pass, so a pass that runs
    // once cannot be bracketed.
    bool after = false;
    // "@0x<base>#<k>" is the kth draw into that target this flip, whatever
    // pass it belongs to. Pass names are not a stable ruler: how many times a
    // Scaleform pass runs depends on what the menu is showing, so a probe
    // placed by occurrence lands somewhere different in the next run.
    std::uint64_t target = 0;
    // "#last": capture at every match, overwriting, so the slot ends up
    // holding the state before the *last* one of the flip. Bracketing a pass
    // that draws several times a frame needs both ends, and the other end
    // cannot be named in advance.
    bool last = false;
};
std::vector<KeepSpec> parse_keep_specs(const char* e) {
    std::vector<KeepSpec> out;
    std::string t(e ? e : "");
    std::size_t pos = 0;
    while (pos < t.size() && out.size() < kKeepSpecs) {
        const std::size_t comma = t.find(',', pos);
        std::string one = t.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? t.size() : comma + 1;
        if (one.empty()) continue;
        KeepSpec k;
        if (const std::size_t hash = one.find('#'); hash != std::string::npos) {
            if (one.compare(hash + 1, std::string::npos, "last") == 0) {
                k.last = true;
            } else {
                k.occurrence = std::max(1u, static_cast<unsigned>(std::strtoul(one.c_str() + hash + 1, nullptr, 10)));
            }
            one = one.substr(0, hash);
        }
        if (!one.empty() && one[0] == '@') {
            k.target = std::strtoull(one.c_str() + 1, nullptr, 0);
            k.rt = true;
            const std::size_t hash = one.find('#');
            k.occurrence = hash == std::string::npos
                               ? 1u
                               : std::max(1u, static_cast<unsigned>(std::strtoul(one.c_str() + hash + 1, nullptr, 10)));
            out.push_back(std::move(k));
            continue;
        }
        if (!one.empty() && one.back() == '+') {
            k.after = true;
            one.pop_back();
        }
        const std::size_t colon = one.find(':');
        k.name = colon == std::string::npos ? one : one.substr(0, colon);
        const std::string rest = colon == std::string::npos ? std::string() : one.substr(colon + 1);
        k.rt = rest == "rt";
        if (!k.rt && !rest.empty()) k.idx = static_cast<std::size_t>(std::strtoul(rest.c_str(), nullptr, 10));
        out.push_back(std::move(k));
    }
    return out;
}
const char* g_keep_input = std::getenv("BBHOST_KEEP_INPUT");
// BBHOST_KEEP_VERTS=<pipeline prefix>: that pass's vertex data, gathered every
// flip into a ring and written out by the change trigger, so the frame that
// went wrong can be read against the frame before it. All of it is guest
// memory read on the CPU - no readback, no broken render pass, none of what
// makes watching a target expensive.
struct KeptVerts {
    std::uint64_t flip = ~0ull;
    int draws = 0;
    std::string text;
    // The ranges those draws read, probed again at the end of the frame. At
    // record time nothing can be stale yet: a mirrored page is checked once
    // per recording and the guest write that strands it comes *after* the
    // draw that checked it. End of frame is where the question is answerable.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::vector<std::uint64_t> range_hash;  // the bytes when the draw was recorded
    bool probed = false;
};
constexpr int kKeptVerts = 4;
KeptVerts g_kept_verts[kKeptVerts];
const char* g_keep_verts = std::getenv("BBHOST_KEEP_VERTS");
// BBHOST_KEEP_VERTS_ROWS=0 keeps only a digest per buffer - a hash and a range
// - instead of every record. The full dump is a hundred and fifty formatted
// lines a draw, which is enough to change the run's timing: three runs in a
// row with it produced no flash at all to compare, while the same build
// without it reproduced the flash twice. Whatever this bug is, it is
// sensitive to that.
const int g_keep_verts_rows = [] {
    const char* e = std::getenv("BBHOST_KEEP_VERTS_ROWS");
    return e ? std::atoi(e) : 8;
}();
// The params as they go into the ring - the one piece of a draw's state the
// guest never sees, and built far from where the rest of the dump is taken.
// Three call sites allocate a slot, and which one a draw uses depends on its
// bindings, so all three report.
void keep_params(const std::string& name, const gcn::StageParams& params, int st) {
    if (glitch_watching_draw()) glitch_watch_params_locked(params, st);
    if (!g_keep_verts || st != 0 || name.find(g_keep_verts) != 0) return;
    KeptVerts& kv = g_kept_verts[hle_video_flip_count() % kKeptVerts];
    if (kv.flip != hle_video_flip_count() || kv.draws > 12) return;
    char line[512];
    // user_sgpr as the *shader* sees it, not as the register file left it: it
    // is rewritten after the draw's own state is read (build_gx_user_data),
    // and it is what every address in the shader is built from. The dump
    // compared the register file's copy and called them identical.
    std::snprintf(line, sizeof(line), "  params user");
    kv.text += line;
    for (int q = 0; q < 16; ++q) {
        std::snprintf(line, sizeof(line), " %08x", params.user_sgpr[q]);
        kv.text += line;
    }
    kv.text += "\n";
    std::snprintf(line, sizeof(line), "  params cb_valid=%08x l1=0x%llx bias", params.cb_valid,
                  static_cast<unsigned long long>(params.l1_table));
    kv.text += line;
    for (int q = 0; q < 8; ++q) {
        std::snprintf(line, sizeof(line), " %u", params.cb_bias_dw[q]);
        kv.text += line;
    }
    kv.text += " stride";
    for (int q = 0; q < 8; ++q) {
        std::snprintf(line, sizeof(line), " %u", params.cb_stride[q]);
        kv.text += line;
    }
    kv.text += " w3";
    for (int q = 0; q < 8; ++q) {
        std::snprintf(line, sizeof(line), " %08x", params.cb_w3[q]);
        kv.text += line;
    }
    kv.text += " vfmt";
    for (int q = 0; q < 8; ++q) {
        std::snprintf(line, sizeof(line), " %08x", params.vertex_formats[q]);
        kv.text += line;
    }
    kv.text += "\n";
}
std::atomic<std::uint64_t> g_cb_realigned{0}, g_cb_realigned_bytes{0};
const bool g_trace = [] {
    const char* e = std::getenv("BBHOST_TRACE_RENDER");
    return e && e[0] == '1';
}();
const int g_debug_ps_color = [] {
    const char* e = std::getenv("BBHOST_DEBUG_PS_COLOR");
    return e ? std::atoi(e) : 0;
}();
const long g_dump_frame = [] {
    const char* e = std::getenv("BBHOST_DUMP_FRAME");
    return e ? std::strtol(e, nullptr, 10) : -1;
}();

void image_barrier(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                   VkAccessFlags src, VkAccessFlags dst, std::uint32_t layers = 1) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, layers};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &b);
}

// Copies of render targets made by the engine's buffer-copy shader, keyed
// by the destination memory; sampled like the targets themselves.
std::map<std::uint64_t, RtImage> g_snapshots;
// For fills (find_rt_for_fill, render_htile_fill_locked): the largest target
// or snapshot made, in bytes, so a fill looks only at targets that start
// within that distance below it; and every address a depth target's HTILE
// was ever given, so a fill of other memory skips the walk for one. Both only
// grow: a stale entry costs a walk, never a missed target.
std::uint64_t g_max_rt_bytes = 0;
std::unordered_set<std::uint64_t> g_htile_bases;

void destroy_rt_image(RtImage& r) {
    invalidate_rt_views(r.base);
    defer_destroy_view(r.view);
    for (auto& [layer, view] : r.layer_views) defer_destroy_view(view);
    r.layer_views.clear();
    defer_destroy_image(r.image, r.memory);
}

// A target's memory comes from the image heap, as a texture's does
// (image_memory_alloc, gpu.cpp): an area's targets are made as it loads -
// ~300 a soak - and a driver allocation each was ~37 ms of the command
// processor in the frame the world first draws. BBHOST_RT_HEAP=0 gives each
// its own allocation again.
bool rt_image_memory(VkImage image, ImageMemory& out) {
    static const bool heap = [] {
        const char* e = std::getenv("BBHOST_RT_HEAP");
        return !(e && e[0] == '0');
    }();
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, image, &req);
    if (heap) {
        if (!image_memory_alloc(req, out)) return false;
    } else {
        out = ImageMemory{};
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &out.memory) != VK_SUCCESS) return false;
        out.size = req.size;
    }
    return vkBindImageMemory(g.device, image, out.memory, out.offset) == VK_SUCCESS;
}

// CB_COLOR*_SLICE / DB_DEPTH_SIZE only carry the *tile-padded* height, but a
// target sampled as a texture is addressed in normalized coordinates over the
// image's height: a 1080-row target padded to 1088 makes every v 1080/1088 too
// small. That is what squashed the movie and left an 8-row band of the clear
// colour along the bottom of the screen. The scissor carries the real height.
// Remember it per target so a later pass with a smaller scissor cannot resize
// the image - rt_image() recreates on a size change, and that costs a flush and
// a vkDeviceWaitIdle.
std::unordered_map<std::uint64_t, std::uint32_t> g_rt_real_height;
std::unordered_map<std::uint64_t, std::uint32_t> g_rt_real_width;

// Fill shaders run before the first draw that creates the image (and again
// after a size/format recreate). The translated compute writes guest memory
// the image never reads, so remember the value and apply it when the image
// appears.
struct PendingFill {
    float rgba[4];
    std::size_t bytes;
};
std::map<std::uint64_t, PendingFill> g_pending_clears;  // ordered: rt_image scans only overlapping fills
// Bumped at every insert and erase of g_pending_clears: a target that scanned
// them at this value has nothing more to take (RtImage::pending_seen).
std::uint64_t g_pending_gen = 1;
std::size_t g_pending_clear_max = 0;                     // largest pending fill, bounds that scan
std::uint64_t g_pending_scans = 0, g_pending_scan_steps = 0;  // apply_pending_clear's scans and the fills they visited (under g.mu)
// The word GX's depth-stencil clear pass (0x2ab5f50) fills HTILE with: ZMASK
// 0 (the tile is cleared) and the tile's z range, zmin at bits 4-17 and zmax
// at 18-31, both floor(depth * 16383) - so the clear depth is in the fill
// itself, to 14 bits, which is exact for the 0 and 1 the game clears to.
// DB_DEPTH_CLEAR, which the hardware expands a cleared tile to, is written by
// the output-merger commit of a later draw (0x2ad2790, bit 2 of its flags) and
// so lags the fill.
float htile_word_depth(std::uint32_t word) { return static_cast<float>(word >> 18) / 16383.0f; }
std::atomic<std::uint64_t> g_htile_fill_register_lag{0};
// HTILE fills that met no depth target yet, with the clear values captured at
// the fill (DB_DEPTH_CLEAR, DB_STENCIL_CLEAR).
struct PendingHtile {
    std::uint32_t word, stencil_clear;  // the HTILE word (htile_word_depth) and DB_STENCIL_CLEAR at the fill
};
std::unordered_map<std::uint64_t, PendingHtile> g_pending_htile;
// The fill's clear values become the target's pending clear; true when one was waiting.
bool take_pending_htile(std::uint64_t key, RtImage& r) {
    const auto it = g_pending_htile.find(key);
    if (it == g_pending_htile.end()) return false;
    r.htile_clear_pending = true;
    const float depth = htile_word_depth(it->second.word);
    std::memcpy(&r.htile_clear_depth, &depth, 4);
    r.htile_clear_stencil = it->second.stencil_clear;
    g_pending_htile.erase(it);
    return true;
}
std::atomic<std::uint64_t> g_fill_pending{0}, g_fill_applied{0};

// True when `v` rounded up to a tiling granularity is exactly `padded`: that is
// what makes a scissor edge believable as the target's real extent rather than
// a pass drawing into part of a larger one. Colour pitch pads to 128 or 256
// pixels (392 -> 512, 960 -> 1024, 1176 -> 1280) and height to 64 rows
// (1080 -> 1088).
bool plausible_extent(std::uint32_t v, std::uint32_t padded) {
    if (!v || v > padded) return false;
    for (std::uint32_t g : {64u, 128u, 256u}) {
        if (((v + g - 1) & ~(g - 1)) == padded) return true;
    }
    return false;
}

// Bumped whenever what rt_image() would answer for the same registers can
// change: a target or snapshot made, dropped or replaced, or an unpadded
// extent learned. target_image() reuses its last answer until then.
std::uint64_t g_rt_gen = 1;

std::uint32_t unpadded(std::unordered_map<std::uint64_t, std::uint32_t>& seen, std::uint64_t base, std::uint32_t padded,
                       std::uint32_t scissor_edge) {
    if (auto it = seen.find(base); it != seen.end() && plausible_extent(it->second, padded)) {
        // A pass cannot draw past its target: a scissor wider or taller than
        // what was learned says the memory holds another target now (the
        // cutscenes' glare pyramid leaves its sizes on addresses the game's
        // reuses at others).
        if (scissor_edge > it->second && scissor_edge != padded && plausible_extent(scissor_edge, padded)) {
            ++g_rt_gen;
            it->second = scissor_edge;
        }
        return it->second;
    }
    if (plausible_extent(scissor_edge, padded) && scissor_edge != padded) {
        ++g_rt_gen;
        seen[base] = scissor_edge;
        return scissor_edge;
    }
    return padded;
}

std::uint32_t unpadded_height(std::uint64_t base, std::uint32_t padded, std::uint32_t scissor_bottom) {
    return unpadded(g_rt_real_height, base, padded, scissor_bottom);
}

// CB_COLOR_PITCH pads the same way CB_COLOR_SLICE does, and a target sampled as
// a texture is addressed in normalized coordinates over the image's width: a
// 392-wide target held in a 512-wide image makes every u 392/512 too small.
// Down a bloom pyramid that compounds, which is why the composited frame was a
// smear along its left edge.
std::uint32_t unpadded_width(std::uint64_t base, std::uint32_t padded, std::uint32_t scissor_right) {
    return unpadded(g_rt_real_width, base, padded, scissor_right);
}

void copy_layers_locked(const RtImage& from, RtImage& to);

RtImage* rt_image(std::uint64_t base, VkFormat format, std::uint32_t width, std::uint32_t height, bool depth,
                  std::uint32_t layers = 1, std::uint64_t slice_bytes = 0) {
    // An image this one replaces whose last write was a fill (RtImage::
    // fill_last): the new image starts with that fill if it covers it. The
    // blood layers' small targets move between sizes as well as between sRGB
    // and UNORM; the game clears one and draws it in its new shape, and the
    // clear went to the old image - a huntsman came out metallic again
    // (2026-09-29, flip 2497) after the same-shape case was fixed.
    // BBHOST_RT_REFILL=0: as before, the new image starts as the image heap left it.
    static const bool refill_on = [] {
        const char* e = std::getenv("BBHOST_RT_REFILL");
        return !(e && e[0] == '0');
    }();
    RtImage refill;  // only its fill_* fields
    auto take_fill = [&](const RtImage& old) {
        if (!refill_on || depth || old.depth || !old.fill_last) return;
        refill.fill_last = true;
        std::memcpy(refill.fill_rgba, old.fill_rgba, sizeof(refill.fill_rgba));
        refill.fill_va = old.fill_va;
        refill.fill_bytes = old.fill_bytes;
    };
    if (auto st = g_snapshots.find(base); st != g_snapshots.end()) {
        RtImage& snap = st->second;
        if (!depth && layers == 1 && snap.initialised && snap.format == format && snap.width == width && snap.height == height) {
            // Copy dispatch wrote this address before any draw created a
            // colour target here (the blur ping-pong dest). Keep those
            // pixels: destroying the snapshot and allocating a new image
            // left 0x1508e0000 black while 0x16ab40000 held the clinic.
            if (auto it = g_rts.find(base); it != g_rts.end()) {
                destroy_rt_image(it->second);
                g_rts.erase(it);
                ++g_rt_gen;
                bump_view_epoch();
            }
            RtImage promoted = std::move(snap);
            g_snapshots.erase(st);
            ++g_rt_gen;
            bump_view_epoch();
            if (g_pending_clears.erase(base)) ++g_pending_gen;
            tex_event(base, static_cast<std::uint64_t>(width) * height * 4, "render target 0x%llx %ux%u format %d: promoted from a snapshot",
                      static_cast<unsigned long long>(base), width, height, format);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 16) {
                host_log("render: promoted snapshot 0x%llx to colour target %ux%u format %d",
                         static_cast<unsigned long long>(base), width, height, format);
            }
            ++g_rt_gen;
            RtImage& kept = (g_rts[base] = std::move(promoted));
            g_max_rt_bytes = std::max<std::uint64_t>(g_max_rt_bytes, rt_size_bytes(kept));
            return &kept;
        }
        g.rt_replacements.fetch_add(1);
        take_fill(st->second);
        destroy_rt_image(st->second);
        g_snapshots.erase(st);
        ++g_rt_gen;
        bump_view_epoch();
    }
    RtImage grown_from;  // a layered target that needs more layers; its layers are copied into the new image
    bool grow = false;
    bool carry = false;  // the same memory in another format of its texel size: grown_from's pixels carried over
    auto it = g_rts.find(base);
    if (it != g_rts.end()) {
        RtImage& r = it->second;
        // A draw into slice k of a layered target names SLICE_MAX = k (a cube's
        // faces are drawn one slice at a time), so fewer slices than the image
        // has is the same target.
        if (r.format == format && r.width == width && r.height == height && r.layers >= layers) {
            if (!r.slice_bytes) r.slice_bytes = slice_bytes;
            apply_pending_clear(r);
            return &r;
        }
        g.rt_replacements.fetch_add(1);
        if (r.format == format && r.width == width && r.height == height) {
            // More slices than before: the faces drawn so far survive. On a
            // cube's first frame this happens once per face.
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 24) {
                host_log("render: target 0x%llx grew from %u to %u layers (layers kept)", static_cast<unsigned long long>(base), r.layers,
                         layers);
            }
            grown_from = r;
            grow = true;
            if (!slice_bytes) slice_bytes = r.slice_bytes;
        } else if (!depth && !r.depth && r.initialised && r.width == width && r.height == height &&
                   format_bytes_per_pixel(r.format) == format_bytes_per_pixel(format)) {
            // The same memory read in another format of its texel size (sRGB
            // and UNORM, or an integer view): on the console the bits stay,
            // so they carry over. The game clears a target, then draws into
            // it in a format the target last had another way - the blood
            // layers' small targets move between sRGB and UNORM - and the
            // clear went to the old image: a new one started as whatever the
            // image heap held there, and a huntsman's layer came out as noise
            // (shiny, metallic) and the Hunter Pistol's magenta.
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 16) {
                host_log("render: target 0x%llx re-created: format %d -> %d %ux%u x%u (contents carried)", static_cast<unsigned long long>(base),
                         r.format, format, width, height, layers);
            }
            grown_from = r;
            carry = true;
            g_rt_carried.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Re-purposed memory: drop the old image. It outlives the work that
            // still references it by one submission, so no device idle is needed.
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 40) {
                host_log("render: target 0x%llx re-created: format %d %ux%u x%u -> %d %ux%u x%u (contents lost)",
                         static_cast<unsigned long long>(base), r.format, r.width, r.height, r.layers, format, width, height, layers);
            }
            g_rt_lost.fetch_add(1, std::memory_order_relaxed);
            take_fill(r);
            destroy_rt_image(r);
        }
        g_rts.erase(it);
        ++g_rt_gen;
        bump_view_epoch();
    }
    RtImage r;
    r.base = base;
    r.format = format;
    r.width = width;
    r.height = height;
    r.depth = depth;
    r.layers = std::max(layers, 1u);
    r.slice_bytes = slice_bytes;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    // Colour targets take views in other formats of their texel size: a T#
    // may read one through an integer format (render_target_view).
    r.mutable_format = !depth;
    ici.flags = r.mutable_format ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
    // Its view formats listed (rt_view_formats), so AMD's drivers keep it
    // compressed: on a Steam Deck at the seed's spawn, 1.5% less GPU time
    // (18.95/18.97 against 19.22/19.29 ms a frame). A view outside the list
    // reads a copy (render_target_view). BBHOST_RT_DCC=0: mutable with no list.
    static const bool listed = [] {
        const char* e = std::getenv("BBHOST_RT_DCC");
        return !(e && e[0] == '0');
    }();
    VkFormat view_formats[5];
    VkImageFormatListCreateInfo format_list{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    if (listed && !depth) {
        format_list.viewFormatCount = rt_view_formats(format, view_formats);
        format_list.pViewFormats = view_formats;
        ici.pNext = &format_list;
        r.format_listed = true;
    }
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = r.layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &r.image) != VK_SUCCESS) {
        host_log("render: image creation failed (format %d %ux%u)", format, width, height);
        if (grow || carry) destroy_rt_image(grown_from);
        return nullptr;
    }
    if (!rt_image_memory(r.image, r.memory)) {
        // Once VRAM is gone this fails every image; a line per failure wrote
        // a gigabyte of log in a three-instance run.
        static std::uint64_t failed = 0;
        ++failed;
        if (failed <= 10 || failed % 10000 == 0) {
            host_log("render: image memory failed (%llu so far)", static_cast<unsigned long long>(failed));
        }
        vkDestroyImage(g.device, r.image, nullptr);
        if (r.memory.memory) defer_destroy_image(VK_NULL_HANDLE, r.memory);
        if (grow || carry) destroy_rt_image(grown_from);
        return nullptr;
    }
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {static_cast<VkImageAspectFlags>(depth ? (VK_IMAGE_ASPECT_DEPTH_BIT | (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ? 0 : VK_IMAGE_ASPECT_STENCIL_BIT)) : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &r.view) != VK_SUCCESS) {
        host_log("render: image view failed");
        vkDestroyImage(g.device, r.image, nullptr);
        defer_destroy_image(VK_NULL_HANDLE, r.memory);
        if (grow || carry) destroy_rt_image(grown_from);
        return nullptr;
    }
    g_rt_created.fetch_add(1);
    tex_event(base, slice_bytes ? slice_bytes * r.layers : static_cast<std::uint64_t>(width) * height * 4,
              "%s target 0x%llx %ux%u x%u format %d created%s", depth ? "depth" : "render", static_cast<unsigned long long>(base),
              width, height, r.layers, format, grow ? " (grown)" : carry ? " (pixels carried from the old format)" : "");
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 24) {
        host_log("render: %s target 0x%llx %ux%u x%u format %d", depth ? "depth" : "color", static_cast<unsigned long long>(base),
                 width, height, r.layers, format);
    }
    ++g_rt_gen;
    RtImage& created = (g_rts[base] = r);
    g_max_rt_bytes = std::max<std::uint64_t>(g_max_rt_bytes, rt_size_bytes(created));
    bump_view_epoch();
    if (grow) {
        created.htile = grown_from.htile;
        created.htile_clear_pending = grown_from.htile_clear_pending;
        created.htile_clear_depth = grown_from.htile_clear_depth;
        created.htile_clear_stencil = grown_from.htile_clear_stencil;
        if (grown_from.initialised) copy_layers_locked(grown_from, created);
        if (grown_from.fill_last) note_fill_last(created, grown_from.fill_va, grown_from.fill_bytes, grown_from.fill_rgba);
        destroy_rt_image(grown_from);
    } else if (carry) {
        copy_layers_locked(grown_from, created);
        if (grown_from.fill_last) note_fill_last(created, grown_from.fill_va, grown_from.fill_bytes, grown_from.fill_rgba);
        destroy_rt_image(grown_from);
    } else if (!depth && textures_carry_into_target_locked(base, width, height, format_bytes_per_pixel(format), created.image)) {
        // A shader stored a texture of its size here first (a compute clear
        // before the first draw): the target starts with it.
        created.initialised = true;
    } else if (refill.fill_last && refill.fill_va <= base && refill.fill_va + refill.fill_bytes >= base + rt_size_bytes(created)) {
        // The old image took the game's clear of this memory and nothing
        // drew since: the clear was meant for this one.
        clear_image_locked(created, refill.fill_rgba, 0, ~0u);
        note_fill_last(created, refill.fill_va, refill.fill_bytes, refill.fill_rgba);
        g_rt_refilled.fetch_add(1, std::memory_order_relaxed);
        tex_event(base, rt_size_bytes(created), "render target 0x%llx %ux%u format %d starts with the fill its old image took (%g %g %g %g)",
                  static_cast<unsigned long long>(base), width, height, static_cast<int>(format), refill.fill_rgba[0], refill.fill_rgba[1],
                  refill.fill_rgba[2], refill.fill_rgba[3]);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("render: target 0x%llx %ux%u format %d re-created after a fill its old image took; it starts with that fill",
                     static_cast<unsigned long long>(base), width, height, static_cast<int>(format));
        }
    } else if (!depth) {
        // Nothing gave the new colour target its pixels: it starts cleared to
        // zero rather than as whatever the image heap held. In the Hunter's
        // Dream a glare level re-created 79x42 (read through a 49x42 T#) kept
        // values near the format's maximum (~63,000) and NaNs in the part no
        // pass draws, unchanged for minutes (2026-10-01, six F12 dumps): any
        // read past the drawn region would flood the glare pyramid. One clear
        // per new target, which is an area load's worth, not a frame's.
        // BBHOST_RT_POISON=1 (checks): magenta instead, so anything that reads
        // pixels no pass drew (a T# over a larger target, a lost clear) shows.
        // BBHOST_RT_ZERO=0: as the heap left it, as before.
        static const bool poison = [] {
            const char* e = std::getenv("BBHOST_RT_POISON");
            return e && e[0] == '1';
        }();
        static const bool zero = [] {
            const char* e = std::getenv("BBHOST_RT_ZERO");
            return !(e && e[0] == '0');
        }();
        if (poison) {
            const float magenta[4] = {1.0f, 0.0f, 1.0f, 1.0f};
            clear_image_locked(created, magenta, 0, ~0u);
        } else if (zero) {
            const float black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            clear_image_locked(created, black, 0, ~0u);
            g_rt_zeroed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    apply_pending_clear(created);
    return &created;
}

// rt_image() with the unpadded extents, as a draw asks for a target: the
// previous answer for this slot (colour 0-7, depth 8) while the registers
// and g_rt_gen are the same. Two hash lookups and an rt_image() per target
// per draw were ~7% of the command processor. A hit still applies what
// rt_image() applies on every use: the pending clears.
// A GX target's real extent from its view's texture
// (GpuDrawInputs::color_extent) instead of guessing it from the scissor
// (unpadded_width/height). BBHOST_GX_VIEW_EXTENTS=0 guesses as before; =2
// guesses, and counts and logs where the guess and the texture disagree.
const int g_view_extents = [] {
    const char* e = std::getenv("BBHOST_GX_VIEW_EXTENTS");
    return e ? std::atoi(e) : 1;
}();
std::uint64_t g_extent_used = 0, g_extent_guessed = 0, g_extent_checked = 0, g_extent_disagree = 0;  // under g.mu
std::uint64_t g_extent_from_gx = 0;  // under g.mu: extents for view-less draws from the GX texture at the target

RtImage* target_image(int slot, std::uint64_t base, VkFormat format, std::uint32_t padded_w, std::uint32_t padded_h,
                      std::uint32_t scissor_right, std::uint32_t scissor_bottom, bool depth, std::uint32_t layers,
                      std::uint64_t slice_bytes, std::uint32_t extent = 0) {
    struct Memo {
        std::uint64_t gen = 0, base = 0, slice_bytes = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t padded_w = 0, padded_h = 0, right = 0, bottom = 0, layers = 0, extent = 0;
        RtImage* rt = nullptr;
    };
    static Memo memo[9];
    Memo& m = memo[slot];
    if (m.rt && m.gen == g_rt_gen && m.base == base && m.format == format && m.padded_w == padded_w && m.padded_h == padded_h &&
        m.right == scissor_right && m.bottom == scissor_bottom && m.layers == layers && m.slice_bytes == slice_bytes && m.extent == extent) {
        apply_pending_clear(*m.rt);
        return m.rt;
    }
    std::uint32_t w = 0, h = 0;
    // No view to take it from (YEBIS's draws, the wrapper's): the live GX
    // texture that begins at the target's memory says its size - the game's
    // own description of what is there now. The learned sizes below are per
    // address, and an address the game reuses for a target of another size
    // kept the old one: the new-character cutscenes' glare pyramid (392x331
    // at its base) left sizes the game's (314x265) then drew into the corner
    // of, and the glare sampled the stale rest - magenta, until a reload.
    // BBHOST_RT_EXTENT_FROM_GX=0: the learned sizes only.
    static const bool extent_from_gx = [] {
        const char* e = std::getenv("BBHOST_RT_EXTENT_FROM_GX");
        return !(e && e[0] == '0');
    }();
    if (!extent && extent_from_gx && !depth) {
        std::uint32_t t[8];
        if (gx_resource_tsharp(base, t)) {
            const std::uint32_t tw = (t[2] & 0x3fff) + 1, th = ((t[2] >> 14) & 0x3fff) + 1;
            if (tw <= padded_w && th <= padded_h) {
                extent = (tw << 16) | th;
                ++g_extent_from_gx;
            }
        }
    }
    const std::uint32_t ew = extent >> 16, eh = extent & 0xffff;
    // The texture's extent stands when it fits the padded one it came padded to.
    const bool use_extent = g_view_extents && extent && ew <= padded_w && eh <= padded_h;
    if (!use_extent || g_view_extents == 2) {
        w = unpadded_width(base, padded_w, scissor_right);
        h = unpadded_height(base, padded_h, scissor_bottom);
    }
    if (use_extent && g_view_extents == 2) ++g_extent_checked;
    if (use_extent && g_view_extents == 2 && (w != ew || h != eh)) {
        if (g_extent_disagree++ < 24) {
            host_log("render: %s target 0x%llx: the scissor makes it %ux%u (padded %ux%u), its view's texture %ux%u",
                     depth ? "depth" : "colour", static_cast<unsigned long long>(base), w, h, padded_w, padded_h, ew, eh);
        }
    }
    if (use_extent && g_view_extents == 1) {
        w = ew;
        h = eh;
        ++g_extent_used;
        // What the scissor would have to teach is known now: draws with no
        // view (the wrapper's, YEBIS's) and the texture cache size this
        // target the same way instead of making it again at another size.
        std::uint32_t& known_w = g_rt_real_width[base];
        std::uint32_t& known_h = g_rt_real_height[base];
        if (known_w != ew || known_h != eh) {
            known_w = ew;
            known_h = eh;
            ++g_rt_gen;
        }
    } else {
        ++g_extent_guessed;
    }
    RtImage* rt = rt_image(base, format, w, h, depth, layers, slice_bytes);
    m = Memo{g_rt_gen, base, slice_bytes, format, padded_w, padded_h, scissor_right, scissor_bottom, layers, extent, rt};
    return rt;
}

VkImageAspectFlags aspect_of(const RtImage& r) {
    if (!r.depth) return VK_IMAGE_ASPECT_COLOR_BIT;
    return VK_IMAGE_ASPECT_DEPTH_BIT | (r.format == VK_FORMAT_D32_SFLOAT || r.format == VK_FORMAT_D16_UNORM ? 0 : VK_IMAGE_ASPECT_STENCIL_BIT);
}

void ensure_initialised(RtImage& r) {
    if (r.initialised) return;
    begin_recording_locked();
    image_barrier(g_cmd(), r.image, aspect_of(r), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, r.layers);
    r.initialised = true;
}

// The attachment view of one layer of a layered target (one cube face); layer 0
// is the target's own view.
VkImageView attachment_view(RtImage& r, std::uint32_t layer) {
    if (layer == 0 || layer >= r.layers) return r.view;
    if (auto it = r.layer_views.find(layer); it != r.layer_views.end()) return it->second;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = r.format;
    vci.subresourceRange = {aspect_of(r), 0, 1, layer, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(g.device, &vci, nullptr, &view) != VK_SUCCESS) return r.view;
    return r.layer_views[layer] = view;
}

// ---- shaders ------------------------------------------------------------------
// Shader programs by code address. Extraction scans memory for the Sony
// footer and is far too slow per draw; the first 16 dwords of the code
// double as the validity check when the game reuses the memory.
struct CachedProgram {
    std::vector<std::uint32_t> words;
    std::string name;
    std::uint64_t hash = 0;
    int fetch_sgpr = -2;  // -2 unknown, -1 none
    std::uint32_t head[16];
    // A program a setting patches (host/shader_patch.h) is read again when the
    // settings change what it should be; -1 for every other program.
    std::int64_t patch_mask = -1;
};
std::unordered_map<std::uint64_t, CachedProgram> g_programs;

CachedProgram* program_at(std::uint64_t va) {
    if (!hle_kernel_va_mapped(va, 64)) return nullptr;
    const auto* p = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
    // The entry an address found last, in front of the map (entries are
    // updated in place, never erased, so the pointer stays good).
    struct Memo {
        std::uint64_t va = 0;
        CachedProgram* cp = nullptr;
    };
    static Memo memo[64];  // under g.mu
    Memo& m = memo[(va >> 8) & 63];
    CachedProgram* hit = m.va == va ? m.cp : nullptr;
    if (!hit) {
        auto it = g_programs.find(va);
        if (it != g_programs.end()) hit = &it->second;
    }
    if (hit && std::memcmp(hit->head, p, 64) == 0 && (hit->patch_mask < 0 || hit->patch_mask == shader_patch_mask())) {
        m = {va, hit};
        return hit;
    }
    CachedProgram cp;
    const std::uint32_t mask = shader_patch_mask();
    if (!extract_program(va, cp.words, cp.name)) return nullptr;
    if (shader_patch_touches(cp.name)) cp.patch_mask = mask;
    cp.hash = fnv1a(cp.words.data(), cp.words.size() * 4);
    std::memcpy(cp.head, p, 64);
    CachedProgram* made = &(g_programs[va] = cp);
    m = {va, made};
    return made;
}

// The fetch shader ends with s_setpc_b64.
// A fetch shader's words and their hash, cached by address. The pointer stays
// valid until the next lookup of the same address finds other words there, so a
// draw can use it throughout.
struct FetchProgram {
    std::vector<std::uint32_t> words;
    std::uint64_t hash = 0;
};
const FetchProgram* fetch_program_at(std::uint64_t va) {
    static std::unordered_map<std::uint64_t, FetchProgram> cache;
    if (!hle_kernel_va_mapped(va, 4)) return nullptr;
    auto cached = cache.find(va);
    if (cached != cache.end() && hle_kernel_va_mapped(va, cached->second.words.size() * 4) &&
        std::memcmp(cached->second.words.data(), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)),
                    cached->second.words.size() * 4) == 0) {
        return &cached->second;
    }
    const auto* p = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
    std::size_t limit = 0;
    while (limit < 16384 && hle_kernel_va_mapped(va + limit * 4, 4)) limit += 1024;
    // One instruction at a time up to the return: a fetch shader is a few dozen
    // dwords, and decoding the whole mapped window after it (up to 16,384
    // dwords) was ~18% of the command processor in the frame where a fight's
    // new input layouts arrive.
    gcn::Inst in;
    for (std::size_t i = 0; i < limit && gcn::decode_one(p, limit, i, in); i += in.size) {
        if (in.enc == gcn::Enc::SOP1 && in.op == 32) {  // s_setpc_b64
            FetchProgram& fp = cache[va];
            fp.words.assign(p, p + i + in.size);
            fp.hash = fnv1a(fp.words.data(), fp.words.size() * 4);
            return &fp;
        }
        if (in.enc == gcn::Enc::SOPP && in.op == 1) break;  // s_endpgm
    }
    return nullptr;
}

// A binding the plan reads from a shader-resource view's record
// finds its view by the view's id (texture_view_by_id); BBHOST_VIEW_BY_ID=0
// by the T#'s words as before, =2 both, counting disagreements.
const int g_view_by_id = [] {
    const char* e = std::getenv("BBHOST_VIEW_BY_ID");
    return e ? std::atoi(e) : 1;
}();
std::uint64_t g_view_by_id_disagree = 0;  // under g.mu

// Programs by object. A GX draw carries its shader and input-layout
// objects' ids (engine/gx_resources.h), which stand for what the object holds
// for its life, so the program and the fetch shader are found once per object
// instead of by code address at every draw (program_at checks 64 bytes of
// code, fetch_program_at all of the fetch shader's). BBHOST_GX_PROGRAM_IDS=0
// finds them by address as before; =2 does both and counts disagreements.
const int g_program_ids = [] {
    const char* e = std::getenv("BBHOST_GX_PROGRAM_IDS");
    return e ? std::atoi(e) : 1;
}();
// A fetch shader's elements (plan_vertex_input: parse_fetch_elements).
struct FetchElement {
    std::uint8_t slot = 0, vdata = 0, count = 0;
    std::uint32_t offset = 0;  // bytes added to the record's address
};
struct ParsedFetch {
    bool ok = false;
    std::vector<FetchElement> elements;
};
struct ProgramById {
    std::uint32_t id = 0;
    std::uint64_t va = 0;
    CachedProgram* cp = nullptr;
    std::int64_t patch_mask = -1;  // the program's, when a setting patches it: a change re-reads it
};
struct FetchById {
    std::uint32_t id = 0;
    std::uint64_t va = 0;
    const FetchProgram* fetch = nullptr;
    const ParsedFetch* parsed = nullptr;  // plan_vertex_input's parse, for `vtx_ud`
    std::uint8_t vtx_ud = 0xff;
};
constexpr std::uint32_t kById = 4096;
ProgramById g_program_by_id[kById];  // under g.mu, direct-mapped by id
FetchById g_fetch_by_id[kById];
std::uint64_t g_by_id_hits = 0, g_by_id_misses = 0, g_by_id_disagree = 0, g_by_id_none = 0, g_by_id_unseen = 0;  // under g.mu

CachedProgram* program_for(std::uint32_t id, std::uint64_t va) {
    if (!id || !g_program_ids) return program_at(va);
    ProgramById& e = g_program_by_id[id & (kById - 1)];
    CachedProgram* cp = nullptr;
    if (e.id == id && e.va == va && (e.patch_mask < 0 || e.patch_mask == shader_patch_mask())) {
        ++g_by_id_hits;
        cp = e.cp;
    } else {
        ++g_by_id_misses;
        cp = program_at(va);
        if (cp) e = {id, va, cp, cp->patch_mask};
        return cp;
    }
    if (g_program_ids == 2 && program_at(va) != cp) {
        if (g_by_id_disagree++ < 8) host_log("render: program object %u at 0x%llx: its id's program is not the address's", id, static_cast<unsigned long long>(va));
        return program_at(va);
    }
    return cp;
}

FetchById* fetch_for(std::uint32_t id, std::uint64_t va, const FetchProgram*& out) {
    if (!id || !g_program_ids) {
        out = fetch_program_at(va);
        return nullptr;
    }
    FetchById& e = g_fetch_by_id[id & (kById - 1)];
    if (e.id == id && e.va == va && e.fetch) {
        ++g_by_id_hits;
        out = e.fetch;
        if (g_program_ids == 2 && fetch_program_at(va) != out) {
            if (g_by_id_disagree++ < 8) host_log("render: input layout %u: its id's fetch shader at 0x%llx is not the address's", id, static_cast<unsigned long long>(va));
            out = fetch_program_at(va);
            e = FetchById{};
            return nullptr;
        }
        return &e;
    }
    ++g_by_id_misses;
    out = fetch_program_at(va);
    if (!out) return nullptr;
    e = {id, va, out, nullptr, 0xff};
    return &e;
}

int fetch_user_sgpr(const gcn::Program& vs) {
    for (const gcn::Inst& in : vs.insts) {
        if (in.enc == gcn::Enc::SOP1 && in.op == 33) {  // s_swappc_b64
            return in.src0 < 104 ? in.src0 : -1;
        }
    }
    return -1;
}

// BBHOST_DUMP_SPIRV=1: every module to build/spv/<name>.spv (for
// spirv-val / spirv-dis when a driver compiler falls over) and a log line
// before each pipeline is created.
const bool g_dump_spirv = [] {
    const char* e = std::getenv("BBHOST_DUMP_SPIRV");
    return e && e[0] == '1';
}();
void dump_spirv(const std::string& name, const std::vector<std::uint32_t>& spirv) {
    if (!g_dump_spirv) return;
    host_mkdir("build/spv");
    const std::string path = "build/spv/" + name + ".spv";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(spirv.data(), 4, spirv.size(), f);
        std::fclose(f);
    }
}

// With BBHOST_DUMP_SPIRV: a module's storage-buffer bindings next to it (index,
// binding, pointer table or V#, path, dwords read), so an offline pass can
// tell which of them survive dead-code elimination.
void dump_buffers(const std::string& name, const gcn::TranslateResult& meta, std::uint32_t tables = ~0u) {
    if (!g_dump_spirv || meta.buffers.empty()) return;
    const std::string path = "build/spv/" + name + ".buffers";
    if (FILE* f = std::fopen(path.c_str(), "w")) {
        std::fprintf(f, "# tables %08x\n", tables);  // table_slots_read
        for (std::size_t i = 0; i < meta.buffers.size(); ++i) {
            const gcn::BufferBinding& b = meta.buffers[i];
            std::fprintf(f, "%zu %u %s %s %u\n", i, b.binding, b.pointer ? "pointer" : "vsharp", b.path.str().c_str(), b.max_dw);
        }
        std::fclose(f);
    }
}

bool make_module(const std::vector<std::uint32_t>& spirv, VkShaderModule& out) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * 4;
    smci.pCode = spirv.data();
    return vkCreateShaderModule(g.device, &smci, nullptr, &out) == VK_SUCCESS;
}

// BBHOST_DECOMP: which shaders run as typed lifts (gcn/lift.h) in their
// no-fallback pipelines. "all" (the default since 2026-09-26) is every pixel
// shader, and every vertex shader on vertex input, that the lifter accepts -
// a rejection keeps the translation, so this can only change a shader the
// lifter reproduced, and every accepted lift replayed byte-identically on the
// world captures (tools/lift_verify.py). "0" turns it off, "1" is the
// pilot pixel shader a22c7f71, anything else is a comma-separated list of VS
// and PS hashes.
//
// It is the source form of a shader, which is what a mod would edit, and it
// compiles about 15% faster for about half the SPIR-V. It does not make a
// frame measurably faster: four alternating soaks put the arms inside the
// run-to-run spread.
bool decomp_selects(const std::string& ps_name) {
    static const std::pair<bool, std::set<std::string>> selection = [] {
        std::pair<bool, std::set<std::string>> s{true, {}};
        const char* e = std::getenv("BBHOST_DECOMP");
        if (!e || !*e || std::strcmp(e, "all") == 0) return s;
        s.first = false;
        if (std::strcmp(e, "0") == 0) return s;
        if (std::strcmp(e, "1") == 0) {
            s.second.insert("a22c7f71");
            return s;
        }
        std::string list(e);
        for (std::size_t at = 0; at <= list.size();) {
            const std::size_t comma = std::min(list.find(',', at), list.size());
            if (comma > at) s.second.insert(list.substr(at, comma - at));
            at = comma + 1;
        }
        return s;
    }();
    return selection.first || selection.second.count(ps_name) != 0;
}

// How a draw bound to `p` runs its pixel shader (GpuStats::draws_ps): only the
// no-fallback variant ever carries a lift.
int draw_ps_kind(const GfxPipeline& p) {
    if (p.ps.lifted) return kDrawPsLifted;
    if (!p.ps.module && p.ps.meta().spirv.empty()) return kDrawPsNone;
    return p.lean ? kDrawPsTranslated : kDrawPsFallback;
}

struct DrawState {
    std::uint64_t vs_va = 0, ps_va = 0, fetch_va = 0;
    std::uint32_t vs_rsrc1 = 0, vs_rsrc2 = 0, ps_rsrc1 = 0, ps_rsrc2 = 0;
    std::uint32_t prim = 0;
    // A tessellated draw's second pass (draw_tessellated): the vertex shader is
    // the domain shader, each patch a domain_level x domain_level grid of quads.
    std::uint32_t domain_level = 0;
    // The same patches through the host's own tessellator instead -
    // a patch list, a generated vertex and control stage, and the domain
    // shader as the evaluation stage. The factors are the hull shader's, read
    // out of it; the level above is then only what the emulated path would
    // have drawn, and is not used.
    bool tess_hw = false;
    std::uint32_t tess_control_points = 1;
    float tess_outer[4] = {1, 1, 1, 1}, tess_inner[2] = {1, 1};
    // And with the control point carried as stage attributes,
    // the LS is the pipeline's own vertex stage - no compute pass, so no
    // render-pass break for it and no buffer standing in for LDS. Its user
    // data rides in the vertex-formats slot, which a patch draw has no use
    // for; the evaluation stage keeps the usual one.
    bool tess_attrs = false;
    std::uint32_t tess_attr_vec4s = 8;
    std::uint64_t ls_va = 0, ls_fetch_va = 0;
    std::uint32_t ls_rsrc1 = 0, ls_rsrc2 = 0;
    std::uint32_t ls_user[16] = {};
    // The game's own hull shader as the control stage (TessDraw::hull): the
    // Forbidden Woods' meshes, triangle patches of three control points with
    // factors the hull works out. Its user data rides in the vertex-formats
    // slot as the LS's does above (the two never meet: this path keeps the LS
    // in its compute pass).
    bool tess_hull = false;
    std::uint64_t hs_va = 0;
    std::uint32_t hs_rsrc1 = 0, hs_rsrc2 = 0, tess_window = 0;
    bool tess_quads = true, tess_cw = true;
    int tess_spacing = 0;
    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool rect = false;
    // attachments
    RtImage* color[8] = {};
    std::uint32_t color_mask[8] = {};
    std::uint32_t color_layer[8] = {};  // CB_COLOR*_VIEW SLICE_START: the layer drawn into
    RtImage* depth = nullptr;
    std::uint32_t depth_layer = 0;  // DB_DEPTH_VIEW SLICE_START
    std::uint32_t depth_control = 0, stencil_control = 0, stencil_ref = 0, stencil_ref_bf = 0;
    float depth_bounds_min = 0.0f, depth_bounds_max = 1.0f;  // DB_DEPTH_BOUNDS_MIN/MAX (dynamic)
    std::uint32_t blend[8] = {};
    std::uint32_t su_sc_mode = 0, clip_cntl = 0, vte_cntl = 0;
    std::uint32_t ps_input_cntl[32] = {};
    std::uint32_t ps_input_ena = 0, ps_in_control = 0, vs_out_cntl = 0, cb_shader_mask = 0, ps_col_format = 0;
    std::uint32_t ps_user[16] = {}, vs_user[16] = {};
    float vport[6] = {};
    VkRect2D scissor{};
    std::uint32_t render_control = 0, depth_clear = 0, stencil_clear = 0;
    // The draw's blend, depth-stencil and rasterizer state as
    // the game described them (engine/gx_state.h), when its GX objects had
    // ids. Set, they - not the registers above - make the pipeline.
    const GxBlendDesc* gx_blend = nullptr;
    const GxDepthStencilDesc* gx_depth = nullptr;
    const GxRasterDesc* gx_raster = nullptr;
    std::uint32_t gx_ids[3] = {};
    std::uint64_t program_fp = 0;  // GpuDrawInputs::program_fp: stands for vte_cntl and the PS/VS fields below it
};

// Everything a graphics pipeline is made from, hashed once as one packed
// record (it was some thirty-five calls, each over a field or two). The
// image dimensions and sampler modes join it in draw_impl.
std::uint64_t pipeline_key(const DrawState& s, std::uint64_t vs_hash, std::uint64_t fetch_hash, std::uint64_t ps_hash) {
    struct Key {
        std::uint64_t vs, fetch, ps, program_fp, ls_va, ls_fetch_va;
        float tess_outer[4], tess_inner[2];
        std::uint32_t prim, domain_level, tess_control_points, tess_attr_vec4s, ls_rsrc1, ls_rsrc2;
        std::uint32_t tess_hw, tess_attrs;
        std::uint64_t hs_va;
        std::uint32_t hs_rsrc1, hs_rsrc2, tess_window, tess_shape;
        // The state objects' ids, or the registers they stand for.
        std::uint32_t state[12];
        std::uint32_t color_mask[8];
        std::uint32_t formats[9];
        // What program_fp stands for, when the draw carries none.
        std::uint32_t vte, ps_input_ena, vs_out_cntl, cb_shader_mask, ps_input_cntl[32];
    } k;
    std::memset(&k, 0, sizeof(k));
    k.vs = vs_hash;
    k.fetch = fetch_hash;
    k.ps = ps_hash;
    k.prim = s.prim;
    k.domain_level = s.domain_level;
    k.tess_hw = s.tess_hw;
    k.tess_control_points = s.tess_control_points;
    std::memcpy(k.tess_outer, s.tess_outer, sizeof(k.tess_outer));
    std::memcpy(k.tess_inner, s.tess_inner, sizeof(k.tess_inner));
    k.tess_attrs = s.tess_attrs;
    if (s.tess_hull) {
        k.hs_va = s.hs_va;
        k.hs_rsrc1 = s.hs_rsrc1;
        k.hs_rsrc2 = s.hs_rsrc2;
        k.tess_window = s.tess_window;
        k.tess_shape = (s.tess_quads ? 1u : 0u) | (s.tess_cw ? 2u : 0u) | static_cast<std::uint32_t>(s.tess_spacing) << 2;
    }
    if (s.tess_attrs) {
        k.tess_attr_vec4s = s.tess_attr_vec4s;
        k.ls_va = s.ls_va;
        k.ls_fetch_va = s.ls_fetch_va;
        k.ls_rsrc1 = s.ls_rsrc1;
        k.ls_rsrc2 = s.ls_rsrc2;
    }
    if (s.gx_blend) {
        std::memcpy(k.state, s.gx_ids, sizeof(s.gx_ids));  // the state objects stand for the registers they were encoded into
    } else {
        k.state[0] = s.depth_control;
        k.state[1] = s.stencil_control;
        std::memcpy(k.state + 2, s.blend, sizeof(s.blend));
        k.state[10] = s.su_sc_mode;
        k.state[11] = s.clip_cntl;
    }
    std::memcpy(k.color_mask, s.color_mask, sizeof(k.color_mask));
    for (int t = 0; t < 8; ++t) k.formats[t] = s.color[t] ? static_cast<std::uint32_t>(s.color[t]->format) : 0u;
    k.formats[8] = s.depth ? static_cast<std::uint32_t>(s.depth->format) : 0u;
    std::size_t bytes = sizeof(k);
    if (s.program_fp) {
        k.program_fp = s.program_fp;
        bytes = offsetof(Key, vte);
    } else {
        k.vte = s.vte_cntl;
        k.ps_input_ena = s.ps_input_ena;
        k.vs_out_cntl = s.vs_out_cntl;
        k.cb_shader_mask = s.cb_shader_mask;
        std::memcpy(k.ps_input_cntl, s.ps_input_cntl, sizeof(k.ps_input_cntl));
    }
    return fnv1a(&k, bytes);
}

// Path-only translation cache: the resource paths of a program do not depend
// on the draw state, so one default translation per code hash tells us which
// T#s to resolve before the real (dimension-aware) translation.
std::unordered_map<std::uint64_t, gcn::TranslateResult> g_paths_cache;  // never erased: references stay valid

}  // namespace

// The per-shader compiles translate each program for its paths as paths_for
// does, with the registers its binary gives, when GX creates it: a draw of a
// program new to the command processor takes that instead of translating it
// again there (~15% of the command processor in the frames that ran long
// while new areas streamed in).
std::mutex g_paths_ready_mu;
std::unordered_map<std::uint64_t, gcn::TranslateResult> g_paths_ready;
std::atomic<std::uint64_t> g_paths_taken{0}, g_paths_translated{0};
namespace {
void manifest_note_paths(const std::vector<std::uint32_t>& words, gcn::Stage stage, std::uint32_t rsrc1, std::uint32_t rsrc2);
}

void offer_paths(const std::vector<std::uint32_t>& words, const gcn::TranslateResult& paths) {
    const std::uint64_t hash = fnv1a(words.data(), words.size() * 4);
    gcn::TranslateResult kept = paths;
    kept.spirv = {};  // what paths_for's callers read is the bindings; ~3000 programs are compiled at creation
    std::lock_guard<std::mutex> lk(g_paths_ready_mu);
    g_paths_ready.emplace(hash, std::move(kept));
}

const gcn::TranslateResult& paths_for(std::uint64_t hash, const std::vector<std::uint32_t>& words, gcn::Stage stage,
                                      std::uint32_t rsrc1, std::uint32_t rsrc2) {
    struct Memo {
        std::uint64_t hash = 0;
        const gcn::TranslateResult* paths = nullptr;
    };
    static Memo memo[64];  // under g.mu; g_paths_cache is never erased
    Memo& m = memo[hash & 63];
    if (m.paths && m.hash == hash) return *m.paths;
    auto it = g_paths_cache.find(hash);
    if (it != g_paths_cache.end()) {
        m = {hash, &it->second};
        return it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g_paths_ready_mu);
        if (auto rd = g_paths_ready.find(hash); rd != g_paths_ready.end()) {
            g_paths_taken.fetch_add(1, std::memory_order_relaxed);
            gcn::TranslateResult& taken = g_paths_cache[hash] = std::move(rd->second);
            g_paths_ready.erase(rd);
            return taken;
        }
    }
    g_paths_translated.fetch_add(1, std::memory_order_relaxed);
    manifest_note_paths(words, stage, rsrc1, rsrc2);  // the next start translates them off the command processor
    const gcn::Program prog = gcn::decode(words.data(), words.size());
    gcn::TranslateOptions o;
    o.stage = stage;
    o.rsrc1 = rsrc1;
    o.rsrc2 = rsrc2;
    return g_paths_cache[hash] = translate_cached(prog, o, TranslationUse::kPaths);
}

namespace {

// Fixed-function pipeline state from the draw's registers. Pipelines are
// created from it and a draw capture records it, so both see the same state.
// Rasterizer, depth and stencil state (part of fixed_state). Pipelines linked
// from libraries set these per draw, so a draw computes them without the
// vectors fixed_state fills.
// State from the descriptions the game gave GX (engine/gx_state.h)
// instead of the registers GX encoded from them, when the draw carries them.
// GX's blend ops, stencil ops and compare functions are Vulkan's in order;
// its blend factors are D3D11's, 0-based.
VkBlendFactor gx_blend_factor(std::uint8_t f) {
    static constexpr VkBlendFactor k[17] = {
        VK_BLEND_FACTOR_ZERO,           VK_BLEND_FACTOR_ONE,           VK_BLEND_FACTOR_SRC_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_FACTOR_DST_ALPHA,      VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_CONSTANT_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
        VK_BLEND_FACTOR_SRC1_ALPHA,     VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA};
    return f < 17 ? k[f] : VK_BLEND_FACTOR_ZERO;
}

VkStencilOpState gx_stencil_face(const GxStencilFace& face) {
    VkStencilOpState o{};
    o.failOp = static_cast<VkStencilOp>(face.fail);
    o.passOp = static_cast<VkStencilOp>(face.pass);
    o.depthFailOp = static_cast<VkStencilOp>(face.depth_fail);
    o.compareOp = static_cast<VkCompareOp>(face.func);
    return o;
}

// Whether a colour format can blend at all. The registers carry the answer as
// CB_COLOR_INFO.BLEND_BYPASS; a description leaves it to the format, as D3D11
// does. Pipelines are also built on worker threads, hence the atomics.
bool format_blends(VkFormat format) {
    static std::atomic<std::int8_t> known[256];  // 0 not asked yet, 1 blends, -1 does not
    const auto i = static_cast<std::uint32_t>(format);
    if (i < 256) {
        if (const std::int8_t k = known[i].load(std::memory_order_relaxed)) return k > 0;
    }
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(g.phys, format, &fp);
    const bool blends = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
    if (i < 256) known[i].store(blends ? 1 : -1, std::memory_order_relaxed);
    return blends;
}

void fixed_raster_depth(const DrawState& s, GfxFixedState& f) {
    f.topology = s.topo;
    // BBHOST_DEBUG_NO_CULL=1 / BBHOST_DEBUG_NO_DEPTH=1: draw everything
    // (is geometry reaching the target at all?).
    static const bool no_cull = [] { const char* e = std::getenv("BBHOST_DEBUG_NO_CULL"); return e && e[0] == '1'; }();
    static const bool no_depth = [] { const char* e = std::getenv("BBHOST_DEBUG_NO_DEPTH"); return e && e[0] == '1'; }();
    const bool stencil_aspect = s.depth && s.depth->format != VK_FORMAT_D32_SFLOAT && s.depth->format != VK_FORMAT_D16_UNORM;
    if (s.gx_raster && s.gx_depth) {
        const GxRasterDesc& r = *s.gx_raster;
        const GxDepthStencilDesc& ds = *s.gx_depth;
        f.cull_mode = no_cull ? 0u : r.cull == 1 ? VK_CULL_MODE_FRONT_BIT : r.cull == 2 ? VK_CULL_MODE_BACK_BIT : 0u;
        f.front_face = r.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        f.polygon_mode = r.fill ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        f.depth_clamp = !r.depth_clip;
        f.depth_test = ds.depth_enable && s.depth && !no_depth;
        f.depth_write = ds.depth_write && s.depth;
        f.depth_compare = static_cast<VkCompareOp>(ds.depth_func);
        f.stencil_test = ds.stencil_enable && stencil_aspect;
        f.depth_bounds_test = ds.bounds && s.depth && !no_depth && g.has_depth_bounds;
        f.front = gx_stencil_face(ds.front);
        f.back = gx_stencil_face(ds.back);
        // The depth bias the register path never read (PA_SU_POLY_OFFSET_*,
        // programmed by the rasterizer flush 0x2ad3370 with D3D11's meaning).
        // BBHOST_DEPTH_BIAS=0 leaves it off, as before.
        static const bool depth_bias = [] {
            const char* e = std::getenv("BBHOST_DEPTH_BIAS");
            return !(e && e[0] == '0');
        }();
        if (depth_bias && (r.depth_bias || r.slope_bias != 0.0f)) {
            f.depth_bias = true;
            f.bias_constant = static_cast<float>(r.depth_bias);
            f.bias_clamp = g.has_depth_bias_clamp ? r.bias_clamp : 0.0f;
            f.bias_slope = r.slope_bias;
        }
        return;
    }
    const std::uint32_t mode = s.su_sc_mode;
    f.cull_mode = no_cull ? 0u : ((mode & 1) ? VK_CULL_MODE_FRONT_BIT : 0) | ((mode & 2) ? VK_CULL_MODE_BACK_BIT : 0);
    f.front_face = (mode & 4) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if ((mode & 0x10) && ((mode >> 5) & 7) == 1) f.polygon_mode = VK_POLYGON_MODE_LINE;  // POLY_MODE with LINES
    f.depth_clamp = (s.clip_cntl & (1u << 16)) != 0;  // ZCLIP_NEAR_DISABLE approximated
    const std::uint32_t dc = s.depth_control;
    f.depth_test = (dc & 2) && s.depth && !no_depth;
    f.depth_write = (dc & 4) && s.depth;
    f.depth_compare = compare_op((dc >> 4) & 7);
    f.stencil_test = (dc & 1) && stencil_aspect;
    // DB_DEPTH_CONTROL bit 3. The deferred lights are full-screen quads that
    // cull by the light's depth range; the bounds themselves are dynamic.
    f.depth_bounds_test = (dc & 8) && s.depth && !no_depth && g.has_depth_bounds;
    const std::uint32_t sc = s.stencil_control;
    f.front.failOp = stencil_op(sc & 0xf);
    f.front.passOp = stencil_op((sc >> 4) & 0xf);
    f.front.depthFailOp = stencil_op((sc >> 8) & 0xf);
    f.front.compareOp = compare_op((dc >> 8) & 7);
    f.back = f.front;
    if (dc & (1u << 7)) {  // BACKFACE_ENABLE
        f.back.failOp = stencil_op((sc >> 12) & 0xf);
        f.back.passOp = stencil_op((sc >> 16) & 0xf);
        f.back.depthFailOp = stencil_op((sc >> 20) & 0xf);
        f.back.compareOp = compare_op((dc >> 20) & 7);
    }
}

GfxFixedState fixed_state(const DrawState& s) {
    GfxFixedState f;
    fixed_raster_depth(s, f);
    const bool stencil_aspect = s.depth && s.depth->format != VK_FORMAT_D32_SFLOAT && s.depth->format != VK_FORMAT_D16_UNORM;
    // Attachment N is CB slot N, gaps included: the pixel shader writes MRT N
    // at location N, and location N is the Nth attachment. Packing the bound
    // slots together moved every export after a gap one target down - the
    // Cainhurst ground has no slot 0, so its MRT1 (normals) landed in slot 2,
    // the albedo, and the snow came out red and purple. A gap below the last
    // bound slot is an attachment with no format (begin_pass gives it no
    // view), which Vulkan discards writes to.
    int last = -1;
    for (int t = 0; t < 8; ++t) {
        if (s.color[t]) last = t;
    }
    for (int t = 0; t <= last; ++t) {
        if (!s.color[t]) {
            f.blends.push_back(VkPipelineColorBlendAttachmentState{});  // write mask 0
            f.color_formats.push_back(VK_FORMAT_UNDEFINED);
            f.color_slots.push_back(t);
            continue;
        }
        VkPipelineColorBlendAttachmentState b{};
        if (s.gx_blend) {
            const GxBlendTarget& rt = s.gx_blend->rt[t];
            b.blendEnable = rt.enable && format_blends(s.color[t]->format) ? VK_TRUE : VK_FALSE;
            b.srcColorBlendFactor = gx_blend_factor(rt.src);
            b.dstColorBlendFactor = gx_blend_factor(rt.dst);
            b.colorBlendOp = static_cast<VkBlendOp>(rt.op);
            b.srcAlphaBlendFactor = gx_blend_factor(rt.src_alpha);
            b.dstAlphaBlendFactor = gx_blend_factor(rt.dst_alpha);
            b.alphaBlendOp = static_cast<VkBlendOp>(rt.op_alpha);
        } else {
            const std::uint32_t bc = s.blend[t];
            b.blendEnable = (bc & (1u << 30)) ? VK_TRUE : VK_FALSE;
            b.srcColorBlendFactor = blend_factor(bc & 0x1f);
            b.colorBlendOp = blend_op((bc >> 5) & 7);
            b.dstColorBlendFactor = blend_factor((bc >> 8) & 0x1f);
            if (bc & (1u << 29)) {  // SEPARATE_ALPHA_BLEND
                b.srcAlphaBlendFactor = blend_factor((bc >> 16) & 0x1f);
                b.alphaBlendOp = blend_op((bc >> 21) & 7);
                b.dstAlphaBlendFactor = blend_factor((bc >> 24) & 0x1f);
            } else {
                b.srcAlphaBlendFactor = b.srcColorBlendFactor;
                b.alphaBlendOp = b.colorBlendOp;
                b.dstAlphaBlendFactor = b.dstColorBlendFactor;
            }
        }
        // The target's write mask where the pixel shader exports (draw_impl
        // makes it from either source; they agree on every draw compared).
        b.colorWriteMask = s.color_mask[t] & 0xf;
        f.blends.push_back(b);
        f.color_formats.push_back(s.color[t]->format);
        f.color_slots.push_back(t);
    }
    if (s.depth) {
        f.depth_format = s.depth->format;
        if (stencil_aspect) f.stencil_format = s.depth->format;
    }
    return f;
}

// BBHOST_GX_STATE=2: at every draw with the three ids, the state the
// descriptions give against the register path's, field by field (the report
// at exit). BLEND_BYPASS (a target format that cannot blend) turns blending
// off on the register path only, so it is counted apart.
enum GxStateField { kGsCull, kGsFront, kGsFill, kGsClamp, kGsDepthTest, kGsDepthWrite, kGsDepthFunc, kGsBounds, kGsStencilTest,
                    kGsStencilOps, kGsStencilMasks, kGsBlendEnable, kGsBlendFactors, kGsWriteMask, kGsFields };
const char* const kGsFieldNames[kGsFields] = {"cull", "front face", "fill", "depth clamp", "depth test", "depth write", "depth func",
                                              "depth bounds", "stencil test", "stencil ops", "stencil masks", "blend enable",
                                              "blend factors", "write mask"};
std::uint64_t g_gs_draws = 0, g_gs_unknown[3] = {}, g_gs_diff[kGsFields] = {}, g_gs_bypass = 0, g_gs_biased = 0;
std::uint64_t g_gs_from_objects = 0, g_gs_from_registers = 0;  // mode 1: GX draws by where their state came from
std::uint64_t g_gs_raster_draws[64] = {}, g_gs_raster_depth_only[64] = {};  // mode 1: draws by rasterizer description

void compare_gx_state(const DrawState& s, const GpuDrawInputs& in, const GxDrawObjects& o) {
    const GxBlendDesc* b = gx_blend_desc(o.blend_id);
    const GxDepthStencilDesc* ds = gx_depth_stencil_desc(o.depth_stencil_id);
    const GxRasterDesc* r = gx_raster_desc(o.raster_id);
    if (!b || !ds || !r) {
        if (!b) ++g_gs_unknown[0];
        if (!ds) ++g_gs_unknown[1];
        if (!r) ++g_gs_unknown[2];
        return;
    }
    ++g_gs_draws;
    if (r->depth_bias || r->slope_bias != 0.0f) ++g_gs_biased;
    // The register side as drawn (compare mode keeps the descriptions off the
    // draw), and the object side exactly as mode 1 would build it.
    const GfxFixedState reg = fixed_state(s);
    DrawState with = s;
    with.gx_blend = b;
    with.gx_depth = ds;
    with.gx_raster = r;
    const GfxFixedState obj = fixed_state(with);
    bool diff[kGsFields] = {};
    diff[kGsCull] = reg.cull_mode != obj.cull_mode;
    diff[kGsFront] = reg.front_face != obj.front_face;
    diff[kGsFill] = reg.polygon_mode != obj.polygon_mode;
    diff[kGsClamp] = reg.depth_clamp != obj.depth_clamp;
    diff[kGsDepthTest] = reg.depth_test != obj.depth_test;
    diff[kGsDepthWrite] = reg.depth_write != obj.depth_write;
    diff[kGsDepthFunc] = reg.depth_test && reg.depth_compare != obj.depth_compare;
    diff[kGsBounds] = reg.depth_bounds_test != obj.depth_bounds_test;
    diff[kGsStencilTest] = reg.stencil_test != obj.stencil_test;
    if (reg.stencil_test && obj.stencil_test) {
        diff[kGsStencilOps] = std::memcmp(&reg.front, &obj.front, sizeof(reg.front)) != 0 ||
                              std::memcmp(&reg.back, &obj.back, sizeof(reg.back)) != 0;
        diff[kGsStencilMasks] = ((s.stencil_ref >> 8) & 0xff) != ds->read_mask || ((s.stencil_ref >> 16) & 0xff) != ds->write_mask;
    }
    for (std::size_t k = 0; k < reg.blends.size() && k < obj.blends.size(); ++k) {
        const VkPipelineColorBlendAttachmentState& a = reg.blends[k];
        const VkPipelineColorBlendAttachmentState& c = obj.blends[k];
        const int t = reg.color_slots[k];
        if (a.blendEnable != c.blendEnable) {
            if (!a.blendEnable && ((in.color[t][4] >> 16) & 1)) {
                ++g_gs_bypass;
            } else {
                diff[kGsBlendEnable] = true;
            }
        } else if (a.blendEnable &&
                   (a.srcColorBlendFactor != c.srcColorBlendFactor || a.dstColorBlendFactor != c.dstColorBlendFactor ||
                    a.colorBlendOp != c.colorBlendOp || a.srcAlphaBlendFactor != c.srcAlphaBlendFactor ||
                    a.dstAlphaBlendFactor != c.dstAlphaBlendFactor || a.alphaBlendOp != c.alphaBlendOp)) {
            diff[kGsBlendFactors] = true;
        }
        if (a.colorWriteMask != c.colorWriteMask) diff[kGsWriteMask] = true;
    }
    for (int k = 0; k < kGsFields; ++k) {
        if (!diff[k]) continue;
        if (g_gs_diff[k]++ < 4) {
            host_log("render: GX state differs from the registers in %s (ids %08x %08x %08x): DB_DEPTH_CONTROL %08x "
                     "DB_STENCIL_CONTROL %08x STENCILREFMASK %08x PA_SU_SC_MODE_CNTL %08x PA_CL_CLIP_CNTL %08x CB_BLEND0 %08x "
                     "mask %x; description: depth %u write %u func %u bounds %u stencil %u masks %02x/%02x, cull %u front-ccw %u "
                     "fill %u clip %u, target 0 blend %u %u/%u/%u %u/%u/%u mask %x (independent %u)",
                     kGsFieldNames[k], o.blend_id, o.depth_stencil_id, o.raster_id, s.depth_control, s.stencil_control, s.stencil_ref,
                     s.su_sc_mode, s.clip_cntl, s.blend[0], s.color_mask[0], ds->depth_enable, ds->depth_write, ds->depth_func,
                     ds->bounds, ds->stencil_enable, ds->read_mask, ds->write_mask, r->cull, r->front_ccw, r->fill, r->depth_clip,
                     b->rt[0].enable, b->rt[0].src, b->rt[0].op, b->rt[0].dst, b->rt[0].src_alpha, b->rt[0].op_alpha,
                     b->rt[0].dst_alpha, b->rt[0].mask, b->independent);
        }
    }
}

void report_gx_state_compare() {
    if (gx_state_mode() == 1) {
        host_log("render: GX draws with their blend, depth-stencil and rasterizer state from the game's descriptions %llu, from the "
                 "registers %llu (objects without ids)",
                 static_cast<unsigned long long>(g_gs_from_objects), static_cast<unsigned long long>(g_gs_from_registers));
        std::string per;
        for (int i = 1; i < 64; ++i) {
            if (!g_gs_raster_draws[i]) continue;
            char one[80];
            std::snprintf(one, sizeof(one), " %08x %llu (depth only %llu);", 0xc0000000u | static_cast<std::uint32_t>(i),
                          static_cast<unsigned long long>(g_gs_raster_draws[i]), static_cast<unsigned long long>(g_gs_raster_depth_only[i]));
            per += one;
        }
        host_log("render: GX draws by rasterizer description:%s", per.c_str());
    }
    if (gx_state_mode() != 2) return;
    std::string diffs;
    for (int k = 0; k < kGsFields; ++k) {
        char one[64];
        std::snprintf(one, sizeof(one), " %s %llu;", kGsFieldNames[k], static_cast<unsigned long long>(g_gs_diff[k]));
        diffs += one;
    }
    host_log("render: GX state against the registers: %llu draws compared (ids unknown: blend %llu, depth-stencil %llu, "
             "rasterizer %llu); differing:%s blending off for BLEND_BYPASS %llu; draws with a depth bias the register path "
             "ignores %llu",
             static_cast<unsigned long long>(g_gs_draws), static_cast<unsigned long long>(g_gs_unknown[0]),
             static_cast<unsigned long long>(g_gs_unknown[1]), static_cast<unsigned long long>(g_gs_unknown[2]), diffs.c_str(),
             static_cast<unsigned long long>(g_gs_bypass), static_cast<unsigned long long>(g_gs_biased));
}

// Vulkan vertex input: the unsigned format that holds a V# data
// format's raw components (packed formats as one dword), so the vertex shader
// converts them as the fetch shader did; UNDEFINED for unknown formats.
VkFormat vertex_format(std::uint32_t dfmt) {
    switch (dfmt) {
        case 1: return VK_FORMAT_R8_UINT;
        case 3: return VK_FORMAT_R8G8_UINT;
        case 10: return VK_FORMAT_R8G8B8A8_UINT;
        case 2: return VK_FORMAT_R16_UINT;
        case 5: return VK_FORMAT_R16G16_UINT;
        case 12: return VK_FORMAT_R16G16B16A16_UINT;
        case 4: return VK_FORMAT_R32_UINT;
        case 11: return VK_FORMAT_R32G32_UINT;
        case 13: return VK_FORMAT_R32G32B32_UINT;
        case 14: return VK_FORMAT_R32G32B32A32_UINT;
        case 6: case 7: case 8: case 9: return VK_FORMAT_R32_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}
std::uint32_t vertex_format_bytes(std::uint32_t dfmt) {
    switch (dfmt) {
        case 1: return 1;
        case 2: case 3: return 2;
        case 11: case 12: return 8;
        case 13: return 12;
        case 14: return 16;
        default: return 4;
    }
}

// Vulkan vertex input for one GX draw: the translator's elements,
// the pipeline's bindings and attributes (attribute n is element n, at
// location n), and each binding's buffer and offset.
struct VertexInputPlan {
    std::vector<gcn::VertexElement> elements;
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    VkBuffer buffer[16] = {};
    VkDeviceSize offset[16] = {};
    std::uint64_t va[16] = {};  // guest address of each binding's first byte (draw captures copy from there)
    std::uint64_t hash = 0;     // pipeline identity: elements, bindings and attributes
};

// Read masks (table_slots_read) run the SPIRV-Tools optimizer, tens of
// milliseconds a module. On the command processor's thread that doubled the
// loading screen and stalled the game whenever a pipeline was created, so
// worker threads compute them. Until a pipeline's mask arrives its table_reads
// stay ~0, and its draws build every table, as with table skipping off. Dump and
// check modes still compute masks in place.
struct TableMaskJob {
    GfxPipeline* pipeline = nullptr;  // g_gfx never erases, so the pipeline outlives the job
    int stage = 0;
    // The module and its buffer list: the stage cache's, shared, or a copy of
    // a stage translated for its pipeline - workers never touch g_gfx.
    std::shared_ptr<const gcn::TranslateResult> meta;
};
struct TableMaskResult {
    GfxPipeline* pipeline;
    int stage;
    std::uint32_t mask, any;
};
struct TableMaskWorkers {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<TableMaskJob> jobs;          // under mu
    std::vector<TableMaskResult> results;  // under mu
};
std::atomic<bool> g_table_masks_ready{false};
std::atomic<std::uint64_t> g_table_masks_queued{0}, g_table_masks_done{0};

TableMaskWorkers& table_mask_workers() {
    static TableMaskWorkers* const workers = [] {
        auto* w = new TableMaskWorkers;  // never destroyed: its threads run until exit
        for (int i = 0; i < 4; ++i) {
            std::thread([w] {
                for (;;) {
                    TableMaskJob job;
                    {
                        std::unique_lock<std::mutex> lk(w->mu);
                        w->cv.wait(lk, [w] { return !w->jobs.empty(); });
                        job = std::move(w->jobs.front());
                        w->jobs.pop_front();
                    }
                    std::uint32_t mask = ~0u, any = ~0u;
                    table_slots_read(*job.meta, mask, &any);
                    std::lock_guard<std::mutex> lk(w->mu);
                    w->results.push_back({job.pipeline, job.stage, mask, any});
                    g_table_masks_ready.store(true, std::memory_order_relaxed);
                    g_table_masks_done.fetch_add(1, std::memory_order_relaxed);
                }
            }).detach();
        }
        return w;
    }();
    return *workers;
}

void queue_table_mask(GfxPipeline& pl, int stage, const ShaderStage& st) {
    TableMaskJob job;
    job.pipeline = &pl;
    job.stage = stage;
    if (st.shared) {
        job.meta = st.shared;
    } else {
        auto copy = std::make_shared<gcn::TranslateResult>();
        copy->spirv = st.own.spirv;
        copy->buffers = st.own.buffers;
        job.meta = std::move(copy);
    }
    TableMaskWorkers& w = table_mask_workers();
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_back(std::move(job));
    }
    g_table_masks_queued.fetch_add(1, std::memory_order_relaxed);
    w.cv.notify_one();
}

// Under g.mu: hands the masks the workers finished to their pipelines.
void apply_table_masks_locked() {
    if (!g_table_masks_ready.load(std::memory_order_relaxed)) return;
    TableMaskWorkers& w = table_mask_workers();
    std::vector<TableMaskResult> done;
    {
        std::lock_guard<std::mutex> lk(w.mu);
        done.swap(w.results);
        g_table_masks_ready.store(false, std::memory_order_relaxed);
    }
    for (const TableMaskResult& r : done) {
        r.pipeline->table_reads[r.stage] = r.mask;
        r.pipeline->user_reads[r.stage] = r.any;
    }
}

// BBHOST_SET_LAYOUTS (on unless 0): each graphics pipeline's sets have a layout
// with only the params buffer and the bindings its stage's translation
// declares, instead of the shared layout's 113, so allocating and binding a set
// covers the descriptors the stage uses. Stages with the same bindings share a
// layout, and pipelines with the same pair share a pipeline layout: a
// no-fallback variant declares its base pipeline's bindings, so it takes the
// sets written for them. Compute dispatches keep the shared layout.
const bool g_set_layouts = [] {
    const char* e = std::getenv("BBHOST_SET_LAYOUTS");
    return !(e && e[0] == '0');
}();
// BBHOST_PIPELINE_LIBRARY=0: monolithic graphics pipelines. By default, where
// the driver fast-links VK_EXT_graphics_pipeline_library, graphics pipelines
// link from pipeline libraries whose shaders are compiled when GX creates them:
// a cold world load stalls 12.6 s
// against 19.4 s, for 2.4% more GPU time and 1.1 GiB more host memory in steady
// state. Needs the per-stage set layouts; both sets take the shared layout
// (build_gfx_pipeline).
bool use_pipeline_library() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_PIPELINE_LIBRARY");
        return !(e && e[0] == '0');
    }();
    return on && g.has_gpl && g_set_layouts;
}
std::mutex g_stage_layout_mu;  // pipelines are built on worker threads too
std::unordered_map<std::uint64_t, VkDescriptorSetLayout> g_stage_set_layouts;  // by binding list, under g_stage_layout_mu
std::unordered_map<std::uint64_t, VkPipelineLayout> g_stage_pipe_layouts;       // by set layout pair, under g_stage_layout_mu

VkDescriptorSetLayout stage_set_layout(const gcn::TranslateResult& meta) {
    std::lock_guard<std::mutex> lk(g_stage_layout_mu);
    std::vector<VkDescriptorSetLayoutBinding> binds;
    binds.push_back({gcn::kBindingParams, params_descriptor_type(), 1, VK_SHADER_STAGE_ALL, nullptr});
    for (const gcn::ImageBinding& b : meta.images) {
        binds.push_back({b.binding, b.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_ALL,
                         nullptr});
    }
    for (const gcn::SamplerBinding& b : meta.samplers) {
        binds.push_back({b.binding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_ALL, nullptr});
    }
    if (meta.bindless) {  // images and samplers are set 3's
        binds.erase(std::remove_if(binds.begin(), binds.end(),
                                   [](const VkDescriptorSetLayoutBinding& b) {
                                       return b.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
                                              b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
                                              b.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER;
                                   }),
                    binds.end());
    }
    for (const gcn::BufferBinding& b : meta.buffers) {
        if (cb_push_on()) break;  // set 2, pushed
        binds.push_back({b.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr});
    }
    std::sort(binds.begin(), binds.end(), [](const VkDescriptorSetLayoutBinding& a, const VkDescriptorSetLayoutBinding& b) {
        return a.binding < b.binding;
    });
    binds.erase(std::unique(binds.begin(), binds.end(),
                            [](const VkDescriptorSetLayoutBinding& a, const VkDescriptorSetLayoutBinding& b) {
                                return a.binding == b.binding && a.descriptorType == b.descriptorType;
                            }),
                binds.end());
    std::uint64_t key = 1469598103934665603ull;
    for (const VkDescriptorSetLayoutBinding& b : binds) {
        key = fnv1a(&b.binding, sizeof(b.binding), key);
        key = fnv1a(&b.descriptorType, sizeof(b.descriptorType), key);
    }
    auto it = g_stage_set_layouts.find(key);
    if (it != g_stage_set_layouts.end()) return it->second;
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = static_cast<std::uint32_t>(binds.size());
    lci.pBindings = binds.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(g.device, &lci, nullptr, &layout) != VK_SUCCESS) {
        host_log("render: descriptor set layout with %zu bindings failed", binds.size());
        return VK_NULL_HANDLE;
    }
    g_stage_set_layouts.emplace(key, layout);
    return layout;
}

VkPipelineLayout stage_pipeline_layout(VkDescriptorSetLayout vs, VkDescriptorSetLayout ps) {
    std::lock_guard<std::mutex> lk(g_stage_layout_mu);
    const VkDescriptorSetLayout sets[4] = {vs, ps, g.cb_push_layout, g.bindless_layout};
    const std::uint64_t key = fnv1a(sets, 2 * sizeof(VkDescriptorSetLayout));
    auto it = g_stage_pipe_layouts.find(key);
    if (it != g_stage_pipe_layouts.end()) return it->second;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = g.bindless ? 4 : cb_push_on() ? 3 : 2;  // set 3: the global views and samplers
    plci.pSetLayouts = sets;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(g.device, &plci, nullptr, &layout) != VK_SUCCESS) return VK_NULL_HANDLE;
    g_stage_pipe_layouts.emplace(key, layout);
    return layout;
}

// Builds a pipeline into `pl`: translation, modules, layouts and the Vulkan
// pipeline. Worker threads run it for no-fallback variants (queue_lean_variant),
// so it touches no command-processor state: the layout caches lock, and the
// only RtImage fields it reads through `s` are the targets' formats.
// Where graphics pipeline builds on the command processor's thread go,
// microseconds: decoding and translating stages, creating their shader
// modules, and vkCreateGraphicsPipelines. Builds on the variant workers are
// not counted (they do not stall the game). A stage's key covers its program
// and the options that change its translation; stages the cache served are
// counted as reused (with BBHOST_STAGE_CACHE=0, translations whose key was
// seen before).
thread_local bool t_pipeline_worker = false;
std::atomic<std::uint64_t> g_pl_translate_us{0}, g_pl_module_us{0}, g_pl_create_us{0}, g_pl_stage_translations{0}, g_pl_stage_repeats{0};
std::mutex g_stage_keys_mu;
std::unordered_set<std::uint64_t> g_stage_keys;
std::uint64_t pl_us_since(std::chrono::steady_clock::time_point t0) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}
// With the stage cache off: count translations whose key was seen before.
void note_stage_key(std::uint64_t key) {
    bool seen = false;
    {
        std::lock_guard<std::mutex> lk(g_stage_keys_mu);
        seen = !g_stage_keys.insert(key).second;
    }
    if (seen && !t_pipeline_worker) g_pl_stage_repeats.fetch_add(1, std::memory_order_relaxed);
}
std::uint64_t dims_modes_key(std::uint64_t h, const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes) {
    for (const auto& d : dims) {
        h = fnv1a(&d.first, 4, h);
        h = fnv1a(&d.second, 1, h);
    }
    for (bool m : modes) h = fnv1a(&m, 1, h);
    return h;
}

// BBHOST_STAGE_CACHE=0 (checks): translate every pipeline's stages again. By
// default a stage's translation and shader module are shared by every pipeline
// whose stage has the same program and translation options: in a world-load
// stall, 122 of 206 stage translations repeated an earlier one. Pipelines and
// modules are never destroyed, so sharing needs no lifetime tracking.
// Translation runs outside the lock; two threads racing on one key both
// translate and the first to store wins.
const bool g_stage_cache_on = [] {
    const char* e = std::getenv("BBHOST_STAGE_CACHE");
    return !(e && e[0] == '0');
}();
// A lifted stage (BBHOST_DECOMP) is a different shader from the translation of
// the same program, so it takes a key of its own rather than being left out of
// the cache: the lift used to re-run for every pipeline that shared the stage,
// which is the opposite of what turning it on is for.
constexpr std::uint64_t kLiftKeySalt = 0x11fd0b1e5ull;
std::uint64_t lift_key(std::uint64_t key, bool lifted) { return lifted ? key ^ kLiftKeySalt : key; }

struct CachedStage {
    gcn::TranslateResult meta;
    VkShaderModule module = VK_NULL_HANDLE;
    // The lift's module rather than the translation's. A lift key holds either:
    // a rejected lift caches the translation under it.
    bool lifted = false;
};
std::mutex g_stage_cache_mu;
std::unordered_map<std::uint64_t, std::shared_ptr<const CachedStage>> g_stage_cache;

std::shared_ptr<const CachedStage> cached_stage(std::uint64_t key) {
    std::lock_guard<std::mutex> lk(g_stage_cache_mu);
    const auto it = g_stage_cache.find(key);
    return it == g_stage_cache.end() ? nullptr : it->second;
}

void cache_stage(std::uint64_t key, const gcn::TranslateResult& meta, VkShaderModule module, bool lifted) {
    auto entry = std::make_shared<CachedStage>();
    entry->meta = meta;
    entry->module = module;
    entry->lifted = lifted;
    std::lock_guard<std::mutex> lk(g_stage_cache_mu);
    g_stage_cache.emplace(key, std::move(entry));
}

// Step 2 of the pipeline work (translating stages when GX creates the shader)
// needs to know how many translation variants a program gets from draw-time
// state. Counted for every stage translation, on any thread; render_report
// prints the summary.
struct StageVariantStats {
    std::mutex mu;
    std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> keys_by_program[2];  // VS, PS
    std::uint64_t translations[2] = {}, program_seen_before[2] = {};
    std::uint64_t vs_with_fetch = 0, vs_with_vertex_input = 0, lean[2] = {};
    std::uint64_t ps_dims_all_2d = 0, ps_modes_default = 0, ps_map_identity = 0, ps_flat_none = 0, ps_early_tests = 0, ps_no_images = 0;
    // Image dimensions predicted from the program alone (gcn::predict_image_dims)
    // against those the bound textures gave: T#s and stage translations.
    std::uint64_t dims_images[2] = {}, dims_images_right[2] = {}, dims_stages[2] = {}, dims_stages_right[2] = {}, dims_conflicts[2] = {};
    std::set<std::string> dims_logged;
};
StageVariantStats g_stage_variants;

// Per-shader compiles, step 4: the shaders GX creates
// (gx_trace.cpp hooks its creators), by footer name as extract_program names
// them, and how long before its first translation each was created.
struct ShaderCreations {
    struct Entry {
        std::uint64_t first_flip = 0, creations = 0;
        bool translated = false;
        // From the binary's tables (Shdr + 0x10): a pixel shader's input
        // semantics; a vertex shader's output semantics and registers, and its
        // input registers and component counts.
        std::vector<std::uint8_t> semantics, registers, in_registers, in_components;
    };
    std::mutex mu;
    std::unordered_map<std::string, Entry> by_name[2];
    std::uint64_t calls[2] = {}, unnamed[2] = {};
    std::uint64_t first_translations[2] = {}, created_before[2] = {};
    std::vector<std::uint64_t> flips_ahead[2];
    std::uint64_t types[4][16] = {};      // per creator (0x2566d00, 0x2566f20, 0x25672d0, 0x25673e0): containers by Shdr type byte
    std::uint64_t other_stage_hits[2] = {};  // first translations found only in the other hook's names
    std::vector<std::string> misses[2];      // first translations never seen created (the first few)
};
ShaderCreations g_creations;

// Per-shader compiles, step 4: compiling a pixel shader's stage and
// fragment-shader library when GX creates it (defined with the libraries).
bool precompile_enabled();
void queue_ps_precompile(std::string name, std::vector<std::uint8_t> container);
void queue_vs_precompile(std::string name, std::vector<std::uint8_t> container);
void queue_compute_precompile(std::string name, std::vector<std::uint8_t> container, std::uint64_t flip);
void precompile_report();

// BBHOST_PRECOMPILE_STAGES (checks): which graphics stages compile at creation,
// bit 0 vertex shaders, bit 1 pixel shaders (default both).
int precompile_stage_mask() {
    static const int mask = [] {
        const char* e = std::getenv("BBHOST_PRECOMPILE_STAGES");
        return e ? std::atoi(e) : 3;
    }();
    return mask;
}
void record_shader_created(int creator, const std::uint8_t* container, std::size_t size, std::uint64_t flip) {
    if (creator < 0 || creator > 3) return;
    std::string name;
    for (std::size_t i = size >= 20 ? size - 20 : 0;; --i) {  // the 28-byte footer sits at the end
        if (i + 20 <= size && std::memcmp(container + i, "OrbShdr", 7) == 0) {
            std::uint32_t h0;
            std::memcpy(&h0, container + i + 16, 4);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%08x", h0);
            name = buf;
            break;
        }
        if (i == 0) break;
    }
    std::lock_guard<std::mutex> lk(g_creations.mu);
    // The container's stage from its Shdr type byte (1 vertex, 2 pixel; the
    // creators also build types 5 and 6), counted per creator.
    int type = -1;
    std::size_t shdr = 0;
    for (std::size_t i = 0; i + 16 <= size && i < 0x100; ++i) {
        if (std::memcmp(container + i, "Shdr", 4) == 0) {
            type = container[i + 8] & 0xf;
            shdr = i;
            break;
        }
    }
    if (type >= 0) g_creations.types[creator][type] += 1;
    if (type == 4) {  // compute shaders: compiled when created (gpu.cpp), unless BBHOST_PRECOMPILE=0
        if (!name.empty() && compute_precompile_enabled() && note_compute_created(name, flip)) {
            queue_compute_precompile(name, std::vector<std::uint8_t>(container, container + size), flip);
        }
        return;
    }
    const int stage = type == 1 ? 0 : type == 2 ? 1 : -1;
    if (stage < 0) return;
    g_creations.calls[stage] += 1;
    if (name.empty()) {
        g_creations.unnamed[stage] += 1;
        return;
    }
    auto [it, fresh] = g_creations.by_name[stage].try_emplace(name);
    it->second.creations += 1;
    if (fresh) {
        it->second.first_flip = flip;
        ShaderCreations::Entry& e = it->second;
        const std::size_t base = shdr + 16;
        if (base + 0x40 <= size) {
            const std::size_t usage = container[base + 3];
            if (stage == 1) {
                const std::size_t n = container[base + 0x38], at = base + 0x3c + usage * 4;
                for (std::size_t k = 0; k < n && at + 2 * k + 2 <= size; ++k) e.semantics.push_back(container[at + 2 * k]);
            } else {
                const std::size_t nin = container[base + 0x24], nout = container[base + 0x25], in_at = base + 0x28 + usage * 4;
                for (std::size_t k = 0; k < nin && in_at + 4 * k + 4 <= size; ++k) {
                    e.in_registers.push_back(container[in_at + 4 * k + 1]);
                    e.in_components.push_back(container[in_at + 4 * k + 2]);
                }
                const std::size_t out_at = in_at + nin * 4;
                for (std::size_t k = 0; k < nout && out_at + 2 * k + 2 <= size; ++k) {
                    e.semantics.push_back(container[out_at + 2 * k]);
                    e.registers.push_back(container[out_at + 2 * k + 1]);
                }
            }
        }
        if (stage == 1 && precompile_enabled() && (precompile_stage_mask() & 2)) {
            queue_ps_precompile(name, std::vector<std::uint8_t>(container, container + size));
        }
        if (stage == 0 && precompile_enabled() && (precompile_stage_mask() & 1)) {
            queue_vs_precompile(name, std::vector<std::uint8_t>(container, container + size));
        }
    }
}

// A program's first translation at a draw: was GX's creation of it seen, and
// how many flips earlier?
void note_first_translation(int stage, const std::string& name) {
    const std::uint64_t flip = hle_video_flip_count();
    std::lock_guard<std::mutex> lk(g_creations.mu);
    ShaderCreations::Entry* e = nullptr;
    if (auto it = g_creations.by_name[stage].find(name); it != g_creations.by_name[stage].end()) e = &it->second;
    static std::set<std::string> translated[2];  // under g_creations.mu
    if (!translated[stage].insert(name).second) return;
    g_creations.first_translations[stage] += 1;
    if (e) {
        g_creations.created_before[stage] += 1;
        g_creations.flips_ahead[stage].push_back(flip >= e->first_flip ? flip - e->first_flip : 0);
    } else if (g_creations.by_name[1 - stage].count(name)) {
        g_creations.other_stage_hits[stage] += 1;
    } else if (g_creations.misses[stage].size() < 12) {
        g_creations.misses[stage].push_back(name);
    }
}

// Vertex shaders at creation: what a vertex
// shader's translation depends on beyond its binary, measured at pipeline
// builds: the vertex input elements (VGPRs, component counts, formats) per program.
struct VsDependencyStats {
    struct Program {
        std::set<std::uint64_t> elements, layouts, links, other;
        bool fetch = false, input = false, fallback = false, lean = false;
    };
    std::unordered_map<std::uint64_t, Program> programs;
    std::uint64_t elements_checked = 0, elements_in_table = 0, table_unknown = 0;
    std::map<std::uint32_t, std::uint64_t> formats;  // data format << 16 | number format << 12 | DST_SEL
};
VsDependencyStats g_vs_deps;  // under g_creations.mu

void note_vs_dependencies(const std::string& vs_name, const std::vector<std::uint32_t>& words, const VertexInputPlan* vertex_input,
                          const std::vector<std::uint8_t>& links, bool lean, const DrawState& s,
                          const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes) {
    const std::uint64_t program = fnv1a(words.data(), words.size() * 4);
    std::lock_guard<std::mutex> lk(g_creations.mu);
    VsDependencyStats::Program& p = g_vs_deps.programs[program];
    (lean ? p.lean : p.fallback) = true;
    (vertex_input ? p.input : p.fetch) = true;
    p.links.insert(fnv1a(links.data(), links.size()));
    const std::uint32_t regs[3] = {s.vs_rsrc1, s.vs_rsrc2, s.vs_out_cntl};
    p.other.insert(dims_modes_key(fnv1a(regs, sizeof(regs)), dims, modes));
    if (!vertex_input) return;
    std::uint64_t layout = 1469598103934665603ull;
    for (const gcn::VertexElement& el : vertex_input->elements) {
        const std::uint32_t l[3] = {el.location, el.vdata, el.count};
        layout = fnv1a(l, sizeof(l), layout);
        g_vs_deps.formats[((el.w3 >> 15) & 0xf) << 16 | ((el.w3 >> 12) & 7) << 12 | (el.w3 & 0xfff)] += 1;
    }
    p.layouts.insert(layout);
    p.elements.insert(fnv1a(vertex_input->elements.data(), vertex_input->elements.size() * sizeof(gcn::VertexElement)));
    const auto it = g_creations.by_name[0].find(vs_name);
    if (it == g_creations.by_name[0].end()) {
        g_vs_deps.table_unknown += 1;
        return;
    }
    const ShaderCreations::Entry& e = it->second;
    for (const gcn::VertexElement& el : vertex_input->elements) {
        g_vs_deps.elements_checked += 1;
        for (std::size_t k = 0; k < e.in_registers.size(); ++k) {
            if (e.in_registers[k] == el.vdata && e.in_components[k] == el.count) {
                g_vs_deps.elements_in_table += 1;
                break;
            }
        }
    }
}

void vs_dependency_report() {
    std::lock_guard<std::mutex> lk(g_creations.mu);
    const VsDependencyStats& d = g_vs_deps;
    std::uint64_t input = 0, multi_elements = 0, multi_layouts = 0, multi_links = 0, multi_other = 0, both_paths = 0, both_variants = 0;
    for (const auto& [h, p] : d.programs) {
        input += p.input;
        multi_elements += p.elements.size() > 1;
        multi_layouts += p.layouts.size() > 1;
        multi_links += p.links.size() > 1;
        multi_other += p.other.size() > 1;
        both_paths += p.fetch && p.input;
        both_variants += p.lean && p.fallback;
    }
    host_log("render: vertex shader dependencies: %zu programs translated, %llu on vertex input; programs with 2+ element sets %llu (2+ element "
             "layouts %llu), 2+ output links %llu, 2+ registers/dims/modes %llu, both fetch and vertex input %llu, both fallback and "
             "no-fallback %llu; elements matching the binary's input table %llu of %llu (%llu translations without the table)",
             d.programs.size(), static_cast<unsigned long long>(input), static_cast<unsigned long long>(multi_elements),
             static_cast<unsigned long long>(multi_layouts), static_cast<unsigned long long>(multi_links), static_cast<unsigned long long>(multi_other),
             static_cast<unsigned long long>(both_paths), static_cast<unsigned long long>(both_variants),
             static_cast<unsigned long long>(d.elements_in_table), static_cast<unsigned long long>(d.elements_checked),
             static_cast<unsigned long long>(d.table_unknown));
    std::string formats;
    for (const auto& [f, count] : d.formats) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " data %u number %u sel %03x x%llu;", f >> 16, (f >> 12) & 7, f & 0xfff, static_cast<unsigned long long>(count));
        formats += buf;
    }
    host_log("render: vertex input element formats:%s", formats.c_str());
}

// BBHOST_PIPELINE_LIBRARY: libraries by key (vertex input interface,
// pre-rasterization shaders, fragment shader, fragment output interface). A key
// another thread is building waits for it.
struct PipelineLibraries {
    std::mutex mu;
    std::condition_variable cv;
    std::unordered_map<std::uint64_t, VkPipeline> ready;  // under mu; VK_NULL_HANDLE when creation failed
    std::unordered_set<std::uint64_t> building;           // under mu
    std::atomic<std::uint64_t> built[4]{}, hits[4]{}, us[4]{};
    std::atomic<std::uint64_t> links{0}, link_us{0};
};
PipelineLibraries g_libs;

// BBHOST_LOG_STAGE_TRANSLATIONS=1: each stage the command processor translates
// itself (a precompiled or cached stage would have saved it), with its
// variant, for finding what the per-shader compiles miss.
const bool g_log_stage_translations = std::getenv("BBHOST_LOG_STAGE_TRANSLATIONS") != nullptr;

void note_stage_translation(int stage, const std::vector<std::uint32_t>& words, std::uint64_t key, bool lean) {
    const std::uint64_t program = fnv1a(words.data(), words.size() * 4);
    if (g_log_stage_translations && !t_pipeline_worker) {
        host_log("render: CP translates %s %016llx (%zu words, %s) at flip %llu", stage ? "PS" : "VS",
                 static_cast<unsigned long long>(program), words.size(), lean ? "no-fallback" : "fallback",
                 static_cast<unsigned long long>(hle_video_flip_count()));
    }
    std::lock_guard<std::mutex> lk(g_stage_variants.mu);
    auto& keys = g_stage_variants.keys_by_program[stage][program];
    if (!keys.empty() && !keys.count(key)) g_stage_variants.program_seen_before[stage] += 1;
    keys.insert(key);
    g_stage_variants.translations[stage] += 1;
    if (lean) g_stage_variants.lean[stage] += 1;
}

// Step 2 of per-shader compiles: would the dimensions
// predicted from the program alone have been the ones the bound textures gave?
void note_image_dims(int stage, const std::string& name, const gcn::Program& program, const gcn::TranslateResult& meta,
                     const std::vector<std::pair<std::uint32_t, bool>>& actual) {
    const std::vector<gcn::PredictedImage> predicted = gcn::predict_image_dims(program, meta);
    std::lock_guard<std::mutex> lk(g_stage_variants.mu);
    bool all_right = true;
    for (std::size_t k = 0; k < predicted.size(); ++k) {
        const std::pair<std::uint32_t, bool> a = k < actual.size() ? actual[k] : std::pair<std::uint32_t, bool>{1, false};
        const bool right = predicted[k].dim == (a.first & 0xff) && predicted[k].arrayed == a.second;
        g_stage_variants.dims_images[stage] += 1;
        if (right) g_stage_variants.dims_images_right[stage] += 1;
        if (predicted[k].conflict) g_stage_variants.dims_conflicts[stage] += 1;
        all_right &= right;
        const std::string key = name + "#" + std::to_string(k);
        if ((!right || predicted[k].conflict) && g_stage_variants.dims_logged.size() < 120 && g_stage_variants.dims_logged.insert(key).second) {
            host_log("render: image dims of %s %s T#%zu (binding %u): predicted dim %u%s%s, bound dim %u%s", stage ? "PS" : "VS", name.c_str(), k,
                     meta.images[k].binding, predicted[k].dim, predicted[k].arrayed ? " arrayed" : "", predicted[k].conflict ? " (conflicting samples)" : "",
                     a.first, a.second ? " arrayed" : "");
        }
    }
    g_stage_variants.dims_stages[stage] += 1;
    if (all_right) g_stage_variants.dims_stages_right[stage] += 1;
}

std::string stage_variant_report() {
    std::lock_guard<std::mutex> lk(g_stage_variants.mu);
    std::string out;
    static const char* const kStage[2] = {"VS", "PS"};
    for (int st = 0; st < 2; ++st) {
        std::uint64_t by_count[5] = {};
        for (const auto& [program, keys] : g_stage_variants.keys_by_program[st]) by_count[std::min<std::size_t>(keys.size(), 4)] += 1;
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      " %s: %zu programs (1 variant %llu, 2 %llu, 3 %llu, 4+ %llu); %llu translations, %llu of them no-fallback, %llu for a "
                      "program already translated with other options;",
                      kStage[st], g_stage_variants.keys_by_program[st].size(), static_cast<unsigned long long>(by_count[1]),
                      static_cast<unsigned long long>(by_count[2]), static_cast<unsigned long long>(by_count[3]),
                      static_cast<unsigned long long>(by_count[4]), static_cast<unsigned long long>(g_stage_variants.translations[st]),
                      static_cast<unsigned long long>(g_stage_variants.lean[st]),
                      static_cast<unsigned long long>(g_stage_variants.program_seen_before[st]));
        out += buf;
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  " VS translations with a fetch shader %llu, with vertex input %llu; PS translations with every image 2D and not arrayed "
                  "%llu (no images %llu), default sampler modes %llu, identity input map %llu, no flat inputs %llu, early fragment tests %llu",
                  static_cast<unsigned long long>(g_stage_variants.vs_with_fetch), static_cast<unsigned long long>(g_stage_variants.vs_with_vertex_input),
                  static_cast<unsigned long long>(g_stage_variants.ps_dims_all_2d), static_cast<unsigned long long>(g_stage_variants.ps_no_images),
                  static_cast<unsigned long long>(g_stage_variants.ps_modes_default), static_cast<unsigned long long>(g_stage_variants.ps_map_identity),
                  static_cast<unsigned long long>(g_stage_variants.ps_flat_none), static_cast<unsigned long long>(g_stage_variants.ps_early_tests));
    out += buf;
    {
        std::lock_guard<std::mutex> lk(g_creations.mu);
        for (int st = 0; st < 2; ++st) {
            std::vector<std::uint64_t> ahead = g_creations.flips_ahead[st];
            std::sort(ahead.begin(), ahead.end());
            const auto pct = [&](double q) { return ahead.empty() ? 0ull : static_cast<unsigned long long>(ahead[static_cast<std::size_t>(q * (ahead.size() - 1))]); };
            std::uint64_t zero = 0;
            for (std::uint64_t a : ahead) zero += a == 0;
            // Logged on their own: the stage report line is long enough to be cut.
            std::string misses;
            for (const std::string& n : g_creations.misses[st]) misses += " " + n;
            host_log("render: %s created by GX: %llu containers, %zu programs (%llu unnamed); first translations %llu, of them created earlier "
                     "%llu (flips ahead: same flip %llu, median %llu, p10 %llu, p90 %llu); found only among the other stage's %llu; never seen:%s",
                     st ? "PS" : "VS", static_cast<unsigned long long>(g_creations.calls[st]), g_creations.by_name[st].size(),
                     static_cast<unsigned long long>(g_creations.unnamed[st]), static_cast<unsigned long long>(g_creations.first_translations[st]),
                     static_cast<unsigned long long>(g_creations.created_before[st]), static_cast<unsigned long long>(zero), pct(0.5), pct(0.1),
                     pct(0.9), static_cast<unsigned long long>(g_creations.other_stage_hits[st]), misses.c_str());
            if (st == 1) {
                static const char* const kCreator[4] = {"0x2566d00", "0x2566f20", "0x25672d0", "0x25673e0"};
                std::string types;
                for (int h = 0; h < 4; ++h) {
                    types += std::string(" ") + kCreator[h] + ":";
                    for (int t = 0; t < 16; ++t) {
                        if (g_creations.types[h][t]) types += " type " + std::to_string(t) + " x" + std::to_string(g_creations.types[h][t]);
                    }
                }
                host_log("render: GX shader creators by container type:%s", types.c_str());
            }
        }
    }
    vs_dependency_report();
    precompile_report();
    if (g_libs.links.load() || g_libs.built[0].load()) {
        std::snprintf(buf, sizeof(buf),
                      "; pipeline libraries built (hits): vertex input %llu (%llu) %.1f s, pre-rasterization %llu (%llu) %.1f s, fragment shader %llu "
                      "(%llu) %.1f s, fragment output %llu (%llu) %.1f s; %llu links in %.1f s",
                      static_cast<unsigned long long>(g_libs.built[0].load()), static_cast<unsigned long long>(g_libs.hits[0].load()),
                      g_libs.us[0].load() / 1e6, static_cast<unsigned long long>(g_libs.built[1].load()),
                      static_cast<unsigned long long>(g_libs.hits[1].load()), g_libs.us[1].load() / 1e6,
                      static_cast<unsigned long long>(g_libs.built[2].load()), static_cast<unsigned long long>(g_libs.hits[2].load()),
                      g_libs.us[2].load() / 1e6, static_cast<unsigned long long>(g_libs.built[3].load()),
                      static_cast<unsigned long long>(g_libs.hits[3].load()), g_libs.us[3].load() / 1e6,
                      static_cast<unsigned long long>(g_libs.links.load()), g_libs.link_us.load() / 1e6);
        out += buf;
    }
    std::snprintf(buf, sizeof(buf),
                  "; image dims predicted from programs: VS %llu of %llu T#s (%llu of %llu stages), PS %llu of %llu T#s (%llu of %llu stages), "
                  "conflicting T#s VS %llu PS %llu",
                  static_cast<unsigned long long>(g_stage_variants.dims_images_right[0]), static_cast<unsigned long long>(g_stage_variants.dims_images[0]),
                  static_cast<unsigned long long>(g_stage_variants.dims_stages_right[0]), static_cast<unsigned long long>(g_stage_variants.dims_stages[0]),
                  static_cast<unsigned long long>(g_stage_variants.dims_images_right[1]), static_cast<unsigned long long>(g_stage_variants.dims_images[1]),
                  static_cast<unsigned long long>(g_stage_variants.dims_stages_right[1]), static_cast<unsigned long long>(g_stage_variants.dims_stages[1]),
                  static_cast<unsigned long long>(g_stage_variants.dims_conflicts[0]), static_cast<unsigned long long>(g_stage_variants.dims_conflicts[1]));
    return out + buf;
}

bool create_gfx_pipeline(GfxPipeline& pl, const DrawState& s, const VertexInputPlan* vertex_input, std::size_t fetch_dwords);

// Linking by register (per-shader compiles, step 6a): every vertex shader
// writes param register r at location r (identity output links), and a pixel
// shader reads its input k at location SPI_PS_INPUT_CNTL_k & 0x1f, the register
// the input map names (OFFSET, bits 0-4, as the input map always read it).
// Neither stage depends on its partner. A pixel shader compiled when GX creates
// it predicts each register as its input's semantic - 0xf, which held for every
// input of a cold world load.
const std::vector<std::uint8_t>& register_links() {
    static const std::vector<std::uint8_t> links = [] {
        std::vector<std::uint8_t> l(32);
        for (std::uint8_t r = 0; r < 32; ++r) l[r] = r;
        return l;
    }();
    return links;
}
// BBHOST_VERTEX_FORMAT_PARAMS=0 (checks): vertex shaders on vertex input convert
// each element with its formats as constants, one translation per format set,
// instead of reading them from their params (step 6c).
const bool g_vertex_format_params = [] {
    const char* e = std::getenv("BBHOST_VERTEX_FORMAT_PARAMS");
    return !(e && e[0] == '0');
}();
// BBHOST_PRECOMPILE_CHECK=1: a draw that finds a stage compiled at creation
// translates it again with its own options and logs any difference.
const bool g_precompile_check = [] {
    const char* e = std::getenv("BBHOST_PRECOMPILE_CHECK");
    return e && e[0] == '1';
}();
void check_precompiled_stage(const char* stage, const std::string& name, const gcn::TranslateResult& cached, const gcn::TranslateResult& again) {
    static std::atomic<int> checked{0}, differing{0};
    checked.fetch_add(1, std::memory_order_relaxed);
    const bool same_bindings = cached.images.size() == again.images.size() && cached.samplers.size() == again.samplers.size() &&
                               cached.buffers.size() == again.buffers.size();
    if (cached.spirv == again.spirv && same_bindings) return;
    std::size_t at = 0;
    while (at < cached.spirv.size() && at < again.spirv.size() && cached.spirv[at] == again.spirv[at]) ++at;
    if (differing.fetch_add(1) < 24) {
        host_log("render: precompile check: %s %s compiled at creation differs from its draw's translation: SPIR-V words %zu against %zu, "
                 "first difference at word %zu; images %zu/%zu, samplers %zu/%zu, buffers %zu/%zu (%d checked)",
                 stage, name.c_str(), cached.spirv.size(), again.spirv.size(), at, cached.images.size(), again.images.size(),
                 cached.samplers.size(), again.samplers.size(), cached.buffers.size(), again.buffers.size(), checked.load());
    }
}
// A pixel shader's input locations (TranslateOptions::ps_input_map), up to the
// last input it reads: `registers[k]` for inputs it reads, k for the others.
std::vector<std::uint8_t> ps_input_locations(std::uint32_t read_inputs, const std::uint32_t* registers) {
    std::vector<std::uint8_t> map;
    for (std::uint32_t k = 0; k < 32; ++k) {
        if (!((read_inputs >> k) & 1)) continue;
        while (map.size() < k) map.push_back(static_cast<std::uint8_t>(map.size()));
        map.push_back(static_cast<std::uint8_t>(registers[k] & 0x1f));
    }
    return map;
}

// A pixel shader's stage translation: what its key and options are made of.
// A draw fills it from the draw state; per-shader compiles (step 4) fill it
// from the shader's binary when GX creates it, so both reach the same cached
// stage and fragment-shader library.
struct PsStageInputs {
    const std::vector<std::uint32_t>* words;
    std::uint32_t rsrc1, rsrc2, ps_input_ena, flat_mask;
    bool early_fragment_tests, no_fallback;
    const std::vector<std::pair<std::uint32_t, bool>>* dims;
    const std::vector<bool>* modes;
    const std::vector<std::uint8_t>* input_map;  // ps_input_locations
    std::uint32_t clip_discard = 0;              // TranslateOptions::ps_clip_discard
};
std::uint64_t ps_stage_key(const PsStageInputs& in) {
    std::uint64_t key = fnv1a(in.words->data(), in.words->size() * 4, 0x9e3779b97f4a7c15ull);
    const std::uint32_t pk[4] = {in.rsrc1, in.rsrc2, in.ps_input_ena, in.flat_mask};
    key = fnv1a(pk, sizeof(pk), key);
    key = fnv1a(&in.early_fragment_tests, 1, key);
    key = fnv1a(&in.no_fallback, 1, key);
    key = fnv1a(in.input_map->data(), in.input_map->size(), key);
    if (in.clip_discard) key = fnv1a(&in.clip_discard, sizeof(in.clip_discard), key ^ 0xc11bd15cull);
    if (g.bindless) key ^= 0xb1d1e55b1d1e55ull;  // another translation
    return dims_modes_key(key, *in.dims, *in.modes);
}
gcn::TranslateOptions ps_translate_options(const PsStageInputs& in) {
    gcn::TranslateOptions po;
    po.stage = gcn::Stage::Pixel;
    po.rsrc1 = in.rsrc1;
    po.rsrc2 = in.rsrc2;
    po.ps_input_ena = in.ps_input_ena;
    po.descriptor_set = 1;
    if (cb_push_on()) {
        po.cb_descriptor_set = 2;
        po.cb_binding_base = gcn::kBindingStorageBuffer0 + gcn::kMaxBuffers;
    }
    po.image_dims = *in.dims;
    po.sampler_force_unnormalized = *in.modes;
    po.bindless = g.bindless;
    po.cb_ssbo = g.cb_ssbo;
    po.cb_no_fallback = in.no_fallback;
    po.exec_known = g.exec_known;
    po.early_fragment_tests = in.early_fragment_tests;
    po.debug_ps = g_debug_ps_color;
    po.ps_flat_mask = in.flat_mask;
    po.ps_input_map = *in.input_map;
    po.ps_clip_discard = in.clip_discard;
    return po;
}
// BBHOST_CLIP_VARYING=0 (checks): clip distances go to the clipper again
// (TranslateOptions::kVsOutCntlClipVarying).
const bool g_clip_varying = [] {
    const char* e = std::getenv("BBHOST_CLIP_VARYING");
    return !(e && e[0] == '0');
}();
// A draw's pixel shader discarding by its clip distances
// (TranslateOptions::kVsOutCntlClipVarying): ps_clip_discard - the count | the
// location << 8 - or 0 to leave them to the clipper. Distances 0-3 alone, no
// cull distances, at the first location past every param the vertex shader
// exports and the pixel shader reads, and one that fits under the fragment
// stage's 128 input components with its built-ins (29 at most).
std::uint32_t ps_clip_discard_count(std::uint32_t vs_out_cntl, const std::vector<std::uint32_t>& vs_words,
                                    const std::vector<std::uint8_t>& ps_input_map) {
    if (!g_clip_varying || !(vs_out_cntl & (1u << 22)) || !(vs_out_cntl & 0xf) || (vs_out_cntl & 0xfff0) || (vs_out_cntl & (1u << 23))) return 0;
    int used = -1;
    for (std::uint8_t l : ps_input_map) {
        if (l < 32) used = std::max(used, static_cast<int>(l));
    }
    const gcn::Program p = gcn::decode(vs_words.data(), vs_words.size());
    for (const gcn::Inst& in : p.insts) {
        if (in.enc == gcn::Enc::EXP && in.tgt >= 32 && in.tgt < 64) used = std::max(used, static_cast<int>(in.tgt) - 32);
    }
    const int location = used + 1;
    if (location > 29) return 0;
    std::uint32_t n = 0;
    for (std::uint32_t k = 0; k < 4; ++k) {
        if ((vs_out_cntl >> k) & 1) n = k + 1;
    }
    return n | static_cast<std::uint32_t>(location) << 8;
}
// The inputs a pixel shader interpolates (bit k: input k).
std::uint32_t ps_read_inputs(const std::vector<std::uint32_t>& words) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    std::uint32_t mask = g_debug_ps_color == 3 ? 2u : 0u;  // the UV view reads input 1
    for (const gcn::Inst& in : p.insts) {
        if (in.enc == gcn::Enc::VINTRP && in.attr < 32) mask |= 1u << in.attr;
    }
    return mask;
}
// Early fragment tests change a pixel shader's result only when it discards
// without replacing depth (the translator then emits none), and whether it can
// discard is up to the translator's EXEC analysis: any translation of the
// program tells, by an OpKill in its SPIR-V. A program known not to discard is
// keyed and translated without them, so draws with and without depth writes
// share its stage, and per-shader compiles build that one.
struct PsDiscards {
    std::mutex mu;
    std::unordered_map<std::uint64_t, bool> by_program;  // program hash -> early tests matter, under mu
} g_ps_discards;
bool spirv_early_tests_matter(const std::vector<std::uint32_t>& spirv) {
    bool kill = false, depth = false;
    for (std::size_t i = 5; i < spirv.size();) {
        const std::uint32_t n = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (n == 0) break;
        if (op == 252 || op == 4416) kill = true;                                             // OpKill, OpTerminateInvocation
        if (op == 16 && n >= 3 && i + 2 < spirv.size() && spirv[i + 2] == 12) depth = true;  // OpExecutionMode DepthReplacing
        i += n;
    }
    return kill && !depth;
}
// Whether a module declares OpExecutionMode EarlyFragmentTests: what a pixel
// shader was translated with, which a capture records for gcnlift.
bool spirv_has_early_fragment_tests(const std::vector<std::uint32_t>& spirv) {
    for (std::size_t i = 5; i < spirv.size();) {
        const std::uint32_t n = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (n == 0) break;
        if (op == 16 && n >= 3 && i + 2 < spirv.size() && spirv[i + 2] == 9) return true;  // OpExecutionMode EarlyFragmentTests
        i += n;
    }
    return false;
}
// The same module without OpExecutionMode EarlyFragmentTests - what the
// translators emit with early_fragment_tests off, since that mode is the
// option's only effect (translate.cpp, lift.cpp). False when it had none.
bool spirv_drop_early_fragment_tests(std::vector<std::uint32_t>& spirv) {
    for (std::size_t i = 5; i < spirv.size();) {
        const std::uint32_t n = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (n == 0) break;
        if (op == 16 && n >= 3 && i + 2 < spirv.size() && spirv[i + 2] == 9) {
            spirv.erase(spirv.begin() + static_cast<std::ptrdiff_t>(i), spirv.begin() + static_cast<std::ptrdiff_t>(i + n));
            return true;
        }
        i += n;
    }
    return false;
}
void note_ps_early_tests(const std::vector<std::uint32_t>& words, const std::vector<std::uint32_t>& spirv) {
    if (spirv.empty()) return;
    const std::uint64_t program = fnv1a(words.data(), words.size() * 4);
    const bool matter = spirv_early_tests_matter(spirv);
    std::lock_guard<std::mutex> lk(g_ps_discards.mu);
    const auto [it, fresh] = g_ps_discards.by_program.emplace(program, matter);
    if (!fresh) it->second = it->second || matter;
}
int ps_early_tests_matter(const std::vector<std::uint32_t>& words) {  // -1 not known yet
    const std::uint64_t program = fnv1a(words.data(), words.size() * 4);
    std::lock_guard<std::mutex> lk(g_ps_discards.mu);
    const auto it = g_ps_discards.by_program.find(program);
    return it == g_ps_discards.by_program.end() ? -1 : it->second ? 1 : 0;
}
bool note_precompile_hit(std::uint64_t ps_key);
void note_precompile_miss(const std::string& name, const PsStageInputs& in);

// A vertex shader's stage translation (step 6d), as PsStageInputs: a draw fills
// it from the draw state, and per-shader compiles from the shader's binary
// (elements from its input table, registers from its register block).
struct VsStageInputs {
    const std::vector<std::uint32_t>* words;
    const std::vector<std::uint32_t>* fetch_words;  // translated only without vertex input
    std::uint32_t rsrc1, rsrc2, out_cntl;
    const std::vector<gcn::VertexElement>* elements;  // vertex input; null without
    bool formats_from_params, no_fallback;
    const std::vector<std::pair<std::uint32_t, bool>>* dims;
    const std::vector<bool>* modes;
    std::uint32_t domain_level = 0;  // DrawState::domain_level: the domain shader of a tessellated draw
    // The same draw through the host's tessellator.
    bool tess_hw = false;
    std::uint32_t tess_control_points = 1;
    bool tess_quads = true, tess_cw = true;
    int tess_spacing = 0;
    bool tess_attrs = false;
    std::uint32_t tess_attr_vec4s = 8;
    std::uint32_t tess_window = 0;  // TranslateOptions::tess_window: the game's own hull runs
};
std::uint64_t vs_stage_key(const VsStageInputs& in) {
    std::uint64_t key = fnv1a(in.words->data(), in.words->size() * 4);
    if (!in.elements) key = fnv1a(in.fetch_words->data(), in.fetch_words->size() * 4, key);  // on vertex input the fetch shader is not translated
    const std::uint32_t vk[5] = {in.rsrc1, in.rsrc2, in.out_cntl, in.elements ? 1u : 0u, in.formats_from_params ? 1u : 0u};
    key = fnv1a(vk, sizeof(vk), key);
    if (in.elements) {
        for (const gcn::VertexElement& el : *in.elements) {
            const std::uint32_t e[4] = {el.location, el.vdata, el.count, in.formats_from_params ? 0u : el.w3};
            key = fnv1a(e, sizeof(e), key);
        }
    }
    key = fnv1a(&in.no_fallback, 1, key);
    if (in.domain_level) key = fnv1a(&in.domain_level, 4, key ^ 0x7e55u);
    if (in.tess_hw) {
        key = fnv1a(&in.tess_control_points, 4, key ^ 0x7e56u);
        key = fnv1a(&in.tess_attrs, 1, key);
        key = fnv1a(&in.tess_attr_vec4s, 4, key);
        if (in.tess_window || !in.tess_quads || !in.tess_cw || in.tess_spacing) {
            const std::uint32_t shape[4] = {in.tess_window, in.tess_quads ? 1u : 0u, in.tess_cw ? 1u : 0u,
                                            static_cast<std::uint32_t>(in.tess_spacing)};
            key = fnv1a(shape, sizeof(shape), key ^ 0x7e57u);
        }
    }
    if (g.bindless) key ^= 0xb1d1e55b1d1e55ull;  // another translation
    return dims_modes_key(key, *in.dims, *in.modes);
}
// The host tessellator's LS as a vertex stage, by what its
// translation takes. Every tessellated pipeline translated its LS again, and
// a fight's new particle effects built 14 of them in one frame.
std::uint64_t tess_ls_key(const std::vector<std::uint32_t>& words, const std::vector<std::uint32_t>* fetch, std::uint32_t rsrc1,
                          std::uint32_t rsrc2, std::uint32_t attr_vec4s) {
    std::uint64_t key = fnv1a(words.data(), words.size() * 4, 0x7e55150a11c0de5ull);
    if (fetch) key = fnv1a(fetch->data(), fetch->size() * 4, key);
    const std::uint32_t k[5] = {rsrc1, rsrc2, attr_vec4s, g.bindless ? 1u : 0u, g.exec_known ? 1u : 0u};
    return fnv1a(k, sizeof(k), key);
}
gcn::TranslateOptions tess_ls_options(std::uint32_t rsrc1, std::uint32_t rsrc2, std::uint32_t attr_vec4s, const gcn::Program* fetch) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.bindless = g.bindless;  // a stage of the same pipeline layout
    o.tess_role = gcn::TranslateOptions::TessRole::LsVertex;
    o.tess_lds_attributes = true;
    o.tess_attr_vec4s = attr_vec4s;
    o.rsrc1 = rsrc1;
    o.rsrc2 = rsrc2;
    o.fetch = fetch;
    o.exec_known = g.exec_known;
    o.descriptor_set = 0;
    return o;
}
// The two stages of a tessellated draw that carry no GCN code, made once:
// the control stage per set of factors (every pipeline of a hull shader
// shares one), the pass-through vertex stage once. Never destroyed, as no
// pipeline's modules are.
VkShaderModule tess_tcs_module(const float outer[4], const float inner[2], std::uint32_t control_points, std::uint32_t attr_vec4s,
                               const std::string& dump_name) {
    static std::mutex mu;
    static std::unordered_map<std::uint64_t, VkShaderModule> modules;  // under mu
    struct {
        float outer[4], inner[2];
        std::uint32_t control_points, attr_vec4s;
    } k{};
    std::memcpy(k.outer, outer, sizeof(k.outer));
    std::memcpy(k.inner, inner, sizeof(k.inner));
    k.control_points = control_points;
    k.attr_vec4s = attr_vec4s;
    const std::uint64_t key = fnv1a(&k, sizeof(k));
    std::lock_guard<std::mutex> lk(mu);
    if (const auto it = modules.find(key); it != modules.end()) return it->second;
    const std::vector<std::uint32_t> tcs = gcn::make_tess_constant_tcs(outer, inner, control_points, attr_vec4s);
    dump_spirv(dump_name + "-tess-tcs", tcs);
    VkShaderModule m = VK_NULL_HANDLE;
    if (!make_module(tcs, m)) return VK_NULL_HANDLE;
    modules.emplace(key, m);
    return m;
}
VkShaderModule tess_passthrough_module(const std::string& dump_name) {
    static std::mutex mu;
    static VkShaderModule module = VK_NULL_HANDLE;  // under mu
    std::lock_guard<std::mutex> lk(mu);
    if (!module) {
        const std::vector<std::uint32_t> tvs = gcn::make_tess_passthrough_vs();
        dump_spirv(dump_name + "-tess-vs", tvs);
        if (!make_module(tvs, module)) module = VK_NULL_HANDLE;
    }
    return module;
}

// The game's own hull shader as a tessellation control stage (DrawState::
// tess_hull), one per program and shape. It binds nothing: its constant
// buffers are read through the page table, and its user data and the LDS
// buffer come from the evaluation stage's params block. Never destroyed, as
// above.
VkShaderModule tess_hull_module(const DrawState& s, const std::string& dump_name) {
    CachedProgram* hs = program_at(s.hs_va);
    if (!hs) return VK_NULL_HANDLE;
    static std::mutex mu;
    static std::unordered_map<std::uint64_t, VkShaderModule> modules;  // under mu; VK_NULL_HANDLE: failed
    const std::uint32_t k[6] = {s.hs_rsrc1, s.hs_rsrc2, s.tess_window, s.tess_control_points,
                                (s.tess_quads ? 1u : 0u) | (s.tess_cw ? 2u : 0u) | static_cast<std::uint32_t>(s.tess_spacing) << 2,
                                g.exec_known ? 1u : 0u};
    const std::uint64_t key = fnv1a(k, sizeof(k), hs->hash ^ 0x4a11ull);
    std::lock_guard<std::mutex> lk(mu);
    if (const auto it = modules.find(key); it != modules.end()) return it->second;
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::TessControl;
    o.tess_role = gcn::TranslateOptions::TessRole::HullTcs;
    o.rsrc1 = s.hs_rsrc1;
    o.rsrc2 = s.hs_rsrc2;
    o.tess_patch_control_points = s.tess_control_points;
    o.tess_window = s.tess_window;
    o.tess_lds_bound = tess_lds_bound();
    o.tess_quads = s.tess_quads;
    o.tess_spacing = s.tess_spacing;
    o.tess_cw = s.tess_cw;
    o.exec_known = g.exec_known;
    o.descriptor_set = 0;
    const gcn::Program prog = gcn::decode(hs->words.data(), hs->words.size());
    const gcn::TranslateResult r = translate_cached(prog, o);
    VkShaderModule m = VK_NULL_HANDLE;
    if (!prog.errors.empty() || !r.ok()) {
        host_log("render: hull shader %s as a control stage: %s", hs->name.c_str(),
                 !prog.errors.empty() ? prog.errors[0].what.c_str() : r.errors[0].c_str());
    } else if (!r.images.empty() || !r.samplers.empty() || !r.buffers.empty()) {
        host_log("render: hull shader %s binds resources (%zu images, %zu samplers, %zu buffers), which would collide with the "
                 "evaluation stage's set", hs->name.c_str(), r.images.size(), r.samplers.size(), r.buffers.size());
    } else {
        dump_spirv(dump_name + "-tess-hs", r.spirv);
        if (!make_module(r.spirv, m)) m = VK_NULL_HANDLE;
    }
    modules.emplace(key, m);
    return m;
}

// The stage manifest (<data root>/bbhost/stage-manifest.bin, BBHOST_STAGE_MANIFEST=0
// turns it off): every stage a draw had to translate itself - the eboot's own
// shaders, which GX never creates (YEBIS, Scaleform, the engine's full-screen
// passes: 59 a soak), and the variants creation-time compiling does not guess -
// with the inputs its translation took. The next start compiles them on the
// precompile workers before the title, so their draws only link. It is kept
// beside the pipeline cache, written whole whenever a 300-flip report finds
// stages recorded since the last write, and at exit; like the pipeline cache
// it is a cache: the stages are translated again, by the running build, from
// the inputs it holds, and one that no longer compiles is dropped.
constexpr std::uint64_t kManifestPathsSeed = 0x70a7b5d1c3e5f701ull;  // a paths entry's key: the program's words from this seed
struct ManifestStage {
    std::uint8_t stage = 0;  // 0 vertex, 1 pixel, 2 a program's resource paths only (its gcn::Stage in domain_level),
                             // 3 the host tessellator's LS vertex stage (tess_ls_key)
    bool lift = false;
    std::string name;
    std::vector<std::uint32_t> words, fetch_words;
    std::uint32_t rsrc1 = 0, rsrc2 = 0, ps_input_ena = 0, flat_mask = 0, out_cntl = 0;
    bool early_fragment_tests = false, no_fallback = false, has_elements = false, formats_from_params = false;
    std::vector<gcn::VertexElement> elements;
    std::vector<std::pair<std::uint32_t, bool>> dims;
    std::vector<bool> modes;
    std::vector<std::uint8_t> input_map;
    std::uint32_t domain_level = 0, tess_control_points = 1, tess_attr_vec4s = 8;
    bool tess_hw = false, tess_attrs = false;
    PsStageInputs ps_inputs() const {
        // A pixel stage keeps its clip_discard where a vertex stage keeps out_cntl.
        return PsStageInputs{&words, rsrc1, rsrc2, ps_input_ena, flat_mask, early_fragment_tests, no_fallback, &dims, &modes, &input_map, out_cntl};
    }
    VsStageInputs vs_inputs() const {
        VsStageInputs in{&words, &fetch_words, rsrc1, rsrc2, out_cntl, has_elements ? &elements : nullptr, formats_from_params, no_fallback,
                         &dims, &modes, domain_level};
        in.tess_hw = tess_hw;
        in.tess_control_points = tess_control_points;
        in.tess_attrs = tess_attrs;
        in.tess_attr_vec4s = tess_attr_vec4s;
        return in;
    }
    std::uint64_t key() const {
        if (stage == 2) return fnv1a(words.data(), words.size() * 4, kManifestPathsSeed);
        if (stage == 3) return tess_ls_key(words, fetch_words.empty() ? nullptr : &fetch_words, rsrc1, rsrc2, tess_attr_vec4s);
        return stage ? lift_key(ps_stage_key(ps_inputs()), lift) : lift_key(vs_stage_key(vs_inputs()), lift);
    }
};
struct StageManifest {
    std::mutex mu;
    std::unordered_map<std::uint64_t, std::shared_ptr<const ManifestStage>> by_key;  // under mu
    std::unordered_set<std::uint64_t> ready;                                         // under mu: compiled at this start
    bool dirty = false;                                                               // under mu
    std::atomic<std::uint64_t> loaded{0}, compiled{0}, failed{0}, recorded{0}, hits{0}, compile_us{0};
};
StageManifest g_manifest;
bool stage_manifest_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_STAGE_MANIFEST");
        return !(e && e[0] == '0');
    }();
    return on;
}
constexpr std::size_t kManifestMax = 16384;
void manifest_add(std::uint64_t key, std::shared_ptr<const ManifestStage> m) {
    std::lock_guard<std::mutex> lk(g_manifest.mu);
    if (g_manifest.by_key.size() >= kManifestMax || !g_manifest.by_key.emplace(key, std::move(m)).second) return;
    g_manifest.dirty = true;
    g_manifest.recorded.fetch_add(1, std::memory_order_relaxed);
}
void manifest_note_ps(std::uint64_t key, const std::string& name, const PsStageInputs& in, bool lift) {
    if (!stage_manifest_on()) return;
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (g_manifest.by_key.count(key)) return;
    }
    auto m = std::make_shared<ManifestStage>();
    m->stage = 1;
    m->lift = lift;
    m->name = name;
    m->words = *in.words;
    m->rsrc1 = in.rsrc1;
    m->rsrc2 = in.rsrc2;
    m->ps_input_ena = in.ps_input_ena;
    m->flat_mask = in.flat_mask;
    m->out_cntl = in.clip_discard;
    m->early_fragment_tests = in.early_fragment_tests;
    m->no_fallback = in.no_fallback;
    m->dims = *in.dims;
    m->modes = *in.modes;
    m->input_map = *in.input_map;
    manifest_add(key, std::move(m));
}
void manifest_note_vs(std::uint64_t key, const std::string& name, const VsStageInputs& in, bool lift) {
    if (!stage_manifest_on()) return;
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (g_manifest.by_key.count(key)) return;
    }
    auto m = std::make_shared<ManifestStage>();
    m->stage = 0;
    m->lift = lift;
    m->name = name;
    m->words = *in.words;
    if (!in.elements) m->fetch_words = *in.fetch_words;
    m->rsrc1 = in.rsrc1;
    m->rsrc2 = in.rsrc2;
    m->out_cntl = in.out_cntl;
    m->has_elements = in.elements != nullptr;
    if (in.elements) m->elements = *in.elements;
    m->formats_from_params = in.formats_from_params;
    m->no_fallback = in.no_fallback;
    m->dims = *in.dims;
    m->modes = *in.modes;
    m->domain_level = in.domain_level;
    m->tess_hw = in.tess_hw;
    m->tess_control_points = in.tess_control_points;
    m->tess_attrs = in.tess_attrs;
    m->tess_attr_vec4s = in.tess_attr_vec4s;
    manifest_add(key, std::move(m));
}
// A program's resource paths that paths_for had to translate on the command
// processor - a compute shader GX never creates, the eboot's own programs: at
// the first dispatches of a fight's new effects, 16% of that frame.
void manifest_note_paths(const std::vector<std::uint32_t>& words, gcn::Stage stage, std::uint32_t rsrc1, std::uint32_t rsrc2) {
    if (!stage_manifest_on()) return;
    const std::uint64_t key = fnv1a(words.data(), words.size() * 4, kManifestPathsSeed);
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (g_manifest.by_key.count(key)) return;
    }
    auto m = std::make_shared<ManifestStage>();
    m->stage = 2;
    m->domain_level = static_cast<std::uint32_t>(stage);
    m->words = words;
    m->rsrc1 = rsrc1;
    m->rsrc2 = rsrc2;
    manifest_add(key, std::move(m));
}
// A tessellated draw's LS translated for its pipeline, for the next start.
void manifest_note_ls(std::uint64_t key, const std::string& name, const std::vector<std::uint32_t>& words,
                      const std::vector<std::uint32_t>* fetch, std::uint32_t rsrc1, std::uint32_t rsrc2, std::uint32_t attr_vec4s) {
    if (!stage_manifest_on()) return;
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (g_manifest.by_key.count(key)) return;
    }
    auto m = std::make_shared<ManifestStage>();
    m->stage = 3;
    m->name = name;
    m->words = words;
    if (fetch) m->fetch_words = *fetch;
    m->rsrc1 = rsrc1;
    m->rsrc2 = rsrc2;
    m->tess_attr_vec4s = attr_vec4s;
    manifest_add(key, std::move(m));
}
// A draw took a stage from the stage cache: count it when the manifest compiled it.
void manifest_note_hit(std::uint64_t key) {
    if (!stage_manifest_on()) return;
    std::lock_guard<std::mutex> lk(g_manifest.mu);
    if (g_manifest.ready.count(key)) g_manifest.hits.fetch_add(1, std::memory_order_relaxed);
}

// BBHOST_TESS_PATCH_CULL=<clip w>: a patch whose centre is nearer the camera
// plane than this is not drawn. A camera-facing quad beside or behind the eye
// has no projection - the clipper turns it into a wedge across the frame, and
// with the particles' additive blending the frame washes out. 0 turns it off.
float tess_patch_cull_near() {
    static const float w = [] {
        const char* e = std::getenv("BBHOST_TESS_PATCH_CULL");
        return e ? static_cast<float>(std::atof(e)) : 0.0f;  // off: it does not stop the wash-out
    }();
    return w;
}

gcn::TranslateOptions vs_translate_options(const VsStageInputs& in, const gcn::Program* fetch) {
    gcn::TranslateOptions vo;
    vo.stage = gcn::Stage::Vertex;
    vo.rsrc1 = in.rsrc1;
    vo.rsrc2 = in.rsrc2;
    vo.vs_out_cntl = in.out_cntl;
    vo.fetch = in.elements ? nullptr : fetch;
    if (in.elements) vo.vertex_input = *in.elements;
    vo.vertex_formats_from_params = in.formats_from_params;
    vo.force_dispatcher = in.elements && g_vs_dispatcher;
    vo.invariant_position = g_vs_invariant;
    vo.link_outputs = true;
    vo.output_links = register_links();
    vo.descriptor_set = 0;
    if (cb_push_on()) vo.cb_descriptor_set = 2;  // bindings from 112, the default
    vo.image_dims = *in.dims;
    vo.sampler_force_unnormalized = *in.modes;
    vo.bindless = g.bindless;
    vo.cb_ssbo = g.cb_ssbo;
    vo.cb_no_fallback = in.no_fallback;
    vo.exec_known = g.exec_known;
    if (in.tess_hw || in.domain_level) {
        // A domain shader has no vertex input: its control points come from
        // the buffer the LS compute pass wrote. Carrying the vertex path's
        // options into it makes the shader read a params member that only a
        // vertex or compute stage declares, which is SPIR-V the driver is
        // free to fall over on - and did.
        vo.vertex_input.clear();
        vo.vertex_formats_from_params = false;
        vo.force_dispatcher = false;
    }
    if (in.tess_hw) {
        // The host's tessellator decides which domain points exist; the shader
        // only has to read the one it is given.
        vo.stage = gcn::Stage::TessEval;
        vo.tess_role = gcn::TranslateOptions::TessRole::DomainTes;
        vo.tess_patch_control_points = in.tess_control_points;
        vo.tess_quads = in.tess_quads;
        vo.tess_spacing = in.tess_spacing;
        vo.tess_cw = in.tess_cw;
        vo.tess_window = in.tess_window;
        vo.tess_lds_bound = tess_lds_bound();
        // The control point arrives as an attribute rather than through a
        // buffer, so the stage reads its own patch and nothing else.
        vo.tess_lds_attributes = in.tess_attrs;
        vo.tess_attr_vec4s = in.tess_attr_vec4s;
    } else if (in.domain_level) {
        vo.tess_role = gcn::TranslateOptions::TessRole::Domain;
        vo.domain_level = in.domain_level;
        vo.tess_lds_bound = tess_lds_bound();
        static const float near_cull = [] {
            const char* e = std::getenv("BBHOST_TESS_NEAR_CULL");
            return e ? static_cast<float>(std::atof(e)) : 0.0f;
        }();
        vo.tess_near_cull = near_cull;
        vo.tess_patch_cull = tess_patch_cull_near() != 0.0f;
        vo.tess_patch_cull_all = tess_patch_cull_near() < 0.0f;
    }
    return vo;
}
bool note_vs_precompile_hit(std::uint64_t vs_key);
void note_vs_precompile_miss(const std::string& name, const VsStageInputs& in);

// Translates a pipeline's stages and creates their layouts; with `create`,
// its Vulkan pipeline too (create_gfx_pipeline).
GfxPipeline& build_gfx_pipeline(GfxPipeline& pl, const DrawState& s, const std::vector<std::uint32_t>& vs_words,
                                const std::vector<std::uint32_t>& fetch_words, const std::vector<std::uint32_t>& ps_words,
                                const std::string& vs_name, const std::string& ps_name,
                                const std::vector<std::pair<std::uint32_t, bool>>& vs_dims,
                                const std::vector<std::pair<std::uint32_t, bool>>& ps_dims,
                                const std::vector<bool>& vs_sampler_modes, const std::vector<bool>& ps_sampler_modes,
                                bool cb_no_fallback, const VertexInputPlan* vertex_input, bool create = true, bool layout_only = false) {
    pl.name = vs_name + "+" + ps_name;
    pl.lean = cb_no_fallback;
    // layout_only: the fallback pipeline laid out from its no-fallback stages,
    // which bind the same resources; the fallback's own translation waits for
    // a draw that has to create it (fallback_for_draw).
    const bool stage_lean = cb_no_fallback || layout_only;
    pl.layout_only = layout_only;
    auto fail = [&](const std::string& why) -> GfxPipeline& {
        host_log("render: pipeline %s: %s", pl.name.c_str(), why.c_str());
        pl.failed = true;
        // BBHOST_TRACE_DRAW=<name prefix>: disassemble the failing stages.
        static const char* trace = std::getenv("BBHOST_TRACE_DRAW");
        if (trace && pl.name.find(trace) == 0) {
            const gcn::Program vp = gcn::decode(vs_words.data(), vs_words.size());
            for (const gcn::Inst& in : vp.insts) host_log("  vs: %s", gcn::format(in).c_str());
            if (!ps_words.empty()) {
                const gcn::Program pp = gcn::decode(ps_words.data(), ps_words.size());
                for (const gcn::Inst& in : pp.insts) host_log("  ps: %s", gcn::format(in).c_str());
            }
        }
        return pl;
    };
    auto t_stage = std::chrono::steady_clock::now();
    const std::string variant = stage_lean ? "-lean" : "";  // dump names of the no-fallback variant
    // Pixel shader first: the vertex shader writes the inputs it reads at their
    // own locations (vs_output_links).
    const bool has_ps = !ps_words.empty();
    std::uint32_t clip_discard = 0;
    if (has_ps) {
        t_stage = std::chrono::steady_clock::now();
        // Input k is read at the location of the param register the input map
        // names (ps_input_locations), whatever vertex shader writes it. FLAT_SHADE
        // counts for the inputs it reads (the register keeps other draws' bits past them).
        const std::uint32_t read_inputs = ps_read_inputs(ps_words);
        std::uint32_t ps_flat_mask = 0;
        for (std::uint32_t k = 0; k < 32; ++k) {
            if (((s.ps_input_cntl[k] >> 10) & 1) && ((read_inputs >> k) & 1)) ps_flat_mask |= 1u << k;  // FLAT_SHADE
        }
        // Without depth or stencil writes, testing before the shader runs
        // cannot change the result, and the driver can then skip the pixels a
        // light's depth bounds reject instead of shading them and discarding.
        // A program an earlier translation showed cannot discard goes without.
        const std::vector<std::uint8_t> ps_input_map = ps_input_locations(read_inputs, s.ps_input_cntl);
        clip_discard = ps_clip_discard_count(s.vs_out_cntl, vs_words, ps_input_map);
        const bool early_fragment_tests = !(s.depth_control & 5) && !clip_discard && ps_early_tests_matter(ps_words) != 0;
        const PsStageInputs ps_in{&ps_words, s.ps_rsrc1, s.ps_rsrc2, s.ps_input_ena, ps_flat_mask, early_fragment_tests, stage_lean,
                                  &ps_dims, &ps_sampler_modes, &ps_input_map, clip_discard};
        const bool ps_lift = stage_lean && decomp_selects(ps_name);
        const std::uint64_t ps_key = lift_key(ps_stage_key(ps_in), ps_lift);
        const bool ps_cacheable = g_stage_cache_on;
        const std::shared_ptr<const CachedStage> ps_hit = ps_cacheable ? cached_stage(ps_key) : nullptr;
        if (ps_hit) {
            pl.ps.take(ps_hit);
            pl.ps.module = ps_hit->module;
            if (!t_pipeline_worker) g_pl_stage_repeats.fetch_add(1, std::memory_order_relaxed);
            manifest_note_hit(ps_key);
            if (note_precompile_hit(ps_key) && g_precompile_check && !ps_lift) {
                const gcn::Program p = gcn::decode(ps_words.data(), ps_words.size());
                check_precompiled_stage("PS", ps_name, ps_hit->meta, gcn::translate(p, ps_translate_options(ps_in)));
            }
        } else {
            const gcn::Program ps_prog = gcn::decode(ps_words.data(), ps_words.size());
            if (!ps_prog.errors.empty()) return fail("PS decode: " + ps_prog.errors[0].what);
            const gcn::TranslateOptions po = ps_translate_options(ps_in);
            pl.ps.fresh() = translate_cached(ps_prog, po);
            note_stage_translation(1, ps_words, ps_key, stage_lean);
            note_ps_early_tests(ps_words, pl.ps.meta().spirv);
            note_precompile_miss(ps_name, ps_in);
            note_image_dims(1, ps_name, ps_prog, pl.ps.meta(), ps_dims);
            note_first_translation(1, ps_name);
            {
                bool all_2d = true, modes_default = true, map_identity = true;
                for (const auto& d : ps_dims) all_2d &= (d.first & 0xff) == 1 && !d.second;
                for (bool m : ps_sampler_modes) modes_default &= !m;
                for (std::uint32_t k = 0; k < 32; ++k) map_identity &= (s.ps_input_cntl[k] & 0x1f) == k;
                std::lock_guard<std::mutex> lk(g_stage_variants.mu);
                if (all_2d) g_stage_variants.ps_dims_all_2d += 1;
                if (ps_dims.empty()) g_stage_variants.ps_no_images += 1;
                if (modes_default) g_stage_variants.ps_modes_default += 1;
                if (map_identity) g_stage_variants.ps_map_identity += 1;
                if (!ps_flat_mask) g_stage_variants.ps_flat_none += 1;
                if (early_fragment_tests) g_stage_variants.ps_early_tests += 1;
            }
            if (!t_pipeline_worker) {
                g_pl_translate_us.fetch_add(pl_us_since(t_stage), std::memory_order_relaxed);
                g_pl_stage_translations.fetch_add(1, std::memory_order_relaxed);
            }
            if (!g_stage_cache_on) note_stage_key(ps_key);
            if (!pl.ps.meta().ok()) return fail("PS: " + pl.ps.meta().errors[0]);
            manifest_note_ps(ps_key, ps_name, ps_in, ps_lift);
            // BBHOST_DECOMP: the no-fallback variant of a selected pixel shader runs
            // its typed lift. The lift declares the translation's bindings, so the
            // descriptor sets are unchanged; a rejection keeps the translation.
            if (ps_lift) {
                gcn::LiftResult lifted = lift_cached(false, ps_prog, po, pl.ps.meta());
                static std::mutex logged_mu;  // no-fallback variants build on the pipeline workers
                static std::set<std::string> logged;
                bool first;
                {
                    std::lock_guard<std::mutex> lk(logged_mu);
                    first = logged.insert(ps_name).second;
                }
                if (lifted.ok()) {
                    if (first) {
                        host_log("render: PS %s lifted (%zu words; translated %zu)", ps_name.c_str(), lifted.spirv.size(),
                                 pl.ps.meta().spirv.size());
                    }
                    pl.ps.fresh().spirv = std::move(lifted.spirv);
                    pl.ps.fresh().wave64_needs = 0;  // quad operations at most, which no subgroup size changes (gcn/wave.h)
                    pl.ps.lifted = true;
                } else if (first) {
                    host_log("render: PS %s not lifted, translated shader kept: %s", ps_name.c_str(), lifted.rejections[0].c_str());
                }
            }
        }
        if (g_dump_spirv || g_token_user_data_check) {
            table_slots_read(pl.ps.meta(), pl.table_reads[1], &pl.user_reads[1]);
        } else if (cb_no_fallback && g_skip_tables) {
            queue_table_mask(pl, 1, pl.ps);
        }
        dump_spirv(ps_name + variant + (pl.ps.lifted ? "-ps-lifted" : "-ps"), pl.ps.meta().spirv);
        dump_buffers(ps_name + variant + "-ps", pl.ps.meta(), pl.table_reads[1]);
        if (!ps_hit) {
            t_stage = std::chrono::steady_clock::now();
            if (!make_module(pl.ps.meta().spirv, pl.ps.module)) return fail("PS module");
            if (!t_pipeline_worker) g_pl_module_us.fetch_add(pl_us_since(t_stage), std::memory_order_relaxed);
            if (ps_cacheable) cache_stage(ps_key, pl.ps.meta(), pl.ps.module, pl.ps.lifted);
        }
    }
    // Vertex shader (+ fetch shader)
    t_stage = std::chrono::steady_clock::now();
    const std::vector<std::uint8_t>& output_links = register_links();
    // Step 6c: on vertex input the shader reads each element's formats from its
    // params (StageParams::vertex_w3), so its translation depends only on the
    // elements' locations, VGPRs and component counts, which the binary's input
    // table gives. Other formats keep them in the shader. Only under libraries,
    // where vertex shaders are compiled at creation: the loader makes a vertex
    // shader ~7% larger, and on the default path that cost a cold load 1.9 s of
    // command-processor compiles for no compile saved.
    const bool vs_formats_from_params = vertex_input && g_vertex_format_params && use_pipeline_library() &&
                                        gcn::vertex_formats_from_params_supported(vertex_input->elements);
    pl.vs_formats_from_params = vs_formats_from_params;
    pl.tess_hw = s.tess_hw;
    pl.tess_control_points = s.tess_control_points;
    const std::uint32_t vs_out_cntl =
        s.vs_out_cntl | (clip_discard ? gcn::TranslateOptions::kVsOutCntlClipVarying |
                                            ((clip_discard >> 8) & 31) << gcn::TranslateOptions::kVsOutCntlClipLocationShift
                                      : 0u);
    const VsStageInputs vs_in{&vs_words, &fetch_words, s.vs_rsrc1, s.vs_rsrc2, vs_out_cntl, vertex_input ? &vertex_input->elements : nullptr,
                              vs_formats_from_params, stage_lean, &vs_dims, &vs_sampler_modes, s.domain_level,
                              s.tess_hw,    s.tess_control_points, s.tess_quads, s.tess_cw, s.tess_spacing, s.tess_attrs, s.tess_attr_vec4s,
                              s.tess_window};
    // BBHOST_DECOMP: the no-fallback variant of a selected vertex shader on
    // vertex input runs its lift, under a key of its own.
    const bool vs_decomp = stage_lean && vertex_input && !g_vs_dispatcher && decomp_selects(vs_name);
    const std::uint64_t vs_key = lift_key(vs_stage_key(vs_in), vs_decomp);
    const bool vs_cacheable = g_stage_cache_on;
    const std::shared_ptr<const CachedStage> vs_hit = vs_cacheable ? cached_stage(vs_key) : nullptr;
    if (vs_hit) {
        pl.vs.take(vs_hit);
        pl.vs.module = vs_hit->module;
        if (!t_pipeline_worker) g_pl_stage_repeats.fetch_add(1, std::memory_order_relaxed);
        manifest_note_hit(vs_key);
        if (note_vs_precompile_hit(vs_key) && g_precompile_check && !vs_decomp) {
            const gcn::Program p = gcn::decode(vs_words.data(), vs_words.size());
            check_precompiled_stage("VS", vs_name, vs_hit->meta, gcn::translate(p, vs_translate_options(vs_in, nullptr)));
        }
    } else {
        const gcn::Program vs_prog = gcn::decode(vs_words.data(), vs_words.size());
        if (!vs_prog.errors.empty()) return fail("VS decode: " + vs_prog.errors[0].what);
        gcn::Program fetch_prog;
        if (!fetch_words.empty()) {
            fetch_prog = gcn::decode(fetch_words.data(), fetch_words.size());
            if (!fetch_prog.errors.empty()) return fail("fetch decode: " + fetch_prog.errors[0].what);
        }
        const gcn::TranslateOptions vo = vs_translate_options(vs_in, fetch_words.empty() ? nullptr : &fetch_prog);
        pl.vs.fresh() = translate_cached(vs_prog, vo);
        note_stage_translation(0, vs_words, vs_key, stage_lean);
        note_vs_precompile_miss(vs_name, vs_in);
        note_image_dims(0, vs_name, vs_prog, pl.vs.meta(), vs_dims);
        note_first_translation(0, vs_name);
        note_vs_dependencies(vs_name, vs_words, vertex_input, output_links, stage_lean, s, vs_dims, vs_sampler_modes);
        {
            std::lock_guard<std::mutex> lk(g_stage_variants.mu);
            if (vo.fetch) g_stage_variants.vs_with_fetch += 1;
            if (vertex_input) g_stage_variants.vs_with_vertex_input += 1;
        }
        if (!t_pipeline_worker) {
            g_pl_translate_us.fetch_add(pl_us_since(t_stage), std::memory_order_relaxed);
            g_pl_stage_translations.fetch_add(1, std::memory_order_relaxed);
        }
        if (!g_stage_cache_on) note_stage_key(vs_key);
        if (!pl.vs.meta().ok()) return fail("VS: " + pl.vs.meta().errors[0]);
        if (!vs_in.tess_window) manifest_note_vs(vs_key, vs_name, vs_in, vs_decomp);  // the manifest keeps no tessellation shape
        // The lift declares the translation's bindings and outputs (lift.cpp
        // checks the outputs), so layouts and linkage are unchanged; a rejection
        // keeps the translation.
        if (vs_decomp) {
            gcn::LiftResult lifted = lift_cached(true, vs_prog, vo, pl.vs.meta());
            static std::mutex logged_mu;  // no-fallback variants build on the pipeline workers
            static std::set<std::string> logged;
            bool first;
            {
                std::lock_guard<std::mutex> lk(logged_mu);
                first = logged.insert(vs_name).second;
            }
            if (lifted.ok()) {
                if (first) {
                    host_log("render: VS %s lifted (%zu words; translated %zu)", vs_name.c_str(), lifted.spirv.size(), pl.vs.meta().spirv.size());
                }
                pl.vs.fresh().spirv = std::move(lifted.spirv);
                pl.vs.lifted = true;
            } else if (first) {
                host_log("render: VS %s not lifted, translated shader kept: %s", vs_name.c_str(), lifted.rejections[0].c_str());
            }
        }
    }
    if (g_dump_spirv || g_token_user_data_check) {
        table_slots_read(pl.vs.meta(), pl.table_reads[0], &pl.user_reads[0]);
    } else if (cb_no_fallback && g_skip_tables) {
        queue_table_mask(pl, 0, pl.vs);
    }
    dump_spirv(vs_name + variant + (pl.vs.lifted ? "-vs-lifted" : "-vs"), pl.vs.meta().spirv);
    dump_buffers(vs_name + variant + "-vs", pl.vs.meta(), pl.table_reads[0]);
    if (!vs_hit) {
        t_stage = std::chrono::steady_clock::now();
        if (!make_module(pl.vs.meta().spirv, pl.vs.module)) return fail("VS module");
        if (!t_pipeline_worker) g_pl_module_us.fetch_add(pl_us_since(t_stage), std::memory_order_relaxed);
        if (vs_cacheable) cache_stage(vs_key, pl.vs.meta(), pl.vs.module, pl.vs.lifted);
    }
    // The LS as the pipeline's own vertex stage, so the control
    // point reaches the evaluation stage through the pipeline and no compute
    // pass has to end the render pass to produce it.
    pl.tess_attrs = s.tess_attrs;
    pl.tess_attr_vec4s = s.tess_attr_vec4s;
    if (pl.tess_attrs) {
        CachedProgram* ls = program_at(s.ls_va);
        const FetchProgram* fetch = s.ls_fetch_va ? fetch_program_at(s.ls_fetch_va) : nullptr;
        if (!ls || (s.ls_fetch_va && !fetch)) return fail("tessellation LS program");
        const std::uint64_t ls_key = tess_ls_key(ls->words, fetch ? &fetch->words : nullptr, s.ls_rsrc1, s.ls_rsrc2, s.tess_attr_vec4s);
        const std::shared_ptr<const CachedStage> ls_hit = g_stage_cache_on ? cached_stage(ls_key) : nullptr;
        gcn::Program prog;
        if (ls_hit) {
            pl.tess_ls.take(ls_hit);
            pl.tess_ls.module = ls_hit->module;
        } else {
            prog = gcn::decode(ls->words.data(), ls->words.size());
            gcn::Program fprog;
            if (fetch) fprog = gcn::decode(fetch->words.data(), fetch->words.size());
            pl.tess_ls.fresh() = translate_cached(prog, tess_ls_options(s.ls_rsrc1, s.ls_rsrc2, s.tess_attr_vec4s, fetch ? &fprog : nullptr));
        }
        if (!prog.errors.empty() || !pl.tess_ls.meta().ok()) {
            host_log("render: tessellation LS %s as a vertex stage: %s", ls->name.c_str(),
                     !prog.errors.empty() ? prog.errors[0].what.c_str() : pl.tess_ls.meta().errors[0].c_str());
            return fail("tessellation LS");
        }
        // Both stages read set 0. The evaluation stage's bindings are the only
        // ones that may be in it, and every LS in this game binds nothing -
        // but if one did, its numbering would land on the other's.
        if (!pl.tess_ls.meta().images.empty() || !pl.tess_ls.meta().samplers.empty() || !pl.tess_ls.meta().buffers.empty()) {
            host_log("render: tessellation LS %s binds resources (%zu images, %zu samplers, %zu buffers), which would collide with "
                     "the evaluation stage's set", ls->name.c_str(), pl.tess_ls.meta().images.size(), pl.tess_ls.meta().samplers.size(),
                     pl.tess_ls.meta().buffers.size());
            return fail("tessellation LS bindings");
        }
        if (!ls_hit) {
            dump_spirv(pl.name + "-tess-ls", pl.tess_ls.meta().spirv);
            if (!make_module(pl.tess_ls.meta().spirv, pl.tess_ls.module)) return fail("tessellation LS module");
            if (g_stage_cache_on) cache_stage(ls_key, pl.tess_ls.meta(), pl.tess_ls.module, false);
            manifest_note_ls(ls_key, ls->name, ls->words, fetch ? &fetch->words : nullptr, s.ls_rsrc1, s.ls_rsrc2, s.tess_attr_vec4s);
        }
    }
    // The two stages of a tessellated draw that carry no GCN code.
    // Shared by every such pipeline with the same factors, which is all of
    // them from one hull shader.
    if (pl.tess_hw && s.tess_hull) {
        pl.tess_tcs = tess_hull_module(s, pl.name);
        if (!pl.tess_tcs) return fail("tessellation hull module");
        pl.tess_vs = tess_passthrough_module(pl.name);
        if (!pl.tess_vs) return fail("tessellation VS module");
        host_log("render: host-tessellated pipeline %s: the game's hull shader at 0x%llx, %s patches of %u, LDS window %u bytes",
                 pl.name.c_str(), static_cast<unsigned long long>(s.hs_va), s.tess_quads ? "quad" : "triangle", s.tess_control_points,
                 s.tess_window);
    } else if (pl.tess_hw) {
        pl.tess_tcs = tess_tcs_module(s.tess_outer, s.tess_inner, s.tess_control_points, pl.tess_attrs ? pl.tess_attr_vec4s : 0, pl.name);
        if (!pl.tess_tcs) return fail("tessellation TCS module");
        if (!pl.tess_attrs) {
            pl.tess_vs = tess_passthrough_module(pl.name);
            if (!pl.tess_vs) return fail("tessellation VS module");
        }
        host_log("render: host-tessellated pipeline %s: control points %u, outer %g %g %g %g, inner %g %g", pl.name.c_str(),
                 s.tess_control_points, s.tess_outer[0], s.tess_outer[1], s.tess_outer[2], s.tess_outer[3], s.tess_inner[0],
                 s.tess_inner[1]);
    }
    // Rect list: geometry shader expanding the triangle into its rectangle.
    if (s.rect) {
        const std::vector<std::uint32_t> gs = gcn::make_rect_geometry_shader(pl.vs.meta().vs_params, pl.vs.meta().vs_clip_count,
                                                                             pl.vs.meta().vs_point_size);
        dump_spirv(pl.name + "-gs", gs);
        pl.gs.fresh().spirv = gs;
        if (!make_module(gs, pl.gs.module)) return fail("GS module");
    }

    if (g_set_layouts) {
        // Pipeline libraries: every vertex shader's set has the shared layout, so a
        // pixel shader's fragment-shader library is laid out the same whatever
        // vertex shader it runs with. (Libraries laid out with their own set only,
        // a null or empty set in the other stage's slot and independent sets,
        // fast-linked into pipelines that lost the device on the RTX 4070 driver.)
        pl.set_layouts[0] = use_pipeline_library() ? g.gfx_set_layout : stage_set_layout(pl.vs.meta());
        // Under libraries both sets take the shared layout, so a vertex shader's
        // pre-rasterization library does not depend on its pixel shader (step 6b).
        pl.set_layouts[1] = use_pipeline_library() ? g.gfx_set_layout : stage_set_layout(pl.ps.meta());  // params only without a PS
        pl.layout = pl.set_layouts[0] && pl.set_layouts[1] ? stage_pipeline_layout(pl.set_layouts[0], pl.set_layouts[1]) : VK_NULL_HANDLE;
        if (!pl.layout) return fail("stage descriptor layouts");
    }
    if (create && !create_gfx_pipeline(pl, s, vertex_input, fetch_words.size())) pl.failed = true;
    return pl;
}

// ---- pipeline libraries (BBHOST_PIPELINE_LIBRARY) --------------------------------
// Per-shader compiles, step 3: a graphics pipeline is
// linked, without link-time optimization, from four libraries. A pixel shader's
// fragment-shader library and a vertex shader's pre-rasterization library are
// compiled once per SPIR-V module and shared by every pipeline using them; the
// vertex input and fragment output interfaces are small. Cull mode, front
// face, depth and stencil state are dynamic, so shader libraries do not depend
// on them.
enum LibraryKind { kLibVertexInput = 0, kLibPreRaster = 1, kLibFragment = 2, kLibOutput = 3 };

// BBHOST_LIBRARY_TRACE=1: log each library build's start and end, with the
// shader that asked for it, so a driver crash inside one names its shader.
const bool g_library_trace = [] {
    const char* e = std::getenv("BBHOST_LIBRARY_TRACE");
    return e && e[0] == '1';
}();
thread_local const char* t_library_owner = nullptr;

template <typename Build>
VkPipeline pipeline_library(LibraryKind kind, std::uint64_t key, const char* what, std::size_t words, Build build) {
    key = fnv1a(&kind, sizeof(kind), key);
    {
        std::unique_lock<std::mutex> lk(g_libs.mu);
        for (;;) {
            if (auto it = g_libs.ready.find(key); it != g_libs.ready.end()) {
                g_libs.hits[kind].fetch_add(1, std::memory_order_relaxed);
                return it->second;
            }
            if (!g_libs.building.count(key)) break;
            g_libs.cv.wait(lk);
        }
        g_libs.building.insert(key);
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (g_library_trace) {
        host_log("render: library %s %016llx start on %s for %s (SPIR-V words %zu)", what, static_cast<unsigned long long>(key),
                 t_pipeline_worker ? "a worker" : "the command processor", t_library_owner ? t_library_owner : "a draw", words);
    }
    const VkPipeline p = build();
    const std::uint64_t us = pl_us_since(t0);
    if (g_library_trace) host_log("render: library %s %016llx done", what, static_cast<unsigned long long>(key));
    g_libs.us[kind].fetch_add(us, std::memory_order_relaxed);
    g_libs.built[kind].fetch_add(1, std::memory_order_relaxed);
    if (us >= 30000) {
        static std::atomic<int> logs[2]{};  // the command processor, workers: per-shader compiles would use up one cap
        if (logs[t_pipeline_worker ? 1 : 0].fetch_add(1) < 200) {
            host_log("render: slow pipeline library %s on %s: %llu ms (SPIR-V words %zu)%s", what, t_pipeline_worker ? "a worker" : "the command processor",
                     static_cast<unsigned long long>(us / 1000), words, p ? "" : ", failed");
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_libs.mu);
        g_libs.building.erase(key);
        g_libs.ready[key] = p;
    }
    g_libs.cv.notify_all();
    return p;
}

// BBHOST_LIBRARY_RELINK=0: keep fast-linked pipelines. By default a worker links
// each one again from the same libraries with link-time optimization, and the
// command processor swaps it in: fast-linked pipelines cost 3.7% more GPU time
// in steady state (step 3).
const bool g_library_relink = [] {
    const char* e = std::getenv("BBHOST_LIBRARY_RELINK");
    return !(e && e[0] == '0');
}();

// BBHOST_PIPELINE_STATS=<name>[,<name>...]: the optimized link of each pipeline
// whose name holds one of them ("9da6aacb+bb10e440", or a shader's hash alone)
// asks the driver for its statistics and internal representations - RADV's are
// registers, instruction counts, NIR, ACO's IR and the disassembly - and writes
// them to build/pipeline-stats/<pipeline>.txt. For seeing what a translator
// change does to the code that runs.
bool pipeline_stats_wanted(const std::string& name) {
    if (!g.pipeline_stats) return false;
    static const std::vector<std::string> wanted = [] {
        std::vector<std::string> v;
        std::string cur;
        for (const char* c = std::getenv("BBHOST_PIPELINE_STATS"); c && *c; ++c) {
            if (*c == ',') {
                if (!cur.empty()) v.push_back(cur);
                cur.clear();
            } else {
                cur += *c;
            }
        }
        if (!cur.empty()) v.push_back(cur);
        return v;
    }();
    for (const std::string& w : wanted) {
        if (w == "*" || name.find(w) != std::string::npos) return true;
    }
    return false;
}

void write_pipeline_stats(VkPipeline p, const std::string& name) {
    static const auto props = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(g.device, "vkGetPipelineExecutablePropertiesKHR"));
    static const auto stats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(g.device, "vkGetPipelineExecutableStatisticsKHR"));
    static const auto irs = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
        vkGetDeviceProcAddr(g.device, "vkGetPipelineExecutableInternalRepresentationsKHR"));
    if (!props || !stats || !irs) return;
    std::error_code ec;
    std::filesystem::create_directories("build/pipeline-stats", ec);
    FILE* f = std::fopen(("build/pipeline-stats/" + name + ".txt").c_str(), "w");
    if (!f) return;
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p;
    std::uint32_t n = 0;
    props(g.device, &pi, &n, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> execs(n, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    props(g.device, &pi, &n, execs.data());
    for (std::uint32_t i = 0; i < n; ++i) {
        std::fprintf(f, "== executable %u: %s (%s), subgroup %u\n", i, execs[i].name, execs[i].description, execs[i].subgroupSize);
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = p;
        ei.executableIndex = i;
        std::uint32_t ns = 0;
        stats(g.device, &ei, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        stats(g.device, &ei, &ns, st.data());
        for (const auto& s : st) {
            switch (s.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::fprintf(f, "%s: %u\n", s.name, s.value.b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::fprintf(f, "%s: %lld\n", s.name, static_cast<long long>(s.value.i64)); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::fprintf(f, "%s: %llu\n", s.name, static_cast<unsigned long long>(s.value.u64)); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::fprintf(f, "%s: %g\n", s.name, s.value.f64); break;
            default: break;
            }
        }
        std::uint32_t nr = 0;
        irs(g.device, &ei, &nr, nullptr);
        std::vector<VkPipelineExecutableInternalRepresentationKHR> rep(nr, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
        irs(g.device, &ei, &nr, rep.data());
        std::vector<std::vector<char>> text(nr);
        for (std::uint32_t k = 0; k < nr; ++k) {
            text[k].resize(rep[k].dataSize + 1);
            rep[k].pData = text[k].data();
        }
        irs(g.device, &ei, &nr, rep.data());
        for (std::uint32_t k = 0; k < nr; ++k) {
            text[k].back() = 0;
            std::fprintf(f, "-- %s (%s)\n%s\n", rep[k].name, rep[k].description, rep[k].isText ? text[k].data() : "(binary)");
        }
    }
    std::fclose(f);
    host_log("render: pipeline %s: statistics and disassembly in build/pipeline-stats", name.c_str());
}

VkPipeline create_library(VkGraphicsPipelineCreateInfo& gpci, VkGraphicsPipelineLibraryFlagsEXT flags, void* next) {
    VkGraphicsPipelineLibraryCreateInfoEXT li{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
    li.flags = flags;
    li.pNext = next;
    gpci.pNext = &li;
    gpci.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
    if (g_library_relink) gpci.flags |= VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;  // what the relink optimizes from
    VkPipeline p = VK_NULL_HANDLE;
    return vkCreateGraphicsPipelines(g.device, PipelineCacheUse().cache, 1, &gpci, nullptr, &p) == VK_SUCCESS ? p : VK_NULL_HANDLE;
}

// A vertex shader's pre-rasterization library, with a rect list's geometry
// shader. Both sets take the shared layout and it gets no target formats (step
// 6b), so it is keyed by the shaders and raster state alone and can be compiled
// before any draw uses it (step 6d).
// The raster states each vertex shader's libraries were built in, for the
// exit report: a library built again only for another state is what dynamic
// depth clamp or polygon mode would save.
struct PreRasterStates {
    std::mutex mu;
    std::unordered_map<std::uint64_t, std::uint32_t> by_vs;  // VS SPIR-V hash -> bit per state (mode * 4 + clamp * 2 + gs)
    std::uint64_t again = 0, again_clamp = 0, again_mode = 0, again_gs = 0;
} g_pre_raster_states;
// The host tessellator's part of a pre-rasterization library: its
// generated vertex stage (the LS translated, or the pass-through), the control
// stage made for the hull shader's factors, and the patch size. The domain
// shader is the library's `vs`, as the evaluation stage.
struct TessLibraryStages {
    VkShaderModule vertex = VK_NULL_HANDLE, control = VK_NULL_HANDLE;
    std::uint32_t control_points = 1;
};

// `vertex_formats`: 48 specialization words for the vertex shader, for an
// optimized relink (relink_pipeline) - 16 vertex-format descriptors
// (gcn::kVertexFormatSpecId), 16 indexed loads' words 3 (gcn::kCbW3SpecId),
// then their 16 strides (gcn::kCbStrideSpecId), 0 where the default (read the
// params) stays; null keeps every default.
VkPipeline pre_raster_library(const ShaderStage& vs, const ShaderStage& gs, VkPolygonMode polygon_mode, bool draw_clamp, VkPipelineLayout layout,
                              const VkDescriptorSetLayout set_layouts[2], const TessLibraryStages* tess = nullptr,
                              const std::uint32_t* vertex_formats = nullptr) {
    const bool depth_clamp = draw_clamp && !g.dynamic_depth_clamp;  // else the draw sets it (DrawLibraryState)
    if (!vertex_formats) {
        const std::uint64_t vs_hash = fnv1a(vs.meta().spirv.data(), vs.meta().spirv.size() * 4);
        const std::uint32_t bit = 1u << ((static_cast<std::uint32_t>(polygon_mode) & 3) * 4 + (depth_clamp ? 2 : 0) + (gs.module ? 1 : 0));
        std::lock_guard<std::mutex> lk(g_pre_raster_states.mu);
        std::uint32_t& seen = g_pre_raster_states.by_vs[vs_hash];
        if (seen && !(seen & bit)) {
            ++g_pre_raster_states.again;
            const std::uint32_t mode_bits = 0xfu << ((static_cast<std::uint32_t>(polygon_mode) & 3) * 4);
            if (!(seen & mode_bits)) ++g_pre_raster_states.again_mode;
            else if (!(seen & (mode_bits & (depth_clamp ? 0xccccccccu : 0x33333333u)))) ++g_pre_raster_states.again_clamp;
            else ++g_pre_raster_states.again_gs;
        }
        seen |= bit;
    }
    std::uint64_t key = fnv1a(vs.meta().spirv.data(), vs.meta().spirv.size() * 4);
    key = fnv1a(gs.meta().spirv.data(), gs.meta().spirv.size() * 4, key);
    const std::uint32_t state[2] = {static_cast<std::uint32_t>(polygon_mode), depth_clamp ? 1u : 0u};
    key = fnv1a(state, sizeof(state), key);
    key = fnv1a(set_layouts, 2 * sizeof(VkDescriptorSetLayout), key);  // every library of a pipeline uses its full layout
    if (tess) {
        // The generated modules are made once each and never destroyed
        // (tess_tcs_module, tess_passthrough_module, the stage cache's LS), so
        // their handles name them for this process's library map.
        const std::uint64_t t[3] = {reinterpret_cast<std::uint64_t>(tess->vertex), reinterpret_cast<std::uint64_t>(tess->control),
                                    tess->control_points};
        key = fnv1a(t, sizeof(t), key ^ 0x7e55000000000000ull);
    }
    if (vertex_formats) key = fnv1a(vertex_formats, 48 * sizeof(std::uint32_t), key ^ 0x5bec000000000000ull);
    return pipeline_library(kLibPreRaster, key, "pre-rasterization", vs.meta().spirv.size(), [&] {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        const auto stage = [&](VkShaderStageFlagBits bit, VkShaderModule m) {
            VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            si.stage = bit;
            si.module = m;
            si.pName = "main";
            stages.push_back(si);
        };
        VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        if (tess) {
            stage(VK_SHADER_STAGE_VERTEX_BIT, tess->vertex);
            stage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tess->control);
            stage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, vs.module);
            ts.patchControlPoints = std::max<std::uint32_t>(1, tess->control_points);
        } else {
            stage(VK_SHADER_STAGE_VERTEX_BIT, vs.module);
        }
        static const auto spec_entries = [] {
            std::array<VkSpecializationMapEntry, 48> e{};
            for (std::uint32_t k = 0; k < 16; ++k) e[k] = {gcn::kVertexFormatSpecId + k, k * 4, 4};
            for (std::uint32_t k = 0; k < 16; ++k) e[16 + k] = {gcn::kCbW3SpecId + k, (16 + k) * 4, 4};
            for (std::uint32_t k = 0; k < 16; ++k) e[32 + k] = {gcn::kCbStrideSpecId + k, (32 + k) * 4, 4};
            return e;
        }();
        const VkSpecializationInfo spec{48, spec_entries.data(), 48 * sizeof(std::uint32_t), vertex_formats};
        if (vertex_formats && !tess) stages.front().pSpecializationInfo = &spec;
        if (gs.module) stage(VK_SHADER_STAGE_GEOMETRY_BIT, gs.module);
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = polygon_mode;
        rs.lineWidth = 1.0f;
        rs.depthClampEnable = depth_clamp ? VK_TRUE : VK_FALSE;
        // Depth bias on with dynamic values (DrawLibraryState), 0 0 0 for a
        // state without one - which is no bias - as D3D11 drivers treat it.
        rs.depthBiasEnable = VK_TRUE;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE,
                                      VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT};
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = g.dynamic_depth_clamp ? 6 : 5;
        dsi.pDynamicStates = dyn;
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpci.stageCount = static_cast<std::uint32_t>(stages.size());
        gpci.pStages = stages.data();
        if (tess) gpci.pTessellationState = &ts;
        gpci.pViewportState = &vp;
        gpci.pRasterizationState = &rs;
        gpci.pDynamicState = &dsi;
        gpci.layout = layout;
        VkPipelineRenderingCreateInfo no_formats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        return create_library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, &no_formats);
    });
}

// ---- Pixel shaders at wave32 (g.ps_wave, decided in gpu.cpp) ----
// The fragment stage's subgroup size: 32 for a program that computes the same
// in a 32-lane subgroup (gcn/wave.h: TranslateResult::wave64_needs 0), the
// device's own size for the others - asked for in so many words, so a stage
// that needs its 64 lanes keeps them - or, with BBHOST_PS_WAVE32=driver, the
// driver's choice. The census counts programs by their SPIR-V, for the report
// line (ps_wave_census) and the log of a program kept at 64.
struct PsWaveCensus {
    std::mutex mu;
    std::unordered_set<std::uint64_t> programs;  // under mu
    std::uint64_t at32 = 0, driver = 0, kept = 0;  // under mu: programs
    std::uint64_t reasons[gcn::kWave64NeedBits] = {};  // under mu: kept programs, by gcn::Wave64Need bit
    std::uint64_t reported = 0;                        // under mu: programs at the last 300-flip line
    std::atomic<std::uint64_t> stages{0};              // fragment stages created with a size asked for
    std::atomic<std::uint64_t> refused{0};             // ... that the driver refused, created again at its default
} g_ps_wave;

// Gives `si` (a fragment stage, pNext unused) its subgroup size through `req`
// (or ALLOW_VARYING_SUBGROUP_SIZE). Returns false where the stage is left as
// it was: BBHOST_PS_WAVE32=0, a device that cannot be asked, a program kept
// at a default the device cannot be asked for by name.
bool fragment_stage_wave(const gcn::TranslateResult& meta, VkPipelineShaderStageCreateInfo& si,
                         VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& req) {
    if (!g.ps_wave) return false;
    const bool wave32 = meta.wave64_needs == 0;
    {
        const std::uint64_t h = fnv1a(meta.spirv.data(), meta.spirv.size() * 4);
        std::lock_guard<std::mutex> lk(g_ps_wave.mu);
        if (g_ps_wave.programs.insert(h).second) {
            if (wave32) {
                ++(g.ps_wave == 1 ? g_ps_wave.driver : g_ps_wave.at32);
            } else {
                ++g_ps_wave.kept;
                for (int b = 0; b < gcn::kWave64NeedBits; ++b) g_ps_wave.reasons[b] += (meta.wave64_needs >> b) & 1;
                if (g_ps_wave.kept <= 20) {
                    host_log("render: pixel shader %016llx (%s) keeps the device's %u-lane subgroups: %s", static_cast<unsigned long long>(h),
                             t_library_owner ? t_library_owner : "a draw's", g.subgroup_size, gcn::wave64_needs_str(meta.wave64_needs).c_str());
                }
            }
        }
    }
    if (wave32 && g.ps_wave == 1) {
        si.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT;
    } else {
        const std::uint32_t size = wave32 ? 32 : g.subgroup_size;
        if (!(g.subgroup_required_stages & VK_SHADER_STAGE_FRAGMENT_BIT) || size < g.subgroup_min || size > g.subgroup_max) return false;
        req = {};
        req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        req.requiredSubgroupSize = size;
        si.pNext = &req;
    }
    g_ps_wave.stages.fetch_add(1, std::memory_order_relaxed);
    return true;
}
// A fragment stage the driver would not create with the size asked for (a
// size it lists for the stage, so that would be its bug): `si` goes back to
// the stage as it was before, at the device's default, for the caller to
// create again.
void fragment_stage_unsized(VkPipelineShaderStageCreateInfo& si) {
    si.pNext = nullptr;
    si.flags &= ~static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT);
}
// The second creation's outcome: counted, and the first few logged, when it
// worked - then the size was why the first failed. When it failed as well the
// size was not the reason, and the caller reports the failure as before.
void fragment_stage_wave_refused(bool created_unsized) {
    if (!created_unsized) return;
    if (g_ps_wave.refused.fetch_add(1, std::memory_order_relaxed) < 4) {
        host_log("render: the driver refused a fragment stage (%s) at the subgroup size asked for; created at its default",
                 t_library_owner ? t_library_owner : "a draw's");
    }
}
// "ps-wave: ..." for the 300-flip report when programs were added since the
// last one (`changed_only`), and for the exit report.
std::string ps_wave_census(bool changed_only) {
    if (!g.ps_wave) return {};
    std::lock_guard<std::mutex> lk(g_ps_wave.mu);
    const std::uint64_t n = g_ps_wave.programs.size();
    if (changed_only && n == g_ps_wave.reported) return {};
    g_ps_wave.reported = n;
    std::string reasons;
    for (int b = 0; b < gcn::kWave64NeedBits; ++b) {
        if (g_ps_wave.reasons[b]) reasons += std::string(reasons.empty() ? " (" : ", ") + gcn::wave64_need_name(b) + " x" + std::to_string(g_ps_wave.reasons[b]);
    }
    if (!reasons.empty()) reasons += ")";
    char buf[320];
    std::snprintf(buf, sizeof(buf), "ps-wave: pixel shaders %llu: wave32 %llu, the driver's choice %llu, kept at %u %llu%s; fragment stages sized %llu, refused %llu",
                  static_cast<unsigned long long>(n), static_cast<unsigned long long>(g_ps_wave.at32),
                  static_cast<unsigned long long>(g_ps_wave.driver), g.subgroup_size, static_cast<unsigned long long>(g_ps_wave.kept), reasons.c_str(),
                  static_cast<unsigned long long>(g_ps_wave.stages.load()), static_cast<unsigned long long>(g_ps_wave.refused.load()));
    return buf;
}

// A pixel shader's fragment-shader library. It takes no target formats (the
// fragment output library has them), so it is keyed by the shader and its set
// layout alone, and can be compiled before any draw uses it (step 4). `layout`
// is the pipeline layout: the shared vertex-shader set layout and `ps_set`.
// `w3`: 16 indexed loads' words 3 then their 16 strides as the pixel shader's
// specialization constants (gcn::kCbW3SpecId, gcn::kCbStrideSpecId), for an
// optimized relink; null keeps the defaults.
VkPipeline fragment_library(const gcn::TranslateResult& meta, VkShaderModule module, VkDescriptorSetLayout ps_set, VkPipelineLayout layout,
                            const std::uint32_t* w3 = nullptr) {
    std::uint64_t fs_key = fnv1a(meta.spirv.data(), meta.spirv.size() * 4, module ? 1 : 2);
    fs_key = fnv1a(&ps_set, sizeof(ps_set), fs_key);
    if (w3) fs_key = fnv1a(w3, 32 * sizeof(std::uint32_t), fs_key ^ 0x3bec000000000000ull);
    return pipeline_library(kLibFragment, fs_key, "fragment shader", meta.spirv.size(), [&] {
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        si.module = module;
        si.pName = "main";
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo wave{};
        const bool sized = module && fragment_stage_wave(meta, si, wave);
        static const auto w3_entries = [] {
            std::array<VkSpecializationMapEntry, 32> e{};
            for (std::uint32_t k = 0; k < 16; ++k) e[k] = {gcn::kCbW3SpecId + k, k * 4, 4};
            for (std::uint32_t k = 0; k < 16; ++k) e[16 + k] = {gcn::kCbStrideSpecId + k, (16 + k) * 4, 4};
            return e;
        }();
        const VkSpecializationInfo spec{32, w3_entries.data(), 32 * sizeof(std::uint32_t), w3};
        if (w3) si.pSpecializationInfo = &spec;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,   VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                                      VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,         VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
                                      VK_DYNAMIC_STATE_DEPTH_BOUNDS};
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = g.has_depth_bounds ? 10 : 8;
        dsi.pDynamicStates = dyn;
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpci.stageCount = module ? 1 : 0;
        gpci.pStages = &si;
        gpci.pMultisampleState = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pDynamicState = &dsi;
        gpci.layout = layout;
        VkPipelineRenderingCreateInfo no_formats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkPipeline p = create_library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, &no_formats);
        if (!p && sized) {
            fragment_stage_unsized(si);
            p = create_library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, &no_formats);
            fragment_stage_wave_refused(p != VK_NULL_HANDLE);
        }
        return p;
    });
}

// ---- per-shader compiles, step 4 (BBHOST_PIPELINE_LIBRARY; BBHOST_PRECOMPILE=0 turns it off) ----
// When GX creates a pixel shader, a worker translates it with what its binary
// says (stage registers, flat inputs from its semantic table, image dimensions
// predicted from the program, no-fallback, early fragment tests) and compiles
// its fragment-shader library. A first draw whose stage inputs agree finds the
// stage and the library ready and only links its pipeline.
bool precompile_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_PRECOMPILE");
        return !(e && e[0] == '0');
    }();
    return on && use_pipeline_library();
}
struct PrecompileJob {
    std::string name;
    std::vector<std::uint8_t> container;
    int type = 2;             // Shdr type: 1 vertex, 2 pixel, 4 compute (gpu.cpp precompile_compute); 0 an optimized relink
    std::uint64_t flip = 0;   // when GX created it
    std::shared_ptr<const ManifestStage> manifest;  // type 7: a stage from the manifest
    // An optimized relink (queue_library_relink): the pipeline, its libraries and layout.
    GfxPipeline* relink = nullptr;
    VkPipeline libraries[4] = {};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    // A relink of a vertex shader that reads its formats from params: its
    // pre-rasterization library again, the pipeline's descriptors given as
    // specialization constants, with the state it was made with.
    bool specialize_formats = false;
    std::uint32_t vertex_formats[16] = {};
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    bool depth_clamp = false;
    // And its indexed loads' words 3 and strides as its first draw had them
    // (by stage).
    std::uint32_t w3[2][16] = {};
    std::uint32_t stride[2][16] = {};
    std::uint16_t w3_mask[2] = {};
};
struct PrecompileRecord {  // what a pixel shader was compiled with
    std::uint32_t rsrc1 = 0, rsrc2 = 0, ps_input_ena = 0, flat_mask = 0;
    bool early_fragment_tests = true, failed = false;
    std::vector<std::uint8_t> input_map;  // ps_input_locations, the registers predicted from the input semantics
    std::vector<std::pair<std::uint32_t, bool>> dims;
    std::vector<bool> modes;
};
struct Precompiler {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<PrecompileJob> jobs;                             // under mu
    std::unordered_map<std::string, int> running;               // under mu: the names a worker is compiling now
    std::unordered_map<std::string, PrecompileRecord> records;  // by pixel-shader name, under mu
    std::unordered_set<std::uint64_t> stage_keys;               // under mu
    std::map<std::string, std::uint64_t> fail_reasons;          // under mu
    // Optimized relinks, for the command processor to swap in (apply_relinks_locked).
    std::vector<std::pair<GfxPipeline*, VkPipeline>> relinked;  // under mu
    std::atomic<std::uint64_t> relinks_queued{0}, relinks_done{0}, relinks_failed{0}, relink_us{0}, relinks_adopted{0};
    std::atomic<std::uint64_t> relinks_specialized{0}, relink_spec_failed{0}, relinks_w3{0};
    // Vertex shaders (step 6d), by name: what each was compiled with.
    struct VsRecord {
        std::uint32_t rsrc1 = 0, rsrc2 = 0, out_cntl = 0;
        std::vector<gcn::VertexElement> elements;
        std::vector<std::pair<std::uint32_t, bool>> dims;
        std::vector<bool> modes;
        bool failed = false;
    };
    std::unordered_map<std::string, VsRecord> vs_records;       // under mu
    std::unordered_set<std::uint64_t> vs_stage_keys;            // under mu
    std::atomic<std::uint64_t> vs_queued{0}, vs_done{0}, vs_failed{0}, vs_translate_us{0}, vs_library_us{0}, vs_stage_hits{0};
    std::uint64_t vs_miss_not_ready = 0, vs_miss_queued = 0, vs_miss_compiling = 0, vs_miss_failed = 0, vs_miss_path = 0, vs_miss_comp = 0, vs_miss_rsrc = 0, vs_miss_out_cntl = 0,
                  vs_miss_formats = 0, vs_miss_elements = 0, vs_miss_dims = 0, vs_miss_modes = 0, vs_miss_other = 0;  // under mu
    std::vector<std::string> not_ready;                         // under mu: the first draw translations with no record
    std::vector<std::string> vs_not_ready;                      // under mu: the same for vertex shaders
    std::atomic<std::uint64_t> queued{0}, done{0}, failed{0}, translate_us{0}, library_us{0}, stage_hits{0};
    std::uint64_t miss_not_ready = 0, miss_queued = 0, miss_compiling = 0, miss_failed = 0, miss_flat = 0, miss_input_map = 0, miss_eft_on = 0, miss_eft_off = 0, miss_dims = 0, miss_modes = 0, miss_ena = 0,
                  miss_rsrc = 0, miss_other = 0, miss_lifted = 0;  // under mu
    // The workers (precompiler()): `boot` of them until the first in-game
    // frame and while a loading screen is up, `steady` otherwise; those past
    // `limit` wait on park_cv.
    std::condition_variable park_cv;
    int limit = 0, boot = 0, steady = 0;  // under mu
    // A run of jobs from the queue's first one to the workers' next idle
    // moment, for the "idle" line: how long the start's compiles took.
    bool busy = false;                                       // under mu
    std::chrono::steady_clock::time_point busy_since{};     // under mu
    std::uint64_t burst_jobs = 0, burst_type[8] = {};        // under mu
    int idle_logs = 0;                                       // under mu
};
void precompile_ps(Precompiler& w, const PrecompileJob& job);
void precompile_vs(Precompiler& w, const PrecompileJob& job);
void precompile_manifest(const ManifestStage& m);
void relink_pipeline(Precompiler& w, const PrecompileJob& job);
// Parallel compiles at the start (Kyo's parallel preload, done natively): GX
// creates its ~3,000 vertex and pixel shaders, and the manifest brings a few
// hundred more, before the first in-game frame - while the game mostly loads -
// so until then, and while a loading screen is up, half the hardware threads
// compile (up to 16, as his preload builds); in game a quarter of them do, as
// before. Set by precompile_boost_update (host_gpu_set_loading,
// host_gpu_world_reached).
std::atomic<bool> g_precompile_boost{true};
std::atomic<Precompiler*> g_precompiler{nullptr};
const auto g_process_start = std::chrono::steady_clock::now();
Precompiler& precompiler() {
    static Precompiler* const p = [] {
        auto* w = new Precompiler;  // never destroyed: its threads run until exit
        // In game a quarter of the hardware threads, 2 to 16. With compute
        // shaders compiled at creation a cold world load is short enough that
        // four workers left 583 pixel shaders queued at its end; eight drained
        // them. With vertex shaders as well (about 330 s of compiles at boot),
        // eight left 966 pixel shaders queued and sixteen drained both queues.
        const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        w->steady = std::clamp(hw / 4, 2, 16);
        w->boot = std::clamp(hw / 2, w->steady, std::max(w->steady, 16));
        // BBHOST_PRECOMPILE_THREADS=<n>: n workers at all times;
        // BBHOST_PRECOMPILE_BOOT_THREADS=<n>: n of them while starting and loading.
        if (const char* e = std::getenv("BBHOST_PRECOMPILE_THREADS")) w->steady = w->boot = std::clamp(std::atoi(e), 1, 64);
        if (const char* e = std::getenv("BBHOST_PRECOMPILE_BOOT_THREADS")) w->boot = std::clamp(std::atoi(e), w->steady, 64);
        {
            // Published before the flag is read: a change that finds no pool
            // has set the flag already, and one that finds it sets the limit.
            std::lock_guard<std::mutex> lk(w->mu);
            g_precompiler.store(w);
            w->limit = g_precompile_boost.load() ? w->boot : w->steady;
        }
        host_log("render: precompile workers: %d until the first in-game frame and while loading, %d in game "
                 "(BBHOST_PRECOMPILE_BOOT_THREADS, BBHOST_PRECOMPILE_THREADS)", w->boot, w->steady);
        for (int i = 0; i < w->boot; ++i) {
            std::thread([w, i] {
                t_pipeline_worker = true;
                host_thread_set_name("bb-compile");
                for (;;) {
                    PrecompileJob job;
                    {
                        std::unique_lock<std::mutex> lk(w->mu);
                        for (;;) {
                            if (i >= w->limit) {
                                // A wake-up meant for a worker under the limit
                                // goes on to one.
                                if (!w->jobs.empty()) w->cv.notify_one();
                                w->park_cv.wait(lk, [w, i] { return i < w->limit; });
                                continue;
                            }
                            if (!w->jobs.empty()) break;
                            w->cv.wait(lk);
                        }
                        job = std::move(w->jobs.front());
                        w->jobs.pop_front();
                        ++w->running[job.name];
                        if (!w->busy) {
                            w->busy = true;
                            w->busy_since = std::chrono::steady_clock::now();
                            w->burst_jobs = 0;
                            std::fill(std::begin(w->burst_type), std::end(w->burst_type), 0);
                        }
                    }
                    while (!g.ok) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    if (job.type == 4) {
                        precompile_compute(job.name, job.container, job.flip);
                    } else if (job.type == 1) {
                        precompile_vs(*w, job);
                    } else if (job.type == 0) {
                        relink_pipeline(*w, job);
                    } else if (job.type == 7) {
                        precompile_manifest(*job.manifest);
                    } else {
                        precompile_ps(*w, job);
                    }
                    std::lock_guard<std::mutex> lk(w->mu);
                    if (--w->running[job.name] == 0) w->running.erase(job.name);
                    ++w->burst_jobs;
                    ++w->burst_type[job.type & 7];
                    if (w->busy && w->jobs.empty() && w->running.empty()) {
                        w->busy = false;
                        // How long a run of compiles took the workers, wall
                        // time: the start's is what parallel compiles and the
                        // translation cache shorten.
                        if (w->burst_jobs >= 32 && w->idle_logs < 16) {
                            ++w->idle_logs;
                            const auto now = std::chrono::steady_clock::now();
                            host_log("render: precompile workers idle: %llu jobs (manifest %llu, pixel %llu, vertex %llu, compute %llu, relinks %llu) "
                                     "in %.2f s on %d workers; %.1f s after start, flip %llu",
                                     static_cast<unsigned long long>(w->burst_jobs), static_cast<unsigned long long>(w->burst_type[7]),
                                     static_cast<unsigned long long>(w->burst_type[2]), static_cast<unsigned long long>(w->burst_type[1]),
                                     static_cast<unsigned long long>(w->burst_type[4]), static_cast<unsigned long long>(w->burst_type[0]),
                                     std::chrono::duration<double>(now - w->busy_since).count(), w->limit,
                                     std::chrono::duration<double>(now - g_process_start).count(),
                                     static_cast<unsigned long long>(hle_video_flip_count()));
                        }
                    }
                }
            }).detach();
        }
        return w;
    }();
    return *p;
}
void queue_ps_precompile(std::string name, std::vector<std::uint8_t> container) {
    Precompiler& w = precompiler();
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_back({std::move(name), std::move(container)});
    }
    w.queued.fetch_add(1, std::memory_order_relaxed);
    w.cv.notify_one();
}
// Compute shaders go to the front: they are few, and a world load's slowest
// compiles (13 s each) are compute shaders its first dispatches wait on.
void queue_compute_precompile(std::string name, std::vector<std::uint8_t> container, std::uint64_t flip) {
    Precompiler& w = precompiler();
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_front({std::move(name), std::move(container), 4, flip});
    }
    w.cv.notify_one();
}
void queue_vs_precompile(std::string name, std::vector<std::uint8_t> container) {
    Precompiler& w = precompiler();
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_back({std::move(name), std::move(container), 1, 0});
    }
    w.vs_queued.fetch_add(1, std::memory_order_relaxed);
    w.cv.notify_one();
}

// The start's and the loading screens' extra workers (g_precompile_boost).
std::atomic<bool> g_world_seen{false}, g_loading_now{false};
void precompile_boost_update(const char* why) {
    const bool on = !g_world_seen.load() || g_loading_now.load();
    if (g_precompile_boost.exchange(on) == on) return;
    Precompiler* w = g_precompiler.load();
    if (!w) return;  // made later, with the limit this says
    int from, to;
    std::size_t waiting;
    {
        std::lock_guard<std::mutex> lk(w->mu);
        from = w->limit;
        w->limit = g_precompile_boost.load() ? w->boot : w->steady;
        to = w->limit;
        waiting = w->jobs.size();
    }
    if (to > from) w->park_cv.notify_all();
    if (from != to) host_log("render: precompile workers %d -> %d (%s); %zu jobs waiting", from, to, why, waiting);
}

// BBHOST_LIBRARY_RELINK: after a fast link, a worker links the same libraries
// again with link-time optimization, behind the shaders GX created, and the
// command processor swaps the result in (apply_relinks_locked).
std::atomic<bool> g_relinks_ready{false};
// BBHOST_RELINK_FORMATS=0: a relink keeps the vertex shader's library, which
// reads the formats from params. By default a vertex shader built with
// vertex_formats_from_params is compiled again for it with the pipeline's own
// formats as constants - the pipeline key holds every element's word 3 - and
// the driver folds the conversions and their selects away (~400 of the
// Steam Deck's hot G-buffer vertex shader's 1018 instructions).
const bool g_relink_formats = [] {
    const char* e = std::getenv("BBHOST_RELINK_FORMATS");
    return !(e && e[0] == '0');
}();
// BBHOST_RELINK_W3=0: a relink keeps the indexed loads' words 3 read from the
// params. By default the relink, queued at the pipeline's first draw, takes
// that draw's words as constants (gcn::kCbW3SpecId) and the driver folds the
// DST_SEL selects away (the Steam Deck's hot G-buffer vertex shader 771 -> 411
// instructions); a later draw whose words differ binds the fast-linked
// pipeline (spec_w3_matches).
const bool g_relink_w3 = [] {
    const char* e = std::getenv("BBHOST_RELINK_W3");
    return !(e && e[0] == '0');
}();
// BBHOST_RELINK_STRIDE=0: the same loads' strides stay read from the params;
// by default they are constants too (gcn::kCbStrideSpecId), checked the same way.
const bool g_relink_stride = [] {
    const char* e = std::getenv("BBHOST_RELINK_STRIDE");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_spec_w3_misses{0}, g_spec_w3_draws{0};
// Under g.mu: the relink link_gfx_pipeline left pending, with the draw's params.
void queue_library_relink(GfxPipeline& pl, const gcn::StageParams params[2]) {
    pl.relink_pending = false;
    Precompiler& w = precompiler();
    PrecompileJob job;
    job.type = 0;
    job.relink = &pl;
    std::copy(pl.relink_libs, pl.relink_libs + 4, job.libraries);
    job.layout = pl.layout;
    job.specialize_formats = pl.relink_formats;
    std::copy(pl.relink_vertex_formats, pl.relink_vertex_formats + 16, job.vertex_formats);
    job.polygon_mode = pl.relink_polygon;
    job.depth_clamp = pl.relink_clamp;
    if (g_relink_w3 && !pl.tess_hw && pl.lean) {
        for (int st = 0; st < 2; ++st) {
            const gcn::TranslateResult& meta = st ? pl.ps.meta() : pl.vs.meta();
            if (st == 1 && !pl.ps.module) continue;
            for (std::size_t i = 0; i < meta.buffers.size() && i < 16; ++i) {
                const std::uint32_t stride = g_relink_stride ? params[st].cb_stride[i] : 0;
                if (!meta.buffers[i].indexed || !(params[st].cb_w3[i] | stride)) continue;
                job.w3[st][i] = params[st].cb_w3[i];
                job.stride[st][i] = stride;
                job.w3_mask[st] |= static_cast<std::uint16_t>(1u << i);
            }
        }
        std::memcpy(pl.spec_w3, job.w3, sizeof(pl.spec_w3));
        std::memcpy(pl.spec_stride, job.stride, sizeof(pl.spec_stride));
        pl.spec_w3_mask[0] = job.w3_mask[0];
        pl.spec_w3_mask[1] = job.w3_mask[1];
    }
    // BBHOST_RELINK_W3_LOG=1: what each relink's indexed loads had.
    static const bool w3_log = [] {
        const char* e = std::getenv("BBHOST_RELINK_W3_LOG");
        return e && e[0] == '1';
    }();
    if (w3_log) {
        std::string what;
        for (int st = 0; st < 2; ++st) {
            const gcn::TranslateResult& meta = st ? pl.ps.meta() : pl.vs.meta();
            for (std::size_t i = 0; i < meta.buffers.size() && i < 16; ++i) {
                char b[64];
                std::snprintf(b, sizeof(b), " %s%zu%s=%08x/%u", st ? "ps" : "vs", i, meta.buffers[i].indexed ? "i" : "", params[st].cb_w3[i],
                              params[st].cb_stride[i]);
                what += b;
            }
        }
        host_log("render: relink %s%s: words 3 masks %04x %04x;%s", pl.name.c_str(), pl.lean ? " (no-fallback)" : "", job.w3_mask[0],
                 job.w3_mask[1], what.c_str());
    }
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_back(std::move(job));
    }
    w.relinks_queued.fetch_add(1, std::memory_order_relaxed);
    w.cv.notify_one();
}
void relink_pipeline(Precompiler& w, const PrecompileJob& job) {
    const auto t0 = std::chrono::steady_clock::now();
    VkPipeline libraries[4];
    std::copy(job.libraries, job.libraries + 4, libraries);
    // The pipeline's stages and layouts are fixed once it is linked, and
    // graphics shader modules are never destroyed.
    const GfxPipeline& pl = *job.relink;
    if (job.specialize_formats || job.w3_mask[0]) {
        std::uint32_t words[48] = {};
        if (job.specialize_formats) std::copy(job.vertex_formats, job.vertex_formats + 16, words);
        std::copy(job.w3[0], job.w3[0] + 16, words + 16);
        std::copy(job.stride[0], job.stride[0] + 16, words + 32);
        if (const VkPipeline pr = pre_raster_library(pl.vs, pl.gs, job.polygon_mode, job.depth_clamp, job.layout, pl.set_layouts, nullptr, words)) {
            libraries[1] = pr;
            if (job.specialize_formats) w.relinks_specialized.fetch_add(1, std::memory_order_relaxed);
        } else {
            w.relink_spec_failed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (job.w3_mask[1]) {
        std::uint32_t words[32] = {};
        std::copy(job.w3[1], job.w3[1] + 16, words);
        std::copy(job.stride[1], job.stride[1] + 16, words + 16);
        if (const VkPipeline fs = fragment_library(pl.ps.meta(), pl.ps.module, pl.set_layouts[1], job.layout, words)) {
            libraries[2] = fs;
        } else {
            w.relink_spec_failed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (job.w3_mask[0] | job.w3_mask[1]) w.relinks_w3.fetch_add(1, std::memory_order_relaxed);
    VkPipelineLibraryCreateInfoKHR lci{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
    lci.libraryCount = 4;
    lci.pLibraries = libraries;
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &lci;
    gpci.flags = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
    gpci.layout = job.layout;
    const bool want_stats = pipeline_stats_wanted(job.relink->name);
    if (want_stats) gpci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    VkPipeline p = VK_NULL_HANDLE;
    const bool ok = vkCreateGraphicsPipelines(g.device, want_stats ? VK_NULL_HANDLE : PipelineCacheUse().cache, 1, &gpci, nullptr, &p) == VK_SUCCESS;
    if (ok && want_stats) {
        // A name has several pipelines (the no-fallback variant and the one with
        // the page-table paths, other fixed state): each gets its own file.
        static std::atomic<int> seq{0};
        write_pipeline_stats(p, job.relink->name + (job.relink->lean ? "-lean-" : "-") + std::to_string(seq.fetch_add(1)));
    }
    w.relink_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
    if (!ok) {
        w.relinks_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.relinked.emplace_back(job.relink, p);
    }
    w.relinks_done.fetch_add(1, std::memory_order_relaxed);
    g_relinks_ready.store(true, std::memory_order_relaxed);
}
// Under g.mu, after apply_lean_builds_locked: swaps optimized relinks into their
// pipelines. Command buffers already recorded with the fast-linked pipeline keep
// it (pipelines are never destroyed). A variant a worker has not handed over yet
// waits for the next call.
void apply_relinks_locked() {
    if (!g_relinks_ready.load(std::memory_order_relaxed)) return;
    Precompiler& w = precompiler();
    std::vector<std::pair<GfxPipeline*, VkPipeline>> done;
    {
        std::lock_guard<std::mutex> lk(w.mu);
        done.swap(w.relinked);
        g_relinks_ready.store(false, std::memory_order_relaxed);
    }
    std::vector<std::pair<GfxPipeline*, VkPipeline>> later;
    for (const auto& [pl, pipeline] : done) {
        if (pl->building) {
            later.emplace_back(pl, pipeline);
            continue;
        }
        if (pl->spec_w3_mask[0] | pl->spec_w3_mask[1]) pl->generic = pl->pipeline;  // spec_w3_matches
        pl->pipeline = pipeline;
        w.relinks_adopted.fetch_add(1, std::memory_order_relaxed);
    }
    if (!later.empty()) {
        std::lock_guard<std::mutex> lk(w.mu);
        w.relinked.insert(w.relinked.end(), later.begin(), later.end());
        g_relinks_ready.store(true, std::memory_order_relaxed);
    }
}
// A translation error without its offset and detail, as a failure reason.
std::string translation_error_kind(const std::string& e) {
    std::string k = e.size() > 8 && e[6] == ':' ? e.substr(8) : e;
    if (const std::size_t p = k.find(" ("); p != std::string::npos) k.resize(p);
    return "translation: " + k;
}
// BBHOST_EFT_TWINS: see precompile_ps.
const bool g_eft_twins = [] {
    const char* e = std::getenv("BBHOST_EFT_TWINS");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_eft_twins_built{0};

void precompile_ps(Precompiler& w, const PrecompileJob& job) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto give_up = [&](const std::string& why) {
        w.failed.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(w.mu);
        w.records[job.name].failed = true;
        w.fail_reasons[why] += 1;
    };
    const std::vector<std::uint8_t>& c = job.container;
    gcn::ShaderCode code;
    std::size_t shdr = 0;
    while (shdr + 0x60 <= c.size() && shdr < 0x100 && std::memcmp(c.data() + shdr, "Shdr", 4) != 0) ++shdr;
    if (shdr + 0x60 > c.size() || shdr >= 0x100 || !gcn::shader_code(c, code) || code.type != 2) return give_up("not a pixel-shader container");
    shader_patch_apply(job.name, code.words);  // the same program the draw will read (host/shader_patch.h)
    std::uint32_t regs[16];
    std::memcpy(regs, c.data() + shdr + 16, sizeof(regs));
    // The Gnm structure at Shdr + 0x10: usage entries at +0x3c, then the input
    // semantic words.
    const std::size_t base = shdr + 16;
    const std::uint32_t declared_inputs = c[base + 0x38];
    const std::size_t input_table = base + 0x3c + static_cast<std::size_t>(c[base + 3]) * 4;
    const gcn::Program prog = gcn::decode(code.words.data(), code.words.size());
    if (!prog.errors.empty()) return give_up("decode");
    gcn::TranslateOptions paths_options;
    paths_options.stage = gcn::Stage::Pixel;
    paths_options.rsrc1 = regs[4];
    paths_options.rsrc2 = regs[5];
    // Its SPIR-V is read too: whether the program can discard (note_ps_early_tests).
    const gcn::TranslateResult paths = translate_cached(prog, paths_options);  // resource paths and the inputs read
    offer_paths(code.words, paths);
    PrecompileRecord rec;
    rec.rsrc1 = regs[4];
    rec.rsrc2 = regs[5];
    rec.ps_input_ena = regs[8];
    // Early fragment tests only for a program that discards: the variant draws
    // without depth writes (lights) ask for.
    note_ps_early_tests(code.words, paths.spirv);
    rec.early_fragment_tests = ps_early_tests_matter(code.words) == 1;
    for (const gcn::PredictedImage& p : gcn::predict_image_dims(prog, paths)) rec.dims.emplace_back(p.dim, p.arrayed);
    rec.modes.assign(paths.samplers.size(), false);
    std::uint32_t read_inputs = 0, registers[32] = {};
    for (std::uint32_t k : paths.ps_inputs) {
        if (k >= 32 || k >= declared_inputs || input_table + 2 * (k + 1) > c.size()) return give_up("an input outside the binary's table");
        std::uint16_t word;
        std::memcpy(&word, c.data() + input_table + 2 * k, 2);
        if (((word >> 12) | (word >> 10)) & 1) rec.flat_mask |= 1u << k;  // FLAT_SHADE, as sub_14873a0 builds it
        // The register the game's input map gives this input: the vertex-shader
        // output with its semantic, which every binary puts at semantic - 0xf.
        const std::uint32_t semantic = word & 0xff;
        if (semantic < 0xf || semantic - 0xf >= 32) return give_up("an input semantic outside the param registers");
        read_inputs |= 1u << k;
        registers[k] = semantic - 0xf;
    }
    rec.input_map = ps_input_locations(read_inputs, registers);
    const PsStageInputs in{&code.words, rec.rsrc1, rec.rsrc2, rec.ps_input_ena, rec.flat_mask, rec.early_fragment_tests, true,
                           &rec.dims, &rec.modes, &rec.input_map};
    const bool lift = decomp_selects(job.name);
    const std::uint64_t key = lift_key(ps_stage_key(in), lift);
    const bool cacheable = g_stage_cache_on;
    ShaderStage stage;
    if (const std::shared_ptr<const CachedStage> hit = cacheable ? cached_stage(key) : nullptr) {
        stage.take(hit);
        stage.module = hit->module;
    } else {
        const gcn::TranslateOptions options = ps_translate_options(in);
        stage.fresh() = translate_cached(prog, options);
        if (!stage.meta().ok()) return give_up(translation_error_kind(stage.meta().errors[0]));
        note_ps_early_tests(code.words, stage.meta().spirv);
        if (lift) {
            gcn::LiftResult lifted = lift_cached(false, prog, options, stage.meta());
            if (lifted.ok()) {
                stage.fresh().spirv = std::move(lifted.spirv);
                stage.fresh().wave64_needs = 0;  // as build_gfx_pipeline's lift
                stage.lifted = true;
            }
        }
        if (!make_module(stage.meta().spirv, stage.module)) return give_up("shader module");
        if (cacheable) cache_stage(key, stage.meta(), stage.module, stage.lifted);
    }
    w.translate_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
    const auto t1 = std::chrono::steady_clock::now();
    const VkDescriptorSetLayout set = g.gfx_set_layout;  // both sets take the shared layout under libraries (build_gfx_pipeline)
    const VkPipelineLayout layout = set ? stage_pipeline_layout(g.gfx_set_layout, set) : VK_NULL_HANDLE;
    dump_spirv("precompile-" + job.name, stage.meta().spirv);
    t_library_owner = job.name.c_str();
    const VkPipeline library = layout ? fragment_library(stage.meta(), stage.module, set, layout) : VK_NULL_HANDLE;
    t_library_owner = nullptr;
    w.library_us.fetch_add(pl_us_since(t1), std::memory_order_relaxed);
    if (!library) return give_up("fragment-shader library");
    // BBHOST_EFT_TWINS (on unless 0): a program that discards is compiled for
    // early fragment tests, the variant the draws without depth writes (the
    // lights) ask for - and a draw of it with depth or stencil writes then
    // translated and built its twin on the command processor, nine of them in
    // a Central Yharnam warp tour, each first visit's hitch (the "early
    // fragment tests (draw without)" misses). The twin is the
    // same module without that one execution mode: derived here, cached
    // under its own key, and its library built beside this one.
    std::uint64_t twin_key = 0;
    if (rec.early_fragment_tests && g_eft_twins) {
        PsStageInputs off = in;
        off.early_fragment_tests = false;
        twin_key = lift_key(ps_stage_key(off), lift);
        const std::shared_ptr<const CachedStage> have = cacheable ? cached_stage(twin_key) : nullptr;
        gcn::TranslateResult twin = have ? have->meta : stage.meta();
        VkShaderModule twin_module = have ? have->module : VK_NULL_HANDLE;
        bool ok = have != nullptr;
        // A program that writes depth gets no early tests from the translator
        // either way: then its twin is the same words, in a module of its own.
        if (!ok) {
            spirv_drop_early_fragment_tests(twin.spirv);
            ok = make_module(twin.spirv, twin_module);
            if (ok && cacheable) cache_stage(twin_key, twin, twin_module, stage.lifted);
        }
        t_library_owner = job.name.c_str();
        ok = ok && fragment_library(twin, twin_module, set, layout) != VK_NULL_HANDLE;
        t_library_owner = nullptr;
        // Uncached, the module served the library build only.
        if (!have && !cacheable && twin_module) vkDestroyShaderModule(g.device, twin_module, nullptr);
        if (ok) g_eft_twins_built.fetch_add(1, std::memory_order_relaxed);
        else twin_key = 0;
    }
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.records[job.name] = std::move(rec);
        w.stage_keys.insert(key);
        if (twin_key) w.stage_keys.insert(twin_key);
    }
    w.done.fetch_add(1, std::memory_order_relaxed);
}
// A stage from the manifest, as the draw that recorded it translated it; then
// its library as creation-time compiling builds one (a tessellated draw's
// stages only go into the stage cache: their pipelines add stages of ours).
void precompile_manifest(const ManifestStage& m) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t key = m.key();
    if (m.stage == 3) {  // the tessellator's LS: into the stage cache, as a pipeline build would put it
        const gcn::Program prog = gcn::decode(m.words.data(), m.words.size());
        gcn::Program fprog;
        if (!m.fetch_words.empty()) fprog = gcn::decode(m.fetch_words.data(), m.fetch_words.size());
        ShaderStage ls;
        bool ok = prog.errors.empty() && fprog.errors.empty();
        if (ok && !(g_stage_cache_on && cached_stage(key))) {
            ls.fresh() = translate_cached(prog, tess_ls_options(m.rsrc1, m.rsrc2, m.tess_attr_vec4s, m.fetch_words.empty() ? nullptr : &fprog));
            ok = ls.meta().ok() && make_module(ls.meta().spirv, ls.module);
            if (ok && g_stage_cache_on) cache_stage(key, ls.meta(), ls.module, false);
        }
        g_manifest.compile_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (ok) {
            g_manifest.ready.insert(key);
            g_manifest.compiled.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_manifest.by_key.erase(key);
            g_manifest.dirty = true;
            g_manifest.failed.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    if (m.stage == 2) {  // resource paths: translated as paths_for would, and offered to it
        const gcn::Program prog = gcn::decode(m.words.data(), m.words.size());
        if (prog.errors.empty()) {
            gcn::TranslateOptions o;
            o.stage = static_cast<gcn::Stage>(m.domain_level);
            o.rsrc1 = m.rsrc1;
            o.rsrc2 = m.rsrc2;
            offer_paths(m.words, translate_cached(prog, o, TranslationUse::kPaths));
        }
        g_manifest.compile_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (prog.errors.empty()) {
            g_manifest.compiled.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_manifest.by_key.erase(key);
            g_manifest.dirty = true;
            g_manifest.failed.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    const gcn::Program prog = gcn::decode(m.words.data(), m.words.size());
    gcn::Program fetch_prog;
    if (!m.fetch_words.empty()) fetch_prog = gcn::decode(m.fetch_words.data(), m.fetch_words.size());
    ShaderStage stage;
    bool ok = prog.errors.empty() && fetch_prog.errors.empty();
    if (const std::shared_ptr<const CachedStage> hit = ok && g_stage_cache_on ? cached_stage(key) : nullptr) {
        stage.take(hit);
        stage.module = hit->module;
    } else if (ok) {
        if (m.stage) {
            const gcn::TranslateOptions options = ps_translate_options(m.ps_inputs());
            stage.fresh() = translate_cached(prog, options);
            if (stage.meta().ok() && m.lift) {
                gcn::LiftResult lifted = lift_cached(false, prog, options, stage.meta());
                if (lifted.ok()) {
                    stage.fresh().spirv = std::move(lifted.spirv);
                    stage.fresh().wave64_needs = 0;  // as build_gfx_pipeline's lift
                    stage.lifted = true;
                }
            }
        } else {
            const gcn::TranslateOptions options = vs_translate_options(m.vs_inputs(), m.fetch_words.empty() ? nullptr : &fetch_prog);
            stage.fresh() = translate_cached(prog, options);
            if (stage.meta().ok() && m.lift) {
                gcn::LiftResult lifted = lift_cached(true, prog, options, stage.meta());
                if (lifted.ok()) {
                    stage.fresh().spirv = std::move(lifted.spirv);
                    stage.lifted = true;
                }
            }
        }
        ok = stage.meta().ok() && make_module(stage.meta().spirv, stage.module);
        if (ok && g_stage_cache_on) cache_stage(key, stage.meta(), stage.module, stage.lifted);
    }
    if (ok && !m.tess_hw && !m.domain_level) {
        const VkDescriptorSetLayout set = g.gfx_set_layout;  // both sets take the shared layout under libraries
        const VkDescriptorSetLayout sets[2] = {set, set};
        const VkPipelineLayout layout = set ? stage_pipeline_layout(set, set) : VK_NULL_HANDLE;
        t_library_owner = m.name.c_str();
        if (m.stage) {
            ok = layout && fragment_library(stage.meta(), stage.module, set, layout) != VK_NULL_HANDLE;
        } else {
            const ShaderStage no_gs;
            ok = layout && pre_raster_library(stage, no_gs, VK_POLYGON_MODE_FILL, false, layout, sets) != VK_NULL_HANDLE;
        }
        t_library_owner = nullptr;
    }
    g_manifest.compile_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_manifest.mu);
    if (ok) {
        g_manifest.ready.insert(key);
        g_manifest.compiled.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_manifest.by_key.erase(key);  // not kept: the running build cannot compile it
        g_manifest.dirty = true;
        g_manifest.failed.fetch_add(1, std::memory_order_relaxed);
    }
}
// A vertex shader when GX creates it (step 6d): its no-fallback stage on vertex
// input, with elements from its input table (location = entry, VGPR =
// register, components) and formats from the params block, and its
// pre-rasterization library with fill mode, no depth clamp and no geometry shader.
void precompile_vs(Precompiler& w, const PrecompileJob& job) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto give_up = [&](const std::string& why) {
        w.vs_failed.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(w.mu);
        w.vs_records[job.name].failed = true;
        w.fail_reasons["VS " + why] += 1;
    };
    const std::vector<std::uint8_t>& c = job.container;
    gcn::ShaderCode code;
    std::size_t shdr = 0;
    while (shdr + 0x60 <= c.size() && shdr < 0x100 && std::memcmp(c.data() + shdr, "Shdr", 4) != 0) ++shdr;
    if (shdr + 0x60 > c.size() || shdr >= 0x100 || !gcn::shader_code(c, code) || code.type != 1) return give_up("not a vertex-shader container");
    if (!g_vertex_format_params) return give_up("formats are constants (BBHOST_VERTEX_FORMAT_PARAMS=0)");
    std::uint32_t regs[16];
    std::memcpy(regs, c.data() + shdr + 16, sizeof(regs));
    const std::size_t base = shdr + 16;
    const std::size_t inputs = c[base + 0x24], input_table = base + 0x28 + static_cast<std::size_t>(c[base + 3]) * 4;
    if (inputs == 0) return give_up("no vertex inputs");
    if (inputs > 16 || input_table + inputs * 4 > c.size()) return give_up("an input table past 16 entries or the container");
    Precompiler::VsRecord rec;
    for (std::size_t k = 0; k < inputs; ++k) {
        gcn::VertexElement el;
        el.location = static_cast<std::uint32_t>(k);
        el.vdata = c[input_table + 4 * k + 1];
        el.count = c[input_table + 4 * k + 2];
        if (el.count == 0 || el.count > 4) return give_up("an input with no components or more than four");
        rec.elements.push_back(el);
    }
    // VsStageRegs from the register block's third word: pgm_lo, pgm_hi, rsrc1,
    // rsrc2, SPI_VS_OUT_CONFIG, SPI_SHADER_POS_FORMAT, PA_CL_VS_OUT_CNTL. The
    // game's sceGnmSetVsShader call adds VGPR_COMP_CNT (bits 24-25) to RSRC1: 3
    // in 38 of 40 captured draws.
    rec.rsrc1 = (regs[4] & ~(3u << 24)) | (3u << 24);
    rec.rsrc2 = regs[5];
    rec.out_cntl = regs[8];
    const gcn::Program prog = gcn::decode(code.words.data(), code.words.size());
    if (!prog.errors.empty()) return give_up("decode");
    gcn::TranslateOptions paths_options;  // as paths_for
    paths_options.stage = gcn::Stage::Vertex;
    paths_options.rsrc1 = rec.rsrc1;
    paths_options.rsrc2 = rec.rsrc2;
    const gcn::TranslateResult paths = translate_cached(prog, paths_options, TranslationUse::kPaths);
    offer_paths(code.words, paths);
    for (const gcn::PredictedImage& p : gcn::predict_image_dims(prog, paths)) rec.dims.emplace_back(p.dim, p.arrayed);
    rec.modes.assign(paths.samplers.size(), false);
    static const std::vector<std::uint32_t> no_fetch;
    const VsStageInputs in{&code.words, &no_fetch, rec.rsrc1, rec.rsrc2, rec.out_cntl, &rec.elements, true, true, &rec.dims, &rec.modes};
    const bool lift = !g_vs_dispatcher && decomp_selects(job.name);  // as build_gfx_pipeline's vs_decomp
    const std::uint64_t key = lift_key(vs_stage_key(in), lift);
    const bool cacheable = g_stage_cache_on;
    ShaderStage stage;
    if (const std::shared_ptr<const CachedStage> hit = cacheable ? cached_stage(key) : nullptr) {
        stage.take(hit);
        stage.module = hit->module;
    } else {
        const gcn::TranslateOptions options = vs_translate_options(in, nullptr);
        stage.fresh() = translate_cached(prog, options);
        if (!stage.meta().ok()) return give_up(translation_error_kind(stage.meta().errors[0]));
        // The lift runs here as well, so a shader compiled at creation is the
        // same stage the draw wants and is not thrown away for it.
        if (lift) {
            gcn::LiftResult lifted = lift_cached(true, prog, options, stage.meta());
            if (lifted.ok()) {
                stage.fresh().spirv = std::move(lifted.spirv);
                stage.lifted = true;
            }
        }
        dump_spirv("precompile-" + job.name + "-vs", stage.meta().spirv);
        if (!make_module(stage.meta().spirv, stage.module)) return give_up("shader module");
        if (cacheable) cache_stage(key, stage.meta(), stage.module, stage.lifted);
    }
    w.vs_translate_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
    const auto t1 = std::chrono::steady_clock::now();
    const VkDescriptorSetLayout sets[2] = {g.gfx_set_layout, g.gfx_set_layout};  // both sets take the shared layout under libraries
    const VkPipelineLayout layout = stage_pipeline_layout(g.gfx_set_layout, g.gfx_set_layout);
    const ShaderStage no_gs;
    t_library_owner = job.name.c_str();
    const VkPipeline library = layout ? pre_raster_library(stage, no_gs, VK_POLYGON_MODE_FILL, false, layout, sets) : VK_NULL_HANDLE;
    t_library_owner = nullptr;
    w.vs_library_us.fetch_add(pl_us_since(t1), std::memory_order_relaxed);
    if (!library) return give_up("pre-rasterization library");
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.vs_records[job.name] = std::move(rec);
        w.vs_stage_keys.insert(key);
    }
    w.vs_done.fetch_add(1, std::memory_order_relaxed);
}
bool note_vs_precompile_hit(std::uint64_t vs_key) {  // true: a stage compiled at creation
    if (!precompile_enabled()) return false;
    Precompiler& w = precompiler();
    std::lock_guard<std::mutex> lk(w.mu);
    if (!w.vs_stage_keys.count(vs_key)) return false;
    w.vs_stage_hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}
// A draw translated a stage with no record of its compile at creation: still
// in the queue (how deep), being compiled that moment, or never queued - GX
// did not create it, or created it after the draw.
void note_not_ready_locked(Precompiler& w, const std::string& name, int type, std::uint64_t& never, std::uint64_t& queued,
                           std::uint64_t& compiling, std::vector<std::string>& first) {
    std::size_t position = 0;
    bool in_queue = false;
    for (const PrecompileJob& j : w.jobs) {
        if (j.type == type && j.name == name) {
            in_queue = true;
            break;
        }
        ++position;
    }
    const bool running = !in_queue && w.running.count(name) != 0;
    ++(in_queue ? queued : running ? compiling : never);
    if (first.size() >= 32) return;
    char buf[96];
    const unsigned long long flip = hle_video_flip_count();
    if (in_queue) {
        std::snprintf(buf, sizeof(buf), " %s@%llu:queued#%zu/%zu", name.c_str(), flip, position, w.jobs.size());
    } else {
        std::snprintf(buf, sizeof(buf), " %s@%llu:%s", name.c_str(), flip, running ? "compiling" : "never-queued");
    }
    first.push_back(buf);
}
// A draw translated a vertex shader after all: why its compile at creation did
// not serve (the first differing input).
void note_vs_precompile_miss(const std::string& name, const VsStageInputs& in) {
    if (!precompile_enabled() || !in.no_fallback) return;  // only the no-fallback variant is compiled ahead
    Precompiler& w = precompiler();
    std::lock_guard<std::mutex> lk(w.mu);
    const auto it = w.vs_records.find(name);
    if (it == w.vs_records.end()) {
        note_not_ready_locked(w, name, 1, w.vs_miss_not_ready, w.vs_miss_queued, w.vs_miss_compiling, w.vs_not_ready);
        return;
    }
    const Precompiler::VsRecord& r = it->second;
    const auto same_elements = [&] {
        if (in.elements->size() != r.elements.size()) return false;
        for (std::size_t k = 0; k < r.elements.size(); ++k) {
            const gcn::VertexElement& a = r.elements[k];
            const gcn::VertexElement& b = (*in.elements)[k];
            if (a.location != b.location || a.vdata != b.vdata || a.count != b.count) return false;
        }
        return true;
    };
    if (r.failed) ++w.vs_miss_failed;
    else if (!in.elements) ++w.vs_miss_path;
    else if ((r.rsrc1 ^ in.rsrc1) & (3u << 24)) ++w.vs_miss_comp;
    else if (r.rsrc1 != in.rsrc1 || r.rsrc2 != in.rsrc2) ++w.vs_miss_rsrc;
    else if (r.out_cntl != in.out_cntl) {
        // The first few, with both values: which bits a draw adds (its clip
        // planes, the clip-distance varying) - what a twin compiled at
        // creation would need to cover.
        if (w.vs_miss_out_cntl++ < 16) {
            host_log("render: vertex shader %s was compiled at creation for PA_CL_VS_OUT_CNTL %08x; a draw wants %08x", name.c_str(),
                     r.out_cntl, in.out_cntl);
        }
    }
    else if (!in.formats_from_params) ++w.vs_miss_formats;
    else if (!same_elements()) ++w.vs_miss_elements;
    else if (r.dims != *in.dims) ++w.vs_miss_dims;
    else if (r.modes != *in.modes) ++w.vs_miss_modes;
    else ++w.vs_miss_other;
}
bool note_precompile_hit(std::uint64_t ps_key) {  // true: a stage compiled at creation
    if (!precompile_enabled()) return false;
    Precompiler& w = precompiler();
    std::lock_guard<std::mutex> lk(w.mu);
    if (!w.stage_keys.count(ps_key)) return false;
    w.stage_hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}
// A draw translated a pixel shader after all: why its compile at creation did
// not serve (the first differing input).
void note_precompile_miss(const std::string& name, const PsStageInputs& in) {
    if (!precompile_enabled() || !in.no_fallback) return;  // only the no-fallback variant is compiled ahead
    Precompiler& w = precompiler();
    std::lock_guard<std::mutex> lk(w.mu);
    if (decomp_selects(name)) ++w.miss_lifted;  // of them lifted: compiled ahead the same way, under the lift's key
    const auto it = w.records.find(name);
    if (it == w.records.end()) {
        note_not_ready_locked(w, name, 2, w.miss_not_ready, w.miss_queued, w.miss_compiling, w.not_ready);
        return;
    }
    const PrecompileRecord& r = it->second;
    if (r.failed) ++w.miss_failed;
    else if (r.flat_mask != in.flat_mask) ++w.miss_flat;
    else if (r.input_map != *in.input_map) ++w.miss_input_map;
    else if (r.early_fragment_tests != in.early_fragment_tests) ++(in.early_fragment_tests ? w.miss_eft_on : w.miss_eft_off);
    else if (r.dims != *in.dims) ++w.miss_dims;
    else if (r.modes != *in.modes) ++w.miss_modes;
    else if (r.ps_input_ena != in.ps_input_ena) ++w.miss_ena;
    else if (r.rsrc1 != in.rsrc1 || r.rsrc2 != in.rsrc2) ++w.miss_rsrc;
    else ++w.miss_other;
}
namespace manifest_file {
constexpr std::uint32_t kMagic = 0x4d534242;  // "BBSM"
// 2: a pixel entry's clip_discard carries its location (count | location << 8).
constexpr std::uint32_t kVersion = 2;
struct Writer {
    std::vector<std::uint8_t> out;
    void u32(std::uint32_t v) { out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(&v), reinterpret_cast<const std::uint8_t*>(&v) + 4); }
    void u8(std::uint8_t v) { out.push_back(v); }
    template <typename T>
    void u32s(const std::vector<T>& v) {
        u32(static_cast<std::uint32_t>(v.size()));
        for (const T& x : v) u32(static_cast<std::uint32_t>(x));
    }
};
struct Reader {
    const std::uint8_t* p;
    const std::uint8_t* end;
    bool ok = true;
    std::uint32_t u32() {
        std::uint32_t v = 0;
        if (end - p < 4) return ok = false, 0;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    std::uint8_t u8() {
        if (p >= end) return ok = false, 0;
        return *p++;
    }
    std::uint32_t count(std::size_t each) {  // a length that fits what is left
        const std::uint32_t n = u32();
        if (static_cast<std::size_t>(end - p) < static_cast<std::size_t>(n) * each) return ok = false, 0;
        return n;
    }
};
void write(Writer& w, const ManifestStage& m) {
    w.u8(m.stage);
    w.u8((m.lift ? 1 : 0) | (m.early_fragment_tests ? 2 : 0) | (m.no_fallback ? 4 : 0) | (m.has_elements ? 8 : 0) | (m.formats_from_params ? 16 : 0) |
         (m.tess_hw ? 32 : 0) | (m.tess_attrs ? 64 : 0));
    w.u32(static_cast<std::uint32_t>(m.name.size()));
    w.out.insert(w.out.end(), m.name.begin(), m.name.end());
    w.u32s(m.words);
    w.u32s(m.fetch_words);
    for (std::uint32_t v : {m.rsrc1, m.rsrc2, m.ps_input_ena, m.flat_mask, m.out_cntl, m.domain_level, m.tess_control_points, m.tess_attr_vec4s}) w.u32(v);
    w.u32(static_cast<std::uint32_t>(m.elements.size()));
    for (const gcn::VertexElement& e : m.elements) {
        w.u32(e.location);
        w.u32(e.vdata);
        w.u32(e.count);
        w.u32(e.w3);
    }
    w.u32(static_cast<std::uint32_t>(m.dims.size()));
    for (const auto& d : m.dims) {
        w.u32(d.first);
        w.u8(d.second ? 1 : 0);
    }
    w.u32(static_cast<std::uint32_t>(m.modes.size()));
    for (bool b : m.modes) w.u8(b ? 1 : 0);
    w.u32(static_cast<std::uint32_t>(m.input_map.size()));
    w.out.insert(w.out.end(), m.input_map.begin(), m.input_map.end());
}
bool read(Reader& r, ManifestStage& m) {
    m.stage = r.u8();
    const std::uint8_t f = r.u8();
    m.lift = f & 1;
    m.early_fragment_tests = f & 2;
    m.no_fallback = f & 4;
    m.has_elements = f & 8;
    m.formats_from_params = f & 16;
    m.tess_hw = f & 32;
    m.tess_attrs = f & 64;
    const std::uint32_t name = r.count(1);
    if (!r.ok) return false;
    m.name.assign(reinterpret_cast<const char*>(r.p), name);
    r.p += name;
    m.words.resize(r.count(4));
    for (std::uint32_t& x : m.words) x = r.u32();
    m.fetch_words.resize(r.count(4));
    for (std::uint32_t& x : m.fetch_words) x = r.u32();
    for (std::uint32_t* v : {&m.rsrc1, &m.rsrc2, &m.ps_input_ena, &m.flat_mask, &m.out_cntl, &m.domain_level, &m.tess_control_points, &m.tess_attr_vec4s}) {
        *v = r.u32();
    }
    m.elements.resize(r.count(16));
    for (gcn::VertexElement& e : m.elements) {
        e.location = r.u32();
        e.vdata = r.u32();
        e.count = r.u32();
        e.w3 = r.u32();
    }
    m.dims.resize(r.count(5));
    for (auto& d : m.dims) {
        d.first = r.u32();
        d.second = r.u8() != 0;
    }
    m.modes.resize(r.count(1));
    for (std::size_t k = 0; k < m.modes.size(); ++k) m.modes[k] = r.u8() != 0;
    m.input_map.resize(r.count(1));
    for (std::uint8_t& x : m.input_map) x = r.u8();
    return r.ok && m.stage <= 3 && (m.stage != 2 || m.domain_level <= static_cast<std::uint32_t>(gcn::Stage::TessEval)) && !m.words.empty();
}
}  // namespace manifest_file

void precompile_report() {
    compute_optimize_report();
    compute_precompile_report();
    translation_cache_report();
    if (const std::string r = ps_wave_census(false); !r.empty()) host_log("render: %s", r.c_str());
    if (!precompile_enabled()) return;
    Precompiler& w = precompiler();
    std::lock_guard<std::mutex> lk(w.mu);
    if (g_library_relink) {
        host_log("render: optimized relinks: %llu queued, %llu linked (%.1f s), %llu failed, %llu swapped in; %llu with the vertex "
                 "formats as constants, %llu with indexed loads' words 3 and strides (%llu of those libraries failed; %llu of %llu draws "
                 "on such a pipeline had other words or strides and drew with the fast-linked one)",
                 static_cast<unsigned long long>(w.relinks_queued.load()), static_cast<unsigned long long>(w.relinks_done.load()),
                 w.relink_us.load() / 1e6, static_cast<unsigned long long>(w.relinks_failed.load()),
                 static_cast<unsigned long long>(w.relinks_adopted.load()), static_cast<unsigned long long>(w.relinks_specialized.load()),
                 static_cast<unsigned long long>(w.relinks_w3.load()), static_cast<unsigned long long>(w.relink_spec_failed.load()),
                 static_cast<unsigned long long>(g_spec_w3_misses.load()), static_cast<unsigned long long>(g_spec_w3_draws.load()));
    }
    host_log("render: vertex shaders compiled at creation: %llu queued, %llu done, %llu failed (translate %.1f s, libraries %.1f s); "
             "pipeline builds that found the stage ready %llu; no-fallback translations it did not serve: never queued %llu, still queued %llu, "
             "being compiled %llu, failed %llu, "
             "without vertex input %llu, VGPR_COMP_CNT %llu, resource registers %llu, output control %llu, constant formats %llu, "
             "elements %llu, image dims %llu, sampler modes %llu, other %llu",
             static_cast<unsigned long long>(w.vs_queued.load()), static_cast<unsigned long long>(w.vs_done.load()),
             static_cast<unsigned long long>(w.vs_failed.load()), w.vs_translate_us.load() / 1e6, w.vs_library_us.load() / 1e6,
             static_cast<unsigned long long>(w.vs_stage_hits.load()), static_cast<unsigned long long>(w.vs_miss_not_ready),
             static_cast<unsigned long long>(w.vs_miss_queued), static_cast<unsigned long long>(w.vs_miss_compiling),
             static_cast<unsigned long long>(w.vs_miss_failed), static_cast<unsigned long long>(w.vs_miss_path),
             static_cast<unsigned long long>(w.vs_miss_comp), static_cast<unsigned long long>(w.vs_miss_rsrc),
             static_cast<unsigned long long>(w.vs_miss_out_cntl), static_cast<unsigned long long>(w.vs_miss_formats),
             static_cast<unsigned long long>(w.vs_miss_elements), static_cast<unsigned long long>(w.vs_miss_dims),
             static_cast<unsigned long long>(w.vs_miss_modes), static_cast<unsigned long long>(w.vs_miss_other));
    if (!w.vs_not_ready.empty()) {
        std::string v;
        for (const std::string& n : w.vs_not_ready) v += n;
        host_log("render: vertex shaders a draw translated with no record of their compile at creation:%s", v.c_str());
    }
    {
        std::lock_guard<std::mutex> lk(g_pre_raster_states.mu);
        host_log("render: pre-rasterization libraries built again for a vertex shader that had one in another raster state: %llu "
                 "(depth clamp %llu, polygon mode %llu, rect-list geometry shader %llu)",
                 static_cast<unsigned long long>(g_pre_raster_states.again), static_cast<unsigned long long>(g_pre_raster_states.again_clamp),
                 static_cast<unsigned long long>(g_pre_raster_states.again_mode), static_cast<unsigned long long>(g_pre_raster_states.again_gs));
    }
    if (stage_manifest_on()) {
        host_log("render: stage manifest: %llu stages from earlier runs, %llu compiled at start (%.1f s), %llu failed and dropped; "
                 "pipeline builds that found one ready %llu; %llu stages recorded in this run",
                 static_cast<unsigned long long>(g_manifest.loaded.load()), static_cast<unsigned long long>(g_manifest.compiled.load()),
                 g_manifest.compile_us.load() / 1e6, static_cast<unsigned long long>(g_manifest.failed.load()),
                 static_cast<unsigned long long>(g_manifest.hits.load()), static_cast<unsigned long long>(g_manifest.recorded.load()));
    }
    std::string names;
    for (const std::string& n : w.not_ready) names += n;
    std::string reasons;
    for (const auto& [why, count] : w.fail_reasons) reasons += " " + why + " x" + std::to_string(count) + ";";
    if (!reasons.empty()) host_log("render: pixel shaders not compiled at creation, by reason:%s", reasons.c_str());
    host_log("render: pixel shaders compiled at creation: %llu queued, %llu done (%llu early-fragment-tests twins beside them), %llu failed, %zu waiting (translate %.1f s, libraries %.1f s); "
             "draws that found the stage ready %llu; draw translations it did not serve: never queued %llu, still queued %llu, being compiled %llu, failed %llu, flat inputs %llu, input map %llu, "
             "early fragment tests (draw with %llu, without %llu), image dims %llu, sampler modes %llu, input enable %llu, resource registers %llu, other %llu, "
             "(of them lifted %llu); no record:%s",
             static_cast<unsigned long long>(w.queued.load()), static_cast<unsigned long long>(w.done.load()),
             static_cast<unsigned long long>(g_eft_twins_built.load()),
             static_cast<unsigned long long>(w.failed.load()), w.jobs.size(), w.translate_us.load() / 1e6, w.library_us.load() / 1e6,
             static_cast<unsigned long long>(w.stage_hits.load()), static_cast<unsigned long long>(w.miss_not_ready),
             static_cast<unsigned long long>(w.miss_queued), static_cast<unsigned long long>(w.miss_compiling),
             static_cast<unsigned long long>(w.miss_failed), static_cast<unsigned long long>(w.miss_flat), static_cast<unsigned long long>(w.miss_input_map), static_cast<unsigned long long>(w.miss_eft_on), static_cast<unsigned long long>(w.miss_eft_off),
             static_cast<unsigned long long>(w.miss_dims), static_cast<unsigned long long>(w.miss_modes), static_cast<unsigned long long>(w.miss_ena),
             static_cast<unsigned long long>(w.miss_rsrc), static_cast<unsigned long long>(w.miss_other), static_cast<unsigned long long>(w.miss_lifted),
             names.c_str());
}

bool link_gfx_pipeline(GfxPipeline& pl, const GfxFixedState& f, const VertexInputPlan* vertex_input) {
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = static_cast<std::uint32_t>(f.color_formats.size());
    ri.pColorAttachmentFormats = f.color_formats.data();
    ri.depthAttachmentFormat = f.depth_format;
    ri.stencilAttachmentFormat = f.stencil_format;
    std::uint64_t formats_key = fnv1a(f.color_formats.data(), f.color_formats.size() * sizeof(VkFormat), 0x84222325cbf29ce4ull);
    const VkFormat ds_formats[2] = {f.depth_format, f.stencil_format};
    formats_key = fnv1a(ds_formats, sizeof(ds_formats), formats_key);
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    const VkPrimitiveTopology topology = pl.tess_hw ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : f.topology;
    std::uint64_t vi_key = fnv1a(&topology, sizeof(topology));
    if (vertex_input) vi_key = fnv1a(&vertex_input->hash, sizeof(vertex_input->hash), vi_key);
    const VkPipeline vi = pipeline_library(kLibVertexInput, vi_key, "vertex input", 0, [&] {
        VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        if (vertex_input) {
            vis.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vertex_input->bindings.size());
            vis.pVertexBindingDescriptions = vertex_input->bindings.data();
            vis.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vertex_input->attributes.size());
            vis.pVertexAttributeDescriptions = vertex_input->attributes.data();
        }
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topology;
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpci.pVertexInputState = &vis;
        gpci.pInputAssemblyState = &ia;
        return create_library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, nullptr);
    });

    TessLibraryStages tess;
    if (pl.tess_hw) {
        tess.vertex = pl.tess_attrs ? pl.tess_ls.module : pl.tess_vs;
        tess.control = pl.tess_tcs;
        tess.control_points = pl.tess_control_points;
    }
    const VkPipeline pr = pre_raster_library(pl.vs, pl.gs, f.polygon_mode, f.depth_clamp, pl.layout, pl.set_layouts, pl.tess_hw ? &tess : nullptr);

    const VkPipeline fs = fragment_library(pl.ps.meta(), pl.ps.module, pl.set_layouts[1], pl.layout);

    std::uint64_t fo_key = fnv1a(f.blends.data(), f.blends.size() * sizeof(VkPipelineColorBlendAttachmentState), formats_key);
    const VkPipeline fo = pipeline_library(kLibOutput, fo_key, "fragment output", 0, [&] {
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = static_cast<std::uint32_t>(f.blends.size());
        cb.pAttachments = f.blends.data();
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = 1;
        dsi.pDynamicStates = dyn;
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpci.pColorBlendState = &cb;
        gpci.pMultisampleState = &ms;
        gpci.pDynamicState = &dsi;
        return create_library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, &ri);
    });

    if (!vi || !pr || !fs || !fo) return false;
    const VkPipeline libs[4] = {vi, pr, fs, fo};
    VkPipelineLibraryCreateInfoKHR lci{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
    lci.libraryCount = 4;
    lci.pLibraries = libs;
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &lci;
    gpci.layout = pl.layout;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = vkCreateGraphicsPipelines(g.device, PipelineCacheUse().cache, 1, &gpci, nullptr, &pl.pipeline) == VK_SUCCESS;
    g_libs.link_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
    g_libs.links.fetch_add(1, std::memory_order_relaxed);
    pl.library = ok;
    if (ok && g_library_relink) {
        // Queued at the first draw, which has the indexed loads' words 3 (queue_library_relink).
        std::copy(libs, libs + 4, pl.relink_libs);
        pl.relink_formats = false;
        if (g_relink_formats && pl.vs_formats_from_params && vertex_input && !pl.tess_hw && !vertex_input->elements.empty()) {
            pl.relink_formats = true;
            for (const gcn::VertexElement& el : vertex_input->elements) {
                if (el.location < 16) pl.relink_vertex_formats[el.location] = gcn::vertex_format_descriptor(el.w3);
            }
        }
        pl.relink_polygon = f.polygon_mode;
        pl.relink_clamp = f.depth_clamp;
        pl.relink_pending = true;
    }
    return ok;
}

// The Vulkan pipeline of a translated and laid-out GfxPipeline, from any draw
// state with its key (the key covers the fixed state and vertex input).
bool create_gfx_pipeline(GfxPipeline& pl, const DrawState& s, const VertexInputPlan* vertex_input, std::size_t fetch_dwords) {
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    auto stage = [&](VkShaderStageFlagBits bit, VkShaderModule m) {
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = bit;
        si.module = m;
        si.pName = "main";
        stages.push_back(si);
    };
    if (pl.tess_hw && (!pl.tess_tcs || !(pl.tess_attrs ? pl.tess_ls.module : pl.tess_vs))) {
        // The generated stages are made in build_gfx_pipeline; a pipeline that
        // reached here without them would hand the driver a null module.
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 4) host_log("render: pipeline %s wants the host tessellator without its stages", pl.name.c_str());
        pl.failed = true;
        return false;
    }
    if (pl.tess_hw) {
        // The host's own tessellator. The control points are
        // already in the buffer the LS compute pass wrote, so the vertex stage
        // carries nothing and the control stage only the hull shader's
        // factors; `pl.vs` holds the domain shader, which is the evaluation
        // stage.
        stage(VK_SHADER_STAGE_VERTEX_BIT, pl.tess_attrs ? pl.tess_ls.module : pl.tess_vs);
        stage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, pl.tess_tcs);
        stage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, pl.vs.module);
    } else {
        stage(VK_SHADER_STAGE_VERTEX_BIT, pl.vs.module);
    }
    if (pl.gs.module) stage(VK_SHADER_STAGE_GEOMETRY_BIT, pl.gs.module);
    if (pl.ps.module) stage(VK_SHADER_STAGE_FRAGMENT_BIT, pl.ps.module);
    // Pixel shaders at wave32: built whole, the fragment stage is given its
    // subgroup size here (linked, in its library: fragment_library).
    const bool linked = use_pipeline_library() && pl.layout;
    const char* const owner = t_library_owner;
    t_library_owner = pl.name.c_str();
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo ps_wave{};
    const bool ps_sized = !linked && pl.ps.module && fragment_stage_wave(pl.ps.meta(), stages.back(), ps_wave);

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (vertex_input) {  // Vulkan vertex input
        vi.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vertex_input->bindings.size());
        vi.pVertexBindingDescriptions = vertex_input->bindings.data();
        vi.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vertex_input->attributes.size());
        vi.pVertexAttributeDescriptions = vertex_input->attributes.data();
    }
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    const GfxFixedState f = fixed_state(s);
    ia.topology = pl.tess_hw ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : f.topology;
    ia.primitiveRestartEnable = VK_FALSE;
    VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    ts.patchControlPoints = std::max<std::uint32_t>(1, pl.tess_control_points);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = f.polygon_mode;
    rs.cullMode = f.cull_mode;
    rs.frontFace = f.front_face;
    rs.lineWidth = 1.0f;
    rs.depthClampEnable = f.depth_clamp ? VK_TRUE : VK_FALSE;
    rs.depthBiasEnable = f.depth_bias ? VK_TRUE : VK_FALSE;
    rs.depthBiasConstantFactor = f.bias_constant;
    rs.depthBiasClamp = f.bias_clamp;
    rs.depthBiasSlopeFactor = f.bias_slope;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = f.depth_test ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = f.depth_write ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = f.depth_compare;
    ds.stencilTestEnable = f.stencil_test ? VK_TRUE : VK_FALSE;
    ds.depthBoundsTestEnable = f.depth_bounds_test ? VK_TRUE : VK_FALSE;
    ds.front = f.front;
    ds.back = f.back;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = static_cast<std::uint32_t>(f.blends.size());
    cb.pAttachments = f.blends.data();
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                  VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                  VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BOUNDS};
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = g.has_depth_bounds ? 7 : 6;
    dsi.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = static_cast<std::uint32_t>(f.color_formats.size());
    ri.pColorAttachmentFormats = f.color_formats.data();
    ri.depthAttachmentFormat = f.depth_format;
    ri.stencilAttachmentFormat = f.stencil_format;
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &ri;
    gpci.stageCount = static_cast<std::uint32_t>(stages.size());
    gpci.pStages = stages.data();
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    if (pl.tess_hw) gpci.pTessellationState = &ts;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dsi;
    gpci.layout = pl.layout ? pl.layout : g.gfx_pipe_layout;
    const auto t0 = std::chrono::steady_clock::now();
    if (g_dump_spirv) host_log("render: creating pipeline %s", pl.name.c_str());
    // The host tessellator's pipelines link from libraries like the rest: built
    // whole, each took ~1.6 ms from a warm cache (50-165 ms cold), 14 of them in
    // the frame after a fight, and left the draw after it to set every dynamic
    // value again.
    bool created = linked ? link_gfx_pipeline(pl, f, vertex_input)
                          : vkCreateGraphicsPipelines(g.device, PipelineCacheUse().cache, 1, &gpci, nullptr, &pl.pipeline) == VK_SUCCESS;
    if (!created && ps_sized) {
        fragment_stage_unsized(stages.back());
        created = vkCreateGraphicsPipelines(g.device, PipelineCacheUse().cache, 1, &gpci, nullptr, &pl.pipeline) == VK_SUCCESS;
        fragment_stage_wave_refused(created);
    }
    t_library_owner = owner;
    if (!created) {
        host_log("render: pipeline %s: vkCreateGraphicsPipelines failed", pl.name.c_str());
        pl.failed = true;
        return false;
    }
    const auto ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    if (!t_pipeline_worker) g_pl_create_us.fetch_add(pl_us_since(t0), std::memory_order_relaxed);
    // Slow driver compiles, with what might make them slow: the stages'
    // SPIR-V sizes and whether this is the page-table fallback or the
    // no-fallback variant.
    if (ms_ >= 30) {
        static std::atomic<int> slow_logs{0};
        if ((slow_logs.load(std::memory_order_relaxed) < 400 && slow_logs.fetch_add(1) < 400)) {
            host_log("render: slow pipeline %s%s on %s: vkCreateGraphicsPipelines %lld ms; SPIR-V words VS %zu, PS %zu%s; VS params %zu, "
                     "PS images %zu, buffers VS %zu PS %zu",
                     pl.name.c_str(), pl.lean ? " (no-fallback)" : "", t_pipeline_worker ? "a worker" : "the command processor",
                     static_cast<long long>(ms_), pl.vs.meta().spirv.size(), pl.ps.meta().spirv.size(), pl.gs.module ? ", with GS" : "",
                     pl.vs.meta().vs_params.size(), pl.ps.meta().images.size(), pl.vs.meta().buffers.size(), pl.ps.meta().buffers.size());
        }
    }
    g.gfx_pipelines.fetch_add(1);
    // BBHOST_LOG_PIPELINES=<name prefix>[,<prefix>...] or `all`: log those
    // pipelines' creation too, past the first 40.
    static const std::vector<std::string> log_names = [] {
        std::vector<std::string> names;
        const char* e = std::getenv("BBHOST_LOG_PIPELINES");
        for (std::string rest = e ? e : ""; !rest.empty();) {
            const std::size_t comma = rest.find(',');
            names.push_back(rest.substr(0, comma));
            rest = comma == std::string::npos ? "" : rest.substr(comma + 1);
        }
        return names;
    }();
    const bool named = std::any_of(log_names.begin(), log_names.end(), [&](const std::string& n) {
        return n == "all" || pl.name.compare(0, n.size(), n) == 0;
    });
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 40 || named) {
        host_log("render: pipeline %s: prim %u, %zu color, depth %d (control %08x, stencil %08x), VS %zu params/%zu images, "
                 "PS %zu images, fetch %zu dwords, vertex input %zu elements in %zu bindings, %lld ms",
                 pl.name.c_str(), s.prim, f.color_formats.size(), s.depth ? s.depth->format : 0, s.depth_control, s.stencil_control,
                 pl.vs.meta().vs_params.size(),
                 pl.vs.meta().images.size(), pl.ps.meta().images.size(), fetch_dwords,
                 vertex_input ? vertex_input->elements.size() : 0, vertex_input ? vertex_input->bindings.size() : 0,
                 static_cast<long long>(ms_));
    }
    return true;
}

// g_gfx by key with the last hits in front (entries are never erased).
GfxPipeline* gfx_lookup(std::uint64_t key) {
    struct Memo {
        std::uint64_t key = 0;
        GfxPipeline* pl = nullptr;
    };
    static Memo memo[64];  // under g.mu
    Memo& m = memo[key & 63];
    if (m.pl && m.key == key) return m.pl;
    auto it = g_gfx.find(key);
    if (it == g_gfx.end()) return nullptr;
    m = {key, &it->second};
    return m.pl;
}

GfxPipeline& gfx_pipeline(const DrawState& s, std::uint64_t key, const std::vector<std::uint32_t>& vs_words,
                          const std::vector<std::uint32_t>& fetch_words, const std::vector<std::uint32_t>& ps_words,
                          const std::string& vs_name, const std::string& ps_name,
                          const std::vector<std::pair<std::uint32_t, bool>>& vs_dims,
                          const std::vector<std::pair<std::uint32_t, bool>>& ps_dims,
                          const std::vector<bool>& vs_sampler_modes, const std::vector<bool>& ps_sampler_modes,
                          bool cb_no_fallback = false, const VertexInputPlan* vertex_input = nullptr, bool create = true,
                          bool layout_only = false) {
    if (GfxPipeline* pl = gfx_lookup(key)) return *pl;
    return build_gfx_pipeline(g_gfx[key], s, vs_words, fetch_words, ps_words, vs_name, ps_name, vs_dims, ps_dims, vs_sampler_modes,
                              ps_sampler_modes, cb_no_fallback, vertex_input, create, layout_only);
}

// Two translations of a stage bind the same resources at the same bindings, so
// descriptor sets written for one fit the other.
bool same_bindings(const gcn::TranslateResult& a, const gcn::TranslateResult& b) {
    if (a.buffers.size() != b.buffers.size() || a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size() ||
        a.vs_params != b.vs_params || a.ps_inputs != b.ps_inputs) {
        return false;
    }
    for (std::size_t i = 0; i < a.buffers.size(); ++i) {
        const gcn::BufferBinding &x = a.buffers[i], &y = b.buffers[i];
        if (x.binding != y.binding || x.max_dw != y.max_dw || x.pointer != y.pointer || x.indexed != y.indexed || x.path.str() != y.path.str()) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.images.size(); ++i) {
        const gcn::ImageBinding &x = a.images[i], &y = b.images[i];
        if (x.binding != y.binding || x.storage != y.storage || x.dim != y.dim || x.arrayed != y.arrayed || x.depth != y.depth ||
            x.cube != y.cube || x.kind != y.kind || x.r128 != y.r128 || x.path.str() != y.path.str()) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.samplers.size(); ++i) {
        if (a.samplers[i].binding != b.samplers[i].binding || a.samplers[i].compare != b.samplers[i].compare ||
            a.samplers[i].path.str() != b.samplers[i].path.str()) {
            return false;
        }
    }
    return true;
}

// BBHOST_ASYNC_PIPELINES (on unless 0): the first draw whose constant buffers
// all bind created its pipeline's no-fallback variant on the command
// processor's thread, and the translation and driver compile stalled the game
// (tens to hundreds of milliseconds with a cold shader cache). A worker builds
// the variant instead, and draws use the fallback pipeline, which renders the
// same, until the command processor hands the variant to its base pipeline.
// Dump and check modes still build in place. BBHOST_DECOMP lifts on the
// workers too: kept in place, its variants cost the command processor 25 s
// more stall time over a cold world load.
const bool g_async_pipelines = [] {
    const char* e = std::getenv("BBHOST_ASYNC_PIPELINES");
    return !(e && e[0] == '0');
}();
// BBHOST_LAYOUT_FROM_LEAN (on unless 0; needs BBHOST_LEAN_FIRST): a new
// pipeline was translated twice on the command processor, the fallback
// variant for its bindings (descriptor sets are written from them) and the
// no-fallback variant it then draws with. The two bind the same resources, so
// the fallback is laid out from its no-fallback stages - which the stage cache
// and the per-shader compiles usually have ready - and translated for itself
// only when a draw has to create it. 227 of 302 translations on the command
// processor in a world walk were fallbacks. Lifted stages are laid out the
// same way: the lift keeps the translation's bindings and is cached under its
// own key (lift_key), so the no-fallback variant takes the stages this lays
// out. They were left out while a lift was not cached, and once the lifter
// became the default that left out every pipeline: 277 fallbacks translated
// on the command processor in a soak, three in four of its translations.
const bool g_layout_from_lean = [] {
    const char* e = std::getenv("BBHOST_LAYOUT_FROM_LEAN");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_fallbacks_deferred{0}, g_fallbacks_translated{0}, g_fallback_binding_mismatch{0};

// BBHOST_LEAN_FIRST (on unless 0): a pipeline's fallback variant, which walks
// the page table for unbound constant buffers, is created only when a draw
// binds it. The first draw whose constant buffers all bind builds the
// no-fallback variant in place instead of waiting on a worker. On a cold
// world load the command processor compiled 166 fallback pipelines, 156 of
// them for pipelines that then built a no-fallback variant as well.
const bool g_lean_first = [] {
    const char* e = std::getenv("BBHOST_LEAN_FIRST");
    return !(e && e[0] == '0');
}();
struct LeanBuildJob {
    GfxPipeline* base = nullptr;     // the pipeline the variant belongs to
    GfxPipeline* variant = nullptr;  // its entry in g_gfx, reserved by the command processor
    DrawState s;                     // copied; its target pointers point at `formats`
    RtImage formats[9];              // only the draw's target formats: colour 0-7, depth
    std::vector<std::uint32_t> vs_words, fetch_words, ps_words;
    std::string vs_name, ps_name;
    std::vector<std::pair<std::uint32_t, bool>> vs_dims, ps_dims;
    std::vector<bool> vs_modes, ps_modes;
    bool has_vertex_input = false;
    VertexInputPlan vertex_input;
};
struct LeanBuilders {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::unique_ptr<LeanBuildJob>> jobs;             // under mu
    std::vector<std::pair<GfxPipeline*, GfxPipeline*>> done;   // (base, variant), under mu
};
std::atomic<bool> g_lean_builds_ready{false};
std::atomic<std::uint64_t> g_lean_builds_queued{0}, g_lean_builds_done{0};

LeanBuilders& lean_builders() {
    static LeanBuilders* const builders = [] {
        auto* w = new LeanBuilders;  // never destroyed: its threads run until exit
        for (int i = 0; i < 2; ++i) {
            std::thread([w] {
                for (;;) {
                    std::unique_ptr<LeanBuildJob> job;
                    {
                        std::unique_lock<std::mutex> lk(w->mu);
                        w->cv.wait(lk, [w] { return !w->jobs.empty(); });
                        job = std::move(w->jobs.front());
                        w->jobs.pop_front();
                    }
                    const LeanBuildJob& j = *job;
                    t_pipeline_worker = true;
                    build_gfx_pipeline(*j.variant, j.s, j.vs_words, j.fetch_words, j.ps_words, j.vs_name, j.ps_name, j.vs_dims, j.ps_dims,
                                       j.vs_modes, j.ps_modes, true, j.has_vertex_input ? &j.vertex_input : nullptr);
                    std::lock_guard<std::mutex> lk(w->mu);
                    w->done.emplace_back(j.base, j.variant);
                    g_lean_builds_ready.store(true, std::memory_order_relaxed);
                    g_lean_builds_done.fetch_add(1, std::memory_order_relaxed);
                }
            }).detach();
        }
        return w;
    }();
    return *builders;
}

// Under g.mu: reserves the variant's entry and hands the build to a worker.
void queue_lean_variant(GfxPipeline& base, std::uint64_t lean_key, const DrawState& s, const std::vector<std::uint32_t>& vs_words,
                        const std::vector<std::uint32_t>& fetch_words, const std::vector<std::uint32_t>& ps_words,
                        const std::string& vs_name, const std::string& ps_name,
                        const std::vector<std::pair<std::uint32_t, bool>>& vs_dims,
                        const std::vector<std::pair<std::uint32_t, bool>>& ps_dims, const std::vector<bool>& vs_modes,
                        const std::vector<bool>& ps_modes, const VertexInputPlan* vertex_input) {
    auto job = std::make_unique<LeanBuildJob>();
    job->base = &base;
    job->variant = &g_gfx[lean_key];
    job->variant->building = true;
    job->s = s;
    for (int t = 0; t < 8; ++t) {
        if (!s.color[t]) continue;
        job->formats[t].format = s.color[t]->format;
        job->s.color[t] = &job->formats[t];
    }
    if (s.depth) {
        job->formats[8].format = s.depth->format;
        job->s.depth = &job->formats[8];
    }
    job->vs_words = vs_words;
    job->fetch_words = fetch_words;
    job->ps_words = ps_words;
    job->vs_name = vs_name;
    job->ps_name = ps_name;
    job->vs_dims = vs_dims;
    job->ps_dims = ps_dims;
    job->vs_modes = vs_modes;
    job->ps_modes = ps_modes;
    if (vertex_input) {
        job->has_vertex_input = true;
        job->vertex_input = *vertex_input;
    }
    base.lean_building = true;
    LeanBuilders& w = lean_builders();
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_back(std::move(job));
    }
    g_lean_builds_queued.fetch_add(1, std::memory_order_relaxed);
    w.cv.notify_one();
}

// Under g.mu: gives finished variants to their base pipelines.
void apply_lean_builds_locked() {
    if (!g_lean_builds_ready.load(std::memory_order_relaxed)) return;
    LeanBuilders& w = lean_builders();
    std::vector<std::pair<GfxPipeline*, GfxPipeline*>> done;
    {
        std::lock_guard<std::mutex> lk(w.mu);
        done.swap(w.done);
        g_lean_builds_ready.store(false, std::memory_order_relaxed);
    }
    for (const auto& [base, variant] : done) {
        variant->building = false;
        base->lean_variant = variant;
        base->lean_building = false;
    }
}

// ---- pass management ------------------------------------------------------------
// The barrier a pass ended between draws still owes, paid now (no pass open).
void pay_pass_barrier_locked() {
    if (g_hazards.owed) {
        if (DrawCmds* c = DrawCmds::open(); !c || !c->pass_barrier()) rec().pass_barrier();
        g_barriers_paid.fetch_add(1, std::memory_order_relaxed);
    }
    g_hazards.owed = false;
    if (!g_hazards.written.empty()) g_hazards.written.clear();  // clear() of an empty set still walks its buckets
    if (!g_hazards.sampled.empty()) g_hazards.sampled.clear();
}

// A pass ended for the next draw's pass: its barrier owed (lazy pass barriers).
void end_pass_for_draw_locked() {
    if (!g_pass.active) return;
    if (!g_lazy_barriers) {
        render_end_pass_locked();
        return;
    }
    g_pass.active = false;
    if (DrawCmds* c = DrawCmds::open(); !c || !c->end_rendering(false)) rec().end_rendering(false);
    for (int t = 0; t < g_pass.ntargets; ++t) g_hazards.written.insert(g_pass.targets[t]);
    g_hazards.owed = true;
    copy_versions_pass_end_locked();
}

// `hazard`: the draw samples a target written since the last barrier (lazy
// pass barriers), so the barrier is paid before it, inside a pass or not.
void begin_pass(const DrawState& s, std::uint64_t key, bool hazard = false) {
    if (g_pass.active && g_pass.key == key && !hazard) return;
    transfer_flush_locked();
    if (hazard && g_pass.active && g_pass.key == key) g_barriers_mid_pass.fetch_add(1, std::memory_order_relaxed);
    end_pass_for_draw_locked();
    if (g_hazards.owed) {
        bool need = hazard;
        for (int t = 0; t < 8 && !need; ++t) {
            if (s.color[t]) need = g_hazards.written.count(s.color[t]->base) || g_hazards.sampled.count(s.color[t]->base);
        }
        if (s.depth && !need) need = g_hazards.written.count(s.depth->base) || g_hazards.sampled.count(s.depth->base);
        if (need) {
            pay_pass_barrier_locked();
        } else {
            g_barriers_skipped.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::vector<VkRenderingAttachmentInfo> colors;
    VkExtent2D extent{0, 0};
    // Attachment N is CB slot N (fixed_state): a gap below the last bound slot
    // is an attachment with no view.
    int last = -1;
    for (int t = 0; t < 8; ++t) {
        if (s.color[t]) last = t;
    }
    for (int t = 0; t <= last; ++t) {
        if (!s.color[t]) {
            colors.push_back(VkRenderingAttachmentInfo{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO});
            continue;
        }
        ensure_initialised(*s.color[t]);
        s.color[t]->fill_last = false;  // drawn into: a fill is no longer its last write
        VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        a.imageView = attachment_view(*s.color[t], s.color_layer[t]);
        a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colors.push_back(a);
        if (!extent.width || s.color[t]->width < extent.width) extent.width = s.color[t]->width;
        if (!extent.height || s.color[t]->height < extent.height) extent.height = s.color[t]->height;
    }
    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo stencil{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (s.depth) {
        ensure_initialised(*s.depth);
        depth.imageView = attachment_view(*s.depth, s.depth_layer);
        depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        stencil = depth;
        if (!extent.width || s.depth->width < extent.width) extent.width = s.depth->width;
        if (!extent.height || s.depth->height < extent.height) extent.height = s.depth->height;
    }
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
    ri.pColorAttachments = colors.data();
    if (s.depth) {
        ri.pDepthAttachment = &depth;
        if (s.depth->format != VK_FORMAT_D32_SFLOAT && s.depth->format != VK_FORMAT_D16_UNORM) ri.pStencilAttachment = &stencil;
    }
    if (g.profile_passes && g_pass_profile_name) {
        profile_begin_locked(g_pass_profile_name);
        g_pass.profiled = true;
    }
    if (DrawCmds* c = DrawCmds::open(); !c || !c->begin_rendering(ri)) vkCmdBeginRendering(g_cmd(), &ri);
    g_pass.active = true;
    g_pass.key = key;
    g_pass.extent = extent;
    g_pass.ntargets = 0;
    for (int t = 0; t < 8; ++t) {
        if (s.color[t]) g_pass.targets[g_pass.ntargets++] = s.color[t]->base;
    }
    if (s.depth) g_pass.targets[g_pass.ntargets++] = s.depth->base;
}

std::uint64_t pass_key(const DrawState& s) {
    std::uint64_t h = 1469598103934665603ull;
    for (int t = 0; t < 8; ++t) {
        const std::uint64_t b = s.color[t] ? s.color[t]->base ^ (static_cast<std::uint64_t>(s.color_layer[t]) << 48) : 0;
        h = fnv1a(&b, 8, h);
    }
    const std::uint64_t d = s.depth ? s.depth->base ^ (static_cast<std::uint64_t>(s.depth_layer) << 48) : 0;
    return fnv1a(&d, 8, h);
}

void clear_depth(RtImage& r, bool depth, bool stencil, float dval, std::uint32_t sval) {
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 12) {
        host_log("render: DB_RENDER_CONTROL clear of 0x%llx: depth=%d (%g) stencil=%d (%u)", static_cast<unsigned long long>(r.base),
                 depth ? 1 : 0, dval, stencil ? 1 : 0, sval);
    }
    render_end_pass_locked();
    ensure_initialised(r);
    VkClearDepthStencilValue v{dval, sval};
    VkImageSubresourceRange range{0, 0, 1, 0, 1};
    if (depth) range.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (stencil && r.format != VK_FORMAT_D32_SFLOAT && r.format != VK_FORMAT_D16_UNORM) range.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!range.aspectMask) return;
    rec().clear_depth_stencil_image(r.image, VK_IMAGE_LAYOUT_GENERAL, v, 1, &range);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    g_clears.fetch_add(1);
}

// ---- resource resolution -----------------------------------------------------
std::uint64_t interp_base(const std::uint32_t* words, bool vsharp) {
    return vsharp ? (static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1] & 0xff) << 32))
                  : (static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1]) << 32));
}

// Walks a ResourcePath with the stage's user SGPRs and guest memory; fills
// `out` with the resource's dwords. Returns false when memory is unmapped.
bool resolve_resource_impl(const gcn::ResourcePath& path, const std::uint32_t* user, int ndw, std::uint32_t* out,
                           std::vector<std::pair<std::uint64_t, std::uint64_t>>* touched = nullptr) {
    if (path.user_sgpr < 0 || path.user_sgpr + 1 >= 16) return false;
    if (path.immediate) {
        if (path.user_sgpr + ndw > 16) return false;
        std::memcpy(out, user + path.user_sgpr, static_cast<std::size_t>(ndw) * 4);
        return true;
    }
    std::uint32_t cur[4] = {user[path.user_sgpr], user[path.user_sgpr + 1], 0, 0};
    if (path.user_sgpr + 3 < 16) {
        cur[2] = user[path.user_sgpr + 2];
        cur[3] = user[path.user_sgpr + 3];
    }
    for (const gcn::ResourceStep& st : path.loads) {
        const std::uint64_t addr = interp_base(cur, st.vsharp) + static_cast<std::uint64_t>(st.offset_dw) * 4;
        if (!hle_kernel_va_mapped(addr, 16)) return false;
        if (touched) touched->push_back({addr, 16});
        read_guest_locked(addr, 16, cur);
    }
    const std::uint64_t addr = interp_base(cur, path.final_vsharp) + static_cast<std::uint64_t>(path.final_offset_dw) * 4;
    if (!hle_kernel_va_mapped(addr, static_cast<std::size_t>(ndw) * 4)) return false;
    if (touched) touched->push_back({addr, static_cast<std::uint64_t>(ndw) * 4});
    // The table was usually dumped by the constant engine a packet ago, into
    // the command stream rather than memory: read it as the draw will see it.
    read_guest_locked(addr, static_cast<std::size_t>(ndw) * 4, out);
    return true;
}

// A GX draw's binding from the records its pending-state objects
// give (GxStageRecords obj_*), mapped through the shader's user-data
// descriptors as compare_gx_bindings does, instead of walking user data
// through the command-buffer tables. False falls back to user data;
// g_gx_resolves counts why.
enum GxResolve { kGxRecord, kGxNoDescriptor, kGxShape, kGxShort, kGxNoObject, kGxResolveKinds };
std::atomic<std::uint64_t> g_gx_resolves[kGxResolveKinds] = {};
std::atomic<std::uint64_t> g_gx_token_resolves[kGxResolveKinds] = {};  // the same, in draws from host-draw tokens
thread_local bool t_token_draw = false;  // host_gpu_draw's current draw came from a host-draw token
// Binding paths token draws resolved through user data (loads or a V# base),
// by path and by where the draw resolved them, for the exit report.
enum ResolveSite { kSiteKey, kSitePrefetch, kSiteBuffers };
thread_local int t_resolve_site = kSiteBuffers;
std::mutex g_token_shape_mu;
std::map<std::pair<std::string, int>, std::uint64_t> g_token_shape_paths;  // (path, site) -> resolves, under g_token_shape_mu

// A stage's user-data descriptors by slot, built once per draw
// (index_gx_stage): a draw resolves a dozen bindings per stage, and each
// scanned the descriptor list for its slot.
struct GxDescIndex {
    const GxStageRecords* st = nullptr;
    std::uint32_t ndesc = 0;
    std::uint8_t of_ud[256];  // descriptor index + 1, 0 for none
};
thread_local GxDescIndex t_desc_index[2];

void index_gx_stage(int k, const GxStageRecords* st) {
    GxDescIndex& ix = t_desc_index[k];
    ix.st = st;
    if (!st) return;
    ix.ndesc = st->ndesc;
    std::memset(ix.of_ud, 0, sizeof(ix.of_ud));
    for (std::uint32_t i = st->ndesc < 64 ? st->ndesc : 64; i-- > 0;) {  // backwards: the first match wins, as before
        ix.of_ud[(st->desc[i] >> 8) & 0xff] = static_cast<std::uint8_t>(i + 1);
    }
}

// A record's first `ndw` dwords. Constant sizes for the widths bindings take:
// a copy of a length known only at run time is a call into libc for every
// binding of every draw (3% of the command processor).
inline void copy_record_dwords(std::uint32_t* out, const std::uint8_t* rec, int ndw) {
    switch (ndw) {
        case 8: std::memcpy(out, rec, 32); break;
        case 4: std::memcpy(out, rec, 16); break;
        case 2: std::memcpy(out, rec, 8); break;
        default: std::memcpy(out, rec, static_cast<std::size_t>(ndw) * 4); break;
    }
}
bool resolve_gx_record(const gcn::ResourcePath& path, const GxStageRecords& st, int ndw, std::uint32_t* out) {
    const auto fail = [](GxResolve why) {
        g_gx_resolves[why].fetch_add(1, std::memory_order_relaxed);
        if (t_token_draw) g_gx_token_resolves[why].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    const GxDescIndex* ix = t_desc_index[0].st == &st && t_desc_index[0].ndesc == st.ndesc   ? &t_desc_index[0]
                            : t_desc_index[1].st == &st && t_desc_index[1].ndesc == st.ndesc ? &t_desc_index[1]
                                                                                              : nullptr;
    const auto find = [&](std::uint32_t ud) -> const std::uint32_t* {
        if (ix) return ud < 256 && ix->of_ud[ud] ? &st.desc[ix->of_ud[ud] - 1] : nullptr;
        for (std::uint32_t i = 0; i < st.ndesc && i < 64; ++i) {
            if (((st.desc[i] >> 8) & 0xff) == ud) return &st.desc[i];
        }
        return nullptr;
    };
    const std::uint32_t* d = path.user_sgpr >= 0 ? find(static_cast<std::uint32_t>(path.user_sgpr)) : nullptr;
    if (!d) return fail(kGxNoDescriptor);
    std::uint32_t type = *d & 0xff, slot = (*d >> 16) & 0xff;
    if (path.immediate) {
        // Tables and blocks: the shader reads the pointer itself.
        if ((type >= 0x8 && type <= 0xb) || type >= 0x15) return fail(kGxShape);
    } else {
        const std::uint32_t off = static_cast<std::uint32_t>(path.final_offset_dw);
        if (!path.loads.empty() || path.final_vsharp) {
            if (t_token_draw) {
                std::lock_guard<std::mutex> lk(g_token_shape_mu);
                const auto key = std::make_pair(path.str(), t_resolve_site);
                if (auto it = g_token_shape_paths.find(key); it != g_token_shape_paths.end()) {
                    ++it->second;
                } else if (g_token_shape_paths.size() < 256) {
                    g_token_shape_paths.emplace(key, 1);
                }
            }
            return fail(kGxShape);
        }
        if ((type == 0x8 || type == 0x9) && off % 8 == 0) {
            slot = off / 8;
        } else if ((type == 0xa || type == 0xb) && off % 4 == 0) {
            slot = off / 4;
        } else if (type == 0x1a) {
            const std::uint32_t* e = find(16 + off);  // extended block: user-data slot 16 + dword offset
            if (!e) return fail(kGxNoDescriptor);
            type = *e & 0xff;
            slot = (*e >> 16) & 0xff;
        } else {
            return fail(kGxShape);
        }
    }
    const std::uint8_t* rec = nullptr;
    std::uint32_t dw = 0;
    switch (type) {
        case 0x0: case 0xd: case 0x1: case 0xe: case 0x8:
            dw = type == 0x0 || type == 0xd ? 4 : 8;
            if (slot < 64 && ((st.obj_tex_set >> slot) & 1)) rec = st.obj_tex + static_cast<std::size_t>(slot) * 32;
            break;
        case 0x2: case 0xf: case 0x3: case 0x10: case 0x9:
            dw = type == 0x2 || type == 0xf ? 4 : 8;
            if (slot < 16 && ((st.obj_smp_set >> slot) & 1)) rec = st.obj_smp + static_cast<std::size_t>(slot) * 32;
            break;
        case 0x4: case 0x11: case 0xa:  // sampler cache: the sampler's first 16 bytes, or the default record
            dw = 4;
            if (slot < 16 && (((st.obj_smp_set | st.obj_smp_default) >> slot) & 1)) {
                rec = st.obj_smp + static_cast<std::size_t>(slot) * 32;
            }
            break;
        case 0x5: case 0x12: case 0xb:
            dw = 4;
            if (slot < 14 && ((st.obj_cb_set >> slot) & 1)) rec = st.obj_cb + static_cast<std::size_t>(slot) * 16;
            break;
        default:
            return fail(kGxShape);
    }
    if (ndw < 0 || static_cast<std::uint32_t>(ndw) > dw) return fail(kGxShort);
    if (!rec) return fail(kGxNoObject);
    copy_record_dwords(out, rec, ndw);
    bump(g_gx_resolves[kGxRecord]);
    if (t_token_draw) bump(g_gx_token_resolves[kGxRecord]);
    return true;
}

// A binding's dwords: from a GX draw's records when it has them and they cover
// the path, otherwise from user data.
bool resolve_binding(const gcn::ResourcePath& path, const std::uint32_t* user, const GxStageRecords* gx, int ndw,
                     std::uint32_t* out) {
    return (gx && resolve_gx_record(path, *gx, ndw, out)) || resolve_resource_impl(path, user, ndw, out);
}

// The same resolution, worked out once per shader.
// Everything resolve_gx_record decides from a binding's path and the stage's
// descriptors - which descriptor, which record array and slot, the record's
// size - is fixed for a shader translation and a descriptor list; only
// whether the slot holds an object, and the record's bytes, change from draw
// to draw. A plan keeps the first part, keyed by the translation and
// GxStageRecords::desc_fp, and a draw checks a bit and copies a record per
// binding: the descriptor lookups and the decode were ~10% of the command
// processor with index_gx_stage. BBHOST_BINDING_PLANS=0 resolves every
// binding as before; =2 does both and counts disagreements.
const int g_binding_plans = [] {
    const char* e = std::getenv("BBHOST_BINDING_PLANS");
    return e ? std::atoi(e) : 1;
}();
struct GxBindingPlan {
    enum Source : std::uint8_t { kFallback, kTex, kSmp, kSmpCache, kCb };
    Source source = kFallback;
    std::uint8_t slot = 0;
    std::uint8_t why = kGxNoDescriptor;  // a fallback's GxResolve
    bool shape_path = false;             // a fallback for loads or a V# base, which token draws log by path
};
struct GxStagePlan {
    const gcn::TranslateResult* meta = nullptr;
    std::uint64_t gen = 0, desc_fp = 0;
    std::uint32_t ndesc = 0;
    std::vector<GxBindingPlan> images, samplers, buffers;
    // Every binding comes from the records (none falls back to user data, and
    // no buffer is a table the shader walks): user data only the shader reads.
    bool all_records = false;
};
std::uint64_t g_binding_plan_gen = 1;  // moves when a pipeline's translation is replaced in place
// Plans live in g_plan_store (never freed, so a pointer stays good) and are
// found through an open-addressed table, with the last few used in front of
// it: a draw asks for four (each stage's key paths and its buffers) and
// consecutive draws mostly repeat them. The descriptors are told apart by
// their 64-bit fingerprint; a map lookup and a memcmp of the list at every
// draw cost most of what the plans saved.
std::vector<std::unique_ptr<GxStagePlan>> g_plan_store;  // under g.mu
struct PlanSlot {
    std::uint64_t key = 0;
    GxStagePlan* plan = nullptr;
};
constexpr std::size_t kPlanSlots = 4096;
PlanSlot g_plan_slots[kPlanSlots];
// In front: the last plan each translation took, direct-mapped by its
// address. A translation nearly always meets one descriptor list - the list
// belongs to the GX shader object - so this answers with two compares.
constexpr std::size_t kPlanFront = 4096;
GxStagePlan* g_plan_front[kPlanFront] = {};
std::uint64_t g_plans_built = 0, g_plans_used = 0, g_plans_disagreed = 0, g_plans_evicted = 0;
// GX stages whose user data was not built because nothing would
// read it, and those built (draw_impl). BBHOST_SKIP_USER_DATA=0 builds all.
std::uint64_t g_ud_skipped = 0, g_ud_built = 0, g_ud_partial = 0;
const bool g_skip_user_data = [] {
    const char* e = std::getenv("BBHOST_SKIP_USER_DATA");
    return !(e && e[0] == '0');
}();

void note_token_shape(const gcn::ResourcePath& path) {
    std::lock_guard<std::mutex> lk(g_token_shape_mu);
    const auto key = std::make_pair(path.str(), t_resolve_site);
    if (auto it = g_token_shape_paths.find(key); it != g_token_shape_paths.end()) {
        ++it->second;
    } else if (g_token_shape_paths.size() < 256) {
        g_token_shape_paths.emplace(key, 1);
    }
}

// resolve_gx_record's decisions for one binding, without a draw's objects.
GxBindingPlan plan_binding(const gcn::ResourcePath& path, const GxStageRecords& st, int ndw) {
    GxBindingPlan p;
    const auto find = [&](std::uint32_t ud) -> const std::uint32_t* {
        for (std::uint32_t i = 0; i < st.ndesc && i < 64; ++i) {
            if (((st.desc[i] >> 8) & 0xff) == ud) return &st.desc[i];
        }
        return nullptr;
    };
    const std::uint32_t* d = path.user_sgpr >= 0 ? find(static_cast<std::uint32_t>(path.user_sgpr)) : nullptr;
    if (!d) return p;
    std::uint32_t type = *d & 0xff, slot = (*d >> 16) & 0xff;
    p.why = kGxShape;
    if (path.immediate) {
        if ((type >= 0x8 && type <= 0xb) || type >= 0x15) return p;  // tables and blocks: the shader reads the pointer itself
    } else {
        const std::uint32_t off = static_cast<std::uint32_t>(path.final_offset_dw);
        if (!path.loads.empty() || path.final_vsharp) {
            p.shape_path = true;
            return p;
        }
        if ((type == 0x8 || type == 0x9) && off % 8 == 0) {
            slot = off / 8;
        } else if ((type == 0xa || type == 0xb) && off % 4 == 0) {
            slot = off / 4;
        } else if (type == 0x1a) {
            const std::uint32_t* e = find(16 + off);  // extended block: user-data slot 16 + dword offset
            if (!e) {
                p.why = kGxNoDescriptor;
                return p;
            }
            type = *e & 0xff;
            slot = (*e >> 16) & 0xff;
        } else {
            return p;
        }
    }
    GxBindingPlan::Source source;
    std::uint32_t dw = 4, slots = 0;
    switch (type) {
        case 0x0: case 0xd: case 0x1: case 0xe: case 0x8:
            source = GxBindingPlan::kTex;
            dw = type == 0x0 || type == 0xd ? 4 : 8;
            slots = 64;
            break;
        case 0x2: case 0xf: case 0x3: case 0x10: case 0x9:
            source = GxBindingPlan::kSmp;
            dw = type == 0x2 || type == 0xf ? 4 : 8;
            slots = 16;
            break;
        case 0x4: case 0x11: case 0xa:
            source = GxBindingPlan::kSmpCache;
            slots = 16;
            break;
        case 0x5: case 0x12: case 0xb:
            source = GxBindingPlan::kCb;
            slots = 14;
            break;
        default:
            return p;
    }
    if (ndw < 0 || static_cast<std::uint32_t>(ndw) > dw) {
        p.why = kGxShort;
        return p;
    }
    if (slot >= slots) {
        p.why = kGxNoObject;
        return p;
    }
    p.source = source;
    p.slot = static_cast<std::uint8_t>(slot);
    p.why = kGxRecord;
    return p;
}

// A draw's half: the slot's object and its record, counted as resolve_gx_record counts.
bool resolve_planned(const GxBindingPlan& p, const gcn::ResourcePath& path, const GxStageRecords& st, int ndw, std::uint32_t* out) {
    const std::uint8_t* rec = nullptr;
    GxResolve why = kGxNoObject;
    switch (p.source) {
        case GxBindingPlan::kTex:
            if ((st.obj_tex_set >> p.slot) & 1) rec = st.obj_tex + static_cast<std::size_t>(p.slot) * 32;
            break;
        case GxBindingPlan::kSmp:
            if ((st.obj_smp_set >> p.slot) & 1) rec = st.obj_smp + static_cast<std::size_t>(p.slot) * 32;
            break;
        case GxBindingPlan::kSmpCache:  // the sampler's first 16 bytes, or the default record
            if (((st.obj_smp_set | st.obj_smp_default) >> p.slot) & 1) rec = st.obj_smp + static_cast<std::size_t>(p.slot) * 32;
            break;
        case GxBindingPlan::kCb:
            if ((st.obj_cb_set >> p.slot) & 1) rec = st.obj_cb + static_cast<std::size_t>(p.slot) * 16;
            break;
        case GxBindingPlan::kFallback:
            why = static_cast<GxResolve>(p.why);
            if (p.shape_path && t_token_draw) note_token_shape(path);
            break;
    }
    if (!rec) {
        bump(g_gx_resolves[why]);
        if (t_token_draw) bump(g_gx_token_resolves[why]);
        return false;
    }
    copy_record_dwords(out, rec, ndw);
    bump(g_gx_resolves[kGxRecord]);
    if (t_token_draw) bump(g_gx_token_resolves[kGxRecord]);
    return true;
}

// The plan for a translation's bindings over a stage's descriptors, built at
// its first draw. A hash collision only rebuilds: the entry checks what it
// was built from.
const GxStagePlan& stage_plan(const gcn::TranslateResult& meta, const GxStageRecords& st) {
    const auto fits = [&](const GxStagePlan* p) {
        return p && p->meta == &meta && p->desc_fp == st.desc_fp && p->ndesc == st.ndesc && p->gen == g_binding_plan_gen;
    };
    const std::uintptr_t m = reinterpret_cast<std::uintptr_t>(&meta);
    GxStagePlan*& front = g_plan_front[(m >> 6) & (kPlanFront - 1)];
    if (fits(front)) {
        ++g_plans_used;
        return *front;
    }
    const std::uint64_t key = ((st.desc_fp ^ (m * 0x9e3779b97f4a7c15ull)) + st.ndesc) | 1;
    std::size_t i = static_cast<std::size_t>(key >> 20) & (kPlanSlots - 1);
    PlanSlot* free_slot = nullptr;
    GxStagePlan* plan = nullptr;
    for (int probe = 0; probe < 8; ++probe, i = (i + 1) & (kPlanSlots - 1)) {
        PlanSlot& s = g_plan_slots[i];
        if (s.key == key && fits(s.plan)) {
            plan = s.plan;
            break;
        }
        if (!free_slot && (s.key == 0 || s.plan->gen != g_binding_plan_gen)) free_slot = &s;
    }
    if (plan) {
        ++g_plans_used;
    } else {
        g_plan_store.push_back(std::make_unique<GxStagePlan>());
        plan = g_plan_store.back().get();
        plan->meta = &meta;
        plan->gen = g_binding_plan_gen;
        plan->desc_fp = st.desc_fp;
        plan->ndesc = st.ndesc;
        for (const gcn::ImageBinding& b : meta.images) plan->images.push_back(plan_binding(b.path, st, b.r128 ? 4 : 8));
        for (const gcn::SamplerBinding& b : meta.samplers) plan->samplers.push_back(plan_binding(b.path, st, 4));
        for (const gcn::BufferBinding& b : meta.buffers) plan->buffers.push_back(plan_binding(b.path, st, b.indexed ? 4 : 2));
        plan->all_records = true;
        for (const GxBindingPlan& b : plan->images) plan->all_records = plan->all_records && b.source != GxBindingPlan::kFallback;
        for (const GxBindingPlan& b : plan->samplers) plan->all_records = plan->all_records && b.source != GxBindingPlan::kFallback;
        for (std::size_t k = 0; k < meta.buffers.size(); ++k) {
            plan->all_records = plan->all_records && !meta.buffers[k].pointer && plan->buffers[k].source != GxBindingPlan::kFallback;
        }
        ++g_plans_built;
        if (!free_slot) {
            free_slot = &g_plan_slots[static_cast<std::size_t>(key >> 20) & (kPlanSlots - 1)];
            ++g_plans_evicted;
        }
        free_slot->key = key;
        free_slot->plan = plan;
    }
    front = plan;
    return *plan;
}

// A binding's dwords through its plan (null: as resolve_binding does).
bool resolve_via_plan(const GxBindingPlan* p, const gcn::ResourcePath& path, const std::uint32_t* user, const GxStageRecords* gx,
                      int ndw, std::uint32_t* out) {
    if (!p) return resolve_binding(path, user, gx, ndw, out);
    const bool ok = resolve_planned(*p, path, *gx, ndw, out) || resolve_resource_impl(path, user, ndw, out);
    if (g_binding_plans == 2) {
        std::uint32_t check[8] = {};
        const bool want = resolve_binding(path, user, gx, ndw, check);
        if (want != ok || (ok && std::memcmp(check, out, static_cast<std::size_t>(ndw) * 4) != 0)) {
            if (g_plans_disagreed++ < 8) {
                host_log("render: binding plan disagreed on %s (%d dwords): plan %s %08x %08x, resolve %s %08x %08x", path.str().c_str(),
                         ndw, ok ? "resolved" : "unresolved", out[0], out[1], want ? "resolved" : "unresolved", check[0], check[1]);
            }
        }
    }
    return ok;
}

struct TsharpInfo {
    std::uint64_t base;
    std::uint32_t dfmt, nfmt, width, height, depth, pitch, type, tiling, base_level, last_level, dst_sel;
};
TsharpInfo decode_tsharp(const std::uint32_t* w) {
    TsharpInfo t{};
    t.base = tsharp_base(w);
    t.dfmt = (w[1] >> 20) & 0x3f;
    t.nfmt = (w[1] >> 26) & 0xf;
    t.width = (w[2] & 0x3fff) + 1;
    t.height = ((w[2] >> 14) & 0x3fff) + 1;
    t.dst_sel = w[3] & 0xfff;
    t.base_level = (w[3] >> 12) & 0xf;
    t.last_level = (w[3] >> 16) & 0xf;
    t.tiling = (w[3] >> 20) & 0x1f;
    t.type = (w[3] >> 28) & 0xf;
    t.depth = (w[4] & 0x1fff) + 1;
    t.pitch = ((w[4] >> 13) & 0x3fff) + 1;
    return t;
}

std::map<std::string, std::uint32_t> g_tsharp_survey;
void survey_images(const gcn::TranslateResult& meta, const std::uint32_t* user, const char* stage) {
    for (const gcn::ImageBinding& b : meta.images) {
        std::uint32_t w[8] = {};
        if (!resolve_resource_impl(b.path, user, 8, w)) {
            const std::string key = std::string(stage) + " unresolved " + b.path.str();
            if (g_tsharp_survey[key]++ == 0) host_log("tsharp: %s", key.c_str());
            continue;
        }
        const TsharpInfo t = decode_tsharp(w);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "%s type=%u dfmt=%u nfmt=%u %ux%ux%u pitch=%u tiling=%u levels=%u..%u dst_sel=%03x",
                      stage, t.type, t.dfmt, t.nfmt, t.width, t.height, t.depth, t.pitch, t.tiling, t.base_level, t.last_level,
                      t.dst_sel);
        if (g_tsharp_survey[buf]++ == 0) host_log("tsharp: %s base=0x%llx", buf, static_cast<unsigned long long>(t.base));
    }
}

std::uint64_t g_display_va_for_dump = 0;
long g_flip_counter = 0;

}  // namespace

void note_shader_created(int stage, const std::uint8_t* container, std::size_t size, std::uint64_t flip) {
    record_shader_created(stage, container, size, flip);
}

// draw_impl's DrawCmds while it resolves a draw (under g.mu).
DrawCmds* g_draw_cmds = nullptr;

DrawCmds* draw_packet_early_locked() {
    if (!g_draw_cmds || g.profile || (g.has_checkpoints && !g.cmd_buffer_marker)) return nullptr;
    g_draw_cmds->start(true);
    return g_draw_cmds->deferred() ? g_draw_cmds : nullptr;
}

void render_end_pass_locked() {
    if (g_pass.active) {
        g_pass.active = false;
        // Into the open draw's packet when it can take it (recorder.cpp).
        if (DrawCmds* c = DrawCmds::open(); !c || !c->end_rendering()) rec().end_rendering(true);
        if (g_pass.profiled) {
            profile_end_locked();
            g_pass.profiled = false;
        }
        g_hazards.owed = false;  // this barrier covers every pass before it too
        if (!g_hazards.written.empty()) g_hazards.written.clear();
        if (!g_hazards.sampled.empty()) g_hazards.sampled.clear();
        copy_versions_pass_end_locked();  // the copy-backs this pass's draws read from their copies, into place
    } else if (g_hazards.owed) {
        pay_pass_barrier_locked();
    }
}

void record_end_rendering(VkCommandBuffer cmd, bool barrier) {
    vkCmdEndRendering(cmd);
    if (barrier) record_pass_barrier(cmd);
}

void record_pass_barrier(VkCommandBuffer cmd) {
    // Attachment writes must be visible to later shader reads / blits.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

namespace {

// Copies the layers `from` has into the same layers of `to`: a layered target
// that grew keeps the slices already drawn.
void copy_layers_locked(const RtImage& from, RtImage& to) {
    begin_recording_locked();
    render_end_pass_locked();
    ensure_initialised(to);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const std::uint32_t layers = std::min(from.layers, to.layers);
    VkImageCopy region{};
    region.srcSubresource = {aspect_of(from), 0, 0, layers};
    region.dstSubresource = {aspect_of(to), 0, 0, layers};
    region.extent = {std::min(from.width, to.width), std::min(from.height, to.height), 1};
    rec().copy_image(from.image, VK_IMAGE_LAYOUT_GENERAL, to.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

}  // namespace

std::size_t format_bytes_per_pixel(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
            return 16;
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8_SNORM:
            return 1;
        case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_D16_UNORM:
            return 2;
        default:
            return 4;
    }
}

std::size_t rt_bytes_per_pixel(const RtImage& r) { return format_bytes_per_pixel(r.format); }

// Recognised fill shader: the whole image is this constant. The V# byte count
// is the tiled allocation, often a few rows larger (or a 1024² slice) than
// width*height*bpp of the unpadded image, so there is no size check here.
void clear_image_locked(RtImage& r, const float rgba[4], std::uint32_t first_layer = 0, std::uint32_t layer_count = ~0u) {
    if (r.base == g_order_target && hle_video_flip_count() >= 2000 && g_order_logs.load() < 120) {
        g_order_logs.fetch_add(1);
        host_log("order: CLEAR 0x%llx rgba=%.3f %.3f %.3f %.3f", static_cast<unsigned long long>(r.base), rgba[0], rgba[1], rgba[2],
                 rgba[3]);
    }
    begin_recording_locked();
    render_end_pass_locked();
    if (!r.initialised) {
        image_barrier(g_cmd(), r.image, aspect_of(r), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, r.layers);
        r.initialised = true;
    }
    first_layer = std::min(first_layer, r.layers - 1);
    VkImageSubresourceRange range{aspect_of(r), 0, 1, first_layer, std::min(layer_count, r.layers - first_layer)};
    if (r.depth) {
        VkClearDepthStencilValue v{};
        v.depth = rgba[0];
        v.stencil = 0;
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 12) {
            host_log("render: fill clear of depth target 0x%llx to %g", static_cast<unsigned long long>(r.base), v.depth);
        }
        rec().clear_depth_stencil_image(r.image, VK_IMAGE_LAYOUT_GENERAL, v, 1, &range);
    } else {
        VkClearColorValue v{};
        const bool bgra = r.format == VK_FORMAT_B8G8R8A8_UNORM || r.format == VK_FORMAT_B8G8R8A8_SRGB;
        v.float32[0] = bgra ? rgba[2] : rgba[0];
        v.float32[1] = rgba[1];
        v.float32[2] = bgra ? rgba[0] : rgba[2];
        v.float32[3] = rgba[3];
        rec().clear_color_image(r.image, VK_IMAGE_LAYOUT_GENERAL, v, 1, &range);
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    g_clears.fetch_add(1);
}

std::size_t rt_size_bytes(const RtImage& r) {
    if (r.layers > 1 && r.slice_bytes) return static_cast<std::size_t>(r.slice_bytes) * r.layers;
    const std::size_t bpp = r.depth ? (r.format == VK_FORMAT_D16_UNORM ? 2 : 4) : rt_bytes_per_pixel(r);
    return static_cast<std::size_t>(r.width) * r.height * bpp;
}

// The layers a fill of guest memory [va, va + bytes) covers: a layered target's
// slices follow each other from its base (a cube face per fill).
void fill_layers(const RtImage& r, std::uint64_t va, std::size_t bytes, std::uint32_t& first, std::uint32_t& count) {
    first = 0;
    count = r.layers;
    if (r.layers <= 1 || !r.slice_bytes || va < r.base) return;
    const std::uint64_t slice = (va - r.base) / r.slice_bytes;
    if (slice >= r.layers) {  // a slice the target does not have yet: the fill waits for it
        count = 0;
        return;
    }
    first = static_cast<std::uint32_t>(slice);
    const std::uint64_t n = std::max<std::uint64_t>(bytes / r.slice_bytes, 1);
    count = static_cast<std::uint32_t>(std::min<std::uint64_t>(n, r.layers - first));
}

// A fill that starts inside a single-layer target, past its base, is not a
// clear of that target: a tiled target's memory cannot be cleared by a linear
// fill that starts mid-image, so the game is clearing memory it is about to
// reuse for something new. The blood layers do exactly this: a new 256x256
// sRGB target is placed 0x28000 bytes into the old one, cleared, then drawn.
// Treating it as the old target's clear wiped the old image and left the new
// one as the image heap left it - a rat's blood layer came out as rainbow
// noise (2026-10-01, F12 flip 16124). BBHOST_FILL_INSIDE=0: as before.
const bool g_fill_inside = [] {
    const char* e = std::getenv("BBHOST_FILL_INSIDE");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_fills_inside{0};
bool fill_starts_inside(const RtImage& r, std::uint64_t va) {
    return g_fill_inside && r.layers <= 1 && va > r.base;
}

bool ranges_overlap(std::uint64_t a, std::size_t na, std::uint64_t b, std::size_t nb) {
    if (!na) na = 1;
    if (!nb) nb = 1;
    return a < b + nb && b < a + na;
}

RtImage* pick_rt(std::uint64_t base) {
    auto it = g_rts.find(base);
    if (it != g_rts.end()) return &it->second;
    auto st = g_snapshots.find(base);
    return st == g_snapshots.end() ? nullptr : &st->second;
}

// A drawn colour or depth target overlapping [va, va + bytes), for the
// stale-read check (textures.cpp): its image is the only copy of what was
// drawn there, so a texture uploaded from that memory reads something else.
const RtImage* rt_overlapping_locked(std::uint64_t va, std::size_t bytes) {
    const RtImage* hit = nullptr;
    auto it = g_rts.lower_bound(va + bytes);
    while (it != g_rts.begin()) {
        --it;
        const RtImage& r = it->second;
        if (r.base + g_max_rt_bytes <= va) break;
        if (r.initialised && ranges_overlap(va, bytes, r.base, rt_size_bytes(r))) hit = &r;
    }
    return hit;
}

RtImage* find_rt_for_fill_walk(std::uint64_t va, std::size_t bytes) {
    if (RtImage* r = pick_rt(va)) return r;
    if (va && va < (1ull << 40) && (va << 8) > va) {
        if (RtImage* r = pick_rt(va << 8)) return r;
    }
    if ((va & 0xff) == 0 && va > 0xff) {
        if (RtImage* r = pick_rt(va >> 8)) return r;
    }
    // The lowest-based target overlapping the fill, else the lowest snapshot
    // (no exact base is left: pick_rt had those). Only targets starting below
    // the fill's end and within the largest target's size of its start can
    // overlap it; walking all of them was ~1.5% of the command processor.
    RtImage* hit = nullptr;
    auto scan = [&](std::map<std::uint64_t, RtImage>& m) {
        auto it = m.lower_bound(va + bytes);
        while (it != m.begin()) {
            --it;
            RtImage& r = it->second;
            if (r.base + g_max_rt_bytes <= va) break;
            if (ranges_overlap(va, bytes, r.base, rt_size_bytes(r))) hit = &r;  // walking down: the last one is the lowest
        }
    };
    scan(g_rts);
    if (!hit) scan(g_snapshots);
    return hit;
}

// The game fills the same memory every frame, so the answer for (va, bytes)
// is kept while no target or snapshot comes or goes (g_rt_gen: every insert
// and erase of either map bumps it). The walk down the targets was ~1% of a
// Steam Deck's command processor. BBHOST_FILL_MEMO=0: walked every time.
RtImage* find_rt_for_fill(std::uint64_t va, std::size_t bytes) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_FILL_MEMO");
        return !(e && e[0] == '0');
    }();
    if (!on) return find_rt_for_fill_walk(va, bytes);
    struct Memo {
        std::uint64_t va = 0, bytes = 0, gen = 0;
        RtImage* r = nullptr;
    };
    static Memo memo[256];
    Memo& m = memo[(va >> 8 ^ va >> 16 ^ bytes) & 255];
    if (m.gen == g_rt_gen && m.va == va && m.bytes == bytes) return m.r;
    RtImage* r = find_rt_for_fill_walk(va, bytes);
    m = Memo{va, bytes, g_rt_gen, r};
    return r;
}

// DMA fill: only when the bytes cover the image.
bool clear_target_locked(std::uint64_t va, std::size_t bytes, const float rgba[4]) {
    auto it = g_rts.find(va);
    if (it == g_rts.end()) return false;
    if (bytes < rt_size_bytes(it->second)) return false;
    clear_image_locked(it->second, rgba);
    note_fill_last(it->second, va, bytes, rgba);
    return true;
}

// The game clears targets with CP DMA fills of their memory; the image is
// the target's real storage here, so clear that instead.
bool render_clear_by_fill_locked(std::uint64_t va, std::size_t bytes, std::uint32_t value) {
    auto it = g_rts.find(va);
    if (it == g_rts.end()) return false;
    RtImage& r = it->second;
    float rgba[4] = {};
    const bool bgra = r.format == VK_FORMAT_B8G8R8A8_UNORM || r.format == VK_FORMAT_B8G8R8A8_SRGB;
    if (r.depth) {
        float f;
        std::memcpy(&f, &value, 4);
        rgba[0] = r.format == VK_FORMAT_D32_SFLOAT || r.format == VK_FORMAT_D32_SFLOAT_S8_UINT ? f : static_cast<float>(value & 0xffffff) / 16777215.0f;
    } else if (value != 0 && rt_bytes_per_pixel(r) == 4 &&
               (r.format == VK_FORMAT_R8G8B8A8_UNORM || r.format == VK_FORMAT_R8G8B8A8_SRGB || bgra)) {
        // Unpacked as the memory layout the shader would see (RGBA order).
        rgba[0] = (value & 0xff) / 255.0f;
        rgba[1] = ((value >> 8) & 0xff) / 255.0f;
        rgba[2] = ((value >> 16) & 0xff) / 255.0f;
        rgba[3] = ((value >> 24) & 0xff) / 255.0f;
        if (bgra) std::swap(rgba[0], rgba[2]);  // clear_target_locked swaps back for BGRA images
    } else if (value != 0 && r.format == VK_FORMAT_R32_SFLOAT) {
        std::memcpy(&rgba[0], &value, 4);
    } else if (value != 0) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 6) {
            host_log("render: clear of target 0x%llx (format %d) with 0x%08x approximated as zero", static_cast<unsigned long long>(va),
                     r.format, value);
        }
    }
    return clear_target_locked(va, bytes, rgba);
}

bool render_clear_target_locked(std::uint64_t va, std::size_t bytes, const float rgba[4]) {
    return clear_target_locked(va, bytes, rgba);
}


bool render_htile_fill_locked(std::uint64_t va, std::size_t bytes, std::uint32_t word, std::uint32_t depth_clear, std::uint32_t stencil_clear) {
    bool any = false;
    if (!g_htile_bases.count(va)) return false;  // no depth target's HTILE was ever here
    for (auto& kv : g_rts) {
        RtImage& r = kv.second;
        if (!r.depth || !r.htile || r.htile != va) continue;
        // HTILE: 4 bytes per 8x8 tile.
        if (bytes < static_cast<std::size_t>((r.width / 8) * (r.height / 8)) * 4) continue;
        r.htile_clear_pending = true;
        const float depth = htile_word_depth(word);
        std::memcpy(&r.htile_clear_depth, &depth, 4);
        r.htile_clear_stencil = stencil_clear;
        if (r.htile_clear_depth != depth_clear) {
            if (g_htile_fill_register_lag.fetch_add(1, std::memory_order_relaxed) < 8) {
                host_log("render: HTILE fill 0x%llx at flip %llu clears to %g (word %08x) while DB_DEPTH_CLEAR still holds %08x",
                         static_cast<unsigned long long>(va), static_cast<unsigned long long>(hle_video_flip_count()), depth, word, depth_clear);
            }
        }
        any = true;
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("render: HTILE fill 0x%llx (%zu bytes) at flip %llu: depth target 0x%llx will clear at its next draw to %g / %02x",
                     static_cast<unsigned long long>(va), bytes, static_cast<unsigned long long>(hle_video_flip_count()),
                     static_cast<unsigned long long>(r.base), depth, stencil_clear & 0xff);
        }
    }
    return any;
}

void apply_pending_clear(RtImage& r) {
    if (!g_pending_htile.empty()) {
        if (r.htile) take_pending_htile(r.htile, r);
        if (r.depth) take_pending_htile(r.base, r);
    }
    // Runs on every rt_image() hit, i.e. for each bound target of every draw,
    // and fills that never meet a target stay pending: scan only the fills
    // that can overlap this target (or start at its base).
    if (g_pending_clears.empty()) return;
    // Nothing came or went since this target last looked: the same answer
    // (~1% of a Steam Deck's command processor was these scans finding
    // nothing). BBHOST_PENDING_SEEN=0: scanned every time.
    static const bool seen_on = [] {
        const char* e = std::getenv("BBHOST_PENDING_SEEN");
        return !(e && e[0] == '0');
    }();
    if (seen_on && r.pending_seen == g_pending_gen) return;
    const std::size_t sz = rt_size_bytes(r);
    const std::uint64_t scan_end = r.base + std::max<std::size_t>(sz, 1);
    ++g_pending_scans;
    for (auto it = g_pending_clears.lower_bound(r.base > g_pending_clear_max ? r.base - g_pending_clear_max : 0);
         it != g_pending_clears.end() && it->first < scan_end;) {
        ++g_pending_scan_steps;
        if (!std::isfinite(it->second.rgba[0])) {
            ++it;
            continue;
        }
        if (!ranges_overlap(it->first, it->second.bytes, r.base, sz) && it->first != r.base) {
            ++it;
            continue;
        }
        if (fill_starts_inside(r, it->first)) {  // waits for the target placed at its start
            ++it;
            continue;
        }
        std::uint32_t first = 0, count = 0;
        fill_layers(r, it->first, it->second.bytes, first, count);
        if (!count) {
            ++it;
            continue;
        }
        clear_image_locked(r, it->second.rgba, first, count);
        note_fill_last(r, it->first, it->second.bytes, it->second.rgba);
        g_fill_applied.fetch_add(1);
        tex_event(r.base, rt_size_bytes(r), "pending fill 0x%llx 0x%zx bytes applied to target 0x%llx %ux%u format %d",
                  static_cast<unsigned long long>(it->first), it->second.bytes, static_cast<unsigned long long>(r.base), r.width, r.height,
                  static_cast<int>(r.format));
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 24) {
            host_log("render: applied pending fill 0x%llx %zu bytes -> target 0x%llx %ux%u",
                     static_cast<unsigned long long>(it->first), it->second.bytes, static_cast<unsigned long long>(r.base), r.width,
                     r.height);
        }
        it = g_pending_clears.erase(it);
        ++g_pending_gen;
    }
    r.pending_seen = g_pending_gen;
}

// BBHOST_FILL_TEXTURES=0 (checks): a fill with no target leaves the textures
// under it as they were uploaded.
const bool g_fill_textures = [] {
    const char* e = std::getenv("BBHOST_FILL_TEXTURES");
    return !(e && e[0] == '0');
}();

bool render_handle_fill_locked(std::uint64_t va, std::size_t bytes, const float rgba[4], std::uint32_t depth_clear,
                               std::uint32_t stencil_clear) {
    RtImage* r = find_rt_for_fill(va, bytes);
    // Found by overlap, starting inside it (not an exact or encoded base):
    // the fill belongs to whatever is placed at `va` next, so it waits there.
    if (r && r->base != va && r->base != (va << 8) && r->base != (va >> 8) && fill_starts_inside(*r, va)) {
        if (g_fills_inside.fetch_add(1, std::memory_order_relaxed) < 16) {
            host_log("render: fill 0x%llx %zu bytes starts inside target 0x%llx %ux%u: pending for what is placed there",
                     static_cast<unsigned long long>(va), bytes, static_cast<unsigned long long>(r->base), r->width, r->height);
        }
        r = nullptr;
    }
    if (r) {
        if (r->depth && !std::isfinite(rgba[0])) {
            std::uint32_t word = 0;
            std::memcpy(&word, &rgba[0], 4);
            r->htile_clear_pending = true;
            const float depth = htile_word_depth(word);
            std::memcpy(&r->htile_clear_depth, &depth, 4);
            r->htile_clear_stencil = stencil_clear;
            return true;
        }
        std::uint32_t first = 0, count = 0;
        fill_layers(*r, va, bytes, first, count);
        if (count) {
            clear_image_locked(*r, rgba, first, count);
            note_fill_last(*r, va, bytes, rgba);
            g_fill_applied.fetch_add(1);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 24 && r->base != va) {
                host_log("render: fill 0x%llx %zu bytes cleared overlapping target 0x%llx %ux%u layers %u+%u",
                         static_cast<unsigned long long>(va), bytes, static_cast<unsigned long long>(r->base), r->width, r->height, first,
                         count);
            }
            return true;
        }
    }
    std::uint32_t word = 0;
    std::memcpy(&word, &rgba[0], 4);
    if (render_htile_fill_locked(va, bytes, word, depth_clear, stencil_clear)) return true;
    // No target: a texture under it takes the fill (textures_fill_surfaces_locked).
    // It still waits for a target placed there later, as before.
    if (g_fill_textures) textures_fill_surfaces_locked(va, bytes, rgba);
    PendingFill pc{};
    std::memcpy(pc.rgba, rgba, sizeof(pc.rgba));
    pc.bytes = bytes;
    g_pending_clears[va] = pc;
    ++g_pending_gen;
    g_pending_clear_max = std::max(g_pending_clear_max, pc.bytes);
    g_pending_htile[va] = PendingHtile{word, stencil_clear};
    g_fill_pending.fetch_add(1);
    tex_event(va, bytes, "fill 0x%llx 0x%zx bytes with %g %g %g %g pending (no image yet)", static_cast<unsigned long long>(va), bytes,
              rgba[0], rgba[1], rgba[2], rgba[3]);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 24) {
        host_log("render: fill of 0x%llx %zu bytes pending (no image yet)", static_cast<unsigned long long>(va), bytes);
    }
    return true;
}

std::string render_fill_stats() {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "pending=%llu applied=%llu starting-inside-a-target=%llu textures=%llu (%llu not clearable); pending now %zu (the largest "
                  "%zu bytes), visited %llu times in %llu scans",
                  static_cast<unsigned long long>(g_fill_pending.load()),
                  static_cast<unsigned long long>(g_fill_applied.load()),
                  static_cast<unsigned long long>(g_fills_inside.load()),
                  static_cast<unsigned long long>(g_fill_surfaces.load()),
                  static_cast<unsigned long long>(g_fill_surfaces_skipped.load()), g_pending_clears.size(), g_pending_clear_max,
                  static_cast<unsigned long long>(g_pending_scan_steps), static_cast<unsigned long long>(g_pending_scans));
    return buf;
}

void render_report() {
    host_log("%s", gx_state_report().c_str());
    if (const std::string b = bindless_report(); !b.empty()) host_log("render: %s", b.c_str());
    if (g_binding_plans) {
        host_log("render: GX stages whose user data was not built (the no-fallback variant touches none, every binding from the "
                 "records) %llu, built %llu (%llu of them only what that variant reads)",
                 static_cast<unsigned long long>(g_ud_skipped), static_cast<unsigned long long>(g_ud_built),
                 static_cast<unsigned long long>(g_ud_partial));
        const std::string checked =
            g_binding_plans == 2 ? "; disagreements with resolving each binding " + std::to_string(g_plans_disagreed) : std::string();
        host_log("render: binding plans: %llu built (%llu pushed out of the table), %llu reused%s",
                 static_cast<unsigned long long>(g_plans_built), static_cast<unsigned long long>(g_plans_evicted),
                 static_cast<unsigned long long>(g_plans_used), checked.c_str());
    }
    report_gx_state_compare();
    if (g_view_extents) {
        host_log("render: GX target extents from their views' textures %llu (%llu of them view-less draws' from the GX texture at the target), "
                 "guessed from the scissor %llu%s",
                 static_cast<unsigned long long>(g_extent_used), static_cast<unsigned long long>(g_extent_from_gx),
                 static_cast<unsigned long long>(g_extent_guessed),
                 g_view_extents == 2 ? ("; compared " + std::to_string(g_extent_checked) + ", disagreeing " + std::to_string(g_extent_disagree)).c_str() : "");
    }
    if (g_view_by_id) {
        host_log("render: %s%s", texture_view_by_id_report().c_str(),
                 g_view_by_id == 2 ? ("; disagreements with the T# " + std::to_string(g_view_by_id_disagree)).c_str() : "");
    }
    if (g_program_ids) {
        const std::string checked = g_program_ids == 2 ? "; disagreements with the address " + std::to_string(g_by_id_disagree) : std::string();
        host_log("render: programs and fetch shaders found by their GX objects' ids %llu, looked up by address for a new id %llu, GX draws "
                 "without shader objects (the wrapper's own: Scaleform, gamma, present) %llu, with a vertex-stage object the registry "
                 "did not see made %llu%s",
                 static_cast<unsigned long long>(g_by_id_hits), static_cast<unsigned long long>(g_by_id_misses),
                 static_cast<unsigned long long>(g_by_id_none), static_cast<unsigned long long>(g_by_id_unseen), checked.c_str());
    }
    host_log("render: resource paths of new programs taken from the per-shader compiles %llu, translated on the command processor %llu",
             static_cast<unsigned long long>(g_paths_taken.load()), static_cast<unsigned long long>(g_paths_translated.load()));
    if (g_fallbacks_deferred.load()) {
        host_log("render: fallback pipelines laid out from their no-fallback stages %llu; translated later for a draw %llu, binding "
                 "differently %llu (BBHOST_LAYOUT_FROM_LEAN)",
                 static_cast<unsigned long long>(g_fallbacks_deferred.load()), static_cast<unsigned long long>(g_fallbacks_translated.load()),
                 static_cast<unsigned long long>(g_fallback_binding_mismatch.load()));
    }
    host_log("render: stage translation variants:%s", stage_variant_report().c_str());
    if (g_token_draws_seen.load()) {
        host_log("render: token draws %llu, with GX inputs %llu, tessellated %llu",
                 static_cast<unsigned long long>(g_token_draws_seen.load()), static_cast<unsigned long long>(g_token_with_gx.load()),
                 static_cast<unsigned long long>(g_token_tess.load()));
    }
    if (g_frames_watched) {
        host_log("render: %llu of %llu frames watched went over the brightness mark (%.1f%%)",
                 static_cast<unsigned long long>(g_frames_bright), static_cast<unsigned long long>(g_frames_watched),
                 100.0 * static_cast<double>(g_frames_bright) / static_cast<double>(g_frames_watched));
    }
    if (const std::uint64_t n = g_tess_draws.load()) {
    if (g_tess_ls_user_gx.load() || g_tess_ls_user_fallback.load()) {
        std::string slots;
        for (int k = 0; k < 16; ++k) {
            if (!g_tess_ls_slot[k].load()) continue;
            char b[64];
            std::snprintf(b, sizeof(b), " s[%d] %llu of %llu;", k, static_cast<unsigned long long>(g_tess_ls_slot_bad[k].load()),
                          static_cast<unsigned long long>(g_tess_ls_slot[k].load()));
            slots += b;
        }
        host_log("render: the LS's user data built from GX for %llu draws (%llu kept the command stream's, %llu with a vertex table of ours); differing by slot:%s",
                 static_cast<unsigned long long>(g_tess_ls_user_gx.load()),
                 static_cast<unsigned long long>(g_tess_ls_user_fallback.load()),
                 static_cast<unsigned long long>(g_tess_ls_table.load()), slots.empty() ? " none" : slots.c_str());
    }
    host_log("render: tessellated draws %llu (%llu patches, %llu vertices; at most %u patches and %u cells a side in a draw), "
                 "refused %llu",
                 static_cast<unsigned long long>(n), static_cast<unsigned long long>(g_tess_patches.load()),
                 static_cast<unsigned long long>(g_tess_vertices.load()), g_tess_max_patches.load(), g_tess_max_level.load(),
                 static_cast<unsigned long long>(g_tess_refused.load()));
        host_log("render: tessellated draws the game's hull shader ran for %llu (%llu patches); LDS buffer accesses %s",
                 static_cast<unsigned long long>(g_tess_hull_draws.load()), static_cast<unsigned long long>(g_tess_hull_patches.load()),
                 tess_lds_bound() ? "bounded to their window and region (BBHOST_TESS_LDS_BOUND=0: unchecked)"
                                  : "unchecked (BBHOST_TESS_LDS_BOUND=0)");
    }
    host_log("render: targets re-created in another format: %llu with their pixels carried over, %llu with them lost (another size "
             "or texel size), %llu of those starting with the fill their old image had last taken; new targets nothing "
             "initialised, cleared to zero: %llu",
             static_cast<unsigned long long>(g_rt_carried.load()), static_cast<unsigned long long>(g_rt_lost.load()),
             static_cast<unsigned long long>(g_rt_refilled.load()), static_cast<unsigned long long>(g_rt_zeroed.load()));
    host_log("render: targets=%llu clears=%llu fill-pending=%llu fill-applied=%llu target-copies=%llu untraced-draws=%llu "
             "vport-scissor-narrowed=%llu%s",
             static_cast<unsigned long long>(g_rt_created.load()), static_cast<unsigned long long>(g_clears.load()),
             static_cast<unsigned long long>(g_fill_pending.load()), static_cast<unsigned long long>(g_fill_applied.load()),
             static_cast<unsigned long long>(g_rt_copies.load()), static_cast<unsigned long long>(g_untraced_draws.load()),
             static_cast<unsigned long long>(g_vport_narrowed.load()), g_vport_scissor ? "" : " (not applied)");
    if (g_gx_resolves[kGxRecord].load() || g_gx_resolves[kGxNoDescriptor].load() || g_gx_resolves[kGxShape].load()) {
        const auto n = [](int k) { return static_cast<unsigned long long>(g_gx_resolves[k].load()); };
        host_log("render: GX bindings from records %llu; from user data: no descriptor %llu, path shape %llu, record shorter "
                 "than the read %llu, no object %llu",
                 n(kGxRecord), n(kGxNoDescriptor), n(kGxShape), n(kGxShort), n(kGxNoObject));
        const auto t = [](int k) { return static_cast<unsigned long long>(g_gx_token_resolves[k].load()); };
        host_log("render: in draws from host-draw tokens: bindings from records %llu; from user data: no descriptor %llu, path shape "
                 "%llu, record shorter than the read %llu, no object %llu",
                 t(kGxRecord), t(kGxNoDescriptor), t(kGxShape), t(kGxShort), t(kGxNoObject));
        std::lock_guard<std::mutex> lk(g_token_shape_mu);
        static const char* const kSites[] = {"pipeline key", "prefetch", "constant buffers"};
        for (const auto& [k, n] : g_token_shape_paths) {
            host_log("render:   token binding through user data: %s in the %s, %llu resolves", k.first.c_str(), kSites[k.second],
                     static_cast<unsigned long long>(n));
        }
    }
    if (g_vertex_tables.load() || g_gx_user_stages.load() || g_gx_user_fallbacks.load()) {
        host_log("render: GX vertex tables from the call %llu; stage user data built %llu, left to the command stream %llu (ring at 0x%llx)",
                 static_cast<unsigned long long>(g_vertex_tables.load()), static_cast<unsigned long long>(g_gx_user_stages.load()),
                 static_cast<unsigned long long>(g_gx_user_fallbacks.load()), static_cast<unsigned long long>(g_gx_ring.base));
    }
    if (g_render_cost_enabled) {
        const auto per = [](int k) {
            const std::uint64_t c = g_render_cost_n[k].load();
            return static_cast<unsigned long long>(c ? g_render_cost_ns[k].load() / c : 0);
        };
        const auto calls = [](int k) { return static_cast<unsigned long long>(g_render_cost_n[k].load()); };
        host_log("render: GX cost ns each: input compare %llu (%llu), stage user data %llu (%llu), vertex table %llu (%llu)",
                 per(kRenderCostCompare), calls(kRenderCostCompare), per(kRenderCostUserData), calls(kRenderCostUserData),
                 per(kRenderCostVertexTable), calls(kRenderCostVertexTable));
        report_render_split(true);
        host_log("render: GX constant buffers sampled: %llu stages, %llu buffers, %llu bytes; changed between the call and the draw %llu",
                 static_cast<unsigned long long>(g_cb_check_stages.load()), static_cast<unsigned long long>(g_cb_check_buffers.load()),
                 static_cast<unsigned long long>(g_cb_check_bytes.load()), static_cast<unsigned long long>(g_cb_check_changed.load()));
    }
    {
        // Page-table walks the shaders keep, by reason: program
        // sites and the draws that ran them, over the no-fallback variants.
        // Draws also by source: from GX records, and of those from host-draw tokens.
        struct WalkCount {
            std::uint64_t sites = 0, draws = 0, gx_draws = 0, token_draws = 0;
        };
        std::map<std::string, WalkCount> walks;  // by reason
        std::uint64_t lean_draws = 0, walk_draws = 0, other_draws = 0;
        std::uint64_t lean_tokens = 0, walk_tokens = 0, other_tokens = 0;
        for (const auto& [key, p] : g_gfx) {
            if (p.building) continue;
            if (!p.lean) {
                other_draws += p.draws;
                other_tokens += p.token_draws;
                continue;
            }
            lean_draws += p.draws;
            lean_tokens += p.token_draws;
            bool any = false;
            for (const gcn::TranslateResult* meta : {&p.vs.meta(), &p.ps.meta()}) {
                for (const auto& [why, sites] : meta->walks) {
                    WalkCount& w = walks[why];
                    w.sites += sites;
                    w.draws += p.draws;
                    w.gx_draws += p.gx_draws;
                    w.token_draws += p.token_draws;
                    any = true;
                }
            }
            if (any) {
                walk_draws += p.draws;
                walk_tokens += p.token_draws;
            }
        }
        host_log("render: prefetch stages that took the pipeline key's binding words %llu, that resolved them again %llu",
                 static_cast<unsigned long long>(g_prefetch_key_words.load()), static_cast<unsigned long long>(g_prefetch_resolved.load()));
        if (g_skip_tables) {
            host_log("render: no-fallback variants built on worker threads %llu of %llu queued",
                     static_cast<unsigned long long>(g_lean_builds_done.load()), static_cast<unsigned long long>(g_lean_builds_queued.load()));
            host_log("render: table read masks computed on worker threads %llu of %llu queued",
                     static_cast<unsigned long long>(g_table_masks_done.load()), static_cast<unsigned long long>(g_table_masks_queued.load()));
            host_log("render: GX draws with user-data tables or blocks left out of the ring %llu (%llu of them); rebuilt for the fallback "
                     "variant %llu, of which bound in full after all %llu",
                     static_cast<unsigned long long>(g_skip_draws.load()), static_cast<unsigned long long>(g_tables_skipped.load()),
                     static_cast<unsigned long long>(g_skip_rebuilds.load()), static_cast<unsigned long long>(g_skip_rebuilds_lean.load()));
        }
        host_log("render: draws on no-fallback pipelines %llu (%llu of them keep page-table walks), on fallback pipelines %llu; "
                 "of them from host-draw tokens %llu (%llu), %llu",
                 static_cast<unsigned long long>(lean_draws), static_cast<unsigned long long>(walk_draws),
                 static_cast<unsigned long long>(other_draws), static_cast<unsigned long long>(lean_tokens),
                 static_cast<unsigned long long>(walk_tokens), static_cast<unsigned long long>(other_tokens));
        for (const auto& [why, v] : walks) {
            host_log("render:   walk %s: %llu sites, %llu draws (from GX records %llu, from tokens %llu)", why.c_str(),
                     static_cast<unsigned long long>(v.sites), static_cast<unsigned long long>(v.draws),
                     static_cast<unsigned long long>(v.gx_draws), static_cast<unsigned long long>(v.token_draws));
            // Which pipelines keep this walk, so the remaining reasons can be
            // read as shaders to fix rather than a count: the architecture
            // criterion is per operation, and these are the operations left.
            std::vector<std::pair<std::uint64_t, const GfxPipeline*>> by_draws;
            for (const auto& [key, p] : g_gfx) {
                if (p.building || !p.lean) continue;
                const bool vs_has = p.vs.meta().walks.count(why) != 0;
                const bool ps_has = p.ps.meta().walks.count(why) != 0;
                if (vs_has || ps_has) by_draws.push_back({p.draws, &p});
            }
            std::sort(by_draws.begin(), by_draws.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t k = 0; k < by_draws.size() && k < 6; ++k) {
                const GfxPipeline& p = *by_draws[k].second;
                const auto sites = [&](const gcn::TranslateResult& m) {
                    const auto it = m.walks.find(why);
                    return it == m.walks.end() ? 0ull : static_cast<unsigned long long>(it->second);
                };
                host_log("render:     %s: %llu draws, vs %llu sites, ps %llu sites", p.name.c_str(),
                         static_cast<unsigned long long>(by_draws[k].first), sites(p.vs.meta()), sites(p.ps.meta()));
            }
        }
        // Which pipelines still have a lean variant reading a user-data table,
        // so the remaining constant-engine reconstruction reads as shaders to
        // fix. s[2] is the extended block (descriptor type 0x1a), s[12] the PS
        // texture table (type 8).
        {
            std::vector<std::pair<std::uint64_t, const GfxPipeline*>> readers;
            for (const auto& [key, p] : g_gfx) {
                if (p.building || !p.token_draws) continue;
                const GfxPipeline* lean = p.lean ? &p : p.lean_variant;
                if (!lean || (!lean->table_reads[0] && !lean->table_reads[1])) continue;
                readers.push_back({p.token_draws, &p});
            }
            std::sort(readers.begin(), readers.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t k = 0; k < readers.size() && k < 8; ++k) {
                const GfxPipeline& p = *readers[k].second;
                const GfxPipeline* lean = p.lean ? &p : p.lean_variant;
                host_log("render:   table reader %s: %llu token draws, vs slots 0x%x, ps slots 0x%x", p.name.c_str(),
                         static_cast<unsigned long long>(readers[k].first), lean->table_reads[0], lean->table_reads[1]);
                if (k < 3) {
                    // What the stage binds, so the reason the table survives is
                    // visible: a pointer buffer over the table means its records
                    // are read in the shader instead of resolved to descriptors.
                    for (const gcn::BufferBinding& b : lean->ps.meta().buffers) {
                        host_log("render:       ps buffer %s%s", b.path.str().c_str(), b.pointer ? " (pointer)" : "");
                    }
                    std::map<std::string, int> image_paths;
                    for (const gcn::ImageBinding& b : lean->ps.meta().images) ++image_paths[b.path.str()];
                    for (const auto& [path, n] : image_paths) host_log("render:       ps image %s x%d", path.c_str(), n);
                }
            }
            host_log("render:   %zu pipelines still read a user-data table", readers.size());
        }
        std::string slots;
        for (int k = 0; k < 16; ++k) {
            if (const std::uint64_t n = g_token_tables_by_slot[k].load()) slots += " s[" + std::to_string(k) + "] " + std::to_string(n);
        }
        host_log("render: tables and blocks still placed in the ring for GX draws %llu, for draws from tokens %llu, by user-data slot:%s",
                 static_cast<unsigned long long>(g_gx_tables_placed.load()), static_cast<unsigned long long>(g_token_tables_placed.load()),
                 slots.empty() ? " none" : slots.c_str());
        std::string stale_slots;
        for (int k = 0; k < 16; ++k) {
            if (const std::uint64_t n = g_token_stale_by_slot[k].load()) stale_slots += " s[" + std::to_string(k) + "] " + std::to_string(n);
        }
        if (g_token_user_data_check) {
            host_log("render: token draw stages checked %llu; that read a register-file user-data slot no descriptor names, nonzero %llu, "
                     "by slot:%s",
                     static_cast<unsigned long long>(g_token_stages_checked.load()), static_cast<unsigned long long>(g_token_stale_stages.load()),
                     stale_slots.empty() ? " none" : stale_slots.c_str());
        }
        std::string set_slots;
        for (int k = 0; k < 16; ++k) {
            if (const std::uint64_t n = g_token_unnamed_set_by_slot[k].load()) set_slots += " s[" + std::to_string(k) + "] " + std::to_string(n);
        }
        if (g_token_user_data_check) {
            host_log("render: token draw stages with an unnamed user-data slot not 0 in the register file %llu, by slot:%s",
                     static_cast<unsigned long long>(g_token_unnamed_set_stages.load()), set_slots.empty() ? " none" : set_slots.c_str());
        }
        if (g_token_user_data_check) {
            host_log("render: token draw stages checked against the variant that drew them %llu; reading an unnamed slot not 0 %llu, "
                     "on fallback variants %llu",
                     static_cast<unsigned long long>(g_token_bound_checked.load()), static_cast<unsigned long long>(g_token_bound_stale.load()),
                     static_cast<unsigned long long>(g_token_bound_stale_fallback.load()));
        }
        host_log("render: HTILE clears run by draws from tokens %llu, with a nonzero clear value %llu; clears by any draw whose "
                 "register clear values at the draw differed from the fill's %llu; fills whose depth DB_DEPTH_CLEAR did not yet hold %llu",
                 static_cast<unsigned long long>(g_token_htile_clears.load()), static_cast<unsigned long long>(g_token_htile_clear_values.load()),
                 static_cast<unsigned long long>(g_htile_clear_value_mismatch.load()), static_cast<unsigned long long>(g_htile_fill_register_lag.load()));
    }
    if (g_vertex_input) {
        host_log("render: GX draws with Vulkan vertex input %llu, fetch shader kept %llu; vertex tables placed for vertex shaders that "
                 "read them %llu; tables the game bound itself, read from memory %llu",
                 static_cast<unsigned long long>(g_vertex_input_draws.load()), static_cast<unsigned long long>(g_vertex_input_kept.load()),
                 static_cast<unsigned long long>(g_vertex_tables_with_input.load()), static_cast<unsigned long long>(g_vertex_tables_read.load()));
    }
    if (const std::uint64_t n = g_set_reused.load() + g_set_built.load()) {
        host_log("render: stage descriptor sets %llu, taken from the stage's previous draw %llu (%.1f%%), written %llu; "
                 "given up for a table rebuild %llu",
                 static_cast<unsigned long long>(n), static_cast<unsigned long long>(g_set_reused.load()),
                 100.0 * g_set_reused.load() / n, static_cast<unsigned long long>(g_set_built.load()),
                 static_cast<unsigned long long>(g_set_reuse_undone.load()));
    }
    if (const std::uint64_t n = g_par_stages.load()) {
        host_log("render: stage params %llu, with the contents of the stage's previous draw %llu (%.1f%%), of a slot earlier in the "
                 "command buffer %llu (%.1f%%); both params and set repeated %llu (%.1f%%)",
                 static_cast<unsigned long long>(n), static_cast<unsigned long long>(g_par_same_as_last.load()),
                 100.0 * g_par_same_as_last.load() / n, static_cast<unsigned long long>(g_par_seen_in_cb.load()),
                 100.0 * g_par_seen_in_cb.load() / n, static_cast<unsigned long long>(g_par_both.load()),
                 100.0 * g_par_both.load() / n);
    }
    textures_report();
}

RtImage* find_render_target(std::uint64_t base) {
    auto it = g_rts.find(base);
    if (it != g_rts.end()) return &it->second;
    auto st = g_snapshots.find(base);
    return st == g_snapshots.end() ? nullptr : &st->second;
}

bool render_copy_target_locked(std::uint64_t src_base, std::uint64_t dst_base, std::size_t bytes) {
    // BBHOST_WATCH_COPY_DST=0x<base>: every whole-target copy into that
    // address, with the flip. A frame can go wrong without any draw being
    // different - the engine also moves whole surfaces around, and a copy from
    // a differently laid out surface would land as a full-screen wash.
    static const std::uint64_t watch_dst = [] {
        const char* e = std::getenv("BBHOST_WATCH_COPY_DST");
        return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    if (watch_dst && dst_base == watch_dst) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 200) {
            RtImage* s0 = find_render_target(src_base);
            host_log("copy: 0x%llx -> 0x%llx %zu bytes at flip %llu (source %s)", static_cast<unsigned long long>(src_base),
                     static_cast<unsigned long long>(dst_base), bytes, static_cast<unsigned long long>(hle_video_flip_count()),
                     s0 ? (s0->depth ? "a depth target" : "a colour target") : "not a render target");
        }
    }
    auto it = g_rts.find(src_base);
    if (it == g_rts.end() || !it->second.initialised) return false;
    RtImage& src = it->second;
    // A depth source becomes an R32_SFLOAT colour snapshot (its Z plane),
    // sampled by the lighting passes as a plain float texture.
    const bool from_depth = src.depth;
    if (from_depth && src.format != VK_FORMAT_D32_SFLOAT && src.format != VK_FORMAT_D32_SFLOAT_S8_UINT) return false;
    const VkFormat snap_format = from_depth ? VK_FORMAT_R32_SFLOAT : src.format;
    const std::size_t src_bytes = static_cast<std::size_t>(src.width) * src.height * (from_depth ? 4 : rt_bytes_per_pixel(src));
    if (bytes < src_bytes) return false;
    // If a render target already lives at the destination, copy into *it*.
    // find_render_target() - and so every sampler - prefers g_rts over
    // g_snapshots, so writing a parallel snapshot here would leave the copy
    // where nothing reads it: the game ping-pongs its full-screen blur between
    // a target it renders into and one it copies into, and the copied half
    // stayed at whatever the render half last left, which is how the composited
    // frame went black once the world started using that chain.
    RtImage* into = nullptr;
    if (auto rt = g_rts.find(dst_base);
        rt != g_rts.end() && !rt->second.depth && rt->second.format == snap_format && rt->second.width == src.width &&
        rt->second.height == src.height) {
        into = &rt->second;
    }
    auto st = g_snapshots.find(dst_base);
    if (!into && st != g_snapshots.end() &&
        (st->second.format != snap_format || st->second.width != src.width || st->second.height != src.height)) {
        g.rt_replacements.fetch_add(1);
        destroy_rt_image(st->second);
        g_snapshots.erase(st);
        ++g_rt_gen;
        bump_view_epoch();
        st = g_snapshots.end();
    }
    if (!into && st == g_snapshots.end()) {
        RtImage r;
        r.base = dst_base;
        r.format = snap_format;
        r.width = src.width;
        r.height = src.height;
        r.depth = false;
        r.mutable_format = true;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = r.format;
        ici.extent = {r.width, r.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        // A depth snapshot is written by the copy's compute pass (depth_copy.cpp).
        r.storage = from_depth && depth_copy_available_locked();
        if (r.storage) ici.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g.device, &ici, nullptr, &r.image) != VK_SUCCESS) return false;
        if (!rt_image_memory(r.image, r.memory)) {
            vkDestroyImage(g.device, r.image, nullptr);
            if (r.memory.memory) defer_destroy_image(VK_NULL_HANDLE, r.memory);
            return false;
        }
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = r.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = r.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(g.device, &vci, nullptr, &r.view) != VK_SUCCESS) {
            vkDestroyImage(g.device, r.image, nullptr);
            defer_destroy_image(VK_NULL_HANDLE, r.memory);
            return false;
        }
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("render: %s target 0x%llx copied to 0x%llx (%ux%u format %d): snapshot image", from_depth ? "depth" : "colour",
                     static_cast<unsigned long long>(src_base), static_cast<unsigned long long>(dst_base), r.width, r.height, r.format);
        }
        ++g_rt_gen;
        st = g_snapshots.emplace(dst_base, r).first;
        g_max_rt_bytes = std::max<std::uint64_t>(g_max_rt_bytes, rt_size_bytes(st->second));
        bump_view_epoch();
    }
    if (!into) {
        // A render target at the destination that did not match is now stale -
        // the copy overwrote the memory it stands for - and samplers would
        // still prefer it over the snapshot. Drop it.
        if (auto rt = g_rts.find(dst_base); rt != g_rts.end()) {
            g.rt_replacements.fetch_add(1);
            destroy_rt_image(rt->second);
            g_rts.erase(rt);
            ++g_rt_gen;
            bump_view_epoch();
        }
    }
    RtImage& dst = into ? *into : st->second;
    begin_recording_locked();
    render_end_pass_locked();
    if (!dst.initialised) {
        image_barrier(g_cmd(), dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        dst.initialised = true;
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    // A depth plane into a snapshot made for it: one compute pass.
    const bool by_compute = from_depth && dst.storage && depth_copy_available_locked();
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = by_compute ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         by_compute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    // BBHOST_GPU_PROFILE: the copy as an entry of its own, by kind and size.
    static std::map<std::string, std::string> copy_names;
    if (g.profile) {
        char key[96];
        std::snprintf(key, sizeof(key), "rt-copy-%s-%ux%u-f%d", by_compute ? "depth-compute" : from_depth ? "depth" : "colour", src.width,
                      src.height, static_cast<int>(src.format));
        profile_begin_locked(&copy_names.emplace(key, key).first->second);
    }
    if (by_compute && depth_copy_record_locked(src.image, src.format, dst.image, src.width, src.height)) {
        if (g.profile) profile_end_locked();
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        rec().pipeline_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        if (g_pending_clears.erase(dst_base)) ++g_pending_gen;
        dst.fill_last = false;
        g_rt_copies.fetch_add(1);
        return true;
    }
    if (by_compute) {
        // The pass could not be recorded: the buffer copies, behind the transfer barrier they expect.
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    if (from_depth) {
        // Depth and colour formats cannot be copied image-to-image: go through a
        // buffer. One, kept and grown: a buffer allocated per copy and freed
        // at the slot's retirement was ~4% of the command processor in the
        // world (vkAllocateMemory and vkFreeMemory, several copies a frame).
        // The barrier above orders each reuse after the last one's reads.
        static DevBuffer scratch;
        if (scratch.size < src_bytes) {
            if (scratch.buffer) defer_destroy(scratch);
            scratch = DevBuffer{};
            std::uint64_t want = 1ull << 20;
            while (want < src_bytes) want <<= 1;
            if (!create_dev_buffer(scratch, want, false)) {
                scratch = DevBuffer{};
                return false;
            }
        }
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        bic.imageExtent = {src.width, src.height, 1};
        rec().copy_image_to_buffer(src.image, VK_IMAGE_LAYOUT_GENERAL, scratch.buffer, 1, &bic);
        VkMemoryBarrier tb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        tb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        tb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &tb, 0, nullptr, 0, nullptr);
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rec().copy_buffer_to_image(scratch.buffer, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &bic);
    } else {
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {src.width, src.height, 1};
        rec().copy_image(src.image, VK_IMAGE_LAYOUT_GENERAL, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }
    if (g.profile) profile_end_locked();
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (g_pending_clears.erase(dst_base)) ++g_pending_gen;
    dst.fill_last = false;
    g_rt_copies.fetch_add(1);
    return true;
}

bool resolve_resource(const gcn::ResourcePath& path, const std::uint32_t* user, int ndw, std::uint32_t* out,
                      std::vector<std::pair<std::uint64_t, std::uint64_t>>* touched) {
    return resolve_resource_impl(path, user, ndw, out, touched);
}

std::vector<bool> sampler_modes_for(const gcn::TranslateResult& meta, const std::uint32_t* user, const GxStageRecords* gx) {
    std::vector<bool> modes;
    for (const auto& sampler : meta.samplers) {
        std::uint32_t words[4]{};
        modes.push_back(resolve_binding(sampler.path, user, gx, 4, words) && (words[0] & (1u << 15)));
    }
    return modes;
}

namespace {
// The image type a translation declares for a binding, from the words it
// resolved to: dimension (spv::Dim) and arrayed.
std::pair<std::uint32_t, bool> image_dim_of(const gcn::ImageBinding& b, bool resolved, const std::uint32_t* w) {
    std::uint32_t dim = 1;
    bool arrayed = false;
    if (resolved) {
        const std::uint32_t type = (w[3] >> 28) & 0xf;
        const std::uint32_t base_array = w[5] & 0x1fff;
        const std::uint32_t last_array = (w[5] >> 13) & 0x1fff;
        // MIMG r128 is a V# sampled as a 1D image. Type without bit 3
        // and no r128 is an invalid T# (shadPS4 Image::Null) — keep 2D
        // so the bind dummy matches the sample.
        if (b.r128) {
            dim = 0;
            arrayed = false;
        } else if ((type & 8u) == 0) {
            dim = 1;
            arrayed = false;
        } else if (tsharp_sample_as_2d(type, base_array, last_array, b.da)) {
            dim = 1;
            arrayed = false;
        } else {
            dim = tsharp_dim(type, arrayed);
        }
        // The texel type (tsharp_kind), for the translator to declare the
        // image with; a depth comparison is always float.
        if (!b.r128 && !b.depth && (type & 8u) != 0) dim |= tsharp_kind(w) << 8;
    }
    return {dim, arrayed};
}
}  // namespace

std::vector<std::pair<std::uint32_t, bool>> image_dims_for(const gcn::TranslateResult& meta, const std::uint32_t* stage_user,
                                                           const GxStageRecords* gx) {
    std::vector<std::pair<std::uint32_t, bool>> dims;
    for (const gcn::ImageBinding& b : meta.images) {
        std::uint32_t w[8] = {};
        const bool resolved = resolve_binding(b.path, stage_user, gx, b.r128 ? 4 : 8, w);
        dims.push_back(image_dim_of(b, resolved, w));
    }
    return dims;
}

void resolve_key_stage(const gcn::TranslateResult* paths, const std::uint32_t* stage_user, const GxStageRecords* gx, KeyStage& out) {
    out.dims.clear();
    out.modes.clear();
    out.images.clear();
    out.samplers.clear();
    if (!paths) return;
    const GxStagePlan* plan = gx && g_binding_plans ? &stage_plan(*paths, *gx) : nullptr;
    for (std::size_t k = 0; k < paths->images.size(); ++k) {
        const gcn::ImageBinding& b = paths->images[k];
        KeyStage::Words& r = out.images.emplace_back();
        r.resolved = resolve_via_plan(plan ? &plan->images[k] : nullptr, b.path, stage_user, gx, b.r128 ? 4 : 8, r.w);
        out.dims.push_back(image_dim_of(b, r.resolved, r.w));
    }
    for (std::size_t k = 0; k < paths->samplers.size(); ++k) {
        const gcn::SamplerBinding& b = paths->samplers[k];
        KeyStage::Words& r = out.samplers.emplace_back();
        r.resolved = resolve_via_plan(plan ? &plan->samplers[k] : nullptr, b.path, stage_user, gx, 4, r.w);
        out.modes.push_back(r.resolved && (r.w[0] & (1u << 15)));
    }
}

bool key_stage_matches(const gcn::TranslateResult& meta, const gcn::TranslateResult& paths) {
    if (meta.images.size() != paths.images.size() || meta.samplers.size() != paths.samplers.size()) return false;
    for (std::size_t k = 0; k < meta.images.size(); ++k) {
        if (meta.images[k].r128 != paths.images[k].r128 || meta.images[k].path.str() != paths.images[k].path.str()) return false;
    }
    for (std::size_t k = 0; k < meta.samplers.size(); ++k) {
        if (meta.samplers[k].path.str() != paths.samplers[k].path.str()) return false;
    }
    return true;
}

void prefetch_stage_images(const gcn::TranslateResult& meta, const std::uint32_t* stage_user, StageImages& out,
                           const GxStageRecords* gx, const KeyStage* resolved) {
    out.images.resize(meta.images.size());
    const GxStagePlan* plan = gx && g_binding_plans && g_view_by_id ? &stage_plan(meta, *gx) : nullptr;
    for (std::size_t k = 0; k < meta.images.size(); ++k) {
        const gcn::ImageBinding& b = meta.images[k];
        StageImages::Image& im = out.images[k];
        im = StageImages::Image{};
        if (resolved) {
            std::memcpy(im.w, resolved->images[k].w, sizeof(im.w));
            im.resolved = resolved->images[k].resolved;
        } else {
            im.resolved = resolve_binding(b.path, stage_user, gx, b.r128 ? 4 : 8, im.w);
        }
        if (!im.resolved && glitch_on() && b.path.user_sgpr >= 0 && b.path.user_sgpr < 15) {
            // The glitch hunt: every image that resolves to nothing, with what
            // its path started from - a frame that went black was a display
            // pass whose LUT's table pointer read as nothing.
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1, std::memory_order_relaxed) < 200) {
                const int u = b.path.user_sgpr;
                const std::uint64_t ptr = static_cast<std::uint64_t>(stage_user[u]) | (static_cast<std::uint64_t>(stage_user[u + 1]) << 32);
                host_log("glitch: draw %llu (flip %llu) %s image %zu (%s) did not resolve: s[%d:%d] = 0x%llx%s, %s, %s; %s", 
                         static_cast<unsigned long long>(g_draw_rec_next), static_cast<unsigned long long>(hle_video_flip_count()),
                         t_draw_pipeline ? t_draw_pipeline : "?", k, b.path.str().c_str(), u, u + 1, static_cast<unsigned long long>(ptr),
                         ptr && hle_kernel_va_mapped(ptr & 0xffffffffffull, 32) ? " (mapped)" : " (not mapped)",
                         resolved ? "from the pipeline key's resolution" : "resolved here", gx ? "GX records" : "no GX records",
                         t_draw_origin);
                if (std::strstr(t_draw_origin, "scaleform")) host_log("glitch:   its Scaleform token: %s", hle_gx_scaleform_last().c_str());
            }
        }
        if (im.resolved) {
            bool ok;
            const GxBindingPlan* bp = plan && k < plan->images.size() ? &plan->images[k] : nullptr;
            const std::uint32_t id = bp && bp->source == GxBindingPlan::kTex && ((gx->obj_tex_set >> bp->slot) & 1) ? gx->obj_tex_id[bp->slot] : 0;
            if (id) {
                im.view = texture_view_by_id(id, im.w, b.storage, ok, im.dim, im.arrayed, b.da, b.r128);
                if (g_view_by_id == 2) {
                    std::uint32_t dim = 0;
                    bool arrayed = false;
                    const VkImageView by_words = texture_view(im.w, b.storage, ok, dim, arrayed, b.da, b.r128);
                    if (by_words != im.view || dim != im.dim || arrayed != im.arrayed) {
                        if (g_view_by_id_disagree++ < 8) host_log("render: view of shader-resource view %u differs from its T#'s", id);
                        im.view = by_words;
                        im.dim = dim;
                        im.arrayed = arrayed;
                    }
                }
            } else {
                im.view = texture_view(im.w, b.storage, ok, im.dim, im.arrayed, b.da, b.r128);
            }
        }
    }
    out.samplers.resize(meta.samplers.size());
    for (std::size_t k = 0; k < meta.samplers.size(); ++k) {
        const gcn::SamplerBinding& b = meta.samplers[k];
        std::uint32_t w[4] = {};
        bool ok;
        if (resolved) {
            std::memcpy(w, resolved->samplers[k].w, sizeof(w));
            ok = resolved->samplers[k].resolved;
        } else {
            ok = resolve_binding(b.path, stage_user, gx, 4, w);
        }
        out.samplers[k] = ok ? sampler_for(w, b.compare) : VK_NULL_HANDLE;
    }
}

// The view each image binding of a stage gets: its texture's, or a dummy of
// the shape the shader declares; null for none at all (left unwritten).
void stage_final_views(const gcn::TranslateResult& meta, const StageImages& pre, const char* pipeline_name,
                       std::vector<VkImageView>& out) {
    out.assign(meta.images.size(), VK_NULL_HANDLE);
    static std::atomic<int> tlog{0};
    const bool yebis = pipeline_name && tlog.load(std::memory_order_relaxed) < 24 &&
                       (std::strncmp(pipeline_name, "7d668276", 8) == 0 || std::strncmp(pipeline_name, "d3ca03f3", 8) == 0 ||
                        std::strncmp(pipeline_name, "1ea5a07e", 8) == 0);
    for (std::size_t k = 0; k < meta.images.size(); ++k) {
        const gcn::ImageBinding& b = meta.images[k];
        const StageImages::Image& im = pre.images[k];
        const std::uint32_t* w = im.w;
        VkImageView view = im.view;
        const bool resolved = im.resolved;
        bool shape_ok = true;
        const std::uint32_t got_dim = im.dim;
        const bool got_arrayed = im.arrayed;
        if (resolved) {
            if (view && (got_dim != b.dim || got_arrayed != b.arrayed)) {
                view = VK_NULL_HANDLE;
                shape_ok = false;
            }
            if (yebis && view && w[0] != 0xffffffffu) {
                if (tlog.fetch_add(1) < 24) {
                    const std::uint64_t tbase = tsharp_base(w);
                    host_log("texture: %s b%u T# base 0x%llx dfmt %u nfmt %u %ux%u type %u tiling %u array %u..%u depth %u da %d r128 %d",
                             pipeline_name, b.binding, static_cast<unsigned long long>(tbase), (w[1] >> 20) & 0x3f,
                             (w[1] >> 26) & 0xf, (w[2] & 0x3fff) + 1, ((w[2] >> 14) & 0x3fff) + 1, (w[3] >> 28) & 0xf,
                             (w[3] >> 20) & 0x1f, w[5] & 0x1fff, (w[5] >> 13) & 0x1fff, (w[4] & 0x1fff) + 1, b.da ? 1 : 0,
                             b.r128 ? 1 : 0);
                }
            }
        }
        if (view && b.depth) {
            // What a depth-compare binding resolves to, once a base: a depth
            // target compares natively; anything in R32F (a colour target, a
            // snapshot of a depth target, a texture) does not on every driver.
            static std::set<std::uint64_t> said;
            const std::uint64_t base = tsharp_base(w);
            if (said.size() < 16 && said.insert(base).second) {
                const auto rt = g_rts.find(base);
                const auto sn = g_snapshots.find(base);
                host_log("render: %s binding %u compares against 0x%llx: %s", pipeline_name ? pipeline_name : "?", b.binding,
                         static_cast<unsigned long long>(base),
                         rt != g_rts.end() ? (rt->second.depth ? "a depth target" : "a colour target")
                                           : sn != g_snapshots.end() ? "a snapshot of a target" : "a texture");
            }
        }
        if (!view) {
            // A binding the shader reads that resolves to nothing is a black
            // texture in the output; say which pipeline and why, once each.
            static std::set<std::string> seen;
            char key[160];
            std::snprintf(key, sizeof(key), "%s|%s|%u", pipeline_name ? pipeline_name : "?", b.path.str().c_str(), b.binding);
            if (seen.size() < 60 && seen.insert(key).second) {
                const std::uint64_t tbase = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8;
                host_log("render: %s binding %u (%s) has no texture: %s, want dim %u%s, got dim %u%s",
                         pipeline_name ? pipeline_name : "?", b.binding, b.path.str().c_str(),
                         !resolved ? "T# unreadable" : !shape_ok ? "wrong shape" : "unsupported format", b.dim,
                         b.arrayed ? " array" : "", got_dim, got_arrayed ? " array" : "");
                host_log("  T# %08x %08x %08x %08x %08x %08x %08x %08x", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                host_log("  reads as T#: base 0x%llx dfmt %u nfmt %u %ux%u type %u tiling %u%s",
                         static_cast<unsigned long long>(tbase), (w[1] >> 20) & 0x3f, (w[1] >> 26) & 0xf,
                         (w[2] & 0x3fff) + 1, ((w[2] >> 14) & 0x3fff) + 1, (w[3] >> 28) & 0xf, (w[3] >> 20) & 0x1f,
                         hle_kernel_va_mapped(tbase, 16) ? "" : " (base not mapped)");
                const std::uint64_t vbase = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xffff) << 32);
                host_log("  reads as V#: base 0x%llx stride %u records %u%s", static_cast<unsigned long long>(vbase),
                         (w[1] >> 16) & 0x3fff, w[2], hle_kernel_va_mapped(vbase, 16) ? "" : " (base not mapped)");
            }
            // Leaving the descriptor unwritten is undefined behaviour that the
            // GPU can fault on, so always bind something of the right shape.
            // Invalid T# (no r128, type&8==0): GCN returns 0, not opaque black.
            const std::uint32_t ttype = (w[3] >> 28) & 0xf;
            const bool invalid_tsharp = resolved && !b.r128 && (ttype & 8u) == 0;
            view = dummy_view_for(b.dim, b.arrayed, invalid_tsharp);
            if (invalid_tsharp) {
                static std::atomic<int> zlogs{0};
                if ((zlogs.load(std::memory_order_relaxed) < 12 && zlogs.fetch_add(1) < 12)) {
                    const std::uint64_t vbase = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
                    host_log("texture: dummy-zero for invalid T# type %u V# 0x%llx (%s b%u)", ttype,
                             static_cast<unsigned long long>(vbase), pipeline_name ? pipeline_name : "?", b.binding);
                }
            }
            if (!view) continue;
            bump(g.dummy_images);
            ++g_draw_dummies;
        }
        out[k] = view;
    }
}

// The writes of a stage's images (views from stage_final_views) and samplers.
void stage_image_writes(VkDescriptorSet set, const gcn::TranslateResult& meta, const StageImages& pre, const std::vector<VkImageView>& views,
                        std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& infos) {
    for (std::size_t k = 0; k < meta.images.size(); ++k) {
        if (!views[k]) continue;
        const gcn::ImageBinding& b = meta.images[k];
        infos.push_back({VK_NULL_HANDLE, views[k], VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet wi{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wi.dstSet = set;
        wi.dstBinding = b.binding;
        wi.descriptorCount = 1;
        wi.descriptorType = b.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        wi.pImageInfo = &infos.back();
        writes.push_back(wi);
    }
    for (std::size_t k = 0; k < meta.samplers.size(); ++k) {
        const gcn::SamplerBinding& b = meta.samplers[k];
        VkSampler smp = pre.samplers[k];
        if (!smp) smp = g.dummy_sampler;
        infos.push_back({smp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
        VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        ws.dstSet = set;
        ws.dstBinding = b.binding;
        ws.descriptorCount = 1;
        ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        ws.pImageInfo = &infos.back();
        writes.push_back(ws);
    }
}

void bind_stage_images(VkDescriptorSet set, const gcn::TranslateResult& meta, const StageImages& pre,
                       std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& infos,
                       const char* pipeline_name) {
    static thread_local std::vector<VkImageView> views;
    stage_final_views(meta, pre, pipeline_name, views);
    stage_image_writes(set, meta, pre, views, writes, infos);
}

bool render_blit_display_locked(VkCommandBuffer cmd, std::uint64_t display_va, VkImage dst, VkRect2D area, VkExtent2D src,
                                std::uint32_t src_x, std::uint32_t src_y, VkImageView dst_view) {
    auto it = g_rts.find(display_va);
    if (it == g_rts.end() || !it->second.initialised) return false;
    RtImage& r = it->second;
    // The display area is the registered buffer size; the target may carry
    // tiling padding rows below it.
    const std::uint32_t x0 = std::min(src_x, r.width), y0 = std::min(src_y, r.height);
    const std::uint32_t sw = src.width && x0 + src.width <= r.width ? src.width : r.width - x0;
    const std::uint32_t sh = src.height && y0 + src.height <= r.height ? src.height : r.height - y0;
    // Smaller than the window (a lower render size, the Deck's 960x600): FSR 1
    // rather than a bilinear stretch (host/fsr.cpp).
    if (sw <= area.extent.width && sh <= area.extent.height && (sw < area.extent.width || sh < area.extent.height) &&
        fsr_upscale_locked(cmd, r.image, r.format, r.width, r.height, x0, y0, sw, sh, dst, area, dst_view))
        return true;
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0] = {static_cast<int32_t>(x0), static_cast<int32_t>(y0), 0};
    blit.srcOffsets[1] = {static_cast<int32_t>(x0 + sw), static_cast<int32_t>(y0 + sh), 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[0] = {area.offset.x, area.offset.y, 0};
    blit.dstOffsets[1] = {area.offset.x + static_cast<int32_t>(area.extent.width), area.offset.y + static_cast<int32_t>(area.extent.height), 1};
    vkCmdBlitImage(cmd, r.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    return true;
}

void render_pipeline_time_us(std::uint64_t out[5]) {
    out[0] = g_pl_translate_us.load();
    out[1] = g_pl_module_us.load();
    out[2] = g_pl_create_us.load();
    out[3] = g_pl_stage_translations.load();
    out[4] = g_pl_stage_repeats.load();
}

bool render_pass_open_locked() { return g_pass.active; }

std::string render_barriers_report() {
    if (!g_lazy_barriers) return "";
    char b[200];
    std::snprintf(b, sizeof(b), "pass barriers (lazy): %llu paid, %llu left out, %llu of the paid in the middle of a pass",
                  static_cast<unsigned long long>(g_barriers_paid.load()), static_cast<unsigned long long>(g_barriers_skipped.load()),
                  static_cast<unsigned long long>(g_barriers_mid_pass.load()));
    return b;
}

}  // namespace gpu

using namespace gpu;

std::string host_gpu_fill_stats() { return render_fill_stats(); }
void host_gpu_pipeline_time_us(std::uint64_t out[5]) { render_pipeline_time_us(out); }

// Dynamic state already recorded into the command buffer being built. Every
// graphics pipeline declares the same dynamic set, and that state lasts for the
// whole recording, so a draw records only what differs from the draw before
// (13 calls into the driver per draw otherwise).
struct RecordedState {
    std::uint64_t serial = ~0ull;  // Gpu::record_serial of the recording this describes
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool viewport_set = false, scissor_set = false, bounds_set = false, stencil_set = false, blend_set = false;
    VkViewport viewport{};
    VkRect2D scissor{};
    float bounds[2]{};
    std::uint32_t stencil[6]{};
    float blend[4]{};
    // Pipelines linked from libraries: the state they take per draw.
    bool library_set = false;
    DrawLibraryState library{};
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceSize index_offset = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
};
static RecordedState g_recorded;

namespace {

// BBHOST_ARENA_MONITOR=1: which GPU reads land in the game's command-arena
// chunks. Storage-buffer
// constants, index buffers and indirect arguments are read when the GPU
// executes the draw, so a chunk freed when the CP walks its retirement could be
// rewritten under a queued draw. Counts per kind, the first examples, and a
// summary every 200,000 bindings.
enum ArenaReadKind { kArenaConstants, kArenaIndex, kArenaIndirect, kArenaKinds };

void note_arena_read(ArenaReadKind kind, std::uint64_t va, std::uint64_t bytes) {
    static const bool on = hle_gnm_arena_monitor();
    if (!on) return;
    static std::atomic<std::uint64_t> total[kArenaKinds]{}, hits[kArenaKinds]{}, hit_bytes[kArenaKinds]{};
    static std::atomic<std::uint64_t> checked{0};
    static const char* const names[kArenaKinds] = {"constants", "index buffer", "indirect args"};
    total[kind].fetch_add(1);
    std::uint64_t chunk = 0;
    if (hle_gx_arena_overlap(va, bytes, &chunk)) {
        hits[kind].fetch_add(1);
        hit_bytes[kind].fetch_add(bytes);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 24) {
            host_log("render: arena read: %s at 0x%llx (%llu bytes) in chunk 0x%llx", names[kind], static_cast<unsigned long long>(va),
                     static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(chunk));
        }
    }
    if (checked.fetch_add(1) % 200000 == 199999) {
        host_log("render: arena reads: constants %llu of %llu (%llu KiB), index buffers %llu of %llu (%llu KiB), indirect %llu of %llu",
                 static_cast<unsigned long long>(hits[kArenaConstants].load()), static_cast<unsigned long long>(total[kArenaConstants].load()),
                 static_cast<unsigned long long>(hit_bytes[kArenaConstants].load() / 1024),
                 static_cast<unsigned long long>(hits[kArenaIndex].load()), static_cast<unsigned long long>(total[kArenaIndex].load()),
                 static_cast<unsigned long long>(hit_bytes[kArenaIndex].load() / 1024),
                 static_cast<unsigned long long>(hits[kArenaIndirect].load()), static_cast<unsigned long long>(total[kArenaIndirect].load()));
    }
}

// YEBIS's constant windows (host_gpu_draw_window): given at a draw's token,
// before the draw, on the command processor's thread. The CPU writes each into
// its ring slot at once, and the draw binds a copy taken as it records -
// a slot comes round again sixteen draws on, while the draw that last read it
// may not have run yet, so the slot itself is never what a draw reads.
struct DrawWindow {
    std::uint64_t dst = 0;
    std::uint32_t bytes = 0;
    std::uint64_t serial = ~0ull;  // Gpu::record_serial its copy was taken in
    DevBuffer copy;
    VkDeviceSize offset = 0;
};
thread_local std::vector<DrawWindow> t_given_windows;  // for the next draw
thread_local std::vector<DrawWindow> t_draw_windows;   // the current draw's
std::atomic<std::uint64_t> g_windows_given{0}, g_windows_bound{0}, g_windows_past{0};

// A binding inside one of the current draw's windows: its copy, made in this
// recording on first use. False when none holds [base, base + bytes).
bool draw_window_locked(std::uint64_t base, std::uint64_t bytes, Located& out) {
    for (DrawWindow& w : t_draw_windows) {
        if (base < w.dst || base - w.dst >= w.bytes) continue;
        if (base - w.dst + bytes > w.bytes) {
            g_windows_past.fetch_add(1, std::memory_order_relaxed);  // read past the window: the slot through its import
            return false;
        }
        if (w.serial != g.record_serial) {
            begin_recording_locked();  // the copy belongs to the recording that binds it
            if (!acquire_staging_locked(w.copy, w.bytes, w.offset)) return false;
            std::memcpy(w.copy.map, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(w.dst)), w.bytes);
            w.serial = g.record_serial;
        }
        out.buffer = w.copy.buffer;
        out.offset = w.offset + (base - w.dst);
        out.avail = w.bytes - (base - w.dst);
        g_windows_bound.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

// TranslateOptions::cb_ssbo: bind each constant buffer the shader reads
// through a storage buffer over the V#'s imported memory. The range starts at
// the V# base aligned down to minStorageBufferOffsetAlignment and the shader
// indexes it from cb_bias_dw. A V# outside imported memory, not on a dword
// boundary, or too long for one binding stays on the page-table path
// (cb_valid bit clear) and gets the dummy buffer.
void resolve_stage_buffers(const gcn::TranslateResult& meta, const std::uint32_t* user, gcn::StageParams& params,
                           std::vector<VkDescriptorBufferInfo>& infos, const GxStageRecords* gx = nullptr) {
    // Why V#s end up on the page-table path, logged every 200,000 resolves.
    enum { kBound, kNoBase, kUnaligned, kUnmapped, kTooLong, kOutcomes };
    static std::uint64_t outcomes[kOutcomes] = {};
    static std::uint64_t resolves = 0;
    const GxStagePlan* plan = gx && g_binding_plans ? &stage_plan(meta, *gx) : nullptr;
    for (std::size_t i = 0; i < meta.buffers.size(); ++i) {
        const gcn::BufferBinding& b = meta.buffers[i];
        VkDescriptorBufferInfo info{g.dummy_ssbo, 0, VK_WHOLE_SIZE};
        std::uint32_t words[4] = {};
        // A pointer buffer reads a whole table, which only the commit builds.
        const bool resolved = b.pointer ? resolve_resource_impl(b.path, user, 2, words)
                                        : resolve_via_plan(plan ? &plan->buffers[i] : nullptr, b.path, user, gx, b.indexed ? 4 : 2, words);
        // A V# read by index is read anywhere in its records (count x stride),
        // in dwords; its stride and word 3 go into the params block.
        VkDeviceSize bytes = static_cast<VkDeviceSize>(b.max_dw) * 4;
        bool shape = true;
        if (b.indexed) {
            const std::uint32_t stride = (words[1] >> 16) & 0x3fff;
            shape = stride != 0 && stride % 4 == 0;
            // BBHOST_CB_RECORDS=1 (with BBHOST_ROBUST=1): bind exactly the V#'s
            // records, not the larger of that and the shader's highest dword.
            // GCN bounds-checks a buffer load against num_records and returns
            // zero past it; ours indexes an SSBO with no check, so an index
            // past the last record reads whatever follows the buffer - and the
            // game's allocator puts these transform buffers at a different
            // address on the frame that flashes, with the same contents. Only
            // what lies past them differs.
            static const bool exact_records = [] {
                const char* e = std::getenv("BBHOST_CB_RECORDS");
                return e && e[0] == '1';
            }();
            bytes = exact_records && words[2] ? static_cast<VkDeviceSize>(words[2]) * stride
                                              : std::max<VkDeviceSize>(bytes, static_cast<VkDeviceSize>(words[2]) * stride);
            params.cb_stride[i] = stride;
            params.cb_w3[i] = words[3];
        }
        // A V# keeps memory-type bits above its 40-bit address; a pointer pair is the address.
        const std::uint64_t hi = b.pointer ? words[1] : (words[1] & 0xff);
        const std::uint64_t base = resolved ? static_cast<std::uint64_t>(words[0]) | (hi << 32) : 0;
        int outcome = !base ? kNoBase : (base & 3) || !shape ? kUnaligned : kUnmapped;
        // Past the longest binding the range takes the page table: no one buffer need hold it.
        Located loc;
        // A copy-back still in its copy (copy versions): the copy, or the copies
        // put in place first when the binding covers part of one.
        int version = outcome == kUnmapped ? copy_version_lookup_locked(base, bytes, &loc) : kCopyVersionNone;
        if (version == kCopyVersionStraddles) {
            copy_versions_put_in_place_locked(kCvStraddle);
            version = kCopyVersionNone;
        }
        if (outcome == kUnmapped && version == kCopyVersionNone && (t_draw_windows.empty() || !draw_window_locked(base, bytes, loc))) {
            loc = locate(base, std::min<std::uint64_t>(bytes, g.ssbo_max_range ? g.ssbo_max_range : bytes));
        }
        if (g_import_audit && base) audit_range(base, bytes, loc.buffer && loc.avail >= bytes ? kUseBound : kUsePageTable);
        if (resolved && !base && bytes <= g.sink.size) {  // a null V#'s stride does not matter: every read lands in the sink page
            // A null table (no slots bound) or a null V# (an unbound constant
            // buffer, which vertex shaders without an inlined fetch shader now
            // bind) reads the sink page through the page table, so reading the
            // sink buffer is the same.
            info = {g.sink.buffer, 0, bytes};
            params.cb_valid |= 1u << i;
            params.cb_bias_dw[i] = 0;
            outcome = kBound;
        } else if (loc.buffer) {
            const VkDeviceSize start = loc.offset & ~(g.ssbo_align - 1);
            const VkDeviceSize bias = loc.offset - start;
            outcome = kTooLong;
            // BBHOST_CB_ALIGN16=1: refuse a binding whose base is not a whole
            // number of vec4s into the bound range, and let it take the page
            // table instead. A buffer read by index lands at bias + k*stride,
            // and a bias that is not a multiple of four dwords cannot be
            // expressed if the load is a vec4 at a vec4 index - it reads half
            // a record off. The menu flash is that: the same bytes at an
            // 8-byte-aligned address instead of a 16-byte one.
            static const bool align16 = [] {
                const char* e = std::getenv("BBHOST_CB_ALIGN16");
                return e && e[0] == '1';
            }();
            // BBHOST_CB_REALIGN=1: when the base is not on a vec4, copy the
            // records into a scratch ring and bind *that* at bias 0. The
            // earlier test refused the binding instead, which sent it to the
            // page-table walk - a different path that could share the fault,
            // so it proved nothing. This changes only the alignment.
            // On unless BBHOST_CB_REALIGN=0. =2 copies only the *already
            // aligned* buffers instead, which is how the two things this copy
            // does were told apart: it aligns the binding and it snapshots the
            // bytes, and mode 2 - snapshot without realignment - still flashes,
            // while mode 1 does not. The alignment is what matters.
            static const int realign_mode = [] {
                const char* e = std::getenv("BBHOST_CB_REALIGN");
                return e ? std::atoi(e) : 1;
            }();
            // =3 (checks): every constant buffer, aligned or not, copied as
            // the draw is recorded - a snapshot of the bytes, so a later CPU
            // write to that memory cannot reach a draw still in flight.
            bool realign = realign_mode == 1 ? (bias % 16) != 0 : realign_mode == 2 ? (bias % 16) == 0 : realign_mode == 3;
            if (t_snapshot_cbs && !realign) {
                realign = true;  // BBHOST_SF_SNAPSHOT: the bytes as recorded
                g_sf_cb_snapshots.fetch_add(1, std::memory_order_relaxed);
            }
            if (glitch_watching_draw()) {
                glitch_watch_read_locked(base, bytes);
                glitch_watch_note_locked("  buffer %zu: base 0x%llx (%llu mod 16) %llu bytes, stride %u, w3 %08x, bound at bias %llu%s%s\n", i,
                                         static_cast<unsigned long long>(base), static_cast<unsigned long long>(base & 15),
                                         static_cast<unsigned long long>(bytes), b.indexed ? params.cb_stride[i] : 0u, words[3],
                                         static_cast<unsigned long long>(bias), realign ? ", realigned" : "",
                                         b.pointer ? ", a pointer" : "");
                realign = realign || glitch_snapshot_draw();  // BBHOST_GLITCH_SNAPSHOT: the bytes as recorded
            }
            static DevBuffer scratch;
            static VkDeviceSize scratch_next = 0;
            constexpr VkDeviceSize kScratch = 64ull << 20;
            if (realign && bytes && bytes <= (1u << 20) && hle_kernel_va_mapped(base, bytes)) {
                if (!scratch.buffer) create_dev_buffer(scratch, kScratch, true);
                if (scratch.map) {
                    if (scratch_next + bytes + 256 > kScratch) scratch_next = 0;
                    const VkDeviceSize at = (scratch_next + 255) & ~VkDeviceSize{255};
                    std::memcpy(static_cast<std::uint8_t*>(scratch.map) + at,
                                reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base)), bytes);
                    copy_versions_overlay_locked(base, bytes, static_cast<std::uint8_t*>(scratch.map) + at);
                    scratch_next = at + bytes;
                    info = {scratch.buffer, at, bytes};
                    params.cb_valid |= 1u << i;
                    params.cb_bias_dw[i] = 0;
                    outcome = kBound;
                    g_cb_realigned.fetch_add(1, std::memory_order_relaxed);
                    g_cb_realigned_bytes.fetch_add(bytes, std::memory_order_relaxed);
                }
            }
            if (outcome == kBound) {
                // already handled by the realign path
            } else if (align16 && (bias % 16) != 0) {
                outcome = kUnaligned;
            } else if (bytes <= loc.avail && bias + bytes <= g.ssbo_max_range) {
                info = {loc.buffer, start, bias + bytes};
                params.cb_valid |= 1u << i;
                params.cb_bias_dw[i] = static_cast<std::uint32_t>(bias / 4);
                outcome = kBound;
                note_arena_read(kArenaConstants, base, bytes);
            }
        }
        if (outcome != kBound) {
            // The shader walks the page table over the V#'s range: its windows
            // imported, and any copy-back still in its copy put in place first.
            if (base && copy_version_lookup_locked(base, bytes, nullptr) != kCopyVersionNone) copy_versions_put_in_place_locked(kCvPageTable);
            if (base) import_windows(base, std::min<std::uint64_t>(bytes, 64ull << 20));
            // Logged once per binding and outcome, 32 at most. A binding the
            // stage's user data was not built for (the no-fallback variant does
            // not read it) fails here at every draw, so what was logged is
            // remembered by the binding, not by a string built each time.
            static std::mutex fail_mu;
            static std::set<std::string> fail_seen;
            static std::unordered_set<std::uint64_t> fail_known;  // under g.mu: binding address and outcome
            const std::uint64_t known_key = reinterpret_cast<std::uintptr_t>(&b) ^ static_cast<std::uint64_t>(outcome);
            const char* why = outcome == kNoBase ? "no base" : outcome == kUnaligned ? "unaligned" : outcome == kUnmapped ? "not imported" : "too long";
            const bool first = fail_known.size() < 4096 && fail_known.insert(known_key).second;
            std::unique_lock<std::mutex> lk(fail_mu, std::defer_lock);
            if (first) lk.lock();
            const std::string what = first ? b.path.str() + " " + why : std::string();
            if (first && fail_seen.size() < 32 && fail_seen.insert(what).second) {
                host_log("render: %s buffer %s: base 0x%llx, %llu bytes%s", b.pointer ? "pointer" : "V#", what.c_str(),
                         static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), b.indexed ? ", indexed" : "");
            }
        }
        ++outcomes[outcome];
        infos.push_back(info);
        if (++resolves % 200000 == 0) {
            host_log("render: storage-buffer constants: %llu bound; page-table path: %llu no base, %llu unaligned, %llu not imported, %llu too long",
                     static_cast<unsigned long long>(outcomes[kBound]), static_cast<unsigned long long>(outcomes[kNoBase]),
                     static_cast<unsigned long long>(outcomes[kUnaligned]), static_cast<unsigned long long>(outcomes[kUnmapped]),
                     static_cast<unsigned long long>(outcomes[kTooLong]));
            host_log("render:   YEBIS windows written by the CPU %llu, bound from their draw's copy %llu, read past their window %llu",
                     static_cast<unsigned long long>(g_windows_given.load()), static_cast<unsigned long long>(g_windows_bound.load()),
                     static_cast<unsigned long long>(g_windows_past.load()));
            host_log("render:   %llu of them copied to an aligned scratch (%llu MiB): a base that is not on a vec4 is read wrongly "
                     "through the binding, which is the menu flash; %llu of those a Scaleform draw's, copied as recorded "
                     "(BBHOST_SF_SNAPSHOT)",
                     static_cast<unsigned long long>(g_cb_realigned.load()),
                     static_cast<unsigned long long>(g_cb_realigned_bytes.load() >> 20),
                     static_cast<unsigned long long>(g_sf_cb_snapshots.load()));
        }
    }
}

void draw_inputs_from_registers(const GpuDraw& d, GpuDrawInputs& in) {
    const std::uint32_t* sh = d.sh;
    const std::uint32_t* cx = d.ctx;
    std::memcpy(in.vs_pgm, &sh[0x48], sizeof(in.vs_pgm));
    std::memcpy(in.ps_pgm, &sh[0x08], sizeof(in.ps_pgm));
    std::memcpy(in.vs_user, &sh[0x4c], sizeof(in.vs_user));
    std::memcpy(in.ps_user, &sh[0x0c], sizeof(in.ps_user));
    in.prim = d.uconfig[0x242];
    in.target_mask = cx[0x8E];
    in.cb_shader_mask = cx[0x8F];
    in.ps_col_format = cx[0x1C5];
    for (int t = 0; t < 8; ++t) {
        std::memcpy(in.color[t], cx + 0x318 + t * 0x0F, sizeof(in.color[t]));
        in.blend[t] = cx[0x1E0 + t];
    }
    std::memcpy(in.blend_const, &cx[0x105], sizeof(in.blend_const));
    in.depth_control = cx[0x200];
    std::memcpy(in.depth_bounds, &cx[0x08], sizeof(in.depth_bounds));
    in.stencil_control = cx[0x10B];
    in.stencil_ref = cx[0x10C];
    in.stencil_ref_bf = cx[0x10D];
    in.render_control = cx[0x0];
    in.depth_clear = cx[0x0B];
    in.stencil_clear = cx[0x0A];
    in.z_info = cx[0x10];
    in.stencil_info = cx[0x11];
    in.z_read_base = cx[0x12];
    in.depth_size = cx[0x16];
    in.htile_base = cx[0x05];
    in.su_sc_mode = cx[0x205];
    in.clip_cntl = cx[0x204];
    in.vte_cntl = cx[0x206];
    in.ps_input_ena = cx[0x1B3];
    in.ps_in_control = cx[0x1B6];
    in.vs_out_cntl = cx[0x207];
    std::memcpy(in.ps_input_cntl, &cx[0x191], sizeof(in.ps_input_cntl));
    std::memcpy(in.vport, &cx[0x10F], sizeof(in.vport));
    in.screen_scissor[0] = cx[0x0C];
    in.screen_scissor[1] = cx[0x0D];
    in.generic_scissor[0] = cx[0x90];
    in.generic_scissor[1] = cx[0x91];
    in.vport_scissor[0] = cx[0x94];
    in.vport_scissor[1] = cx[0x95];
    in.sc_mode_cntl_0 = cx[0x292];
    in.base_vertex = static_cast<std::int32_t>(cx[0x102]);
}

// The scissor rectangle a draw renders with: the generic scissor intersected
// with the screen scissor (a zero bottom-right corner on one leaves the
// other), or everything when that is empty. With `vport`, viewport scissor 0
// is intersected too while PA_SC_MODE_CNTL_0 enables it (GX sets its
// WINDOW_OFFSET_DISABLE bit); that intersection may be empty.
struct DrawScissor {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};
DrawScissor draw_scissor(const GpuDrawInputs& in, bool vport) {
    const auto tl = [](std::uint32_t v) { return std::make_pair(static_cast<int>(v & 0x7fff), static_cast<int>((v >> 16) & 0x7fff)); };
    const auto [sx0, sy0] = tl(in.screen_scissor[0]);
    const auto [sx1, sy1] = tl(in.screen_scissor[1]);
    const auto [gx0, gy0] = tl(in.generic_scissor[0]);
    const auto [gx1, gy1] = tl(in.generic_scissor[1]);
    DrawScissor s;
    s.x0 = std::max(sx0, gx0);
    s.y0 = std::max(sy0, gy0);
    s.x1 = sx1 && gx1 ? std::min(sx1, gx1) : std::max(sx1, gx1);
    s.y1 = sy1 && gy1 ? std::min(sy1, gy1) : std::max(sy1, gy1);
    if (s.x1 <= s.x0 || s.y1 <= s.y0) s = {0, 0, 16384, 16384};
    if (vport && (in.sc_mode_cntl_0 & 2)) {
        const auto [vx0, vy0] = tl(in.vport_scissor[0]);
        const auto [vx1, vy1] = tl(in.vport_scissor[1]);
        s.x0 = std::max(s.x0, vx0);
        s.y0 = std::max(s.y0, vy0);
        s.x1 = std::max(s.x0, std::min(s.x1, vx1));
        s.y1 = std::max(s.y0, std::min(s.y1, vy1));
    }
    return s;
}

// BBHOST_GX_BACKEND=2: the inputs the GX backend built for a draw against the
// ones from the register file, counting only what the decode would use:
// targets outside CB_TARGET_MASK, depth-target fields without a depth target,
// stencil values with the test off, PS input entries past the PS's inputs,
// the viewport scissor while disabled and clear values without a clear are
// not compared. The generic scissor counts through the rectangle it makes.
// User data, which the GX side still copies from the registers, is skipped.
struct InputCompare {
    std::map<std::string, std::uint64_t> fields;
    std::uint64_t draws = 0, differing = 0;
    // How often the conditional comparisons ran: draws with PS input
    // entries, with the viewport scissor enabled, with a clear bit, and
    // whose register-file generic scissor narrows the screen scissor.
    std::uint64_t ps_inputs = 0, vport_scissor = 0, clears = 0, generic_narrows = 0;
};
InputCompare g_input_compare;  // under g.mu

bool target_used(const GpuDrawInputs& in, int t) {
    return ((in.target_mask >> (4 * t)) & 0xf) && in.color[t][0] && ((in.color[t][4] >> 2) & 0x1f);
}

void compare_draw_inputs(const GpuDrawInputs& r, const GpuDrawInputs& x) {
    InputCompare& c = g_input_compare;
    ++c.draws;
    bool any = false;
    const auto diff = [&](const std::string& name, std::uint64_t a, std::uint64_t b) {
        if (a == b) return;
        any = true;
        if (++c.fields[name] <= 3) {
            host_log("gx-compare: %s: registers 0x%llx, GX 0x%llx", name.c_str(), static_cast<unsigned long long>(a),
                     static_cast<unsigned long long>(b));
        }
    };
    const auto diff_f = [&](const std::string& name, float a, float b) {
        if (std::fabs(a - b) <= 0.01f || (std::isnan(a) && std::isnan(b))) return;
        any = true;
        if (++c.fields[name] <= 3) host_log("gx-compare: %s: registers %g, GX %g", name.c_str(), a, b);
    };
    static const char* const kPgm[4] = {"pgm_lo", "pgm_hi", "rsrc1", "rsrc2"};
    for (int i = 0; i < 4; ++i) {
        diff(std::string("vs ") + kPgm[i], r.vs_pgm[i], x.vs_pgm[i]);
        diff(std::string("ps ") + kPgm[i], r.ps_pgm[i], x.ps_pgm[i]);
    }
    diff("prim", r.prim, x.prim);
    diff("cb_shader_mask", r.cb_shader_mask, x.cb_shader_mask);
    for (int t = 0; t < 8; ++t) {
        const bool ur = target_used(r, t), ux = target_used(x, t);
        const std::string rt = "rt" + std::to_string(t) + " ";
        diff(rt + "used", ur, ux);
        if (ur != ux) {
            static int details = 0;
            if (details++ < 6) {
                host_log("gx-compare: rt%d registers mask %x base %08x info %08x, GX mask %x base %08x info %08x; PS writes %x",
                         t, (r.target_mask >> (4 * t)) & 0xf, r.color[t][0], r.color[t][4], (x.target_mask >> (4 * t)) & 0xf,
                         x.color[t][0], x.color[t][4], (r.cb_shader_mask >> (4 * t)) & 0xf);
            }
        }
        if (!ur || !ux) continue;
        diff(rt + "mask", (r.target_mask >> (4 * t)) & 0xf, (x.target_mask >> (4 * t)) & 0xf);
        diff(rt + "base", r.color[t][0], x.color[t][0]);
        diff(rt + "pitch", r.color[t][1] & 0x7ff, x.color[t][1] & 0x7ff);
        diff(rt + "slice", r.color[t][2] & 0x3fffff, x.color[t][2] & 0x3fffff);
        diff(rt + "info", r.color[t][4], x.color[t][4]);
        diff(rt + "blend", r.blend[t], x.blend[t]);
    }
    for (int k = 0; k < 4; ++k) diff_f("blend constant " + std::to_string(k), r.blend_const[k], x.blend_const[k]);
    diff("depth_control", r.depth_control, x.depth_control);
    if (r.depth_control & 1) {
        diff("stencil_control", r.stencil_control, x.stencil_control);
        diff("stencil_ref", r.stencil_ref, x.stencil_ref);
        diff("stencil_ref_bf", r.stencil_ref_bf, x.stencil_ref_bf);
    }
    if (r.depth_control & 8) {
        diff_f("depth bounds min", r.depth_bounds[0], x.depth_bounds[0]);
        diff_f("depth bounds max", r.depth_bounds[1], x.depth_bounds[1]);
    }
    const auto depth_used = [](const GpuDrawInputs& in) {
        return ((in.depth_control & 3) || (in.render_control & 3)) && in.z_read_base;
    };
    const bool dr = depth_used(r), dx = depth_used(x);
    diff("depth used", dr, dx);
    if (dr && dx) {
        diff("z_info", r.z_info, x.z_info);
        diff("stencil_info", r.stencil_info, x.stencil_info);
        diff("z_read_base", r.z_read_base, x.z_read_base);
        diff("depth_size", r.depth_size, x.depth_size);
        diff("htile_base", r.htile_base, x.htile_base);
    }
    diff("su_sc_mode", r.su_sc_mode, x.su_sc_mode);
    diff("clip_cntl", r.clip_cntl, x.clip_cntl);
    diff("vte_cntl", r.vte_cntl, x.vte_cntl);
    diff("ps_input_ena", r.ps_input_ena, x.ps_input_ena);
    diff("ps_in_control", r.ps_in_control, x.ps_in_control);
    diff("vs_out_cntl", r.vs_out_cntl, x.vs_out_cntl);
    for (std::uint32_t k = 0; k < x.ps_input_count && k < 32; ++k) {
        if (r.ps_input_cntl[k] != x.ps_input_cntl[k]) {
            diff("ps_input_cntl " + std::to_string(k), r.ps_input_cntl[k], x.ps_input_cntl[k]);
            break;
        }
    }
    diff("sc_mode_cntl_0", r.sc_mode_cntl_0 & 3, x.sc_mode_cntl_0 & 3);
    if ((r.sc_mode_cntl_0 & 2) && (x.sc_mode_cntl_0 & 2)) {
        diff("vport scissor tl", r.vport_scissor[0], x.vport_scissor[0]);
        diff("vport scissor br", r.vport_scissor[1], x.vport_scissor[1]);
    }
    // No GX draw flush writes DB_RENDER_CONTROL or the clear values.
    diff("render_control", r.render_control, x.render_control);
    if ((r.render_control & 3) || (x.render_control & 3)) {
        diff("depth_clear", r.depth_clear, x.depth_clear);
        diff("stencil_clear", r.stencil_clear & 0xff, x.stencil_clear & 0xff);
    }
    if (x.ps_input_count) ++c.ps_inputs;
    if ((r.sc_mode_cntl_0 & 2) || (x.sc_mode_cntl_0 & 2)) ++c.vport_scissor;
    if ((r.render_control & 3) || (x.render_control & 3)) ++c.clears;
    const DrawScissor sr = draw_scissor(r, true), sx = draw_scissor(x, true);
    {
        GpuDrawInputs screen_only;
        screen_only.screen_scissor[0] = r.screen_scissor[0];
        screen_only.screen_scissor[1] = r.screen_scissor[1];
        const DrawScissor so = draw_scissor(screen_only, false), sg = draw_scissor(r, false);
        if (so.x0 != sg.x0 || so.y0 != sg.y0 || so.x1 != sg.x1 || so.y1 != sg.y1) ++c.generic_narrows;
    }
    diff("scissor x0", static_cast<std::uint32_t>(sr.x0), static_cast<std::uint32_t>(sx.x0));
    diff("scissor y0", static_cast<std::uint32_t>(sr.y0), static_cast<std::uint32_t>(sx.y0));
    diff("scissor x1", static_cast<std::uint32_t>(sr.x1), static_cast<std::uint32_t>(sx.x1));
    diff("scissor y1", static_cast<std::uint32_t>(sr.y1), static_cast<std::uint32_t>(sx.y1));
    if ((r.vte_cntl & 1) || r.vte_cntl == 0) {
        static const char* const kVport[6] = {"vport xscale", "vport xoffset", "vport yscale",
                                              "vport yoffset", "vport zscale",  "vport zoffset"};
        for (int k = 0; k < 6; ++k) diff_f(kVport[k], r.vport[k], x.vport[k]);
    }
    diff("screen scissor tl", r.screen_scissor[0], x.screen_scissor[0]);
    diff("screen scissor br", r.screen_scissor[1], x.screen_scissor[1]);
    diff("base_vertex", static_cast<std::uint32_t>(r.base_vertex), static_cast<std::uint32_t>(x.base_vertex));
    if (any) ++c.differing;
    if (c.draws % 20000 == 0) {
        std::vector<std::pair<std::string, std::uint64_t>> v(c.fields.begin(), c.fields.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string s;
        for (std::size_t i = 0; i < v.size() && i < 16; ++i) s += " " + v[i].first + ":" + std::to_string(v[i].second);
        host_log("gx-compare: %llu draws compared, %llu with differences;%s", static_cast<unsigned long long>(c.draws),
                 static_cast<unsigned long long>(c.differing), s.c_str());
        host_log("gx-compare: draws with PS input entries %llu, viewport scissor enabled %llu, a clear bit %llu, "
                 "a narrowing generic scissor %llu",
                 static_cast<unsigned long long>(c.ps_inputs), static_cast<unsigned long long>(c.vport_scissor),
                 static_cast<unsigned long long>(c.clears), static_cast<unsigned long long>(c.generic_narrows));
    }
}

// BBHOST_GX_BACKEND=2: a GX draw's geometry from the call
// against its draw packet, and the input layout's fetch shader against the one
// the VS user data binds.
struct GeometryCompare {
    std::uint64_t draws = 0, count = 0, instances = 0, indexed = 0, index_va = 0, index_type = 0, index_unknown = 0;
    std::uint64_t fetch_compared = 0, fetch = 0;
};
GeometryCompare g_geometry_compare;  // under g.mu

void compare_gx_geometry(const GpuDraw& d, const GxDrawObjects& o) {
    GeometryCompare& c = g_geometry_compare;
    ++c.draws;
    const auto bad = [](std::uint64_t& n, const char* what, std::uint64_t packet, std::uint64_t gx) {
        if (n++ < 3) {
            host_log("gx-compare: geometry %s: packet 0x%llx, GX 0x%llx", what, static_cast<unsigned long long>(packet),
                     static_cast<unsigned long long>(gx));
        }
    };
    if (d.index_count != o.count) bad(c.count, "count", d.index_count, o.count);
    if (d.instance_count != o.instances) bad(c.instances, "instances", d.instance_count, o.instances);
    if ((d.index_va != 0) != o.indexed) {
        bad(c.indexed, "indexed", d.index_va != 0, o.indexed);
    } else if (o.indexed && !o.index_known) {
        ++c.index_unknown;
    } else if (o.indexed) {
        if (d.index_va != o.index_va) bad(c.index_va, "first index address", d.index_va, o.index_va);
        if (d.index_type != o.index_type) bad(c.index_type, "index type", d.index_type, o.index_type);
    }
    if (c.draws % 20000 == 0) {
        host_log("gx-compare: geometry of %llu draws: differing count %llu, instances %llu, indexed %llu, first index "
                 "address %llu, index type %llu; renamed index buffers %llu; fetch shaders compared %llu, differing %llu",
                 static_cast<unsigned long long>(c.draws), static_cast<unsigned long long>(c.count),
                 static_cast<unsigned long long>(c.instances), static_cast<unsigned long long>(c.indexed),
                 static_cast<unsigned long long>(c.index_va), static_cast<unsigned long long>(c.index_type),
                 static_cast<unsigned long long>(c.index_unknown), static_cast<unsigned long long>(c.fetch_compared),
                 static_cast<unsigned long long>(c.fetch));
    }
}

void compare_gx_fetch(std::uint64_t user_fetch, const GxDrawObjects& o) {
    GeometryCompare& c = g_geometry_compare;
    ++c.fetch_compared;
    if (user_fetch != o.fetch_va && c.fetch++ < 3) {
        host_log("gx-compare: fetch shader: VS user data 0x%llx, input layout 0x%llx", static_cast<unsigned long long>(user_fetch),
                 static_cast<unsigned long long>(o.fetch_va));
    }
}

// Reserves bytes in the ring (8-byte aligned), copies them there and returns
// their address; 0 when no ring could be mapped. Tables come to a few MiB a
// frame at most, so the 128 MiB ring does not reuse a copy before the queued
// work reading it has run.
std::uint64_t gx_ring_place(const std::uint8_t* data, std::uint64_t bytes, std::uint64_t align = 8) {
    GxRing& ring = g_gx_ring;
    std::uint64_t va = 0;
    {
        std::lock_guard<std::mutex> lk(ring.mu);
        if (!ring.base && !ring.failed) {
            ring.size = 128ull << 20;
            ring.base = hle_kernel_map_host(ring.size);
            ring.failed = ring.base == 0;
            if (ring.failed) host_log("render: no host mapping for GX tables; they stay on the command stream");
        }
        if (!ring.base || !bytes || bytes > ring.size) return 0;
        ring.cursor = (ring.cursor + align - 1) & ~(align - 1);
        if (ring.cursor + bytes > ring.size) ring.cursor = 0;
        va = ring.base + ring.cursor;
        ring.cursor += bytes;
    }
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va)), data, static_cast<std::size_t>(bytes));
    return va;
}

// The renderer's placements (under g.mu): a ring mapped since the last page
// table rebuild enters the tables before anything in it is bound.
std::uint64_t ring_place(const std::uint8_t* data, std::uint64_t bytes, std::uint64_t align = 8) {
    const std::uint64_t va = gx_ring_place(data, bytes, align);
    return va && rebuild_page_tables() ? va : 0;
}

// A GX draw's vertex table (slots 0 to the highest one built, zeroes where
// none was) in the ring; 0 when the draw has none.
std::uint64_t place_vertex_table(const GxDrawObjects& o) {
    if (o.vtx_ud >= 15 || !o.vtx_valid) return 0;
    std::uint32_t slots = 16;
    while (slots && !((o.vtx_valid >> (slots - 1)) & 1)) --slots;
    std::uint8_t table[16 * 16] = {};
    for (std::uint32_t k = 0; k < slots; ++k) {
        if ((o.vtx_valid >> k) & 1) std::memcpy(table + 16 * k, o.vtx_rec[k], 16);
    }
    const std::uint64_t va = ring_place(table, static_cast<std::uint64_t>(slots) * 16);
    if (va) g_vertex_tables.fetch_add(1, std::memory_order_relaxed);
    return va;
}

// BBHOST_LOG_FETCH=1: each distinct fetch shader, with the
// formats and strides of the vertex table built at the call, logged once, to
// see which shapes the GX input layouts generate before replacing them with
// Vulkan vertex input. Under g.mu.
void log_fetch_shader(std::uint64_t fetch_hash, const std::vector<std::uint32_t>& words, const GxDrawObjects& o) {
    std::uint64_t key = fetch_hash;
    for (int k = 0; k < 16; ++k) {
        if (!((o.vtx_valid >> k) & 1)) continue;
        const std::uint64_t v = o.vtx_rec[k][3] | (static_cast<std::uint64_t>((o.vtx_rec[k][1] >> 16) & 0x3fff) << 32) |
                                (static_cast<std::uint64_t>(k) << 48);
        key = (key ^ v) * 1099511628211ull;
    }
    static std::set<std::uint64_t> seen;
    if (seen.size() >= 400 || !seen.insert(key).second) return;
    host_log("fetch-survey: fetch %016llx at 0x%llx, table in s[%u], slots %04x, instances %u, %s", static_cast<unsigned long long>(fetch_hash),
             static_cast<unsigned long long>(o.fetch_va), o.vtx_ud, o.vtx_valid, o.instances, o.indexed ? "indexed" : "not indexed");
    for (int k = 0; k < 16; ++k) {
        if (!((o.vtx_valid >> k) & 1)) continue;
        const std::uint32_t* w = o.vtx_rec[k];
        host_log("fetch-survey:   V#%d stride=%u records=%u dfmt=%u nfmt=%u dst_sel=%03x (%08x %08x %08x %08x)", k, (w[1] >> 16) & 0x3fff, w[2],
                 (w[3] >> 15) & 0xf, (w[3] >> 12) & 7, w[3] & 0xfff, w[0], w[1], w[2], w[3]);
    }
    const gcn::Program fp = gcn::decode(words.data(), words.size());
    for (const gcn::Inst& in : fp.insts) host_log("fetch-survey:   %s", gcn::format(in).c_str());
}

// A fetch shader in the shape the GX input layouts generate (all 28 in the
// world window): each element's V# record
// loaded from the vertex table (s_load_dwordx4 from the table pair at a
// constant dword offset, 4 dwords a slot), then buffer_load_format_* into
// VGPRs indexed by v0 at a constant offset, then the return. Anything else
// keeps the fetch shader.
// `stop`, when given: where and why the parse stopped, for the log.
bool parse_fetch_elements(const std::vector<std::uint32_t>& words, std::uint32_t table_ud, std::vector<FetchElement>& out,
                          std::string* stop = nullptr) {
    out.clear();
    const auto fail = [&](const gcn::Inst* in, const char* why) {
        if (stop) {
            char b[160];
            if (in) {
                std::snprintf(b, sizeof(b), "%s (%s; src0 s[%u] sdst %u vaddr v%u srsrc s[%u] idxen %d offen %d soffset %u)", gcn::mnemonic(*in), why,
                              in->src0, in->sdst, in->vaddr, in->srsrc, in->idxen ? 1 : 0, in->offen ? 1 : 0, in->soffset);
            } else {
                std::snprintf(b, sizeof(b), "%s", why);
            }
            *stop = b;
        }
        return false;
    };
    const gcn::Program p = gcn::decode(words.data(), words.size());
    if (!p.errors.empty()) return fail(nullptr, "decode error");
    std::map<std::uint16_t, std::uint8_t> record_at;  // SGPR base of a loaded record -> its table slot
    bool returned = false;
    for (const gcn::Inst& in : p.insts) {
        if (returned) return fail(&in, "after the return");
        const char* name = gcn::mnemonic(in);
        if (in.enc == gcn::Enc::SMRD && !std::strcmp(name, "s_load_dwordx4")) {
            const std::uint32_t dwords = in.imm_flag ? static_cast<std::uint32_t>(in.imm) : in.has_literal ? in.literal : ~0u;
            if (in.src0 != table_ud || dwords == ~0u || dwords % 4 || dwords / 4 >= 16 || in.sdst + 4 > 104) return fail(&in, "record load");
            record_at[in.sdst] = static_cast<std::uint8_t>(dwords / 4);
        } else if (!std::strcmp(name, "s_waitcnt")) {
            // the loads above land here
        } else if (in.enc == gcn::Enc::MUBUF && in.op <= 3) {  // buffer_load_format_x .. _xyzw
            const auto it = record_at.find(in.srsrc);
            if (it == record_at.end() || in.vaddr != 0 || !in.idxen || in.offen || in.addr64 || in.soffset < 128 || in.soffset > 192) {
                return fail(&in, "element load");
            }
            out.push_back({it->second, static_cast<std::uint8_t>(in.vdata), static_cast<std::uint8_t>(in.op + 1),
                           static_cast<std::uint32_t>(in.offset12) + (in.soffset - 128u)});
        } else if (in.enc == gcn::Enc::SOP1 && in.op == 32) {  // s_setpc_b64: return
            returned = true;
        } else {
            return fail(&in, "another instruction");
        }
    }
    if (!returned) return fail(nullptr, "no return");
    if (out.empty() || out.size() > 16) return fail(nullptr, "no elements or more than 16");
    return true;
}

bool vertex_format_usable(VkFormat f) {
    static std::int8_t core[256];  // under g.mu: 0 not asked yet, 1 usable, -1 not
    static std::unordered_map<int, bool> usable;
    const bool is_core = static_cast<std::uint32_t>(f) < 256;
    if (is_core && core[f]) return core[f] > 0;
    if (!is_core) {
        const auto it = usable.find(f);
        if (it != usable.end()) return it->second;
    }
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(g.phys, f, &fp);
    const bool ok = (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0;
    if (is_core) core[f] = ok ? 1 : -1;
    else usable[f] = ok;
    return ok;
}

// A draw whose vertex table the game binds itself (0x73e670's: the commit's
// vertex builder left no slot, o.vtx_ud 0xff): its fetch shader loads every
// record from one user-data pair, so the records are read from that table
// now, as the fetch shader would read them on the GPU, and the draw can take
// Vulkan vertex input like the rest instead of walking the page table for
// every vertex. False when the fetch shader loads from more than one pair or
// a record is unreadable. BBHOST_VERTEX_TABLE_FROM_MEMORY=1 turns it on: it
// moved ~174k draws a run onto vertex input but paid for nothing - the Steam
// Deck at the spawn within noise (GPU 16.80/16.73 against 16.76/16.69 ms a
// frame, windowed 57.1/57.1 against 57.0/57.3 fps), and on the RTX 4070 two of
// those pipelines got slower (72 -> 183 and 64 -> 97 us a draw: their vertex
// data stays in host memory, which its fixed-function fetch reads over PCIe).
const bool g_vertex_table_from_memory = [] {
    const char* e = std::getenv("BBHOST_VERTEX_TABLE_FROM_MEMORY");
    return e && e[0] == '1';
}();
bool vertex_table_from_memory(const std::vector<std::uint32_t>& fetch_words, std::uint64_t fetch_hash, const std::uint32_t* vs_user,
                              const GxDrawObjects& o, GxDrawObjects& out) {
    struct Loads {
        int ud = -1;          // the user-data pair every record load reads from; -1 none or several
        std::uint16_t slots = 0;  // the table slots loaded
    };
    static std::unordered_map<std::uint64_t, Loads> by_fetch;  // under g.mu; never erased
    auto it = by_fetch.find(fetch_hash);
    if (it == by_fetch.end()) {
        Loads l;
        const gcn::Program p = gcn::decode(fetch_words.data(), fetch_words.size());
        bool ok = p.errors.empty();
        for (const gcn::Inst& in : p.insts) {
            if (!ok || in.enc != gcn::Enc::SMRD || std::strcmp(gcn::mnemonic(in), "s_load_dwordx4") != 0) continue;
            const std::uint32_t dwords = in.imm_flag ? static_cast<std::uint32_t>(in.imm) : in.has_literal ? in.literal : ~0u;
            if (in.src0 >= 15 || (l.ud >= 0 && l.ud != static_cast<int>(in.src0)) || dwords == ~0u || dwords % 4 || dwords / 4 >= 16) {
                ok = false;
                continue;
            }
            l.ud = static_cast<int>(in.src0);
            l.slots |= static_cast<std::uint16_t>(1u << (dwords / 4));
        }
        if (!ok) l = Loads{};
        it = by_fetch.emplace(fetch_hash, l).first;
    }
    const Loads& l = it->second;
    if (l.ud < 0 || !l.slots) return false;
    const std::uint64_t table = (static_cast<std::uint64_t>(vs_user[l.ud]) | (static_cast<std::uint64_t>(vs_user[l.ud + 1]) << 32)) & ((1ull << 48) - 1);
    if (!table) return false;
    out = o;
    out.vtx_ud = static_cast<std::uint8_t>(l.ud);
    out.vtx_valid = 0;
    for (std::uint32_t slot = 0; slot < 16; ++slot) {
        if (!((l.slots >> slot) & 1)) continue;
        const std::uint64_t at = table + 16ull * slot;
        if (!hle_kernel_va_mapped(at, 16)) return false;
        read_guest_locked(at, 16, out.vtx_rec[slot]);
        out.vtx_valid |= static_cast<std::uint16_t>(1u << slot);
    }
    g_vertex_tables_read.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Builds the plan from the draw's fetch shader and the vertex table built at
// the call. False when the fetch shader has another shape, a record is
// missing, or an element's data does not lie in one imported buffer (a
// record's count times its stride); the draw then runs the fetch shader.
// Under g.mu.
// `memo` (the input layout's by-id entry, or null) keeps the parse for the
// layout: the map lookup then happens once per layout, not once per draw.
bool plan_vertex_input(const std::vector<std::uint32_t>& fetch_words, std::uint64_t fetch_hash, const GxDrawObjects& o,
                       VertexInputPlan& p, FetchById* memo = nullptr) {
    auto keep = [&](const char* why) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 16) {
            host_log("render: fetch shader %016llx at 0x%llx kept: %s", static_cast<unsigned long long>(fetch_hash),
                     static_cast<unsigned long long>(o.fetch_va), why);
        }
        return false;
    };
    static std::unordered_map<std::uint64_t, ParsedFetch> parsed;  // by fetch shader and table slot; never erased
    const ParsedFetch* pf = memo && memo->parsed && memo->vtx_ud == o.vtx_ud ? memo->parsed : nullptr;
    if (!pf) {
        const std::uint64_t pkey = fnv1a(&o.vtx_ud, sizeof(o.vtx_ud), fetch_hash);
        auto it = parsed.find(pkey);
        if (it == parsed.end()) {
            ParsedFetch r;
            std::string stop;
            r.ok = o.vtx_ud < 15 && parse_fetch_elements(fetch_words, o.vtx_ud, r.elements, &stop);
            it = parsed.emplace(pkey, std::move(r)).first;
            if (!it->second.ok) {
                static std::atomic<int> shapes{0};
                if (shapes.fetch_add(1) < 24) {
                    std::string hex;
                    for (std::size_t k = 0; k < fetch_words.size() && k < 32; ++k) {
                        char b[12];
                        std::snprintf(b, sizeof(b), " %08x", fetch_words[k]);
                        hex += b;
                    }
                    host_log("render: fetch shader %016llx (%zu dwords, table at s[%u], caller 0x%llx) is not the GX input-layout shape: %s;%s",
                             static_cast<unsigned long long>(fetch_hash), fetch_words.size(), o.vtx_ud, static_cast<unsigned long long>(o.caller),
                             stop.empty() ? "no table slot" : stop.c_str(), hex.c_str());
                }
                return keep("not the GX input-layout shape");
            }
        }
        pf = &it->second;
        if (memo) {
            memo->parsed = pf;
            memo->vtx_ud = o.vtx_ud;
        }
    }
    if (!pf->ok) return false;
    const std::vector<FetchElement>& els = pf->elements;
    const std::size_t n = els.size();
    struct Placed {
        VkBuffer buffer;
        VkDeviceSize offset;
        std::uint32_t stride;
        VkFormat format;
        std::uint64_t va;
    } placed[16];
    p.elements.clear();
    // The device-local copy of each vertex record, once per record: its
    // elements share it (buffer_shadow.cpp).
    struct RecordShadow {
        const std::uint32_t* w;
        Located whole;
    } shadows[16];
    std::size_t n_shadows = 0;
    for (std::size_t e = 0; e < n; ++e) {
        const FetchElement& fe = els[e];
        if (!((o.vtx_valid >> fe.slot) & 1)) return keep("the vertex table has no record for an element");
        const std::uint32_t* w = o.vtx_rec[fe.slot];
        const std::uint32_t dfmt = (w[3] >> 15) & 0xf, stride = (w[1] >> 16) & 0x3fff, bytes = vertex_format_bytes(dfmt);
        const VkFormat fmt = vertex_format(dfmt);
        if (fmt == VK_FORMAT_UNDEFINED || !vertex_format_usable(fmt)) return keep("a data format with no vertex format");
        if (stride < bytes) return keep("a stride shorter than its element");
        const std::uint64_t va = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) + fe.offset;
        const std::uint64_t need = w[2] ? static_cast<std::uint64_t>(w[2] - 1) * stride + bytes : bytes;
        Located loc;
        const int version = copy_version_lookup_locked(va, need, &loc);
        if (version == kCopyVersionStraddles) copy_versions_put_in_place_locked(kCvStraddle);
        if (version != kCopyVersionInside) loc = locate(va, need);
        if (g_import_audit) audit_range(va, need, kUseMirror);
        if (!loc.buffer || loc.avail < need) return keep("vertex data not within one imported buffer");
        // Read from device memory when the record's whole extent can be kept
        // there (buffer_shadow.cpp); every element of the record shares it.
        const RecordShadow* rs = nullptr;
        for (std::size_t k = 0; k < n_shadows; ++k) {
            if (std::memcmp(shadows[k].w, w, 12) == 0) rs = &shadows[k];
        }
        if (!rs) {
            const std::uint64_t record_va = va - fe.offset;
            const std::uint64_t record_bytes = w[2] ? static_cast<std::uint64_t>(w[2]) * stride : fe.offset + bytes;
            RecordShadow& ns = shadows[n_shadows++];
            ns.w = w;
            ns.whole = {};
            if (loc.offset >= fe.offset && loc.avail + fe.offset >= record_bytes) {
                ns.whole = {loc.buffer, loc.offset - fe.offset, record_bytes};
                shadow_locate_locked(record_va, record_bytes, ns.whole);
                if (ns.whole.buffer == loc.buffer) ns.whole = {};  // stays in host memory
            }
            rs = &ns;
        }
        if (rs->whole.buffer) {
            loc.buffer = rs->whole.buffer;
            loc.offset = rs->whole.offset + fe.offset;
        }
        gcn::VertexElement ve;
        ve.location = static_cast<std::uint32_t>(e);
        ve.vdata = fe.vdata;
        ve.count = fe.count;
        ve.w3 = w[3];
        p.elements.push_back(ve);
        placed[e] = {loc.buffer, loc.offset, stride, fmt, va};
    }
    // Every element reads address + v0 * stride, so elements in one buffer
    // with one stride can share a binding at their lowest offset with the rest
    // as attribute offsets, as a D3D11 stream holds its elements at
    // AlignedByteOffset. Offsets past what every device allows (2047) get
    // their own bindings.
    auto bind = [&](bool per_element) {
        p.bindings.clear();
        p.attributes.clear();
        std::uint32_t binding_of[16];
        for (std::size_t e = 0; e < n; ++e) {
            binding_of[e] = UINT32_MAX;
            for (std::uint32_t k = 0; !per_element && k < p.bindings.size(); ++k) {
                if (p.buffer[k] == placed[e].buffer && p.bindings[k].stride == placed[e].stride) binding_of[e] = k;
            }
            if (binding_of[e] == UINT32_MAX) {
                binding_of[e] = static_cast<std::uint32_t>(p.bindings.size());
                p.bindings.push_back({binding_of[e], placed[e].stride, VK_VERTEX_INPUT_RATE_VERTEX});
                p.buffer[binding_of[e]] = placed[e].buffer;
                p.offset[binding_of[e]] = placed[e].offset;
                p.va[binding_of[e]] = placed[e].va;
            } else if (placed[e].offset < p.offset[binding_of[e]]) {
                p.offset[binding_of[e]] = placed[e].offset;
                p.va[binding_of[e]] = placed[e].va;
            }
        }
        for (std::size_t e = 0; e < n; ++e) {
            const VkDeviceSize rel = placed[e].offset - p.offset[binding_of[e]];
            if (rel > 2047) return false;
            p.attributes.push_back({static_cast<std::uint32_t>(e), binding_of[e], placed[e].format, static_cast<std::uint32_t>(rel)});
        }
        return true;
    };
    if (!bind(g_vertex_input == 2)) bind(true);
    std::uint64_t h = fnv1a("vertex input", 12);
    h = fnv1a(p.elements.data(), p.elements.size() * sizeof(gcn::VertexElement), h);
    h = fnv1a(p.bindings.data(), p.bindings.size() * sizeof(VkVertexInputBindingDescription), h);
    h = fnv1a(p.attributes.data(), p.attributes.size() * sizeof(VkVertexInputAttributeDescription), h);
    p.hash = h;
    return true;
}

// The record a resource slot's pending-state object gives for a descriptor
// type (the object-built records, GxStageRecords obj_*); null without one.
// A tessellated draw run by the game's own hull (TessDraw::hull): the V# of
// the tessellation constants the host built, standing in for constant buffer
// 0x13 - which no GX object fills - while that draw's user data is built.
// The hull and domain shaders ask for it as a direct record (type 5) or in
// the extended block (type 0x12, the Moonside Lake's water).
thread_local const std::uint32_t* t_tess_constants = nullptr;

const std::uint8_t* gx_object_record(const GxStageRecords& st, std::uint32_t type, std::uint32_t slot) {
    if (t_tess_constants && slot == 0x13 && (type == 0x5 || type == 0x12 || type == 0xb)) {
        return reinterpret_cast<const std::uint8_t*>(t_tess_constants);
    }
    switch (type) {
        case 0x0: case 0xd: case 0x1: case 0xe: case 0x8:
            return slot < 64 && ((st.obj_tex_set >> slot) & 1) ? st.obj_tex + static_cast<std::size_t>(slot) * 32 : nullptr;
        case 0x2: case 0xf: case 0x3: case 0x10: case 0x9:
            return slot < 16 && ((st.obj_smp_set >> slot) & 1) ? st.obj_smp + static_cast<std::size_t>(slot) * 32 : nullptr;
        case 0x4: case 0x11: case 0xa:
            return slot < 16 && (((st.obj_smp_set | st.obj_smp_default) >> slot) & 1) ? st.obj_smp + static_cast<std::size_t>(slot) * 32 : nullptr;
        case 0x5: case 0x12: case 0xb:
            return slot < 14 && ((st.obj_cb_set >> slot) & 1) ? st.obj_cb + static_cast<std::size_t>(slot) * 16 : nullptr;
        default:
            return nullptr;
    }
}

// A GX draw stage's user data built from its records, as the
// commit (0x2abc320) writes it through 0x14761a0 / 0x1476240 / 0x14762f0 /
// 0x1476390 / 0x1476420:
// - types 0-5: the slot's record at the user-data slot, 4 dwords (8 for 1 and
//   3); type 6: the dword 0x20;
// - types 8-0xb: a table over the slots the shader's masks cover (textures
//   32 bytes a slot, samplers 32, sampler caches 16, constant buffers 16),
//   from slot 0 to the last set one, its address in two dwords; nothing when
//   the mask is empty;
// - types 0xd-0x13 and 0x15-0x18 do the same in the extended area at dword
//   (user-data slot - 16); type 0x1a copies `extra` extended dwords from the
//   resource-slot dword and binds the copy's address.
// Slots without an object stay zero. Types 7 and 0x14 (the 0xc0 block),
// texture slots past 63 and malformed descriptors are not covered.
enum GxUserResult { kUserBuilt, kUserType, kUserSlot, kUserRing, kUserResults };
struct GxUserData {
    std::uint32_t user[16] = {};
    std::uint16_t set = 0;           // user-data dwords written
    std::uint64_t ext_set[4] = {};   // extended dwords written (not pointers)
    struct Range {
        std::uint8_t ud = 0;         // user-data slot of the pointer
        std::uint64_t va = 0;        // the ring copy
        std::uint32_t stride = 0;    // table: bytes a slot; block: 4
        std::uint64_t slots = 0;     // table: the mask's slots
        std::uint32_t first = 0, count = 0;  // block: extended dwords copied
    } ranges[16];
    int nranges = 0;
    std::uint16_t skipped = 0;       // user-data slots whose table or block table_mask left out (pointer zero)
    bool partial = false;            // read_mask left some descriptor out: the fallback variant needs a full build
};

// `table_mask` (bit n: user-data slot n) leaves out tables and blocks at
// slots it does not cover: their pointers stay zero and nothing goes into the
// ring (the tables a no-fallback variant does not read). Tables
// in the extended area are reached through the block and follow its slot.
// The descriptors build_gx_user_data builds from, for one descriptor list and
// pair of masks: which it keeps and in what order (the list's own, with the
// extended block's pointer last), whether any was left out, and where the
// block's pointer lives - worked out once, since the list belongs to the
// shader object, rather than walked twice over at every draw. Direct-mapped by
// the list's fingerprint (GxStageRecords::desc_fp) and the masks.
struct UserBuildList {
    std::uint64_t key = 0;
    std::uint8_t idx[64];
    std::uint8_t n0 = 0, n = 0;  // [0, n0): the first pass, [n0, n): the block
    std::uint8_t block_ud = 0xff;
    bool partial = false;
};
UserBuildList g_user_lists[1024];  // under g.mu
std::uint64_t g_user_lists_built = 0, g_user_lists_used = 0;

// `read_mask` (bit n: user-data dword n; the no-fallback variant's
// user_reads) leaves out what that variant never touches - a record, a
// constant, a table's or the extended block's pointer - and says so in
// `partial`, so a draw that ends on the fallback variant builds it all.
GxUserResult build_gx_user_data(const GxStageRecords& st, GxUserData& out, std::uint32_t table_mask = ~0u,
                                std::uint32_t read_mask = ~0u) {
    // Only what is read without a mask is reset: `user` is read where `set`
    // says, `ranges` up to `nranges`, and the extended block is zeroed only
    // over what a block copies out (below). Clearing all of it was ~1.7 KiB
    // of stores a stage, at every draw.
    out.set = 0;
    std::memset(out.ext_set, 0, sizeof(out.ext_set));
    out.nranges = 0;
    out.skipped = 0;
    out.partial = false;
    std::uint32_t ext[256];
    std::uint64_t ext_written[4] = {};
    static std::vector<std::uint8_t> buf;  // under g.mu
    std::uint32_t block_ud = 0xff;
    for (std::uint32_t i = 0; i < st.ndesc && i < 64; ++i) {
        if ((st.desc[i] & 0xff) == 0x1a) block_ud = (st.desc[i] >> 8) & 0xff;
    }
    // The kept descriptors in order, from the cache when the list has one.
    UserBuildList* list = nullptr;
    std::uint64_t list_key = 0;
    if (st.desc_fp && st.ndesc <= 64) {
        const std::uint64_t key = ((st.desc_fp ^ (static_cast<std::uint64_t>(read_mask) << 32) ^ table_mask) * 0x9e3779b97f4a7c15ull) | 1;
        list_key = key;
        UserBuildList& e = g_user_lists[(key >> 40) & 1023];
        if (e.key == key) {
            list = &e;
            ++g_user_lists_used;
        } else {
            // Filled below as the descriptors are walked, and kept only if the
            // walk finishes (a malformed list returns early and is not kept).
            e.key = 0;
            e.n0 = e.n = 0;
            e.partial = false;
            e.block_ud = static_cast<std::uint8_t>(block_ud);
            list = &e;
        }
    }
    const bool list_ready = list && list->key != 0;
    const auto wanted = [&](std::uint32_t ud) { return ud + 1 >= 32 || ((table_mask >> ud) & 3) != 0; };
    // Whether the variant reads any of user-data dwords [ud, ud + n).
    const auto read = [&](std::uint32_t ud, std::uint32_t n) {
        return ud >= 16 || ((read_mask >> ud) & ((1u << n) - 1)) != 0;
    };
    // Extended entries are reached only through the block's pointer.
    const bool block_read = block_ud == 0xff || read(block_ud, 2);
    const std::uint32_t null_pointer[2] = {0, 0};
    const auto put_user = [&](std::uint32_t ud, const std::uint32_t* w, std::uint32_t n) {
        if (ud + n > 16) return false;
        std::memcpy(out.user + ud, w, n * 4u);
        out.set = static_cast<std::uint16_t>(out.set | (((1u << n) - 1) << ud));
        return true;
    };
    const auto put_ext = [&](std::uint32_t off, const std::uint32_t* w, std::uint32_t n, bool compared) {
        if (off + n > 256) return false;
        std::memcpy(ext + off, w, n * 4u);
        for (std::uint32_t k = 0; k < n; ++k) {
            ext_written[(off + k) >> 6] |= 1ull << ((off + k) & 63);
            if (compared) out.ext_set[(off + k) >> 6] |= 1ull << ((off + k) & 63);
        }
        return true;
    };
    if (list_ready) out.partial = list->partial;
    const std::uint32_t steps = list_ready ? list->n : 2 * std::min<std::uint32_t>(st.ndesc, 64);
    for (std::uint32_t step = 0; step < steps; ++step) {
        const std::uint32_t walk = std::min<std::uint32_t>(st.ndesc, 64);
        const int pass = list_ready ? (step >= list->n0 ? 1 : 0) : (step >= walk ? 1 : 0);
        const std::uint32_t i = list_ready ? list->idx[step] : step % walk;
        {
            const std::uint32_t d = st.desc[i];
            const std::uint32_t type = d & 0xff, ud = (d >> 8) & 0xff, slot = (d >> 16) & 0xff, extra = d >> 24;
            if (!list_ready && (type == 0x1a) != (pass == 1)) continue;
            const bool in_ext = type >= 0xd && type <= 0x18;  // 0x14 (the context block's pointer) among them
            if (in_ext != (ud >= 16)) return kUserType;
            const std::uint32_t off = ud - (in_ext ? 16 : 0);
            if (!list_ready) {
                // What the descriptor writes: 8 dwords for a T# or S# pair, 4 for a
                // record, 1 for a constant, 2 for a pointer.
                const std::uint32_t n = type == 0x1 || type == 0x3 ? 8 : type <= 0x5 ? 4 : type == 0x6 ? 1 : 2;
                if (in_ext ? !block_read : !read(ud, n)) {
                    out.partial = true;
                    if (list) list->partial = true;
                    continue;
                }
                if (list) {
                    list->idx[list->n++] = static_cast<std::uint8_t>(i);
                    if (pass == 0) list->n0 = list->n;
                }
            }
            switch (type) {
                case 0x0: case 0x1: case 0x2: case 0x3: case 0x4: case 0x5:
                case 0xd: case 0xe: case 0xf: case 0x10: case 0x11: case 0x12: {
                    const std::uint32_t n = type == 0x1 || type == 0x3 || type == 0xe || type == 0x10 ? 8 : 4;
                    std::uint32_t w[8] = {};
                    if (const std::uint8_t* rec = gx_object_record(st, type, slot)) std::memcpy(w, rec, n * 4u);
                    if (!(in_ext ? put_ext(off, w, n, true) : put_user(off, w, n))) return kUserSlot;
                    break;
                }
                case 0x6: case 0x13: {
                    const std::uint32_t w = 0x20;
                    if (!(in_ext ? put_ext(off, &w, 1, true) : put_user(off, &w, 1))) return kUserSlot;
                    break;
                }
                case 0x8: case 0x9: case 0xa: case 0xb: case 0x15: case 0x16: case 0x17: case 0x18: {
                    const std::uint32_t source = type >= 0x15 ? type - 0xd : type;
                    const std::uint32_t stride = source <= 0x9 ? 32 : 16;
                    std::uint64_t slots = 0;
                    if (source == 0x8) {
                        if (st.mask[1]) return kUserSlot;
                        slots = st.mask[0];
                    } else if (source == 0x9) {
                        slots = (st.mask[2] >> 36) & 0xffff;
                    } else if (source == 0xa) {
                        slots = st.mask[2] & 0xffff;
                    } else {
                        slots = (st.mask[2] >> 16) & 0xfffff;
                    }
                    if (!slots) break;  // the commit binds nothing
                    if (!(in_ext ? block_ud == 0xff || wanted(block_ud) : wanted(ud))) {
                        if (!(in_ext ? put_ext(off, null_pointer, 2, false) : put_user(off, null_pointer, 2))) return kUserSlot;
                        if (!in_ext) out.skipped = static_cast<std::uint16_t>(out.skipped | (1u << ud));
                        break;
                    }
                    const std::uint32_t first = static_cast<std::uint32_t>(__builtin_ctzll(slots));
                    const std::uint32_t last = 63 - static_cast<std::uint32_t>(__builtin_clzll(slots));
                    buf.assign(static_cast<std::size_t>(last + 1) * stride, 0);
                    for (std::uint32_t s = first; s <= last; ++s) {
                        if (const std::uint8_t* rec = gx_object_record(st, source, s)) std::memcpy(buf.data() + s * stride, rec, stride);
                    }
                    const std::uint64_t va = ring_place(buf.data(), buf.size());
                    if (!va) return kUserRing;
                    const std::uint32_t w[2] = {static_cast<std::uint32_t>(va), static_cast<std::uint32_t>(va >> 32)};
                    if (!(in_ext ? put_ext(off, w, 2, false) : put_user(off, w, 2))) return kUserSlot;
                    if (!in_ext && out.nranges < 16) out.ranges[out.nranges++] = {static_cast<std::uint8_t>(ud), va, stride, slots, 0, 0};
                    break;
                }
                case 0x7: case 0x14: {
                    // The context's 0xc0-byte block (0x2ace5e0 copies it into GX's
                    // ring and hands the shader the copy's address).
                    if (!st.has_ctx_block) return kUserSlot;
                    if (!(in_ext ? block_ud == 0xff || wanted(block_ud) : wanted(ud))) {
                        if (!(in_ext ? put_ext(off, null_pointer, 2, false) : put_user(off, null_pointer, 2))) return kUserSlot;
                        if (!in_ext) out.skipped = static_cast<std::uint16_t>(out.skipped | (1u << ud));
                        break;
                    }
                    const std::uint64_t va = ring_place(st.ctx_block, sizeof(st.ctx_block));
                    if (!va) return kUserRing;
                    const std::uint32_t w[2] = {static_cast<std::uint32_t>(va), static_cast<std::uint32_t>(va >> 32)};
                    if (!(in_ext ? put_ext(off, w, 2, false) : put_user(off, w, 2))) return kUserSlot;
                    break;
                }
                case 0x1a: {
                    if (!extra || slot + extra > 256) return kUserSlot;
                    if (!wanted(ud)) {
                        if (!put_user(off, null_pointer, 2)) return kUserSlot;
                        out.skipped = static_cast<std::uint16_t>(out.skipped | (1u << ud));
                        break;
                    }
                    for (std::uint32_t k = slot; k < slot + extra; ++k) {
                        if (!((ext_written[k >> 6] >> (k & 63)) & 1)) ext[k] = 0;  // what no descriptor wrote reads as zero
                    }
                    const std::uint64_t va = ring_place(reinterpret_cast<const std::uint8_t*>(ext + slot), static_cast<std::uint64_t>(extra) * 4);
                    if (!va) return kUserRing;
                    const std::uint32_t w[2] = {static_cast<std::uint32_t>(va), static_cast<std::uint32_t>(va >> 32)};
                    if (!put_user(off, w, 2)) return kUserSlot;
                    if (out.nranges < 16) out.ranges[out.nranges++] = {static_cast<std::uint8_t>(ud), va, 4, 0, slot, extra};
                    break;
                }
                default:
                    return kUserType;
            }
        }
    }
    if (list && !list_ready) {
        list->key = list_key;  // the walk finished: the list holds what it built
        ++g_user_lists_built;
    }
    return kUserBuilt;
}

// BBHOST_GX_BACKEND=2: a stage's user data built from its
// records against the command stream's. Plain dwords compare directly;
// tables compare each masked slot's bytes and extended blocks each written
// dword, against the guest's copy at the command stream's pointer.
struct UserDataCompare {
    std::uint64_t stages = 0, results[kUserResults] = {}, dwords = 0, dwords_differ = 0, units = 0, units_differ = 0, unreadable = 0;
    int logs = 0;
};
UserDataCompare g_user_compare;  // under g.mu

void compare_gx_user_data(const std::uint32_t* reg, const GxStageRecords& st, int stage) {
    UserDataCompare& c = g_user_compare;
    ++c.stages;
    GxUserData u;
    const GxUserResult result = build_gx_user_data(st, u);
    ++c.results[result];
    const char* name = stage ? "PS" : "VS";
    if (result == kUserBuilt) {
        std::uint32_t pointers = 0;
        for (int i = 0; i < u.nranges; ++i) pointers |= 3u << u.ranges[i].ud;
        for (int k = 0; k < 16; ++k) {
            if (!((u.set >> k) & 1) || ((pointers >> k) & 1)) continue;
            ++c.dwords;
            if (reg[k] != u.user[k]) {
                ++c.dwords_differ;
                if (c.logs++ < 12) host_log("gx-user: %s user %d: registers %08x, built %08x", name, k, reg[k], u.user[k]);
            }
        }
        for (int i = 0; i < u.nranges; ++i) {
            const GxUserData::Range& r = u.ranges[i];
            const std::uint64_t guest = static_cast<std::uint64_t>(reg[r.ud]) | (static_cast<std::uint64_t>(reg[r.ud + 1]) << 32);
            const auto unit = [&](std::uint64_t at, std::uint32_t bytes, const char* what, std::uint32_t index) {
                ++c.units;
                std::uint8_t a[32] = {};
                if (!hle_kernel_va_mapped(guest + at, bytes)) {
                    ++c.unreadable;
                    return;
                }
                read_guest_locked(guest + at, bytes, a);
                if (std::memcmp(a, reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(r.va + at)), bytes) != 0) {
                    ++c.units_differ;
                    std::uint32_t g[2] = {}, b[2] = {};
                    std::memcpy(g, a, 8);
                    std::memcpy(b, reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(r.va + at)), 8);
                    if (c.logs++ < 12) {
                        host_log("gx-user: %s user %u %s %u: registers' copy %08x %08x, built %08x %08x", name, r.ud, what, index, g[0], g[1],
                                 b[0], b[1]);
                    }
                }
            };
            if (r.slots) {
                for (std::uint32_t s = 0; s < 64; ++s) {
                    if ((r.slots >> s) & 1) unit(static_cast<std::uint64_t>(s) * r.stride, r.stride, "table slot", s);
                }
            } else {
                for (std::uint32_t k = 0; k < r.count; ++k) {
                    const std::uint32_t e = r.first + k;
                    if ((u.ext_set[e >> 6] >> (e & 63)) & 1) unit(static_cast<std::uint64_t>(k) * 4, 4, "block dword", e);
                }
            }
        }
    }
    if (c.stages % 20000 == 0) {
        host_log("gx-user: %llu stages: built %llu (uncovered type %llu, slot %llu, no ring %llu); plain dwords %llu, differing %llu; "
                 "table slots and block dwords %llu, differing %llu, unreadable %llu",
                 static_cast<unsigned long long>(c.stages), static_cast<unsigned long long>(c.results[kUserBuilt]),
                 static_cast<unsigned long long>(c.results[kUserType]), static_cast<unsigned long long>(c.results[kUserSlot]),
                 static_cast<unsigned long long>(c.results[kUserRing]), static_cast<unsigned long long>(c.dwords),
                 static_cast<unsigned long long>(c.dwords_differ), static_cast<unsigned long long>(c.units),
                 static_cast<unsigned long long>(c.units_differ), static_cast<unsigned long long>(c.unreadable));
    }
}

// BBHOST_GX_BACKEND=2: each V# record a GX draw's fetch shader
// loads (SMRD from the table in VS user data) against the record the call
// built natively (build_vertex_table in gx_trace.cpp). One compared draw in
// 64; `gx-vertex:` lines.
struct VertexTableProbe {
    std::uint64_t samples = 0, records = 0, unreadable = 0, equal = 0, differ = 0, not_built = 0;
    int differ_logs = 0, missing_logs = 0;
};
VertexTableProbe g_vertex_probe;  // under g.mu

void probe_gx_vertex_table(const std::vector<std::uint32_t>& fetch_words, const std::uint32_t* vs_user, const GxDrawObjects& o) {
    VertexTableProbe& p = g_vertex_probe;
    static std::uint64_t calls = 0;
    if (calls++ % 64) return;
    ++p.samples;
    const gcn::Program fp = gcn::decode(fetch_words.data(), fetch_words.size());
    for (const gcn::Inst& in : fp.insts) {
        if (in.enc != gcn::Enc::SMRD || in.op >= 8 || in.src0 >= 15) continue;
        const std::uint64_t table = static_cast<std::uint64_t>(vs_user[in.src0]) | (static_cast<std::uint64_t>(vs_user[in.src0 + 1]) << 32);
        const std::uint64_t at = table + static_cast<std::uint64_t>(in.imm) * 4;
        if (!hle_kernel_va_mapped(at, 16)) {
            ++p.unreadable;
            continue;
        }
        std::uint32_t w[4] = {};
        read_guest_locked(at, 16, w);
        ++p.records;
        const std::uint32_t slot = in.imm / 4;
        if (in.src0 != o.vtx_ud || in.imm % 4 || slot >= 16 || !((o.vtx_valid >> slot) & 1)) {
            ++p.not_built;
            if (p.missing_logs++ < 6) {
                host_log("gx-vertex: load s%u+%u (%08x %08x %08x %08x) not built: table at s%u, slots %04x", in.src0, in.imm, w[0], w[1],
                         w[2], w[3], o.vtx_ud, o.vtx_valid);
            }
            continue;
        }
        const std::uint32_t* b = o.vtx_rec[slot];
        if (std::memcmp(w, b, sizeof(w)) == 0) {
            ++p.equal;
        } else {
            ++p.differ;
            if (p.differ_logs++ < 8) {
                host_log("gx-vertex: slot %u loaded %08x %08x %08x %08x, built %08x %08x %08x %08x", slot, w[0], w[1], w[2], w[3], b[0], b[1],
                         b[2], b[3]);
            }
        }
    }
    if (p.samples % 2000 == 0) {
        host_log("gx-vertex: %llu sampled draws, %llu records loaded: equal to the built record %llu, differing %llu, not built %llu, "
                 "unreadable %llu",
                 static_cast<unsigned long long>(p.samples), static_cast<unsigned long long>(p.records),
                 static_cast<unsigned long long>(p.equal), static_cast<unsigned long long>(p.differ),
                 static_cast<unsigned long long>(p.not_built), static_cast<unsigned long long>(p.unreadable));
    }
}

}  // namespace

void host_gpu_depth_clear(std::uint64_t base, std::uint32_t word, std::uint32_t stencil) {
    std::lock_guard<GpuMutex> lk(g.mu);
    bool any = false;
    for (auto& kv : g_rts) {
        RtImage& r = kv.second;
        if (!r.depth || r.base != base) continue;
        r.htile_clear_pending = true;
        const float depth = htile_word_depth(word);
        std::memcpy(&r.htile_clear_depth, &depth, 4);
        r.htile_clear_stencil = stencil;
        any = true;
    }
    if (!any) g_pending_htile[base] = PendingHtile{word, stencil};
}

std::uint64_t host_gpu_ring_place(const void* data, std::size_t bytes, std::size_t align) {
    if (!data || !bytes || !align || (align & (align - 1))) return 0;
    std::lock_guard<GpuMutex> lk(g.mu);
    return ring_place(static_cast<const std::uint8_t*>(data), bytes, align < 8 ? 8 : align);
}


namespace {

// The register-encoded inputs of a draw, recorded with a draw capture.
json::Value draw_inputs_json(const GpuDrawInputs& in) {
    const auto words = [](const std::uint32_t* w, std::size_t n) {
        json::Value a = json::Value::make_array();
        for (std::size_t i = 0; i < n; ++i) a.push(json::hex(w[i]));
        return a;
    };
    const auto floats = [](const float* f, std::size_t n) {
        json::Value a = json::Value::make_array();
        for (std::size_t i = 0; i < n; ++i) a.push(json::f32(f[i]));
        return a;
    };
    json::Value r = json::Value::make_object();
    r.set("vs_pgm", words(in.vs_pgm, 4));
    r.set("ps_pgm", words(in.ps_pgm, 4));
    r.set("vs_user", words(in.vs_user, 16));
    r.set("ps_user", words(in.ps_user, 16));
    r.set("prim", in.prim);
    r.set("target_mask", json::hex(in.target_mask));
    r.set("cb_shader_mask", json::hex(in.cb_shader_mask));
    json::Value color = json::Value::make_array();
    for (int t = 0; t < 8; ++t) color.push(words(in.color[t], 5));
    r.set("color_base_pitch_slice_view_info", color);
    r.set("blend", words(in.blend, 8));
    r.set("blend_const", floats(in.blend_const, 4));
    r.set("depth_control", json::hex(in.depth_control));
    r.set("stencil_control", json::hex(in.stencil_control));
    r.set("stencil_ref", json::hex(in.stencil_ref));
    r.set("stencil_ref_bf", json::hex(in.stencil_ref_bf));
    r.set("depth_bounds", floats(in.depth_bounds, 2));
    r.set("render_control", json::hex(in.render_control));
    r.set("depth_clear", json::hex(in.depth_clear));
    r.set("stencil_clear", json::hex(in.stencil_clear));
    r.set("z_info", json::hex(in.z_info));
    r.set("stencil_info", json::hex(in.stencil_info));
    r.set("z_read_base", json::hex(in.z_read_base));
    r.set("depth_size", json::hex(in.depth_size));
    r.set("htile_base", json::hex(in.htile_base));
    r.set("su_sc_mode", json::hex(in.su_sc_mode));
    r.set("clip_cntl", json::hex(in.clip_cntl));
    r.set("vte_cntl", json::hex(in.vte_cntl));
    r.set("ps_input_ena", json::hex(in.ps_input_ena));
    r.set("ps_in_control", json::hex(in.ps_in_control));
    r.set("vs_out_cntl", json::hex(in.vs_out_cntl));
    r.set("ps_input_cntl", words(in.ps_input_cntl, 32));
    r.set("vport", floats(in.vport, 6));
    r.set("screen_scissor", words(in.screen_scissor, 2));
    r.set("generic_scissor", words(in.generic_scissor, 2));
    r.set("vport_scissor", words(in.vport_scissor, 2));
    r.set("sc_mode_cntl_0", json::hex(in.sc_mode_cntl_0));
    r.set("base_vertex", in.base_vertex);
    return r;
}

// The user-data
// descriptor table a GX draw's shader objects carry (+0xe0, count at +0xe8),
// logged beside the translator's binding paths for the first pipelines, so the
// descriptor layout is fixed before bindings are resolved from GX slots.
// Called under g.mu.
void probe_gx_descriptors(const GfxPipeline& pl, const GxDrawObjects& gx) {
    static std::set<std::string> seen;
    if (seen.size() >= 12 || !seen.insert(pl.name).second) return;
    host_log("gx-bind: pipeline %s from GX call site 0x%llx", pl.name.c_str(), static_cast<unsigned long long>(gx.caller));
    const auto stage = [&](const char* name, std::uint64_t shader, const gcn::TranslateResult& meta) {
        if (!shader || !hle_kernel_va_mapped(shader + 0xe0, 16)) return;
        std::uint64_t table = 0, count = 0;
        read_guest_locked(shader + 0xe0, 8, &table);
        read_guest_locked(shader + 0xe8, 8, &count);
        std::string dwords;
        const std::uint64_t n = std::min<std::uint64_t>(count & 0xffffffffull, 48);
        if (table && hle_kernel_va_mapped(table, n * 4)) {
            for (std::uint64_t i = 0; i < n; ++i) {
                std::uint32_t w = 0;
                read_guest_locked(table + i * 4, 4, &w);
                char buf[16];
                std::snprintf(buf, sizeof(buf), " %08x", w);
                dwords += buf;
            }
        }
        host_log("gx-bind:   %s shader 0x%llx: descriptors at 0x%llx, +0xe8 = 0x%llx:%s", name, static_cast<unsigned long long>(shader),
                 static_cast<unsigned long long>(table), static_cast<unsigned long long>(count), dwords.c_str());
        for (const gcn::ImageBinding& b : meta.images) {
            host_log("gx-bind:     %s image binding %u: %s%s", name, b.binding, b.path.str().c_str(), b.cube ? " (cube)" : "");
        }
        for (const gcn::SamplerBinding& b : meta.samplers) {
            host_log("gx-bind:     %s sampler binding %u: %s", name, b.binding, b.path.str().c_str());
        }
        for (const gcn::BufferBinding& b : meta.buffers) {
            host_log("gx-bind:     %s buffer binding %u: %s (%s, %u dwords)", name, b.binding, b.path.str().c_str(),
                     b.pointer ? "pointer" : "V#", b.max_dw);
        }
    };
    stage("VS", gx.shader[0], pl.vs.meta());
    stage("PS", gx.shader[4], pl.ps.meta());
}

// Every translated binding of a sampled GX draw resolved from
// the records the GX commit wrote, through the shader's user-data descriptors,
// against the PM4 path's resolution from user data. Called under g.mu.
struct GxBindCompare {
    std::uint64_t draws = 0, bindings = 0, no_descriptor = 0, unsupported = 0, unresolved = 0, pointer = 0;
    std::uint64_t equal[32] = {}, differ[32] = {};
    std::uint64_t obj_equal[32] = {}, obj_differ[32] = {}, obj_missing[32] = {};  // object-built record vs the commit's
    std::uint64_t ext_equal[2] = {}, ext_differ[2] = {};                           // extended blocks read as pointer buffers, VS/PS
    std::uint64_t cb_same_address = 0, cb_renamed = 0;  // differing constant-buffer records: flags only, or another address
    std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> sites;  // call site -> (bindings, differing)
    std::map<std::string, std::uint64_t> examples;                          // logged per stage/kind/type
};
GxBindCompare g_bind_compare;

// The record a descriptor type selects for a resource slot, and how many
// dwords of it the shader sees; null for types that are not records.
const std::uint8_t* gx_record(const GxStageRecords& st, std::uint8_t type, std::uint32_t slot, std::uint32_t& dw) {
    const auto pick = [&](const std::uint8_t* array, std::size_t bytes, std::uint32_t elem, std::uint32_t words) -> const std::uint8_t* {
        if ((static_cast<std::size_t>(slot) + 1) * elem > bytes) return nullptr;
        dw = words;
        return array + static_cast<std::size_t>(slot) * elem;
    };
    switch (type) {
        case 0x0: case 0xd: return pick(st.tex, sizeof(st.tex), 32, 4);
        case 0x1: case 0xe: case 0x8: return pick(st.tex, sizeof(st.tex), 32, 8);
        case 0x2: case 0xf: return pick(st.smp, sizeof(st.smp), 32, 4);
        case 0x3: case 0x10: case 0x9: return pick(st.smp, sizeof(st.smp), 32, 8);
        case 0x4: case 0x11: case 0xa: return pick(st.smp_cache, sizeof(st.smp_cache), 16, 4);
        case 0x5: case 0x12: case 0xb: return pick(st.cb, sizeof(st.cb), 16, 4);
        default: return nullptr;
    }
}

void compare_gx_bindings(const GfxPipeline& pl, const std::uint32_t* vs_user, const std::uint32_t* ps_user, const GxDrawObjects& gx) {
    GxBindCompare& c = g_bind_compare;
    ++c.draws;
    const auto stage = [&](int k, const gcn::TranslateResult& meta, const std::uint32_t* user) {
        const GxStageRecords& st = gx.records->stage[k];
        const auto find = [&](std::uint32_t ud) -> const std::uint32_t* {
            for (std::uint32_t i = 0; i < st.ndesc; ++i) {
                if (((st.desc[i] >> 8) & 0xff) == ud) return &st.desc[i];
            }
            return nullptr;
        };
        const auto one = [&](const gcn::ResourcePath& path, int ndw, const char* kind, std::uint32_t binding) {
            ++c.bindings;
            auto& site = c.sites[gx.caller];
            ++site.first;
            const std::uint32_t* d = path.user_sgpr >= 0 ? find(static_cast<std::uint32_t>(path.user_sgpr)) : nullptr;
            if (!d) {
                ++c.no_descriptor;
                return;
            }
            std::uint8_t type = *d & 0xff;
            std::uint32_t slot = (*d >> 16) & 0xff;
            if (!path.immediate) {
                const std::uint32_t off = static_cast<std::uint32_t>(path.final_offset_dw);
                if (!path.loads.empty() || path.final_vsharp) {
                    ++c.unsupported;
                    return;
                }
                if ((type == 0x8 || type == 0x9) && off % 8 == 0) {
                    slot = off / 8;
                } else if ((type == 0xa || type == 0xb) && off % 4 == 0) {
                    slot = off / 4;
                } else if (type == 0x1a) {
                    const std::uint32_t* e = find(16 + off);  // extended block: user-data slot 16 + dword offset
                    if (!e) {
                        ++c.no_descriptor;
                        return;
                    }
                    type = *e & 0xff;
                    slot = (*e >> 16) & 0xff;
                } else {
                    ++c.unsupported;
                    return;
                }
            }
            std::uint32_t rec_dw = 0;
            const std::uint8_t* rec = gx_record(st, type, slot, rec_dw);
            if (!rec) {
                ++c.unsupported;
                return;
            }
            // The record a native draw would build from the pending-state
            // object, against the one the commit wrote.
            {
                const std::uint8_t* obj = nullptr;
                std::uint32_t bytes = rec_dw * 4;
                bool set = false;
                switch (type) {
                    case 0x0: case 0xd: case 0x1: case 0xe: case 0x8:
                        set = slot < 64 && ((st.obj_tex_set >> slot) & 1);
                        obj = st.obj_tex + slot * 32;
                        break;
                    case 0x2: case 0xf: case 0x3: case 0x10: case 0x9: case 0x4: case 0x11: case 0xa:
                        // An empty slot has only the 16-byte sampler-cache default.
                        set = slot < 16 && (((st.obj_smp_set >> slot) & 1) || (bytes <= 16 && ((st.obj_smp_default >> slot) & 1)));
                        obj = st.obj_smp + slot * 32;
                        break;
                    case 0x5: case 0x12: case 0xb:
                        set = slot < 14 && ((st.obj_cb_set >> slot) & 1);
                        obj = st.obj_cb + slot * 16;
                        break;
                    default: break;
                }
                if (!set) {
                    ++c.obj_missing[type & 0x1f];
                } else if (std::memcmp(obj, rec, bytes) == 0) {
                    ++c.obj_equal[type & 0x1f];
                } else {
                    ++c.obj_differ[type & 0x1f];
                    if (type == 0x5 || type == 0xb || type == 0x12) {
                        std::uint32_t o[2] = {}, r[2] = {};
                        std::memcpy(o, obj, 8);
                        std::memcpy(r, rec, 8);
                        (o[0] == r[0] && (o[1] & 0xff) == (r[1] & 0xff) ? c.cb_same_address : c.cb_renamed)++;
                    }
                    char key[48];
                    std::snprintf(key, sizeof(key), "obj %s %s 0x%x", k ? "PS" : "VS", kind, type);
                    if (c.examples[key]++ < 3) {
                        std::string a, b;
                        for (std::uint32_t i = 0; i < bytes / 4; ++i) {
                            std::uint32_t x = 0, y = 0;
                            std::memcpy(&x, obj + i * 4u, 4);
                            std::memcpy(&y, rec + i * 4u, 4);
                            char buf[12];
                            std::snprintf(buf, sizeof(buf), " %08x", x);
                            a += buf;
                            std::snprintf(buf, sizeof(buf), " %08x", y);
                            b += buf;
                        }
                        host_log("gx-bind-compare: %s %s binding %u slot %u from site 0x%llx: object%s, commit%s", pl.name.c_str(), key,
                                 binding, slot, static_cast<unsigned long long>(gx.caller), a.c_str(), b.c_str());
                    }
                }
            }
            std::uint32_t words[8] = {};
            if (!resolve_resource_impl(path, user, ndw, words)) {
                ++c.unresolved;
                if (c.examples["unresolved"]++ < 6) {
                    const int u = path.user_sgpr;
                    host_log("gx-bind-compare: unresolved on the PM4 side: %s %s %s binding %u %s: descriptor %08x -> type 0x%x slot %u; "
                             "user data %08x %08x",
                             pl.name.c_str(), k ? "PS" : "VS", kind, binding, path.str().c_str(), *d, type, slot,
                             u >= 0 && u < 16 ? user[u] : 0, u >= 0 && u + 1 < 16 ? user[u + 1] : 0);
                }
                return;
            }
            const std::uint32_t n = std::min<std::uint32_t>(static_cast<std::uint32_t>(ndw), rec_dw);
            if (std::memcmp(words, rec, n * 4u) == 0) {
                ++c.equal[type & 0x1f];
                return;
            }
            ++c.differ[type & 0x1f];
            ++site.second;
            char key[48];
            std::snprintf(key, sizeof(key), "%s %s 0x%x", k ? "PS" : "VS", kind, type);
            if (c.examples[key]++ < 3) {
                std::string a, b;
                for (std::uint32_t i = 0; i < n; ++i) {
                    char buf[12];
                    std::snprintf(buf, sizeof(buf), " %08x", words[i]);
                    a += buf;
                    std::uint32_t r = 0;
                    std::memcpy(&r, rec + i * 4u, 4);
                    std::snprintf(buf, sizeof(buf), " %08x", r);
                    b += buf;
                }
                host_log("gx-bind-compare: %s %s binding %u %s from site 0x%llx: descriptor %08x -> type 0x%x slot %u; user data%s, GX%s",
                         pl.name.c_str(), key, binding, path.str().c_str(), static_cast<unsigned long long>(gx.caller), *d, type, slot,
                         a.c_str(), b.c_str());
            }
        };
        for (const gcn::ImageBinding& b : meta.images) one(b.path, b.r128 ? 4 : 8, "image", b.binding);
        for (const gcn::SamplerBinding& b : meta.samplers) one(b.path, 4, "sampler", b.binding);
        for (const gcn::BufferBinding& b : meta.buffers) {
            if (!b.pointer) {
                one(b.path, 2, "buffer", b.binding);
                continue;
            }
            // A whole extended block (type 0x1a) read as a buffer: the guest
            // block against the commit's extended array, over the ranges the
            // shader's extended descriptors cover. Tables stay uncompared.
            const std::uint32_t* d = b.path.immediate && b.path.user_sgpr >= 0 ? find(static_cast<std::uint32_t>(b.path.user_sgpr)) : nullptr;
            if (!d || (*d & 0xff) != 0x1a) {
                ++c.pointer;
                continue;
            }
            std::uint32_t words[2] = {};
            const std::uint32_t dw = std::min<std::uint32_t>(std::min<std::uint32_t>(b.max_dw, (*d >> 24) & 0xff), 64);
            const std::uint64_t base = resolve_resource_impl(b.path, user, 2, words)
                                           ? static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1]) << 32)
                                           : 0;
            if (!base || !dw || !hle_kernel_va_mapped(base, static_cast<std::size_t>(dw) * 4)) {
                ++c.unresolved;
                continue;
            }
            std::uint32_t block[64] = {};
            read_guest_locked(base, static_cast<std::size_t>(dw) * 4, block);
            bool any = false, same = true;
            for (std::uint32_t i = 0; i < st.ndesc; ++i) {
                const std::uint32_t ud = (st.desc[i] >> 8) & 0xff;
                std::uint32_t size = 0;
                switch (st.desc[i] & 0xff) {
                    case 0xd: case 0xf: case 0x11: case 0x12: size = 4; break;
                    case 0xe: case 0x10: size = 8; break;
                    default: break;
                }
                if (ud < 16 || !size || ud - 16 + size > dw) continue;
                any = true;
                if (std::memcmp(block + (ud - 16), st.ext + (ud - 16) * 4, size * 4) != 0) same = false;
            }
            if (!any) {
                ++c.pointer;
            } else if (same) {
                ++c.ext_equal[k ? 1 : 0];
            } else {
                ++c.ext_differ[k ? 1 : 0];
                if (c.examples[k ? "ext PS" : "ext VS"]++ < 3) {
                    host_log("gx-bind-compare: %s %s extended block at 0x%llx (%u dwords) differs from the commit's copy",
                             pl.name.c_str(), k ? "PS" : "VS", static_cast<unsigned long long>(base), dw);
                }
            }
        }
    };
    stage(0, pl.vs.meta(), vs_user);
    stage(1, pl.ps.meta(), ps_user);
    if (c.draws % 5000 != 0) return;
    std::string types;
    for (int t = 0; t < 32; ++t) {
        if (!c.equal[t] && !c.differ[t]) continue;
        char buf[48];
        std::snprintf(buf, sizeof(buf), " 0x%x %llu/%llu", t, static_cast<unsigned long long>(c.equal[t]),
                      static_cast<unsigned long long>(c.differ[t]));
        types += buf;
    }
    std::string sites;
    int shown = 0;
    for (const auto& [site, counts] : c.sites) {
        if (!counts.second || shown++ >= 8) continue;
        char buf[64];
        std::snprintf(buf, sizeof(buf), " 0x%llx %llu/%llu", static_cast<unsigned long long>(site),
                      static_cast<unsigned long long>(counts.second), static_cast<unsigned long long>(counts.first));
        sites += buf;
    }
    host_log("gx-bind-compare: %llu sampled draws, %llu bindings; equal/differing by type:%s; no descriptor %llu, unsupported %llu, "
             "unresolved %llu, pointer buffers %llu; call sites with differences (differing/bindings):%s",
             static_cast<unsigned long long>(c.draws), static_cast<unsigned long long>(c.bindings), types.c_str(),
             static_cast<unsigned long long>(c.no_descriptor), static_cast<unsigned long long>(c.unsupported),
             static_cast<unsigned long long>(c.unresolved), static_cast<unsigned long long>(c.pointer), sites.empty() ? " none" : sites.c_str());
    std::string objects;
    for (int t = 0; t < 32; ++t) {
        if (!c.obj_equal[t] && !c.obj_differ[t] && !c.obj_missing[t]) continue;
        char buf[64];
        std::snprintf(buf, sizeof(buf), " 0x%x %llu/%llu/%llu", t, static_cast<unsigned long long>(c.obj_equal[t]),
                      static_cast<unsigned long long>(c.obj_differ[t]), static_cast<unsigned long long>(c.obj_missing[t]));
        objects += buf;
    }
    host_log("gx-bind-compare: records built from the pending-state objects vs the commit's, equal/differing/no object by type:%s; "
             "differing constant buffers: %llu same address, %llu renamed; extended blocks equal/differing VS %llu/%llu, PS %llu/%llu",
             objects.c_str(), static_cast<unsigned long long>(c.cb_same_address), static_cast<unsigned long long>(c.cb_renamed),
             static_cast<unsigned long long>(c.ext_equal[0]), static_cast<unsigned long long>(c.ext_differ[0]),
             static_cast<unsigned long long>(c.ext_equal[1]), static_cast<unsigned long long>(c.ext_differ[1]));
}

}  // namespace

// BBHOST_LOG_DRAW_STATE=<first flip>,<last flip>[,<file>]: one line per draw in
// those flips, with where it came from (a host-draw token, a GX draw packet or
// another draw packet) and the inputs it is decoded from, so two runs can be
// compared draw by draw. Default file: build/draw-state.log.
static void log_draw_state(const GpuDraw& d, const GpuDrawInputs& in) {
    struct Config {
        std::uint64_t first = 1, last = 0;
        FILE* file = nullptr;
    };
    static const Config config = [] {
        Config c;
        const char* e = std::getenv("BBHOST_LOG_DRAW_STATE");
        unsigned long long first = 0, last = 0;
        char path[512] = "build/draw-state.log";
        if (!e || std::sscanf(e, "%llu,%llu,%511s", &first, &last, path) < 2) return c;
        c.first = first;
        c.last = last;
        c.file = std::fopen(path, "w");
        if (!c.file) host_log("render: BBHOST_LOG_DRAW_STATE: cannot open %s", path);
        return c;
    }();
    if (!config.file) return;
    const std::uint64_t flip = hle_video_flip_count();
    if (flip < config.first || flip > config.last) return;
    const char* source = d.gx_token ? "token" : d.gx && d.gx_render ? "gx" : "packets";
    std::fprintf(config.file, "%llu %s %u %u %d %s\n", static_cast<unsigned long long>(flip), source, d.index_count, d.instance_count,
                 d.index_va || d.indirect_va ? 1 : 0, json::dump(draw_inputs_json(in)).c_str());
    std::fflush(config.file);
}

namespace {
// BBHOST_TESS_PROBE=1: for each distinct tessellated draw (VGT_SHADER_STAGES_EN
// with LS/HS on) - which the renderer cannot draw yet - log its tessellation
// registers once and write its LS, HS, VS (the domain shader) and PS code to
// tmp/tess/<stage>-<address>.bin for tools/gcndis.
void probe_tessellated_draw(const GpuDraw& d) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_TESS_PROBE");
        return e && e[0] == '1';
    }();
    const std::uint32_t stages = d.ctx[0x2d5];
    if (!on || !(stages & 0x7)) return;
    const auto pgm = [&](std::uint32_t lo) {
        return (static_cast<std::uint64_t>(d.sh[lo + 1] & 0xff) << 40) | (static_cast<std::uint64_t>(d.sh[lo]) << 8);
    };
    const std::uint64_t ls = pgm(0x148), hs = pgm(0x108), vs = pgm(0x48), ps = pgm(0x08);
    static std::mutex mu;
    static std::set<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>> seen;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (seen.size() >= 64 || !seen.insert({ls, hs, vs, ps}).second) return;
    }
    host_log("tess: draw with stages %08x: LS 0x%llx (rsrc %08x %08x) HS 0x%llx (rsrc %08x %08x) VS/DS 0x%llx (rsrc %08x %08x) PS 0x%llx; "
             "VGT_LS_HS_CONFIG %08x (patches %u, in cp %u, out cp %u) VGT_TF_PARAM %08x (type %u, partition %u, topology %u) "
             "HOS max %g min %g; prim %u, index count %u, instances %u%s; TF ring size %08x, offchip %08x, TF base %08x",
             stages, static_cast<unsigned long long>(ls), d.sh[0x14a], d.sh[0x14b], static_cast<unsigned long long>(hs), d.sh[0x10a],
             d.sh[0x10b], static_cast<unsigned long long>(vs), d.sh[0x4a], d.sh[0x4b], static_cast<unsigned long long>(ps), d.ctx[0x2d6],
             d.ctx[0x2d6] & 0xff, (d.ctx[0x2d6] >> 8) & 0x3f, (d.ctx[0x2d6] >> 14) & 0x3f, d.ctx[0x2db], d.ctx[0x2db] & 3,
             (d.ctx[0x2db] >> 2) & 7, (d.ctx[0x2db] >> 5) & 7, reinterpret_cast<const float&>(d.ctx[0x286]),
             reinterpret_cast<const float&>(d.ctx[0x287]), d.uconfig[0x242], d.index_count, d.instance_count, d.index_va ? " indexed" : "",
             d.uconfig[0x24e], d.uconfig[0x24f], d.uconfig[0x250]);
    host_log("tess:   user data LS %08x %08x %08x %08x %08x %08x %08x %08x | HS %08x %08x %08x %08x %08x %08x %08x %08x | "
             "VS %08x %08x %08x %08x %08x %08x %08x %08x",
             d.sh[0x14c], d.sh[0x14d], d.sh[0x14e], d.sh[0x14f], d.sh[0x150], d.sh[0x151], d.sh[0x152], d.sh[0x153], d.sh[0x10c],
             d.sh[0x10d], d.sh[0x10e], d.sh[0x10f], d.sh[0x110], d.sh[0x111], d.sh[0x112], d.sh[0x113], d.sh[0x4c], d.sh[0x4d],
             d.sh[0x4e], d.sh[0x4f], d.sh[0x50], d.sh[0x51], d.sh[0x52], d.sh[0x53]);
    // The constants the domain shader fades a near particle with: its V# in
    // user SGPRs 4-7, dwords 0x50-0x5f.
    {
        const std::uint64_t base = static_cast<std::uint64_t>(d.sh[0x50]) | (static_cast<std::uint64_t>(d.sh[0x51] & 0xfff) << 32);
        float cb[16] = {};
        if (base && hle_kernel_va_mapped(base + 0x140, sizeof(cb))) {
            std::memcpy(cb, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base + 0x140)), sizeof(cb));
        }
        host_log("tess:   DS constants at 0x%llx + 0x140: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g",
                 static_cast<unsigned long long>(base), cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7], cb[8], cb[9],
                 cb[10], cb[11], cb[12], cb[13], cb[14], cb[15]);
    }
    // The domain shader's tessellation constants: the V# in its user SGPRs 8-11.
    {
        const std::uint64_t base = static_cast<std::uint64_t>(d.sh[0x54]) | (static_cast<std::uint64_t>(d.sh[0x55] & 0xffff) << 32);
        std::uint32_t tc[12] = {};
        if (base && hle_kernel_va_mapped(base, sizeof(tc))) std::memcpy(tc, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base)), sizeof(tc));
        host_log("tess:   DS user 8-11 %08x %08x %08x %08x -> constants at 0x%llx: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                 d.sh[0x54], d.sh[0x55], d.sh[0x56], d.sh[0x57], static_cast<unsigned long long>(base), tc[0], tc[1], tc[2], tc[3], tc[4],
                 tc[5], tc[6], tc[7], tc[8], tc[9], tc[10], tc[11]);
        const std::uint64_t hbase = static_cast<std::uint64_t>(d.sh[0x10c]) | (static_cast<std::uint64_t>(d.sh[0x10d] & 0xffff) << 32);
        std::uint32_t hv[16] = {};
        if (hbase && hle_kernel_va_mapped(hbase, sizeof(hv))) std::memcpy(hv, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(hbase)), sizeof(hv));
        host_log("tess:   HS table at 0x%llx: +0x24 V# %08x %08x %08x %08x; HS user 4-7 V# %08x %08x %08x %08x", static_cast<unsigned long long>(hbase),
                 hv[9], hv[10], hv[11], hv[12], d.sh[0x110], d.sh[0x111], d.sh[0x112], d.sh[0x113]);
    }
    std::error_code ec;
    std::filesystem::create_directories("tmp/tess", ec);
    const std::pair<const char*, std::uint64_t> progs[4] = {{"ls", ls}, {"hs", hs}, {"vs", vs}, {"ps", ps}};
    for (const auto& [name, va] : progs) {
        if (!va || !hle_kernel_va_mapped(va, 4)) continue;
        char path[96];
        std::snprintf(path, sizeof(path), "tmp/tess/%s-%llx.bin", name, static_cast<unsigned long long>(va));
        if (std::filesystem::exists(path)) continue;
        // Up to s_endpgm (0xbf810000), 16 KiB at most.
        const auto* w = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
        std::size_t n = 0;
        while (n < 4096 && hle_kernel_va_mapped(va + n * 4, 4) && w[n] != 0xbf810000u) ++n;
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(w, 4, n + 1, f);
            std::fclose(f);
        }
    }
}
}  // namespace

// Tessellated draws. The game's
// particles - fire, item glints, the lamp's travel light, blood mist - are
// patches of one control point: the LS writes the particle to LDS, the HS
// writes constant tessellation factors, and the domain shader reads the
// particle back and places one corner of a camera-facing quad per domain
// coordinate. Vulkan's tessellation stages would need the LDS traffic turned
// into patch variables; the shape is simple enough to emulate instead:
//   1. the LS runs as a compute pass over the control points, its LDS a
//      buffer (TessRole::LsCompute), and
//   2. the draw runs as a triangle list of domain_level x domain_level quads a
//      patch, the domain shader its vertex shader (TessRole::Domain) reading
//      that buffer; the HS is not run - its factors are VGT_HOS_MAX_TESS_LEVEL.
// Other shapes (several control points, other domains, indirect draws) still
// fail, logged once each.
namespace {
// The LDS ring the tessellation LS pass writes (tess_ls_pass_locked): a
// region for each recording with tessellated draws in it, 16 regions of
// 16 MiB.
constexpr std::uint64_t kTessRegions = 16;
constexpr std::uint64_t kTessRegionBytes = 16ull << 20;
constexpr std::uint64_t kTessRingBytes = kTessRegions * kTessRegionBytes;

// Where the ring is and the LDS of the last tessellated draws in it, for the
// device-fault report (gpu::tess_lds_describe, host/tess_lds.h): which draw's
// LDS a faulting address is in or just past. Its own lock: the report can come
// while the renderer's is held.
std::mutex g_tess_lds_mu;
std::uint64_t g_tess_ring_address = 0;
constexpr std::size_t kTessLdsUses = 32;
TessLdsUse g_tess_lds_uses[kTessLdsUses];
std::uint64_t g_tess_lds_next = 0;

struct TessDraw {
    std::uint32_t count = 0, level = 1;
    // Drawn by the host's tessellator, with the hull shader's own
    // factors, instead of the topology the domain prologue invents.
    bool hw = false;
    std::uint32_t control_points = 1;
    float outer[4] = {1, 1, 1, 1}, inner[2] = {1, 1};
    // BBHOST_TESS_HW=2: and the LS is the pipeline's own vertex stage, so
    // there is no compute pass and no buffer between them.
    bool attrs = false;
    std::uint32_t attr_vec4s = 8;
    std::uint64_t ls_va = 0, ls_fetch_va = 0;
    std::uint32_t ls_rsrc1 = 0, ls_rsrc2 = 0;
    std::uint32_t ls_user[16] = {};
    std::uint64_t lds_address = 0, hs_va = 0;
    // The bytes of the ring from lds_address this draw was given: the end
    // its stages' LDS accesses are checked against (StageParams::lds_bytes).
    std::uint32_t lds_bytes = 0;
    float clamp_lo = 1.0f, clamp_hi = 1.0f;  // VGT_HOS_MIN/MAX_TESS_LEVEL
    // What the domain prologue needs to leave a patch out (tess_patch_cull):
    // the row of the projection in the domain shader's own constants that
    // gives clip w, and where a patch sits in the region.
    float cull_w[4] = {0, 0, 0, 0};
    std::uint32_t patch_stride = 0, patch_base = 0;
    // The game's own hull shader as the control stage (gx_hull_layout): the
    // Forbidden Woods' meshes. `count` is then control points, not patches.
    // Every patch has an LDS window of its own, and the tessellation
    // constants describe one patch; the hull and domain shaders read them
    // through a constant-buffer descriptor (slot 0x13) no GX object fills,
    // so the LS pass places them in the ring and puts their V# there.
    bool hull = false;
    std::uint32_t window = 0, hs_rsrc1 = 0, hs_rsrc2 = 0;
    // The draw's instances: the trees are instanced, and the LS reads each
    // one's transform by its instance id. The LS pass runs the points once
    // per instance and the draw is one patch list of all of them, so
    // gl_PrimitiveID still names a window.
    std::uint32_t instances = 1, first_instance = 0;
    std::uint32_t first_point = 0;  // the chunk's first point (index-buffer position), a whole number of patches in
    bool quads = true, cw = true;
    int spacing = 0;
    std::uint32_t tess_constants[12] = {};
    std::uint32_t tess_vsharp[4] = {};
    std::uint32_t hs_user[16] = {};
};
thread_local const TessDraw* t_tess = nullptr;

// The tessellation constants' V# where a stage's descriptors ask for constant
// buffer 0x13 directly in its user SGPRs (type 5: the woods' hull and domain
// shaders, user SGPRs 12-15) - written whatever the stage's read mask left
// out of its user-data build, as a binding resolved from user data needs it.
// The extended block's (type 0x12) comes from gx_object_record.
void put_tess_constants(const GxStageRecords& st, const std::uint32_t vsharp[4], std::uint32_t user[16]) {
    for (std::uint32_t i = 0; i < st.ndesc && i < 64; ++i) {
        const std::uint32_t d = st.desc[i];
        const std::uint32_t ud = (d >> 8) & 0xff;
        if ((d & 0xff) == 0x5 && ((d >> 16) & 0xff) == 0x13 && ud + 4 <= 16) std::memcpy(user + ud, vsharp, 16);
    }
}

// Vertices a patch: one quad at level 1, and above it a fan of 4L triangles
// from the patch's centre to the points around its edge (gcn/translate.cpp).
std::uint32_t tess_vertices_a_patch(std::uint32_t level) { return level <= 1 ? 6u : 12u * level; }

// How fine a patch may be drawn: one quad, which is what a hull shader that
// writes factors of 1 asks for exactly. BBHOST_TESS_LEVEL raises it, to work
// on the freeze the finer levels bring.
const std::uint32_t kTessLevelCap = [] {
    const char* e = std::getenv("BBHOST_TESS_LEVEL");
    return e ? static_cast<std::uint32_t>(std::clamp(std::atoi(e), 1, 8)) : 1u;
}();

// How finely a patch is tessellated is the hull shader's business: it writes
// the factors to the tessellation-factor ring, and VGT_HOS_MAX/MIN_TESS_LEVEL
// only clamp them. The hull shaders behind these patches write one constant
// (a v_mov_b32 of an inline constant, stored to the ring), so read it out of
// the program instead of running the stage. 0 when the factors are not that
// simple, and the draw falls back to one quad a patch - which is what a
// camera-facing particle needs, and never the runaway subdivision that
// reading the clamp register as a level produced.
bool hs_constant_factors(const gcn::Program& p, float out[6]);
std::pair<float, float> hs_constant_factor(const gcn::Program& p) {
    float f[6] = {};
    if (!hs_constant_factors(p, f)) return {0.0f, 0.0f};
    return {std::max({f[0], f[1], f[2], f[3]}), std::max(f[4], f[5])};
}
bool hs_constant_factors(const gcn::Program& p, float out[6]) {
    constexpr int kVgprs = 256;
    std::array<float, kVgprs> value{};
    std::array<int, kVgprs> writes{};
    std::array<bool, kVgprs> constant{};
    const auto mark = [&](int first, int n) {
        for (int k = first; k < first + n && k < kVgprs; ++k) {
            if (k >= 0) ++writes[k];
        }
    };
    for (const gcn::Inst& in : p.insts) {
        switch (in.enc) {
        case gcn::Enc::VOP1:
        case gcn::Enc::VOP2:
        case gcn::Enc::VOP3:
        case gcn::Enc::VINTRP: mark(in.dst, 1); break;  // a 64-bit result would take two, and none writes a factor
        case gcn::Enc::DS: mark(in.dst, 2); break;  // a read's destination; a write has none
        case gcn::Enc::MUBUF:
        case gcn::Enc::MTBUF:
            // Loads and atomics write vdata; stores (4-7, 24-31) read it.
            if (!((in.op >= 4 && in.op <= 7) || (in.op >= 24 && in.op <= 31))) mark(in.vdata, 4);
            break;
        case gcn::Enc::MIMG: mark(in.vdata, 4); break;
        default: break;
        }
        // v_mov_b32 of an inline float constant or a literal.
        if (in.enc == gcn::Enc::VOP1 && in.op == 1 && in.dst < kVgprs) {
            float f = 0.0f;
            bool ok = false;
            if (in.src0 >= gcn::kConstHalf && in.src0 <= gcn::kConstHalf + 7) {
                static const float inl[8] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f};
                f = inl[in.src0 - gcn::kConstHalf];
                ok = true;
            } else if (in.src0 == gcn::kLiteral && in.has_literal) {
                std::memcpy(&f, &in.literal, 4);
                ok = std::isfinite(f);
            }
            value[in.dst] = f;
            constant[in.dst] = ok;
        }
    }
    // The six factors a quad patch has, in the order they are written: four
    // edges, then the two inner ones.
    float factors[6] = {};
    int written = 0;
    for (const gcn::Inst& in : p.insts) {
        if (in.enc != gcn::Enc::MUBUF || in.op < 28 || in.op > 31) continue;  // buffer_store_dword..x4
        const int n = in.op == 28 ? 1 : in.op == 29 ? 2 : in.op == 31 ? 3 : 4;
        for (int k = 0; k < n; ++k) {
            const int reg = in.vdata + k;
            if (reg >= kVgprs || !constant[reg] || writes[reg] != 1) return false;
            if (written < 6) factors[written] = value[reg];
            ++written;
        }
    }
    if (written != 6) return false;
    for (int k = 0; k < 6; ++k) out[k] = factors[k];
    return true;
}

// A tessellated draw's state from its GX objects when they are
// there. A draw that took a token wrote no registers for the command stream to
// have them in, and these are the fields the shadow says its packets set.
const GpuDrawInputs* tess_gx(const GpuDraw& d) {
    return d.gx && d.gx_render && d.gx->tess ? d.gx : nullptr;
}

void probe_ls_input(std::uint64_t fetch_va, const std::uint32_t* user, std::uint64_t ds_va, const GxDrawObjects* gx);  // below

// BBHOST_TESS_PROBE=1 for a GX draw (its stages are in the GX inputs, not the
// register file probe_tessellated_draw reads): once per program set, its
// tessellation state and the four programs in tmp/tess/.
void probe_refused_gx_tess(const GpuDraw& d, const GpuDrawInputs& gi, const char* why) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_TESS_PROBE");
        return e && e[0] == '1';
    }();
    if (!on) return;
    const auto pgm = [](const std::uint32_t* p) {
        return (static_cast<std::uint64_t>(p[1] & 0xff) << 40) | (static_cast<std::uint64_t>(p[0]) << 8);
    };
    const std::uint64_t ls = pgm(gi.ls_pgm), hs = pgm(gi.hs_pgm), vs = pgm(gi.vs_pgm), ps = pgm(gi.ps_pgm);
    static std::mutex mu;
    static std::set<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>> seen;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (seen.size() >= 64 || !seen.insert({ls, hs, vs, ps}).second) return;
    }
    host_log("tess: GX draw (%s): LS 0x%llx rsrc %08x %08x, HS 0x%llx rsrc %08x %08x, DS 0x%llx rsrc %08x %08x, PS 0x%llx; "
             "VGT_LS_HS_CONFIG %08x VGT_TF_PARAM %08x HOS %g..%g",
             why, static_cast<unsigned long long>(ls), gi.ls_pgm[2], gi.ls_pgm[3], static_cast<unsigned long long>(hs), gi.hs_pgm[2],
             gi.hs_pgm[3], static_cast<unsigned long long>(vs), gi.vs_pgm[2], gi.vs_pgm[3], static_cast<unsigned long long>(ps),
             gi.ls_hs_config, gi.tf_param, reinterpret_cast<const float&>(gi.hos_min), reinterpret_cast<const float&>(gi.hos_max));
    {
        // The hull's user data, and the tessellation constants its V# in user
        // SGPRs 12-15 points at (strides, patch counts, output bases).
        // From the GX hull record (kGxRecordHs) when the draw has records,
        // else the register file (empty for a draw that took a token).
        std::uint32_t hu_built[16] = {};
        const std::uint32_t* hu = &d.sh[0x10c];
        if (d.gx_objects && d.gx_objects->records) {
            GxUserData ud;
            const GxStageRecords& hr = d.gx_objects->records->stage[kGxRecordHs];
            const GxStageRecords& dr = d.gx_objects->records->stage[kGxRecordSet0];
            std::string hd, dd;
            for (std::uint32_t i = 0; i < hr.ndesc && i < 16; ++i) { char b[16]; std::snprintf(b, sizeof(b), " %08x", hr.desc[i]); hd += b; }
            for (std::uint32_t i = 0; i < dr.ndesc && i < 16; ++i) { char b[16]; std::snprintf(b, sizeof(b), " %08x", dr.desc[i]); dd += b; }
            host_log("tess:   HS descriptors%s | DS descriptors%s", hd.c_str(), dd.c_str());
            // The Gnm shader structures (object +0x100): registers from +8,
            // then the stage's own constants (the LS stride; the hull's
            // control-point counts, patch constants and strides).
            for (int k = 0; k < 3; ++k) {
                const std::uint64_t obj = d.gx_objects->shader[k];
                std::uint64_t gnm = 0;
                std::uint32_t w[20] = {};
                if (obj && hle_kernel_va_mapped(obj + 0x100, 8)) std::memcpy(&gnm, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(obj + 0x100)), 8);
                if (gnm && hle_kernel_va_mapped(gnm, sizeof(w))) std::memcpy(w, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(gnm)), sizeof(w));
                std::string line;
                for (std::uint32_t v : w) { char b[16]; std::snprintf(b, sizeof(b), " %08x", v); line += b; }
                host_log("tess:   gnm %s 0x%llx:%s", k == 0 ? "LS" : k == 1 ? "HS" : "DS", static_cast<unsigned long long>(gnm), line.c_str());
            }
            if (build_gx_user_data(hr, ud, ~0u) == kUserBuilt) {
                std::memcpy(hu_built, ud.user, sizeof(hu_built));
                hu = hu_built;
            }
        }
        const std::uint64_t cb = static_cast<std::uint64_t>(hu[12]) | (static_cast<std::uint64_t>(hu[13] & 0xffff) << 32);
        std::uint32_t tc[16] = {};
        if (cb && hle_kernel_va_mapped(cb, sizeof(tc))) std::memcpy(tc, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cb)), sizeof(tc));
        host_log("tess:   HS user %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x -> 0x%llx: "
                 "%u %u %u %u | %u %u %u %u | %08x %08x %08x %08x | %08x %08x %08x %08x",
                 hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6], hu[7], hu[8], hu[9], hu[10], hu[11], hu[12], hu[13], hu[14], hu[15], static_cast<unsigned long long>(cb), tc[0], tc[1], tc[2], tc[3], tc[4], tc[5], tc[6], tc[7], tc[8], tc[9],
                 tc[10], tc[11], tc[12], tc[13], tc[14], tc[15]);
    }
    {
        const std::uint32_t* lu = &d.sh[0x14c];
        const std::uint32_t* vu = &d.sh[0x4c];
        const std::uint64_t tc_va = static_cast<std::uint64_t>(vu[8]) | (static_cast<std::uint64_t>(vu[9] & 0xffff) << 32);
        std::uint32_t tc[16] = {};
        if (tc_va && hle_kernel_va_mapped(tc_va, sizeof(tc))) std::memcpy(tc, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(tc_va)), sizeof(tc));
        host_log("tess:   LS user %08x %08x %08x %08x %08x %08x %08x %08x | DS user %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                 lu[0], lu[1], lu[2], lu[3], lu[4], lu[5], lu[6], lu[7], vu[0], vu[1], vu[2], vu[3], vu[4], vu[5], vu[6], vu[7], vu[8], vu[9],
                 vu[10], vu[11]);
        host_log("tess:   tess constants (DS user 8-9) 0x%llx: %u %u %u %u %u %u %u %u | %08x %08x %08x %08x %08x %08x %08x %08x",
                 static_cast<unsigned long long>(tc_va), tc[0], tc[1], tc[2], tc[3], tc[4], tc[5], tc[6], tc[7], tc[8], tc[9], tc[10], tc[11],
                 tc[12], tc[13], tc[14], tc[15]);
    }
    std::error_code ec;
    std::filesystem::create_directories("tmp/tess", ec);
    const std::pair<const char*, std::uint64_t> progs[4] = {{"ls", ls}, {"hs", hs}, {"vs", vs}, {"ps", ps}};
    for (const auto& [name, va] : progs) {
        if (!va || !hle_kernel_va_mapped(va, 4)) continue;
        char path[96];
        std::snprintf(path, sizeof(path), "tmp/tess/%s-%llx.bin", name, static_cast<unsigned long long>(va));
        const auto* w = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
        std::size_t n = 0;
        while (n < 4096 && hle_kernel_va_mapped(va + n * 4, 4) && w[n] != 0xbf810000u) ++n;
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(w, 4, n + 1, f);
            std::fclose(f);
        }
    }
}

// The Forbidden Woods' meshes: triangle patches of three control points
// whose hull shader is no passthrough - it reads the LS's control points out
// of LDS, writes its own outputs and patch constants after them and works the
// factors out. How that LDS is laid out is in the Gnm shader structures the
// GX objects point at (object +0x100), after each stage's registers: the LS's
// stride (dword 7) and the hull's control-point counts, patch-constant count,
// control-point stride and first edge factor index (dwords 9-15). The hull
// shaders hardcode the same strides; the domain shaders read them from the
// tessellation constants, which this builds for one patch.
bool gx_hull_layout(const GpuDraw& d, const GpuDrawInputs& gi, TessDraw& t) {
    const GxDrawObjects& o = *d.gx_objects;
    const auto gnm = [](std::uint64_t obj, std::uint32_t* w, std::size_t n) {
        std::uint64_t at = 0;
        if (!obj || !hle_kernel_va_mapped(obj + 0x100, 8)) return false;
        std::memcpy(&at, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(obj + 0x100)), 8);
        if (!at || !hle_kernel_va_mapped(at, n * 4)) return false;
        std::memcpy(w, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), n * 4);
        return true;
    };
    std::uint32_t ls[8] = {}, hs[16] = {};
    if (!gnm(o.shader[0], ls, 8) || !gnm(o.shader[1], hs, 16)) return false;
    // The structures belong to the programs this draw runs.
    if (ls[2] != gi.ls_pgm[0] || hs[2] != gi.hs_pgm[0]) return false;
    const std::uint32_t in_cp = (gi.ls_hs_config >> 8) & 0x3f, out_cp = (gi.ls_hs_config >> 14) & 0x3f;
    const std::uint32_t ls_stride = ls[7], cp_stride = hs[12], npc = hs[11];
    if (hs[9] != in_cp || hs[10] != out_cp || in_cp != out_cp || in_cp < 1 || in_cp > 32) return false;
    if (!ls_stride || ls_stride % 16 || ls_stride > 0x400 || cp_stride % 16 || cp_stride > 0x400 || npc > 64) return false;
    if (!(g.subgroup_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) return false;  // exec is a ballot
    const std::uint32_t inputs = in_cp * ls_stride, outputs = out_cp * cp_stride, consts = npc * 16;
    // TessellationDataConstantBuffer for a threadgroup of one patch.
    const std::uint32_t tc[9] = {ls_stride, cp_stride, 1, inputs, consts, inputs + outputs, outputs, 0, hs[15]};
    std::memcpy(t.tess_constants, tc, sizeof(tc));
    t.hull = true;
    t.window = (inputs + outputs + consts + 15) & ~15u;
    t.hs_rsrc1 = gi.hs_pgm[2];
    t.hs_rsrc2 = gi.hs_pgm[3];
    // VGT_TF_PARAM: partitioning in bits 2-4 (integer, pow2, odd, even),
    // the output primitive in 5-7 (2 clockwise triangles, 3 counter-clockwise).
    const std::uint32_t part = (gi.tf_param >> 2) & 7, topo = (gi.tf_param >> 5) & 7;
    t.quads = (gi.tf_param & 3) == 2;
    t.spacing = part == 2 ? 1 : part == 3 ? 2 : 0;
    t.cw = topo != 3;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1, std::memory_order_relaxed) < 4) {
        host_log("render: the game's hull shader at 0x%llx runs as the control stage: %u control points, LS stride %u, CP stride %u, "
                 "%u patch constants, window %u bytes, VGT_TF_PARAM %08x",
                 static_cast<unsigned long long>((static_cast<std::uint64_t>(gi.hs_pgm[1] & 0xff) << 40) |
                                                 (static_cast<std::uint64_t>(gi.hs_pgm[0]) << 8)),
                 in_cp, ls_stride, cp_stride, npc, t.window, gi.tf_param);
    }
    return true;
}

bool tess_plan(const GpuDraw& d, TessDraw& t) {
    const GpuDrawInputs* gi = tess_gx(d);
    const std::uint32_t stages = gi ? gi->stages : d.ctx[0x2d5];
    const std::uint32_t cfg = gi ? gi->ls_hs_config : d.ctx[0x2d6];
    const std::uint32_t tf = gi ? gi->tf_param : d.ctx[0x2db];
    const std::uint32_t prim = gi ? 9u : d.uconfig[0x242];
    const char* why = nullptr;
    const std::uint32_t in_cp = (cfg >> 8) & 0x3f, out_cp = (cfg >> 14) & 0x3f;
    // A patch of more than one control point is the game's own hull shader
    // at work (the Forbidden Woods' meshes), which runs as the control stage
    // when its GX objects say how its LDS is laid out.
    const bool many = in_cp != 1 || out_cp != 1;
    static const bool hull_on = [] {  // BBHOST_TESS_HULL=0: refuse them, as before 2026-10-02
        const char* e = std::getenv("BBHOST_TESS_HULL");
        return !(e && e[0] == '0');
    }();
    if ((stages & 3) != 1 || !(stages & 4) || (stages & 0x38) || ((stages >> 6) & 3) != 1) why = "stages other than LS, HS and a domain VS";
    else if (prim != 9) why = "not a patch list";
    else if (many && (!hull_on || !gi || !d.gx_objects || (tf & 3) != 1 || !gx_hull_layout(d, *gi, t))) why = "more than one control point a patch";
    else if (!many && (tf & 3) != 2) why = "not a quad domain";
    else if (d.indirect_va) why = "an indirect draw";
    if (why) {
        g_tess_refused.fetch_add(1, std::memory_order_relaxed);
        if (gi) probe_refused_gx_tess(d, *gi, why);
        static std::mutex mu;
        static std::set<std::string> logged;
        std::lock_guard<std::mutex> lk(mu);
        if (logged.size() < 16 && logged.insert(why).second) {
            host_log("render: tessellated draw not drawn: %s (stages %08x, VGT_LS_HS_CONFIG %08x, VGT_TF_PARAM %08x, prim %u)", why, stages,
                     cfg, tf, prim);
        }
        return false;
    }
    if (t.hull && gi) probe_refused_gx_tess(d, *gi, "the game's hull runs it");  // BBHOST_TESS_PROBE: its stages too
    t.count = d.index_count;
    t.control_points = std::max<std::uint32_t>(1, in_cp);
    if (t.hull) {
        t.count -= t.count % t.control_points;  // whole patches
        t.instances = std::max<std::uint32_t>(1, d.instance_count);
    }
    t.hs_va = gi ? (static_cast<std::uint64_t>(gi->hs_pgm[1] & 0xff) << 40) | (static_cast<std::uint64_t>(gi->hs_pgm[0]) << 8)
                 : (static_cast<std::uint64_t>(d.sh[0x109] & 0xff) << 40) | (static_cast<std::uint64_t>(d.sh[0x108]) << 8);
    float lo = 1.0f, hi = 1.0f;
    std::memcpy(&hi, gi ? &gi->hos_max : &d.ctx[0x286], 4);  // VGT_HOS_MAX_TESS_LEVEL
    std::memcpy(&lo, gi ? &gi->hos_min : &d.ctx[0x287], 4);  // VGT_HOS_MIN_TESS_LEVEL
    t.clamp_lo = std::isfinite(lo) ? lo : 1.0f;
    t.clamp_hi = std::isfinite(hi) ? hi : 1.0f;
    return t.count > 0;
}

// Pass 1: the LS over the draw's control points, into a region of the LDS ring.
// Cells a side for this draw's patches: the hull shader's constant factor,
// clamped as the hardware clamps it. BBHOST_TESS_LEVEL=<n> overrides it.
std::uint32_t tess_level_locked(const TessDraw& t) {
    static const std::uint32_t level_env = [] {
        const char* e = std::getenv("BBHOST_TESS_LEVEL");
        return e ? static_cast<std::uint32_t>(std::clamp(std::atoi(e), 1, 8)) : 0u;
    }();
    if (level_env) return level_env;
    static std::unordered_map<std::uint64_t, std::pair<float, float>> factors;  // outer, inner by hull shader hash
    CachedProgram* hs = program_at(t.hs_va);
    if (!hs) return 1;
    auto it = factors.find(hs->hash);
    if (it == factors.end()) {
        const std::pair<float, float> f = hs_constant_factor(gcn::decode(hs->words.data(), hs->words.size()));
        it = factors.emplace(hs->hash, f).first;
        host_log("render: tessellation factors %g outer, %g inner from %s at 0x%llx%s", f.first, f.second, hs->name.c_str(),
                 static_cast<unsigned long long>(t.hs_va), f.first > 0 ? "" : " (not constant: one quad a patch)");
    }
    // An inner factor above 1 would put points inside the patch, which the
    // fan the domain prologue draws does not have: one quad a patch instead.
    if (it->second.first <= 0.0f || it->second.second > 1.0f) return 1;
    const float f = std::clamp(it->second.first, std::max(1.0f, t.clamp_lo), std::max(1.0f, t.clamp_hi));
    return static_cast<std::uint32_t>(std::clamp(std::ceil(f), 1.0f, 8.0f));
}

// BBHOST_TESS_HW=1 draws these patches with the host's own
// tessellator instead of the topology the domain prologue invents. It needs
// the device to support tessellation and the hull shader's factors to be
// constants in it, which is what these hull shaders write; anything else keeps
// the emulated path.
bool tess_hw_locked(TessDraw& t) {
    // On by default where the device has a tessellator: it draws the patches
    // the emulation had to refuse (90% of them), it ends the frames that
    // washed out, and with the LS as the pipeline's vertex stage it costs
    // nothing measurable - 11.99 ms of main-loop work a frame against 12.14
    // for the emulation, which is inside the run-to-run spread.
    // BBHOST_TESS_HW=0 goes back to the emulation, 1 keeps the LS in its own
    // compute pass (which costs 8 ms), 2 is the default.
    static const int want = [] {
        const char* e = std::getenv("BBHOST_TESS_HW");
        return e ? std::atoi(e) : 2;
    }();
    if (want <= 0 || !g.has_tessellation) return false;
    t.attrs = want >= 2;
    if (t.hull) {
        // The game's own hull: its factors are its business, and its LDS is
        // a buffer the LS pass writes.
        t.attrs = false;
        return true;
    }
    static std::unordered_map<std::uint64_t, std::array<float, 6>> cache;  // by hull shader hash
    CachedProgram* hs = program_at(t.hs_va);
    if (!hs) return false;
    auto it = cache.find(hs->hash);
    if (it == cache.end()) {
        std::array<float, 6> f{};
        const bool ok = hs_constant_factors(gcn::decode(hs->words.data(), hs->words.size()), f.data());
        if (!ok) f.fill(0.0f);
        it = cache.emplace(hs->hash, f).first;
        host_log("render: host tessellation for %s at 0x%llx: factors %g %g %g %g outer, %g %g inner%s", hs->name.c_str(),
                 static_cast<unsigned long long>(t.hs_va), f[0], f[1], f[2], f[3], f[4], f[5],
                 ok ? "" : " (not constant: the emulated path)");
    }
    if (it->second[0] <= 0.0f) return false;
    // The hardware clamps what the hull shader asks for.
    const float lo = std::max(1.0f, t.clamp_lo), hi = std::max(1.0f, t.clamp_hi);
    for (int k = 0; k < 4; ++k) t.outer[k] = std::clamp(it->second[k], lo, hi);
    for (int k = 0; k < 2; ++k) t.inner[k] = std::clamp(it->second[4 + k], lo, hi);
    return true;
}

// A tessellated GX draw's LS user data from its records (kGxRecordLs), as
// the LS-as-vertex-stage path builds it: what its descriptors name, its fetch
// shader where the program reads it, and the vertex table GX names from the
// ring. A draw that took a token wrote no registers to read it from instead.
bool gx_ls_user_locked(const GpuDraw& d, const CachedProgram& ls, std::uint32_t out[16]) {
    if (!d.gx_objects || !d.gx_objects->records) return false;
    GxUserData u;
    if (build_gx_user_data(d.gx_objects->records->stage[kGxRecordLs], u, ~0u) != kUserBuilt) return false;
    std::memset(out, 0, 16 * sizeof(std::uint32_t));
    for (int k = 0; k < 16; ++k) {
        if ((u.set >> k) & 1) out[k] = u.user[k];
    }
    const int fs = ls.fetch_sgpr;
    if (fs >= 0 && fs + 1 < 16 && d.gx_objects->fetch_va) {
        out[fs] = static_cast<std::uint32_t>(d.gx_objects->fetch_va);
        out[fs + 1] = static_cast<std::uint32_t>(d.gx_objects->fetch_va >> 32);
    }
    if (d.gx_objects->vtx_ud < 15 && d.gx_objects->vtx_valid) {
        const std::uint64_t table = place_vertex_table(*d.gx_objects);
        if (!table) return false;
        out[d.gx_objects->vtx_ud] = static_cast<std::uint32_t>(table);
        out[d.gx_objects->vtx_ud + 1] = static_cast<std::uint32_t>(table >> 32);
    }
    return true;
}

// Why a hull draw's LS pass gave up, the first times each reason comes up.
bool tess_hull_refused(const char* why) {
    static std::mutex mu;
    static std::map<std::string, std::uint64_t> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen[why]++ < 2) host_log("render: the game's hull: no LS pass (%s)", why);
    return false;
}

bool tess_ls_pass_locked(const GpuDraw& d, TessDraw& t) {
    const GpuDrawInputs* plan_gx = tess_gx(d);
    t.hw = tess_hw_locked(t);
    t.level = tess_level_locked(t);
    if (t.hull && !t.hw) {
        // Nothing but the host's tessellator can run the game's hull.
        g_tess_refused.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // A patch whose hull shader asks for a finer mesh than we draw is left
    // out rather than drawn as one quad: its domain shader bends the patch
    // around its middle (a ribbon), and the corners alone put the wrong shape
    // on the screen - a flare that washes the frame out near a fire. Drawing
    // the fan it asks for freezes the game a few seconds in, with the guest's
    // own loop still running while it stops asking for frames and the
    // renderer, the GPU and the presenting thread all idle, so the
    // finer levels wait for that (BBHOST_TESS_LEVEL forces one).
    if (t.level > kTessLevelCap && !t.hw) {
        g_tess_refused.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) == 0) {
            host_log("render: patches asking for %u cells a side are not drawn (BBHOST_TESS_LEVEL=%u draws them)", t.level, t.level);
        }
        t.level = kTessLevelCap;
        return false;
    }
    // BBHOST_TESS_SKIP=<hash>[,<hash>]: leave those domain shaders' patches
    // undrawn, and BBHOST_TESS_ONLY=<hash>[,...] draws only theirs - for
    // finding which effect a bad patch belongs to.
    static const std::string skip = [] {
        const char* e = std::getenv("BBHOST_TESS_SKIP");
        return std::string(e ? e : "");
    }();
    static const std::string only = [] {
        const char* e = std::getenv("BBHOST_TESS_ONLY");
        return std::string(e ? e : "");
    }();
    {
        const std::uint64_t ds_va =
            plan_gx ? (static_cast<std::uint64_t>(plan_gx->vs_pgm[1] & 0xff) << 40) | (static_cast<std::uint64_t>(plan_gx->vs_pgm[0]) << 8)
                    : (static_cast<std::uint64_t>(d.sh[0x49] & 0xff) << 40) | (static_cast<std::uint64_t>(d.sh[0x48]) << 8);
        CachedProgram* ds = program_at(ds_va);
        if (ds) {
            static const bool zpass_trace = [] {
                const char* e = std::getenv("BBHOST_ZPASS_TRACE");
                return e && e[0] == '1';
            }();
            if (zpass_trace) {
                static std::atomic<int> dlogs{0};
                if (dlogs.fetch_add(1) < 60) {
                    host_log("zpass: patches from %s at draw %llu (%u patches)", ds->name.c_str(),
                             static_cast<unsigned long long>(hle_gnm_draw_count()), t.count);
                }
            }
            static std::set<std::string> seen;
            if (seen.size() < 64 && seen.insert(ds->name).second) {
                host_log("render: tessellated patches from domain shader %s at 0x%llx, %u cells a side", ds->name.c_str(),
                         static_cast<unsigned long long>(ds_va), t.level);
            }
        }
        if (ds && (!skip.empty() || !only.empty())) {
            const bool named = !ds->name.empty() &&
                               (!skip.empty() ? skip.find(ds->name) != std::string::npos : only.find(ds->name) != std::string::npos);
            if (!skip.empty() ? named : !named) return false;
        }
    }
    {
        std::uint32_t was = g_tess_max_patches.load(std::memory_order_relaxed);
        while (t.count > was && !g_tess_max_patches.compare_exchange_weak(was, t.count, std::memory_order_relaxed)) {
        }
        was = g_tess_max_level.load(std::memory_order_relaxed);
        while (t.level > was && !g_tess_max_level.compare_exchange_weak(was, t.level, std::memory_order_relaxed)) {
        }
        g_tess_vertices.fetch_add(static_cast<std::uint64_t>(t.count) * tess_vertices_a_patch(t.level), std::memory_order_relaxed);
    }
    const auto pgm = [&](std::uint32_t lo) {
        return (static_cast<std::uint64_t>(d.sh[lo + 1] & 0xff) << 40) | (static_cast<std::uint64_t>(d.sh[lo]) << 8);
    };
    const std::uint32_t* user = &d.sh[0x14c];  // SPI_SHADER_USER_DATA_LS_0..15
    const std::uint32_t rsrc1 = plan_gx ? plan_gx->ls_pgm[2] : d.sh[0x14a];
    const std::uint32_t rsrc2 = plan_gx ? plan_gx->ls_pgm[3] : d.sh[0x14b];
    CachedProgram* ls = program_at(plan_gx ? (static_cast<std::uint64_t>(plan_gx->ls_pgm[1] & 0xff) << 40) |
                                                 (static_cast<std::uint64_t>(plan_gx->ls_pgm[0]) << 8)
                                           : pgm(0x148));
    if (!ls) return false;
    if (ls->fetch_sgpr == -2) ls->fetch_sgpr = fetch_user_sgpr(gcn::decode(ls->words.data(), ls->words.size()));
    std::uint32_t gx_ls_user[16];
    if (t.hull) {
        if (!gx_ls_user_locked(d, *ls, gx_ls_user)) return tess_hull_refused("the LS's user data");
        user = gx_ls_user;
    }
    const FetchProgram* fetch = nullptr;
    if (ls->fetch_sgpr >= 0 && ls->fetch_sgpr + 1 < 16) {
        // A draw that took a token wrote no user-data registers, so GX names
        // its fetch shader instead of the command stream holding it.
        const std::uint64_t fva =
            plan_gx && d.gx_objects
                ? d.gx_objects->fetch_va
                : static_cast<std::uint64_t>(user[ls->fetch_sgpr]) | (static_cast<std::uint64_t>(user[ls->fetch_sgpr + 1]) << 32);
        if (!(fetch = fetch_program_at(fva))) return false;
    }
    std::uint64_t stride = 1024, base = 0;
    {
        const std::uint64_t tc_va = static_cast<std::uint64_t>(d.sh[0x54]) | (static_cast<std::uint64_t>(d.sh[0x55] & 0xffff) << 32);
        std::uint32_t tc[8] = {};
        if (tc_va && hle_kernel_va_mapped(tc_va, sizeof(tc))) {
            std::memcpy(tc, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(tc_va)), sizeof(tc));
            stride = std::clamp<std::uint64_t>(std::max({tc[0], tc[1], tc[6]}), 16, 65536);
            base = std::min<std::uint64_t>(tc[3], 65536);
        }
    }

    // With the control point carried through the pipeline there
    // is nothing for a compute pass to do. Take what the vertex stage needs of
    // the LS and leave - no pipeline, no ring, no dispatch, and no render pass
    // ended to make room for one.
    if (t.hw && t.attrs) {
        // One slot per 16 bytes of the control point. The stride is 0x80 or
        // 0x90 in this game, and the index is taken modulo the count, so a
        // fixed sixteen slots (256 bytes) covers both, addresses inside the
        // patch still land where they belong, and the count does not depend on
        // the tessellation constants - which a draw that took a token has no
        // register to point at. Anything wider than that is refused rather
        // than folded onto itself.
        // The stride comes from the tessellation constants, which a draw that
        // took a token has no register pointing at - and the attribute count
        // does not depend on it. Only check it where it is actually known.
        if (!plan_gx && stride > 16 * 16) {
            g_tess_refused.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) == 0) {
                host_log("render: a control point of %llu bytes is wider than the attributes carry",
                         static_cast<unsigned long long>(stride));
            }
            return false;
        }
        t.attr_vec4s = 16;
        const GpuDrawInputs* gi = tess_gx(d);
        t.ls_va = gi ? (static_cast<std::uint64_t>(gi->ls_pgm[1] & 0xff) << 40) | (static_cast<std::uint64_t>(gi->ls_pgm[0]) << 8)
                     : pgm(0x148);
        // The fetch shader: GX names it directly, and the command stream keeps
        // it in the slot the LS's own program reads.
        t.ls_fetch_va = gi && d.gx_objects ? d.gx_objects->fetch_va : 0;
        if (!gi && ls->fetch_sgpr >= 0 && ls->fetch_sgpr + 1 < 16) {
            t.ls_fetch_va = static_cast<std::uint64_t>(user[ls->fetch_sgpr]) |
                            (static_cast<std::uint64_t>(user[ls->fetch_sgpr + 1]) << 32);
        }
        t.ls_rsrc1 = gi ? gi->ls_pgm[2] : rsrc1;
        t.ls_rsrc2 = gi ? gi->ls_pgm[3] : rsrc2;
        std::memcpy(t.ls_user, user, sizeof(t.ls_user));
        probe_ls_input(t.ls_fetch_va, user,
                       plan_gx ? (static_cast<std::uint64_t>(plan_gx->vs_pgm[1] & 0xff) << 40) | (static_cast<std::uint64_t>(plan_gx->vs_pgm[0]) << 8)
                               : (static_cast<std::uint64_t>(d.sh[0x49] & 0xff) << 40) | (static_cast<std::uint64_t>(d.sh[0x48]) << 8),
                       plan_gx ? d.gx_objects : nullptr);
        return true;
    }

    struct LsPipe {
        bool failed = false;
        VkPipeline pipeline = VK_NULL_HANDLE;
        bool standin = false;  // unoptimized until bb-cs-optimize has built it (first_compute_pipeline)
        std::uint64_t opt_seen = 0;
        gcn::TranslateResult meta;
        std::string name;
    };
    static std::unordered_map<std::uint64_t, LsPipe> pipes;
    const std::uint64_t fetch_hash = fetch ? fetch->hash : 0;
    const std::uint32_t rs[4] = {rsrc1, rsrc2, t.hull ? t.window : 0u, t.hull ? t.control_points : 0u};
    const std::uint64_t key = fnv1a(rs, sizeof(rs), fnv1a(&fetch_hash, 8, ls->hash ^ 0x15c0ull));
    LsPipe& pl = pipes[key];
    constexpr std::uint64_t kLsOptimizeTag = 0x15c0000000000000ull;  // its optimized build's key, apart from the dispatches'
    if (pl.standin) adopt_optimized_compute(key ^ kLsOptimizeTag, pl.pipeline, pl.standin, pl.opt_seen);
    if (!pl.pipeline && !pl.failed) {
        pl.name = ls->name + "-ls";
        const gcn::Program prog = gcn::decode(ls->words.data(), ls->words.size());
        gcn::Program fprog;
        if (fetch) fprog = gcn::decode(fetch->words.data(), fetch->words.size());
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Compute;
        o.tess_role = gcn::TranslateOptions::TessRole::LsCompute;
        o.rsrc1 = rsrc1;
        o.rsrc2 = rsrc2;
        o.cs_threads[0] = 64;
        o.fetch = fetch ? &fprog : nullptr;
        o.exec_known = g.exec_known;
        o.tess_lds_bound = tess_lds_bound();
        if (t.hull) {
            o.tess_window = t.window;
            o.tess_patch_control_points = t.control_points;
        }
        pl.meta = translate_cached(prog, o);
        VkShaderModule module = VK_NULL_HANDLE;
        if (!prog.errors.empty() || !pl.meta.ok()) {
            host_log("render: tessellation LS %s: %s", pl.name.c_str(),
                     !prog.errors.empty() ? prog.errors[0].what.c_str() : pl.meta.errors[0].c_str());
            pl.failed = true;
        } else if (!make_module(pl.meta.spirv, module)) {
            pl.failed = true;
        } else {
            dump_spirv(pl.name, pl.meta.spirv);
            pl.pipeline = first_compute_pipeline(module, key ^ kLsOptimizeTag, pl.name, pl.meta.spirv, pl.standin, pl.name);
            if (!pl.pipeline) pl.failed = true;
            vkDestroyShaderModule(g.device, module, nullptr);
            host_log("render: tessellation LS %s%s (%zu images)", pl.name.c_str(), pl.failed ? " failed" : " ready", pl.meta.images.size());
        }
    }
    if (pl.failed) return false;
    // Room for every control point and the invocations past the count
    // (which write slack slots), at the larger of the LS's and the domain
    // shader's strides as the tessellation constants give them (the V# in the
    // domain shader's user SGPRs 8-11: LS stride, CP stride, patches, base,
    // patch-constant size and base, patch stride).
    // The game's own hull: a window a patch, the invocations past the count
    // (up to a workgroup) in the windows after the last.
    const std::uint64_t bytes = t.hull ? ((static_cast<std::uint64_t>(t.count) * t.instances + 64) / t.control_points + 2) * t.window + 256
                                       : (static_cast<std::uint64_t>(t.count) + 64) * stride + base + 256;
    // The ring belongs to the recordings in flight: a region has to hold its
    // patches until the draw that reads them has run, and that draw can end up
    // in a later command buffer than the dispatch. Each recording with
    // tessellated draws takes the next region, and a region is reused only
    // once the submission it went into and the two after it have run.
    // Sharing one offset let a later dispatch overwrite patches an earlier
    // draw had not read yet, which drew one effect with another's particles -
    // huge quads across the screen. (Regions went round with the recordings,
    // twice as many as submissions in flight, until 16 submissions in flight
    // would have made them 8 MiB.)
    static DevBuffer ring;
    constexpr std::uint64_t kRingBytes = kTessRingBytes;
    constexpr std::uint64_t kRegions = kTessRegions;
    constexpr std::uint64_t kRegionBytes = kTessRegionBytes;
    struct Region {
        std::uint64_t off = 0, serial = ~0ull;  // Gpu::record_serial of the recording it belongs to
        std::uint64_t submission = 0;           // Gpu::flushes when that recording began: its submission serial
    };
    static Region regions[kRegions];
    static std::uint64_t taken = 0;  // regions handed out
    // Device-local: the stages behind it read and write it all draw long - the
    // game's own hull shaders and their domain shaders, at up to 16 cells a
    // side, cost 26 ms of GPU a frame in the Forbidden Woods with it in host
    // memory. Host-visible only for BBHOST_TESS_DEBUG and
    // BBHOST_DUMP_ON_BRIGHT, which read it back.
    static const bool ring_read_back = std::getenv("BBHOST_TESS_DEBUG") || std::getenv("BBHOST_DUMP_ON_BRIGHT");
    if (!ring.buffer) {
        if (!create_dev_buffer(ring, kRingBytes, ring_read_back, ring_read_back)) return false;
        std::lock_guard<std::mutex> lk(g_tess_lds_mu);
        g_tess_ring_address = ring.address;
    }
    if (bytes > kRegionBytes) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 4) {
            host_log("render: a tessellated draw needs %llu bytes of LDS ring, more than a region's %llu: not drawn",
                     static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(kRegionBytes));
        }
        return false;
    }
    begin_recording_locked();  // this recording's serial is the one the region belongs to
    Region* region = taken ? &regions[(taken - 1) % kRegions] : nullptr;
    if (region && region->serial == g.record_serial && region->off + bytes > kRegionBytes) {
        submit_locked();  // the next recording, with a region of its own
        begin_recording_locked();
    }
    if (!region || region->serial != g.record_serial) {
        region = &regions[taken++ % kRegions];
        if (region->serial != ~0ull) wait_submission_locked(region->submission + 2);
        *region = {0, g.record_serial, g.flushes};
    }
    t.patch_stride = static_cast<std::uint32_t>(stride);
    t.patch_base = static_cast<std::uint32_t>(base);
    {
        // The projection's w row, from the domain shader's constant buffer
        // (its V# is in the vertex stage's user SGPRs 4-7, dwords 12-15).
        const std::uint64_t cb = static_cast<std::uint64_t>(d.sh[0x50]) | (static_cast<std::uint64_t>(d.sh[0x51] & 0xfff) << 32);
        if (cb && hle_kernel_va_mapped(cb + 48, sizeof(t.cull_w))) {
            std::memcpy(t.cull_w, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cb + 48)), sizeof(t.cull_w));
        }
    }
    t.lds_address = ring.address + static_cast<std::uint64_t>(region - regions) * kRegionBytes + region->off;
    t.lds_bytes = static_cast<std::uint32_t>(bytes);  // at most a region (checked above)
    region->off = (region->off + bytes + 255) & ~255ull;
    {
        std::lock_guard<std::mutex> lk(g_tess_lds_mu);
        g_tess_lds_uses[g_tess_lds_next++ % kTessLdsUses] = {g_draw_rec_next, t.lds_address, bytes, t.hull ? t.window : 0u};
    }
    if (t.hull) {
        // The tessellation constants for one patch, and the hull's user data
        // with their V# in it.
        const std::uint64_t tc = ring_place(reinterpret_cast<const std::uint8_t*>(t.tess_constants), sizeof(t.tess_constants), 16);
        if (!tc) return tess_hull_refused("ring room for the tessellation constants");
        t.tess_vsharp[0] = static_cast<std::uint32_t>(tc);
        t.tess_vsharp[1] = static_cast<std::uint32_t>((tc >> 32) & 0xffff) | 16u << 16;  // stride 16
        t.tess_vsharp[2] = sizeof(t.tess_constants) / 16;
        t.tess_vsharp[3] = 0x00077fac;  // x, y, z, w; 32-bit float, as the game's constant buffers are
        GxUserData hu;
        t_tess_constants = t.tess_vsharp;
        const GxUserResult hr = build_gx_user_data(d.gx_objects->records->stage[kGxRecordHs], hu, ~0u);
        t_tess_constants = nullptr;
        if (hr != kUserBuilt) return tess_hull_refused("the hull's user data");
        std::memset(t.hs_user, 0, sizeof(t.hs_user));
        for (int k = 0; k < 16; ++k) {
            if ((hu.set >> k) & 1) t.hs_user[k] = hu.user[k];
        }
    }

    static thread_local StageImages images;
    prefetch_stage_images(pl.meta, user, images);
    if (g.queued + 1 >= kMaxQueued * kStageSlots) submit_locked();
    begin_recording_locked();
    transfer_flush_locked();
    render_end_pass_locked();
    gcn::StageParams params{};
    params.l1_table = g.l1.address;
    params.lds_address = t.lds_address;
    params.lds_bytes = t.lds_bytes;
    std::memcpy(params.user_sgpr, user, sizeof(params.user_sgpr));
    params.vertex_formats[gcn::kTessIndexLo] = static_cast<std::uint32_t>(d.index_va);
    params.vertex_formats[gcn::kTessIndexHi] = static_cast<std::uint32_t>(d.index_va >> 32);
    params.vertex_formats[gcn::kTessCount] = t.count;
    params.vertex_formats[gcn::kTessInstances] = t.hull ? t.instances : 1;
    params.vertex_formats[gcn::kTessFirstInstance] = t.hull ? t.first_instance : 0;
    params.vertex_formats[gcn::kTessFirstPoint] = t.hull ? t.first_point : 0;
    params.vertex_formats[gcn::kTessIndexType] = d.index_type;
    // VGT_INDX_OFFSET: the GX inputs' for a token draw (the register file is not kept for draws).
    params.vertex_formats[gcn::kTessBaseVertex] = d.gx && d.gx_render ? static_cast<std::uint32_t>(d.gx->base_vertex) : d.ctx[0x102];
    VkDescriptorBufferInfo ubi{};
    alloc_params_slot_locked(params, ubi);
    VkDescriptorSet set = alloc_set_locked();
    if (!set) return false;
    std::vector<VkWriteDescriptorSet> writes;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = gcn::kBindingParams;
    w.descriptorCount = 1;
    w.descriptorType = params_descriptor_type();
    w.pBufferInfo = set_cache_on() ? &g_params_desc : &ubi;
    writes.push_back(w);
    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(pl.meta.images.size() + pl.meta.samplers.size() + 1);
    bind_stage_images(set, pl.meta, images, writes, infos, pl.name.c_str());
    vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    // BBHOST_TESS_CLEAR=1: zero the region first, so a patch the LS does not
    // write draws as a degenerate quad instead of whatever the ring held.
    static const bool clear_region = [] {
        const char* e = std::getenv("BBHOST_TESS_CLEAR");
        return e && e[0] == '1';
    }();
    if (clear_region) {
        vkCmdFillBuffer(g_cmd(), ring.buffer, t.lds_address - ring.address, bytes, 0);
        VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fb, 0, nullptr, 0,
                             nullptr);
    }
    vkCmdBindPipeline(g_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline);
    const std::uint32_t params_offset = static_cast<std::uint32_t>(ubi.offset);
    vkCmdBindDescriptorSets(g_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe_layout, 0, 1, &set, set_cache_on() ? 1 : 0, &params_offset);
    vkCmdDispatch(g_cmd(), (t.count * (t.hull ? t.instances : 1) + 63) / 64, 1, 1);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    // BBHOST_TESS_DEBUG=4: where each patch of a draw lands on the screen,
    // to find the one that fills the frame. The domain shader builds a world
    // position out of the patch's three rows and the domain coordinate -
    // component k is row k dotted with (u, v, dword 2, dword 3), dwords 0-3
    // being the offsets - and projects it with the matrix in its own
    // constants, so the same arithmetic here gives the quad's size in clip
    // space. Read a flip later, when the LS pass has certainly run, rather
    // than flushing the GPU for every draw.
    static const int scan = [] {
        const char* e = std::getenv("BBHOST_TESS_DEBUG");
        return e ? std::atoi(e) : 0;
    }();
    if (ring.map && (scan == 4 || g_dump_patches.load(std::memory_order_relaxed))) {
        struct Pending {
            std::uint64_t at, flip, cb;
            std::uint32_t count, stride;
            std::string name, ds;
        };
        static std::deque<Pending> pending;
        const std::uint64_t flip = hle_video_flip_count();
        static std::uint64_t scanned_flip = 0;
        const std::uint64_t cb_va = static_cast<std::uint64_t>(d.sh[0x50]) | (static_cast<std::uint64_t>(d.sh[0x51] & 0xfff) << 32);
        const std::uint64_t ds_at = (static_cast<std::uint64_t>(d.sh[0x49] & 0xff) << 40) | (static_cast<std::uint64_t>(d.sh[0x48]) << 8);
        CachedProgram* ds_prog = program_at(ds_at);
        pending.push_back({t.lds_address, flip, cb_va, t.count, static_cast<std::uint32_t>(stride), pl.name,
                           ds_prog ? ds_prog->name : std::string("?")});
        // Scan a flip's draws together, once that flip is over and the GPU
        // has been waited for: a region is another draw's a dozen recordings
        // later, so reading one late reads somebody else's patches.
        if (flip == scanned_flip || pending.front().flip >= flip) return true;
        scanned_flip = flip;
        flush_locked();
        while (!pending.empty() && pending.front().flip < flip) {
            const Pending p = pending.front();
            pending.pop_front();
            float m[16] = {};
            if (!p.cb || !hle_kernel_va_mapped(p.cb, sizeof(m))) continue;
            std::memcpy(m, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(p.cb)), sizeof(m));
            const auto* region = static_cast<const std::uint8_t*>(ring.map) + (p.at - ring.address);
            float worst = 0.0f, worst_w = 0.0f;
            std::uint32_t worst_k = 0, all_zero = 0;
            for (std::uint32_t k = 0; k < p.count; ++k) {
                const auto* f = reinterpret_cast<const float*>(region + static_cast<std::size_t>(k) * p.stride);
                bool zero = true;
                for (std::uint32_t w = 0; w < p.stride / 4 && zero; ++w) zero = f[w] == 0.0f;
                if (zero) {
                    ++all_zero;
                    continue;
                }
                float lo_x = 1e30f, hi_x = -1e30f, lo_y = 1e30f, hi_y = -1e30f, near_w = 1e30f;
                bool ok = true;
                for (int corner = 0; corner < 4 && ok; ++corner) {
                    const float cu = (corner & 1) ? 0.5f : -0.5f, cv = (corner & 2) ? 0.5f : -0.5f;
                    float c[3] = {};
                    for (int r = 0; r < 3; ++r) {
                        const float* row = f + 12 + r * 4;
                        c[r] = row[0] * (f[0] + cu) + row[1] * (f[1] + cv) + row[2] * f[2] + row[3] * f[3];
                    }
                    const float x = m[0] * c[0] + m[1] * c[1] + m[2] * c[2] + m[3];
                    const float y = m[4] * c[0] + m[5] * c[1] + m[6] * c[2] + m[7];
                    const float w = m[12] * c[0] + m[13] * c[1] + m[14] * c[2] + m[15];
                    ok = std::isfinite(x) && std::isfinite(y) && std::isfinite(w);
                    lo_x = std::min(lo_x, x / w);
                    hi_x = std::max(hi_x, x / w);
                    lo_y = std::min(lo_y, y / w);
                    hi_y = std::max(hi_y, y / w);
                    near_w = std::min(near_w, w);
                }
                const float screens = std::max(hi_x - lo_x, hi_y - lo_y) * 0.5f;  // 1.0 is the whole frame
                // Skip the ones the near cull drops: what is left is a quad
                // that is genuinely enormous at a normal distance.
                if (near_w < 2.0f) continue;
                if (ok && std::isfinite(screens) && screens > worst) {
                    worst = screens;
                    worst_w = near_w;
                    worst_k = k;
                }
            }
            if (g_dump_patches.load(std::memory_order_relaxed)) {
                // Raw patches of the flip the watch flagged, for reading back
                // offline: a header a draw, then its region.
                static FILE* out = nullptr;
                static std::uint64_t out_flip = 0;
                if (!out || out_flip != p.flip) {
                    if (out) std::fclose(out);
                    char path[96];
                    std::snprintf(path, sizeof(path), "build/patches-%llu.bin", static_cast<unsigned long long>(p.flip));
                    out = std::fopen(path, "wb");
                    out_flip = p.flip;
                    if (out) host_log("dump: the patches of flip %llu -> %s", static_cast<unsigned long long>(p.flip), path);
                }
                if (out) {
                    char head[96] = {};
                    std::snprintf(head, sizeof(head), "%s %s %u %u", p.name.c_str(), p.ds.c_str(), p.count, p.stride);
                    std::fwrite(head, 1, sizeof(head), out);
                    std::fwrite(m, 1, sizeof(m), out);  // the projection its constants hold
                    std::fwrite(region, 1, static_cast<std::size_t>(p.count) * p.stride, out);
                    std::fflush(out);
                }
            }
            if (all_zero) {
                static std::atomic<int> zlogs{0};
                if (zlogs.fetch_add(1) < 20) {
                    host_log("tess scan: flip %llu %s: %u of %u patches were left unwritten by the LS pass",
                             static_cast<unsigned long long>(p.flip), p.name.c_str(), all_zero, p.count);
                }
            }
            if (worst > 3.0f) {
                const auto* f = reinterpret_cast<const float*>(region + static_cast<std::size_t>(worst_k) * p.stride);
                host_log("tess scan: flip %llu %s -> %s, %u patches: patch %u covers %.1f screens, nearest corner w %g; dwords "
                         "%g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g",
                         static_cast<unsigned long long>(p.flip), p.name.c_str(), p.ds.c_str(), p.count, worst_k, worst, worst_w, f[0], f[1],
                         f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17],
                         f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27]);
            }
        }
        while (pending.size() > 4096) pending.pop_front();
    }
    // BBHOST_TESS_DEBUG=1: the first draws' control points as the LS wrote them.
    static const bool debug = [] {
        const char* e = std::getenv("BBHOST_TESS_DEBUG");
        return e && e[0] == '1';
    }();
    static int debug_logs = 0;
    if (debug && debug_logs < 12) {
        ++debug_logs;
        flush_locked();
        const auto* f = reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(ring.map) + (t.lds_address - ring.address));
        const auto* u = reinterpret_cast<const std::uint32_t*>(f);
        host_log("tess debug: %s count %u level %u index 0x%llx (type %u) base vertex %d stride %llu; point 0: %g %g %g %g | %g %g %g %g | "
                 "%g %g %g %g | %g %g %g %g | %08x %08x %08x %08x",
                 pl.name.c_str(), t.count, t.level, static_cast<unsigned long long>(d.index_va), d.index_type, static_cast<int>(d.ctx[0x102]),
                 static_cast<unsigned long long>(stride), f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12],
                 f[13], f[14], f[15], u[16], u[17], u[18], u[19]);
        for (std::uint32_t k = 0; k < t.count && k < 4; ++k) {
            const float* q = f + k * stride / 4;
            host_log("tess debug:   point %u: pivot %g %g %g %g; rows %g %g %g %g / %g %g %g %g / %g %g %g %g", k, q[0], q[1], q[2], q[3],
                     q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23]);
        }
    }
    return true;
}


// The LS's input (BBHOST_TESS_PROBE=1, once per domain shader): the V#s its
// fetch shader loads from the vertex table (s_load_dwordx4 from a user-SGPR
// pair plus an offset), resolved from the draw's user data - the
// control-point buffer the game's particle code wrote. BBHOST_TESS_WATCH=
// <domain hash> arms a write watch on the first one for a draw whose domain
// shader has that hash, so the watch report names the writer.
void probe_ls_input(std::uint64_t fetch_va, const std::uint32_t* user, std::uint64_t ds_va, const GxDrawObjects* gx) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_TESS_PROBE");
        return e && e[0] == '1';
    }();
    if (!on) return;
    CachedProgram* dsp = program_at(ds_va);
    static std::map<std::uint64_t, int> seen;  // the first eight draws of each domain shader
    if (!dsp || seen.size() > 64 || seen[ds_va]++ >= 8) return;
    const FetchProgram* fp = fetch_va ? fetch_program_at(fetch_va) : nullptr;
    std::string line;
    std::uint64_t first_base = 0, first_bytes = 0;
    if (fp) {
        std::error_code ec;
        std::filesystem::create_directories("tmp/tess", ec);
        char path[96];
        std::snprintf(path, sizeof(path), "tmp/tess/fetch-%llx.bin", static_cast<unsigned long long>(fetch_va));
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(fp->words.data(), 4, fp->words.size(), f);
            std::fclose(f);
        }
        const gcn::Program fprog = gcn::decode(fp->words.data(), fp->words.size());
        for (const gcn::Inst& in : fprog.insts) {
            if (in.enc != gcn::Enc::SMRD || std::strcmp(gcn::mnemonic(in), "s_load_dwordx4") != 0) continue;
            const unsigned sb = in.src0;
            if (sb + 1 >= 16) continue;
            const std::uint64_t off = in.imm_flag ? static_cast<std::uint64_t>(in.imm) * 4 : static_cast<std::uint64_t>(in.literal);
            const std::uint64_t table = static_cast<std::uint64_t>(user[sb]) | (static_cast<std::uint64_t>(user[sb + 1]) << 32);
            std::uint32_t v[4] = {};
            std::uint64_t home = 0;  // the buffer object's own address, where the CPU writes before the rename
            if (gx && !table) {
                // A GX draw: the table is built from the call's records, not on the command stream.
                const std::uint64_t slot = off / 16;
                if (slot >= 16 || !((gx->vtx_valid >> slot) & 1)) continue;
                std::memcpy(v, gx->vtx_rec[slot], 16);
                if (gx->vtx_obj[slot]) {
                    host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(gx->vtx_obj[slot] + 0x20)), &home, 8);
                }
            } else {
                const std::uint64_t at = table + off;
                if (!hle_kernel_va_mapped(at, 16)) {
                    char nb[64];
                    std::snprintf(nb, sizeof(nb), " s[%u:%u] table 0x%llx unmapped;", sb, sb + 1, static_cast<unsigned long long>(at));
                    line += nb;
                    continue;
                }
                std::memcpy(v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), 16);
            }
            // The GX-built records keep the base in dword1's low 12 bits and flags above (build_vertex_table).
            const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xfff) << 32);
            const std::uint32_t stride = (v[1] >> 16) & 0x3fff, records = v[2];
            char buf[96];
            std::snprintf(buf, sizeof(buf), " s[%u:%u]+0x%x -> base 0x%llx (home 0x%llx) stride %u records %u;", sb, sb + 1,
                          in.imm_flag ? in.imm * 4 : static_cast<int>(in.literal), static_cast<unsigned long long>(base),
                          static_cast<unsigned long long>(home), stride, records);
            line += buf;
            if (!first_base && base && stride) {
                first_base = home ? home : base;
                first_bytes = static_cast<std::uint64_t>(stride) * records;
            }
        }
    }
    host_log("tess: %s LS fetch 0x%llx (%zu words) user %08x %08x %08x %08x V#s:%s", dsp->name.c_str(),
             static_cast<unsigned long long>(fetch_va), fp ? fp->words.size() : 0, user[0], user[1], user[2], user[3],
             line.empty() ? " (none)" : line.c_str());
    static const char* want = std::getenv("BBHOST_TESS_WATCH");
    static bool armed = false;
    if (want && want[0] && !armed && first_base && dsp->name.find(want) != std::string::npos) {
        armed = true;
        // The buffer's home, where the game writes before the GX renames
        // and copies it; through the GPU alias when direct memory has one.
        static const bool cpu_view = std::getenv("BBHOST_TESS_WATCH_CPU") != nullptr;  // the CPU view instead of the alias
        const std::uint64_t alias = cpu_view ? 0 : hle_kernel_gpu_alias(first_base);
        const std::uint64_t at = alias ? alias : first_base;
        host_log("tess: watching the LS input of %s: [0x%llx, +0x%llx) (first page%s)", dsp->name.c_str(),
                 static_cast<unsigned long long>(at), static_cast<unsigned long long>(first_bytes),
                 alias ? ", the GPU alias" : "");
        hle_watch_arm(at, at + first_bytes);
    }
}


}  // namespace

static bool draw_impl(const GpuDraw& d);

// One tessellated draw (or one chunk of an instanced hull draw's
// instances): its LS pass and its draw.
static bool tess_draw_one(const GpuDraw& d, TessDraw& t) {
    // BBHOST_TESS_HULL_TRACE=1: every draw the game's own hull runs for -
    // which, how many control points, and how far it got.
    static const bool hull_trace = [] {
        const char* e = std::getenv("BBHOST_TESS_HULL_TRACE");
        return e && e[0] == '1';
    }();
    {
        std::lock_guard<GpuMutex> lock(g.mu);
        // A device that is gone draws nothing, and its draws are not the LS
        // pass's failures: the RX 7600 XT's log put the 23,314 tessellated
        // draws the game made after its device loss under "tessellation LS
        // pass", which read as draws left out while the LS compiled.
        if (!init_locked()) {
            draw_failed(kFailGpuGone);
            return false;
        }
        if (!tess_ls_pass_locked(d, t)) {
            if (hull_trace && t.hull) {
                host_log("tess hull: flip %llu HS 0x%llx %u points: LS pass failed", static_cast<unsigned long long>(hle_video_flip_count()),
                         static_cast<unsigned long long>(t.hs_va), t.count);
            }
            draw_failed(kFailTessLs);
            return false;
        }
    }
    static const int tess_debug = [] {  // 2: the LS pass alone, no draw (a diagnostic)
        const char* e = std::getenv("BBHOST_TESS_DEBUG");
        return e ? std::atoi(e) : 0;
    }();
    if (tess_debug == 2) return true;
    GpuDraw dd = d;
    // The host's tessellator takes a patch list: one vertex a control point,
    // and it decides the rest. The emulated path has to hand over every vertex
    // of the topology it invents, and its LS pass has already resolved the
    // index buffer into the ring, so it draws without one. With the LS as the
    // vertex stage the draw is the draw the game asked for, index buffer and
    // all.
    if (!t.attrs) dd.index_va = 0;
    dd.index_count = t.hull ? t.count * t.instances : t.hw ? t.count * t.control_points : t.count * tess_vertices_a_patch(t.level);
    // BBHOST_TESS_DEBUG=7: record the draw with nothing to draw. Everything
    // it does apart from running vertices - its pipeline, its sets, its
    // parameters, its state - still happens.
    static const bool no_vertices = [] {
        const char* e = std::getenv("BBHOST_TESS_DEBUG");
        return e && e[0] == '7';
    }();
    if (no_vertices) dd.index_count = 0;
    dd.instance_count = 1;
    t_tess = &t;
    t_tess_constants = t.hull ? t.tess_vsharp : nullptr;
    const bool ok = draw_impl(dd);
    t_tess_constants = nullptr;
    t_tess = nullptr;
    if (hull_trace && t.hull) {
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        host_log("tess hull: flip %llu at %.3f instances %u HS 0x%llx %u points (%u patches), index 0x%llx base %d, lds 0x%llx: %s",
                 static_cast<unsigned long long>(hle_video_flip_count()), now.tv_sec + now.tv_nsec * 1e-9, d.instance_count, static_cast<unsigned long long>(t.hs_va), t.count,
                 t.count / t.control_points, static_cast<unsigned long long>(d.index_va),
                 d.gx && d.gx_render ? static_cast<int>(d.gx->base_vertex) : 0, static_cast<unsigned long long>(t.lds_address),
                 ok ? "drawn" : "draw failed");
    }
    g.draw_calls.fetch_sub(1, std::memory_order_relaxed);  // draw_impl counted it again
    g_tess_draws.fetch_add(1, std::memory_order_relaxed);
    g_tess_patches.fetch_add(t.hull ? t.count * t.instances / t.control_points : t.count, std::memory_order_relaxed);
    if (t.hull) {
        // Where the game's hull draws keep their windows, the first times: a
        // device loss on one names the draw (AMD's markers), and this its
        // patches against the region its stages are held inside.
        const std::uint32_t patches = t.count * t.instances / t.control_points;
        g_tess_hull_draws.fetch_add(1, std::memory_order_relaxed);
        g_tess_hull_patches.fetch_add(patches, std::memory_order_relaxed);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1, std::memory_order_relaxed) < 4) {
            host_log("render: the game's hull draws %u patches (%u points x %u instances) into %u bytes of LDS ring, windows of %u bytes, "
                     "accesses %s: %s",
                     patches, t.count, t.instances, t.lds_bytes, t.window, tess_lds_bound() ? "bounded" : "unchecked (BBHOST_TESS_LDS_BOUND=0)",
                     ok ? "drawn" : "draw failed");
        }
    }
    return ok;
}

bool host_gpu_draw(const GpuDraw& d) {
    // The windows given since the last draw are this one's (host_gpu_draw_window).
    t_draw_windows.swap(t_given_windows);
    t_given_windows.clear();
    // Under a SET_PREDICATION whose block already reads occluded: not drawn,
    // as the PS4's command processor would skip it (occlusion.cpp).
    if (!occlusion_draw_entry()) return true;
    probe_tessellated_draw(d);
    // Whether this is a tessellated draw at all. A draw that took a token
    // wrote no VGT_SHADER_STAGES_EN for the command stream to hold, so its GX
    // inputs are what say so; without them the register file does.
    {
        const GpuDrawInputs* gi = tess_gx(d);
        if (d.gx_token) {
            g_token_draws_seen.fetch_add(1, std::memory_order_relaxed);
            if (d.gx) g_token_with_gx.fetch_add(1, std::memory_order_relaxed);
            if (d.gx && d.gx->tess) g_token_tess.fetch_add(1, std::memory_order_relaxed);
        }
        if (!(gi ? gi->stages & 0x7 : d.ctx[0x2d5] & 0x7)) return draw_impl(d);
    }
    g.draw_calls.fetch_add(1, std::memory_order_relaxed);
    static const bool tess_on = [] {  // BBHOST_TESS=0: tessellated draws are skipped, as before 2026-09-19
        const char* e = std::getenv("BBHOST_TESS");
        return !(e && e[0] == '0');
    }();
    if (!tess_on) {
        draw_failed(kFailTessOff);
        return false;
    }
    TessDraw t;
    if (!tess_plan(d, t)) {
        draw_failed(kFailTessPlan);
        return false;
    }
    // BBHOST_TESS_HULL_SKIP=<points>[:<k>]: a diagnostic - leave out the hull
    // draws of that many control points (only the k-th of them in a flip).
    if (t.hull) {
        static const std::pair<std::uint32_t, int> skip = [] {
            const char* e = std::getenv("BBHOST_TESS_HULL_SKIP");
            if (!e) return std::pair<std::uint32_t, int>{0, -1};
            const char* c = std::strchr(e, ':');
            return std::pair<std::uint32_t, int>{static_cast<std::uint32_t>(std::atoi(e)), c ? std::atoi(c + 1) : -1};
        }();
        static std::uint64_t flip = 0;
        static int seen = 0;
        if (skip.first && t.count == skip.first) {
            if (hle_video_flip_count() != flip) flip = hle_video_flip_count(), seen = 0;
            if (skip.second < 0 || seen++ == skip.second) return true;
        }
    }
    // A hull draw whose windows would not fit a ring region goes in chunks,
    // each its own LS pass and draw: whole instances while one instance fits
    // (four of the 14,715-point roots took 20 MiB), else each instance in
    // ranges of patches (Byrgenwerth's tree tops: 19,117 patches, 16 MiB, in
    // one). Patches are independent, and neither the hull nor the domain
    // shaders read their absolute patch id (v0, v3), only their window.
    if (t.hull) {
        const std::uint64_t budget = kTessRegionBytes / 2;
        const std::uint32_t patches = t.count / t.control_points;
        const std::uint64_t per = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(patches + 1) * t.window);
        bool ok = true;
        if (per <= budget) {
            const std::uint32_t fit = static_cast<std::uint32_t>(budget / per);
            if (t.instances <= fit) return tess_draw_one(d, t);
            const std::uint32_t total = t.instances;
            for (std::uint32_t first = 0; first < total; first += fit) {
                TessDraw c = t;
                c.first_instance = first;
                c.instances = std::min(fit, total - first);
                ok = tess_draw_one(d, c) && ok;
            }
            return ok;
        }
        const std::uint32_t fit = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, budget / t.window - 64));
        for (std::uint32_t inst = 0; inst < t.instances; ++inst) {
            for (std::uint32_t p = 0; p < patches; p += fit) {
                TessDraw c = t;
                c.first_instance = inst;
                c.instances = 1;
                c.first_point = p * t.control_points;
                c.count = std::min(fit, patches - p) * t.control_points;
                ok = tess_draw_one(d, c) && ok;
            }
        }
        return ok;
    }
    return tess_draw_one(d, t);
}

static bool draw_impl(const GpuDraw& d) {
    g_draw_dummies = 0;  // the draw record takes the count of this draw's
    t_draw_pipeline = nullptr;
    g.draw_calls.fetch_add(1, std::memory_order_relaxed);
    // BBHOST_PASS_BREAK=<n>: end the render pass before every n-th draw, the
    // way a tessellated draw's compute pass has to. A check on whether that
    // churn is what washes a frame out.
    static const std::uint64_t pass_break = [] {
        const char* e = std::getenv("BBHOST_PASS_BREAK");
        return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    if (pass_break) {
        static std::atomic<std::uint64_t> seen{0};
        if (seen.fetch_add(1) % pass_break == 0) {
            std::lock_guard<GpuMutex> lock(g.mu);
            if (g.ok && init_locked()) {
                begin_recording_locked();
                transfer_flush_locked();
                render_end_pass_locked();
                // BBHOST_PASS_BREAK_SUBMIT=1: submit there too, as a draw that
                // runs out of parameter slots makes the renderer do.
                static const bool also_submit = [] {
                    const char* e = std::getenv("BBHOST_PASS_BREAK_SUBMIT");
                    return e && e[0] == '1';
                }();
                if (also_submit) submit_locked();
            }
        }
    }
    // Everything below runs under the one GPU mutex, so book the time spent
    // waiting for it separately: it is the other command-processor threads
    // and the fence waits, not this draw.
    // One draw in kPhaseSample is timed, and booked that many times over.
    static thread_local std::uint32_t phase_seq = 0;
    const std::uint32_t phase_weight = ++phase_seq % kPhaseSample == 0 ? kPhaseSample : 0;
    GpuPhaseTimer timer(kPhaseLock, phase_weight);
    std::lock_guard<GpuMutex> lock(g.mu);
    timer.next(kPhaseDraw);
    RenderCostStamp draw_stamp;
    t_token_draw = d.gx_token;
    apply_table_masks_locked();
    apply_lean_builds_locked();
    apply_relinks_locked();
    if (!init_locked()) return false;
    // The draw's commands (recorder.cpp): its packet starts with its
    // descriptor writes below, or earlier for a region copy (draw_packet_early_locked).
    DrawCmds cmds;
    struct CurrentDraw {
        DrawCmds* prev;
        explicit CurrentDraw(DrawCmds* c) : prev(g_draw_cmds) { g_draw_cmds = c; }
        ~CurrentDraw() { g_draw_cmds = prev; }
    } current_draw(&cmds);
    const std::uint32_t* cx = d.ctx;
    // BBHOST_GX_BACKEND=1: the state comes from the GX objects the engine's
    // draw call used instead of the register file. A draw
    // GX describes whole reads no register: the file is read only for the
    // draws that take it (packets, YEBIS's partial descriptions) and to
    // compare. Both input blocks are ~900 bytes, so they are not built fresh
    // for every draw.
    const bool gx_whole = d.gx && d.gx_render && !d.gx_partial;
    static thread_local GpuDrawInputs reg_inputs;
    if (!gx_whole || d.gx_compare) {
        reg_inputs = GpuDrawInputs{};
        draw_inputs_from_registers(d, reg_inputs);
    }
    if (d.gx && d.gx_compare) {
        RenderCostTimer timer(kRenderCostCompare);
        compare_draw_inputs(reg_inputs, *d.gx);
    }
    static thread_local GpuDrawInputs merged;
    if (d.gx && d.gx_render && d.gx_partial) {
        merged = reg_inputs;
        std::memcpy(merged.vs_pgm, d.gx->vs_pgm, sizeof(merged.vs_pgm));
        std::memcpy(merged.ps_pgm, d.gx->ps_pgm, sizeof(merged.ps_pgm));
        std::memcpy(merged.vs_user, d.gx->vs_user, sizeof(merged.vs_user));
        std::memcpy(merged.ps_user, d.gx->ps_user, sizeof(merged.ps_user));
        std::memcpy(merged.ps_input_cntl, d.gx->ps_input_cntl, sizeof(merged.ps_input_cntl));
        merged.ps_input_count = d.gx->ps_input_count;
        merged.vs_out_cntl = d.gx->vs_out_cntl;
        merged.ps_input_ena = d.gx->ps_input_ena;
        merged.ps_in_control = d.gx->ps_in_control;
        merged.cb_shader_mask = d.gx->cb_shader_mask;
        merged.ps_col_format = d.gx->ps_col_format;
    }
    const GpuDrawInputs& in = d.gx && d.gx_render ? (d.gx_partial ? merged : *d.gx) : reg_inputs;
    log_draw_state(d, in);
    draw_stamp.to(kRenderCostRegisters);
    DrawState s;
    s.vs_va = (static_cast<std::uint64_t>(in.vs_pgm[1] & 0xff) << 40) | (static_cast<std::uint64_t>(in.vs_pgm[0]) << 8);
    s.ps_va = (static_cast<std::uint64_t>(in.ps_pgm[1] & 0xff) << 40) | (static_cast<std::uint64_t>(in.ps_pgm[0]) << 8);
    s.vs_rsrc1 = in.vs_pgm[2];
    s.vs_rsrc2 = in.vs_pgm[3];
    s.ps_rsrc1 = in.ps_pgm[2];
    s.ps_rsrc2 = in.ps_pgm[3];
    std::memcpy(s.vs_user, in.vs_user, sizeof(s.vs_user));
    std::memcpy(s.ps_user, in.ps_user, sizeof(s.ps_user));
    // A tessellated draw's patches become triangles (draw_tessellated), unless
    // the host's own tessellator draws them, when they stay a patch list.
    s.prim = t_tess ? (t_tess->hw ? 9u : 4u) : in.prim;
    s.domain_level = t_tess && !t_tess->hw ? t_tess->level : 0;
    if (t_tess && t_tess->hw) {
        s.tess_hw = true;
        s.tess_control_points = t_tess->control_points;
        std::memcpy(s.tess_outer, t_tess->outer, sizeof(s.tess_outer));
        std::memcpy(s.tess_inner, t_tess->inner, sizeof(s.tess_inner));
        s.tess_attrs = t_tess->attrs;
        s.tess_attr_vec4s = t_tess->attr_vec4s;
        s.ls_va = t_tess->ls_va;
        s.ls_fetch_va = t_tess->ls_fetch_va;
        s.ls_rsrc1 = t_tess->ls_rsrc1;
        s.ls_rsrc2 = t_tess->ls_rsrc2;
        std::memcpy(s.ls_user, t_tess->ls_user, sizeof(s.ls_user));
        if (t_tess->hull) {
            s.tess_hull = true;
            s.hs_va = t_tess->hs_va;
            s.hs_rsrc1 = t_tess->hs_rsrc1;
            s.hs_rsrc2 = t_tess->hs_rsrc2;
            s.tess_window = t_tess->window;
            s.tess_quads = t_tess->quads;
            s.tess_cw = t_tess->cw;
            s.tess_spacing = t_tess->spacing;
        }
    }
    {
        // A tessellated draw whose pipeline is built without the domain role
        // would run its domain shader as a plain vertex shader: garbage
        // geometry. Count both ways round.
        static std::atomic<std::uint64_t> tess_without{0}, plain_with{0};
        if (t_tess && !s.domain_level && !s.tess_hw) {
            if (tess_without.fetch_add(1) == 0) host_log("render: a tessellated draw with no domain level");
        }
        if (!t_tess && s.domain_level) {
            if (plain_with.fetch_add(1) == 0) host_log("render: a plain draw with a domain level");
        }
    }
    if (!primitive_topology(s.prim, s.topo, s.rect)) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 4) host_log("render: unsupported primitive type %u", s.prim);
        draw_failed(kFailPrimitive);
        return false;
    }
    if (!s.vs_va) {
        draw_failed(kFailNoVs);
        return false;
    }
    if (!rebuild_page_tables()) return false;
    // Render targets
    const std::uint32_t target_mask = in.target_mask;
    const DrawScissor target_rect = draw_scissor(in, false);
    DrawScissor draw_rect = target_rect;
    if (in.sc_mode_cntl_0 & 2) {
        const DrawScissor narrowed = draw_scissor(in, true);
        if (narrowed.x0 != target_rect.x0 || narrowed.y0 != target_rect.y0 || narrowed.x1 != target_rect.x1 ||
            narrowed.y1 != target_rect.y1) {
            bump(g_vport_narrowed);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 16) {
                host_log("render: viewport scissor narrows a draw to (%d,%d)-(%d,%d) from (%d,%d)-(%d,%d); viewport xscale %g "
                         "xoffset %g yscale %g yoffset %g; PS 0x%08x, CB_COLOR0_INFO %08x",
                         narrowed.x0, narrowed.y0, narrowed.x1, narrowed.y1, target_rect.x0, target_rect.y0, target_rect.x1,
                         target_rect.y1, in.vport[0], in.vport[1], in.vport[2], in.vport[3], in.ps_pgm[0], in.color[0][4]);
            }
            if (g_vport_scissor) draw_rect = narrowed;
        }
    }
    s.scissor = {{draw_rect.x0, draw_rect.y0},
                 {static_cast<std::uint32_t>(draw_rect.x1 - draw_rect.x0), static_cast<std::uint32_t>(draw_rect.y1 - draw_rect.y0)}};

    // The screen and generic scissors' bottom edge is the target's real
    // height (see unpadded_height); the CB registers only give the padded
    // one. The viewport scissor is per draw, so it does not size targets.
    const int x1 = target_rect.x1, y1 = target_rect.y1;
    const std::uint32_t scissor_bottom = y1 > 0 && y1 < 16384 ? static_cast<std::uint32_t>(y1) : 0u;
    const std::uint32_t scissor_right = x1 > 0 && x1 < 16384 ? static_cast<std::uint32_t>(x1) : 0u;
    s.cb_shader_mask = in.cb_shader_mask;
    s.ps_col_format = in.ps_col_format;
    for (int t = 0; t < 8; ++t) {
        const std::uint32_t* c = in.color[t];
        const std::uint32_t mask = (target_mask >> (4 * t)) & 0xf;
        const CbInfo info = CbInfo::from(c[4]);
        if (!mask || !c[0] || info.format == 0) continue;
        const VkFormat fmt = cb_format(info);
        if (fmt == VK_FORMAT_UNDEFINED) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 6) host_log("render: unsupported color format %u type %u swap %u", info.format, info.number_type, info.comp_swap);
            continue;
        }
        const std::uint32_t pitch = ((c[1] & 0x7ff) + 1) * 8;
        const std::uint32_t slice_px = ((c[2] & 0x3fffff) + 1) * 64;
        const std::uint32_t padded_h = pitch ? slice_px / pitch : 0;
        if (!pitch || !padded_h) continue;
        // CB_COLOR*_VIEW: SLICE_START [10:0], SLICE_MAX [23:13] (as upstream
        // shadPS4 regs_color.h). A layered target - a cube's six faces - has
        // SLICE_MAX + 1 slices of `pitch * padded_h` pixels each in guest memory,
        // and the draw renders into SLICE_START. A LINEAR_GENERAL target keeps
        // the low address byte in SLICE_START instead and starts at slice 0.
        const bool linear_general = ((c[4] >> 7) & 1) != 0;
        const std::uint64_t base = (static_cast<std::uint64_t>(c[0]) << 8) | (linear_general ? (c[3] & 0xff) : 0u);
        const std::uint32_t layers = ((c[3] >> 13) & 0x7ff) + 1;
        const std::uint32_t slice_start = linear_general ? 0 : std::min(c[3] & 0x7ff, layers - 1);
        s.color[t] = target_image(t, base, fmt, pitch, padded_h, scissor_right, scissor_bottom, false, layers,
                                  static_cast<std::uint64_t>(pitch) * padded_h * format_bytes_per_pixel(fmt), in.color_extent[t]);
        s.color_layer[t] = slice_start;
        if (layers > 1) {
            static std::set<std::uint64_t> logged;
            if (logged.size() < 24 && logged.insert(base).second) {
                host_log("render: layered colour target 0x%llx: %u layers, slice %u bytes (first drawn layer %u)",
                         static_cast<unsigned long long>(base), layers, pitch * padded_h * static_cast<std::uint32_t>(format_bytes_per_pixel(fmt)),
                         slice_start);
            }
        }
        // The channels the pixel shader exports to this target: CB_SHADER_MASK,
        // which Gnm sets from SPI_SHADER_COL_FORMAT (32_R exports R alone,
        // 32_GR red and green, 32_AR red and alpha). The export's other
        // components are whatever the registers held - the hardware drops
        // them. Writing them put a leftover ~3000 into the green of the
        // G-buffer's emissive target (0x174ce0000), and every candle and
        // lantern in the Hunter's Dream glowed green through the lighting.
        // A shader that exports no colour (a depth pass, or no pixel shader at
        // all) has a zero mask and writes none - not whatever a fragment
        // shader without that output leaves in the attachment.
        s.color_mask[t] = mask & ((in.cb_shader_mask >> (4 * t)) & 0xf);
        // BBHOST_TESS_DEBUG=6: a tessellated draw writes no colour, so what is
        // left of it is its state and its pass, not its pixels.
        static const bool no_colour = [] {
            const char* e = std::getenv("BBHOST_TESS_DEBUG");
            return e && e[0] == '6';
        }();
        if (no_colour && t_tess) s.color_mask[t] = 0;
        s.blend[t] = in.blend[t];
        // CB_COLOR_INFO.BLEND_BYPASS (bit 16): the target writes the shader's
        // colour unblended whatever CB_BLEND_CONTROL says.
        if ((c[4] >> 16) & 1) {
            if (s.blend[t] & (1u << 30)) {
                static std::set<std::uint64_t> logged;
                if (logged.size() < 16 && logged.insert(base).second) {
                    host_log("render: colour target 0x%llx has BLEND_BYPASS with blending enabled - drawn unblended",
                             static_cast<unsigned long long>(base));
                }
            }
            s.blend[t] &= ~(1u << 30);
        }
        // Compressed exports in a 16-bit integer or normalized export format
        // (SPI_SHADER_COL_FORMAT 5-8) unpack as those numbers, not as halves;
        // the translator unpacks every compressed export as FP16. Not seen in
        // the areas checked (FP16 and 32_ABGR only) - say so if one appears.
        if (const std::uint32_t cf = (in.ps_col_format >> (4 * t)) & 0xf; cf >= 5 && cf <= 8) {
            static std::set<std::uint32_t> logged;
            if (logged.size() < 16 && logged.insert(in.ps_pgm[0] ^ (t << 28)).second) {
                host_log("render: PS 0x%x exports target %d in SPI_SHADER_COL_FORMAT %u (16-bit %s); compressed exports "
                         "are unpacked as FP16", in.ps_pgm[0], t, cf, cf == 5 ? "unorm" : cf == 6 ? "snorm" : cf == 7 ? "uint" : "sint");
            }
        }
    }
    s.depth_control = in.depth_control;
    s.depth_bounds_min = in.depth_bounds[0];
    s.depth_bounds_max = in.depth_bounds[1];
    s.stencil_control = in.stencil_control;
    s.stencil_ref = in.stencil_ref;
    s.stencil_ref_bf = in.stencil_ref_bf;
    s.render_control = in.render_control;
    s.depth_clear = in.depth_clear;
    s.stencil_clear = in.stencil_clear;
    const std::uint32_t z_info = in.z_info, stencil_info = in.stencil_info;
    const bool wants_depth = (s.depth_control & 3) != 0 || (s.render_control & 3) != 0;
    if (wants_depth && in.z_read_base) {
        const VkFormat df = db_format(z_info, stencil_info);
        if (df != VK_FORMAT_UNDEFINED) {
            const std::uint32_t size = in.depth_size;
            const std::uint32_t w = ((size & 0x7ff) + 1) * 8;
            const std::uint32_t h = (((size >> 11) & 0x7ff) + 1) * 8;
            const std::uint64_t dbase = static_cast<std::uint64_t>(in.z_read_base) << 8;
            if (w && h) {
                // DB_DEPTH_VIEW selects the slice as CB_COLOR*_VIEW does; the
                // slices of DEPTH_SIZE's padded w x h follow each other.
                const std::uint32_t layers = ((in.depth_view >> 13) & 0x7ff) + 1;
                const std::size_t bpp = df == VK_FORMAT_D16_UNORM ? 2 : 4;
                s.depth = target_image(8, dbase, df, w, h, scissor_right, scissor_bottom, true, layers,
                                       static_cast<std::uint64_t>(w) * h * bpp, in.depth_extent);
                s.depth_layer = std::min(in.depth_view & 0x7ff, layers - 1);
                if (layers > 1) {
                    static std::set<std::uint64_t> logged;
                    if (logged.size() < 24 && logged.insert(dbase).second) {
                        host_log("render: layered depth target 0x%llx: %u layers (first drawn layer %u)",
                                 static_cast<unsigned long long>(dbase), layers, s.depth_layer);
                    }
                }
            }
            if (s.depth) {
                const std::uint64_t htile = static_cast<std::uint64_t>(in.htile_base) << 8;  // DB_HTILE_DATA_BASE
                if (htile != s.depth->htile) {
                    host_log("render: depth target 0x%llx htile at 0x%llx (z_info %08x)", static_cast<unsigned long long>(s.depth->base),
                             static_cast<unsigned long long>(htile), z_info);
                }
                s.depth->htile = htile;
                if (htile) g_htile_bases.insert(htile);
                if (!g_pending_htile.empty()) take_pending_htile(htile, *s.depth);
                if (s.depth->htile_clear_pending) {
                    // The clear values are the fill's: a draw from
                    // a token reads no register, and the register file at a
                    // packet draw is checked against them.
                    const std::uint32_t depth_clear = s.depth->htile_clear_depth, stencil_clear = s.depth->htile_clear_stencil;
                    if (d.gx_token) {
                        g_token_htile_clears.fetch_add(1, std::memory_order_relaxed);
                        if (depth_clear || (stencil_clear & 0xff)) g_token_htile_clear_values.fetch_add(1, std::memory_order_relaxed);
                    }
                    // The check: the register file at the draw (which a token
                    // draw does not read, but the CP still has) against the fill's.
                    const std::uint32_t reg_depth = d.gx_token ? d.ctx[0x0B] : s.depth_clear;
                    const std::uint32_t reg_stencil = d.gx_token ? d.ctx[0x0A] : s.stencil_clear;
                    if (d.registers && (depth_clear != reg_depth || (stencil_clear & 0xff) != (reg_stencil & 0xff))) {
                        if (g_htile_clear_value_mismatch.fetch_add(1, std::memory_order_relaxed) < 12) {
                            host_log("render: HTILE clear of 0x%llx at flip %llu: the fill captured depth %08x stencil %02x, the register file at the "
                                     "draw holds %08x / %02x (%s)",
                                     static_cast<unsigned long long>(s.depth->base), static_cast<unsigned long long>(hle_video_flip_count()),
                                     depth_clear, stencil_clear & 0xff, reg_depth, reg_stencil & 0xff, d.gx_token ? "token" : "packet");
                        }
                    }
                    s.depth->htile_clear_pending = false;
                    begin_recording_locked();
                    float dval;
                    std::memcpy(&dval, &depth_clear, 4);
                    clear_depth(*s.depth, true, true, dval, stencil_clear & 0xff);
                }
            }
        }
    }
    s.su_sc_mode = in.su_sc_mode;
    s.clip_cntl = in.clip_cntl;
    s.vte_cntl = in.vte_cntl;
    s.ps_input_ena = in.ps_input_ena;
    s.ps_in_control = in.ps_in_control;
    s.vs_out_cntl = in.vs_out_cntl;
    s.program_fp = in.program_fp;
    std::memcpy(s.ps_input_cntl, in.ps_input_cntl, sizeof(s.ps_input_cntl));
    std::memcpy(s.vport, in.vport, sizeof(s.vport));
    bool any_color = false;
    for (int t = 0; t < 8; ++t) any_color |= s.color[t] != nullptr;
    // Depth/stencil clears requested through DB_RENDER_CONTROL.
    if (s.depth && (s.render_control & 3)) {
        begin_recording_locked();
        float dval;
        std::memcpy(&dval, &s.depth_clear, 4);
        clear_depth(*s.depth, (s.render_control & 1) != 0, (s.render_control & 2) != 0, dval, s.stencil_clear & 0xff);
    }
    if (!any_color && !s.depth) {
        g_untraced_draws.fetch_add(1);
        g.draws_empty.fetch_add(1, std::memory_order_relaxed);
        return true;  // nothing to render into
    }
    if (!any_color && !s.ps_va && !(s.depth_control & 5)) {
        g.draws_empty.fetch_add(1, std::memory_order_relaxed);
        return true;  // no colour, no depth write: nothing observable
    }
    if (d.gx_objects && gx_state_mode() == 2) {
        compare_gx_state(s, in, *d.gx_objects);
    } else if (d.gx_objects && gx_state_mode() == 1) {
        // The pipeline's state from the game's descriptions.
        const GxDrawObjects& o = *d.gx_objects;
        s.gx_blend = gx_blend_desc(o.blend_id);
        s.gx_depth = gx_depth_stencil_desc(o.depth_stencil_id);
        s.gx_raster = gx_raster_desc(o.raster_id);
        if (s.gx_blend && s.gx_depth && s.gx_raster) {
            s.gx_ids[0] = o.blend_id;
            s.gx_ids[1] = o.depth_stencil_id;
            s.gx_ids[2] = o.raster_id;
            ++g_gs_from_objects;
            if (const std::uint32_t i = o.raster_id & 0x3fffffff; i < 64) {
                ++g_gs_raster_draws[i];
                if (!any_color) ++g_gs_raster_depth_only[i];
            }
        } else {
            s.gx_blend = nullptr;
            s.gx_depth = nullptr;
            s.gx_raster = nullptr;
            ++g_gs_from_registers;
        }
    }

    draw_stamp.to(kRenderCostTargets);
    // Shaders: by their GX objects' ids where the draw has them,
    // else by code address.
    const FetchProgram* fetch = nullptr;  // the fetch shader, when the VS takes one
    const GxDrawObjects* ids = d.gx && d.gx_render && !d.gx_partial ? d.gx_objects : nullptr;
    CachedProgram* vs_cp = program_for(ids ? ids->vs_prog_id : 0, s.vs_va);
    if (!vs_cp) {
        draw_failed(kFailVsProgram);
        return false;
    }
    if (ids && !ids->vs_prog_id) {
        // The wrapper's own draws (Scaleform, the gamma pass, the present)
        // carry no shader objects; one that does and has no id was made
        // where the registry did not see it.
        const bool object = ids->shader[0] || ids->shader[2];
        ++(object ? g_by_id_unseen : g_by_id_none);
        static std::set<std::string> logged;
        if (object && logged.size() < 16 && logged.insert(vs_cp->name).second) {
            host_log("render: a GX draw from 0x%llx has vertex program %s (object 0x%llx, stages %llx %llx %llx) with no object id",
                     static_cast<unsigned long long>(ids->caller), vs_cp->name.c_str(), static_cast<unsigned long long>(ids->shader[0]),
                     static_cast<unsigned long long>(ids->shader[1]), static_cast<unsigned long long>(ids->shader[2]),
                     static_cast<unsigned long long>(ids->shader[4]));
        }
    }
    CachedProgram* ps_cp = s.ps_va ? program_for(ids ? ids->ps_prog_id : 0, s.ps_va) : nullptr;
    if (s.ps_va && !ps_cp) {
        draw_failed(kFailPsProgram);
        return false;
    }
    const std::vector<std::uint32_t>& vs_words = vs_cp->words;
    const std::string& vs_name = vs_cp->name;
    static const std::vector<std::uint32_t> no_words;
    static const std::string no_name = "none";
    const std::vector<std::uint32_t>& ps_words = ps_cp ? ps_cp->words : no_words;
    const std::string& ps_name = ps_cp ? ps_cp->name : no_name;
    std::uint64_t vs_hash = vs_cp->hash;
    std::uint64_t fetch_hash = 0;
    // GX draws carry their geometry from the call: compare mode
    // checks it here; render mode already drew its counts and index buffer
    // from it (gnm_exec) and takes the fetch shader below.
    const GxDrawObjects* gx_geometry = d.gx && d.gx_objects && d.gx_objects->geometry ? d.gx_objects : nullptr;
    const bool gx_geometry_compare = gx_geometry && d.gx_compare && !d.gx_render;
    std::uint64_t vertex_table = 0;  // render mode: the draw's vertex table in the host ring
    static thread_local VertexInputPlan vertex_input;
    bool use_vertex_input = false;  // BBHOST_VERTEX_INPUT: vertex_input replaces the fetch shader
    if (gx_geometry_compare) compare_gx_geometry(d, *gx_geometry);
    {
        if (vs_cp->fetch_sgpr == -2) {
            const gcn::Program vs_prog = gcn::decode(vs_words.data(), vs_words.size());
            vs_cp->fetch_sgpr = fetch_user_sgpr(vs_prog);
        }
        const int fs = vs_cp->fetch_sgpr;
        if (fs >= 0 && fs + 1 < 16) {
            const std::uint64_t user_fetch =
                static_cast<std::uint64_t>(s.vs_user[fs]) | (static_cast<std::uint64_t>(s.vs_user[fs + 1]) << 32);
            if (gx_geometry_compare) compare_gx_fetch(user_fetch, *gx_geometry);
            const bool layout_fetch = gx_geometry && d.gx_render && gx_geometry->fetch_va;
            s.fetch_va = layout_fetch ? gx_geometry->fetch_va : user_fetch;
            // The input layout's fetch shader, by the layout's id.
            FetchById* layout = nullptr;
            if (s.fetch_va) layout = fetch_for(layout_fetch && ids ? ids->il_id : 0, s.fetch_va, fetch);
            if (!s.fetch_va || !fetch) {
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1) < 4) {
                    host_log("render: VS %s expects a fetch shader in s[%d:%d] but 0x%llx is unreadable", vs_name.c_str(), fs,
                             fs + 1, static_cast<unsigned long long>(s.fetch_va));
                }
                draw_failed(kFailFetch);
                return false;
            }
            fetch_hash = fetch->hash;
            if (gx_geometry_compare) probe_gx_vertex_table(fetch->words, s.vs_user, *gx_geometry);
            static const bool log_fetch = [] {
                const char* e = std::getenv("BBHOST_LOG_FETCH");
                return e && e[0] == '1';
            }();
            if (log_fetch && gx_geometry) log_fetch_shader(fetch_hash, fetch->words, *gx_geometry);
            if (gx_geometry && d.gx_render && g_vertex_input) {
                const GxDrawObjects* geometry = gx_geometry;
                static thread_local GxDrawObjects own_table;
                if (geometry->vtx_ud == 0xff && g_vertex_table_from_memory &&
                    vertex_table_from_memory(fetch->words, fetch_hash, s.vs_user, *geometry, own_table)) {
                    geometry = &own_table;
                }
                use_vertex_input = plan_vertex_input(fetch->words, fetch_hash, *geometry, vertex_input, layout);
                bump(use_vertex_input ? g_vertex_input_draws : g_vertex_input_kept);
            }
            if (gx_geometry && d.gx_render && !use_vertex_input) {
                RenderCostTimer timer(kRenderCostVertexTable);
                vertex_table = place_vertex_table(*gx_geometry);
            }
        }
    }
    const std::vector<std::uint32_t>& fetch_words = fetch ? fetch->words : no_words;
    draw_stamp.to(kRenderCostPrograms);
    const std::uint64_t ps_hash = ps_cp ? ps_cp->hash : 0;
    // Rendering from GX resolves resources from the draw's
    // records where they cover a binding.
    // BBHOST_SF_SNAPSHOT=0: Scaleform draws read their constant buffers from
    // guest memory when the GPU runs them, like every other draw. The game
    // writes a frame's UI uniforms (per-quad transforms and colours, ~100
    // bytes a draw) at the same addresses the frame before used, trusting
    // the GPU to be done with them; ours can be a frame behind, and a draw
    // then read the next frame's transform - a quad pulled over the whole
    // screen, flooded blue or green (the glitch hunt: 3
    // of 3 such frames had their inputs rewritten in flight, against 2.7% of
    // Scaleform draws). Copied when the command processor records the draw,
    // they are what it saw with everything else it resolved then.
    static const bool sf_snapshot = [] {
        const char* e = std::getenv("BBHOST_SF_SNAPSHOT");
        return !(e && e[0] == '0');
    }();
    t_snapshot_cbs = sf_snapshot && d.gx_token && d.gx_token_kind && std::strcmp(d.gx_token_kind, "scaleform") == 0;
    const GxDrawRecords* gx_records = d.gx && d.gx_render && d.gx_objects ? d.gx_objects->records : nullptr;
    const GxStageRecords* gx_stage[2] = {gx_records ? &gx_records->stage[0] : nullptr, gx_records ? &gx_records->stage[1] : nullptr};
    if (glitch_on()) {
        std::snprintf(t_draw_origin, sizeof(t_draw_origin), "%s token, gx inputs %s, gx_render %s, objects %s, records %s, partial %s, "
                      "called at flip %llu (now %llu)",
                      d.gx_token ? (d.gx_token_kind ? d.gx_token_kind : "a") : "no (packets)", d.gx ? "yes" : "no", d.gx_render ? "yes" : "no", d.gx_objects ? "yes" : "no",
                      d.gx_objects && d.gx_objects->records ? "yes" : "no", d.gx_partial ? "yes" : "no",
                      static_cast<unsigned long long>(d.gx_objects ? d.gx_objects->call_flip : 0),
                      static_cast<unsigned long long>(hle_video_flip_count()));
    }
    // The entries the image prefetch will look its views up in, asked for now
    // and together: the table is larger than the caches, and one at a time
    // each is a miss. The key and the pipeline come first.
    if (g_view_by_id) {
        for (const GxStageRecords* st : gx_stage) {
            if (!st) continue;
            for (std::uint64_t m = st->obj_tex_set; m; m &= m - 1) texture_view_by_id_prefetch(st->obj_tex_id[__builtin_ctzll(m)]);
        }
    }
    // With binding plans the index is only a reset: the few resolutions left
    // outside the plans scan the descriptors, and a recycled record set must
    // not find the last draw's index.
    index_gx_stage(0, g_binding_plans ? nullptr : gx_stage[0]);
    index_gx_stage(1, g_binding_plans ? nullptr : gx_stage[1]);
    if (g_render_cost_enabled && gx_records) {
        // BBHOST_GX_COST: sampled constant buffers against their contents at the call.
        for (int k = 0; k < 2; ++k) {
            const GxStageRecords& st = gx_records->stage[k];
            if (!st.cb_hashed) continue;
            g_cb_check_stages.fetch_add(1, std::memory_order_relaxed);
            for (int j = 0; j < 14; ++j) {
                if (!((st.cb_hashed >> j) & 1)) continue;
                std::uint32_t rec[4];
                std::memcpy(rec, st.obj_cb + j * 16, sizeof(rec));
                const std::uint64_t addr = rec[0] | (static_cast<std::uint64_t>(rec[1] & 0xfff) << 32);
                const std::uint64_t bytes = static_cast<std::uint64_t>(rec[2]) * 16;
                if (!hle_kernel_va_mapped(addr, bytes)) continue;
                g_cb_check_buffers.fetch_add(1, std::memory_order_relaxed);
                g_cb_check_bytes.fetch_add(bytes, std::memory_order_relaxed);
                if (render_cb_fnv(addr, bytes) != st.cb_hash[j]) {
                    g_cb_check_changed.fetch_add(1, std::memory_order_relaxed);
                    // BBHOST_WATCH_CHANGED_CB=1: watch the first buffer seen changing
                    // after its call, to find the code that writes it (the watch
                    // report prints at exit).
                    static const bool watch_changed = [] {
                        const char* e = std::getenv("BBHOST_WATCH_CHANGED_CB");
                        return e && e[0] == '1';
                    }();
                    static std::atomic<bool> watched{false};
                    if (watch_changed && !watched.exchange(true)) {
                        g_watch_cb_lo.store(addr);
                        g_watch_cb_hi.store(addr + bytes);
                        // Nothing wrote buffer 9 through its CPU address
                        // (two watch runs); GX updates it with GPU copies. Watch
                        // another CPU mapping of the same physical
                        // memory when there is one, else the GPU alias, else the
                        // address itself.
                        const std::uint64_t other = hle_kernel_mappings_of(addr);
                        const std::uint64_t alias = hle_kernel_gpu_alias(addr);
                        const std::uint64_t watch_at = other ? other : alias ? alias : addr;
                        hle_watch_arm(watch_at, watch_at + bytes);
                    }
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) < 16) {
                        float f[8] = {};
                        std::memcpy(f, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(addr)),
                                    static_cast<std::size_t>(std::min<std::uint64_t>(bytes, sizeof(f))));
                        host_log("render: GX %s constant buffer %d at 0x%llx (%llu bytes) changed after the call (flip %llu, now %llu; "
                                 "site 0x%llx, shader %s): %g %g %g %g %g %g %g %g",
                                 k ? "PS" : "VS", j, static_cast<unsigned long long>(addr), static_cast<unsigned long long>(bytes),
                                 static_cast<unsigned long long>(d.gx_objects->call_flip),
                                 static_cast<unsigned long long>(hle_video_flip_count()),
                                 static_cast<unsigned long long>(d.gx_objects->caller), k ? ps_name.c_str() : vs_name.c_str(), f[0],
                                 f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
                    }
                }
            }
        }
    }
    draw_stamp.to(kRenderCostCbCheck);
    // Image dimensions from the resolved T#s decide the translated image types,
    // and sampler modes the unnormalized samplers. Each binding resolves once,
    // here; the prefetch takes the words (KeyStage).
    static thread_local KeyStage key_stages[2];
    const gcn::TranslateResult& vs_paths = paths_for(vs_hash, vs_words, gcn::Stage::Vertex, s.vs_rsrc1, s.vs_rsrc2);
    const gcn::TranslateResult* const ps_paths =
        ps_words.empty() ? nullptr : &paths_for(ps_hash, ps_words, gcn::Stage::Pixel, s.ps_rsrc1, s.ps_rsrc2);
    t_resolve_site = kSiteKey;
    resolve_key_stage(&vs_paths, s.vs_user, gx_stage[0], key_stages[0]);
    resolve_key_stage(ps_paths, s.ps_user, gx_stage[1], key_stages[1]);
    const std::vector<std::pair<std::uint32_t, bool>>& vs_dims = key_stages[0].dims;
    const std::vector<std::pair<std::uint32_t, bool>>& ps_dims = key_stages[1].dims;
    const std::vector<bool>& vs_sampler_modes = key_stages[0].modes;
    const std::vector<bool>& ps_sampler_modes = key_stages[1].modes;
    std::uint64_t key = pipeline_key(s, vs_hash, use_vertex_input ? vertex_input.hash : fetch_hash, ps_hash);
    const VertexInputPlan* vertex_plan = use_vertex_input ? &vertex_input : nullptr;
    {
        // The sampler modes and image dimensions the translation depends on,
        // packed and hashed once; a separator ends each list.
        static thread_local std::vector<std::uint32_t> shape;
        shape.clear();
        for (const bool mode : vs_sampler_modes) shape.push_back(mode ? 1u : 0u);
        shape.push_back(~0u);
        for (const bool mode : ps_sampler_modes) shape.push_back(mode ? 1u : 0u);
        shape.push_back(~0u);
        for (const auto& d2 : vs_dims) shape.push_back(d2.first | (d2.second ? 0x80000000u : 0u));
        shape.push_back(~0u);
        for (const auto& d2 : ps_dims) shape.push_back(d2.first | (d2.second ? 0x80000000u : 0u));
        key = fnv1a(shape.data(), shape.size() * sizeof(std::uint32_t), key);
    }
    draw_stamp.to(kRenderCostKey);
    GpuPhaseTimer pipeline_timer(kPhasePipeline, phase_weight);
    // BBHOST_LEAN_FIRST: translate and lay out the fallback pipeline, but create
    // its Vulkan pipeline only when a draw binds it (below).
    const bool lean_first = g_lean_first && g.cb_lean;
    const bool layout_only = lean_first && g_layout_from_lean && !gfx_lookup(key);
    if (layout_only) g_fallbacks_deferred.fetch_add(1, std::memory_order_relaxed);
    GfxPipeline& pl = gfx_pipeline(s, key, vs_words, fetch_words, ps_words, vs_name, ps_name, vs_dims, ps_dims,
                                   vs_sampler_modes, ps_sampler_modes, false, vertex_plan, !lean_first, layout_only);
    pipeline_timer.next(kPhaseBind);
    if (pl.failed) {
        draw_failed(kFailPipeline);
        // Which pipelines, once each: the exit report only counts them.
        static std::set<std::string> said;
        if (said.size() < 24 && said.insert(pl.name).second) {
            host_log("render: pipeline %s failed to build; its draws are skipped (first at flip %llu)", pl.name.c_str(),
                     static_cast<unsigned long long>(hle_video_flip_count()));
        }
        return false;
    }
    // A pipeline that reads memory through the page table could read a
    // copy-back's destination before its copy is in place (copy versions).
    if (copy_versions_pending_locked() && (!pl.vs.meta().walks.empty() || !pl.ps.meta().walks.empty())) {
        copy_versions_put_in_place_locked(kCvWalks);
    }
    // A draw given CPU-written windows (YEBIS) reads them from its own copies
    // only through bound constant buffers; a page-table walk would read the
    // ring slot itself, which the CPU may already have rewritten for a draw
    // sixteen on. Said once a pipeline; none known to.
    if (!t_draw_windows.empty() && !pl.building && (!pl.vs.meta().walks.empty() || !pl.ps.meta().walks.empty())) {
        static std::set<std::string> said;
        if (said.size() < 16 && said.insert(pl.name).second) {
            host_log("render: YEBIS pipeline %s keeps page-table walks (vs %zu, ps %zu reasons): its windows can be read rewritten",
                     pl.name.c_str(), pl.vs.meta().walks.size(), pl.ps.meta().walks.size());
        }
    }
    draw_stamp.to(kRenderCostLookup);

    if (g_trace) {
        survey_images(pl.vs.meta(), s.vs_user, "VS");
        survey_images(pl.ps.meta(), s.ps_user, "PS");
    }
    if (d.gx_objects) {
        probe_gx_descriptors(pl, *d.gx_objects);
        if (d.gx_objects->records && !d.gx_render) {
            compare_gx_bindings(pl, s.vs_user, s.ps_user, *d.gx_objects);
            compare_gx_user_data(s.vs_user, d.gx_objects->records->stage[0], 0);
            if (d.gx_objects->shader[4]) compare_gx_user_data(s.ps_user, d.gx_objects->records->stage[1], 1);
        }
    }
    // Resolved once per draw: the prefetch uploads outside the render pass and
    // the bind below writes the same views (it used to resolve and look up
    // every texture again, and hash hot ones twice).
    static thread_local StageImages stage_images[2];
    {
        GpuPhaseTimer t(kPhasePrefetch, phase_weight);
        if (!pl.paths_checked) {
            pl.paths_checked = true;
            pl.paths_match[0] = key_stage_matches(pl.vs.meta(), vs_paths);
            pl.paths_match[1] = ps_paths ? key_stage_matches(pl.ps.meta(), *ps_paths) : pl.ps.meta().images.empty() && pl.ps.meta().samplers.empty();
        }
        for (int st = 0; st < 2; ++st) bump(pl.paths_match[st] ? g_prefetch_key_words : g_prefetch_resolved);
        t_resolve_site = kSitePrefetch;
        t_draw_pipeline = pl.name.c_str();
        // BBHOST_GLITCH_ORIGIN=<pipeline prefix>: where that pipeline's draws
        // come from, every time it changes (the first 60 changes).
        static const char* origin_of = std::getenv("BBHOST_GLITCH_ORIGIN");
        if (origin_of && glitch_on() && pl.name.compare(0, std::strlen(origin_of), origin_of) == 0) {
            static std::string last;
            static int changes = 0;
            if (last != t_draw_origin && changes++ < 60) {
                last = t_draw_origin;
                host_log("glitch: draw %llu (flip %llu) %s comes from: %s", static_cast<unsigned long long>(g_draw_rec_next),
                         static_cast<unsigned long long>(hle_video_flip_count()), pl.name.c_str(), t_draw_origin);
            }
            static int samples = 0;
            if (std::strstr(t_draw_origin, "scaleform") && (samples < 4 || hle_video_flip_count() % 2000 == 0) && samples++ < 40) {
                host_log("glitch:   draw %llu (flip %llu), its Scaleform token: %s", static_cast<unsigned long long>(g_draw_rec_next),
                         static_cast<unsigned long long>(hle_video_flip_count()), hle_gx_scaleform_last().c_str());
            }
        }
        prefetch_stage_images(pl.vs.meta(), s.vs_user, stage_images[0], gx_stage[0], pl.paths_match[0] ? &key_stages[0] : nullptr);
        prefetch_stage_images(pl.ps.meta(), s.ps_user, stage_images[1], gx_stage[1], pl.paths_match[1] ? &key_stages[1] : nullptr);
        t_resolve_site = kSiteBuffers;
    }
    draw_stamp.to(kRenderCostPrefetch);
    // BBHOST_CAPTURE_DRAW (draw_capture.h): run everything recorded so far
    // before this draw's sets exist, so memory and images hold its inputs.
    const bool capture = capture_candidate(pl.name, d.index_count, hle_video_flip_count(), pl.lean_variant && !pl.lean_variant->failed);
    if (capture) capture_settle_locked();
    // BBHOST_WATCH_PS_IMAGE=<pipeline prefix>:<image index>: for every draw of
    // that pipeline, resolve the image's T# and hash the memory behind it, and
    // log a line whenever either changes. One draw's detail says what a pass
    // reads; this says *when* what it reads changed, which is the question
    // when one frame in a hundred comes out wrong.
    if (g_keep_verts && pl.name.find(g_keep_verts) == 0) {
        KeptVerts& kv = g_kept_verts[hle_video_flip_count() % kKeptVerts];
        if (kv.flip != hle_video_flip_count()) {
            kv.flip = hle_video_flip_count();
            kv.draws = 0;
            kv.text.clear();
            kv.ranges.clear();
            kv.range_hash.clear();
            kv.probed = false;
        }
        if (kv.draws++ < 12) {
            char line[512];
            auto add = [&kv, &line] { kv.text += line; };
            std::snprintf(line, sizeof(line), "draw %llu %s n=%u prim=%u index_va=0x%llx type=%u vp=(%g,%g %gx%g)\n",
                          static_cast<unsigned long long>(g_draw_rec_next), pl.name.c_str(), d.index_count, s.prim,
                          static_cast<unsigned long long>(d.index_va), d.index_type, s.vport[1] - s.vport[0], s.vport[3] - s.vport[2],
                          s.vport[0] * 2, s.vport[2] * 2);
            add();
            std::snprintf(line, sizeof(line),
                          "  vs user %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                          s.vs_user[0], s.vs_user[1], s.vs_user[2], s.vs_user[3], s.vs_user[4], s.vs_user[5], s.vs_user[6],
                          s.vs_user[7], s.vs_user[8], s.vs_user[9], s.vs_user[10], s.vs_user[11], s.vs_user[12], s.vs_user[13],
                          s.vs_user[14], s.vs_user[15]);
            add();
            // Every user SGPR pair that points at mapped memory: the first 16
            // dwords, as hex and as floats. A quad that covers the screen has
            // its transform somewhere in here.
            for (int k = 0; k + 1 < 16; k += 2) {
                // The high dword carries the top bits of the address *and*,
                // when the pair is a V#, a stride. Taking only the low byte as
                // address bits covers both; an earlier filter insisted on a
                // high byte of 1 and so skipped every pointer in this heap,
                // which is where the packed vertices' transform lives.
                const std::uint64_t ptr = static_cast<std::uint64_t>(s.vs_user[k]) |
                                          (static_cast<std::uint64_t>(s.vs_user[k + 1] & 0xff) << 32);
                if (!ptr || !hle_kernel_va_mapped(ptr, 128)) continue;
                const std::uint32_t stride = (s.vs_user[k + 1] >> 16) & 0x3fff;
                const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(ptr));
                const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(ptr));
                const std::uint32_t recs = k + 2 < 16 ? s.vs_user[k + 2] : 0;
                const std::uint64_t pbytes = std::min<std::uint64_t>(static_cast<std::uint64_t>(recs ? recs : 8) * (stride ? stride : 16), 4096);
                std::uint64_t ph = 1469598103934665603ull;
                if (hle_kernel_va_mapped(ptr, pbytes)) {
                    const auto* pu = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(ptr));
                    for (std::uint64_t q = 0; q < pbytes; ++q) ph = (ph ^ pu[q]) * 1099511628211ull;
                }
                std::snprintf(line, sizeof(line), "  s[%d:%d] -> 0x%llx stride=%u records=%u digest %llu hash=%016llx\n", k, k + 1,
                              static_cast<unsigned long long>(ptr), stride, recs, static_cast<unsigned long long>(pbytes),
                              static_cast<unsigned long long>(ph));
                add();
                // As many records as the draw says it has, not a fixed eight: a
                // draw with thirty-six of them hid any difference past the
                // first eight, and the first eight were identical.
                const std::uint32_t count = k + 2 < 16 ? s.vs_user[k + 2] : 0;
                // Deep only where the per-quad transforms live (a strided
                // pair); the plain pointers get a header's worth. Dumping
                // everything at every draw of every frame made the run too
                // slow to reach the world at all.
                const int rows = g_keep_verts_rows <= 0
                                     ? 0
                                     : stride ? static_cast<int>(std::min<std::uint32_t>(count ? count : 8u, 48u)) : 8;
                if (!hle_kernel_va_mapped(ptr, static_cast<std::size_t>(rows) * 16)) continue;
                for (int r = 0; r < rows; ++r) {
                    std::snprintf(line, sizeof(line), "    +%03x %08x %08x %08x %08x  %g %g %g %g\n", r * 16, u[r * 4], u[r * 4 + 1],
                                  u[r * 4 + 2], u[r * 4 + 3], f[r * 4], f[r * 4 + 1], f[r * 4 + 2], f[r * 4 + 3]);
                    add();
                }
            }
            // The fetch shader's V# table, and the vertices behind each one.
            const std::uint64_t tbl = static_cast<std::uint64_t>(s.vs_user[2]) | (static_cast<std::uint64_t>(s.vs_user[3]) << 32);
            if (hle_kernel_va_mapped(tbl, 16 * 8)) {
                const auto* v = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tbl));
                for (int k = 0; k < 4; ++k) {
                    const std::uint32_t* w = v + k * 4;
                    if (!w[0] && !w[1]) break;
                    const std::uint64_t vb = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
                    const std::uint32_t stride = (w[1] >> 16) & 0x3fff;
                    {
                        char probe[256];
                        shadow_probe_locked(vb, static_cast<std::uint64_t>(w[2] ? w[2] : 1) * (stride ? stride : 16), probe,
                                            sizeof(probe));
                        std::snprintf(line, sizeof(line), "  V#%d shadow %s\n", k, probe);
                        add();
                        const std::uint64_t vbytes = static_cast<std::uint64_t>(w[2] ? w[2] : 1) * (stride ? stride : 16);
                        if (kv.ranges.size() < 64) {
                            kv.ranges.emplace_back(vb, vbytes);
                            kv.range_hash.push_back(hle_kernel_va_mapped(vb, vbytes)
                                                        ? fnv1a(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(vb)), vbytes)
                                                        : 0);
                        }
                    }
                    std::snprintf(line, sizeof(line), "  V#%d base=0x%llx stride=%u records=%u dfmt=%u nfmt=%u dst_sel=%03x\n", k,
                                  static_cast<unsigned long long>(vb), stride, w[2], (w[3] >> 15) & 0xf, (w[3] >> 12) & 7,
                                  w[3] & 0xfff);
                    add();
                    // Every record the V# declares, up to forty-eight. Twelve
                    // was a guess that left the back half of the biggest draw
                    // uncompared, which is exactly where a difference would
                    // have been invisible.
                    {
                        const std::uint64_t vb_bytes = static_cast<std::uint64_t>(w[2] ? w[2] : 1) * (stride ? stride : 16);
                        std::uint64_t vh = 1469598103934665603ull;
                        if (hle_kernel_va_mapped(vb, vb_bytes)) {
                            const auto* vu = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(vb));
                            for (std::uint64_t q = 0; q < vb_bytes; ++q) vh = (vh ^ vu[q]) * 1099511628211ull;
                        }
                        std::snprintf(line, sizeof(line), "  V#%d digest %llu bytes hash=%016llx\n", k,
                                      static_cast<unsigned long long>(vb_bytes), static_cast<unsigned long long>(vh));
                        add();
                    }
                    const int vrows = g_keep_verts_rows <= 0
                                          ? 0
                                          : static_cast<int>(std::min<std::uint32_t>(w[2] ? w[2] : 8u,
                                                                                     static_cast<std::uint32_t>(g_keep_verts_rows)));
                    if (!stride || !hle_kernel_va_mapped(vb, static_cast<std::size_t>(stride) * vrows + 16)) continue;
                    for (int r = 0; r < vrows; ++r) {
                        const auto* rf = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vb + static_cast<std::uint64_t>(r) * stride));
                        const auto* ru = reinterpret_cast<const std::uint32_t*>(rf);
                        std::snprintf(line, sizeof(line), "    v%d: %08x %08x %08x %08x (%g %g %g %g)\n", r, ru[0], ru[1], ru[2],
                                      ru[3], rf[0], rf[1], rf[2], rf[3]);
                        add();
                    }
                }
            }
            // Which buffer each vertex binding actually comes from. The
            // vertex-input path may bind a device-local shadow of the guest's
            // memory instead of the import (buffer_shadow.cpp), and the fetch
            // path never does - so "shadow or import, and at what offset" is
            // the one thing that can differ between two frames handed the same
            // bytes.
            std::snprintf(line, sizeof(line), "  vertex input %s, %zu binding(s)\n", use_vertex_input ? "on" : "off (fetch shader)",
                          use_vertex_input ? vertex_input.bindings.size() : std::size_t{0});
            add();
            if (use_vertex_input) {
                for (std::size_t b = 0; b < vertex_input.bindings.size() && b < 16; ++b) {
                    std::snprintf(line, sizeof(line), "    binding %zu buffer=%p offset=%llu stride=%u va=0x%llx\n", b,
                                  static_cast<const void*>(vertex_input.buffer[b]),
                                  static_cast<unsigned long long>(vertex_input.offset[b]), vertex_input.bindings[b].stride,
                                  static_cast<unsigned long long>(vertex_input.va[b]));
                    add();
                }
            }
            // The pixel side: its user data, every image and sampler it
            // resolves, and a hash of the memory behind each image. The
            // vertices being identical puts the difference here or in our own
            // state, so both have to be on the record for the same frame.
            std::snprintf(line, sizeof(line), "  ps user %08x %08x %08x %08x %08x %08x %08x %08x\n", s.ps_user[0], s.ps_user[1],
                          s.ps_user[2], s.ps_user[3], s.ps_user[4], s.ps_user[5], s.ps_user[6], s.ps_user[7]);
            add();
            for (int k = 0; k + 1 < 8; k += 2) {
                const std::uint64_t ptr = static_cast<std::uint64_t>(s.ps_user[k]) |
                                          (static_cast<std::uint64_t>(s.ps_user[k + 1] & 0xff) << 32);
                if (!ptr || !hle_kernel_va_mapped(ptr, 128)) continue;
                const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(ptr));
                const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(ptr));
                std::snprintf(line, sizeof(line), "  ps s[%d:%d] -> 0x%llx\n", k, k + 1, static_cast<unsigned long long>(ptr));
                add();
                for (int r = 0; r < 8; ++r) {
                    std::snprintf(line, sizeof(line), "    +%03x %08x %08x %08x %08x  %g %g %g %g\n", r * 16, u[r * 4], u[r * 4 + 1],
                                  u[r * 4 + 2], u[r * 4 + 3], f[r * 4], f[r * 4 + 1], f[r * 4 + 2], f[r * 4 + 3]);
                    add();
                }
            }
            for (std::size_t ii = 0; ii < pl.ps.meta().images.size(); ++ii) {
                std::uint32_t w[8] = {};
                if (!resolve_resource_impl(pl.ps.meta().images[ii].path, s.ps_user, 8, w)) continue;
                const std::uint64_t tb = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8;
                // The whole surface, not its first 4 KiB. A 4 KiB hash covers
                // one row of a 1024-wide atlas, so it said "identical" about
                // 3% of the texture and nothing about the rest.
                const std::uint32_t tw = (w[2] & 0x3fff) + 1, th = ((w[2] >> 14) & 0x3fff) + 1;
                const std::uint64_t tbytes = std::min<std::uint64_t>(static_cast<std::uint64_t>(tw) * th * 4, 4ull << 20);
                std::uint64_t hash = 1469598103934665603ull;
                std::size_t nonzero = 0;
                if (hle_kernel_va_mapped(tb, tbytes)) {
                    const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tb));
                    for (std::uint64_t q = 0; q < tbytes / 4; ++q) {
                        hash = (hash ^ u[q]) * 1099511628211ull;
                        nonzero += u[q] != 0;
                    }
                }
                std::snprintf(line, sizeof(line),
                              "  ps image %zu T# %08x %08x %08x %08x %08x %08x %08x %08x base=0x%llx %ux%u hash=%016llx nonzero=%zu\n",
                              ii, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], static_cast<unsigned long long>(tb), tw, th,
                              static_cast<unsigned long long>(hash), nonzero);
                add();
            }
            for (std::size_t si2 = 0; si2 < pl.ps.meta().samplers.size(); ++si2) {
                std::uint32_t w[4] = {};
                if (!resolve_resource_impl(pl.ps.meta().samplers[si2].path, s.ps_user, 4, w)) continue;
                std::snprintf(line, sizeof(line), "  ps sampler %zu S# %08x %08x %08x %08x\n", si2, w[0], w[1], w[2], w[3]);
                add();
            }
            if (d.index_va) {
                char probe[256];
                shadow_probe_locked(d.index_va, static_cast<std::uint64_t>(d.index_count) * (d.index_type ? 4u : 2u), probe,
                                    sizeof(probe));
                std::snprintf(line, sizeof(line), "  index shadow [0x%llx +%u] %s\n", static_cast<unsigned long long>(d.index_va),
                              d.index_count * (d.index_type ? 4u : 2u), probe);
                add();
                const std::uint64_t ibytes = static_cast<std::uint64_t>(d.index_count) * (d.index_type ? 4u : 2u);
                if (kv.ranges.size() < 64) {
                    kv.ranges.emplace_back(d.index_va, ibytes);
                    kv.range_hash.push_back(hle_kernel_va_mapped(d.index_va, ibytes)
                                                ? fnv1a(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(d.index_va)), ibytes)
                                                : 0);
                }
            }
            if (d.index_va && hle_kernel_va_mapped(d.index_va, static_cast<std::size_t>(d.index_count) * (d.index_type ? 4 : 2) + 8)) {
                const auto* i16 = reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(d.index_va));
                const auto* i32 = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(d.index_va));
                // Every index, not twelve of them. A draw of sixty had 48 that
                // were never compared, and one index out of range is enough to
                // fetch a vertex from nowhere.
                const std::uint32_t nidx = std::min<std::uint32_t>(d.index_count, 192);
                std::uint32_t lo_i = ~0u, hi_i = 0;
                std::uint64_t ihash = 1469598103934665603ull;
                for (std::uint32_t q = 0; q < nidx; ++q) {
                    const std::uint32_t v = d.index_type ? i32[q] : i16[q];
                    lo_i = std::min(lo_i, v);
                    hi_i = std::max(hi_i, v);
                    ihash = (ihash ^ v) * 1099511628211ull;
                }
                // One line, one snprintf: a call per index made the dump heavy
                // enough to change the run's timing, and two runs in a row then
                // produced no flash at all to compare.
                std::snprintf(line, sizeof(line), "  indices%s n=%u range %u..%u hash=%016llx\n", d.index_type ? "32" : "16", nidx,
                              lo_i, hi_i, static_cast<unsigned long long>(ihash));
                add();
            }
        }
    }
    // BBHOST_ONLY_PASS=<prefix>@0x<target>: into that target, draw only that
    // pass. Everything else there is dropped, so what reaches the screen is
    // exactly what the pass puts on it - which answers "does one of its quads
    // cover the screen?" by looking, with no readback and no broken render
    // pass, the two things this driver will not take.
    static const char* only_pass = std::getenv("BBHOST_ONLY_PASS");
    if (only_pass) {
        static const std::string oname = [] {
            const std::string t(only_pass);
            const std::size_t at = t.find('@');
            return at == std::string::npos ? t : t.substr(0, at);
        }();
        static const std::uint64_t otarget = [] {
            const char* at = std::strchr(only_pass, '@');
            return at ? std::strtoull(at + 1, nullptr, 0) : 0ull;
        }();
        if (s.color[0] && (!otarget || s.color[0]->base == otarget) && pl.name.find(oname) != 0) return true;
    }
    // BBHOST_SKIP_PASS=<prefix>[,<prefix>...]: drop those draws. Readbacks
    // cost a broken render pass and this driver will not take many of those,
    // so the safe way to ask "is it this pass?" is to remove it and look at
    // the screen. No copies, no pass breaks, no risk to the run.
    static const char* skip_pass = std::getenv("BBHOST_SKIP_PASS");
    if (skip_pass) {
        static const std::vector<std::string> names = [] {
            std::vector<std::string> out;
            std::string t(skip_pass);
            std::size_t pos = 0;
            while (pos <= t.size()) {
                const std::size_t c = t.find(',', pos);
                std::string one = t.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                if (!one.empty()) out.push_back(one);
                if (c == std::string::npos) break;
                pos = c + 1;
            }
            return out;
        }();
        for (const std::string& n : names) {
            if (pl.name.find(n) == 0) {
                static std::atomic<std::uint64_t> skipped{0};
                if (skipped.fetch_add(1) % 100000 == 0) {
                    host_log("render: skipping %s (BBHOST_SKIP_PASS), %llu so far", pl.name.c_str(),
                             static_cast<unsigned long long>(skipped.load()));
                }
                return true;
            }
        }
    }
    if (g_keep_input) {
        // ":rt" keeps the pass's own colour target instead of an image it
        // reads - which, named on the pass *after* the one under suspicion, is
        // that one's output.
        static const std::vector<KeepSpec> specs = parse_keep_specs(g_keep_input);
        for (std::size_t si = 0; si < specs.size(); ++si) {
            const KeepSpec& spec = specs[si];
            const std::string& kname = spec.name;
            const std::size_t kidx = spec.idx;
            const bool krt = spec.rt;
            if (spec.target && !(s.color[0] && s.color[0]->base == spec.target)) continue;
            static bool armed[kKeepSpecs] = {};
            static std::uint64_t armed_flip[kKeepSpecs] = {};
            if (spec.after) {
                // Arm on the nth draw of the pass, then copy at the next draw
                // into its target: "#6+" is the composite immediately after
                // that pass's sixth draw, which is the only way to attribute a
                // change to one draw when its neighbours vary by frame.
                static unsigned after_seen[kKeepSpecs] = {};
                static std::uint64_t after_flip[kKeepSpecs] = {};
                if (after_flip[si] != hle_video_flip_count()) {
                    after_flip[si] = hle_video_flip_count();
                    after_seen[si] = 0;
                    armed[si] = false;
                }
                if (pl.name.find(kname) == 0 && s.color[0]) {
                    if (++after_seen[si] == spec.occurrence) {
                        armed[si] = true;
                        armed_flip[si] = hle_video_flip_count();
                    }
                    continue;
                }
                if (!armed[si] || armed_flip[si] != hle_video_flip_count() || !s.color[0]) continue;
                armed[si] = false;
            } else if (!spec.target && (pl.name.find(kname) != 0 || !(krt || kidx < pl.ps.meta().images.size()))) {
                continue;
            }
            std::uint32_t w[8] = {};
            RtImage* src = nullptr;
            if (krt || spec.after) {
                src = s.color[0];
            } else if (resolve_resource_impl(pl.ps.meta().images[kidx].path, s.ps_user, 8, w)) {
                const std::uint64_t base = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8;
                src = find_render_target(base);
            }
            if (src && src->initialised && !src->depth) {
                KeptInput& k = g_kept_inputs[si][hle_video_flip_count() % kKeptInputs];
                // The *first* match of this flip only. A pass that runs thirty
                // times a frame would otherwise mean thirty full-screen
                // readbacks and thirty broken render passes, which is what put
                // the driver on the floor; the draw index in the file name
                // says exactly where the one copy was taken.
                if (!spec.last && k.filled && k.flip == hle_video_flip_count()) continue;
                static unsigned seen_this_flip[kKeepSpecs] = {};
                static std::uint64_t seen_flip[kKeepSpecs] = {};
                if (seen_flip[si] != hle_video_flip_count()) {
                    seen_flip[si] = hle_video_flip_count();
                    seen_this_flip[si] = 0;
                }
                if (!spec.last && ++seen_this_flip[si] != spec.occurrence) continue;
                const std::size_t bpp = rt_bytes_per_pixel(*src);
                const std::uint64_t bytes = static_cast<std::uint64_t>(src->width) * src->height * bpp;
                if (k.buf.buffer == VK_NULL_HANDLE || k.width != src->width || k.height != src->height || k.bpp != bpp) {
                    if (k.buf.buffer) {
                        vkDestroyBuffer(g.device, k.buf.buffer, nullptr);
                        vkFreeMemory(g.device, k.buf.memory, nullptr);
                        k.buf = DevBuffer{};
                    }
                    if (create_dev_buffer(k.buf, bytes, true, true)) {
                        k.width = src->width;
                        k.height = src->height;
                        k.bpp = bpp;
                    }
                }
                if (k.buf.buffer) {
                    render_end_pass_locked();  // a copy cannot be recorded inside one
                    k.format = src->format;
                    VkBufferImageCopy full{};
                    full.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    full.imageExtent = {src->width, src->height, 1};
                    vkCmdCopyImageToBuffer(g_cmd(), src->image, VK_IMAGE_LAYOUT_GENERAL, k.buf.buffer, 1, &full);
                    k.flip = hle_video_flip_count();
                    k.landed_at = g.flushes;
                    k.at = g_draw_rec_next;
                    k.base = src->base;
                    std::snprintf(k.pass, sizeof(k.pass), "%s", pl.name.c_str());
                    k.filled = true;
                }
            }
        }
    }
    static const char* watch_img = std::getenv("BBHOST_WATCH_PS_IMAGE");
    if (watch_img) {
        static const std::string wname = [] {
            const std::string t(watch_img);
            const std::size_t c = t.find(':');
            return c == std::string::npos ? t : t.substr(0, c);
        }();
        static const std::size_t widx = [] {
            const char* c = std::strchr(watch_img, ':');
            return c ? static_cast<std::size_t>(std::strtoul(c + 1, nullptr, 10)) : std::size_t{0};
        }();
        // An index of 99 watches every image the pass reads.
        for (std::size_t ii = 0; ii < pl.ps.meta().images.size(); ++ii) {
            if (pl.name.find(wname) != 0 || (widx != 99 && ii != widx)) continue;
            std::uint32_t w[8] = {};
            if (resolve_resource_impl(pl.ps.meta().images[ii].path, s.ps_user, 8, w)) {
                const std::uint64_t tb = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8;
                // Enough of the surface to tell one curve from another: a 1D
                // array's three layers are small and adjacent.
                std::uint64_t hash = 1469598103934665603ull;
                std::uint32_t first[8] = {};
                if (hle_kernel_va_mapped(tb, 4096)) {
                    const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tb));
                    for (int k = 0; k < 1024; ++k) hash = (hash ^ u[k]) * 1099511628211ull;
                    for (int k = 0; k < 8; ++k) first[k] = u[k];
                }
                static std::uint64_t last_hash[8] = {};
                static std::uint32_t last_t[8][8] = {};
                if (ii < 8 && (hash != last_hash[ii] || std::memcmp(last_t[ii], w, sizeof(w)) != 0)) {
                    last_hash[ii] = hash;
                    std::memcpy(last_t[ii], w, sizeof(w));
                    host_log("watch: %s image %zu at flip %llu: T# %08x %08x %08x %08x %08x %08x %08x %08x "
                             "(base 0x%llx) memory %08x %08x %08x %08x %08x %08x %08x %08x hash %016llx",
                             pl.name.c_str(), ii, static_cast<unsigned long long>(hle_video_flip_count()), w[0], w[1], w[2], w[3],
                             w[4], w[5], w[6], w[7], static_cast<unsigned long long>(tb), first[0], first[1], first[2], first[3],
                             first[4], first[5], first[6], first[7], static_cast<unsigned long long>(hash));
                }
            }
        }
    }
    static const char* trace_draw_name = std::getenv("BBHOST_TRACE_DRAW");
    if (trace_draw_name) {
        // BBHOST_TRACE_MIN_FLIP=N: trace the first matching draw from flip N on.
        static const std::uint64_t trace_min_flip = [] {
            const char* e = std::getenv("BBHOST_TRACE_MIN_FLIP");
            return e ? std::strtoull(e, nullptr, 10) : 0ull;
        }();
        static std::atomic<int> detail{0};
        // "<name>@0x<base>": only a draw whose first PS image sits at that address.
        static const std::string trace_name = [] {
            std::string t(trace_draw_name);
            const std::size_t cut = t.find_first_of("@%");
            return cut == std::string::npos ? t : t.substr(0, cut);
        }();
        // "%<n>": only draws with at least n indices (hidden UI elements draw 3 zero indices).
        static const std::uint32_t trace_min_count = [] {
            const char* pc = std::strchr(trace_draw_name, '%');
            return pc ? static_cast<std::uint32_t>(std::strtoul(pc + 1, nullptr, 10)) : 0u;
        }();
        static const std::uint64_t trace_tex = [] {
            const char* at = std::strchr(trace_draw_name, '@');
            return at ? std::strtoull(at + 1, nullptr, 0) : 0ull;
        }();
        bool tex_ok = true;
        if (trace_tex && pl.name.find(trace_name) == 0 && !pl.ps.meta().images.empty()) {
            std::uint32_t w[8] = {};
            tex_ok = resolve_resource_impl(pl.ps.meta().images[0].path, s.ps_user, 8, w) &&
                     ((static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8) == trace_tex;
        }
        if (pl.name.find(trace_name) == 0 && tex_ok && d.index_count >= trace_min_count && hle_video_flip_count() >= trace_min_flip &&
            detail.fetch_add(1) < 1) {
            // Execute all recorded work so far (dispatches earlier in this
            // submission may GPU-fill this draw's vertex/index buffers) and
            // wait, so the reads below reflect post-GPU memory, not the
            // pre-execution CPU state.
            flush_locked();
            {
                QueueGuard queue;
                vkQueueWaitIdle(g.queue);  // not the device: the presenter's queue is not ours to wait on
            }
            host_log("render: detail for %s: index_va=0x%llx type=%u count=%u", pl.name.c_str(),
                     static_cast<unsigned long long>(d.index_va), d.index_type, d.index_count);
            host_log("  stages_en=%08x prim=%u ia_multi_vgt=%08x vgt_gs_mode=%08x index_type=%u", cx[0x2D5], s.prim, cx[0x2AA], cx[0x290], d.index_type);
            host_log("  depth: control=%08x render_control=%08x clear=%08x vte=%08x vport xs=%g xo=%g ys=%g yo=%g zs=%g zo=%g -> vk %g..%g; target 0x%llx",
                     s.depth_control, s.render_control, s.depth_clear, s.vte_cntl, s.vport[0], s.vport[1], s.vport[2], s.vport[3],
                     s.vport[4], s.vport[5], s.vport[5], s.vport[5] + s.vport[4], static_cast<unsigned long long>(s.depth ? s.depth->base : 0));
            host_log("  vs user: %08x %08x %08x %08x %08x %08x %08x %08x | %08x %08x %08x %08x %08x %08x %08x %08x", s.vs_user[0], s.vs_user[1],
                     s.vs_user[2], s.vs_user[3], s.vs_user[4], s.vs_user[5], s.vs_user[6], s.vs_user[7], s.vs_user[8], s.vs_user[9], s.vs_user[10],
                     s.vs_user[11], s.vs_user[12], s.vs_user[13], s.vs_user[14], s.vs_user[15]);
            // Every user SGPR pair that points at mapped memory: 16 dwords of it
            // (EUD tables and V#s show up this way), and V#s in the user data
            // get their first 16 floats printed.
            for (int k = 0; k + 1 < 16; k += 2) {
                const std::uint64_t ptr = static_cast<std::uint64_t>(s.vs_user[k]) | (static_cast<std::uint64_t>(s.vs_user[k + 1]) << 32);
                if ((s.vs_user[k + 1] & 0xff) == 1 && hle_kernel_va_mapped(ptr, 64)) {
                    const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(ptr));
                    host_log("  vs s[%d:%d] -> 0x%llx: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x", k, k + 1,
                             static_cast<unsigned long long>(ptr), u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
                }
            }
            for (int k = 0; k + 3 < 16; k += 4) {
                const std::uint32_t* w = &s.vs_user[k];
                const std::uint64_t vb = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
                const std::uint32_t stride = (w[1] >> 16) & 0x3fff;
                if (w[2] && (w[3] >> 12) && hle_kernel_va_mapped(vb, 64)) {
                    const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vb));
                    host_log("  vs V# s[%d:%d] base=0x%llx stride=%u records=%u: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g", k, k + 3,
                             static_cast<unsigned long long>(vb), stride, w[2], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11],
                             f[12], f[13], f[14], f[15]);
                }
            }
            if (d.index_va && hle_kernel_va_mapped(d.index_va, 64)) {
                const auto* i16 = reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(d.index_va));
                const auto* i32 = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(d.index_va));
                if (d.index_type) host_log("  indices32: %u %u %u %u %u %u %u %u", i32[0], i32[1], i32[2], i32[3], i32[4], i32[5], i32[6], i32[7]);
                else host_log("  indices16: %u %u %u %u %u %u %u %u %u %u %u %u", i16[0], i16[1], i16[2], i16[3], i16[4], i16[5], i16[6], i16[7], i16[8], i16[9], i16[10], i16[11]);
            }
            const gcn::Program fp = gcn::decode(fetch_words.data(), fetch_words.size());
            for (const gcn::Inst& in : fp.insts) host_log("  fetch: %s", gcn::format(in).c_str());
            // The fetch shader's V# table (user s[2:3] by convention): base, stride, records, format.
            {
                const std::uint64_t tbl = static_cast<std::uint64_t>(s.vs_user[2]) | (static_cast<std::uint64_t>(s.vs_user[3]) << 32);
                if (hle_kernel_va_mapped(tbl, 16 * 8)) {
                    const auto* v = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tbl));
                    for (int k = 0; k < 8; ++k) {
                        const std::uint32_t* w = v + k * 4;
                        if (!w[0] && !w[1]) break;
                        const std::uint64_t vb = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
                        host_log("  fetch V#%d: %08x %08x %08x %08x base=0x%llx stride=%u records=%u dfmt=%u nfmt=%u dst_sel=%03x", k, w[0], w[1], w[2], w[3],
                                 static_cast<unsigned long long>(vb), (w[1] >> 16) & 0x3fff, w[2], (w[3] >> 15) & 0xf, (w[3] >> 12) & 7, w[3] & 0xfff);
                        // BBHOST_WATCH_FILL=1: watch this fill draw's first vertex
                        // buffer so the next frame's CPU write to it reveals the
                        // guest GFx code that computes the (zero) corners.
                        static const bool watch_fill = [] {
                            const char* e = std::getenv("BBHOST_WATCH_FILL");
                            return e && e[0] == '1';
                        }();
                        if (watch_fill && k == 0 && vb) {
                            // Watch the whole page holding this fill draw's dynamic
                            // vertex buffer; GFx rotates buffers within it, so log
                            // every writer RIP that touches the page.
                            const std::uint64_t page = vb & ~std::uint64_t(0xfff);
                            hle_watch_arm(page, page + 0x1000);
                        }
                        if (hle_kernel_va_mapped(vb, 32)) {
                            const auto* d = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(vb));
                            const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vb));
                            host_log("    data: %08x %08x %08x %08x %08x %08x %08x %08x (floats %g %g %g %g)", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], f[0], f[1], f[2], f[3]);
                            // Whole records for a small vertex buffer: a short
                            // quad or a bad corner is only visible across all of
                            // them, not in the first two vertices.
                            const std::uint32_t stride_b = (w[1] >> 16) & 0x3fff;
                            const std::uint32_t records = w[2];
                            if (stride_b && records && records <= 16 &&
                                hle_kernel_va_mapped(vb, static_cast<std::size_t>(stride_b) * records)) {
                                for (std::uint32_t rec = 0; rec < records; ++rec) {
                                    const auto* rf = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vb) + rec * stride_b);
                                    const auto* ru = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(vb) + rec * stride_b);
                                    host_log("      rec%u: %08x %08x %08x (floats %g %g %g)", rec, ru[0], ru[1],
                                             stride_b >= 12 ? ru[2] : 0u, rf[0], rf[1], stride_b >= 12 ? rf[2] : 0.0f);
                                }
                            }
                        }
                    }
                }
            }
            const gcn::Program vp2 = gcn::decode(vs_words.data(), vs_words.size());
            int n = 0;
            for (const gcn::Inst& in : vp2.insts) { if (n++ < 400) host_log("  vs: %s", gcn::format(in).c_str()); }
            if (!ps_words.empty()) {
                const gcn::Program pp = gcn::decode(ps_words.data(), ps_words.size());
                n = 0;
                for (const gcn::Inst& in : pp.insts) { if (n++ < 400) host_log("  ps: %s", gcn::format(in).c_str()); }
                host_log("  ps user: %08x %08x %08x %08x %08x %08x %08x %08x", s.ps_user[0], s.ps_user[1], s.ps_user[2], s.ps_user[3],
                         s.ps_user[4], s.ps_user[5], s.ps_user[6], s.ps_user[7]);
                for (const gcn::ImageBinding& b : pl.ps.meta().images) {
                    std::uint32_t w[8] = {};
                    if (resolve_resource_impl(b.path, s.ps_user, 8, w)) {
                        host_log("  ps image %s: %08x %08x %08x %08x %08x %08x %08x %08x", b.path.str().c_str(), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                        const std::uint64_t tb = (static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32)) << 8;
                        if (hle_kernel_va_mapped(tb, 4096)) {
                            const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tb));
                            const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(tb));
                            std::uint32_t nz = 0;
                            for (int k = 0; k < 1024; ++k) nz += u[k] != 0;
                            host_log("    memory: %08x %08x %08x %08x %08x %08x %08x %08x (floats %g %g %g %g) nonzero dwords in first 4K: %u",
                                     u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], f[0], f[1], f[2], f[3], nz);
                        }
                    }
                }
                for (const gcn::SamplerBinding& b : pl.ps.meta().samplers) {
                    std::uint32_t w[4] = {};
                    if (resolve_resource_impl(b.path, s.ps_user, 4, w)) host_log("  ps sampler %s: %08x %08x %08x %08x", b.path.str().c_str(), w[0], w[1], w[2], w[3]);
                }
            }
            // Vertex buffer table: the fetch shader's s_load base pair.
            for (const gcn::Inst& in : fp.insts) {
                if (in.enc == gcn::Enc::SMRD && in.op < 8 && in.src0 < 16) {
                    const std::uint64_t tbl = static_cast<std::uint64_t>(s.vs_user[in.src0]) | (static_cast<std::uint64_t>(s.vs_user[in.src0 + 1]) << 32);
                    const std::uint64_t at = tbl + static_cast<std::uint64_t>(in.imm) * 4;
                    if (hle_kernel_va_mapped(at, 16)) {
                        const auto* v = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(at));
                        const std::uint64_t vb = static_cast<std::uint64_t>(v[0]) | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
                        host_log("  V# at +0x%x: %08x %08x %08x %08x -> base 0x%llx stride %u records %u", in.imm * 4, v[0], v[1], v[2], v[3],
                                 static_cast<unsigned long long>(vb), (v[1] >> 16) & 0x3fff, v[2]);
                        if (hle_kernel_va_mapped(vb, 128)) {
                            const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vb));
                            const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(vb));
                            for (int r = 0; r < 4; ++r) {
                                host_log("    data+%02x: %08x %08x %08x %08x %08x %08x %08x %08x (floats %g %g %g %g %g %g %g %g)", r * 32,
                                         u[r * 8], u[r * 8 + 1], u[r * 8 + 2], u[r * 8 + 3], u[r * 8 + 4], u[r * 8 + 5], u[r * 8 + 6], u[r * 8 + 7],
                                         f[r * 8], f[r * 8 + 1], f[r * 8 + 2], f[r * 8 + 3], f[r * 8 + 4], f[r * 8 + 5], f[r * 8 + 6], f[r * 8 + 7]);
                            }
                        }
                    }
                }
            }
        }
    }
    draw_stamp.to(kRenderCostChecks);
    // Vertex input replaces the fetch shader, but a vertex shader can still read
    // the vertex table itself (per-instance records). Place the draw's table
    // when the no-fallback variant reads its slot, or when no variant says:
    // otherwise the slot keeps the register file's table, which for a draw
    // from a host-draw token belongs to another draw.
    if (gx_geometry && d.gx_render && use_vertex_input && !vertex_table && gx_geometry->vtx_ud < 15) {
        const GfxPipeline* const lean = pl.lean_variant;
        const std::uint32_t reads = lean && !lean->failed ? lean->table_reads[0] : ~0u;
        if ((reads >> gx_geometry->vtx_ud) & 1) {
            vertex_table = place_vertex_table(*gx_geometry);
            if (vertex_table) g_vertex_tables_with_input.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (g.queued + 2 >= kMaxQueued * kStageSlots) {
        // A frame with more draws than the params ring holds is submitted in
        // the middle. Where that boundary falls depends on how many draws the
        // frame has, which is exactly what removing a pass changes - so it has
        // to be on the record next to the frame that went wrong.
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 400) {
            host_log("render: mid-frame submit at draw %llu of flip %llu (%u params slots queued)",
                     static_cast<unsigned long long>(g_draw_rec_next), static_cast<unsigned long long>(hle_video_flip_count()),
                     g.queued);
        }
        submit_locked();
    }
    begin_recording_locked();
    // Descriptor sets: VS (set 0) and PS (set 1). The write lists are reused
    // from draw to draw (host_gpu_draw does not nest), and reserved before they
    // fill: writes point into infos and buffer_infos, which must not move.
    VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorBufferInfo ubi[2]{};
    const std::size_t image_writes =
        pl.vs.meta().images.size() + pl.vs.meta().samplers.size() + pl.ps.meta().images.size() + pl.ps.meta().samplers.size();
    const std::size_t buffer_writes = pl.vs.meta().buffers.size() + pl.ps.meta().buffers.size();
    static thread_local std::vector<VkWriteDescriptorSet> writes;
    static thread_local std::vector<VkDescriptorImageInfo> infos;
    static thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
    writes.clear();
    writes.reserve(2 + image_writes + buffer_writes);
    infos.clear();
    infos.reserve(2 * image_writes + 4);
    buffer_infos.clear();
    buffer_infos.reserve(buffer_writes);
    bool all_bound = true, any_buffers = false;  // every constant buffer of both stages bound
    gcn::StageParams stage_params[2]{};
    std::size_t stage_first_buffer[2] = {};
    // BBHOST_GX_SKIP_TABLES: the tables the draw's no-fallback variant reads,
    // when that variant exists (the only one that can skip any).
    std::uint32_t table_masks[2] = {~0u, ~0u};
    std::uint16_t skipped_tables[2] = {};
    bool user_skipped[2] = {false, false};  // the stage's user data was not built at all
    // The user-data dwords that variant touches at all; the rest
    // is not built (build_gx_user_data's read_mask), and user_partial says so.
    std::uint32_t read_masks[2] = {~0u, ~0u};
    bool user_partial[2] = {false, false};
    if (g_skip_tables && g.cb_lean && gx_records) {
        const GfxPipeline* lean = pl.lean_variant;
        if (lean && !lean->failed && lean->pipeline) {
            table_masks[0] = lean->table_reads[0];
            table_masks[1] = lean->table_reads[1];
            if (g_skip_user_data) {
                read_masks[0] = lean->user_reads[0];
                read_masks[1] = lean->user_reads[1];
            }
        }
    }
    draw_stamp.to(kRenderCostSetup);
    static StageSetCache set_cache[2][kSetCacheWays];  // g.mu is held
    static int set_cache_next[2] = {0, 0};
    bool stage_reused[2] = {false, false};  // this stage took an earlier draw's set, or one the set cache keeps
    const bool use_set_cache = set_cache_on() && !capture;  // captures describe buffers by their ranges

    int reused_way[2] = {-1, -1};
    std::uint32_t stage_named[2] = {0, 0};  // user-data dwords each stage built from GX
    for (int st = 0; st < 2; ++st) {
        const gcn::TranslateResult& meta = st == 0 ? pl.vs.meta() : pl.ps.meta();
        const std::size_t stage_writes = writes.size();
        gcn::StageParams params{};
        params.l1_table = g.l1.address;
        std::memcpy(params.user_sgpr, st == 0 ? s.vs_user : s.ps_user, sizeof(params.user_sgpr));
        if (st == 0 && t_tess) {
            params.lds_address = t_tess->lds_address;
            params.lds_bytes = t_tess->lds_bytes;
            std::memcpy(params.patch_cull_w, t_tess->cull_w, sizeof(params.patch_cull_w));
            params.patch_stride = t_tess->patch_stride;
            params.patch_base = t_tess->patch_base;
            params.patch_cull_near = tess_patch_cull_near();
        }
        std::uint32_t named_user = 0;  // user-data dwords built from GX below
        if (gx_stage[st] && !t_tess && g_skip_user_data && g_binding_plans) {
            // The PS4 user data this stage's program would read -
            // records in user SGPRs, tables in the ring - built only for a
            // reader. The no-fallback variant touches none (its liveness mask)
            // and every binding comes from the records, so none is built. Should
            // the draw end on the fallback variant after all, the table rebuild
            // below builds it all then.
            const GfxPipeline* lean = pl.lean_variant;
            user_skipped[st] = lean && !lean->failed && !lean->user_reads[st] && stage_plan(meta, *gx_stage[st]).all_records;
            ++(user_skipped[st] ? g_ud_skipped : g_ud_built);
        }
        if (gx_stage[st] && (st == 0 || d.gx_objects->shader[4]) && !user_skipped[st]) {
            // Rendering from GX: the user data the stage's descriptors name,
            // built from its records with tables in the host ring, and the
            // input layout's fetch shader. Other slots keep the command
            // stream's values.
            GxUserData native;
            GxUserResult built;
            {
                RenderCostTimer timer(kRenderCostUserData);
                built = build_gx_user_data(*gx_stage[st], native, table_masks[st], read_masks[st]);
            }
            user_partial[st] = native.partial;
            if (native.partial) ++g_ud_partial;
            if (built == kUserBuilt) {
                skipped_tables[st] = native.skipped;
                bump(g_gx_tables_placed, static_cast<std::uint64_t>(native.nranges));
                if (d.gx_token) {
                    bump(g_token_tables_placed, static_cast<std::uint64_t>(native.nranges));
                    for (int r = 0; r < native.nranges; ++r) {
                        if (native.ranges[r].ud < 16) bump(g_token_tables_by_slot[native.ranges[r].ud]);
                    }
                }
                for (int k = 0; k < 16; ++k) {
                    if ((native.set >> k) & 1) params.user_sgpr[k] = native.user[k];
                }
                named_user = native.set;
                const int fs = vs_cp->fetch_sgpr;
                if (st == 0 && fs >= 0 && fs + 1 < 16 && gx_geometry && gx_geometry->fetch_va) {
                    params.user_sgpr[fs] = static_cast<std::uint32_t>(gx_geometry->fetch_va);
                    params.user_sgpr[fs + 1] = static_cast<std::uint32_t>(gx_geometry->fetch_va >> 32);
                    named_user |= 3u << fs;
                }
                bump(g_gx_user_stages);
            } else {
                bump(g_gx_user_fallbacks);
            }
        }
        if (st == 0 && vertex_table) {
            // The fetch shader reads the draw's vertex table from the host ring.
            // Set before the buffers resolve: a binding over this slot would
            // otherwise take the register file's table, which for a draw from
            // a host-draw token belongs to another draw.
            params.user_sgpr[gx_geometry->vtx_ud] = static_cast<std::uint32_t>(vertex_table);
            params.user_sgpr[gx_geometry->vtx_ud + 1] = static_cast<std::uint32_t>(vertex_table >> 32);
            named_user |= 3u << gx_geometry->vtx_ud;
        }
        stage_named[st] = named_user;
        if (d.gx_token && g_token_user_data_check) {
            std::uint32_t unnamed_set = 0;
            for (int k = 0; k < 16; ++k) {
                if (!((named_user >> k) & 1) && params.user_sgpr[k]) unnamed_set |= 1u << k;
            }
            if (unnamed_set) {
                bump(g_token_unnamed_set_stages, 1);
                for (int k = 0; k < 16; ++k) {
                    if ((unnamed_set >> k) & 1) bump(g_token_unnamed_set_by_slot[k]);
                }
            }
        }
        if (d.gx_token && g_token_user_data_check && pl.lean_variant && !pl.lean_variant->failed) {
            // The register file's user data belongs to the last packet draw.
            const std::uint32_t unnamed_reads = pl.lean_variant->table_reads[st] & ~named_user & 0xffff;
            std::uint32_t stale = 0;
            for (int k = 0; k < 16; ++k) {
                if (((unnamed_reads >> k) & 1) && params.user_sgpr[k]) stale |= 1u << k;
            }
            bump(g_token_stages_checked, 1);
            if (stale) {
                bump(g_token_stale_stages, 1);
                static std::set<std::string> logged;
                if (logged.size() < 8 && logged.insert(pl.name + (st ? " PS" : " VS")).second) {
                    std::string values, descs;
                    for (int k = 0; k < 16; ++k) {
                        if (!((stale >> k) & 1)) continue;
                        char buf[24];
                        std::snprintf(buf, sizeof(buf), " s[%d]=%08x", k, params.user_sgpr[k]);
                        values += buf;
                    }
                    for (std::uint32_t i = 0; i < gx_stage[st]->ndesc && i < 24; ++i) {
                        const std::uint32_t dw = gx_stage[st]->desc[i];
                        char buf[24];
                        std::snprintf(buf, sizeof(buf), " %x@s[%u]", dw & 0xff, (dw >> 8) & 0xff);
                        descs += buf;
                    }
                    host_log("render: token draw %s %s reads unnamed user data:%s; its descriptors (type@slot):%s; vertex table slot %u, "
                             "vertex input %d", pl.name.c_str(), st ? "PS" : "VS", values.c_str(), descs.c_str(),
                             gx_geometry ? static_cast<unsigned>(gx_geometry->vtx_ud) : 0xffu, use_vertex_input ? 1 : 0);
                }
                for (int k = 0; k < 16; ++k) {
                    if ((stale >> k) & 1) bump(g_token_stale_by_slot[k]);
                }
            }
        }
        draw_stamp.to(kRenderCostStageUser);
        const std::size_t first_buffer = buffer_infos.size();
        if (st == 0) {
            glitch_watch_draw_locked(pl.name);
            if (glitch_watching_draw()) glitch_watch_note_locked("  from: %s\n", t_draw_origin);
        }
        if (st == 0 && t_tess && t_tess->hull) {
            // The game's own hull: its user data in the vertex-formats slot
            // (the evaluation stage keeps the usual one), and the domain
            // shader's tessellation constants where its descriptors want them.
            std::memcpy(params.vertex_formats, t_tess->hs_user, sizeof(params.vertex_formats));
            if (gx_stage[0]) put_tess_constants(*gx_stage[0], t_tess->tess_vsharp, params.user_sgpr);
        }
        resolve_stage_buffers(meta, params.user_sgpr, params, buffer_infos, gx_stage[st]);
        if (!meta.buffers.empty()) {
            any_buffers = true;
            all_bound = all_bound && params.cb_valid == (1u << meta.buffers.size()) - 1;
        }
        if (st == 0 && use_vertex_input) {
            // Step 6c: the conversions a vertex shader built with vertex_formats_from_params reads.
            for (const gcn::VertexElement& el : vertex_input.elements) {
                if (el.location < 16) params.vertex_formats[el.location] = gcn::vertex_format_descriptor(el.w3);
            }
        }
        if (st == 0 && pl.tess_attrs) {
            // The LS's own user data. Two stages of one pipeline
            // read this set and theirs differ, so the evaluation stage keeps
            // the usual place - it is the one whose resources were resolved
            // above - and the LS takes the slot a patch draw's vertex formats
            // would have been in.
            //
            // From its GX records where there are any, since a draw
            // that took a token wrote no registers for the command stream to
            // have them in. The LS binds nothing, so only its named dwords and
            // its fetch shader are wanted; everything else keeps what the
            // command stream had.
            std::memcpy(params.vertex_formats, s.ls_user, sizeof(params.vertex_formats));
            // BBHOST_GLITCH_WATCH: what the LS fetches, through the page table
            // rather than a binding - the V#s in the table its fetch shader
            // loads from s[2:3], as a vertex shader's does.
            if (glitch_watching_draw()) {
                const std::uint64_t tbl = static_cast<std::uint64_t>(s.ls_user[2]) | (static_cast<std::uint64_t>(s.ls_user[3] & 0xff) << 32);
                if (tbl && hle_kernel_va_mapped(tbl, 16 * 8)) {
                    const auto* v = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tbl));
                    for (int k = 0; k < 8; ++k) {
                        const std::uint32_t* w = v + k * 4;
                        const std::uint64_t vb = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
                        const std::uint32_t stride = (w[1] >> 16) & 0x3fff;
                        if (!vb || !w[2]) continue;
                        glitch_watch_read_locked(vb, static_cast<std::uint64_t>(w[2]) * (stride ? stride : 1), 2);
                    }
                }
            }
            // Straight off the objects: these draws still come through packets,
            // so `gx_records` (which wants gx_render) is null for them, and
            // the point is to check what GX gives before anything draws from
            // it. Sampled draws only, which is enough to count.
            const GxStageRecords* ls_rec =
                d.gx_objects && d.gx_objects->records && d.gx_objects->shader[1] ? &d.gx_objects->records->stage[kGxRecordLs] : nullptr;
            if (ls_rec) {
                GxUserData ls_native;
                if (build_gx_user_data(*ls_rec, ls_native, ~0u) == kUserBuilt) {
                    for (int k = 0; k < 16; ++k) {
                        if ((ls_native.set >> k) & 1) params.vertex_formats[k] = ls_native.user[k];
                    }
                    // The LS has no descriptors at all: its user data is two
                    // pointers, and they go in different slots. The fetch
                    // shader sits where the program reads it, and the vertex
                    // table - eight records the commit builds into the command
                    // stream, which is why the register changes every draw -
                    // at the slot GX names, from the ring instead.
                    const CachedProgram* ls_cp = program_at(s.ls_va);
                    const int ls_fs = ls_cp ? ls_cp->fetch_sgpr : -1;
                    if (ls_fs >= 0 && ls_fs + 1 < 16 && d.gx_objects->fetch_va) {
                        params.vertex_formats[ls_fs] = static_cast<std::uint32_t>(d.gx_objects->fetch_va);
                        params.vertex_formats[ls_fs + 1] = static_cast<std::uint32_t>(d.gx_objects->fetch_va >> 32);
                    }
                    if (d.gx_objects->vtx_ud < 15 && d.gx_objects->vtx_valid) {
                        if (const std::uint64_t table = place_vertex_table(*d.gx_objects)) {
                            params.vertex_formats[d.gx_objects->vtx_ud] = static_cast<std::uint32_t>(table);
                            params.vertex_formats[d.gx_objects->vtx_ud + 1] = static_cast<std::uint32_t>(table >> 32);
                            g_tess_ls_table.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    // While these draws still come through packets the command
                    // stream's copy is right, so it is what the GX-built one
                    // is checked against - per slot, so a disagreement names
                    // itself rather than just counting.
                    // Per slot against the command stream, which is still
                    // right while these draws write it. The vertex table's two
                    // are ours on purpose - the game's points into its command
                    // stream, ours into the ring - so they are counted apart
                    // rather than called a disagreement.
                    const int tbl = d.gx_objects->vtx_ud < 15 ? d.gx_objects->vtx_ud : -1;
                    for (int k = 0; k < 16; ++k) {
                        if (k == tbl || k == tbl + 1) continue;
                        if (!params.vertex_formats[k] && !s.ls_user[k]) continue;
                        g_tess_ls_slot[k].fetch_add(1, std::memory_order_relaxed);
                        if (params.vertex_formats[k] != s.ls_user[k]) {
                            g_tess_ls_slot_bad[k].fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    {
                        // What the LS's user data is made of: its descriptors,
                        // what GX named, and what the command stream holds.
                        static std::atomic<int> logs{0};
                        if (logs.fetch_add(1, std::memory_order_relaxed) < 4) {
                            std::string desc, reg, got;
                            for (std::uint32_t i = 0; i < ls_rec->ndesc && i < 16; ++i) {
                                char b[24];
                                std::snprintf(b, sizeof(b), " %08x", ls_rec->desc[i]);
                                desc += b;
                            }
                            for (int k = 0; k < 16; ++k) {
                                char b[24];
                                std::snprintf(b, sizeof(b), " %08x", s.ls_user[k]);
                                reg += b;
                                std::snprintf(b, sizeof(b), " %08x", params.vertex_formats[k]);
                                got += b;
                            }
                            host_log("render: LS user data: %u descriptors%s; named %04x", ls_rec->ndesc, desc.c_str(), ls_native.set);
                            host_log("render:   registers%s", reg.c_str());
                            host_log("render:   built    %s", got.c_str());
                            host_log("render:   fetch 0x%llx at s[%u], vtx_valid %04x",
                                     static_cast<unsigned long long>(d.gx_objects->fetch_va), d.gx_objects->vtx_ud,
                                     d.gx_objects->vtx_valid);
                        }
                    }
                    g_tess_ls_user_gx.fetch_add(1, std::memory_order_relaxed);
                } else {
                    g_tess_ls_user_fallback.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        // Bindless, the stage's views and samplers are slots in
        // the global set, named in its params block before it is placed.
        static thread_local std::vector<VkImageView> views;
        if (meta.bindless) {
            stage_final_views(meta, stage_images[st], pl.name.c_str(), views);
            for (std::size_t k = 0; k < meta.images.size() && k < 16; ++k) {
                params.image_index[k] = bindless_view_slot_locked(views[k], meta.images[k].storage);
            }
            for (std::size_t k = 0; k < meta.samplers.size() && k < 16; ++k) {
                const VkSampler smp = stage_images[st].samplers[k] ? stage_images[st].samplers[k] : g.dummy_sampler;
                params.sampler_index[k] = bindless_sampler_slot_locked(smp);
            }
        }
        if (use_set_cache) {
            // A set with the same views, samplers and buffers made before is
            // bound as it is, with this draw's params block as its offset.
            keep_params(pl.name, params, st);
        alloc_params_slot_locked(params, ubi[st]);
            stage_params[st] = params;
            stage_first_buffer[st] = first_buffer;
            draw_stamp.to(kRenderCostBuffers);
            const VkDescriptorSetLayout layout = pl.set_layouts[st] ? pl.set_layouts[st] : g.gfx_set_layout;
            static thread_local std::vector<std::uint64_t> material;
            if (!meta.bindless) stage_final_views(meta, stage_images[st], pl.name.c_str(), views);
            const auto handle = [](auto h) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h)); };
            material.clear();
            material.reserve(1 + 2 * (meta.images.size() + meta.samplers.size()) + 4 * meta.buffers.size());
            material.push_back(handle(layout));
            for (std::size_t k = 0; k < meta.images.size() && !meta.bindless; ++k) {
                material.push_back((static_cast<std::uint64_t>(meta.images[k].binding) << 1) | (meta.images[k].storage ? 1u : 0u));
                material.push_back(handle(views[k]));
            }
            for (std::size_t k = 0; k < meta.samplers.size() && !meta.bindless; ++k) {
                const VkSampler smp = stage_images[st].samplers[k] ? stage_images[st].samplers[k] : g.dummy_sampler;
                material.push_back(meta.samplers[k].binding);
                material.push_back(handle(smp));
            }
            for (std::size_t i = 0; i < meta.buffers.size() && !cb_push_on(); ++i) {
                const VkDescriptorBufferInfo& bi = buffer_infos[first_buffer + i];
                material.push_back(meta.buffers[i].binding);
                material.push_back(handle(bi.buffer));
                material.push_back(bi.offset);
                material.push_back(bi.range);
            }
            const std::uint64_t key = fnv1a(material.data(), material.size() * sizeof(std::uint64_t));
            if (const VkDescriptorSet hit = set_cache_find(key, material)) {
                sets[st] = hit;
                stage_reused[st] = true;  // a table rebuild takes a set of its own
                draw_stamp.to(kRenderCostSets);
                draw_stamp.to(kRenderCostImages);
                continue;
            }
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VkDescriptorSet set = set_cache_alloc(layout, pool);
            const bool cached = set != VK_NULL_HANDLE;
            if (!cached) set = alloc_set_locked(pl.set_layouts[st] ? pl.set_layouts[st] : g.gfx_set_layout);
            if (!set) {
                draw_failed(kFailSet);
                return false;
            }
            sets[st] = set;
            bump(g_set_built);
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set;
            w.dstBinding = gcn::kBindingParams;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            w.pBufferInfo = &g_params_desc;
            writes.push_back(w);
            for (std::size_t i = 0; i < meta.buffers.size() && !cb_push_on(); ++i) {
                VkWriteDescriptorSet bw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                bw.dstSet = set;
                bw.dstBinding = meta.buffers[i].binding;
                bw.descriptorCount = 1;
                bw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bw.pBufferInfo = &buffer_infos[first_buffer + i];
                writes.push_back(bw);
            }
            draw_stamp.to(kRenderCostSets);
            if (!meta.bindless) stage_image_writes(set, meta, stage_images[st], views, writes, infos);
            draw_stamp.to(kRenderCostImages);
            if (cached) {
                static const std::vector<VkImageView> no_views;
                set_cache_insert(key, material, set, pool, meta.bindless ? no_views : views);
                stage_reused[st] = true;
            }
            continue;
        }
        // Everything the set is built from is resolved now, so the previous
        // draw's set for this stage can be taken as it stands.
        int hit_way = -1;
        if (g_set_reuse && !set_cache_on()) {
            // Most recently written first: the previous draw's set is the
            // likeliest match by far, and a scan that found it last paid for
            // every other way's compare on the way there.
            for (int k = 1; k <= kSetCacheWays; ++k) {
                const int way = (set_cache_next[st] - k + kSetCacheWays) % kSetCacheWays;
                if (stage_set_matches(set_cache[st][way], pl.set_layouts[st], &meta, params, buffer_infos.data() + first_buffer,
                                      meta.buffers.size(), stage_images[st])) {
                    hit_way = way;
                    break;
                }
            }
        }
        if (hit_way >= 0) {
            sets[st] = set_cache[st][hit_way].set;
            ubi[st] = set_cache[st][hit_way].ubi;
            stage_reused[st] = true;
            reused_way[st] = hit_way;
            bump(g_set_reused);
            stage_params[st] = params;
            stage_first_buffer[st] = first_buffer;
            draw_stamp.to(kRenderCostBuffers);
            draw_stamp.to(kRenderCostSets);
            draw_stamp.to(kRenderCostImages);
            continue;
        }
        keep_params(pl.name, params, st);
        alloc_params_slot_locked(params, ubi[st]);
        stage_params[st] = params;
        stage_first_buffer[st] = first_buffer;
        draw_stamp.to(kRenderCostBuffers);
        sets[st] = alloc_set_locked(pl.set_layouts[st] ? pl.set_layouts[st] : g.gfx_set_layout);
        bump(g_set_built);
        if (!sets[st]) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 4) {
                host_log("render: descriptor set allocation failed (%u params slots queued in this command buffer)", g.queued);
            }
            draw_failed(kFailSetAlloc);
            return false;
        }
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = sets[st];
        w.dstBinding = gcn::kBindingParams;
        w.descriptorCount = 1;
        w.descriptorType = params_descriptor_type();
        w.pBufferInfo = set_cache_on() ? &g_params_desc : &ubi[st];
        writes.push_back(w);
        for (std::size_t i = 0; i < meta.buffers.size() && !cb_push_on(); ++i) {
            VkWriteDescriptorSet bw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            bw.dstSet = sets[st];
            bw.dstBinding = meta.buffers[i].binding;
            bw.descriptorCount = 1;
            bw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bw.pBufferInfo = &buffer_infos[first_buffer + i];
            writes.push_back(bw);
        }
        draw_stamp.to(kRenderCostSets);
        if (!meta.bindless) bind_stage_images(sets[st], meta, stage_images[st], writes, infos, pl.name.c_str());
        draw_stamp.to(kRenderCostImages);
        if (g_set_reuse && !set_cache_on()) {
            stage_set_remember(set_cache[st][set_cache_next[st]], pl.set_layouts[st], &meta, params,
                               buffer_infos.data() + first_buffer, meta.buffers.size(), stage_images[st], sets[st], ubi[st]);
            set_cache_next[st] = (set_cache_next[st] + 1) % kSetCacheWays;
        }
        if (g_render_cost_enabled) {
            const bool set_repeated = note_stage_set(st, writes.data() + stage_writes, writes.size() - stage_writes);
            note_stage_params(st, stage_params[st], set_repeated);
            draw_stamp.skip();
        }
    }
    // From here the draw's descriptor writes and commands go to the recorder
    // thread (recorder.cpp), except while something that must see them in
    // place is on: NVIDIA's checkpoints, GPU profiling, a draw capture. AMD's
    // markers ride in the packet (DrawCmds::marker): a start after a lost
    // device recorded every draw in place, draining the stream at each one -
    // 2.2 million drains a 300-flip window against ~360, on the Radeon 8060S
    // at 60 fps (2026-10-08).
    cmds.start(!capture && !g.profile && !(g.has_checkpoints && !g.cmd_buffer_marker));
    if (set_cache_on()) {
        cmds.take_sets(writes, infos, buffer_infos, &g_params_desc, 1);
    } else {
        cmds.take_sets(writes, infos, buffer_infos, ubi, 2);
    }
    draw_stamp.to(kRenderCostUpdate);

    // Every constant buffer bound: draw with the variant translated without the
    // page-table fallback, where the driver can drop the walks and the V#
    // loads that only fed them. Same bindings, so the sets written above fit.
    GfxPipeline* bind_pl = &pl;
    if (g.cb_lean && any_buffers && all_bound) {
        if (!pl.lean_variant && !pl.lean_building) {
            const std::uint64_t lean_key = fnv1a("cb-lean", 7, key);
            // A worker builds the variant only while a created fallback can draw
            // meanwhile; otherwise (BBHOST_LEAN_FIRST) it is built here, and the
            // fallback is never created unless a draw lacks a binding.
            if (g_async_pipelines && pl.pipeline && !g_dump_spirv && !g_token_user_data_check) {
                queue_lean_variant(pl, lean_key, s, vs_words, fetch_words, ps_words, vs_name, ps_name, vs_dims, ps_dims, vs_sampler_modes,
                                   ps_sampler_modes, vertex_plan);
            } else {
                pl.lean_variant = &gfx_pipeline(s, lean_key, vs_words, fetch_words, ps_words, vs_name, ps_name, vs_dims, ps_dims,
                                                vs_sampler_modes, ps_sampler_modes, true, vertex_plan);
            }
        }
        if (pl.lean_variant) {
            GfxPipeline& lean = *pl.lean_variant;
            if (!lean.failed && lean.layout == pl.layout) bind_pl = &lean;  // the sets fit its layout
        }
    }
    if (skipped_tables[0] | skipped_tables[1]) {
        bump(g_skip_draws, 1);
        bump(g_tables_skipped, static_cast<std::uint64_t>(__builtin_popcount(skipped_tables[0]) + __builtin_popcount(skipped_tables[1])));
    }
    if ((skipped_tables[0] | skipped_tables[1] | user_skipped[0] | user_skipped[1] | user_partial[0] | user_partial[1]) && !bind_pl->lean) {
        // The fallback variant walks every table: build the skipped ones, and
        // rewrite the stage's params and storage buffers in place (the draw's
        // sets are not bound yet).
        g_skip_rebuilds.fetch_add(1, std::memory_order_relaxed);
        std::vector<VkWriteDescriptorSet> rewrites;
        std::vector<VkDescriptorImageInfo> rewrite_infos;
        rewrite_infos.reserve(2 * image_writes + 4);
        for (int st = 0; st < 2; ++st) {
            GxUserData native;
            if (!(skipped_tables[st] || user_skipped[st] || user_partial[st]) || build_gx_user_data(*gx_stage[st], native) != kUserBuilt) continue;
            gcn::StageParams& params = stage_params[st];
            for (int k = 0; k < 16; ++k) {
                if ((native.set >> k) & 1) params.user_sgpr[k] = native.user[k];
            }
            const int fs = vs_cp->fetch_sgpr;
            if (st == 0 && fs >= 0 && fs + 1 < 16 && gx_geometry && gx_geometry->fetch_va) {
                params.user_sgpr[fs] = static_cast<std::uint32_t>(gx_geometry->fetch_va);
                params.user_sgpr[fs + 1] = static_cast<std::uint32_t>(gx_geometry->fetch_va >> 32);
            }
            if (st == 0 && vertex_table) {
                // As in the first pass: the vertex table's slot before the buffers resolve.
                params.user_sgpr[gx_geometry->vtx_ud] = static_cast<std::uint32_t>(vertex_table);
                params.user_sgpr[gx_geometry->vtx_ud + 1] = static_cast<std::uint32_t>(vertex_table >> 32);
            }
            if (st == 0 && t_tess && t_tess->hull) put_tess_constants(*gx_stage[0], t_tess->tess_vsharp, params.user_sgpr);
            const gcn::TranslateResult& meta = st == 0 ? pl.vs.meta() : pl.ps.meta();
            std::vector<VkDescriptorBufferInfo> again;
            params.cb_valid = 0;
            resolve_stage_buffers(meta, params.user_sgpr, params, again, gx_stage[st]);
            std::copy(again.begin(), again.end(), buffer_infos.begin() + static_cast<std::ptrdiff_t>(stage_first_buffer[st]));
            if (stage_reused[st] && !cb_push_on()) {
                // This stage took the previous draw's set and params slot, and
                // that draw is already recorded: rewriting either in place
                // would change it. Take a fresh set and slot and write the
                // whole stage into them instead. (With constant buffers pushed,
                // the set holds none, and the params slot is this draw's own.)
                const VkDescriptorSet fresh = alloc_set_locked(pl.set_layouts[st] ? pl.set_layouts[st] : g.gfx_set_layout);
                if (!fresh) {
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1) < 4) {
                        host_log("render: no descriptor set for a table rebuild; leaving the shared set alone");
                    }
                    continue;  // better an unrebuilt table than a recorded draw changed under it
                }
                sets[st] = fresh;
                stage_reused[st] = false;
                if (reused_way[st] >= 0) set_cache[st][reused_way[st]].valid = false;
                g_set_reuse_undone.fetch_add(1, std::memory_order_relaxed);
                keep_params(pl.name, params, st);
        alloc_params_slot_locked(params, ubi[st]);
                VkWriteDescriptorSet pw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                pw.dstSet = fresh;
                pw.dstBinding = gcn::kBindingParams;
                pw.descriptorCount = 1;
                pw.descriptorType = params_descriptor_type();
                pw.pBufferInfo = set_cache_on() ? &g_params_desc : &ubi[st];
                rewrites.push_back(pw);
                if (!meta.bindless) bind_stage_images(fresh, meta, stage_images[st], rewrites, rewrite_infos, pl.name.c_str());
            }
            rewrite_params_slot_locked(params, ubi[st]);
            for (std::size_t i = 0; i < meta.buffers.size() && !cb_push_on(); ++i) {
                VkWriteDescriptorSet bw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                bw.dstSet = sets[st];
                bw.dstBinding = meta.buffers[i].binding;
                bw.descriptorCount = 1;
                bw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bw.pBufferInfo = &buffer_infos[stage_first_buffer[st] + i];
                rewrites.push_back(bw);
            }
        }
        cmds.update_sets(rewrites.data(), rewrites.size());
        // A binding that resolved through a skipped table may bind now: then
        // the no-fallback variant, whose masks were used, serves the draw.
        bool bound = true;
        for (int st = 0; st < 2; ++st) {
            const gcn::TranslateResult& meta = st == 0 ? pl.vs.meta() : pl.ps.meta();
            if (!meta.buffers.empty()) bound = bound && stage_params[st].cb_valid == (1u << meta.buffers.size()) - 1;
        }
        GfxPipeline* const lean = pl.lean_variant;
        if (bound && any_buffers && g.cb_lean && lean && !lean->failed && lean->pipeline && lean->layout == pl.layout) {
            bind_pl = lean;
            g_skip_rebuilds_lean.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ++bind_pl->draws;
    if (gx_records) ++bind_pl->gx_draws;
    if (d.gx_token) {
        ++bind_pl->token_draws;
        // Against the variant that draws, fallback variants included: unnamed
        // user-data slots it reads whose register value is not 0.
        for (int st = 0; g_token_user_data_check && st < 2; ++st) {
            const std::uint32_t reads = bind_pl->table_reads[st] & ~stage_named[st] & 0xffff;
            std::uint32_t stale = 0;
            for (int k = 0; k < 16; ++k) {
                if (((reads >> k) & 1) && stage_params[st].user_sgpr[k]) stale |= 1u << k;
            }
            bump(g_token_bound_checked, 1);
            if (stale) {
                bump(g_token_bound_stale, 1);
                if (!bind_pl->lean) bump(g_token_bound_stale_fallback, 1);
            }
        }
    }
    CapturePtr capture_session;
    if (capture) {
        CaptureDraw cd;
        cd.pipeline = pl.name;
        cd.lean = bind_pl != &pl;
        cd.flip = hle_video_flip_count();
        cd.draw_index = g.draws.load();
        cd.source = d.gx_token ? "token" : "packets";
        for (int st = 0; st < 2; ++st) {
            CaptureStage& cs = cd.stages[st];
            // Bindings as the sets were written, modules as bound: the lean
            // variant keeps the same bindings.
            cs.meta = st == 0 ? &pl.vs.meta() : &pl.ps.meta();
            const std::vector<std::uint32_t>& spirv = st == 0 ? bind_pl->vs.meta().spirv : bind_pl->ps.meta().spirv;
            cs.spirv = spirv.empty() ? nullptr : &spirv;
            cs.gcn_words = st == 0 ? &vs_words : &ps_words;
            cs.params = stage_params[st];
            cs.images = &stage_images[st];
            const auto first = buffer_infos.begin() + static_cast<std::ptrdiff_t>(stage_first_buffer[st]);
            cs.buffers.assign(first, first + static_cast<std::ptrdiff_t>(cs.meta->buffers.size()));
        }
        cd.gs_spirv = bind_pl->gs.module ? &bind_pl->gs.meta().spirv : nullptr;
        cd.fetch_words = fetch_words.empty() ? nullptr : &fetch_words;
        if (use_vertex_input) {
            cd.vertex_input = true;
            cd.vinput.elements = vertex_input.elements;
            cd.vinput.bindings = vertex_input.bindings;
            cd.vinput.attributes = vertex_input.attributes;
            cd.vinput.binding_va.assign(vertex_input.va, vertex_input.va + vertex_input.bindings.size());
        }
        cd.fixed = fixed_state(s);
        for (int t = 0; t < 8; ++t) {
            cd.color[t] = s.color[t];
            cd.color_layer[t] = s.color_layer[t];
        }
        cd.depth = s.depth;
        cd.depth_layer = s.depth_layer;
        cd.count = d.index_count;
        cd.instances = d.instance_count;
        cd.base_vertex = in.base_vertex;
        cd.index_va = d.index_va;
        cd.index_type = d.index_type;
        cd.indirect_va = d.indirect_va;
        cd.registers = draw_inputs_json(in);
        cd.translate = json::Value::make_object();
        cd.translate.set("cb_ssbo", g.cb_ssbo);
        cd.translate.set("cb_no_fallback", cd.lean);
        cd.translate.set("exec_known", g.exec_known);
        // What the bound pixel shader was translated with: draws key early tests
        // only for programs that can discard (ps_early_tests_matter).
        cd.translate.set("early_fragment_tests", spirv_has_early_fragment_tests(bind_pl->ps.meta().spirv));
        cd.translate.set("debug_ps", g_debug_ps_color);
        cd.translate.set("rect_geometry_shader", s.rect);
        cd.translate.set("ps_lifted", bind_pl->ps.lifted);
        cd.translate.set("vs_lifted", bind_pl->vs.lifted);
        cd.translate.set("vertex_formats_from_params", bind_pl->vs_formats_from_params);
        cd.translate.set("link_by_register", true);  // gcnlift: identity output links, pixel-shader inputs at SPI_PS_INPUT_CNTL's registers
        cd.translate.set("vertex_input", use_vertex_input);
        cd.translate.set("vs_out_cntl", json::hex(s.vs_out_cntl));  // with the vertex_input elements, what gcnlift needs for the VS
        cd.translate.set("vs_invariant", g_vs_invariant);
        capture_session = capture_begin_locked(std::move(cd));
    }
    draw_stamp.to(kRenderCostVariant);
    // Indices or indirect arguments across a copy-back's edge: its copies go
    // in place before the pass, which a copy cannot be recorded inside.
    if (copy_versions_pending_locked() &&
        ((d.index_va && copy_version_lookup_locked(d.index_va, static_cast<std::uint64_t>(d.index_count) * (d.index_type == 1 ? 4 : 2),
                                                   nullptr) == kCopyVersionStraddles) ||
         (d.indirect_va && copy_version_lookup_locked(d.indirect_va, 20, nullptr) == kCopyVersionStraddles))) {
        copy_versions_put_in_place_locked(kCvStraddle);
    }
    // Lazy pass barriers: what this draw samples against the targets written
    // since the last barrier, and noted for the passes after it.
    bool sample_hazard = false;
    if (g_lazy_barriers) {
        for (int st = 0; st < 2; ++st) {
            for (const StageImages::Image& im : stage_images[st].images) {
                if (!im.resolved) continue;
                const std::uint64_t base = tsharp_base(im.w);
                if (g_hazards.owed && g_hazards.written.count(base)) sample_hazard = true;
                g_hazards.sampled.insert(base);
            }
        }
    }
    g_pass_profile_name = &bind_pl->profile_name;
    begin_pass(s, pass_key(s), sample_hazard);
    draw_stamp.to(kRenderCostPass);
    // Drivers finalise a pipeline's ISA on first use, so a pathological
    // translated shader shows up as a hang in vkCmdBindPipeline rather than in
    // pipeline creation. Name it before binding so a hung log says which.
    if (!bind_pl->bound_once) {
        bind_pl->bound_once = true;
        const std::string bind_note = bind_pl == &pl ? std::string()
                                                     : std::string(", no fallback") + (bind_pl->vs.lifted ? ", lifted VS" : "") +
                                                           (bind_pl->ps.lifted ? ", lifted PS" : "");
        host_log("render: first bind of pipeline %s (storage-buffer constants: VS %zu, PS %zu%s)", pl.name.c_str(),
                 pl.vs.meta().buffers.size(), pl.ps.meta().buffers.size(), bind_note.c_str());
        if (pl.vs.meta().unnormalized_samples || pl.ps.meta().unnormalized_samples) {
            host_log("render: %s unnormalized image_sample (texel coords / QuerySize) vs=%u ps=%u", pl.name.c_str(),
                     pl.vs.meta().unnormalized_samples, pl.ps.meta().unnormalized_samples);
        }
        if (!pl.vs.meta().store_buffers.empty() || pl.vs.meta().untraced_stores || !pl.ps.meta().store_buffers.empty() ||
            pl.ps.meta().untraced_stores) {
            host_log("render: %s stores to buffers: VS %zu (+%u untraced), PS %zu (+%u untraced)", pl.name.c_str(),
                     pl.vs.meta().store_buffers.size(), pl.vs.meta().untraced_stores, pl.ps.meta().store_buffers.size(),
                     pl.ps.meta().untraced_stores);
        }
    }
    if (g_recorded.serial != g.record_serial) {
        g_recorded = RecordedState{};
        g_recorded.serial = g.record_serial;
    }
    // BBHOST_LAYOUT_FROM_LEAN: a draw that reads constant buffers through the
    // page table needs the fallback's own translation. Without constant
    // buffers the two variants translate alike, and its stages serve.
    if (!bind_pl->pipeline && bind_pl->layout_only && !bind_pl->failed && any_buffers) {
        const gcn::TranslateResult vs_was = bind_pl->vs.meta(), ps_was = bind_pl->ps.meta();
        build_gfx_pipeline(*bind_pl, s, vs_words, fetch_words, ps_words, vs_name, ps_name, vs_dims, ps_dims, vs_sampler_modes,
                           ps_sampler_modes, false, vertex_plan, false);
        ++g_binding_plan_gen;  // its translations were replaced in place, and binding plans are keyed by them
        g_fallbacks_translated.fetch_add(1, std::memory_order_relaxed);
        if (!bind_pl->failed && (!same_bindings(vs_was, bind_pl->vs.meta()) || !same_bindings(ps_was, bind_pl->ps.meta()))) {
            // This draw's sets were written for the other translation's
            // bindings; the draws after it write them for these.
            if (g_fallback_binding_mismatch.fetch_add(1, std::memory_order_relaxed) < 8) {
                host_log("render: pipeline %s: its fallback binds differently from its no-fallback stages; one draw skipped",
                         bind_pl->name.c_str());
            }
            draw_failed(kFailFallbackBinding);
            return false;
        }
    }
    // BBHOST_LEAN_FIRST: the fallback pipeline is created by the first draw that binds it.
    if (!bind_pl->pipeline && (bind_pl->failed || !create_gfx_pipeline(*bind_pl, s, vertex_plan, fetch_words.size()))) {
        draw_failed(kFailPipelineCreate);
        return false;
    }
    if (bind_pl->relink_pending) queue_library_relink(*bind_pl, stage_params);
    // A relink specialized with its first draw's indexed loads' words 3 serves
    // the draws that have the same; the rest take the fast-linked pipeline.
    VkPipeline bind_vk = bind_pl->pipeline;
    if (bind_pl->generic) {
        bump(g_spec_w3_draws);
        bool same = true;
        for (int st = 0; st < 2 && same; ++st) {
            for (std::uint32_t m = bind_pl->spec_w3_mask[st]; m && same; m &= m - 1) {
                const int i = __builtin_ctz(m);
                same = stage_params[st].cb_w3[i] == bind_pl->spec_w3[st][i] &&
                       (!bind_pl->spec_stride[st][i] || stage_params[st].cb_stride[i] == bind_pl->spec_stride[st][i]);
            }
        }
        if (!same) {
            bind_vk = bind_pl->generic;
            bump(g_spec_w3_misses);
        }
    }
    if (g_recorded.pipeline != bind_vk) {
        cmds.bind_pipeline(bind_vk);
        g_recorded.pipeline = bind_vk;
        // A pipeline with this state static (the host tessellator's are built
        // whole) leaves the dynamic values invalid: the next library draw sets
        // them all again, not only what changed.
        if (!bind_pl->library) g_recorded.library_set = false;
    }
    if (bind_pl->library) {  // BBHOST_PIPELINE_LIBRARY: rasterizer, depth and stencil state per draw
        GfxFixedState d;
        fixed_raster_depth(s, d);
        DrawLibraryState now;
        std::memset(&now, 0, sizeof(now));  // padding compares too
        now.cull = d.cull_mode;
        now.front = d.front_face;
        now.depth_test = d.depth_test;
        now.depth_write = d.depth_write;
        now.bounds_test = d.depth_bounds_test;
        now.stencil_test = d.stencil_test;
        now.depth_compare = d.depth_compare;
        // BBHOST_DEPTH_ALWAYS=<pipeline prefix> (checks): that pipeline's depth
        // test always passes - whether a multipass surface loses pixels to its
        // own first pass's depth.
        static const char* depth_always = std::getenv("BBHOST_DEPTH_ALWAYS");
        if (depth_always && *depth_always && bind_pl->name.rfind(depth_always, 0) == 0) now.depth_compare = VK_COMPARE_OP_ALWAYS;
        now.front_ops = d.front;
        now.back_ops = d.back;
        if (d.depth_bias) {
            now.bias_constant = d.bias_constant;
            now.bias_clamp = d.bias_clamp;
            now.bias_slope = d.bias_slope;
        }
        now.depth_clamp = g.dynamic_depth_clamp && d.depth_clamp ? VK_TRUE : VK_FALSE;
        if (!g_recorded.library_set || std::memcmp(&g_recorded.library, &now, sizeof(now)) != 0) {
            cmds.library_state(now, g.has_depth_bounds);
            g_recorded.library = now;
            g_recorded.library_set = true;
        }
    }
    const std::uint32_t params_offsets[2] = {static_cast<std::uint32_t>(ubi[0].offset), static_cast<std::uint32_t>(ubi[1].offset)};
    const VkPipelineLayout draw_layout = bind_pl->layout ? bind_pl->layout : g.gfx_pipe_layout;
    cmds.bind_sets(draw_layout, sets, set_cache_on() ? params_offsets : nullptr, bind_pl->vs.meta().bindless);
    if (cb_push_on() && any_buffers) {
        // The draw's constant buffers, each bound over its own range.
        std::uint32_t bindings[2 * gcn::kMaxBuffers];
        VkDescriptorBufferInfo ranges[2 * gcn::kMaxBuffers];
        std::uint32_t n = 0;
        for (int st = 0; st < 2; ++st) {
            const gcn::TranslateResult& meta = st == 0 ? pl.vs.meta() : pl.ps.meta();
            for (std::size_t i = 0; i < meta.buffers.size() && n < 2 * gcn::kMaxBuffers; ++i, ++n) {
                bindings[n] = meta.buffers[i].binding;
                ranges[n] = buffer_infos[stage_first_buffer[st] + i];
            }
        }
        cmds.push_buffers(draw_layout, bindings, ranges, n);
    }
    draw_stamp.to(kRenderCostBindCmds);
    // Viewport: screen = offset + ndc * scale (y scale is negative on GCN for a y-down framebuffer).
    VkViewport vp{};
    const bool vport_enabled = (s.vte_cntl & 0x1) != 0 || s.vte_cntl == 0;
    if (vport_enabled && s.vport[0] != 0.0f) {
        vp.x = s.vport[1] - s.vport[0];
        vp.width = 2.0f * s.vport[0];
        vp.y = s.vport[3] - s.vport[2];
        vp.height = 2.0f * s.vport[2];
        vp.minDepth = s.vport[5];
        vp.maxDepth = s.vport[5] + s.vport[4];
    } else {
        vp.x = 0;
        vp.y = 0;
        vp.width = static_cast<float>(g_pass.extent.width);
        vp.height = static_cast<float>(g_pass.extent.height);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
    }
    // A reversed range (zscale < 0) stays reversed: Vulkan only requires
    // both ends inside [0, 1].
    vp.minDepth = std::clamp(vp.minDepth, 0.0f, 1.0f);
    vp.maxDepth = std::clamp(vp.maxDepth, 0.0f, 1.0f);
    if (!g_recorded.viewport_set || std::memcmp(&g_recorded.viewport, &vp, sizeof(vp)) != 0) {
        cmds.viewport(vp);
        g_recorded.viewport = vp;
        g_recorded.viewport_set = true;
    }
    VkRect2D sc = s.scissor;
    sc.extent.width = std::min<std::uint32_t>(sc.extent.width, g_pass.extent.width > static_cast<std::uint32_t>(sc.offset.x) ? g_pass.extent.width - sc.offset.x : 0);
    sc.extent.height = std::min<std::uint32_t>(sc.extent.height, g_pass.extent.height > static_cast<std::uint32_t>(sc.offset.y) ? g_pass.extent.height - sc.offset.y : 0);
    if (!sc.extent.width || !sc.extent.height) {
        return true;  // fully scissored out
    }
    if (!g_recorded.scissor_set || std::memcmp(&g_recorded.scissor, &sc, sizeof(sc)) != 0) {
        cmds.scissor(sc);
        g_recorded.scissor = sc;
        g_recorded.scissor_set = true;
    }
    float bounds[2] = {0.0f, 1.0f};
    if (g.has_depth_bounds) {
        auto bound = [](float v, float fallback) { return std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : fallback; };
        bounds[0] = bound(s.depth_bounds_min, 0.0f);
        bounds[1] = bound(s.depth_bounds_max, 1.0f);
        if (!g_recorded.bounds_set || std::memcmp(g_recorded.bounds, bounds, sizeof(bounds)) != 0) {
            cmds.depth_bounds(bounds[0], bounds[1]);
            std::memcpy(g_recorded.bounds, bounds, sizeof(bounds));
            g_recorded.bounds_set = true;
        }
    }
    const std::uint32_t stencil[6] = {s.stencil_ref & 0xff,         s.stencil_ref_bf & 0xff,         (s.stencil_ref >> 8) & 0xff,
                                      (s.stencil_ref_bf >> 8) & 0xff, (s.stencil_ref >> 16) & 0xff, (s.stencil_ref_bf >> 16) & 0xff};
    if (!g_recorded.stencil_set || std::memcmp(g_recorded.stencil, stencil, sizeof(stencil)) != 0) {
        cmds.stencil(stencil);
        std::memcpy(g_recorded.stencil, stencil, sizeof(stencil));
        g_recorded.stencil_set = true;
    }
    float blend_const[4];
    std::memcpy(blend_const, in.blend_const, sizeof(blend_const));
    if (!g_recorded.blend_set || std::memcmp(g_recorded.blend, blend_const, sizeof(blend_const)) != 0) {
        cmds.blend_constants(blend_const);
        std::memcpy(g_recorded.blend, blend_const, sizeof(blend_const));
        g_recorded.blend_set = true;
    }
    draw_stamp.to(kRenderCostDynamic);
    // VGT_INDX_OFFSET (context reg 0x102) is the base vertex the hardware
    // adds to every index before the vertex shader sees it in V0. We inline
    // the fetch shader faithfully (so any user-SGPR vertex add still applies);
    // this is the separate hardware add, passed as vertexOffset/firstVertex.
    const std::int32_t vtx_off = t_tess ? 0 : in.base_vertex;  // the LS pass took the index buffer and base vertex
    if (vtx_off) {
        static std::atomic<int> vlogs{0};
        if ((vlogs.load(std::memory_order_relaxed) < 16 && vlogs.fetch_add(1) < 16))
            host_log("render: draw %s base_vertex=%d (VGT_INDX_OFFSET) count=%u", pl.name.c_str(), vtx_off, d.index_count);
    }
    if (d.index_va) {
        if (g_import_audit && d.indirect_va) audit_range(d.index_va, static_cast<std::uint64_t>(d.index_count) * (d.index_type == 1 ? 4 : 2), kUseIndex);
        const std::uint64_t index_bytes = static_cast<std::uint64_t>(d.index_count) * (d.index_type == 1 ? 4 : 2);
        Located loc;
        const int version = copy_version_lookup_locked(d.index_va, index_bytes, &loc);  // a straddle ended the pass above
        if (version != kCopyVersionInside) {
            loc = locate(d.index_va, index_bytes);
            if (!d.indirect_va) shadow_locate_locked(d.index_va, index_bytes, loc);
        }
        if (!loc.buffer) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 4) host_log("render: index buffer 0x%llx not in imported memory", static_cast<unsigned long long>(d.index_va));
            draw_failed(kFailIndexBuffer);
            return false;
        }
        note_arena_read(kArenaIndex, d.index_va, static_cast<std::uint64_t>(d.index_count) * (d.index_type == 1 ? 4 : 2));
        if (glitch_watching_draw()) glitch_watch_read_locked(d.index_va, static_cast<std::uint64_t>(d.index_count) * (d.index_type == 1 ? 4 : 2), 1);
        const VkIndexType itype = d.index_type == 1 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
        if (g_recorded.index_buffer != loc.buffer || g_recorded.index_offset != loc.offset || g_recorded.index_type != itype) {
            cmds.index_buffer(loc.buffer, loc.offset, itype);
            g_recorded.index_buffer = loc.buffer;
            g_recorded.index_offset = loc.offset;
            g_recorded.index_type = itype;
        }
    }
    if (use_vertex_input) {
        cmds.vertex_buffers(static_cast<std::uint32_t>(vertex_input.bindings.size()), vertex_input.buffer, vertex_input.offset);
    }
    draw_stamp.to(kRenderCostGeometry);
    if (g.cmd_buffer_marker) {
        cmds.marker(gpu_marker_value(1, g_draw_rec_next));
    } else {
        gpu_checkpoint(1, g_draw_rec_next);
    }
    if (bind_pl->profile_name.empty()) {
        const bool walks = !bind_pl->vs.meta().walks.empty() || !bind_pl->ps.meta().walks.empty();
        bind_pl->profile_name = pl.name + (bind_pl->lean ? "" : "~fallback") + (walks ? "~walks" : "");
        static int logs = 0;  // under g.mu
        if (walks && logs < 400) {
            ++logs;
            std::string why;
            for (int st = 0; st < 2; ++st) {
                for (const auto& [site, n] : (st ? bind_pl->ps.meta() : bind_pl->vs.meta()).walks) {
                    if (why.size() < 400) why += std::string(why.empty() ? "" : "; ") + (st ? "PS " : "VS ") + site + " x" + std::to_string(n);
                }
            }
            host_log("render: pipeline %s%s keeps page-table walks: %s", pl.name.c_str(), bind_pl->lean ? "" : " (fallback)", why.c_str());
        }
    }
    if (!g.profile_passes) profile_begin_locked(&bind_pl->profile_name);
    if (d.indirect_va) {
        if (g_import_audit) audit_range(d.indirect_va, 20, kUseIndex);
        Located loc;
        const int version = copy_version_lookup_locked(d.indirect_va, 20, &loc);  // a straddle ended the pass above
        if (version != kCopyVersionInside) loc = locate(d.indirect_va, 20);
        if (!loc.buffer) {
            if (!g.profile_passes) profile_end_locked();
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 8)
                host_log("render: indirect args 0x%llx not in imported memory", static_cast<unsigned long long>(d.indirect_va));
            draw_failed(kFailIndirectArgs);
            return false;
        }
        note_arena_read(kArenaIndirect, d.indirect_va, d.index_va ? 20 : 16);
        DrawCall call;
        call.kind = d.index_va ? DrawCall::kIndexedIndirect : DrawCall::kIndirect;
        call.buffer = loc.buffer;
        call.offset = loc.offset;
        glitch_draw_locked(call);
        occlusion_draw_locked(call);
        cmds.draw(call);
    } else if (d.index_va) {
        DrawCall call;
        call.kind = DrawCall::kIndexed;
        call.count = d.index_count;
        call.instances = d.instance_count;
        call.vertex_offset = vtx_off;
        glitch_draw_locked(call);
        occlusion_draw_locked(call);
        cmds.draw(call);
    } else {
        DrawCall call;
        call.kind = DrawCall::kDirect;
        call.count = d.index_count;
        call.instances = d.instance_count;
        call.vertex_offset = vtx_off;
        glitch_draw_locked(call);
        occlusion_draw_locked(call);
        cmds.draw(call);
    }
    if (!g.profile_passes) profile_end_locked();
    cmds.publish();  // the diagnostics below may record or flush: they come after this draw
    bump(g.draws);  // the one writer, under g.mu: no locked add
    bump(g.draws_ps[draw_ps_kind(*bind_pl)]);
    draw_stamp.to(kRenderCostRecord);
    if (g_render_cost_enabled) report_render_split(false);
    if (capture_session) {
        CaptureDynamic dyn;
        dyn.viewport = vp;
        dyn.scissor = sc;
        dyn.render_area = g_pass.extent;
        dyn.depth_bounds_dynamic = g.has_depth_bounds;
        std::memcpy(dyn.depth_bounds, bounds, sizeof(bounds));
        std::memcpy(dyn.stencil, stencil, sizeof(stencil));
        std::memcpy(dyn.blend_constants, blend_const, sizeof(blend_const));
        capture_finish_locked(std::move(capture_session), dyn);
    }
    // BBHOST_TRACE_TARGET=0xbase: for the frame after F12 (or "0xbase@flip",
    // that flip), after every draw that binds that colour target, read it
    // back and log each draw that raises a channel's largest value - which
    // draw put a value there. Float targets only (RGBA16F, R11G11B10). Slow:
    // one readback a draw, for one frame. Found the G-buffer draws whose
    // emissive went green in the Hunter's Dream.
    static std::uint64_t trace_flip = 0;
    static const std::uint64_t trace_target = [] {
        const char* e = std::getenv("BBHOST_TRACE_TARGET");
        if (!e) return 0ull;
        if (const char* at = std::strchr(e, '@')) trace_flip = std::strtoull(at + 1, nullptr, 0);
        return std::strtoull(e, nullptr, 0);
    }();
    // Without a flip, F12 arms it: the whole frame after the one F12 lands in.
    static std::uint64_t f12_trace_flip = 0;
    if (trace_target && !trace_flip && !f12_trace_flip && g_dump_request.load(std::memory_order_acquire)) {
        f12_trace_flip = hle_video_flip_count() + 1;
    }
    if (trace_target && hle_video_flip_count() == (trace_flip ? trace_flip : f12_trace_flip)) {
        for (int t = 0; t < 8; ++t) {
            RtImage* rt = s.color[t];
            if (!rt || rt->base != trace_target) continue;
            static float seen[3] = {0.f, 0.f, 0.f};
            static std::uint64_t seen_flip = ~0ull;
            if (seen_flip != hle_video_flip_count()) {
                seen_flip = hle_video_flip_count();
                seen[0] = seen[1] = seen[2] = 0.f;
            }
            float mx[3] = {0.f, 0.f, 0.f};
            if (gpu::rt_float_max_locked(*rt, mx) && (mx[0] > seen[0] * 1.1f + 1.f || mx[1] > seen[1] * 1.1f + 1.f ||
                                                      mx[2] > seen[2] * 1.1f + 1.f)) {
                host_log("trace: 0x%llx (slot %d) max %g %g %g after %s n=%u mask %x smask %08x colfmt %08x",
                         static_cast<unsigned long long>(trace_target), t, mx[0], mx[1], mx[2], pl.name.c_str(), d.index_count,
                         s.color_mask[t], s.cb_shader_mask, s.ps_col_format);
                for (int k = 0; k < 3; ++k) seen[k] = std::max(seen[k], mx[k]);
            }
        }
    }
    // BBHOST_DUMP_AT_DRAW=<pipeline name prefix>:<occurrence>: after that
    // draw, wait for the GPU and write every colour target to
    // build/rtd-<base>.ppm (the state in the middle of a frame).
    static const std::string dump_at = [] {
        const char* e = std::getenv("BBHOST_DUMP_AT_DRAW");
        return std::string(e ? e : "");
    }();
    if (!dump_at.empty()) {
        static std::uint64_t seen = 0;
        static bool done = false;
        // "<name>:<occurrence>[:<min flip>]": occurrences count from that flip on.
        std::string want = dump_at;
        std::uint64_t occurrence = 1, min_flip = 0;
        std::uint32_t min_count = 0;
        const std::size_t c1 = want.find(':');
        if (c1 != std::string::npos) {
            const std::string rest = want.substr(c1 + 1);
            want = want.substr(0, c1);
            const std::size_t c2 = rest.find(':');
            occurrence = std::strtoull(rest.c_str(), nullptr, 10);
            if (c2 != std::string::npos) min_flip = std::strtoull(rest.c_str() + c2 + 1, nullptr, 10);
        }
        if (const std::size_t pc = want.find('%'); pc != std::string::npos) {  // "%<n>": at least n indices
            min_count = static_cast<std::uint32_t>(std::strtoul(want.c_str() + pc + 1, nullptr, 10));
            want = want.substr(0, pc);
        }
        if (!done && pl.name.find(want) == 0 && d.index_count >= min_count && hle_video_flip_count() >= min_flip && ++seen == occurrence) {
            done = true;
            host_log("render: dumping every colour target after draw %llu of %s (targets %s / depth 0x%llx)",
                     static_cast<unsigned long long>(occurrence), pl.name.c_str(), s.color[0] ? "bound" : "none",
                     static_cast<unsigned long long>(s.depth ? s.depth->base : 0));
            flush_locked();
            for (auto& kv : g_rts) {
                if (!kv.second.initialised) continue;
                char pth[96];
                std::snprintf(pth, sizeof(pth), "build/rtd-%llx%s.ppm", static_cast<unsigned long long>(kv.first), kv.second.depth ? "-depth" : "");
                gpu::dump_rt_locked(kv.second, pth);
            }
        }
    }
    // d3ca03f3+111fce32 is the first YEBIS colour write (lerp of the lighting
    // blit at attr0 toward a second sample at attr1). Later draws overwrite
    // 0x157010000, so an F12 of that RT is not this pass. While F12 is armed,
    // snapshot dest and the sampled source before they are reused.
    // Two pipeline names compared on every draw were 0.4% of the Steam Deck's
    // command processor: once a pipeline.
    if (pl.yebis_probe < 0) {
        pl.yebis_probe = pl.name.compare(0, 17, "d3ca03f3+111fce32") == 0 ? 1 : pl.name.compare(0, 17, "7d668276+e0305cef") == 0 ? 2 : 0;
    }
    if (pl.yebis_probe == 1) {
        static std::atomic<int> ylog{0};
        if (ylog.fetch_add(1) == 0) {
            host_log("yebis: first-write %s n=%u prim=%u ena=%08x cntl0=%08x cntl1=%08x rt0=0x%llx", pl.name.c_str(),
                     d.index_count, s.prim, s.ps_input_ena, s.ps_input_cntl[0], s.ps_input_cntl[1],
                     static_cast<unsigned long long>(s.color[0] ? s.color[0]->base : 0));
            const std::uint32_t* vw = &s.vs_user[8];
            const std::uint64_t vb = static_cast<std::uint64_t>(vw[0]) | (static_cast<std::uint64_t>(vw[1] & 0xff) << 32);
            if (hle_kernel_va_mapped(vb + 0x14, 8)) {
                const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vb + 0x14));
                host_log("yebis: VS scale s[8:11]+0x14 = %g %g (V# base 0x%llx stride %u records %u)", f[0], f[1],
                         static_cast<unsigned long long>(vb), (vw[1] >> 16) & 0x3fff, vw[2]);
            }
            const std::uint64_t tbl = static_cast<std::uint64_t>(s.ps_user[2]) | (static_cast<std::uint64_t>(s.ps_user[3]) << 32);
            if (hle_kernel_va_mapped(tbl, 64)) {
                const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tbl));
                const std::uint64_t cb = static_cast<std::uint64_t>(u[12]) | (static_cast<std::uint64_t>(u[13] & 0xff) << 32);
                host_log("yebis: PS s[2:3] table 0x%llx T# %08x %08x %08x %08x  C# %08x %08x %08x %08x",
                         static_cast<unsigned long long>(tbl), u[0], u[1], u[2], u[3], u[12], u[13], u[14], u[15]);
                if (hle_kernel_va_mapped(cb + 0x130, 8)) {
                    const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(cb + 0x130));
                    host_log("yebis: lerp k,b at C#+0x4c dwords = %g %g (base 0x%llx)", f[0], f[1],
                             static_cast<unsigned long long>(cb));
                }
            }
            const std::uint64_t vtbl = static_cast<std::uint64_t>(s.vs_user[2]) | (static_cast<std::uint64_t>(s.vs_user[3]) << 32);
            if (hle_kernel_va_mapped(vtbl, 16)) {
                const auto* v = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(vtbl));
                const std::uint64_t vtx = static_cast<std::uint64_t>(v[0]) | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
                const std::uint32_t stride = (v[1] >> 16) & 0x3fff;
                host_log("yebis: fetch V#0 base 0x%llx stride %u records %u", static_cast<unsigned long long>(vtx), stride,
                         v[2]);
                if (stride && stride <= 64 && hle_kernel_va_mapped(vtx, static_cast<std::size_t>(stride) * 3)) {
                    for (int rec = 0; rec < 3; ++rec) {
                        const auto* rf = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(vtx) + rec * stride);
                        host_log("yebis: vert%u %g %g %g %g  %g %g %g %g", rec, rf[0], rf[1],
                                 stride >= 12 ? rf[2] : 0.f, stride >= 16 ? rf[3] : 0.f, stride >= 20 ? rf[4] : 0.f,
                                 stride >= 24 ? rf[5] : 0.f, stride >= 28 ? rf[6] : 0.f, stride >= 32 ? rf[7] : 0.f);
                    }
                }
            }
            if (d.index_va && hle_kernel_va_mapped(d.index_va, 8)) {
                const auto* i16 = reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(d.index_va));
                host_log("yebis: indices %u %u %u (type %u)", i16[0], i16[1], i16[2], d.index_type);
            }
        }
        static bool dumped_this_request = false;
        if (!g_dump_request.load(std::memory_order_acquire)) {
            dumped_this_request = false;
        } else if (!dumped_this_request && s.color[0]) {
            dumped_this_request = true;
            const std::uint64_t flip = hle_video_flip_count();
            std::string pth = capture_file(flip, "yebis-first");
            host_log("yebis: dumping dest 0x%llx after %s to %s", static_cast<unsigned long long>(s.color[0]->base),
                     pl.name.c_str(), pth.c_str());
            flush_locked();
            gpu::dump_rt_locked(*s.color[0], pth.c_str());
            if (pl.ps.meta().images.size() >= 1) {
                std::uint32_t tw[8] = {};
                if (resolve_resource_impl(pl.ps.meta().images[0].path, s.ps_user, 8, tw)) {
                    const std::uint64_t src = tsharp_base(tw);
                    auto it = g_rts.find(src);
                    if (it != g_rts.end() && it->second.initialised) {
                        pth = capture_file(flip, "yebis-src");
                        host_log("yebis: dumping source 0x%llx to %s", static_cast<unsigned long long>(src), pth.c_str());
                        gpu::dump_rt_locked(it->second, pth.c_str());
                    }
                }
            }
        }
    }
    if (pl.yebis_probe == 2) {
        static std::atomic<int> blog{0};
        if (blog.fetch_add(1) == 0) {
            const std::uint64_t tbl = static_cast<std::uint64_t>(s.ps_user[2]) | (static_cast<std::uint64_t>(s.ps_user[3]) << 32);
            if (hle_kernel_va_mapped(tbl, 64)) {
                const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(tbl));
                const std::uint64_t cb = static_cast<std::uint64_t>(u[12]) | (static_cast<std::uint64_t>(u[13] & 0xff) << 32);
                host_log("yebis: e0305cef C# base 0x%llx T# %08x %08x %08x %08x", static_cast<unsigned long long>(cb), u[0],
                         u[1], u[2], u[3]);
                auto load4 = [&](std::uint32_t dw, const char* tag) {
                    const std::uint64_t at = cb + static_cast<std::uint64_t>(dw) * 4;
                    if (!hle_kernel_va_mapped(at, 16)) return;
                    const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(at));
                    host_log("yebis: e0305cef %s +0x%x dw = %g %g %g %g", tag, dw, f[0], f[1], f[2], f[3]);
                };
                load4(0x274, "s0-3 blurred result scale");
                load4(0x2e8, "s0-3 kernel scale/bias");
                load4(0x2ec, "s36-39 tap offset");
                load4(0x2f2, "s32 thresh / s34-35");
                load4(0x2f4, "s34-35");
                load4(0x2f8, "s40-43 uv xform");
            }
            if (!fetch_words.empty()) {
                const gcn::Program fp = gcn::decode(fetch_words.data(), fetch_words.size());
                int n = 0;
                for (const gcn::Inst& in : fp.insts) {
                    if (n++ >= 40) break;
                    host_log("  yebis fetch: %s", gcn::format(in).c_str());
                }
            }
        }
        static bool dumped_blur = false;
        if (!g_dump_request.load(std::memory_order_acquire)) {
            dumped_blur = false;
        } else if (!dumped_blur && s.color[0]) {
            dumped_blur = true;
            const std::uint64_t flip = hle_video_flip_count();
            const std::string pth = capture_file(flip, "yebis-blur");
            host_log("yebis: dumping dest 0x%llx after %s to %s", static_cast<unsigned long long>(s.color[0]->base),
                     pl.name.c_str(), pth.c_str());
            flush_locked();
            gpu::dump_rt_locked(*s.color[0], pth.c_str());
        }
    }
    if (g_order_target && s.color[0] && s.color[0]->base == g_order_target && hle_video_flip_count() >= 2000 && g_order_logs.load() < 120) {
        g_order_logs.fetch_add(1);
        host_log("order: DRAW 0x%llx %s n=%u", static_cast<unsigned long long>(g_order_target), pl.name.c_str(), d.index_count);
    }
    {
        DrawRec& r = g_draw_recs[g_draw_rec_next++ % kDrawRecs];
        const std::size_t name_len = std::min<std::size_t>(pl.name.size(), sizeof(r.name) - 1);  // no snprintf: every draw
        std::memcpy(r.name, pl.name.data(), name_len);
        r.name[name_len] = 0;
        r.count = d.index_count;
        r.inst = d.instance_count;
        r.prim = s.prim;
        r.vp[0] = vp.x; r.vp[1] = vp.y; r.vp[2] = vp.width; r.vp[3] = vp.height;
        r.sc[0] = sc.offset.x; r.sc[1] = sc.offset.y; r.sc[2] = static_cast<std::int32_t>(sc.extent.width); r.sc[3] = static_cast<std::int32_t>(sc.extent.height);
        r.rt0 = s.color[0] ? s.color[0]->base : 0;
        for (int k = 0; k < 7; ++k) r.rt[k] = s.color[k + 1] ? s.color[k + 1]->base : 0;
        r.depth = s.depth ? s.depth->base : 0;
        r.blend0 = s.blend[0];
        r.depth_ctl = s.depth_control;
        r.mask0 = s.color_mask[0];
        r.shader_mask = s.cb_shader_mask;
        r.col_format = s.ps_col_format;
        r.indexed = d.index_va != 0;
        r.index_type = d.index_type;
        r.base_vertex = vtx_off;
        r.index_va = d.index_va;
        r.indirect_va = d.indirect_va;
        for (std::size_t k = 0; k < 2; ++k) {
            const bool bound = use_vertex_input && k < vertex_input.bindings.size();
            r.vb[k] = bound ? vertex_input.va[k] : 0;
            r.vb_stride[k] = bound ? vertex_input.bindings[k].stride : 0;
        }
        r.dummies = g_draw_dummies;
        g_draw_dummies = 0;
        r.tex0 = 0;
        // The T#s the prefetch resolved: from a GX draw's records where they
        // cover the binding. Walking s.ps_user again would read, for a draw
        // from a host-draw token, the tables of the last packet draw.
        const StageImages& ps_images = stage_images[1];
        for (std::size_t k = 0; k < 8; ++k) {
            r.tex[k] = 0;
            if (k < pl.ps.meta().images.size() && k < ps_images.images.size() && !pl.ps.meta().images[k].r128 &&
                ps_images.images[k].resolved) {
                r.tex[k] = tsharp_base(ps_images.images[k].w);
                std::memcpy(r.tsharp[k], ps_images.images[k].w, sizeof(r.tsharp[k]));
            }
        }
        r.tex0 = r.tex[0];
    }
    if (g_trace) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 200) {
            host_log("render: draw %s count=%u inst=%u prim=%u vp=(%.0f,%.0f %.0fx%.0f) sc=(%d,%d %ux%u) rt0=0x%llx depth=0x%llx%s",
                     pl.name.c_str(), d.index_count, d.instance_count, s.prim, vp.x, vp.y, vp.width, vp.height, sc.offset.x,
                     sc.offset.y, sc.extent.width, sc.extent.height,
                     static_cast<unsigned long long>(s.color[0] ? s.color[0]->base : 0),
                     static_cast<unsigned long long>(s.depth ? s.depth->base : 0), d.index_va ? " indexed" : "");
        }
    }
    return true;
}

bool host_gpu_blit_display(void* cmd, std::uint64_t display_va, void* dst_image, std::int32_t dst_x, std::int32_t dst_y,
                           std::uint32_t dst_w, std::uint32_t dst_h, std::uint32_t src_width, std::uint32_t src_height, std::uint32_t src_x,
                           std::uint32_t src_y, void* dst_storage_view) {
    return render_blit_display_locked(static_cast<VkCommandBuffer>(cmd), display_va, static_cast<VkImage>(dst_image),
                                      {{dst_x, dst_y}, {dst_w, dst_h}}, {src_width, src_height}, src_x, src_y,
                                      static_cast<VkImageView>(dst_storage_view));
}

bool host_gpu_display_image(std::uint64_t display_va, void** image, std::uint32_t* format, std::uint32_t* width,
                            std::uint32_t* height) {
    // What render_blit_display_locked reads, for the presenter's own pass
    // (host/present_pass.cpp): the target is in GENERAL, as the blit takes it.
    auto it = g_rts.find(display_va);
    if (it == g_rts.end() || !it->second.initialised || !it->second.image || it->second.depth) return false;
    *image = it->second.image;
    *format = static_cast<std::uint32_t>(it->second.format);
    *width = it->second.width;
    *height = it->second.height;
    return true;
}

namespace gpu {
bool dump_rt_locked(RtImage& r, const char* path);
}

// Writes the displayed render target to a binary PPM (BBHOST_DUMP_FRAME).
// Called from the watchdog when the game has stopped flipping, so the thread
// that is stuck is very likely holding g.mu: read the draw ring WITHOUT the
// lock. The values can be torn; a torn line is still better than no report.
namespace gpu {
std::string tess_lds_describe(std::uint64_t addr, std::uint64_t precision) {
    std::lock_guard<std::mutex> lk(g_tess_lds_mu);
    if (!g_tess_ring_address) return {};
    TessLdsUse uses[kTessLdsUses];
    const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(g_tess_lds_next, kTessLdsUses));
    for (std::size_t k = 0; k < n; ++k) uses[k] = g_tess_lds_uses[(g_tess_lds_next - 1 - k) % kTessLdsUses];  // newest first
    return tess_lds_where(g_tess_ring_address, kTessRingBytes, kTessRegionBytes, uses, n, addr, precision);
}

void describe_draw_record(std::uint64_t index) {
    if (index >= g_draw_rec_next || (g_draw_rec_next - index) > kDrawRecs) {
        host_log("    draw %llu is no longer in the ring", static_cast<unsigned long long>(index));
        return;
    }
    const DrawRec& r = g_draw_recs[index % kDrawRecs];
    host_log("    draw[%llu] %s n=%u inst=%u prim=%u vp=(%.0f,%.0f %.0fx%.0f) sc=(%d,%d %dx%d) rt0=0x%llx depth=0x%llx tex=0x%llx,0x%llx,0x%llx,0x%llx%s",
             static_cast<unsigned long long>(index), r.name, r.count, r.inst, r.prim, r.vp[0], r.vp[1], r.vp[2], r.vp[3], r.sc[0],
             r.sc[1], r.sc[2], r.sc[3], static_cast<unsigned long long>(r.rt0), static_cast<unsigned long long>(r.depth),
             static_cast<unsigned long long>(r.tex[0]), static_cast<unsigned long long>(r.tex[1]),
             static_cast<unsigned long long>(r.tex[2]), static_cast<unsigned long long>(r.tex[3]), r.indexed ? " idx" : "");
}
}  // namespace gpu

void host_gpu_hang_report() {
    compute_compiling_report();
    {
        // Every watchdog that reports a hang says where the presenter is and
        // who holds the renderer's lock, not only the GPU's own.
        double present_for = 0;
        const char* const step = host_present_step(present_for);
        host_log("hang: %s; the presenting thread is %s (%.1f s); the renderer's lock is held by %s", submit_thread_report().c_str(), step,
                 present_for, GpuMutex::name_of(g.mu.holder()).c_str());
        // Where the command stream's recorder is (recorder.cpp).
        if (const std::string s = stream_last_op(); !s.empty()) host_log("hang: %s", s.c_str());
    }
    const std::uint64_t next = g_draw_rec_next;
    const std::uint64_t first = next > 6 ? next - 6 : 0;
    host_log("hang: %llu draws recorded, %llu dispatches, %llu queued, pass %s",
             static_cast<unsigned long long>(g.draws.load()), static_cast<unsigned long long>(g.dispatches.load()),
             static_cast<unsigned long long>(g.queued), g_pass.active ? "open" : "closed");
    for (std::uint64_t i = first; i < next; ++i) {
        const DrawRec& r = g_draw_recs[i % kDrawRecs];
        host_log("  hang draw[%llu] %s n=%u prim=%u vp=(%.0f,%.0f %.0fx%.0f) rt0=0x%llx tex=0x%llx,0x%llx",
                 static_cast<unsigned long long>(i), r.name, r.count, r.prim, r.vp[0], r.vp[1], r.vp[2], r.vp[3],
                 static_cast<unsigned long long>(r.rt0), static_cast<unsigned long long>(r.tex[0]),
                 static_cast<unsigned long long>(r.tex[1]));
    }
}

namespace {

// The folder as the player finds it: absolute, in the system's own
// separators, as UTF-8 (a Windows user folder's name need not be ASCII).
std::string capture_dir_shown(const std::string& dir) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(dir, ec);
    if (ec) return dir;
    const std::u8string u = p.lexically_normal().make_preferred().u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

// A folder of its own for each F12 capture, beside the logs: logs/ when
// there is one (the packages' run-bbhost writes its logs there), else
// build/; BBHOST_CAPTURE_DIR names another place. Made at the key press, so
// a folder that cannot be made is in the log then.
std::string new_capture_dir() {
    std::error_code ec;
    std::string base;
    if (const char* e = std::getenv("BBHOST_CAPTURE_DIR"); e && e[0]) {
        base = e;
    } else {
        base = std::filesystem::is_directory("logs", ec) ? "logs" : "build";
    }
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char name[40];
    std::strftime(name, sizeof(name), "f12-%Y%m%d-%H%M%S", &tm);
    std::string dir = base + "/" + name;
    for (int k = 2; std::filesystem::exists(dir, ec) && k < 100; ++k) dir = base + "/" + name + "-" + std::to_string(k);
    std::filesystem::create_directories(dir, ec);
    if (ec) host_log("dump: F12 cannot make the folder %s: %s", capture_dir_shown(dir).c_str(), ec.message().c_str());
    return dir;
}

}  // namespace

void host_gpu_request_dump() {
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (g_dump_request.load(std::memory_order_acquire)) return;  // one already waits for its flip
    g_capture_dir = new_capture_dir();
    g_dump_request.store(true, std::memory_order_release);
    host_log("dump: F12 capture on the next flip, into %s", capture_dir_shown(g_capture_dir).c_str());
}
bool host_gpu_take_dump_request(std::string* dir) {
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (!g_dump_request.exchange(false, std::memory_order_acq_rel)) return false;
    if (dir) *dir = g_capture_dir;
    return true;
}
std::string host_gpu_capture_dir() {
    std::lock_guard<std::mutex> lk(g_capture_mu);
    return g_capture_dir;
}
// PPM costs nothing to write (a world frame's 69 images in 0.2 s) and zips to
// little more than PNG does (71 MB against 59); PNG is a fifth of the size on
// disk and opens in any image viewer, for 4 s of a background thread.
const char* host_gpu_capture_ext() {
    static const bool png = [] {
        const char* e = std::getenv("BBHOST_F12_PNG");
        return e && e[0] == '1';
    }();
    return png ? ".png" : ".ppm";
}

// BBHOST_RT_REFILL_TEST=1 (video.cpp, once): the blood layers' lost clear on
// the real device, at an address no guest resource uses. A 128x128 sRGB
// target takes a fill of a 256x256 RGBA8 target's bytes, then a 256x256 UNORM
// target is asked for there: every texel must be the fill. Then the same with
// a fill too small for the new target, which must not be carried.
void host_gpu_refill_selftest() {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return;
    constexpr std::uint64_t kBase = 0x7e00000000ull;  // outside the guest's windows
    const float rgba[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    const std::uint8_t want[4] = {64, 128, 191, 255};  // UNORM of rgba
    // The texels of a target that match `want`, or -1.
    auto matching = [&](RtImage& r, const std::uint8_t* want) -> long {
        DevBuffer staging;
        const std::uint64_t n = static_cast<std::uint64_t>(r.width) * r.height;
        if (!create_dev_buffer(staging, n * 4, true)) return -1;
        begin_recording_locked();
        render_end_pass_locked();
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {r.width, r.height, 1};
        vkCmdCopyImageToBuffer(g_cmd(), r.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
        flush_locked();
        const auto* px = static_cast<const std::uint8_t*>(staging.map);
        long same = 0;
        for (std::uint64_t i = 0; i < n; ++i) {
            bool ok = true;
            for (int k = 0; k < 4; ++k) ok = ok && std::abs(px[i * 4 + k] - want[k]) <= 1;
            same += ok;
        }
        vkDestroyBuffer(g.device, staging.buffer, nullptr);
        vkFreeMemory(g.device, staging.memory, nullptr);
        return same;
    };
    auto drop = [&] {
        if (auto it = g_rts.find(kBase); it != g_rts.end()) {
            destroy_rt_image(it->second);
            g_rts.erase(it);
            ++g_rt_gen;
            bump_view_epoch();
        }
    };
    struct Case {
        const char* name;
        std::size_t fill_bytes;
        bool carried;
    };
    const Case cases[] = {{"a fill covering the new target", 256 * 256 * 4, true},
                          {"a fill smaller than the new target", 128 * 128 * 4, false}};
    for (const Case& c : cases) {
        drop();
        if (!rt_image(kBase, VK_FORMAT_R8G8B8A8_SRGB, 128, 128, false)) {
            host_log("refill test: no target");
            return;
        }
        render_handle_fill_locked(kBase, c.fill_bytes, rgba, 0, 0);
        const std::uint64_t before = g_rt_refilled.load();
        RtImage* fresh = rt_image(kBase, VK_FORMAT_R8G8B8A8_UNORM, 256, 256, false);
        if (!fresh) {
            host_log("refill test: no re-created target");
            drop();
            return;
        }
        const bool carried = g_rt_refilled.load() != before;
        // Only a carried fill initialised the image; an uncarried one may sit
        // on the memory the last case filled, so its texels prove nothing.
        const long same = carried ? matching(*fresh, want) : -1;
        const bool pass = carried == c.carried && (!carried || same == 256 * 256);
        host_log("refill test: %s: %s%s - %s", c.name, carried ? "carried" : "not carried",
                 carried ? (", " + std::to_string(same) + " of 65536 texels hold it").c_str() : "", pass ? "PASS" : "FAIL");
    }
    drop();

    // The rat's case (F12 flip 16124): a drawn 256x256 sRGB target, a clear
    // of a 256x256 target's bytes starting 0x28000 into it, then that target.
    // The new one must start with the clear, and the old one keep its pixels.
    {
        constexpr std::uint64_t kInside = kBase + 0x28000;
        // Colours of 0 and 1 only: the targets are sRGB, and these are the
        // same bytes whichever way they are encoded.
        const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        const std::uint8_t want_red[4] = {255, 0, 0, 255};
        const float magenta[4] = {1.0f, 0.0f, 1.0f, 1.0f};
        const std::uint8_t want_magenta[4] = {255, 0, 255, 255};
        RtImage* old = rt_image(kBase, VK_FORMAT_R8G8B8A8_SRGB, 256, 256, false);
        if (!old) {
            host_log("refill test: no target");
            return;
        }
        clear_image_locked(*old, red, 0, ~0u);  // what was drawn there
        old->initialised = true;
        old->fill_last = false;
        render_handle_fill_locked(kInside, 256 * 256 * 4, magenta, 0, 0);
        RtImage* fresh = rt_image(kInside, VK_FORMAT_R8G8B8A8_SRGB, 256, 256, false);
        old = pick_rt(kBase);
        if (!fresh || !old) {
            host_log("refill test: lost a target");
        } else {
            const long fresh_same = matching(*fresh, want_magenta), old_red = matching(*old, want_red);
            const bool pass = fresh_same == 256 * 256 && old_red == 256 * 256;
            host_log("refill test: a clear starting inside a drawn target: the new target holds it in %ld of 65536 texels, "
                     "the old one kept its pixels in %ld - %s",
                     fresh_same, old_red, pass ? "PASS" : "FAIL");
        }
        for (const std::uint64_t b : {kBase, kInside}) {
            if (auto it = g_rts.find(b); it != g_rts.end()) {
                destroy_rt_image(it->second);
                g_rts.erase(it);
            }
            if (g_pending_clears.erase(b)) ++g_pending_gen;
        }
        ++g_rt_gen;
        bump_view_epoch();
    }
}

// BBHOST_DUMP_ALL=1 also writes every colour target as build/rt-<base>.ppm.
// `all_targets` (F12) does the same, named next to `path`.
// BBHOST_DUMP_DRAWS=<n> logs the draw list on its own - same lines, no images.
namespace {
// A colour or depth target's texels, read back, to be written as a PPM
// (write_target_image) - on the renderer's thread or, for F12, on one of its own.
struct TargetPixels {
    std::string path;
    std::uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool depth = false;
    std::size_t bpp = 0;
    std::vector<std::uint8_t> px;
    // F12: the texels stay in their host-cached staging buffer, which the
    // writer reads and then frees (no copy on the renderer's lock).
    DevBuffer staging;
    const std::uint8_t* data() const { return staging.map ? static_cast<const std::uint8_t*>(staging.map) : px.data(); }
};

bool record_target_readback(RtImage& r, DevBuffer& staging, TargetPixels& out);
void take_target_pixels(DevBuffer& staging, TargetPixels& t);
bool write_target_image(const TargetPixels& t);
}  // namespace

// F12 is a request from someone playing: it must not stop the game for long.
// It used to read back and write every render target the renderer had ever
// made (~390) and up to 400 sampled textures, one GPU wait and one file each,
// all under the renderer's lock - two minutes of a frozen game on NFS. Now
// the draw list is built in memory, the targets the recorded draws used are
// read back with one wait, and a thread of its own writes the files.
// BBHOST_F12_ALL_TARGETS=1 takes every target again; BBHOST_F12_TEXTURES=1
// also saves the sampled textures (on the renderer's lock, as before).
// BBHOST_STALL_TEST=<flip>:<ms>: hold the renderer's lock that long at that
// flip, the way a frame dump does. The game has to survive a stalled frame -
// F12 used to stall it for minutes and the game froze behind it.
void host_gpu_stall_test(unsigned ms) {
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return;
    flush_locked();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    host_log("stall test: held the renderer %.0f ms",
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

// BBHOST_DUMP_ON_BRIGHT=<mean>: watch how bright the frame the game just
// flipped is, and ask for a dump of the next one when it goes over. A frame
// that washes out is over in a moment and nobody can press F12 in time; this
// catches it with its draw list. The display is blitted down to 16x16 on the
// GPU and read back a flip later, so no frame waits for it.
void host_gpu_watch_display(std::uint64_t display_va) {
    static const double bright = [] {
        const char* e = std::getenv("BBHOST_DUMP_ON_BRIGHT");
        return e ? std::atof(e) : 0.0;
    }();
    // BBHOST_DUMP_ON_CHANGE=<mean abs difference, 0..1>: catch a frame that is
    // simply *unlike* the one before it, whatever colour it went. A threshold
    // on brightness has to be guessed per symptom - the frame that flashes
    // green passes 0.35, a grey one may not - and a frame that jumps and jumps
    // back is the shape all of these have in common.
    static const double change = [] {
        const char* e = std::getenv("BBHOST_DUMP_ON_CHANGE");
        return e ? std::atof(e) : 0.0;
    }();
    // BBHOST_DUMP_ON_CHANGE_ALSO=0x<base>: keep this target beside the display
    // one, for the same flips. The display target is the end of the chain, so
    // on its own it cannot say whether a frame went wrong in the last pass or
    // arrived wrong - keeping that pass's input answers it in one run.
    static const std::uint64_t also = [] {
        const char* e = std::getenv("BBHOST_DUMP_ON_CHANGE_ALSO");
        return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    if ((bright <= 0 && change <= 0) || !display_va) return;
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return;
    RtImage* rt = find_render_target(display_va);
    if (!rt || !rt->initialised) return;
    constexpr std::uint32_t kSide = 16;
    // One readback per flip, in a ring: waiting for a copy to *complete*
    // before recording the next one costs frames, and a one-frame flash is
    // exactly what gets lost. Each slot is read when its own submission has
    // retired, so the measurement stays honest and every flip is still
    // sampled.
    constexpr int kSmall = 16;  // submissions retire several flips behind the flip that records them
    struct Small {
        DevBuffer buf;
        std::uint64_t recorded_at = 0;  // the submission serial the copy went into
        std::uint64_t flip = 0;
        bool pending = false;
    };
    static Small smalls[kSmall];
    static int small_next = 0;
    static VkImage small = VK_NULL_HANDLE;
    static VkDeviceMemory small_mem = VK_NULL_HANDLE;
    static std::uint64_t measured_flip = 0;  // the flip of the slot being measured
    // The frame that flashes is over before the readback says so, and asking
    // for a dump then gets the *next* one - which for a one-frame flash is
    // already back to normal. So keep the last few frames at full size and
    // write out the one that was actually measured (and the one before it, to
    // compare against). Only while the watcher is on: this is 8 MiB a flip.
    struct KeptOne {
        DevBuffer buf;
        std::uint32_t width = 0, height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::size_t bpp = 0;
        bool filled = false;
    };
    struct Kept {
        KeptOne t[2];  // [0] the display target, [1] BBHOST_DUMP_ON_CHANGE_ALSO
        std::uint64_t flip = ~0ull, landed_at = 0;
        std::uint64_t rec_at = 0;  // g_draw_rec_next when this frame was kept
    };
    constexpr int kKept = 4;  // the detection lag is one flip, so four is plenty.
    // Not more than this: these are host-visible readbacks, and past about
    // 128 MiB of them together the NVIDIA driver faults inside its own copy
    // path (SIGSEGV in libnvidia-glcore) rather than failing an allocation.
    static Kept kept[kKept];
    // What the flip actually pointed at, for the last few flips. A frame that
    // goes wrong all over at once is more likely to be the wrong *target*
    // presented than every pass in it drawing differently, and that is a
    // question about the flip, not about the draws.
    struct Flipped {
        std::uint64_t flip = ~0ull, va = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t width = 0, height = 0;
        bool snapshot = false, depth = false, mutable_format = false;
        std::uint32_t layers = 0;
    };
    static Flipped flips[8];
    {
        Flipped& f = flips[hle_video_flip_count() % 8];
        f.flip = hle_video_flip_count();
        f.va = display_va;
        f.format = rt->format;
        f.width = rt->width;
        f.height = rt->height;
        f.snapshot = g_rts.find(display_va) == g_rts.end();
        f.depth = rt->depth;
        f.mutable_format = rt->mutable_format;
        f.layers = rt->layers;
    }
    static double last_mean = -1.0;
    static std::uint8_t last_small[kSide * kSide * 4] = {};
    static bool have_last = false;
    if (!smalls[0].buf.buffer) {
        for (Small& sm : smalls) {
            if (!create_dev_buffer(sm.buf, kSide * kSide * 4, true, true)) return;
        }
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {kSide, kSide, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g.device, &ici, nullptr, &small) != VK_SUCCESS) return;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(g.device, small, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
        if (mai.memoryTypeIndex == UINT32_MAX) return;
        if (vkAllocateMemory(g.device, &mai, nullptr, &small_mem) != VK_SUCCESS) return;
        vkBindImageMemory(g.device, small, small_mem, 0);
    }
    // Last flip's copy, if its submission has gone through.
    // `flushes` counts submissions *started*; a copy recorded when it read S
    // lands in serial S and has run only once `completed_submits` passes S.
    // Testing the wrong one reads a staging buffer the GPU has not filled yet,
    // which made the same ring slot give two different images when two
    // triggers a flip apart both wrote it out.
    // The oldest slot whose copy has retired, in flip order.
    Small* ready = nullptr;
    for (Small& sm : smalls) {
        if (!sm.pending || g.completed_submits <= sm.recorded_at || !sm.buf.map) continue;
        if (!ready || sm.flip < ready->flip) ready = &sm;
    }
    if (ready) {
        ready->pending = false;
        measured_flip = ready->flip;
        const auto* px = static_cast<const std::uint8_t*>(ready->buf.map);
        double sum = 0;
        for (std::uint32_t i = 0; i < kSide * kSide; ++i) sum += (px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2]) / (3.0 * 255.0);
        const double mean = sum / (kSide * kSide);
        // How unlike the frame before it this one is.
        double moved = 0.0;
        if (have_last) {
            double d = 0;
            for (std::uint32_t i = 0; i < kSide * kSide; ++i) {
                for (int c = 0; c < 3; ++c) {
                    d += std::abs(static_cast<int>(px[i * 4 + c]) - static_cast<int>(last_small[i * 4 + c])) / 255.0;
                }
            }
            moved = d / (kSide * kSide * 3);
        }
        std::memcpy(last_small, px, sizeof(last_small));
        have_last = true;
        (void)last_mean;
        // A quiet frame right after a bright one, from the same view, is what
        // the bright one has to be compared against.
        static bool had_bright = false, had_quiet = false;
        const bool quiet_one = had_bright && !had_quiet && mean > 0.05 && mean < bright * 0.4;
        if (quiet_one) had_quiet = true;
        if (mean > bright) had_bright = true;
        // Keep the patches of the flips around a bright one.
        static std::uint64_t patches_until = 0;
        if (mean > bright) patches_until = hle_video_flip_count() + 2;
        g_dump_patches.store(hle_video_flip_count() <= patches_until, std::memory_order_relaxed);
        ++g_frames_watched;
        if (mean > bright) ++g_frames_bright;
        const bool jumped = change > 0 && moved > change;
        if (jumped) ++g_frames_bright;
        if ((bright > 0 && (mean > bright || quiet_one)) || jumped) {
            static std::atomic<int> asked{0};
            if (asked.fetch_add(1) < 8) {
                host_log("dump: the frame at flip %llu averages %.2f and moved %.3f from the one before (%s) - writing it and "
                         "the one before, and dumping the next (now at flip %llu, %llu behind)",
                         static_cast<unsigned long long>(measured_flip), mean, moved,
                         jumped ? "a jump" : quiet_one ? "a quiet one to compare" : "over the mark",
                         static_cast<unsigned long long>(hle_video_flip_count()),
                         static_cast<unsigned long long>(hle_video_flip_count() - measured_flip));
                for (int back = 5; back >= 0; --back) {
                    const std::uint64_t want = measured_flip - static_cast<std::uint64_t>(back);
                    const Flipped& f = flips[want % 8];
                    if (f.flip != want) continue;
                    host_log("dump:   flip %llu presented 0x%llx %ux%u format %d%s%s%s layers %u",
                             static_cast<unsigned long long>(f.flip), static_cast<unsigned long long>(f.va), f.width, f.height,
                             static_cast<int>(f.format), f.snapshot ? " (a snapshot)" : "", f.depth ? " depth" : "",
                             f.mutable_format ? " mutable" : "", f.layers);
                }
                // The frame itself, out of the ring, before anything else runs.
                for (int back = 0; back <= 1; ++back) {
                    const std::uint64_t want = measured_flip - static_cast<std::uint64_t>(back);
                    if (g_keep_verts) {
                        for (const KeptVerts& kv : g_kept_verts) {
                            if (kv.flip != want || kv.text.empty()) continue;
                            char vp[160];
                            std::snprintf(vp, sizeof(vp), "build/flash-%llu%s-verts.txt", static_cast<unsigned long long>(want),
                                          back ? "-before" : "");
                            auto text = std::make_shared<std::string>(kv.text);
                            auto path = std::make_shared<std::string>(vp);
                            std::thread([text, path] {
                                if (FILE* f = std::fopen(path->c_str(), "wb")) {
                                    std::fwrite(text->data(), 1, text->size(), f);
                                    std::fclose(f);
                                }
                            }).detach();
                            host_log("dump: %s vertex data for flip %llu -> %s", g_keep_verts,
                                     static_cast<unsigned long long>(want), vp);
                        }
                    }
                    for (Kept& k : kept) {
                        if (k.flip != want || g.completed_submits <= k.landed_at) continue;
                        for (int n = 1; n >= 1; --n) {  // the extra target, written beside the display one
                            const KeptOne& e = k.t[n];
                            if (!e.filled || !e.buf.map) continue;
                            auto x = std::make_shared<TargetPixels>();
                            x->width = e.width;
                            x->height = e.height;
                            x->format = e.format;
                            x->bpp = e.bpp;
                            char xp[160];
                            std::snprintf(xp, sizeof(xp), "build/flash-%llu%s-%llx.ppm", static_cast<unsigned long long>(want),
                                          back ? "-before" : "", static_cast<unsigned long long>(also));
                            x->path = xp;
                            const auto* xpx = static_cast<const std::uint8_t*>(e.buf.map);
                            x->px.assign(xpx, xpx + static_cast<std::size_t>(e.width) * e.height * e.bpp);
                            std::thread([x] { write_target_image(*x); }).detach();
                        }
                        // Every kept input, named by the flip it was taken at: the
                        // blit for the frame presented at a flip runs a flip or two
                        // earlier, and which one is what the file names settle.
                        for (auto& row : g_kept_inputs) for (KeptInput& in : row) {
                            if (back || !in.filled || !in.buf.map || g.completed_submits <= in.landed_at) continue;
                            if (in.flip > want) continue;  // a later frame's capture is not this frame's evidence
                            auto x = std::make_shared<TargetPixels>();
                            x->width = in.width;
                            x->height = in.height;
                            x->format = in.format;
                            x->bpp = in.bpp;
                            char xp[160];
                            std::snprintf(xp, sizeof(xp), "build/flash-%llu-in%llu-d%llu-%llx-%s.ppm",
                                          static_cast<unsigned long long>(want), static_cast<unsigned long long>(in.flip),
                                          static_cast<unsigned long long>(in.at), static_cast<unsigned long long>(in.base),
                                          in.pass);
                            x->path = xp;
                            const auto* xpx = static_cast<const std::uint8_t*>(in.buf.map);
                            x->px.assign(xpx, xpx + static_cast<std::size_t>(in.width) * in.height * in.bpp);
                            std::thread([x] { write_target_image(*x); }).detach();
                        }
                        if (!k.t[0].filled || !k.t[0].buf.map) continue;
                        auto t = std::make_shared<TargetPixels>();
                        t->width = k.t[0].width;
                        t->height = k.t[0].height;
                        t->format = k.t[0].format;
                        t->bpp = k.t[0].bpp;
                        char path[128];
                        std::snprintf(path, sizeof(path), "build/flash-%llu%s.ppm", static_cast<unsigned long long>(want),
                                      back ? "-before" : "");
                        t->path = path;
                        const auto* px = static_cast<const std::uint8_t*>(k.t[0].buf.map);
                        t->px.assign(px, px + static_cast<std::size_t>(k.t[0].width) * k.t[0].height * k.t[0].bpp);
                        // The draws of that frame, from the ring: everything
                        // recorded between the frame before it was kept and it
                        // was. The dump the watcher asks for lands on the
                        // frame *after*, which is the wrong frame's list.
                        std::string list;
                        std::uint64_t from = 0;
                        for (const Kept& prev : kept) {
                            if (prev.flip + 1 == want) from = prev.rec_at;
                        }
                        if (from && k.rec_at > from) {
                            const std::uint64_t lo = std::max(from, k.rec_at > kDrawRecs ? k.rec_at - kDrawRecs : 0);
                            char line[512];
                            for (std::uint64_t i = lo; i < k.rec_at; ++i) {
                                const DrawRec& dr = g_draw_recs[i % kDrawRecs];
                                std::snprintf(line, sizeof(line),
                                              "draw[%llu] %s n=%u inst=%u prim=%u vp=(%.0f,%.0f %.0fx%.0f) rt0=0x%llx depth=0x%llx "
                                              "tex=0x%llx,0x%llx,0x%llx,0x%llx blend=%08x dctl=%08x mask=%x%s\n",
                                              static_cast<unsigned long long>(i), dr.name, dr.count, dr.inst, dr.prim, dr.vp[0],
                                              dr.vp[1], dr.vp[2], dr.vp[3], static_cast<unsigned long long>(dr.rt0),
                                              static_cast<unsigned long long>(dr.depth), static_cast<unsigned long long>(dr.tex[0]),
                                              static_cast<unsigned long long>(dr.tex[1]), static_cast<unsigned long long>(dr.tex[2]),
                                              static_cast<unsigned long long>(dr.tex[3]), dr.blend0, dr.depth_ctl, dr.mask0,
                                              dr.indexed ? " idx" : "");
                                list += line;
                            }
                        }
                        auto text = std::make_shared<std::string>(std::move(list));
                        auto dpath = std::make_shared<std::string>(t->path.substr(0, t->path.size() - 4) + "-draws.txt");
                        std::thread([t, text, dpath] {
                            write_target_image(*t);
                            if (text->empty()) return;
                            if (FILE* f = std::fopen(dpath->c_str(), "wb")) {
                                std::fwrite(text->data(), 1, text->size(), f);
                                std::fclose(f);
                            }
                        }).detach();
                        host_log("dump: kept flip %llu -> %s (%zu draws)", static_cast<unsigned long long>(want), t->path.c_str(),
                                 static_cast<std::size_t>(std::count(text->begin(), text->end(), '\n')));
                        break;
                    }
                }
                host_gpu_request_dump();
            }
        }
    }
    // Sample every flip, into a slot nothing is waiting on. Overwriting a
    // pending one loses the copy *and* the measurement, which is how a ring
    // shallower than the submission pipeline stopped firing at all.
    Small* free_slot = nullptr;
    for (int i = 0; i < kSmall && !free_slot; ++i) {
        Small& c = smalls[(small_next + i) % kSmall];
        if (!c.pending) free_slot = &c;
    }
    if (!free_slot) return;
    small_next = static_cast<int>(free_slot - smalls + 1) % kSmall;
    Small& slot = *free_slot;
    begin_recording_locked();
    transfer_flush_locked();
    render_end_pass_locked();
    VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_src.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_src.srcQueueFamilyIndex = to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = small;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    to_src.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &to_src);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<std::int32_t>(rt->width), static_cast<std::int32_t>(rt->height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {static_cast<std::int32_t>(kSide), static_cast<std::int32_t>(kSide), 1};
    vkCmdBlitImage(g_cmd(), rt->image, VK_IMAGE_LAYOUT_GENERAL, small, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);
    VkImageMemoryBarrier to_read = to_src;
    to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &to_read);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {kSide, kSide, 1};
    vkCmdCopyImageToBuffer(g_cmd(), small, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.buf.buffer, 1, &copy);
    slot.recorded_at = g.flushes;
    slot.flip = hle_video_flip_count();
    slot.pending = true;
    // And the frame itself, into the ring, so the one that flashes can be
    // written out rather than the one after it.
    {
        Kept& k = kept[slot.flip % kKept];
        RtImage* srcs[2] = {rt, also ? find_render_target(also) : nullptr};
        bool any = false;
        for (int n = 0; n < 2; ++n) {
            KeptOne& one = k.t[n];
            one.filled = false;
            RtImage* src = srcs[n];
            if (!src || !src->initialised || src->depth) continue;
            const std::size_t bpp = rt_bytes_per_pixel(*src);
            const std::uint64_t bytes = static_cast<std::uint64_t>(src->width) * src->height * bpp;
            if (one.buf.buffer == VK_NULL_HANDLE || one.width != src->width || one.height != src->height || one.bpp != bpp) {
                if (one.buf.buffer) {
                    vkDestroyBuffer(g.device, one.buf.buffer, nullptr);
                    vkFreeMemory(g.device, one.buf.memory, nullptr);
                    one.buf = DevBuffer{};
                }
                if (create_dev_buffer(one.buf, bytes, true, true)) {
                    one.width = src->width;
                    one.height = src->height;
                    one.bpp = bpp;
                }
            }
            if (!one.buf.buffer) continue;
            one.format = src->format;
            VkBufferImageCopy full{};
            full.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            full.imageExtent = {src->width, src->height, 1};
            vkCmdCopyImageToBuffer(g_cmd(), src->image, VK_IMAGE_LAYOUT_GENERAL, one.buf.buffer, 1, &full);
            one.filled = true;
            any = true;
        }
        if (any) {
            k.flip = slot.flip;
            k.landed_at = g.flushes;
            k.rec_at = g_draw_rec_next;
        }
    }
    if (g_keep_verts) {
        // Whichever frame has just finished - the draws are recorded on the
        // command-processor thread and the counter has usually moved on by the
        // time the flip runs, so keying this to the current flip probed
        // nothing at all.
        KeptVerts* pick = nullptr;
        for (KeptVerts& c : g_kept_verts) {
            if (c.probed || c.ranges.empty() || c.flip > hle_video_flip_count()) continue;
            if (!pick || c.flip > pick->flip) pick = &c;
        }
        if (pick) {
            KeptVerts& kv = *pick;
            kv.probed = true;
            char probe[256], line[400];
            std::snprintf(line, sizeof(line), "-- at the end of frame %llu (now flip %llu) --\n",
                          static_cast<unsigned long long>(kv.flip), static_cast<unsigned long long>(hle_video_flip_count()));
            kv.text += line;
            for (std::size_t r = 0; r < kv.ranges.size(); ++r) {
                const auto& [va, bytes] = kv.ranges[r];
                shadow_probe_locked(va, bytes, probe, sizeof(probe));
                // The bytes now against the bytes when the draw was recorded.
                // The mirror's copy is made at submit, so it holds what the
                // range says *here*; if that differs, the draw recorded
                // earlier read the later contents.
                const std::uint64_t now = hle_kernel_va_mapped(va, bytes)
                                              ? fnv1a(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), bytes)
                                              : 0;
                const bool moved = r < kv.range_hash.size() && now != kv.range_hash[r];
                std::snprintf(line, sizeof(line), "  [0x%llx +%llu] %s%s\n", static_cast<unsigned long long>(va),
                              static_cast<unsigned long long>(bytes), probe, moved ? "  <= REWRITTEN SINCE THE DRAW" : "");
                kv.text += line;
            }
        }
    }
    // BBHOST_SYNC_FRAME=1: wait for the GPU at every flip. The game
    // double-buffers the dynamic vertex and constant buffers a menu pass
    // reads, rotating between two addresses a frame; if our submissions run
    // behind, the GPU reads one the guest has already started rewriting. Every
    // byte handed to the pass is identical when read at record time, which is
    // exactly what a race like that looks like from the CPU. If the flash goes
    // when the GPU is kept in step, that is the mechanism.
    static const bool sync_frame = [] {
        const char* e = std::getenv("BBHOST_SYNC_FRAME");
        return e && e[0] == '1';
    }();
    // Submit these copies now rather than letting them wait for the frame's
    // own flush. Waiting for a copy to *complete* before trusting it is
    // correct but costs latency, and the frame ring is only a few flips deep -
    // left to the natural flush the detection fell so far behind that the
    // frame it named had already rotated out and nothing was written at all.
    flush_locked();
    if (sync_frame) {
        QueueGuard queue;
        vkQueueWaitIdle(g.queue);
    }
}

namespace {

// A dump's path without its ".png" or ".ppm", and that extension: what is
// written beside it is named after it, in its format.
std::string image_stem(const char* path, std::string* ext) {
    std::string stem = path ? path : "build/rt";
    *ext = ".ppm";
    if (stem.size() > 4 && (stem.ends_with(".ppm") || stem.ends_with(".png"))) {
        *ext = stem.substr(stem.size() - 4);
        stem.resize(stem.size() - 4);
    }
    return stem;
}

// <stem>-<kind>-<base in hex><ext>
std::string image_beside(const std::string& stem, const char* kind, std::uint64_t base, const std::string& ext) {
    char hex[24];
    std::snprintf(hex, sizeof(hex), "%llx", static_cast<unsigned long long>(base));
    return stem + "-" + kind + "-" + hex + ext;
}

}  // namespace

bool host_gpu_dump_display(std::uint64_t display_va, const char* path, bool all_targets) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return false;
    flush_locked();
    // BBHOST_DUMP_ALL=1 keeps the default window of recorded draws;
    // BBHOST_DUMP_ALL=<n> lists the last n, up to the ring size, which is what
    // it takes to see a whole world frame.
    static const std::size_t dump_all_env = [] {
        const char* e = std::getenv("BBHOST_DUMP_ALL");
        if (!e || !e[0]) return std::size_t{0};
        const long v = std::strtol(e, nullptr, 10);
        return v <= 1 ? std::size_t{400} : std::min<std::size_t>(static_cast<std::size_t>(v), kDrawRecs);
    }();
    // BBHOST_DUMP_DRAWS=<n> is the draw list **without** the target images.
    // Reading back every colour target is what costs the memory - 140 of them
    // at a world frame - and the list is what says which draw wrote what, so
    // asking for one without the other is worth a switch of its own.
    static const std::size_t dump_draws_env = [] {
        const char* e = std::getenv("BBHOST_DUMP_DRAWS");
        if (!e || !e[0]) return std::size_t{0};
        const long v = std::strtol(e, nullptr, 10);
        return v <= 1 ? std::size_t{400} : std::min<std::size_t>(static_cast<std::size_t>(v), kDrawRecs);
    }();
    const std::size_t dump_all = all_targets && dump_all_env == 0 ? std::size_t{400} : dump_all_env;
    const std::size_t dump_list = dump_all ? dump_all : dump_draws_env;
    if (dump_list) {
        const std::uint64_t first = g_draw_rec_next > dump_list ? g_draw_rec_next - dump_list : 0;
        for (std::uint64_t i = first; i < g_draw_rec_next; ++i) {
            const DrawRec& r = g_draw_recs[i % kDrawRecs];
            host_log("  draw[%llu] %s n=%u inst=%u prim=%u vp=(%.0f,%.0f %.0fx%.0f) sc=(%d,%d %dx%d) rt0=0x%llx depth=0x%llx tex=0x%llx,0x%llx,0x%llx,0x%llx blend=%08x dctl=%08x mask=%x%s",
                     static_cast<unsigned long long>(i), r.name, r.count, r.inst, r.prim, r.vp[0], r.vp[1], r.vp[2], r.vp[3], r.sc[0],
                     r.sc[1], r.sc[2], r.sc[3], static_cast<unsigned long long>(r.rt0), static_cast<unsigned long long>(r.depth),
                     static_cast<unsigned long long>(r.tex[0]), static_cast<unsigned long long>(r.tex[1]), static_cast<unsigned long long>(r.tex[2]),
                     static_cast<unsigned long long>(r.tex[3]), r.blend0, r.depth_ctl, r.mask0, r.indexed ? " idx" : "");
        }
    }
    // F12 (all_targets): the draw list and every texture it sampled, into a
    // file beside the images - so a dump is readable without the run's log.
    // Each texture is described (T#; render target or cached surface, and
    // how that surface has been kept current) and each cached surface saved
    // as <stem>-tex-<base> (.png or .ppm, as the display's), which is what the
    // draws actually sampled.
    static const bool f12_textures = [] {
        const char* e = std::getenv("BBHOST_F12_TEXTURES");
        return e && e[0] == '1';
    }();
    static const bool f12_all_targets = [] {
        const char* e = std::getenv("BBHOST_F12_ALL_TARGETS");
        return e && e[0] == '1';
    }();
    const auto f12_start = std::chrono::steady_clock::now();
    std::string draws_path;
    char* draws_text = nullptr;
    std::size_t draws_len = 0;
    if (all_targets && dump_all) {
        std::string ext;
        const std::string stem = image_stem(path, &ext);
        const std::string txt = stem + "-draws.txt";
        draws_path = txt;
#if !defined(_WIN32)
        FILE* f = open_memstream(&draws_text, &draws_len);  // written with the images, off the lock
#else
        FILE* f = std::fopen(txt.c_str(), "w");
#endif
        if (f) {
            std::map<std::uint64_t, std::array<std::uint32_t, 8>> seen;
            // The whole ring, not the log's 400: a world frame's G-buffer pass
            // is further back than that.
            const std::uint64_t first = g_draw_rec_next > kDrawRecs ? g_draw_rec_next - kDrawRecs : 0;
            for (std::uint64_t i = first; i < g_draw_rec_next; ++i) {
                const DrawRec& r = g_draw_recs[i % kDrawRecs];
                std::fprintf(f, "draw[%llu] %s n=%u inst=%u prim=%u vp=(%.0f,%.0f %.0fx%.0f) rt0=0x%llx depth=0x%llx "
                                "blend=%08x dctl=%08x mask=%x%s ",
                             static_cast<unsigned long long>(i), r.name, r.count, r.inst, r.prim, r.vp[0], r.vp[1], r.vp[2],
                             r.vp[3], static_cast<unsigned long long>(r.rt0), static_cast<unsigned long long>(r.depth), r.blend0,
                             r.depth_ctl, r.mask0, r.indexed ? " idx" : "");
                std::fprintf(f, "smask=%08x colfmt=%08x ", r.shader_mask, r.col_format);
                for (int k = 0; k < 7; ++k) {
                    if (r.rt[k]) std::fprintf(f, "rt%d=0x%llx ", k + 1, static_cast<unsigned long long>(r.rt[k]));
                }
                std::fputs("tex=", f);
                for (int k = 0; k < 8; ++k) {
                    std::fprintf(f, "%s0x%llx", k ? "," : "", static_cast<unsigned long long>(r.tex[k]));
                    if (r.tex[k]) {
                        std::array<std::uint32_t, 8> w{};
                        std::memcpy(w.data(), r.tsharp[k], sizeof(r.tsharp[k]));
                        seen.emplace(r.tex[k], w);
                    }
                }
                std::fputc('\n', f);
            }
            std::fprintf(f, "\n%zu textures\n", seen.size());
            int saved = 0;
            for (const auto& kv : seen) {
                const std::uint32_t* w = kv.second.data();
                std::fprintf(f, "tex 0x%llx: T# dfmt %u nfmt %u %ux%u depth %u pitch %u type %u tiling %u levels %u..%u "
                                "array %u..%u [%08x %08x %08x %08x %08x %08x %08x %08x]\n",
                             static_cast<unsigned long long>(kv.first), (w[1] >> 20) & 0x3f, (w[1] >> 26) & 0xf,
                             (w[2] & 0x3fff) + 1, ((w[2] >> 14) & 0x3fff) + 1, (w[4] & 0x1fff) + 1, ((w[4] >> 13) & 0x3fff) + 1,
                             (w[3] >> 28) & 0xf, (w[3] >> 20) & 0x1f, (w[3] >> 12) & 0xf, (w[3] >> 16) & 0xf, w[5] & 0x1fff,
                             (w[5] >> 13) & 0x1fff, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                if (RtImage* rt = find_render_target(kv.first)) {
                    std::fprintf(f, "    render target vkformat %d %ux%u x%u layers%s%s\n", static_cast<int>(rt->format),
                                 rt->width, rt->height, rt->layers, rt->initialised ? "" : " (never drawn)",
                                 rt->depth ? " depth" : "");
                }
                const std::string d = texture_describe_locked(kv.first);
                if (!d.empty()) {
                    std::fprintf(f, "    %s\n", d.c_str());
                    if (f12_textures && saved < 400 && texture_dump_locked(kv.first, image_beside(stem, "tex", kv.first, ext).c_str())) {
                        ++saved;
                    }
                }
                // What touched its memory: the events that made it what it is.
                std::uint64_t span = texture_src_bytes_locked(kv.first);
                if (!span) span = static_cast<std::uint64_t>((w[2] & 0x3fff) + 1) * (((w[2] >> 14) & 0x3fff) + 1) * 4;
                tex_events_for(f, kv.first, span);
            }
            tex_events_write(f);
            std::fclose(f);
            host_log("dump: %s - %zu textures described, %d saved%s", txt.c_str(), seen.size(), saved,
                     f12_textures ? "" : " (BBHOST_F12_TEXTURES=1 saves them)");
        }
    }
    if (all_targets) {
        // The targets the recorded draws wrote, the display, and with
        // BBHOST_F12_ALL_TARGETS every colour target there is.
        std::string ext;
        const std::string stem = image_stem(path, &ext);
        std::set<std::uint64_t> want;
        if (f12_all_targets) {
            for (auto& kv : g_rts) want.insert(kv.first);
        } else {
            const std::uint64_t first = g_draw_rec_next > kDrawRecs ? g_draw_rec_next - kDrawRecs : 0;
            for (std::uint64_t i = first; i < g_draw_rec_next; ++i) {
                const DrawRec& r = g_draw_recs[i % kDrawRecs];
                if (r.rt0) want.insert(r.rt0);
                for (std::uint64_t t : r.rt) {
                    if (t) want.insert(t);
                }
            }
        }
        struct Pending {
            DevBuffer staging;
            std::unique_ptr<TargetPixels> t;
        };
        const auto t_list = std::chrono::steady_clock::now();
        std::vector<Pending> pending;
        auto add = [&](RtImage& rt, const std::string& p) {
            Pending q;
            q.t = std::make_unique<TargetPixels>();
            q.t->path = p;
            if (record_target_readback(rt, q.staging, *q.t)) pending.push_back(std::move(q));
        };
        for (std::uint64_t base : want) {
            auto it = g_rts.find(base);
            if (it == g_rts.end() || it->second.depth || !it->second.initialised) continue;
            add(it->second, image_beside(stem, "rt", base, ext));
        }
        auto disp = g_rts.find(display_va);
        const bool have_display = disp != g_rts.end() && disp->second.initialised;
        if (have_display) add(disp->second, path ? path : "build/f12.ppm");
        const auto t_record = std::chrono::steady_clock::now();
        flush_locked();  // one wait for every copy
        const auto t_flush = std::chrono::steady_clock::now();
        std::vector<std::unique_ptr<TargetPixels>> images;
        for (Pending& q : pending) {
            q.t->staging = q.staging;  // the writer reads it in place and frees it
            images.push_back(std::move(q.t));
        }
        const double held_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f12_start).count();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        host_log("dump: F12 held the renderer %.0f ms (draw list %.0f, copies %.0f, GPU %.0f, read %.0f); writing %zu images%s in the "
                 "background",
                 held_ms, ms(f12_start, t_list), ms(t_list, t_record), ms(t_record, t_flush), ms(t_flush, std::chrono::steady_clock::now()),
                 images.size(), draws_text ? " and the draw list" : "");
        // Where it all went, for the player to send: the capture's folder.
        const std::string folder = capture_dir_shown(std::filesystem::path(stem).parent_path().string());
        std::thread([images = std::move(images), draws_path, draws_text, draws_len, folder]() mutable {
            if (draws_text) {
                if (FILE* f = std::fopen(draws_path.c_str(), "w")) {
                    std::fwrite(draws_text, 1, draws_len, f);
                    std::fclose(f);
                }
                std::free(draws_text);
            }
            std::size_t failed = 0;
            for (const auto& t : images) {
                if (!write_target_image(*t)) ++failed;
                vkDestroyBuffer(g.device, t->staging.buffer, nullptr);
                vkFreeMemory(g.device, t->staging.memory, nullptr);
            }
            if (failed) host_log("dump: F12 could not write %zu of the %zu images", failed, images.size());
            host_log("dump: F12 files written to %s - send that folder with the log", folder.c_str());
        }).detach();
        if (!have_display) {
            host_log("render: no render target at display address 0x%llx to dump", static_cast<unsigned long long>(display_va));
        }
        return have_display;
    }
    if (dump_all) {  // BBHOST_DUMP_ALL without F12 (F12 returned above)
        for (auto& kv : g_rts) {
            if (kv.second.depth || !kv.second.initialised) continue;
            char p[64];
            std::snprintf(p, sizeof(p), "build/rt-%llx.ppm", static_cast<unsigned long long>(kv.first));
            dump_rt_locked(kv.second, p);
        }
    }
    // BBHOST_DUMP_TEXTURE=0xbase[,0xbase..]: the sampled surfaces at those
    // guest addresses (as the shaders see them) to build/tex-<base>.ppm.
    static const std::vector<std::uint64_t> tex_bases = [] {
        std::vector<std::uint64_t> v;
        const char* e = std::getenv("BBHOST_DUMP_TEXTURE");
        while (e && *e) {
            char* end = nullptr;
            const unsigned long long b = std::strtoull(e, &end, 0);
            if (end == e) break;
            v.push_back(b);
            e = *end == ',' ? end + 1 : end;
        }
        return v;
    }();
    for (std::uint64_t b : tex_bases) {
        char p[96];
        std::snprintf(p, sizeof(p), "build/tex-%llx.ppm", static_cast<unsigned long long>(b));
        if (!texture_dump_locked(b, p)) host_log("render: no sampled surface at 0x%llx to dump", static_cast<unsigned long long>(b));
        // BBHOST_DUMP_TEXTURE_LEVELS=1: every level and layer too, as -l<level>-s<layer>.
        if (std::getenv("BBHOST_DUMP_TEXTURE_LEVELS")) {
            for (std::uint32_t lv = 0; lv < 16; ++lv) {
                for (std::uint32_t ly = 0; ly < 6; ++ly) {
                    char q[128];
                    std::snprintf(q, sizeof(q), "build/tex-%llx-l%u-s%u.ppm", static_cast<unsigned long long>(b), lv, ly);
                    texture_dump_locked(b, q, lv, ly);
                }
            }
        }
    }
    auto it = g_rts.find(display_va);
    if (it == g_rts.end() || !it->second.initialised) {
        host_log("render: no render target at display address 0x%llx to dump", static_cast<unsigned long long>(display_va));
        return false;
    }
    return dump_rt_locked(it->second, path);
}

namespace {
float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    std::uint32_t bits;
    if (exp == 0) bits = (sign << 31) | (man ? (127 - 15 - 9 + 32 - __builtin_clz(man)) << 23 | ((man << (__builtin_clz(man) - 8)) & 0x7fffff) : 0);
    else if (exp == 31) bits = (sign << 31) | 0x7f800000 | (man << 13);
    else bits = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
std::uint8_t tone8(float v) {
    if (!(v > 0.f)) return 0;
    if (v >= 1.f) return 255;
    return static_cast<std::uint8_t>(v * 255.f + 0.5f);
}
}  // namespace

// The largest finite value of each colour channel of a float target
// (RGBA16F or R11G11B10); waits for the GPU. BBHOST_TRACE_TARGET's readback.
bool gpu::rt_float_max_locked(RtImage& r, float* out3) {
    const bool half4 = r.format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (r.depth || (!half4 && r.format != VK_FORMAT_B10G11R11_UFLOAT_PACK32)) return false;
    const std::size_t bpp = rt_bytes_per_pixel(r);
    DevBuffer staging;
    if (!create_dev_buffer(staging, static_cast<std::uint64_t>(r.width) * r.height * bpp, true)) return false;
    begin_recording_locked();
    render_end_pass_locked();
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {r.width, r.height, 1};
    vkCmdCopyImageToBuffer(g_cmd(), r.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
    flush_locked();
    const auto* px = static_cast<const std::uint8_t*>(staging.map);
    out3[0] = out3[1] = out3[2] = 0.f;
    auto note = [&](int k, float v) {
        if (std::isfinite(v)) out3[k] = std::max(out3[k], v);
    };
    for (std::size_t i = 0, n = static_cast<std::size_t>(r.width) * r.height; i < n; ++i) {
        const std::uint8_t* p = px + i * bpp;
        if (half4) {
            std::uint16_t h[3];
            std::memcpy(h, p, 6);
            for (int k = 0; k < 3; ++k) note(k, half_to_float(h[k]));
        } else {
            std::uint32_t v;
            std::memcpy(&v, p, 4);
            note(0, half_to_float(static_cast<std::uint16_t>((v & 0x7ff) << 4)));
            note(1, half_to_float(static_cast<std::uint16_t>(((v >> 11) & 0x7ff) << 4)));
            note(2, half_to_float(static_cast<std::uint16_t>(((v >> 22) & 0x3ff) << 5)));
        }
    }
    vkDestroyBuffer(g.device, staging.buffer, nullptr);
    vkFreeMemory(g.device, staging.memory, nullptr);
    return true;
}

namespace {

// Records the copy of r into a new host-visible staging buffer; the caller
// flushes once for all of them and takes the bytes (take_target_pixels).
bool record_target_readback(RtImage& r, DevBuffer& staging, TargetPixels& out) {
    const bool depth = r.depth;
    if (depth && r.format != VK_FORMAT_D32_SFLOAT && r.format != VK_FORMAT_D32_SFLOAT_S8_UINT) return false;
    out.width = r.width;
    out.height = r.height;
    out.format = r.format;
    out.depth = depth;
    out.bpp = depth ? 4 : rt_bytes_per_pixel(r);
    if (!create_dev_buffer(staging, static_cast<std::uint64_t>(r.width) * r.height * out.bpp, true, true)) return false;
    begin_recording_locked();
    render_end_pass_locked();
    VkBufferImageCopy region{};
    region.imageSubresource = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {r.width, r.height, 1};
    vkCmdCopyImageToBuffer(g_cmd(), r.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
    return true;
}

void take_target_pixels(DevBuffer& staging, TargetPixels& t) {
    const auto* px = static_cast<const std::uint8_t*>(staging.map);
    t.px.assign(px, px + static_cast<std::size_t>(t.width) * t.height * t.bpp);
    vkDestroyBuffer(g.device, staging.buffer, nullptr);
    vkFreeMemory(g.device, staging.memory, nullptr);
}

// No Vulkan and no renderer state: safe off the renderer's lock. PNG when
// the path ends in ".png", PPM otherwise (core/image_file.h).
bool write_target_image(const TargetPixels& t) {
    const VkFormat fmt = t.format;
    const bool bgra = fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB;
    const bool rgba8 = bgra || fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_SRGB;
    ImageFile f;
    if (!f.open(t.path, t.width, t.height)) {
        host_log("render: cannot write %s", t.path.c_str());
        return false;
    }
    // Float targets: what the 8-bit image clamps away - each channel's largest
    // finite value and how many texels are infinite or NaN.
    float fmax[3] = {0.f, 0.f, 0.f};
    std::uint64_t finf[3] = {}, fnan[3] = {};
    auto note_float = [&](int k, float v) {
        if (std::isnan(v)) ++fnan[k];
        else if (std::isinf(v)) ++finf[k];
        else fmax[k] = std::max(fmax[k], v);
    };
    std::vector<std::uint8_t> row(static_cast<std::size_t>(t.width) * 3);
    for (std::uint32_t y = 0; y < t.height; ++y) {
        for (std::uint32_t x = 0; x < t.width; ++x) {
            const std::uint8_t* p = t.data() + (static_cast<std::size_t>(y) * t.width + x) * t.bpp;
            if (rgba8) {
                row[x * 3 + 0] = bgra ? p[2] : p[0];
                row[x * 3 + 1] = p[1];
                row[x * 3 + 2] = bgra ? p[0] : p[2];
            } else if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
                std::uint16_t h[3];
                std::memcpy(h, p, 6);
                for (int k = 0; k < 3; ++k) {
                    note_float(k, half_to_float(h[k]));
                    row[x * 3 + k] = tone8(half_to_float(h[k]));
                }
            } else if (fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                std::uint32_t v;
                std::memcpy(&v, p, 4);
                // 11-bit floats: 5 exp, 6 mantissa; 10-bit: 5 exp, 5 mantissa
                auto f11 = [](std::uint32_t b) { return half_to_float(static_cast<std::uint16_t>((b & 0x7ff) << 4)); };
                auto f10 = [](std::uint32_t b) { return half_to_float(static_cast<std::uint16_t>((b & 0x3ff) << 5)); };
                note_float(0, f11(v));
                note_float(1, f11(v >> 11));
                note_float(2, f10(v >> 22));
                row[x * 3 + 0] = tone8(f11(v));
                row[x * 3 + 1] = tone8(f11(v >> 11));
                row[x * 3 + 2] = tone8(f10(v >> 22));
            } else if (fmt == VK_FORMAT_R32_SFLOAT || t.depth) {
                float fv;
                std::memcpy(&fv, p, 4);
                row[x * 3 + 0] = row[x * 3 + 1] = row[x * 3 + 2] = tone8(fv);
            } else {
                row[x * 3 + 0] = row[x * 3 + 1] = row[x * 3 + 2] = p[0];
            }
        }
        f.row(row.data());
    }
    if (!f.close()) {
        host_log("render: writing %s failed", t.path.c_str());
        return false;
    }
    host_log("render: wrote %s (%ux%u, format %d)", t.path.c_str(), t.width, t.height, fmt);
    if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT || fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        host_log("render: %s float channels: max %g %g %g, infinite %llu %llu %llu, NaN %llu %llu %llu", t.path.c_str(), fmax[0], fmax[1],
                 fmax[2], static_cast<unsigned long long>(finf[0]), static_cast<unsigned long long>(finf[1]),
                 static_cast<unsigned long long>(finf[2]), static_cast<unsigned long long>(fnan[0]),
                 static_cast<unsigned long long>(fnan[1]), static_cast<unsigned long long>(fnan[2]));
    }
    return true;
}

}  // namespace

bool gpu::dump_rt_locked(RtImage& r, const char* path) {
    DevBuffer staging;
    TargetPixels t;
    t.path = path;
    if (!record_target_readback(r, staging, t)) return false;
    flush_locked();
    take_target_pixels(staging, t);
    return write_target_image(t);
}

namespace gpu {

// A loading screen up or down, and the first in-game frame (engine/loading.cpp
// through gpu.cpp): the precompile workers' number (g_precompile_boost).
void precompile_set_loading(bool loading) {
    g_loading_now.store(loading);
    precompile_boost_update(loading ? "a loading screen" : "the loading screen gone");
}
void precompile_set_world_reached() {
    g_world_seen.store(true);
    precompile_boost_update("the first in-game frame");
}

// At the end of device init: the manifest's stages go to the precompile
// workers, ahead of anything GX creates.
void stage_manifest_load(const std::string& path) {
    if (!stage_manifest_on() || !precompile_enabled() || path.empty()) return;
    std::ifstream in(path, std::ios::binary);
    const std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    manifest_file::Reader r{data.data(), data.data() + data.size()};
    if (data.size() < 12 || r.u32() != manifest_file::kMagic || r.u32() != manifest_file::kVersion) return;
    const std::uint32_t n = r.u32();
    std::vector<std::shared_ptr<const ManifestStage>> stages;
    for (std::uint32_t k = 0; k < n && k < kManifestMax; ++k) {
        auto m = std::make_shared<ManifestStage>();
        if (!manifest_file::read(r, *m)) break;
        stages.push_back(std::move(m));
    }
    Precompiler& w = precompiler();
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        for (const auto& m : stages) g_manifest.by_key.emplace(m->key(), m);
    }
    {
        std::lock_guard<std::mutex> lk(w.mu);
        for (auto it = stages.rbegin(); it != stages.rend(); ++it) {
            PrecompileJob job;
            job.name = (*it)->name;
            job.type = 7;
            job.manifest = *it;
            w.jobs.push_front(std::move(job));
        }
    }
    g_manifest.loaded.store(stages.size());
    w.cv.notify_all();
    host_log("render: stage manifest: %zu stages from earlier runs queued for their compile", stages.size());
}

// The whole manifest, when stages were recorded since the last write.
void stage_manifest_save(const std::string& path) {
    if (!stage_manifest_on() || path.empty()) return;
    static std::mutex writing;  // the report's thread and the exit's
    std::lock_guard<std::mutex> wl(writing);
    manifest_file::Writer out;
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (!g_manifest.dirty) return;
        g_manifest.dirty = false;
        out.u32(manifest_file::kMagic);
        out.u32(manifest_file::kVersion);
        out.u32(static_cast<std::uint32_t>(g_manifest.by_key.size()));
        for (int pass = 0; pass < 2; ++pass) {  // stages, then paths: a reader that knows only stages stops at the first paths entry
            for (const auto& [key, m] : g_manifest.by_key) {
                if ((m->stage >= 2) == (pass == 1)) manifest_file::write(out, *m);
            }
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    bool ok = false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(out.out.data()), static_cast<std::streamsize>(out.out.size()));
        ok = static_cast<bool>(f);
    }
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) host_log("render: saving the stage manifest to %s failed", path.c_str());
}

// With every 300-flip report, on a thread of its own: a stage recorded is
// kept even if the run never exits cleanly (the pipeline cache's own save
// waits for three minutes and a few MiB of growth).
void stage_manifest_save_async(const std::string& path) {
    static std::atomic<bool> saving{false};
    if (!stage_manifest_on() || path.empty()) return;
    {
        std::lock_guard<std::mutex> lk(g_manifest.mu);
        if (!g_manifest.dirty) return;
    }
    if (saving.exchange(true)) return;
    std::thread([path] {
        stage_manifest_save(path);
        saving.store(false);
    }).detach();
}

}  // namespace gpu

std::string host_gpu_ps_wave_report() { return gpu::ps_wave_census(true); }

bool host_gpu_draw_window(std::uint64_t dst, const void* data, std::uint32_t bytes) {
    static const bool gpu_writes = [] {
        const char* e = std::getenv("BBHOST_YEBIS_WINDOWS");
        return e && std::strcmp(e, "gpu") == 0;
    }();
    if (gpu_writes || !bytes || bytes > (64u << 10) || !hle_kernel_va_mapped(dst, bytes)) return false;
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), data, bytes);
    DrawWindow w;
    w.dst = dst;
    w.bytes = bytes;
    t_given_windows.push_back(w);
    g_windows_given.fetch_add(1, std::memory_order_relaxed);
    return true;
}
