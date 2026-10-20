// host/vertex_plan.h without a GPU: a pipeline recipe recorded where draws
// pack their 8- and 16-bit vertex elements (fast-linked pipeline libraries,
// formats from the params), replayed where draws do not pack them (formats as
// constants: no libraries, BBHOST_VERTEX_FORMAT_PARAMS=0, BBHOST_VERTEX_PACK=0).
// The replayed plan, taken apart again, must be the plan such a draw makes -
// its elements and attributes byte for byte, which is all of the pipeline's
// vertex identity besides the bindings - and the vertex shader stage key of a
// packed element must differ from an unpacked one's where formats are
// constants (a stage that reads packed dwords must never serve a draw that
// gives one component a channel), while an unpacked element's key stays its
// word 3, as before the packing was keyed (no key a default run makes moves),
// and a stage reading its formats from the params keys neither.
#include "host/vertex_plan.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

namespace {

// A V# word 3: data and number format, DST_SEL x y z w.
std::uint32_t word3(std::uint32_t dfmt, std::uint32_t nfmt) { return dfmt << 15 | nfmt << 12 | 4u | 5u << 3 | 6u << 6 | 7u << 9; }

struct Element {
    std::uint32_t dfmt, nfmt, packed;  // packed: what pack_vertex_input gives the element where draws pack
    VkFormat packed_format;
};
// A skinned mesh's record as the game's vertex layouts have it: a float3
// position, an 8_8_8_8 snorm normal, 16_16 float texture coordinates, 8_8_8_8
// uint bone indices and unorm weights, 16_16_16_16 snorm tangent.
const Element kMesh[] = {
    {13, 7, 0, VK_FORMAT_UNDEFINED},
    {10, 1, 8, VK_FORMAT_R32_UINT},
    {5, 7, 16, VK_FORMAT_R32_UINT},
    {10, 4, 8, VK_FORMAT_R32_UINT},
    {10, 0, 8, VK_FORMAT_R32_UINT},
    {12, 1, 16, VK_FORMAT_R32G32_UINT},
};

struct Plan {
    std::vector<gcn::VertexElement> elements;
    std::vector<VkVertexInputAttributeDescription> attributes;
};
Plan plan(bool packed) {
    Plan p;
    std::uint32_t offset = 0, vgpr = 4;
    for (std::uint32_t k = 0; k < sizeof(kMesh) / sizeof(kMesh[0]); ++k) {
        const Element& e = kMesh[k];
        gcn::VertexElement el;
        el.location = k;
        el.vdata = vgpr;
        el.count = 4;
        el.w3 = word3(e.dfmt, e.nfmt);
        el.packed = packed ? e.packed : 0;
        p.elements.push_back(el);
        p.attributes.push_back({k, 0, packed && e.packed ? e.packed_format : gpu::vertex_format(e.dfmt), offset});
        offset += gpu::vertex_format_bytes(e.dfmt);
        vgpr += 4;
    }
    return p;
}
bool same(const Plan& a, const Plan& b) {
    return a.elements.size() == b.elements.size() && a.attributes.size() == b.attributes.size() &&
           std::memcmp(a.elements.data(), b.elements.data(), a.elements.size() * sizeof(gcn::VertexElement)) == 0 &&
           std::memcmp(a.attributes.data(), b.attributes.data(), a.attributes.size() * sizeof(VkVertexInputAttributeDescription)) == 0;
}

}  // namespace

int main() {
    // The formats one component a channel takes and their sizes.
    CHECK(gpu::vertex_format(10) == VK_FORMAT_R8G8B8A8_UINT && gpu::vertex_format(5) == VK_FORMAT_R16G16_UINT);
    CHECK(gpu::vertex_format(12) == VK_FORMAT_R16G16B16A16_UINT && gpu::vertex_format(13) == VK_FORMAT_R32G32B32_UINT);
    CHECK(gpu::vertex_format(7) == VK_FORMAT_R32_UINT && gpu::vertex_format(0) == VK_FORMAT_UNDEFINED);
    CHECK(gpu::vertex_format_bytes(10) == 4 && gpu::vertex_format_bytes(12) == 8 && gpu::vertex_format_bytes(13) == 12);

    // The recipe recorded packed, replayed where draws do not pack: the draw's own plan.
    Plan recorded = plan(true);
    const Plan draw = plan(false);
    CHECK(!same(recorded, draw));
    CHECK(gpu::unpack_vertex_attributes(recorded.elements, recorded.attributes));
    CHECK(same(recorded, draw));
    CHECK(!gpu::unpack_vertex_attributes(recorded.elements, recorded.attributes));  // nothing left to take apart
    Plan unpacked = draw;
    CHECK(!gpu::unpack_vertex_attributes(unpacked.elements, unpacked.attributes) && same(unpacked, draw));

    // The stage key's words.
    const Plan packed = plan(true);
    for (std::size_t k = 0; k < packed.elements.size(); ++k) {
        const gcn::VertexElement& p = packed.elements[k];
        const gcn::VertexElement& u = draw.elements[k];
        // An unpacked element's word is its word 3, as before: a default run's keys do not move.
        CHECK(gpu::vertex_stage_key_word(u, false, true) == u.w3);
        CHECK(gpu::vertex_stage_key_word(u, false, false) == u.w3);
        // Constant formats: packed and unpacked stages never share a key.
        if (p.packed) CHECK(gpu::vertex_stage_key_word(p, false, true) != gpu::vertex_stage_key_word(u, false, true));
        else CHECK(gpu::vertex_stage_key_word(p, false, true) == gpu::vertex_stage_key_word(u, false, true));
        // As before the packing was keyed (BBHOST_VERTEX_KEY_PACKING=0), they did: the collision.
        CHECK(gpu::vertex_stage_key_word(p, false, false) == gpu::vertex_stage_key_word(u, false, false));
        // Formats from the params: the descriptor says how an element arrives; no word.
        CHECK(gpu::vertex_stage_key_word(p, true, true) == 0 && gpu::vertex_stage_key_word(u, true, true) == 0);
    }

    if (g_failures) {
        std::fprintf(stderr, "vertex_plan_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("vertex_plan_test ok: a recipe recorded packed, replayed where draws do not pack, is the draw's plan byte for byte; "
                "constant-format stages of packed and unpacked elements never share a key, an unpacked element's key is its word 3 "
                "as before, params-format stages key neither\n");
    return 0;
}
