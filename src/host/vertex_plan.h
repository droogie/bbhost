#pragma once
// What a vertex input plan's identity is made of, where a pipeline recorded in
// one run must find the stage a draw of another builds (render.cpp
// plan_vertex_input, pack_vertex_input, replan_recorded_vertex_input,
// vs_stage_key): pure functions, unit-tested without a GPU
// (tests/vertex_plan_test.cpp).
#include "gcn/translate.h"

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace gpu {

// Vulkan vertex input: the unsigned format that holds a V# data format's raw
// components (packed formats as one dword), so the vertex shader converts them
// as the fetch shader did; UNDEFINED for unknown formats. One component a
// channel: an element pack_vertex_input packs arrives instead as an R16, R32
// or R32G32 UINT attribute (gcn::VertexElement::packed).
inline VkFormat vertex_format(std::uint32_t dfmt) {
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
inline std::uint32_t vertex_format_bytes(std::uint32_t dfmt) {
    switch (dfmt) {
        case 1: return 1;
        case 2: case 3: return 2;
        case 11: case 12: return 8;
        case 13: return 12;
        case 14: return 16;
        default: return 4;
    }
}

// The word a vertex shader stage's key holds for an element (vs_stage_key):
// none where its formats come from the params (the descriptor there says how
// it arrives), else its word 3 - and, with `key_packing`, how it arrives in
// the bits word 3 leaves free (24 up), since a translation that takes it
// packed reads its attribute's dwords apart and one that takes it one
// component a channel does not. An unpacked element's word is its word 3
// alone, as before.
inline std::uint32_t vertex_stage_key_word(const gcn::VertexElement& el, bool formats_from_params, bool key_packing) {
    if (formats_from_params) return 0;
    return el.w3 | (key_packing ? el.packed << 24 : 0u);
}

// A plan's packed elements taken apart again, one component a channel, as a
// draw that does not pack plans them: each element's packing cleared and its
// attribute given vertex_format's format. True when one was packed.
inline bool unpack_vertex_attributes(std::vector<gcn::VertexElement>& elements, std::vector<VkVertexInputAttributeDescription>& attributes) {
    bool any = false;
    for (VkVertexInputAttributeDescription& a : attributes) {
        if (a.location >= elements.size()) continue;
        gcn::VertexElement& el = elements[a.location];
        if (!el.packed) continue;
        el.packed = 0;
        a.format = vertex_format((el.w3 >> 15) & 0xf);
        any = true;
    }
    return any;
}

}  // namespace gpu
