// GX's render-state objects as host objects.
//
// GX makes blend, depth-stencil and rasterizer state the way D3D11 does: a
// description goes through a builder (0x2578050 blend, 0x25782b0
// depth-stencil, 0x2578550 rasterizer) into an immutable object, and a draw
// binds it by pointer (pending state +0x4d0, +0x4d8, +0x240). The builder
// encodes Gnm registers into the object, and the command processor has been
// decoding those registers back at every draw. This registry keeps what the
// game asked for instead: the description, which each builder also copies
// into the object, parsed into the fields below under an id the draw carries.
//
// Objects are identified by content, at the first draw that binds them.
// Hooking the creators (0x25658e0, 0x25659a0, 0x2565a80) sees almost none
// of them: the engine calls the builders itself on temporaries and copies the
// results where its draws bind them (0x25aeca0 alone builds 15 blend, 6
// depth-stencil and 12 rasterizer states), and GX builds its own defaults in
// place (0x2ad3280, 0x2ad3a00). So the recording thread, at the draw call,
// fingerprints the object's register words and looks its address up; a new
// or changed object is parsed from its own copy of the description.
//
// GX's enums are D3D11's, 0-based (the tables at 0x4c6fc50, 0x4c6fca0,
// 0x4c6fcc0, 0x4c6fe20): blend factors ZERO..INV_SRC1_ALPHA without D3D11's
// gap; ops ADD SUBTRACT REV_SUBTRACT MIN MAX, stencil ops KEEP..DECR and
// compare NEVER..ALWAYS in Vulkan's order; fill 0 solid, cull 0/1/2 none,
// front, back.
//
// An object's address is a lookup key only while the object is alive, and
// the command processor draws long after the game recorded the draw, when
// the address may belong to a newer object. So the id is taken at the call
// (gx_trace.cpp's snapshot) and travels with the draw. Ids name interned
// descriptions: identical states share one, and none is ever reused.
//
// BBHOST_GX_STATE=0 leaves it off; =2 (compare) checks, at every draw that
// carries ids, the Vulkan state the descriptions give against what the
// register path gives (render.cpp) and reports the differences at exit.
#pragma once

#include <cstdint>
#include <string>

struct ElfImage;

struct GxBlendTarget {
    std::uint8_t enable = 0, mask = 0;
    std::uint8_t src = 0, dst = 0, op = 0, src_alpha = 0, dst_alpha = 0, op_alpha = 0;
};
struct GxBlendDesc {
    std::uint8_t alpha_to_coverage = 0, independent = 0;
    GxBlendTarget rt[8];
};
struct GxStencilFace {
    std::uint8_t fail = 0, depth_fail = 0, pass = 0, func = 0;
};
struct GxDepthStencilDesc {
    std::uint8_t depth_enable = 0, depth_write = 0, depth_func = 0, bounds = 0;
    std::uint8_t stencil_enable = 0, read_mask = 0, write_mask = 0, pad = 0;
    GxStencilFace front, back;
};
struct GxRasterDesc {
    std::uint8_t fill = 0, cull = 0, front_ccw = 0, depth_clip = 0, scissor = 0, multisample = 0, pad[2] = {};
    std::int32_t depth_bias = 0;
    float bias_clamp = 0.0f, slope_bias = 0.0f;
};

void gx_state_install(ElfImage* image);
// 0 off, 1 on, 2 compare (BBHOST_GX_STATE).
int gx_state_mode();
// A state object's id at the draw call, where the object is known to be
// alive; 0 when the registry never saw it created.
std::uint32_t gx_blend_id(std::uint64_t object);
std::uint32_t gx_depth_stencil_id(std::uint64_t object);
std::uint32_t gx_raster_id(std::uint64_t object);
// A plugin's own draws restyled (plugin API 16), on this thread until set back
// to 0, as the or of: kGxDrawLines - the rasterizer state's copy that draws
// lines (fill 1), pulled toward the viewer by a depth bias so the lines win
// over the faces they outline, and the depth not written; kGxDrawBlack - the
// blend state's copy that writes zero to every target; kGxDrawSubtract - the
// blend state's copy that writes the draw's colour less the target's, once a
// pixel (pulled toward the viewer a little, tested less-than, depth written).
constexpr int kGxDrawLines = 1, kGxDrawBlack = 2, kGxDrawSubtract = 4;
void gx_draw_style(int style);
// The description behind an id; null for 0 or an unknown id. Stable for the run.
const GxBlendDesc* gx_blend_desc(std::uint32_t id);
const GxDepthStencilDesc* gx_depth_stencil_desc(std::uint32_t id);
const GxRasterDesc* gx_raster_desc(std::uint32_t id);
// Exit report: objects created, distinct descriptions, lookups.
std::string gx_state_report();
