// Whether a draw writes every pixel of a colour target (host/full_write.h):
// each state rule alone, then the geometry - a rectangle list, a strip and a
// list of two triangles, one large triangle - covering the square, short of it
// by a hair, crossed, degenerate, behind the eye and past the depth range.
#include "host/full_write.h"

#include <cstdio>
#include <initializer_list>

namespace {

int g_failures = 0;
#define CHECK_EQ(a, b)                                                                                                   \
    do {                                                                                                                 \
        const int a_ = (a), b_ = (b);                                                                                    \
        if (a_ != b_) {                                                                                                  \
            std::fprintf(stderr, "%s:%d: %s is %s (%d), wanted %s (%d)\n", __FILE__, __LINE__, #a, full_write::fail_name(a_), a_, \
                         full_write::fail_name(b_), b_);                                                                 \
            ++g_failures;                                                                                                \
        }                                                                                                                \
    } while (0)

using full_write::Input;

// A full-target draw that passes every state rule: 1920x1080, every channel, the viewport and scissor its size.
Input base() {
    Input in;
    in.width = 1920;
    in.height = 1080;
    in.mask = 0xf;
    in.channels = 0xf;
    in.vp_w = 1920;
    in.vp_h = 1080;
    in.sc_w = 1920;
    in.sc_h = 1080;
    in.known_vs = true;
    return in;
}
void verts(Input& in, int prim, std::initializer_list<std::initializer_list<float>> v) {
    in.prim = prim;
    in.vertices = 0;
    for (const auto& p : v) {
        int k = 0;
        for (float c : p) in.pos[in.vertices][k++] = c;
        if (k == 2) {
            in.pos[in.vertices][2] = 0.5f;
            in.pos[in.vertices][3] = 1.0f;
        }
        ++in.vertices;
    }
}

void state_rules() {
    Input in = base();
    verts(in, full_write::kPrimRectList, {{-1, -1}, {1, -1}, {-1, 1}});
    CHECK_EQ(full_write::check(in), full_write::kNone);
    { Input t = in; t.mask = 0x7; CHECK_EQ(full_write::check(t), full_write::kMask); }
    { Input t = in; t.mask = 0x7; t.channels = 0x7; CHECK_EQ(full_write::check(t), full_write::kNone); }  // an RGB format
    { Input t = in; t.blend = true; CHECK_EQ(full_write::check(t), full_write::kBlend); }
    { Input t = in; t.depth_rejects = true; CHECK_EQ(full_write::check(t), full_write::kDepth); }
    { Input t = in; t.stencil = true; CHECK_EQ(full_write::check(t), full_write::kStencil); }
    { Input t = in; t.bounds = true; CHECK_EQ(full_write::check(t), full_write::kBounds); }
    { Input t = in; t.coverage = true; CHECK_EQ(full_write::check(t), full_write::kCoverage); }
    { Input t = in; t.discard = true; CHECK_EQ(full_write::check(t), full_write::kDiscard); }
    { Input t = in; t.branches = true; CHECK_EQ(full_write::check(t), full_write::kBranches); }
    { Input t = in; t.raster_kill = true; CHECK_EQ(full_write::check(t), full_write::kRasterKill); }
    { Input t = in; t.cull = true; CHECK_EQ(full_write::check(t), full_write::kCull); }
    { Input t = in; t.solid = false; CHECK_EQ(full_write::check(t), full_write::kFill); }
    { Input t = in; t.layers = 6; CHECK_EQ(full_write::check(t), full_write::kLayers); }
    { Input t = in; t.vp_w = 1919; CHECK_EQ(full_write::check(t), full_write::kViewport); }
    { Input t = in; t.vp_x = 1; CHECK_EQ(full_write::check(t), full_write::kViewport); }
    { Input t = in; t.vp_y = 1080; t.vp_h = -1080; CHECK_EQ(full_write::check(t), full_write::kNone); }  // a flipped viewport
    { Input t = in; t.vp_x = -10; t.vp_w = 1940; CHECK_EQ(full_write::check(t), full_write::kNone); }   // larger than the target
    { Input t = in; t.sc_h = 1079; CHECK_EQ(full_write::check(t), full_write::kScissor); }
    { Input t = in; t.sc_y = 1; CHECK_EQ(full_write::check(t), full_write::kScissor); }
    { Input t = in; t.known_vs = false; CHECK_EQ(full_write::check(t), full_write::kShader); }
    { Input t = in; t.simple = false; CHECK_EQ(full_write::check(t), full_write::kPrimitive); }
    // The render area (the smallest attachment), a texture at the target, a pipeline variant not built, bytes not known.
    { Input t = in; t.area_w = 1919; CHECK_EQ(full_write::check(t), full_write::kArea); }
    { Input t = in; t.area_h = 1079; CHECK_EQ(full_write::check(t), full_write::kArea); }
    { Input t = in; t.area_w = 3840; t.area_h = 2160; CHECK_EQ(full_write::check(t), full_write::kNone); }  // larger attachments
    { Input t = in; t.feedback = true; CHECK_EQ(full_write::check(t), full_write::kFeedback); }
    { Input t = in; t.variant_unknown = true; CHECK_EQ(full_write::check(t), full_write::kVariant); }
    { Input t = in; t.bytes_unknown = true; CHECK_EQ(full_write::check(t), full_write::kBytes); }
    {
        Input t = in;  // bytes not known and the geometry short: the geometry is said first
        t.bytes_unknown = true;
        verts(t, full_write::kPrimRectList, {{-1, -1}, {0.5f, -1}, {-1, 1}});
        CHECK_EQ(full_write::check(t), full_write::kGeometry);
    }
}

void geometry_cases() {
    Input in = base();
    // Rectangle lists.
    verts(in, full_write::kPrimRectList, {{-1, -1}, {1, -1}, {-1, 1}});
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimRectList, {{-1, 1}, {1, 1}, {-1, -1}});  // the other way up
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimRectList, {{-1, -1}, {0.9999f, -1}, {-1, 1}});  // a hair short
    CHECK_EQ(full_write::check(in), full_write::kGeometry);
    verts(in, full_write::kPrimRectList, {{-2, -2}, {2, -2}, {-2, 2}});  // larger
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimRectList, {{-1, -1}, {1, -1}, {1, -1}});  // degenerate
    CHECK_EQ(full_write::check(in), full_write::kGeometry);
    // Strips of two triangles.
    verts(in, full_write::kPrimTriangleStrip, {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}});
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimTriangleStrip, {{-1, -1}, {1, 1}, {1, -1}, {-1, 1}});  // crossed: the shared edge is a side
    CHECK_EQ(full_write::check(in), full_write::kGeometry);
    verts(in, full_write::kPrimTriangleStrip, {{-1, -1}, {1, -1}, {-1, 0.5f}, {1, 0.5f}});  // the top left out
    CHECK_EQ(full_write::check(in), full_write::kGeometry);
    // Lists.
    verts(in, full_write::kPrimTriangleList, {{-1, -1}, {1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 1}});
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimTriangleList, {{-1, -1}, {1, -1}, {-1, 1}, {1, -1}, {1, 1}, {0, 1}});  // a gap at the top
    CHECK_EQ(full_write::check(in), full_write::kGeometry);
    verts(in, full_write::kPrimTriangleList, {{-1, -1}, {3, -1}, {-1, 3}});  // one large triangle
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimTriangleList, {{-1, -1}, {1, -1}, {-1, 1}});  // half the square
    CHECK_EQ(full_write::check(in), full_write::kGeometry);
    verts(in, full_write::kPrimTriangleList, {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}});  // four vertices of a list
    CHECK_EQ(full_write::check(in), full_write::kPrimitive);
    verts(in, full_write::kPrimOther, {{-1, -1}, {1, -1}, {-1, 1}});
    CHECK_EQ(full_write::check(in), full_write::kPrimitive);
    // Clip space: w and the depth range.
    verts(in, full_write::kPrimRectList, {{-2, -2, 0.5f, 2}, {2, -2, 0.5f, 2}, {-2, 2, 0.5f, 2}});  // w = 2
    CHECK_EQ(full_write::check(in), full_write::kNone);
    verts(in, full_write::kPrimRectList, {{-1, -1, 0.5f, 0}, {1, -1, 0.5f, 1}, {-1, 1, 0.5f, 1}});
    CHECK_EQ(full_write::check(in), full_write::kClip);
    verts(in, full_write::kPrimRectList, {{-1, -1, 1.5f, 1}, {1, -1, 0.5f, 1}, {-1, 1, 0.5f, 1}});
    CHECK_EQ(full_write::check(in), full_write::kClip);
    in.depth_clamp = true;
    CHECK_EQ(full_write::check(in), full_write::kNone);
    in.depth_clamp = false;
    verts(in, full_write::kPrimRectList, {{-1, -1, -0.1f, 1}, {1, -1, 0.5f, 1}, {-1, 1, 0.5f, 1}});
    CHECK_EQ(full_write::check(in), full_write::kClip);
}

}  // namespace

int main() {
    state_rules();
    geometry_cases();
    if (g_failures) {
        std::fprintf(stderr, "full_write_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("full_write_test: ok\n");
    return 0;
}
