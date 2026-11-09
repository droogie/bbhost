// host/render_api.h (plugin API 27's frame info): a frame's camera from the
// view block's terms - a known camera with a 43-degree vertical field of view
// at 16:9, depth 0 at 0.05 m and 1 at 3013 m (the game's, as its log says) -
// checked against the matrices' own definitions and against the host's own
// decode of a pixel (motion_at.glsl's, in C++ here): the basis and eye, the
// view's inverse, a point's clip position, its depth and its pixel, the way
// back from a pixel and its depth to the world, the jittered projection, and
// a depth that runs the other way. Then the cut test's thresholds, the
// reset's reasons, and the generations the render hooks hand out.
#include "host/render_api.h"

#include <cmath>
#include <cstdio>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

constexpr double kPi = 3.14159265358979323846;

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// m (column-major) times (x, y, z, w).
void apply(const float (&m)[16], const double (&v)[4], double (&out)[4]) {
    for (int r = 0; r < 4; ++r) out[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
}

struct Known {
    double eye[3] = {100.0, 20.0, -50.0};
    double right[3], up[3], forward[3];
    float tan_x, tan_y, a, r2w;
    render_api::ViewTerms terms;
};

// A camera turned 30 degrees about y and looking 10 degrees down.
Known known(double near_z, double far_z) {
    Known k;
    const double yaw = 30.0 * kPi / 180.0, pitch = -10.0 * kPi / 180.0;
    k.forward[0] = std::cos(pitch) * std::sin(yaw);
    k.forward[1] = std::sin(pitch);
    k.forward[2] = std::cos(pitch) * std::cos(yaw);
    // right = world up x forward, up = forward x right (x right, y up, z forward).
    const double wu[3] = {0.0, 1.0, 0.0};
    k.right[0] = wu[1] * k.forward[2] - wu[2] * k.forward[1];
    k.right[1] = wu[2] * k.forward[0] - wu[0] * k.forward[2];
    k.right[2] = wu[0] * k.forward[1] - wu[1] * k.forward[0];
    const double rl = std::sqrt(k.right[0] * k.right[0] + k.right[1] * k.right[1] + k.right[2] * k.right[2]);
    for (double& c : k.right) c /= rl;
    k.up[0] = k.forward[1] * k.right[2] - k.forward[2] * k.right[1];
    k.up[1] = k.forward[2] * k.right[0] - k.forward[0] * k.right[2];
    k.up[2] = k.forward[0] * k.right[1] - k.forward[1] * k.right[0];
    k.tan_y = static_cast<float>(std::tan(21.5 * kPi / 180.0));
    k.tan_x = k.tan_y * 16.0f / 9.0f;
    // raw = a + r2w / z: 0 at the near plane, 1 at the far one.
    k.a = static_cast<float>(far_z / (far_z - near_z));
    k.r2w = static_cast<float>(near_z * far_z / (near_z - far_z));
    k.terms.tan_x = k.tan_x;
    k.terms.tan_y = k.tan_y;
    k.terms.a = k.a;
    k.terms.r2w = k.r2w;
    for (int r = 0; r < 3; ++r) {
        k.terms.rows[4 * r + 0] = static_cast<float>(k.right[r]);
        k.terms.rows[4 * r + 1] = static_cast<float>(k.up[r]);
        k.terms.rows[4 * r + 2] = static_cast<float>(k.forward[r]);
        k.terms.rows[4 * r + 3] = static_cast<float>(k.eye[r]);
    }
    return k;
}

// The world point at view (x, y, z).
void world_of(const Known& k, const double (&v)[3], double (&w)[4]) {
    for (int r = 0; r < 3; ++r) w[r] = k.eye[r] + k.right[r] * v[0] + k.up[r] * v[1] + k.forward[r] * v[2];
    w[3] = 1.0;
}

void test_camera() {
    const Known k = known(0.05, 3013.0);
    BbFrameCamera cam{};
    bool reversed = true;
    render_api::frame_camera(k.terms, &cam, &reversed);
    CHECK(!reversed);
    for (int r = 0; r < 3; ++r) {
        CHECK(near(cam.camera.position[r], k.eye[r], 1e-4));
        CHECK(near(cam.camera.right[r], k.right[r], 1e-6));
        CHECK(near(cam.camera.up[r], k.up[r], 1e-6));
        CHECK(near(cam.camera.forward[r], k.forward[r], 1e-6));
    }
    CHECK(cam.camera.position[3] == 1.0f && cam.camera.right[3] == 0.0f && cam.camera.up[3] == 0.0f && cam.camera.forward[3] == 0.0f);
    CHECK(near(cam.camera.fov_y, 43.0 * kPi / 180.0, 1e-5));
    CHECK(near(cam.camera.aspect, 16.0 / 9.0, 1e-5));
    CHECK(near(cam.camera.near_z, 0.05, 1e-4));
    // The terms are floats, as the game's are: 1 - a holds the far plane to
    // about 0.2% (3017.5 m here).
    CHECK(near(cam.camera.far_z, 3013.0, 30.0));

    // The view: the eye at the origin, ten metres ahead (0, 0, 10), three right (3, 0, 0).
    const double ahead[3] = {0.0, 0.0, 10.0}, aside[3] = {3.0, 0.0, 0.0};
    double w[4], v[4];
    world_of(k, ahead, w);
    apply(cam.view, w, v);
    CHECK(near(v[0], 0.0, 1e-3) && near(v[1], 0.0, 1e-3) && near(v[2], 10.0, 1e-3) && near(v[3], 1.0, 1e-6));
    world_of(k, aside, w);
    apply(cam.view, w, v);
    CHECK(near(v[0], 3.0, 1e-3) && near(v[1], 0.0, 1e-3) && near(v[2], 0.0, 1e-3));

    // A point at view (1, 2, 10): its clip position through proj, then the same through view_proj from the world.
    const double p[3] = {1.0, 2.0, 10.0};
    const double pv[4] = {p[0], p[1], p[2], 1.0};
    double c[4];
    apply(cam.proj, pv, c);
    CHECK(near(c[3], 10.0, 1e-5));  // w is the view's z
    const double ndc[3] = {c[0] / c[3], c[1] / c[3], c[2] / c[3]};
    CHECK(near(ndc[0], p[0] / (k.tan_x * p[2]), 1e-5));
    CHECK(near(ndc[1], -p[1] / (k.tan_y * p[2]), 1e-5));  // up in the view is up the image: y down
    CHECK(near(ndc[2], k.a + k.r2w / p[2], 1e-6));        // the depth image's value
    world_of(k, p, w);
    double c2[4];
    apply(cam.view_proj, w, c2);
    for (int r = 0; r < 4; ++r) CHECK(near(c2[r], c[r], 2e-3));

    // The host's own decode of that pixel (motion_at.glsl): uv, then the view point.
    const double uv[2] = {(ndc[0] + 1.0) * 0.5, (ndc[1] + 1.0) * 0.5};
    const double gx = (uv[0] * 2.0 - 1.0) * k.tan_x, gy = -(uv[1] * 2.0 - 1.0) * k.tan_y;
    const double d = k.r2w / (ndc[2] - k.a);
    CHECK(near(gx * d, p[0], 1e-3) && near(gy * d, p[1], 1e-3) && near(d, p[2], 1e-3));

    // Back to the world from the pixel and its depth.
    const double clip[4] = {ndc[0], ndc[1], ndc[2], 1.0};
    double back[4];
    apply(cam.inv_view_proj, clip, back);
    for (int r = 0; r < 3; ++r) CHECK(near(back[r] / back[3], w[r], 5e-3));

    // inv_view_proj * view_proj is the identity.
    for (int col = 0; col < 4; ++col) {
        const double e[4] = {col == 0 ? 1.0 : 0.0, col == 1 ? 1.0 : 0.0, col == 2 ? 1.0 : 0.0, col == 3 ? 1.0 : 0.0};
        double t[4], i[4];
        apply(cam.view_proj, e, t);
        apply(cam.inv_view_proj, t, i);
        for (int r = 0; r < 4; ++r) CHECK(near(i[r], e[r], 1e-3));
    }

    // The jittered projection (bbhost_plugin.h): m[8] and m[9] move a point by the jitter's render pixels.
    const double jx = 0.25, jy = -0.375, rw = 1920.0, rh = 1080.0;
    float jp[16];
    for (int i = 0; i < 16; ++i) jp[i] = cam.proj[i];
    jp[8] = static_cast<float>(2.0 * jx / rw);
    jp[9] = static_cast<float>(2.0 * jy / rh);
    double cj[4];
    apply(jp, pv, cj);
    const double px = ((ndc[0] + 1.0) * 0.5) * rw, pxj = ((cj[0] / cj[3] + 1.0) * 0.5) * rw;
    const double py = ((ndc[1] + 1.0) * 0.5) * rh, pyj = ((cj[1] / cj[3] + 1.0) * 0.5) * rh;
    CHECK(near(pxj - px, jx, 1e-3) && near(pyj - py, jy, 1e-3));

    // A depth that runs the other way: the same planes, reversed.
    render_api::ViewTerms rev = k.terms;
    rev.a = 1.0f - k.a;
    rev.r2w = -k.r2w;
    BbFrameCamera rc{};
    render_api::frame_camera(rev, &rc, &reversed);
    CHECK(reversed);
    CHECK(near(rc.camera.near_z, 0.05, 1e-4) && near(rc.camera.far_z, 3013.0, 30.0));
    double rcv[4];
    const double at_near[4] = {0.0, 0.0, 0.05, 1.0};
    apply(rc.proj, at_near, rcv);
    CHECK(near(rcv[2] / rcv[3], 1.0, 1e-4));

    // No decode at all: the stand-ins, and no division by zero.
    render_api::ViewTerms none = k.terms;
    none.r2w = 0.0f;
    BbFrameCamera nc{};
    render_api::frame_camera(none, &nc, &reversed);
    CHECK(nc.camera.near_z == 0.1f && nc.camera.far_z == 10000.0f && !reversed);
}

BbCamera turned(const BbCamera& c, double degrees) {
    BbCamera t = c;
    const double a = degrees * kPi / 180.0;
    // About the up axis: forward toward right.
    for (int r = 0; r < 3; ++r) t.forward[r] = static_cast<float>(std::cos(a) * c.forward[r] + std::sin(a) * c.right[r]);
    return t;
}

void test_cut() {
    const Known k = known(0.05, 3013.0);
    BbFrameCamera cam{};
    bool reversed = false;
    render_api::frame_camera(k.terms, &cam, &reversed);
    const BbCamera& c = cam.camera;
    CHECK(!render_api::frame_cut(c, c));
    BbCamera moved = c;
    moved.position[0] += 3.0f;
    CHECK(!render_api::frame_cut(moved, c));
    moved.position[2] += 3.0f;  // 4.24 m
    CHECK(render_api::frame_cut(moved, c));
    CHECK(!render_api::frame_cut(turned(c, 70.0), c));
    CHECK(render_api::frame_cut(turned(c, 80.0), c));
    BbCamera wider = c;
    wider.fov_y += static_cast<float>(9.0 * kPi / 180.0);
    CHECK(!render_api::frame_cut(wider, c));
    wider.fov_y += static_cast<float>(2.0 * kPi / 180.0);
    CHECK(render_api::frame_cut(wider, c));
}

void test_reset() {
    using render_api::FrameShape;
    const FrameShape a{1920, 1080, 1920, 1080, BB_UPSCALE_TAA};
    CHECK(!render_api::frame_reset(false, a, a, true, true, false, 1.0 / 60.0));
    CHECK(render_api::frame_reset(true, a, a, true, true, false, 0.0));        // the first frame
    FrameShape b = a;
    b.render_w = 1280;
    CHECK(render_api::frame_reset(false, a, b, true, true, false, 0.016));     // a new render size
    b = a;
    b.output_h = 1440;
    CHECK(render_api::frame_reset(false, a, b, true, true, false, 0.016));     // a new output size
    b = a;
    b.method = BB_UPSCALE_DLSS;
    CHECK(render_api::frame_reset(false, a, b, true, true, false, 0.016));     // another method
    CHECK(render_api::frame_reset(false, a, a, false, true, false, 0.016));    // no camera this frame
    CHECK(render_api::frame_reset(false, a, a, true, false, false, 0.016));    // none the last
    CHECK(render_api::frame_reset(false, a, a, true, true, true, 0.016));      // a cut
    CHECK(!render_api::frame_reset(false, a, a, true, true, false, 0.25));
    CHECK(render_api::frame_reset(false, a, a, true, true, false, 0.26));      // a pause
}

// The images' generations: the same image keeps its number, anything of it
// changed takes a new one, the host letting go of it takes a new one even
// when the next image comes back under the same handle, and no number is
// handed out twice across the slots.
void test_generations() {
    render_api::ImageSlot scene, depth;
    std::uint64_t serial = 0;
    CHECK(render_api::image_generation(scene, 0, 0, 97, 1920, 1080, serial) == 0);  // no image: none
    const std::uint64_t g1 = render_api::image_generation(scene, 0x1000, 0x2000, 97, 1920, 1080, serial);
    CHECK(g1 != 0);
    CHECK(render_api::image_generation(scene, 0x1000, 0x2000, 97, 1920, 1080, serial) == g1);
    const std::uint64_t d1 = render_api::image_generation(depth, 0x3000, 0x4000, 130, 1920, 1080, serial);
    CHECK(d1 != g1);
    const std::uint64_t g2 = render_api::image_generation(scene, 0x1000, 0x2000, 97, 2560, 1440, serial);  // a Resolution change
    CHECK(g2 != g1 && g2 != d1);
    const std::uint64_t g3 = render_api::image_generation(scene, 0x5000, 0x2000, 97, 2560, 1440, serial);  // another image
    CHECK(g3 != g2);
    const std::uint64_t g4 = render_api::image_generation(scene, 0x5000, 0x6000, 97, 2560, 1440, serial);  // another view
    CHECK(g4 != g3);
    const std::uint64_t g5 = render_api::image_generation(scene, 0x5000, 0x6000, 64, 2560, 1440, serial);  // another format
    CHECK(g5 != g4);
    render_api::image_gone(scene, 0x7777);  // someone else's
    CHECK(render_api::image_generation(scene, 0x5000, 0x6000, 64, 2560, 1440, serial) == g5);
    render_api::image_gone(scene, 0x5000);  // let go of: the same handle again is a new image
    const std::uint64_t g6 = render_api::image_generation(scene, 0x5000, 0x6000, 64, 2560, 1440, serial);
    CHECK(g6 != g5);
    render_api::image_gone(scene, 0x6000);  // its view
    CHECK(render_api::image_generation(scene, 0x5000, 0x6000, 64, 2560, 1440, serial) != g6);
    CHECK(render_api::image_generation(depth, 0x3000, 0x4000, 130, 1920, 1080, serial) == d1);
}

}  // namespace

int main() {
    test_camera();
    test_cut();
    test_reset();
    test_generations();
    if (g_failures) {
        std::fprintf(stderr, "render_api_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("render_api_test: the cameras, cuts, resets and image generations as plugin API 27 defines them\n");
    return 0;
}
