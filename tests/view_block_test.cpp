// The engine's per-view block test (host/view_block.h). Blocks as the game
// gave them - their numbers read from its memory at 1920x1080, 2560x1080,
// 1280x800, 3840x1080 and at 1920x1080 with the field of view 20% wider,
// where the inverse projection's (0, 0, 0, 1) row came out a rounding off:
// w 0.999999881, 0.99999994, 1.00000012 - and blocks made at many shapes and
// fields of view with the rows a few ulps off, must all be taken; buffers
// that are not the block must be refused. Then four mutants, each of which a
// check here must catch: the exact test it replaced, the test without the
// projection check, a test too loose, and an exact cheap pre-test.
#include "host/view_block.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

namespace vb = gpu::view_block;
using Block = std::vector<float>;

float from_bits(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

float ulps(float v, int n) {
    for (; n > 0; --n) v = std::nextafter(v, 2.0f);
    for (; n < 0; ++n) v = std::nextafter(v, 0.0f);
    return v;
}

// A block laid out as the engine's: the inverse projection (the tangents, the
// third row's w, the depth terms), the projection's diagonal and its depth
// rows, the camera (a turn about y and an eye), other floats some numbers.
Block make_block(float tan_x, float tan_y, float p00, float p11, float inv_w, float cam_w = 1.0f, float zero = 0.0f) {
    Block f(vb::kFloats, 0.0f);
    for (std::size_t i = 0; i < vb::kFloats; ++i) f[i] = 0.25f * static_cast<float>(i % 7);  // (the rest of the block)
    float* ip = f.data() + vb::kInvProj;
    for (int i = 0; i < 16; ++i) ip[i] = 0.0f;
    ip[0] = tan_x;
    ip[5] = tan_y;
    ip[8] = zero;  // the third row: (0, 0, 0, 1) on paper
    ip[10] = -zero;
    ip[11] = inv_w;
    ip[14] = -19.9728737f;  // the depth terms as the game had them
    ip[15] = 19.9732075f;
    float* p = f.data() + vb::kProj;
    for (int i = 0; i < 16; ++i) p[i] = 0.0f;
    p[0] = p00;
    p[5] = p11;
    p[10] = 1.00002f;
    p[11] = -0.0500671f;
    p[14] = 1.0f;
    float* c = f.data() + vb::kCamera;
    const float a = 0.6f, ca = std::cos(a), sa = std::sin(a);
    const float rows[16] = {ca, 0.0f, sa, -9.6f, 0.0f, 1.0f, 0.0f, -4.5f, -sa, 0.0f, ca, -18.9f, zero, 0.0f, 0.0f, cam_w};
    std::memcpy(c, rows, sizeof(rows));
    return f;
}

// The game's own, read at each screen (tangents, projection diagonal, the
// inverse projection's w by its bits).
struct Seen {
    const char* where;
    float tan_x, tan_y, p00, p11;
    std::uint32_t w_bits;
};
const Seen kSeen[] = {
    {"1920x1080", 0.700285256f, 0.393910468f, 1.42798948f, 2.53864789f, 0x3f800000u},
    {"2560x1080", 0.933713615f, 0.393910438f, 1.07099211f, 2.53864789f, 0x3f7ffffeu},
    {"1280x800", 0.630256772f, 0.393910468f, 1.5866549f, 2.53864789f, 0x3f7fffffu},
    {"3840x1080", 1.40057051f, 0.393910468f, 0.713994741f, 2.53864789f, 0x3f800000u},
    {"1920x1080, field of view +20%", 0.859407842f, 0.483416945f, 1.16359198f, 2.06860781f, 0x3f800001u},
};

using Rule = const char* (*)(const float*);
using PreTest = bool (*)(const float*);

// The checks: the number of wrong answers a rule (and a pre-test) gives.
int run_checks(Rule rule, PreTest pre, bool verbose) {
    int wrong = 0;
    auto take = [&](const Block& b, const char* what) {
        const char* why = rule(b.data());
        const bool ok = !why && pre(b.data());
        if (!ok) {
            ++wrong;
            if (verbose) std::fprintf(stderr, "  refused %s: %s\n", what, why ? why : "the cheap pre-test");
        }
    };
    auto refuse = [&](const Block& b, const char* what) {
        if (!rule(b.data())) {
            ++wrong;
            if (verbose) std::fprintf(stderr, "  taken, though not the block: %s\n", what);
        }
    };
    // Taken: what the game gave.
    for (const Seen& s : kSeen) take(make_block(s.tan_x, s.tan_y, s.p00, s.p11, from_bits(s.w_bits)), s.where);
    // Taken: many shapes and fields of view, the (0, 0, 0, 1) rows a few ulps
    // off either way, their zeros a hair off.
    const float shapes[] = {16.0f / 9.0f, 2560.0f / 1080.0f, 3440.0f / 1440.0f, 3840.0f / 1600.0f, 32.0f / 9.0f, 1.6f, 4.0f / 3.0f, 1.0f, 48.0f / 9.0f};
    const float fovs[] = {38.0f, 43.0f, 47.3f, 51.6f, 64.5f, 86.0f};  // degrees, vertical (+0% .. +100%)
    for (float shape : shapes) {
        for (float fov : fovs) {
            const float tan_y = std::tan(fov * 0.5f * 3.14159265f / 180.0f), tan_x = tan_y * shape;
            for (int n = -3; n <= 3; ++n) {
                for (float zero : {0.0f, 1e-8f, -1e-8f}) {
                    char what[96];
                    std::snprintf(what, sizeof(what), "shape %.3f, field of view %.1f, w %+d ulps, zeros %g", shape, fov, n, zero);
                    take(make_block(tan_x, tan_y, 1.0f / tan_x, 1.0f / tan_y, ulps(1.0f, n), ulps(1.0f, -n), zero), what);
                }
            }
        }
    }
    // Refused: buffers that are not the block.
    const Seen& s = kSeen[1];
    auto bad = [&](auto change, const char* what) {
        Block b = make_block(s.tan_x, s.tan_y, s.p00, s.p11, 1.0f);
        change(b);
        refuse(b, what);
    };
    bad([](Block& b) { b[vb::kInvProj + 11] = 1.001f; }, "the inverse projection's w 1.001");
    bad([](Block& b) { b[vb::kInvProj + 11] = 0.999f; }, "the inverse projection's w 0.999");
    bad([](Block& b) { b[vb::kInvProj + 8] = 1e-3f; }, "the inverse projection's third row's x 0.001");
    bad([](Block& b) { b[vb::kCamera + 15] = 1.001f; }, "the camera's w 1.001");
    bad([](Block& b) { b[vb::kCamera + 13] = -1e-3f; }, "the camera's last row's y -0.001");
    bad([](Block& b) { b[vb::kInvProj + 0] = 0.05f; }, "a horizontal tangent of 0.05");
    bad([](Block& b) { b[vb::kInvProj + 5] = 12.0f; }, "a vertical tangent of 12");
    bad([](Block& b) { b[vb::kProj + 0] *= 1.01f; }, "a projection 1% off the tangent");
    bad([](Block& b) { b[vb::kProj + 5] = 0.0f; }, "no projection beside the tangents");
    bad([](Block& b) { b[vb::kInvProj + 12] = 1e-3f; }, "the depth terms' x 0.001");
    bad([](Block& b) { b[vb::kInvProj + 14] = 0.0f; }, "no depth term");
    bad([](Block& b) { b[100] = std::nanf(""); }, "a NaN");
    bad([](Block& b) { b[3] = INFINITY; }, "an infinity");
    {
        Block zeros(vb::kFloats, 0.0f), ones(vb::kFloats, 1.0f), identities(vb::kFloats, 0.0f), noise(vb::kFloats);
        for (std::size_t i = 0; i < vb::kFloats; i += 5) identities[i] = 1.0f;  // a run of 4x4 identities
        std::uint32_t x = 12345u;
        for (float& v : noise) {
            x = x * 1664525u + 1013904223u;
            v = static_cast<float>(x >> 8) / 16777216.0f;
        }
        refuse(zeros, "zeros");
        refuse(ones, "ones");
        refuse(identities, "identity matrices");
        refuse(noise, "noise");
    }
    return wrong;
}

// The mutants.
const char* mutant_exact(const float* f) {  // c0ee3984's test: the rows exactly (0, 0, 0, 1), no projection check
    for (std::size_t i = 0; i < vb::kFloats; ++i) {
        if (!std::isfinite(f[i])) return "not finite";
    }
    const float tx = f[36], ty = f[41];
    if (!(tx > 0.1f && tx < 10.0f && ty > 0.1f && ty < 10.0f)) return "tangents";
    if (f[44] != 0.0f || f[45] != 0.0f || f[46] != 0.0f || f[47] != 1.0f) return "row";
    if (f[192] != 0.0f || f[193] != 0.0f || f[194] != 0.0f || f[195] != 1.0f) return "camera";
    if (!(std::fabs(f[50]) > 1e-6f) || std::fabs(f[48]) > 1e-4f || std::fabs(f[49]) > 1e-4f) return "depth";
    return nullptr;
}
bool mutant_exact_pre(const float* f) { return f[47] == 1.0f && f[195] == 1.0f; }
const char* mutant_no_projection(const float* f) {  // the rows within a rounding, the tangents unchecked
    Block b(f, f + vb::kFloats);
    b[vb::kProj + 0] = 1.0f / b[vb::kInvProj + 0];
    b[vb::kProj + 5] = 1.0f / b[vb::kInvProj + 5];
    return vb::refusal(b.data());
}
const char* mutant_loose(const float* f) {  // the rows within 1e-2
    Block b(f, f + vb::kFloats);
    for (int k : {44, 45, 46, 192, 193, 194}) {
        if (std::fabs(b[k]) <= 1e-2f) b[k] = 0.0f;
    }
    for (int k : {47, 195}) {
        if (std::fabs(b[k] - 1.0f) <= 1e-2f) b[k] = 1.0f;
    }
    return vb::refusal(b.data());
}

}  // namespace

int main() {
    const int wrong = run_checks(&vb::refusal, &vb::may_be, true);
    CHECK(wrong == 0);
    std::printf("view_block_test: %zu blocks as the game gave them, the made ones and the refusals: %d wrong\n", sizeof(kSeen) / sizeof(kSeen[0]),
                wrong);
    struct Mutant {
        const char* name;
        Rule rule;
        PreTest pre;
    } mutants[] = {{"the exact test (as before)", &mutant_exact, &mutant_exact_pre},
                   {"no projection check", &mutant_no_projection, &vb::may_be},
                   {"rows within 1e-2", &mutant_loose, &vb::may_be},
                   {"an exact cheap pre-test", &vb::refusal, &mutant_exact_pre}};
    for (const Mutant& m : mutants) {
        const int caught = run_checks(m.rule, m.pre, false);
        std::printf("view_block_test: mutant '%s': %d answers wrong%s\n", m.name, caught, caught ? " (caught)" : " - NOT CAUGHT");
        CHECK(caught > 0);
    }
    if (g_failures) {
        std::fprintf(stderr, "view_block_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("view_block_test: ok\n");
    return 0;
}
