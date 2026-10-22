// Draw-replay comparison and manifest JSON: the checks a replay verdict rests
// on, without a GPU.
#include "replay/compare.h"
#include "replay/json.h"

#include <vulkan/vulkan_core.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

#include <unistd.h>

static int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static void test_json_round_trip() {
    json::Value root = json::Value::make_object();
    root.set("schema", "bbhost-draw-capture");
    root.set("version", 1);
    root.set("address", json::hex(0x1438ad910ull));
    root.set("text", "quote \" backslash \\ newline \n tab \t");
    json::Value list = json::Value::make_array();
    list.push(true);
    list.push(nullptr);
    list.push(-2.5);
    list.push(json::f32(std::numeric_limits<float>::quiet_NaN()));
    root.set("list", list);
    root.set("version", 2);  // replaces, keeps order
    const std::string text = json::dump(root);
    json::Value back;
    std::string error;
    CHECK(json::parse(text, back, error));
    CHECK(json::u64(back, "version") == 2);
    CHECK(back.object[1].first == "version");
    CHECK(json::u64(back, "address") == 0x1438ad910ull);
    CHECK(json::str(back, "text") == "quote \" backslash \\ newline \n tab \t");
    const auto& items = json::arr(back, "list");
    CHECK(items.size() == 4 && items[0].boolean && items[1].type == json::Value::Type::Null && items[2].number == -2.5);
    const float nan = json::as_f32(items[3], "list[3]");
    json::Value holder = json::Value::make_object();
    holder.set("x", json::f32(-2.5f));
    CHECK(json::read_f32(holder, "x") == -2.5f);
    std::uint32_t bits;
    std::memcpy(&bits, &nan, 4);
    CHECK(bits == 0x7fc00000u);

    CHECK(json::parse("{\"a\": \"\\u00e9\\ud83d\\ude00\"}", back, error));
    CHECK(json::str(back, "a") == "\xc3\xa9\xf0\x9f\x98\x80");
    for (const char* bad : {"{\"a\": 1,}", "{\"a\": 1, \"a\": 2}", "[1, 2", "{\"a\" 1}", "01x", "\"\\ud800\"", "[1] x"}) {
        CHECK(!json::parse(bad, back, error));
    }
    json::Value obj;
    CHECK(json::parse("{\"n\": 1.5, \"s\": \"0xzz\", \"big\": 1e300}", obj, error));
    bool threw = false;
    try { json::u64(obj, "n"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { json::u64(obj, "s"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { json::u64(obj, "big"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { json::member(obj, "missing"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

static void test_layouts() {
    std::uint32_t bytes = 0, bw = 0, bh = 0;
    CHECK(replay::texel_layout(VK_FORMAT_R16G16B16A16_SFLOAT, replay::kAspectColor, bytes, bw, bh) && bytes == 8 && bw == 1);
    CHECK(replay::texel_layout(VK_FORMAT_D32_SFLOAT_S8_UINT, replay::kAspectDepth, bytes, bw, bh) && bytes == 4);
    CHECK(replay::texel_layout(VK_FORMAT_D32_SFLOAT_S8_UINT, replay::kAspectStencil, bytes, bw, bh) && bytes == 1);
    CHECK(!replay::texel_layout(VK_FORMAT_D32_SFLOAT_S8_UINT, replay::kAspectColor, bytes, bw, bh));
    CHECK(!replay::texel_layout(VK_FORMAT_R16G16B16A16_SFLOAT, replay::kAspectDepth, bytes, bw, bh));
    CHECK(replay::texel_layout(VK_FORMAT_BC7_UNORM_BLOCK, replay::kAspectColor, bytes, bw, bh) && bytes == 16 && bw == 4 && bh == 4);
    CHECK(replay::subresource_bytes(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, replay::kAspectColor, 5, 3, 1) == 16);
    CHECK(replay::subresource_bytes(VK_FORMAT_B10G11R11_UFLOAT_PACK32, replay::kAspectColor, 1920, 1080, 1) == 1920ull * 1080 * 4);
    CHECK(replay::subresource_bytes(VK_FORMAT_R8G8B8A8_UNORM, replay::kAspectColor, 4, 4, 3) == 4 * 4 * 3 * 4);
    CHECK(replay::subresource_bytes(VK_FORMAT_UNDEFINED, replay::kAspectColor, 4, 4, 1) == 0);
}

static void test_decode() {
    std::string error;
    replay::Plane p;
    // Half floats: 1.0, -2.0, +inf, NaN / smallest subnormal, 65504, -0, 0.5.
    const std::uint16_t half[8] = {0x3c00, 0xc000, 0x7c00, 0x7e00, 0x0001, 0x7bff, 0x8000, 0x3800};
    CHECK(replay::decode_plane(VK_FORMAT_R16G16B16A16_SFLOAT, replay::kAspectColor, reinterpret_cast<const std::uint8_t*>(half),
                               sizeof(half), 2, 1, p, error));
    CHECK(p.channels == 4 && p.values.size() == 8);
    CHECK(p.values[0] == 1.0 && p.values[1] == -2.0 && std::isinf(p.values[2]) && std::isnan(p.values[3]));
    CHECK(p.values[4] == std::ldexp(1.0, -24) && p.values[5] == 65504.0 && p.values[6] == 0.0 && std::signbit(p.values[6]));
    CHECK(p.values[7] == 0.5);
    CHECK(!replay::decode_plane(VK_FORMAT_R16G16B16A16_SFLOAT, replay::kAspectColor, reinterpret_cast<const std::uint8_t*>(half),
                                sizeof(half) - 2, 2, 1, p, error));

    // B10G11R11: R = 1.0, G = 0.5, B = 2.0.
    const std::uint32_t packed = (15u << 6) | ((14u << 6) << 11) | ((16u << 5) << 22);
    CHECK(replay::decode_plane(VK_FORMAT_B10G11R11_UFLOAT_PACK32, replay::kAspectColor, reinterpret_cast<const std::uint8_t*>(&packed),
                               4, 1, 1, p, error));
    CHECK(p.channels == 3 && p.values[0] == 1.0 && p.values[1] == 0.5 && p.values[2] == 2.0);

    const std::uint8_t rgba[4] = {0, 255, 51, 128};
    CHECK(replay::decode_plane(VK_FORMAT_B8G8R8A8_UNORM, replay::kAspectColor, rgba, 4, 1, 1, p, error));
    CHECK(p.values[0] == 0.0 && p.values[1] == 1.0 && p.values[2] == 0.2 && p.values[3] == 128 / 255.0);

    const std::uint32_t d24 = 0xab800000u;  // stencil bits above, depth 0x800000
    CHECK(replay::decode_plane(VK_FORMAT_D24_UNORM_S8_UINT, replay::kAspectDepth, reinterpret_cast<const std::uint8_t*>(&d24), 4, 1, 1,
                               p, error));
    CHECK(p.values[0] == 0x800000 / 16777215.0);
    const float depth = 0.25f;
    CHECK(replay::decode_plane(VK_FORMAT_D32_SFLOAT_S8_UINT, replay::kAspectDepth, reinterpret_cast<const std::uint8_t*>(&depth), 4, 1,
                               1, p, error));
    CHECK(p.values[0] == 0.25);
    const std::uint8_t stencil = 200;
    CHECK(replay::decode_plane(VK_FORMAT_D32_SFLOAT_S8_UINT, replay::kAspectStencil, &stencil, 1, 1, 1, p, error));
    CHECK(p.values[0] == 200.0);
    CHECK(!replay::decode_plane(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, replay::kAspectColor, rgba, 4, 1, 1, p, error));
}

static void test_tolerance() {
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    replay::Tolerance exact;
    CHECK(replay::values_match(1.0, 1.0, exact));
    CHECK(replay::values_match(0.0, -0.0, exact));
    CHECK(!replay::values_match(1.0, std::nextafter(1.0, 2.0), exact));
    CHECK(replay::values_match(1.0, 1.25, {0.25, 0.0}));
    CHECK(!replay::values_match(1.0, 1.25, {0.2499, 0.0}));
    CHECK(replay::values_match(100.0, 101.0, {0.0, 0.01}));   // 1 <= 0.01 * 101
    CHECK(!replay::values_match(100.0, 102.0, {0.0, 0.01}));  // 2 > 0.01 * 102
    CHECK(replay::values_match(nan, nan, exact));
    CHECK(!replay::values_match(nan, 1.0, {1e9, 1e9}));
    CHECK(replay::values_match(inf, inf, exact));
    CHECK(!replay::values_match(inf, -inf, {1e9, 1e9}));
    CHECK(!replay::values_match(inf, 1e300, {1e9, 1e9}));
}

static void test_compare() {
    replay::Plane a;
    a.width = 3;
    a.height = 2;
    a.channels = 2;
    for (int i = 0; i < 12; ++i) a.values.push_back(i * 0.5);
    replay::Plane b = a;
    replay::Comparison same = replay::compare_planes(a, b, {});
    CHECK(same.match() && same.texels == 6 && same.texels_different == 0 && same.first_failures.empty());

    // The perturbation a replay self-test uses: one channel of one texel.
    b.values[(1 * 3 + 2) * 2 + 1] += 0.5;
    replay::Comparison one = replay::compare_planes(a, b, {});
    CHECK(!one.match() && one.texels_failed == 1 && one.texels_different == 1);
    CHECK(one.first_failures.size() == 1 && one.first_failures[0].x == 2 && one.first_failures[0].y == 1 &&
          one.first_failures[0].channel == 1);
    CHECK(one.per_channel[1].failed == 1 && one.per_channel[1].max_abs == 0.5 && one.per_channel[1].max_x == 2 &&
          one.per_channel[1].max_y == 1 && one.per_channel[0].failed == 0);
    CHECK(one.texel_failed[5] == 1 && one.texel_error[5] == 0.5f);
    replay::Comparison tolerated = replay::compare_planes(a, b, {0.5, 0.0});
    CHECK(tolerated.match() && tolerated.texels_different == 1);

    b = a;
    b.values[0] = std::numeric_limits<double>::quiet_NaN();
    replay::Comparison nan = replay::compare_planes(a, b, {1.0, 1.0});
    CHECK(!nan.match() && nan.per_channel[0].nonfinite_b == 1 && nan.per_channel[0].nonfinite_mismatch == 1 &&
          std::isinf(nan.texel_error[0]));
    json::Value report = replay::to_json(nan);
    std::string error;
    json::Value back;
    CHECK(json::parse(json::dump(report), back, error));
    CHECK(!json::flag(back, "match") && json::str(json::arr(back, "first_failures")[0], "b") == "nan");
    // The file is this process's own, in the temp folder: two test runs at
    // once from one build folder (its working folder) would share it.
    const std::string heatmap =
        (std::filesystem::temp_directory_path() / ("bbhost-replay-compare-test-" + std::to_string(getpid()) + ".ppm")).string();
    CHECK(replay::write_heatmap(heatmap, nan));
    std::remove(heatmap.c_str());

    b = a;
    b.width = 2;
    CHECK(!replay::compare_planes(a, b, {}).shape_ok);
}

int main() {
    test_json_round_trip();
    test_layouts();
    test_decode();
    test_tolerance();
    test_compare();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("replay_compare_test: all checks passed");
    return 0;
}
