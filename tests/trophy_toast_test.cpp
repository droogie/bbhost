// host/trophy_toast.cpp: what the popup says and draws, against a stand-in
// for the host overlay that records the calls. One scenario per run (the popup
// shows one trophy at a time, for several seconds), named on the command line:
//
//   nopack            no trophy pack was read (no key, say): one line, "Trophy
//                     unlocked - See docs for full behavior", no grade square, no icon
//   pack              the pack is read: the grade's line and the name, a square, the icon
//   noicon            the trophy has no icon in the pack: the square alone
//   testmode          BBHOST_TROPHY_TOAST_TEST=1 with the pack: the first popup is the
//                     bronze trophy (the others follow, one at a time)
//   testmode_nopack   the same with no pack: the one-line popup
//   room <vertices>   the pack, with that much vertex room left in the frame: the
//                     icon's mosaic at 32x32, at 16x16, or not at all
#include "core/trophy_pack.h"
#include "host/overlay.h"
#include "host/trophy_toast.h"

#include "trophy_test_pack.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

std::vector<std::string> g_texts;
int g_rects = 0, g_tris = 0;
std::size_t g_room = 100000;
}  // namespace

// The overlay the popup draws on, recording instead of drawing.
void host_overlay_rect(float, float, float, float, std::uint32_t) { ++g_rects; }
void host_overlay_tri(float, float, float, float, float, float, std::uint32_t) { ++g_tris; }
float host_overlay_text(float, float, float scale, std::uint32_t, const char* text) {
    g_texts.emplace_back(text);
    return 8.0f * scale * static_cast<float>(std::strlen(text));
}
float host_overlay_text_width(float scale, const char* text) { return 8.0f * scale * static_cast<float>(std::strlen(text)); }
std::size_t host_overlay_room() { return g_room; }

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "";
    unsetenv("BBHOST_TROPHY_TOAST");
    if (scenario.rfind("testmode", 0) == 0) {
        setenv("BBHOST_TROPHY_TOAST_TEST", "1", 1);
    } else {
        unsetenv("BBHOST_TROPHY_TOAST_TEST");
    }
    if (argc > 2) g_room = static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 10));

    int id = 0;
    if (scenario == "nopack") {
        id = 5;
    } else if (scenario == "pack" || scenario == "room") {
        id = 0;
    } else if (scenario == "noicon") {
        id = 2;
    } else if (scenario == "testmode" || scenario == "testmode_nopack") {
        id = -1;  // nothing is pushed here: the test mode does it
    } else {
        std::printf("trophy_toast_test: unknown scenario \"%s\"\n", scenario.c_str());
        return 2;
    }

    // The icons are read from the pack file when a popup starts, so it stays
    // until the draw below has run.
    char tmpl[] = "/tmp/trophy_toast_test_XXXXXX";
    std::filesystem::path dir;
    if (scenario != "nopack" && scenario != "testmode_nopack") {
        if (!mkdtemp(tmpl)) return 1;
        dir = tmpl;
        std::ofstream(dir / "t.trp", std::ios::binary).write(reinterpret_cast<const char*>(kTestTrp), static_cast<std::streamsize>(kTestTrpSize));
        std::vector<char> np(32, 0);
        np.insert(np.end(), kTestCommId, kTestCommId + 12);
        std::ofstream(dir / "n.dat", std::ios::binary).write(np.data(), static_cast<std::streamsize>(np.size()));
        std::string err;
        CHECK(trophy_pack_load((dir / "t.trp").string(), (dir / "n.dat").string(), kTestTrophyKey, &err));
    }

    if (id >= 0) host_trophy_toast_push(id);
    host_trophy_toast_draw(1920.0f, 1080.0f);

    if (scenario == "nopack") {
        CHECK(g_texts.size() == 1);
        CHECK(!g_texts.empty() && g_texts[0] == "Trophy unlocked - See docs for full behavior");
        CHECK(g_tris == 0);
        CHECK(g_rects == 5);  // the box and its four border strips: no grade square
    } else if (scenario == "testmode_nopack") {
        CHECK(g_texts.size() == 1 && g_texts[0] == "Trophy unlocked - See docs for full behavior");
    } else if (scenario == "testmode") {  // the pack has bronze, silver, gold and platinum: bronze first
        CHECK(g_texts.size() == 2);
        CHECK(g_texts.size() == 2 && g_texts[0] == "Bronze trophy earned" && g_texts[1] == "Caf? Bronze");
    } else if (scenario == "noicon") {
        CHECK(g_texts.size() == 2);
        CHECK(g_texts.size() == 2 && g_texts[0] == "Silver trophy earned" && g_texts[1] == "Silver");
        CHECK(g_tris == 0);
        CHECK(g_rects == 6);  // and the grade's square
    } else if (scenario == "pack") {
        CHECK(g_texts.size() == 2);
        CHECK(g_texts.size() == 2 && g_texts[0] == "Gold trophy earned" && g_texts[1] == "Gold & Glory");
        CHECK(g_rects == 6);
        CHECK(g_tris == 96);  // the icon at 32x32: 48 runs of two triangles
    } else {          // room
        CHECK(g_texts.size() == 2 && g_texts[1] == "Gold & Glory");
        CHECK(g_rects == 6);
        std::printf("trophy_toast_test: room %zu: %d triangles\n", g_room, g_tris);
        // 32x32 needs its 48 runs' vertices and 1500 to spare, 16x16 its 24 runs'.
        const int want = g_room >= 48 * 6 + 1500 ? 96 : g_room >= 24 * 6 + 1500 ? 48 : 0;
        CHECK(g_tris == want);
    }

    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    if (g_fail) {
        std::printf("trophy_toast_test %s: %d failed\n", scenario.c_str(), g_fail);
        return 1;
    }
    std::printf("trophy_toast_test %s: ok\n", scenario.c_str());
    return 0;
}
