// The PC options' generated menu assets (engine/menu_assets.cpp, which it
// includes) against the game's own files: the option movie's PC Graphics
// section has the seven widget rows its handler fills (engine/option_menu.cpp:
// anti-aliasing, ambient occlusion, HLSL Shaders, three sliders, the
// resolution list) with the Defaults label below them, the other sections keep
// their slots, every language's message bundle carries HLSL Shaders' text and
// line help, and every PC row's line help fits the menus' help line. Skips (77)
// without the dump (BBHOST_APP0).
#include "test_app0.h"

#include "engine/menu_assets.cpp"

#include <cstdio>
#include <fstream>
#include <iterator>

// menu_assets.cpp's file hooks, which only menu_assets_ensure calls.
const char* hle_fs_data_root() { return ""; }
const char* hle_fs_app0_root() { return ""; }
const char* hle_fs_update_root() { return ""; }
std::string hle_fs_game_file(const std::string& rel) { return rel; }
void hle_fs_set_generated_root(const char*) {}

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

std::optional<Bytes> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    return Bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// The rows ("Item_N_0") of the main-timeline section named `name`.
std::set<std::string> section_rows(const Bytes& movie, const char* name) {
    for (const Tag& t : tags(movie, body_start(movie), movie.size())) {
        if (t.code != kPlace3 && t.code != kPlace2) continue;
        const Placement p = placement(movie, t.code, t.s);
        if (!p.name || *p.name != name || !p.cid) continue;
        const auto sprite = find_sprite(movie, *p.cid);
        if (!sprite) return {};
        std::set<std::string> rows;
        for (const std::string& n : names_in_sprite(movie, sprite->s, sprite->e)) {
            if (starts_with(n, "Item_") && ends_with(n, "_0")) rows.insert(n);
        }
        return rows;
    }
    return {};
}

std::set<std::string> items(int n) {
    std::set<std::string> out;
    for (int k = 0; k < n; ++k) out.insert("Item_" + std::to_string(k) + "_0");
    return out;
}

}  // namespace

int main() {
    const std::string movie_path = test_app0_file("dvdroot_ps4/menu/optionsetting.gfx");
    const auto src = movie_path.empty() ? std::nullopt : read_bytes(movie_path);
    if (!src) {
        std::fprintf(stderr, "no dvdroot_ps4/menu/optionsetting.gfx (BBHOST_APP0); skipped\n");
        return kTestSkip;
    }
    // 1. The movie: PC Graphics has rows 0..6 and its Defaults label as row 7.
    Bytes movie;
    try {
        movie = build_option_movie(*src);
    } catch (const Fail& f) {
        std::fprintf(stderr, "build_option_movie: %s\n", f.why.c_str());
        return 1;
    }
    const std::set<std::string> gfx = section_rows(movie, "PCGraphics");
    CHECK(gfx == items(8));  // seven widget rows and the Defaults label
    // PC Settings and PC Effects keep six rows and the label.
    CHECK(section_rows(movie, "PCSetting") == items(7));
    CHECK(section_rows(movie, "PCEffects") == items(7));
    std::printf("PCGraphics rows:");
    for (const std::string& r : gfx) std::printf(" %s", r.c_str());
    std::printf("\n");

    // 2. The strings: HLSL Shaders' id in the built-in table, and in every
    // language's bundle once it is built.
    const auto rows = read_messages();
    CHECK(rows.count(117009) && rows.at(117009).text == "HLSL シェーダー" && rows.at(117009).help.find("再起動") != std::string::npos);
    int languages = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(test_app0_file("dvdroot_ps4/msg"), ec)) {
        const auto raw = read_bytes((e.path() / "menu.msgbnd.dcx").string());
        if (!raw) continue;
        Bytes built;
        try {
            built = build_menu_messages(*raw);
        } catch (const Fail& f) {
            std::fprintf(stderr, "%s: %s\n", e.path().filename().string().c_str(), f.why.c_str());
            ++g_failures;
            continue;
        }
        const std::size_t dca = find(built, "DCA\0", 4);
        CHECK(dca != std::string::npos);
        if (dca == std::string::npos) continue;
        const Bnd4 bnd = bnd_read(inflate_all(built, dca + be32(built, dca + 4)));
        bool text = false, help = false;
        for (const BndFile& f : bnd.files) {
            if (f.id != kMenuText && f.id != kLineHelp) continue;
            const Fmg ids = fmg_read(f.data);
            const auto it = ids.find(117009);
            if (it == ids.end() || !it->second) continue;
            if (f.id == kMenuText) text = *it->second == utf16("HLSL シェーダー");
            else help = *it->second == utf16(rows.at(117009).help);
        }
        CHECK(text && help);
        ++languages;
    }
    CHECK(languages > 0);
    std::printf("%d language bundles carry HLSL Shaders\n", languages);

    // 3. Every PC row's line help fits the help line, which shows one line
    // and cuts it off past the dialog (about 130 characters of its font; the
    // Resolution row's 198 and Frame Rate's 175 were cut): 96 at most. The
    // System list's rows (110006..110013) and the port's sections (116000 on).
    int helps = 0;
    for (const auto& [id, row] : rows) {
        const bool pc_row = (id >= 110006 && id <= 110013) || (id >= 116000 && id < 125000);
        if (!pc_row || row.help.empty()) continue;
        ++helps;
        if (utf16(row.help).size() / 2 > 48) {
            std::fprintf(stderr, "%d's help is %zu UTF-16 units (48 at most): %s\n", static_cast<int>(id), utf16(row.help).size() / 2, row.help.c_str());
            ++g_failures;
        }
    }
    CHECK(helps > 40);
    std::printf("%d PC rows' help within 48 UTF-16 units\n", helps);
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("menu assets: ok\n");
    return 0;
}
