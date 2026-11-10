// The PC options' generated menu assets (engine/menu_assets.cpp, which it
// includes) against the game's own files: the option movie's PC Graphics
// section has the seven widget rows its handler fills (engine/option_menu.cpp:
// anti-aliasing, ambient occlusion, HLSL Shaders, three sliders, the
// resolution list) with the Defaults label below them, the other sections keep
// their slots, every section's rows are stacked top-down (each over the rows
// after it, which a row's pick list opens over; three mutants of that, each of
// which the check must catch), every language's message bundle carries HLSL
// Shaders' text and line help, and every PC row's line help fits the menus'
// help line. Skips (77) without the dump (BBHOST_APP0).
#include "test_app0.h"

#define MENU_ASSETS_MUTANTS  // the stacking's mutants (g_menu_assets_mutant)
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

// The sections build_option_movie makes.
const char* const kSections[] = {"PCSetting", "PCGraphics", "PCControls", "PCKeys", "PCEffects",
                                 "PCCamera",  "PCDeck",     "PCEnhance",  "PCUpscale"};

// The faults in the stacking of the main-timeline section `name`'s rows: each
// row (Item_<n>_0, the Defaults label the last) has to be at a depth above
// every row's after it - a row's pick list is drawn at the row's depth and
// opens down over them - and every frame's move at a row's depth has to keep
// that row within half a row's pitch of its place (the fade moves each row,
// not another into its place). With `log`, the depths and each fault.
int stack_faults(const Bytes& movie, const char* name, bool log) {
    std::optional<std::uint16_t> cid;
    for (const Tag& t : tags(movie, body_start(movie), movie.size())) {
        if (t.code != kPlace3 && t.code != kPlace2) continue;
        const Placement p = placement(movie, t.code, t.s);
        if (p.name && *p.name == name && p.cid) cid = p.cid;
    }
    const auto sprite = cid ? find_sprite(movie, *cid) : std::nullopt;
    if (!sprite) {
        if (log) std::fprintf(stderr, "%s: no section\n", name);
        return 1;
    }
    struct Row {
        std::uint16_t depth;
        std::int64_t tx, ty;
    };
    std::map<int, Row> rows;  // Item_<n>_0's n
    for (const Tag& t : tags(movie, sprite->s + 4, sprite->e)) {
        if (t.code != kPlace2 && t.code != kPlace3) continue;
        const Placement p = placement(movie, t.code, t.s);
        if (!p.cid || !p.has_mat || !p.name || !starts_with(*p.name, "Item_") || !ends_with(*p.name, "_0")) continue;
        rows[std::atoi(p.name->substr(5, p.name->size() - 7).c_str())] = Row{p.depth, p.tx, p.ty};
    }
    if (!rows.count(0) || !rows.count(1)) {
        if (log) std::fprintf(stderr, "%s: no Item_0_0 and Item_1_0\n", name);
        return 1;
    }
    int faults = 0;
    std::string depths;
    std::map<std::uint16_t, int> at;  // depth -> row
    const Row* before = nullptr;
    for (const auto& [n, r] : rows) {
        depths += " " + std::to_string(r.depth);
        if (before && r.depth >= before->depth) {
            ++faults;
            if (log) std::fprintf(stderr, "%s: Item_%d_0 at depth %d, not below the row before it (%d)\n", name, n, r.depth, before->depth);
        }
        before = &r;
        at[r.depth] = n;
    }
    const std::int64_t half_pitch = std::llabs(rows[1].ty - rows[0].ty) / 2;
    int frame = 0;
    for (const Tag& t : tags(movie, sprite->s + 4, sprite->e)) {
        if (t.code == kShowFrame) ++frame;
        if (t.code != kPlace2 && t.code != kPlace3) continue;
        const Placement p = placement(movie, t.code, t.s);
        const auto it = at.find(p.depth);
        if (p.cid || !p.has_mat || it == at.end()) continue;  // a move of a row's depth
        const Row& r = rows[it->second];
        if (p.tx != r.tx || std::llabs(p.ty - r.ty) > half_pitch) {
            ++faults;
            if (log)
                std::fprintf(stderr, "%s: frame %d moves depth %d (Item_%d_0, at %.1f,%.1f) to %.1f,%.1f\n", name, frame, p.depth, it->second,
                             r.tx / 20.0, r.ty / 20.0, p.tx / 20.0, p.ty / 20.0);
        }
    }
    if (log) std::printf("%s: Item_0_0..Item_%d_0 at depths%s\n", name, rows.rbegin()->first, depths.c_str());
    return faults;
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
    // PC Settings has seven rows on Windows (one added, as PC Graphics'
    // seventh), six elsewhere, and the label; PC Effects keeps six rows and
    // the label.
    CHECK(section_rows(movie, "PCSetting") == items(kPcSettingSlots + 1));
    CHECK(section_rows(movie, "PCEffects") == items(7));
    // PC Upscaling: a slot for each of its rows - seven on Windows, six
    // elsewhere (no Frame generation) - and the label.
    CHECK(section_rows(movie, "PCUpscale") == items(kPcUpscaleSlots + 1));
    // PC Enhancements: a row for each of its toggles (engine/option_menu.cpp's
    // g_enhance) and the Defaults label.
    CHECK(section_rows(movie, "PCEnhance") == items(7));
    std::printf("PCGraphics rows:");
    for (const std::string& r : gfx) std::printf(" %s", r.c_str());
    std::printf("\n");

    // Every section's rows stacked top-down, so a list opened on a row covers
    // the rows under it and the Defaults label; then three mutants of the
    // stacking, each of which the check must catch - the first is the movie
    // as it was made before (Item_0_0 lowest, the label on top, PC Upscaling's
    // Upscaling Quality value drawn over Upscaling's open list).
    int stack = 0;
    for (const char* s : kSections) stack += stack_faults(movie, s, true);
    CHECK(stack == 0);
    static const char* const kStackMutants[] = {"", "the rows left as the donor stacks them (Item_0_0 lowest, the label on top)",
                                                "the frames' moves left at the rows' old depths",
                                                "the last row (the Defaults label) left where it was"};
    int caught = 0;
    for (int m = 1; m <= 3; ++m) {
        g_menu_assets_mutant = m;
        int faults = 0;
        try {
            const Bytes mutant = build_option_movie(*src);
            for (const char* s : kSections) faults += stack_faults(mutant, s, false);
        } catch (const Fail& f) {
            std::fprintf(stderr, "stacking mutant %d: build_option_movie: %s\n", m, f.why.c_str());
        }
        caught += faults > 0;
        std::printf("  stacking mutant %d (%s): %d faults, %s\n", m, kStackMutants[m], faults, faults ? "caught" : "NOT CAUGHT");
    }
    g_menu_assets_mutant = 0;
    CHECK(caught == 3);

    // 2. The strings: HLSL Shaders' id in the built-in table, and in every
    // language's bundle once it is built.
    const auto rows = read_messages();
    CHECK(rows.count(117009) && rows.at(117009).text == "HLSL Shaders" && rows.at(117009).help.find("restart") != std::string::npos);
    CHECK(rows.count(116008) && rows.at(116008).text == "Texture Quality");
    CHECK(rows.count(116009) && rows.at(116009).text == "Stay Borderless");
    CHECK(rows.count(116045) && rows.count(116042) && rows.count(116043) && rows.count(116044) && rows.at(116044).text == "High");
    CHECK(rows.count(124005) && rows.at(124005).text == "Full Phantom HP" && rows.at(124005).help.find("restart") != std::string::npos);
    CHECK(rows.count(124006) && rows.at(124006).text == "Auto-Replenish Items" &&
          rows.at(124006).help.find("restart") != std::string::npos);
    int languages = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(test_app0_file("dvdroot_ps4/msg"), ec)) {
        const auto raw = read_bytes((e.path() / "menu.msgbnd.dcx").string());
        if (!raw) continue;
        Bytes built;
        try {
            built = build_menu_messages(*raw, false);
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
            if (f.id == kMenuText) text = *it->second == utf16("HLSL Shaders");
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
    // System list's rows (110006..110014) and the port's sections (116000 to
    // PC Upscaling's 125999).
    int helps = 0;
    for (const auto& [id, row] : rows) {
        const bool pc_row = (id >= 110006 && id <= 110014) || (id >= 116000 && id < 126000);
        if (!pc_row || row.help.empty()) continue;
        ++helps;
        if (row.help.size() > 96) {
            std::fprintf(stderr, "%d's help is %zu characters (96 at most): %s\n", static_cast<int>(id), row.help.size(), row.help.c_str());
            ++g_failures;
        }
    }
    CHECK(helps > 40);
    std::printf("%d PC rows' help within 96 characters\n", helps);
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("menu assets: ok\n");
    return 0;
}
