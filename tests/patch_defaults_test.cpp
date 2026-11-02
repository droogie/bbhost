// The patches that ship (patches/*.toml), read as the game reads them
// (engine/patch_manifest_file.h): the ones on by default - their own [patch]
// enabled on, nothing in bbhost.toml - are exactly the list below, and Enable
// Screen Space Reflections is not among them. A manifest without `enabled`
// reads as on, so a dropped line, a typo in it or a parser change that turns a
// patch on for every player fails here first. Nor does anything a player
// starts from turn one on: the first-start template (config_write_template)
// and the example config the packages ship have no [patches] key set to true.
#include "core/config.h"
#include "engine/patch_manifest_file.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <unistd.h>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

namespace fs = std::filesystem;

// The patches on by their own switch as of v0.2.25, each with the setting it
// also waits for ("" none): one more or one fewer is a decision, made here.
const std::map<std::string, std::string> kOnByDefault = {
    {"easier-run-range", ""},
    {"ingame-post-processors", "streaming.all_post_processors"},
    {"old-hunters", ""},
    {"performance-v3-decals-on", ""},
    {"quick-reentry", "loading.quick_reentry"},
};

// The [patches] keys a config file sets to on.
std::set<std::string> patches_turned_on(const fs::path& file) {
    std::map<std::string, std::string> kv;
    std::string err;
    std::set<std::string> on;
    if (!config_parse_toml(file.string(), &kv, &err)) {
        std::printf("  %s: %s\n", file.string().c_str(), err.c_str());
        CHECK(false);
        return on;
    }
    for (const auto& [k, v] : kv) {
        if (k.rfind("patches.", 0) == 0 && !(v == "false" || v == "0" || v == "\"false\"")) on.insert(k);
    }
    return on;
}

}  // namespace

int main() {
    std::map<std::string, std::string> on;  // name -> option
    std::set<std::string> seen;
    for (const auto& e : fs::directory_iterator(BB_PATCHES_DIR)) {
        if (e.path().extension() != ".toml") continue;
        patch_file::Manifest m;
        std::string err;
        const bool ok = patch_file::parse(e.path().string(), &m, &err);
        CHECK(ok);
        if (!ok) {
            std::printf("  %s: %s\n", e.path().string().c_str(), err.c_str());
            continue;
        }
        CHECK(seen.insert(m.name).second);  // one file a name
        if (m.enabled) on[m.name] = m.option;
    }
    CHECK(seen.size() > kOnByDefault.size());
    for (const auto& [name, option] : on) {
        if (!kOnByDefault.count(name)) std::printf("  %s is on by default and not on the list\n", name.c_str());
        else if (kOnByDefault.at(name) != option) std::printf("  %s now waits for \"%s\"\n", name.c_str(), option.c_str());
    }
    for (const auto& [name, option] : kOnByDefault) {
        if (!on.count(name)) std::printf("  %s is on the list and not on by default\n", name.c_str());
    }
    CHECK(on == kOnByDefault);
    CHECK(seen.count("screen-space-reflections") == 1);
    CHECK(on.count("screen-space-reflections") == 0);

    // The guard catches a dropped line: Screen Space Reflections' own file
    // without its `enabled = false` reads as on.
    const fs::path dir = fs::temp_directory_path() / ("patch_defaults_test." + std::to_string(::getpid()));
    fs::create_directories(dir);
    {
        std::ifstream in(fs::path(BB_PATCHES_DIR) / "screen-space-reflections.toml");
        std::stringstream all;
        all << in.rdbuf();
        std::string text = all.str(), kept;
        int dropped = 0;
        std::istringstream lines(text);
        for (std::string line; std::getline(lines, line);) {
            if (line.rfind("enabled", 0) == 0) {
                ++dropped;
                continue;
            }
            kept += line + "\n";
        }
        CHECK(dropped == 1);
        std::ofstream(dir / "without-enabled.toml") << kept;
        patch_file::Manifest m;
        std::string err;
        CHECK(patch_file::parse((dir / "without-enabled.toml").string(), &m, &err));
        CHECK(m.name == "screen-space-reflections" && m.enabled);
    }

    // What a player starts from turns nothing on.
    CHECK(config_write_template((dir / "template.toml").string()));
    CHECK(patches_turned_on(dir / "template.toml").empty());
    CHECK(patches_turned_on(fs::path(BB_PATCHES_DIR).parent_path() / "bbhost.example.toml").empty());
    fs::remove_all(dir);

    std::printf("patch_defaults_test: %zu manifests, %zu on by default; %s\n", seen.size(), on.size(), g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
