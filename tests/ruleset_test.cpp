// The session's online rules (host/ruleset.h) and the patch manifests that put
// an entry in them (engine/patch_manifest.h: [patch] gameplay = true): a
// gameplay patch must make the rules say modded, so the server keeps its
// players' bloodstains, messages and statistics apart from vanilla ones.
#include "core/config.h"
#include "host/ruleset.h"

#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

// What plugins_enter_world accepts in a ruleset.
bool ruleset_chars_ok(const std::string& s) {
    for (const char c : s) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ';' || c == '=' || c == ',' ||
              c == '-'))
            return false;
    }
    return true;
}

std::string unquote(const std::string& v) {
    return v.size() >= 2 && v.front() == '"' && v.back() == '"' ? v.substr(1, v.size() - 2) : v;
}

void test_join() {
    CHECK(ruleset_join({}) == "vanilla");
    CHECK(ruleset_join({""}) == "vanilla");
    CHECK(ruleset_join({"randomizer;seed=abc", "boss_rush"}) == "boss_rush,randomizer;seed=abc");
    CHECK(ruleset_join({"patch.rally-window", "patch.rally-window"}) == "patch.rally-window");
    // A gameplay patch alone is enough to leave vanilla.
    CHECK(ruleset_join({patch_rule_token("debug-dash")}) == "patch.debug-dash");
    CHECK(ruleset_join({patch_rule_token("debug-dash"), "mutators"}) == "mutators,patch.debug-dash");
}

void test_token() {
    CHECK(patch_rule_token("debug-dash") == "patch.debug-dash");
    CHECK(patch_rule_token("analog-run-range-40") == "patch.analog-run-range-40");
    CHECK(patch_rule_token("Bigger Weapons 1.2x") == "patch.bigger_weapons_1.2x");
    CHECK(patch_rule_token("a,b;c=d") == "patch.a_b_c_d");
    CHECK(ruleset_chars_ok(patch_rule_token("Weird/Name:\\x")));
}

// bbhost's own manifests: each parses and names a usable rules token, and
// exactly the ones listed here say they change the game - none since
// 2026-10-28 (Kyo: the gameplay flags do not ship); a mod's still can.
void test_manifests() {
    const std::set<std::string> gameplay = {};
    std::set<std::string> seen, marked;
    for (const auto& e : std::filesystem::directory_iterator(BB_PATCHES_DIR)) {
        if (e.path().extension() != ".toml") continue;
        std::map<std::string, std::string> kv;
        std::string err;
        const bool ok = config_parse_toml(e.path().string(), &kv, &err);
        CHECK(ok);
        if (!ok) {
            std::printf("  %s: %s\n", e.path().string().c_str(), err.c_str());
            continue;
        }
        const std::string name = unquote(kv["patch.name"]);
        CHECK(!name.empty());
        seen.insert(name);
        CHECK(ruleset_chars_ok(patch_rule_token(name)));
        const std::string gp = unquote(kv.count("patch.gameplay") ? kv["patch.gameplay"] : "");
        if (gp == "true" || gp == "1") marked.insert(name);
        if (gameplay.count(name) && !marked.count(name)) std::printf("  %s changes the game but does not say so\n", name.c_str());
        if (!gameplay.count(name) && marked.count(name)) std::printf("  %s says it changes the game\n", name.c_str());
    }
    for (const std::string& g : gameplay) CHECK(seen.count(g) == 1);  // every one ships
    CHECK(marked == gameplay);
}

}  // namespace

int main() {
    test_join();
    test_token();
    test_manifests();
    if (g_fail) {
        std::printf("ruleset_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("ruleset_test: ok\n");
    return 0;
}
