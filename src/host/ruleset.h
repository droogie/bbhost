// The session's online rules, as every request to the server carries them
// (X-BBHost-Ruleset): what this player's world is played by, so the server
// keeps modded players' bloodstains, messages and statistics apart from
// vanilla ones. host/plugins.cpp builds it from the gameplay plugins and the
// patch manifests that change the game (engine/patch_manifest.h).
#pragma once

#include <algorithm>
#include <string>
#include <vector>

// The entries, sorted and comma separated; "vanilla" when there are none.
inline std::string ruleset_join(std::vector<std::string> parts) {
    std::sort(parts.begin(), parts.end());
    parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
    std::string out;
    for (const std::string& p : parts) {
        if (p.empty()) continue;
        out += (out.empty() ? "" : ",") + p;
    }
    return out.empty() ? "vanilla" : out;
}

// A gameplay patch manifest's entry: "patch." and its name, lower case, in the
// characters a ruleset may hold ([a-z0-9_.-], what plugins_enter_world takes
// from a host); any other character becomes '_'. "debug-dash" ->
// "patch.debug-dash". The prefix keeps a patch apart from a plugin of the
// same name.
inline std::string patch_rule_token(const std::string& name) {
    std::string out = "patch.";
    for (char c : name) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
        out += ok ? c : '_';
    }
    return out;
}
