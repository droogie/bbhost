#pragma once

// One patch manifest file as engine/patch_manifest.cpp reads it (the format is
// engine/patch_manifest.h's and docs/modding.md's): its [patch] fields and its
// sites, nothing checked against an eboot or applied. A unit of its own, with
// nothing of the game in it, so a test reads the shipped manifests exactly as
// the game does (tests/patch_defaults_test.cpp).

#include <cstdint>
#include <string>
#include <vector>

namespace patch_file {

struct Site {
    std::string id;
    std::uint64_t address = 0;  // guest VA at the preferred base
    std::vector<std::uint8_t> expect, bytes;
    bool code = true;
    // Patches this site steps aside for: while any of them is on, this site
    // is neither checked nor written - two patches that write the same bytes
    // differently, the one named here having the last word.
    std::vector<std::string> yields;
};

struct Manifest {
    std::string path, name, title, eboot, description, option, group;
    bool enabled = true;
    bool gameplay = false;  // [patch] gameplay: changes the game (the session's online rules say so)
    bool live = true;       // [patch] live: it may change while the game runs (patch_manifest_set_live)
    int order = 0;  // [patch] order: its place in the list it came from (later wins)
    std::vector<Site> sites;
};

// Reads `path`: false, and why in *err, when it is not a manifest (no name or
// eboot, a site without its address, expect and bytes of one length, none).
// A missing [patch] enabled (or `default`, an earlier port's name for it)
// reads as on.
bool parse(const std::string& path, Manifest* m, std::string* err);

}  // namespace patch_file
