// Patch manifests: byte patches to the loaded eboot as data
// files, so a patch is a file next to the host rather than code in it, and a
// mod can ship its own. docs/modding.md is the format; the short of it:
//
//   [patch]
//   name = "skip-intro"
//   eboot = "941f887a..."            # the SHA-256 the addresses belong to
//   description = "..."
//   option = "startup.skip_intro"    # optional: a config switch that enables it
//   [site.fromsoft]
//   address = "0x04d99138"           # guest VA at the preferred 0x400000 base
//   expect = "4c 00 6f 00"           # what must be there
//   bytes = "00 00 00 00"            # what is written
//   kind = "code"                    # or "data": how the page is protected after
//
// A manifest applies only when its eboot hash is the loaded image's, and
// only whole: every site's expected bytes are checked before any is
// written. Files come from <exe dir>/patches, <exe dir>/../patches (a build
// tree) and <mods>/patches, in that order; a later file with the same name
// replaces an earlier one.
//
// A player leaves one out with `<name> = false` under [patches] in
// bbhost.toml, which the setup window's Patches tab writes; one whose [patch]
// says `enabled = false` is off until `<name> = true` there. A site may name
// patches it steps aside for (`yields = ["other"]`): while one of them is on,
// that site is neither checked nor written.
//
// `order = N` is the patch's place in the list it came from (Kyo's shadPS4
// list): manifests apply in that order, then by name, each site checked
// against the eboot's own bytes, so where two write the same bytes the later
// one's stand - as shadPS4 applies the list. `group = "x"` makes variants of
// one patch exclusive: one of a group is on at a time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ElfImage;

// Reads the manifests and applies the ones that match; one log line each.
void patch_manifests_apply(ElfImage* image);

// One manifest as the setup window's Patches tab lists it (host/launcher.cpp):
// read the way patch_manifests_apply reads it, nothing applied.
struct PatchListing {
    std::string path;  // the file it was read from
    std::string name, description, option, eboot, group;
    bool enabled = true;  // its own [patch] enabled
    int order = 0;        // its own [patch] order
    std::string error;    // why it does not parse ("" when it does)
};
// Every manifest patch_manifests_apply would read, by file name, a later
// directory's replacing an earlier one's. `mods` is the mods directory ("" for none).
std::vector<PatchListing> patch_manifests_list(const std::string& mods);
// Whether the player left this manifest out: [patches] <name> = false.
bool patch_manifest_off(const std::string& name);
// Whether this manifest applies by its switch: [patches] <name> when set,
// else its own [patch] enabled (false: off until the player turns it on).
bool patch_manifest_on(const std::string& name, bool enabled);
// Whether `option` is one a manifest may name (startup.skip_intro, ...).
bool patch_manifest_option_known(const std::string& option);
// The applied manifest that changed any byte of [address, address + n)
// (guest addresses at the preferred base, as manifests give them), or null.
// The host's own patches leave such bytes as the manifest wrote them rather
// than refuse over them (engine/frame_rate.cpp).
const char* patch_manifest_owner(std::uint64_t address, std::size_t n);
