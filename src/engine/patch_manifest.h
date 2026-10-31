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
// tree), the per-user folder's patches (config_user_dir(): a player's own,
// which no release carries) and <mods>/patches, in that order; a later file
// with the same name replaces an earlier one.
//
// A player leaves one out with `<name> = false` under [patches] in
// bbhost.toml, which the setup window's Patches tab and the patch menu in the
// game (F9) write; one whose [patch] says `enabled = false` is off until
// `<name> = true` there. A site may name patches it steps aside for
// (`yields = ["other"]`): while one of them is on, that site is neither
// checked nor written. `title` is the name the menus show (the patch list's
// own, "Disable SSAO"); without one they show `name`.
//
// The patch menu in the game turns a patch on or off while the game runs
// (patch_manifest_set_live): its bytes written, or the eboot's own put back,
// at once. `live = false` marks one that only takes effect as the game starts
// (render targets sized as they are made, code bbhost rewrites as it runs):
// the menu saves its switch for the next start and leaves the code alone.
//
// bbhost's own byte patches that a setting switches (Skip company logos) are
// carried in the binary as manifests no menu lists; a file of the same name
// replaces one.
//
// `gameplay = true` marks a patch that changes the game (how far the player
// runs, how long the rally window lasts, a debug move): while it is on, the
// session's online rules carry "patch.<name>" (X-BBHost-Ruleset), so the
// server keeps its bloodstains, messages and statistics apart from vanilla
// ones, as it does for gameplay plugins. A patch without it is taken as not
// changing the game (graphics, loading, the intro).
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
    std::string name, title, description, option, eboot, group;
    bool enabled = true;   // its own [patch] enabled
    bool gameplay = false; // its own [patch] gameplay: it changes the game
    bool live = true;      // its own [patch] live: it can change while the game runs
    int order = 0;         // its own [patch] order
    std::string error;    // why it does not parse ("" when it does)
};
// Every manifest patch_manifests_apply would read, a later directory's
// replacing an earlier one's with the same file name, in the order they apply
// (order, then name). `mods` is the mods directory ("" for none).
std::vector<PatchListing> patch_manifests_list(const std::string& mods);
// The name the menus show: its title, or its name with spaces for dashes.
inline std::string patch_display_name(const PatchListing& p) {
    if (!p.title.empty()) return p.title;
    std::string n = p.name;
    for (char& c : n) c = c == '-' ? ' ' : c;
    return n;
}
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
// The applied manifests that change the game ([patch] gameplay = true), by name,
// sorted: each puts patch_rule_token(name) in the session's online rules
// (host/ruleset.h, host/plugins.cpp own_ruleset).
std::vector<std::string> patch_manifests_gameplay();
// Whether patch_manifests_apply has run: the list above is final after it.
bool patch_manifests_done();

// A manifest while the game runs, for the patch menu (host/ingame_menu.cpp).
struct PatchLiveState {
    bool known = false;    // read for this eboot at the start
    bool applied = false;  // its bytes are in the game's code now
    bool can_change = false;  // it may be put in or taken out now
    std::string why;       // why not, when it may not ("" when it may)
};
PatchLiveState patch_manifest_live_state(const std::string& name);
// Puts a manifest's bytes in (on) or the eboot's own back (off) while the
// game runs. Code it puts in first, the jumps to it last; taken out, the
// jumps go first and code it added where the eboot had zeros stays (a thread
// may still be inside it). A site that fits one aligned 8-byte word is one
// store; a longer code site first jumps to itself, so a thread reaching it
// waits while the rest is written. True when the bytes are as asked; false,
// and why, when they cannot change now (another of its group cannot go,
// bytes that are no longer what it expects, live = false): the switch, which
// the caller saves, then applies at the next start.
bool patch_manifest_set_live(const std::string& name, bool on, std::string* why);
