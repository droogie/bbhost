#include "engine/patch_manifest.h"
#include "engine/patch_manifest_file.h"

#include "core/config.h"
#include "core/elf.h"
#include "core/memory.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/render_scale.h"
#include "gcn/container.h"
#include "hle/fs.h"
#include "host/options.h"
#include "host/ruleset.h"
#include "host/scene_look.h"
#include "host/settings.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using patch_file::Manifest;
using patch_file::Site;

// What the applied manifests wrote, by guest address at the preferred base:
// the manifest each byte is now, and the eboot's own byte under it. A later
// manifest's site is checked against those, so it may write over an earlier
// one's bytes (patch_manifest_owner answers the host's own patches).
// The applied manifests that change the game, by name (patch_manifests_gameplay),
// and whether patch_manifests_apply has run.
std::mutex g_gameplay_mu;
std::vector<std::string> g_gameplay;  // under g_gameplay_mu
std::atomic<bool> g_manifests_done{false};

struct Written {
    std::string owner;
    std::uint8_t original = 0;
};
std::map<std::uint64_t, Written> g_written;  // under g_live_mu once the game runs

// Every manifest read for this eboot at the start, in the order they apply,
// and whether its bytes are in now: what the patch menu turns on and off while
// the game runs (patch_manifest_set_live). The image's memory, which lives as
// long as the run. A manifest bbhost's own patches left a value to
// (patch_manifest_owner: the frame-rate conversion skips what one wrote) keeps
// its bytes until the next start: taken out, that value would be the eboot's,
// not the conversion's.
struct Loaded {
    Manifest m;
    bool applied = false;
};
std::mutex g_live_mu;
std::vector<Loaded> g_loaded;         // under g_live_mu
GuestMemory* g_mem = nullptr;         // under g_live_mu
std::set<std::string> g_deferred_to;  // under g_live_mu
std::set<std::string> g_built_in;     // bbhost's own that no file replaced (under g_live_mu)


bool shadow_resolution_on();  // (the Shadow resolution setting's sizes, or Performance mode's, differ from the game's: below)

// The config switches a manifest may name.
bool option_on(const std::string& option, bool* known) {
    *known = true;
    if (option == "startup.skip_intro") return config().skip_intro;
    if (option == "loading.quick_reentry") return config().quick_reentry;
    if (option == "streaming.all_post_processors") return config().all_post_processors;
    if (option == "debug.free_camera") return host_settings().debug_camera;  // F10 > Game, read once
    if (option == "startup.performance_mode") return host_settings().performance_mode;  // F10 > Game, read once
    if (option == "graphics.ssr") return host_settings().ssr;  // F10 > GRAPHICS, read once
    if (option == "graphics.hd_motion_blur") return host_settings().hd_motion_blur;  // F10 > GRAPHICS, read once
    if (option == "graphics.shadow_resolution") return shadow_resolution_on();  // F10 > GRAPHICS (and Performance mode), read once
    if (option == "camera.fov_wide") return host_settings().fov_scale > 1.0f;  // then live (engine/camera.cpp)
    if (option == "graphics.render_scale") return render_scale_plan(nullptr);  // Upscaling and its quality, read once
    if (option == "plugin.scene_rgba8") return scene_look_colour_8bit();  // a plugin's scene_colour_bits (the Bloodborne PS1's), then live
    *known = false;
    return false;
}

// Whether a site steps aside for a patch that is on (Site::yields).
bool yielded(const Site& s, const std::set<std::string>& on) {
    for (const std::string& y : s.yields) {
        if (on.count(y)) return true;
    }
    return false;
}

// The eboot's own byte at a guest address (preferred base): under whatever an
// applied manifest wrote there.
std::uint8_t original_byte(std::uint64_t address, std::uint8_t now) {
    const auto it = g_written.find(address);
    return it == g_written.end() ? now : it->second.original;
}

// Whether it applied: every site written, already as it writes it, or left to
// a patch it yields to (false: refused, or a page could not be unprotected).
bool apply(ElfImage* image, const Manifest& m, const std::set<std::string>& on) {
    GuestMemory& mem = image->mem;
    // Every site checked before any is written - against the eboot's own
    // bytes, under any an earlier manifest wrote: manifests apply in their
    // list's order (order, then name), and where two write the same bytes the
    // later one's stand, as shadPS4 applies Kyo's list.
    for (const Site& s : m.sites) {
        if (yielded(s, on)) continue;
        if (s.address < kPreferredGuestSlide) {
            host_log("patch: %s refused: site %s address 0x%llx is below the image", m.name.c_str(), s.id.c_str(),
                     static_cast<unsigned long long>(s.address));
            return false;
        }
        const std::uint64_t off = s.address - kPreferredGuestSlide;
        if (off >= mem.size || s.expect.size() > mem.size - off) {
            host_log("patch: %s refused: site %s at 0x%llx is outside the image", m.name.c_str(), s.id.c_str(),
                     static_cast<unsigned long long>(s.address));
            return false;
        }
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, mem.slide + off));
        // Already applied (the bytes are what the patch writes) is not a
        // refusal: the same file from two directories, or a rerun.
        if (std::memcmp(p, s.bytes.data(), s.bytes.size()) == 0) continue;
        for (std::size_t i = 0; i < s.expect.size(); ++i) {
            if (p[i] == s.expect[i] || p[i] == s.bytes[i] || original_byte(s.address + i, p[i]) == s.expect[i]) continue;
            host_log("patch: %s refused: unexpected bytes at 0x%llx (site %s)", m.name.c_str(),
                     static_cast<unsigned long long>(s.address), s.id.c_str());
            return false;
        }
    }
    int written = 0, stepped_aside = 0, already = 0;
    std::string aside, over;
    for (const Site& s : m.sites) {
        if (yielded(s, on)) {
            ++stepped_aside;
            for (const std::string& y : s.yields) {
                if (on.count(y) && aside.find(y) == std::string::npos) aside += (aside.empty() ? "" : ", ") + y;
            }
            continue;
        }
        const std::uint64_t at = mem.slide + (s.address - kPreferredGuestSlide);
        auto* p = static_cast<std::uint8_t*>(guest_ptr(mem, at));
        if (std::memcmp(p, s.bytes.data(), s.bytes.size()) == 0) {
            already += s.bytes != s.expect;  // another patch wrote the same (a site that writes what is there counts neither way)
            continue;
        }
        const std::uint64_t lo = at & ~0xfffull, hi = (at + s.bytes.size() + 0xfff) & ~0xfffull;
        const bool ok = s.code ? guest_protect_rwx(&mem, lo, hi - lo) : guest_protect_rw(&mem, lo, hi - lo);
        if (!ok) {
            host_log("patch: %s: cannot unprotect 0x%llx (site %s); the rest is left alone", m.name.c_str(),
                     static_cast<unsigned long long>(s.address), s.id.c_str());
            return false;
        }
        for (std::size_t i = 0; i < s.bytes.size(); ++i) {
            if (p[i] == s.bytes[i]) continue;
            const auto [it, fresh] = g_written.try_emplace(s.address + i, Written{m.name, p[i]});
            if (!fresh && it->second.owner != m.name) {
                if (over.find(it->second.owner) == std::string::npos) over += (over.empty() ? "" : ", ") + it->second.owner;
                it->second.owner = m.name;
            }
        }
        std::memcpy(p, s.bytes.data(), s.bytes.size());
        if (s.code && !guest_protect_rx(&mem, lo, hi - lo)) {
            host_log("patch: %s: cannot restore the protection of 0x%llx", m.name.c_str(),
                     static_cast<unsigned long long>(s.address));
            std::abort();
        }
        ++written;
    }
    std::string notes;
    if (already) notes += " (" + std::to_string(already) + " already as it writes them)";
    if (stepped_aside) notes += " (" + std::to_string(stepped_aside) + " left to " + aside + ")";
    if (!over.empty()) notes += " (over " + over + "'s bytes, as it comes after in the list)";
    host_log("patch: %s: %d of %zu sites written%s%s%s", m.name.c_str(), written, m.sites.size(), notes.c_str(),
             m.description.empty() ? "" : " - ", m.description.c_str());
    return true;
}

// The directories manifests come from, in order: the package's, a build
// tree's, the player's own (beside the per-user bbhost.toml: patches of their
// own that no copy of bbhost ships, and every copy reads), the mods
// directory's.
std::vector<std::string> manifest_dirs(const std::string& mods) {
    std::vector<std::string> dirs = {config_exe_dir() + "/patches", config_exe_dir() + "/../patches", config_user_dir() + "/patches"};
    if (!mods.empty()) dirs.push_back(mods + "/patches");
    return dirs;
}

// Whether a texture pack (a TPF, inflated) names a texture: its name table
// holds each name NUL-terminated, in UTF-16 or in single bytes.
bool tpf_names(const std::vector<std::uint8_t>& tpf, const std::string& name) {
    std::vector<std::uint8_t> wide, narrow(name.begin(), name.end());
    for (const char c : name) wide.insert(wide.end(), {static_cast<std::uint8_t>(c), 0});
    wide.insert(wide.end(), {0, 0});
    narrow.push_back(0);
    return std::search(tpf.begin(), tpf.end(), wide.begin(), wide.end()) != tpf.end() ||
           std::search(tpf.begin(), tpf.end(), narrow.begin(), narrow.end()) != tpf.end();
}

// The Old Hunters edition looks for the DLC's own files, which came with the
// update that brought it (1.07), so a game folder holding the base game's
// versions has none of them: the edition's title art, MENU_Title_00004 in
// menu/title.tpf.dcx - without it the title movie fills its picture flat
// green (a player's game folder, 2026-10-07) - and the three areas' maps,
// m34-m36 (their lamps are the Old Hunters' headstone's, slots 44-54).
// What is missing, or "" when it is all there; read as the game will read it,
// update folder and mods included.
std::string old_hunters_missing() {
    const std::string pack = hle_fs_map_path("/app0/dvdroot_ps4/menu/title.tpf.dcx");
    std::ifstream in(pack, std::ios::binary);
    if (!in) return "menu/title.tpf.dcx";
    const std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::vector<std::uint8_t> tpf = gcn::dcx_decompress(raw);
    if (!tpf_names(tpf.empty() ? raw : tpf, "MENU_Title_00004")) return "Old Hunters title art (MENU_Title_00004 in menu/title.tpf.dcx)";
    for (const char* map : {"m34_00_00_00", "m35_00_00_00", "m36_00_00_00"}) {
        std::error_code ec;
        if (!std::filesystem::is_directory(hle_fs_map_path(("/app0/dvdroot_ps4/map/" + std::string(map)).c_str()), ec))
            return std::string("map/") + map + " (an Old Hunters area)";
    }
    return {};
}

// bbhost's own byte patches, carried in the binary rather than as files: a
// setting switches each, and no patch list shows one beside its setting. A
// manifest file of the same name replaces it (an older package's).
//
// performance-mode, Performance mode (startup.performance_mode): Kyo's Giga
// Performance Patch's post-effects. The scene's constructor sets its feature
// flags with one `mov dword [rbx+0x2968], 0x01010101` (SSAO +0x2968, motion
// blur +0x2969, anti-aliasing +0x296a) and the dynamic lights' shadows at
// +0x296e (imm at 0x26c2538); the patch made each 0. Since 2026-11-05 SSAO,
// anti-aliasing and the dynamic lights' shadows are cleared on each renderer
// by the settings' own switches over the same bytes instead
// (engine/graphics_patch.cpp settings_changed), which follow a change while
// the game runs; the motion blur's stays here (the velocity map's, kept for an
// upscaler). The directional shadow (+0x296c, the sun's and the moon's: the
// buildings' shadows) stays. YEBIS's apply (sub_25d5f80) jumps past every
// depth-of-field pass (0x25d7bbc) and reads GlareShape as 0, no bloom
// (0x25d72f0) - code read each frame, so put in and taken out while the game
// runs (host/options.cpp, patch_manifest_refresh_live). The map's fog is left
// as the game draws it (the Fog setting still scales it). Chromatic
// aberration is its own setting's mask (engine/graphics_patch.cpp), cleared
// while this is on.
Manifest performance_mode_manifest() {
    Manifest m;
    m.name = "performance-mode";
    m.title = "Performance mode";
    m.eboot = kEboot109Sha256;
    m.description = "every post-effect and YEBIS pass off: SSAO, motion blur, anti-aliasing, depth of field, chromatic "
                    "aberration, bloom and the dynamic lights' shadows";
    m.option = "startup.performance_mode";
    m.order = 11;
    const std::vector<std::uint8_t> one = {0x01}, zero = {0x00};
    // Motion blur's flag also draws the velocity map, which an upscaler's or
    // frame generation's motion takes the player's and the enemies' own
    // movement from (host/upscale_pass.cpp): with either the flag stays and
    // the blur is taken off the Motion blur setting's way (its shader patched
    // away, the dilation skipped - host/shader_patch.cpp,
    // engine/graphics_patch.cpp, which also follows a later change of either).
    if (!host_opt_motion_wanted()) m.sites.push_back(Site{"motion-blur", 0x026c2549, one, zero, true, {}});
    m.sites.push_back(Site{"dof", 0x025d7bbc, {0x0f, 0x84, 0xde, 0x00, 0x00, 0x00}, {0xe9, 0xdf, 0x00, 0x00, 0x00, 0x90}, true, {}});
    m.sites.push_back(Site{"bloom", 0x025d72f0, {0x8b, 0xb3, 0xd8, 0x00, 0x00, 0x00}, {0x31, 0xf6, 0x0f, 0x1f, 0x40, 0x00}, true, {}});
    return m;
}

// performance, Performance patch (always on: the renders the rest of bbhost's
// pipeline - the upscaler among it - is built over; [patches] performance =
// false in bbhost.toml leaves it out, for a test): Kyo's
// Performance Patch v3 (Velomap + Camera Blur, Decals On) from his shadPS4
// list, site for site, which were patches/performance-v3-velomap.toml and,
// without the velocity map, performance-v3-decals-on.toml - less the camera
// blur's force (0x26c265d 00 -> 01): the Motion blur setting decides it - and
// its model level bias's (0x216fc09, the mesh draw's read of it made -2,
// Full): the Model detail setting decides it (graphics_patch.cpp). The
// velocity map stays, at 640x360 (0x26c3de4/0x26c3df0), with motion blur's
// half and quarter depth at 640x360 and 320x180 (0x125b1a3/0x125b1d6, which
// Screen space reflections puts back at the game's sizes when it is on). A
// 2026-10-03 review of it, entry by entry, found parts emulator-specific,
// parts that leave SSAO, depth of field and distortion reading stale buffers
// (the skipped depth and scene copies), and parts it judged harmful (the file
// cache's access-time skips, a condition wait made a `ret`, a cave calling
// with a misaligned stack, the assert and DL_PANIC suppression). It is kept
// byte for byte as players have run it, until each part is done natively.
Manifest performance_manifest() {
    Manifest m;
    m.name = "performance";
    m.title = "Performance patch";
    m.eboot = kEboot109Sha256;
    m.description = "Kyo's Performance Patch v3 (Velomap, Decals On), without its camera blur's and model detail's forces";
    m.order = 21;
    m.sites = {
#include "engine/performance_v3_sites.inc"
    };
    return m;
}

// fov-uncap, Uncapped field of view (camera.fov_wide: Field of view above
// +0%): the camera's field of view is clamped to 38-48 degrees where the
// option table's value is applied (0x183af4e, `cmovbe rax, rsi`), so a wider
// one stopped at 48 - the game's 43 at +12%. The clamp comes out while the
// setting is above +0% and goes back at +0%, live (engine/camera.cpp
// settings_changed), so the game's own camera keeps it. Was
// patches/fov-uncap.toml, off unless turned on.
Manifest fov_uncap_manifest() {
    Manifest m;
    m.name = "fov-uncap";
    m.title = "Uncapped field of view";
    m.eboot = kEboot109Sha256;
    m.description = "the field of view is no longer capped at 48 degrees";
    m.option = "camera.fov_wide";
    m.order = 1040;
    m.sites.push_back(Site{"clamp", 0x0183af4e, {0x48, 0x0f, 0x46, 0xc6}, {0x48, 0x89, 0xf0, 0x90}, true, {}});
    return m;
}

// ssr, Screen space reflections (graphics.ssr): Kyo's 1.09 patch from his
// shadPS4 list ("Enable Screen Space Reflections (READ NOTE)"), site for site,
// which was patches/screen-space-reflections.toml. The scene context's
// reflection pass on (its targets made and the pass run) - its ray march's step
// scale at 0x12a8cfd left at the game's 1.0 (the list's 0.0 is dearer: 7.8 ms
// a frame at 1080p on the Radeon 8060S against 4.2; BBHOST_SSR_STEP=list) - its two targets at a quarter
// of the render size (Reflection quality Quarter, the default) or the game's
// half (Half), in format 10 instead of r12d's, and the depth renders it reads at the
// game's own half and quarter sizes - four sites that write the game's code
// back over a performance patch's fixed sizes - as 32-bit float (0x125b1bf:
// 0x14 -> 0x28). Two calls and two bytes as shipped (0x26a10c0, 0x26a2829,
// 0x26c3d7d, 0x26c4fd7) undo the performance patches' changes to what it
// reads. It loads after every patch file, so its bytes win. Its Fade To Viewer
// goes to -10000 with it (the last two sites).
Manifest ssr_manifest() {
    Manifest m;
    m.name = "ssr";
    m.title = "Screen space reflections";
    m.eboot = kEboot109Sha256;
    m.description = "the reflection pass on, the depth renders it reads at the game's own sizes";
    m.option = "graphics.ssr";
    m.order = 9000;
    auto site = [&](const char* id, std::uint64_t at, std::vector<std::uint8_t> expect, std::vector<std::uint8_t> bytes) {
        m.sites.push_back(Site{id, at, std::move(expect), std::move(bytes), true, {}});
    };
    site("enable-a", 0x026c4fd7, {0x01}, {0x01});
    site("enable-b", 0x026c4fd3, {0x77}, {0x74});        // ja -> je
    site("enable-c", 0x026c3d2f, {0x74, 0x53}, {0x90, 0x90});  // je 0x26c3d84 -> nops
    site("enable-d", 0x026c3d7d, {0x01}, {0x01});
    const std::vector<std::uint8_t> half_w = {0x8b, 0x03, 0xd1, 0xe8, 0x89, 0x04, 0x24},
                                    half_h = {0x8b, 0x43, 0x04, 0xd1, 0xe8, 0x89, 0x44, 0x24, 0x04},
                                    quarter_w = {0x8b, 0x03, 0xc1, 0xe8, 0x02, 0x89, 0x04, 0x24},
                                    quarter_h = {0x8b, 0x43, 0x04, 0xc1, 0xe8, 0x02, 0x89, 0x44, 0x24, 0x04};
    site("depth-half-width", 0x0125b1a3, half_w, half_w);
    site("depth-half-height", 0x0125b1aa, half_h, half_h);
    site("depth-quarter-width", 0x0125b1d6, quarter_w, quarter_w);
    site("depth-quarter-height", 0x0125b1de, quarter_h, quarter_h);
    site("depth-format", 0x0125b1bf, {0x14}, {0x28});
    // The game's own step scale, 1.0, unless BBHOST_SSR_STEP=list asks for
    // the list's 0.0 (about twice the ray march's cost: the 100 fps and more
    // Kyo saw SSR take, 2026-11-05).
    if (const char* e = std::getenv("BBHOST_SSR_STEP"); e && std::strcmp(e, "list") == 0)
        site("step-scale", 0x012a8cfd, {0x00, 0x00, 0x80, 0x3f}, {0x00, 0x00, 0x00, 0x00});
    site("call-a", 0x026a10c0, {0xe8, 0xbb, 0xe0, 0xfe, 0xff}, {0xe8, 0xbb, 0xe0, 0xfe, 0xff});
    site("call-b", 0x026a2829, {0xe8, 0x52, 0xc9, 0xfe, 0xff}, {0xe8, 0x52, 0xc9, 0xfe, 0xff});
    site("size-a", 0x026c429a, {0x44, 0x89, 0xe1}, {0x6a, 0x0a, 0x59});  // mov ecx, r12d -> push 10; pop rcx
    if (host_settings().ssr_half) {
        site("size-b", 0x026c42b4,
             {0xc7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x31, 0xc9, 0x41, 0xb9, 0x14, 0x00, 0x00, 0x00, 0x45, 0x89, 0xe0},
             {0x83, 0x24, 0x24, 0x00, 0x31, 0xc9, 0x41, 0xb9, 0x14, 0x00, 0x00, 0x00, 0x41, 0xb8, 0x0a, 0x00, 0x00, 0x00});
    } else {
        // Reflection quality Quarter (patches/screen-space-reflections-quarter.toml
        // until 2026-11-05): the two targets at a quarter of the render size each
        // way. The game halves with `shr esi, 1; shr edx, 1`; `shr r32, 2` is a
        // byte longer, so each block is written whole in its own bytes - the
        // first loads `mov r8d, 20` as `push 20; pop r8`, the second (which takes
        // size-b's place, mov r8d, 10 among it) `mov r9d, 20` the same way. No
        // branch lands inside either block.
        site("quarter-a", 0x026c4286, {0x41, 0x8b, 0x37, 0x41, 0x8b, 0x57, 0x04, 0xd1, 0xee, 0xd1, 0xea, 0x41, 0xb8, 0x14, 0x00, 0x00, 0x00},
             {0x41, 0x8b, 0x37, 0x41, 0x8b, 0x57, 0x04, 0xc1, 0xee, 0x02, 0xc1, 0xea, 0x02, 0x6a, 0x14, 0x41, 0x58});
        site("quarter-b", 0x026c42a9,
             {0x41, 0x8b, 0x37, 0x41, 0x8b, 0x57, 0x04, 0xd1, 0xee, 0xd1, 0xea, 0xc7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x31, 0xc9, 0x41,
              0xb9, 0x14, 0x00, 0x00, 0x00, 0x45, 0x89, 0xe0},
             {0x41, 0x8b, 0x37, 0x41, 0x8b, 0x57, 0x04, 0xc1, 0xee, 0x02, 0xc1, 0xea, 0x02, 0x83, 0x24, 0x24, 0x00, 0x31, 0xc9, 0x6a, 0x14,
              0x41, 0x59, 0x41, 0xb8, 0x0a, 0x00, 0x00, 0x00});
    }
    // [Ssr] Fade To Viewer Param - the scene context's +0x7680, set in its
    // constructor's movabs - -10000 over the game's -0.8, as the shadPS4 list
    // has it (the reflections stay where they face the camera), and the debug
    // menu's slider for it reaching down to it (its minimum loads -10000 at
    // 0x4c80a34 instead of -1.0) so it shows the value.
    site("fade-to-viewer", 0x02695a5b, {0xcd, 0xcc, 0x4c, 0xbf}, {0x00, 0x40, 0x1c, 0xc6});
    site("fade-to-viewer-slider", 0x026b8d65, {0xc5, 0xfa, 0x10, 0x05, 0x1f, 0x0f, 0x67, 0x02}, {0xc5, 0xfa, 0x10, 0x05, 0xc7, 0x7c, 0x5c, 0x02});
    return m;
}

// hd-motion-blur, HD motion blur (graphics.hd_motion_blur): Kyo's HD Motion
// Blur patches (360p, 1080p, 1440p, 4K - patches/hd-motion-blur-*.toml until
// now), each of which set motion blur's buffers to one fixed size, made a
// share of the render size instead (HD motion blur quality: a quarter,
// half - the game's own - or full). The three are where the game halves or
// quarters the scene's size for them: the velocity map (0x26c3de4, the
// scene context's target at +0xce0: `mov eax, [r15]; shr eax, 1` and the
// same for the height) and the half- and quarter-size depth renders it blurs
// with (0x125b1a3, 0x125b1d6), which the performance patch fixes at 640x360
// and 320x180 and Screen space reflections gives back the game's halves and
// quarters. Each becomes one 64-bit load of the {width, height} pair, a
// shift of both halves and one store - fewer bytes than the game's, xmm0 free
// there (a call that clobbers it follows, and nothing between reads it) -
// with the velocity map and the half depth at the chosen share and the
// quarter depth at half of it, as the patches had them (1080p: 1920x1080 and
// 960x540). A share of the scene's own size follows any render size,
// Resolution's and the upscaler's smaller one alike. The camera blur's switch
// (the scene context's +0x2a10, "Camera Blur (for Test)" in the debug menu,
// 0x26c265d) comes on with it, as in the patches. After Screen space
// reflections (order 9000), as the patches came after it in the list, so its
// sizes are the ones that stand.
Manifest hd_motion_blur_manifest() {
    Manifest m;
    m.name = "hd-motion-blur";
    m.title = "HD motion blur";
    m.eboot = kEboot109Sha256;
    m.description = "motion blur's velocity map and depth at a share of the render size, the camera blur on";
    m.option = "graphics.hd_motion_blur";
    m.order = 9100;
    // How far each is shifted down: the velocity map and the half depth by
    // 2, 1 or 0 (a quarter, half, full), the quarter depth one more.
    // While a plugin's look draws the 3D small, a share of it near half the
    // window's height (render_scale_blur_shift): a negative shift is one left.
    const int shift = render_scale_blur_shift(2 - std::clamp(host_settings().hd_motion_blur_res, 0, 2));
    const auto op = [](int k) { return static_cast<std::uint8_t>(k < 0 ? 0xf0 : 0xd0); };  // vpslld / vpsrld xmm0, xmm0, n
    const auto by = [](int k) { return static_cast<std::uint8_t>(k < 0 ? -k : k); };
    auto site = [&](const char* id, std::uint64_t at, std::vector<std::uint8_t> expect, std::vector<std::uint8_t> bytes) {
        bytes.resize(expect.size(), 0x90);
        m.sites.push_back(Site{id, at, std::move(expect), std::move(bytes), true, {}});
    };
    // vmovq xmm0, [r15]; vpsrld (vpslld) xmm0, xmm0, n; vmovq [rsp+0x200], xmm0
    site("velomap-size", 0x026c3de4,
         {0x41, 0x8b, 0x07, 0xd1, 0xe8, 0x89, 0x84, 0x24, 0x00, 0x02, 0x00, 0x00, 0x41,
          0x8b, 0x47, 0x04, 0xd1, 0xe8, 0x89, 0x84, 0x24, 0x04, 0x02, 0x00, 0x00},
         {0xc4, 0xc1, 0x7a, 0x7e, 0x07, 0xc5, 0xf9, 0x72, op(shift), by(shift), 0xc5, 0xf9, 0xd6, 0x84, 0x24, 0x00, 0x02, 0x00, 0x00});
    // vmovq xmm0, [rbx]; vpsrld xmm0, xmm0, n; vmovq [rsp], xmm0
    site("depth-half-size", 0x0125b1a3,
         {0x8b, 0x03, 0xd1, 0xe8, 0x89, 0x04, 0x24, 0x8b, 0x43, 0x04, 0xd1, 0xe8, 0x89, 0x44, 0x24, 0x04},
         {0xc5, 0xfa, 0x7e, 0x03, 0xc5, 0xf9, 0x72, op(shift), by(shift), 0xc5, 0xf9, 0xd6, 0x04, 0x24});
    site("depth-quarter-size", 0x0125b1d6,
         {0x8b, 0x03, 0xc1, 0xe8, 0x02, 0x89, 0x04, 0x24, 0x8b, 0x43, 0x04, 0xc1, 0xe8, 0x02, 0x89, 0x44, 0x24, 0x04},
         {0xc5, 0xfa, 0x7e, 0x03, 0xc5, 0xf9, 0x72, op(shift + 1), by(shift + 1), 0xc5, 0xf9, 0xd6, 0x04, 0x24});
    site("camera-blur", 0x026c265d, {0x00}, {0x01});  // mov byte [rbx+0x2a10], 0 -> 1
    return m;
}

// velomap-16f, the motion blur's velocity map as 16-bit float (always on):
// the scene context's two velocity targets (+0xce0, +0xd38) are made in the
// game's format 0x1c, RGBA8, and the meshes write their motion into it at a
// step of about a pixel at 1080p - coarse for the upscaler, which takes the
// meshes' own motion from it (host/upscale_pass.cpp). Format 0x0a, the lit
// scene's RGBA16F, keeps it exact; the game's blur reads it the same. One
// byte, the immediate of `mov dword [rsp+0x208], 0x1c` (0x26c3dfd).
// BBHOST_VELOMAP_RES=full|half|quarter (a test, HD motion blur off): the map
// at that share of the render size, without HD motion blur's camera blur.
Manifest velomap_manifest() {
    Manifest m;
    m.name = "velomap-16f";
    m.title = "16-bit velocity map";
    m.eboot = kEboot109Sha256;
    m.description = "motion blur's velocity map in 16-bit float, for the upscaler's motion";
    m.order = 9150;  // after HD motion blur: rebuilt motion vectors' size stands
    // Rebuilt motion vectors (the Motion vectors setting): the map at the
    // render size, in the bits asked for - the game's codes are DXGI's
    // (0x1c R8G8B8A8_UNORM, 0x0a R16G16B16A16_FLOAT, 0x02 R32G32B32A32_FLOAT).
    const int rebuilt = host_opt_motion_rebuilt_bits();
    const char* bits = std::getenv("BBHOST_VELOMAP_8BIT");  // =1: the game's RGBA8 kept (a test)
    const std::uint8_t format = rebuilt == 8 ? 0x1c : rebuilt == 32 ? 0x02 : 0x0a;
    if (format != 0x1c && !(bits && bits[0] == '1')) m.sites.push_back(Site{"velomap-format", 0x026c3e04, {0x1c}, {format}, true, {}});
    const char* e = std::getenv("BBHOST_VELOMAP_RES");
    if (rebuilt || (e && !host_settings().hd_motion_blur)) {
        const std::string r(e && !rebuilt ? e : "full");
        const int shift = r == "full" ? 0 : r == "quarter" ? 2 : 1;
        std::vector<std::uint8_t> bytes = {0xc4, 0xc1, 0x7a, 0x7e, 0x07, 0xc5, 0xf9, 0x72, 0xd0, static_cast<std::uint8_t>(shift),
                                           0xc5, 0xf9, 0xd6, 0x84, 0x24, 0x00, 0x02, 0x00, 0x00};
        bytes.resize(25, 0x90);
        m.sites.push_back(Site{"velomap-size", 0x026c3de4,
                               {0x41, 0x8b, 0x07, 0xd1, 0xe8, 0x89, 0x84, 0x24, 0x00, 0x02, 0x00, 0x00, 0x41,
                                0x8b, 0x47, 0x04, 0xd1, 0xe8, 0x89, 0x84, 0x24, 0x04, 0x02, 0x00, 0x00},
                               std::move(bytes), true, {}});
        // The depth it is drawn against, at the same share (as HD motion blur).
        m.sites.push_back(Site{"depth-half-size", 0x0125b1a3,
                               {0x8b, 0x03, 0xd1, 0xe8, 0x89, 0x04, 0x24, 0x8b, 0x43, 0x04, 0xd1, 0xe8, 0x89, 0x44, 0x24, 0x04},
                               {0xc5, 0xfa, 0x7e, 0x03, 0xc5, 0xf9, 0x72, 0xd0, static_cast<std::uint8_t>(shift), 0xc5, 0xf9, 0xd6, 0x04,
                                0x24, 0x90, 0x90},
                               true, {}});
        m.sites.push_back(Site{"depth-quarter-size", 0x0125b1d6,
                               {0x8b, 0x03, 0xc1, 0xe8, 0x02, 0x89, 0x04, 0x24, 0x8b, 0x43, 0x04, 0xc1, 0xe8, 0x02, 0x89, 0x44, 0x24, 0x04},
                               {0xc5, 0xfa, 0x7e, 0x03, 0xc5, 0xf9, 0x72, 0xd0, static_cast<std::uint8_t>(shift + 1), 0xc5, 0xf9, 0xd6, 0x04,
                                0x24, 0x90, 0x90, 0x90, 0x90},
                               true, {}});
    }
    return m;
}

// scene-rgba8, the scene's colour targets in 8 bits a channel (plugin API 18
// scene_colour_bits: the Bloodborne PS1's 8-bit colour): Kyo's "rgba8f color space
// (Perf Increase)" patch, site for site (patches/rgba8f-color-space.toml). The
// scene context's Initialize (sub_26c39f0) makes its colour targets in the
// formats these bytes name - the game's 0x0a, RGBA16F, which the Performance
// patch makes 0x1a (R11G11B10F) and, at 0x26c4b6f, 0x18 (R10G10B10A2) - and
// here they are all 0x1c, RGBA8; the 0x1c at 0x26c4984, which the Performance
// patch makes 0x1a, goes back to it. Two sites the Performance patch writes the
// same way already (the flag at 0x26c1084, the jne at 0x26c4ac6). After every
// built-in that touches the scene's targets, so its bytes stand; live: the
// targets are made again with it (engine/render_scale.cpp).
Manifest scene_rgba8_manifest() {
    Manifest m;
    m.name = "scene-rgba8";
    m.title = "8-bit scene colour";
    m.eboot = kEboot109Sha256;
    m.description = "the scene's colour targets in RGBA8, as the rgba8f patch makes them";
    m.option = "plugin.scene_rgba8";
    m.order = 9300;
    m.sites = {
#include "engine/scene_rgba8_sites.inc"
    };
    return m;
}

// all-languages, Every text language (always on): the game's text languages
// are CSLocalize's, and which of them a disc offers is its region's - the
// region the game makes of param.sfo's USER_DEFINED_PARAM_1 (the US disc's
// Americas: English (US), French, Spanish (Latin America), Portuguese
// (Brazil); a European disc's 14 European languages; Japan's Japanese
// alone). A static initializer (0x23b9680) builds a 32-byte entry per region
// at 0x59994e0: the list System > Language shows (+0, its count +8), what a
// language - the player's pick, or the system language's while there is
// none - becomes there (+0x10, 21 entries; Europe's turns Japanese into
// English (UK)) and the language past the end of that (+0x18). The sixth,
// region 5's (the build's own: a USER_DEFINED_PARAM_1 of 0), lists all 21
// and keeps each as it is. Its list, its count and its mapping now stand for
// every region's, each region's last-resort language left as it was.
//
// The region itself is untouched: it also picks the title id the game goes
// online as, its server address (region 5 has none) and whether Circle
// decides - which is why the community's Unlock Game Region patch, which made
// the game region 5, took online play with it. That patch file is retired
// (retired_file). A save with no pick of its own follows the system language,
// which the region's console answers (hle/system.cpp), so a European disc
// still starts in English (UK).
Manifest all_languages_manifest() {
    Manifest m;
    m.name = "all-languages";
    m.title = "Every text language";
    m.eboot = kEboot109Sha256;
    m.description = "System > Language lists all 21 of the game's text languages, whatever the disc's region";
    m.order = 12;
    auto site = [&](const char* id, std::uint64_t at, std::vector<std::uint8_t> expect, std::vector<std::uint8_t> bytes) {
        m.sites.push_back(Site{id, at, std::move(expect), std::move(bytes), true, {}});
    };
    // Per region: the list's lea (rip+disp32 to 0x4b368f0, region 5's), the
    // count's mov (21) and the mapping's lea (0x4b36950, region 5's).
    site("japan-list", 0x023b96b3, {0xf1, 0xcf, 0x77, 0x02}, {0x39, 0xd2, 0x77, 0x02});
    site("japan-count", 0x023b96c4, {0x01, 0x00, 0x00, 0x00}, {0x15, 0x00, 0x00, 0x00});
    site("japan-remap", 0x023b96cb, {0xe1, 0xcf, 0x77, 0x02}, {0x81, 0xd2, 0x77, 0x02});
    site("americas-list", 0x023b96e3, {0x29, 0xd0, 0x77, 0x02}, {0x09, 0xd2, 0x77, 0x02});
    site("americas-count", 0x023b96f4, {0x04, 0x00, 0x00, 0x00}, {0x15, 0x00, 0x00, 0x00});
    site("americas-remap", 0x023b96fb, {0x21, 0xd0, 0x77, 0x02}, {0x51, 0xd2, 0x77, 0x02});
    site("europe-list", 0x023b9713, {0x69, 0xd0, 0x77, 0x02}, {0xd9, 0xd1, 0x77, 0x02});
    site("europe-count", 0x023b9724, {0x0e, 0x00, 0x00, 0x00}, {0x15, 0x00, 0x00, 0x00});
    site("europe-remap", 0x023b972b, {0x91, 0xd0, 0x77, 0x02}, {0x21, 0xd2, 0x77, 0x02});
    site("asia-list", 0x023b9743, {0xd9, 0xd0, 0x77, 0x02}, {0xa9, 0xd1, 0x77, 0x02});
    site("asia-count", 0x023b9754, {0x04, 0x00, 0x00, 0x00}, {0x15, 0x00, 0x00, 0x00});
    site("asia-remap", 0x023b975b, {0xd1, 0xd0, 0x77, 0x02}, {0xf1, 0xd1, 0x77, 0x02});
    site("region4-list", 0x023b9773, {0x0d, 0xd1, 0x77, 0x02}, {0x79, 0xd1, 0x77, 0x02});
    site("region4-count", 0x023b9784, {0x01, 0x00, 0x00, 0x00}, {0x15, 0x00, 0x00, 0x00});
    site("region4-remap", 0x023b978b, {0x01, 0xd1, 0x77, 0x02}, {0xc1, 0xd1, 0x77, 0x02});
    return m;
}

// render-scale, the 3D at the Upscaling quality's render size
// (graphics.render_scale: on whenever live resolution is - BBHOST_RENDER_SCALE=0
// leaves it out): the decoupled half of Kyo's "Decoupled Render Res" patches,
// the scene contexts' size reads made immediates that engine/render_scale.cpp
// writes - at the start from the Resolution and the quality, the full size
// with no upscaler, and again before each live rebuild. engine/render_scale.cpp
// says which sites and why. Not live as a manifest: its sizes change by the
// rebuild, its sites stay.
Manifest render_scale_manifest() {
    Manifest m;
    m.name = "render-scale";
    m.title = "Render scale";
    m.eboot = kEboot109Sha256;
    m.option = "graphics.render_scale";
    m.order = 9200;  // after hd-motion-blur (9100), whose sites are the velocity map's and the blur depth's: none of these
    m.live = false;
    RenderScale rs;
    if (!render_scale_plan(&rs)) rs = RenderScale{1920, 1080, 1920, 1080, 0};  // the bytes it would write, for the expectations
    m.description = "the scene contexts at the Upscaling quality's render size (at the start " + std::to_string(rs.render_w) + "x" +
                    std::to_string(rs.render_h) + " for " + std::to_string(rs.output_w) + "x" + std::to_string(rs.output_h) + ")";
    m.sites = render_scale_sites(rs);
    return m;
}

// shadow-resolution, Shadow resolution (graphics.shadow_resolution): the
// shadow maps at the setting's sizes - Lowest, Low, Default (the game's own,
// no patch), High, Ultra, Giga - or Performance mode's in Default's place.
//
// The sun's cascades: sub_12ac210 sets the shadow object up from the
// renderer's ShadowParam block, whose +0x90 is "Shadow Map Resolution", an
// index into 0x4ca9668 (2048, 1024, 2048, 4096; 2048 for anything else) -
// most areas ask for 3. The size is written in the table's place: `mov eax,
// size` and a jump past the lookup to its compare with the object's size, so
// 512 (no entry of the table's) is a size as any other.
//
// The dynamic lights': the light shadow manager (the graphics manager's
// +0x5878) is made by sub_26440b0 from a block its caller fills at 0x261b0f3:
// the spot lights' (lamps') maps 512x512 and the point lights' (torches')
// cube faces 256x256, each in four LODs that halve (2, 4, 6 and 10 spot maps,
// 2 cubes a LOD), and a size the caller's own block gives (when non-zero) in
// their place - not taken here. A spot map is a D32 depth target, with an
// RGBA8 one per LOD that its maps share; a cube is six RGBA8 faces, with a
// D32 face-size depth per LOD (the game's codes are DXGI's: 0x28, 0x1c).
// They come from the render-target heap (GFX_GraphicsPrivate) as the game
// starts, which the stock heap does not hold at the larger sizes - at all
// 4096, the first area's meshes were not made and the game read through the
// null - so the heap grows by shadow_resolution_heap_mib
// (engine/live_resolution.cpp). Measured at all 4096 (Radeon 8060S, 1080p, an
// indoor room with lamps): the heap's free 1663 MiB less, 69 -> 45 fps.
constexpr std::uint64_t kShadowSunSite = 0x012ac231, kShadowSpotSite = 0x0261b0f3, kShadowCubeSizeSite = 0x0261b116;
struct ShadowSizes {
    std::uint32_t sun, spot, cube;  // 0: the game's own
};
// The setting's entries (host/options.cpp), Default the game's own; then
// Performance mode's.
constexpr ShadowSizes kShadowTiers[] = {{512, 128, 64},    {1024, 256, 128},   {0, 0, 0},
                                        {4096, 1024, 512}, {4096, 2048, 1024}, {4096, 4096, 4096}};
constexpr ShadowSizes kShadowPerformanceMode = {1024, 256, 128};
// The setting's entry for the game's own sizes.
constexpr int kShadowDefaultTier = 2;
// Whether Performance mode's sizes run: only while Shadow resolution is
// Default - a size the player chose wins (Giga under Performance mode ran at
// 1024 until 2026-11-05, and looked like no change at all).
bool shadow_performance_sizes() {
    return host_settings().performance_mode && host_settings().shadow_resolution == kShadowDefaultTier;
}
// The sizes this run asks for: the setting's, or Performance mode's in
// Default's place.
ShadowSizes shadow_sizes() {
    if (shadow_performance_sizes()) return kShadowPerformanceMode;
    return kShadowTiers[std::clamp(host_settings().shadow_resolution, 0, 5)];
}
bool shadow_resolution_on() { return shadow_sizes().sun != 0; }

Manifest shadow_resolution_manifest() {
    const ShadowSizes z = shadow_sizes();
    Manifest m;
    m.name = "shadow-resolution";
    m.title = "Shadow resolution";
    m.eboot = kEboot109Sha256;
    m.description = "the shadow maps: the sun's " + std::to_string(z.sun ? z.sun : 4096) + ", the lamps' " +
                    std::to_string(z.spot ? z.spot : 512) + ", the torches' cube faces " + std::to_string(z.cube ? z.cube : 256) +
                    (shadow_performance_sizes() ? " (Performance mode)" : "");
    m.option = "graphics.shadow_resolution";
    m.order = 9050;
    m.live = false;
    auto imm = [](std::vector<std::uint8_t> op, std::uint32_t v) {
        for (int k = 0; k < 4; ++k) op.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
        return op;
    };
    // movsxd rcx, dword [r14+0x90]; mov eax, 0x800 -> mov eax, size; jmp 0x12ac24d (past the table's lookup)
    std::vector<std::uint8_t> sun = imm({0xb8}, z.sun);
    sun.insert(sun.end(), {0xeb, 0x15, 0x90, 0x90, 0x90, 0x90, 0x90});
    m.sites.push_back(Site{"sun-resolution", kShadowSunSite, {0x49, 0x63, 0x8e, 0x90, 0x00, 0x00, 0x00, 0xb8, 0x00, 0x08, 0x00, 0x00},
                           std::move(sun), true, {}});
    // mov dword [rbp-0x58], 512 (the spot maps) and [rbp-0x44], 256 (the cube faces)
    m.sites.push_back(Site{"spot-size", kShadowSpotSite, imm({0xc7, 0x45, 0xa8}, 512), imm({0xc7, 0x45, 0xa8}, z.spot), true, {}});
    m.sites.push_back(Site{"cube-size", kShadowCubeSizeSite, imm({0xc7, 0x45, 0xbc}, 256), imm({0xc7, 0x45, 0xbc}, z.cube), true, {}});
    // test eax, eax; je -> jmp: the caller's sizes are not taken
    m.sites.push_back(Site{"spot-keep", 0x0261b13f, {0x74, 0x03}, {0xeb, 0x03}, true, {}});
    m.sites.push_back(Site{"cube-keep", 0x0261b149, {0x74, 0x03}, {0xeb, 0x03}, true, {}});
    return m;
}


// skip-intro, Skip company logos (startup.skip_intro): illusion's 1.09 Skip
// Intro. The three company logos (FromSoftware, Japan Studio, SCE) are
// entries the startup flow plays by name; emptying the first two UTF-16
// characters of each name skips them. The warning that the last session may
// not have ended with Exit Game - shown after any session that ended another
// way, which on PC is closing the window - is a step the flow builds only in
// its state 6 (sub_202edf0: a dialog of message 400003, group 204); the jne
// there becoming a jmp builds the step after it, as every other state does
// (the community list's 0x0202EE1C EB73). Story movies are untouched.
std::vector<Manifest> built_in_manifests() {
    Manifest m;
    m.name = "skip-intro";
    m.title = "Skip company logos";
    m.eboot = kEboot109Sha256;
    m.description = "the three company logos and the Exit Game warning are skipped at startup";
    m.option = "startup.skip_intro";
    m.order = 10;
    const std::vector<std::uint8_t> name = {0x4c, 0x00, 0x6f, 0x00}, none(4, 0);
    m.sites.push_back(Site{"logo-fromsoft", 0x04d99138, name, none, true, {}});
    m.sites.push_back(Site{"logo-jstudio", 0x04d99154, name, none, true, {}});
    m.sites.push_back(Site{"logo-sce", 0x04d9916e, name, none, true, {}});
    m.sites.push_back(Site{"startup-warning", 0x0202ee1c, {0x75, 0x73}, {0xeb, 0x73}, true, {}});
    std::vector<Manifest> out;
    out.push_back(std::move(m));
    out.push_back(performance_mode_manifest());
    out.push_back(performance_manifest());
    out.push_back(ssr_manifest());
    out.push_back(render_scale_manifest());
    out.push_back(fov_uncap_manifest());
    out.push_back(all_languages_manifest());
    out.push_back(shadow_resolution_manifest());
    out.push_back(hd_motion_blur_manifest());
    out.push_back(velomap_manifest());
    out.push_back(scene_rgba8_manifest());
    return out;
}

// Patch files bbhost does itself now: the Screen Space Reflections patches
// (the game's half size and the quarter-size variant), Performance Patch v3's
// variants and the uncapped field of view are the Graphics page's Screen
// space reflections, the built-in performance patch and the Field of view
// setting, the HD Motion Blur patches are HD motion blur, Unlock Game Region
// is all-languages without its online cost, the Intel Black Tonemap Fix's
// six bytes are the performance patch's own (always on), and Disable Dynamic
// Light Shadows is the Dynamic light shadows setting. A
// package unpacked over an older one still has them in its patches folder;
// they are passed over, not applied twice.
bool retired_file(const std::string& file) {
    return file.rfind("screen-space-reflections", 0) == 0 || file.rfind("performance-v3-", 0) == 0 || file == "fov-uncap.toml" ||
           file.rfind("hd-motion-blur-", 0) == 0 || file == "unlock-game-region.toml" || file == "intel-tonemap-fix.toml" ||
           file == "disable-dynamic-light-shadow.toml";
}

// By file name, a later directory's file replacing an earlier one's.
std::map<std::string, std::string> manifest_files(const std::vector<std::string>& dirs) {
    std::map<std::string, std::string> files;
    for (const std::string& dir : dirs) {
        std::vector<std::string> names;
        if (!host_list_dir(dir.c_str(), &names)) continue;
        for (const std::string& n : names) {
            if (retired_file(n)) continue;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".toml") == 0) files[n] = dir + "/" + n;
        }
    }
    return files;
}

}  // namespace

bool patch_manifest_option_known(const std::string& option) {
    bool known = false;
    option_on(option, &known);
    return known;
}

bool patch_manifest_on(const std::string& name, bool enabled) {
    const std::string v = config_value("patches." + name);
    if (!v.empty()) return !(v == "false" || v == "0");
    return enabled && !patch_manifest_off(name);
}

bool patch_manifest_off(const std::string& name) {
    const std::string v = config_value("patches." + name);
    if (!v.empty()) return v == "false" || v == "0";
    // The Old Hunters was a PC enhancement in v0.2.9, `old_hunters` under
    // [world]: a bbhost.toml (a package's, a player's) that turned it off there
    // keeps it off until [patches] says otherwise.
    if (name == "old-hunters") {
        const std::string w = config_value("world.old_hunters");
        return w == "false" || w == "0";
    }
    return false;
}

std::vector<PatchListing> patch_manifests_list(const std::string& mods) {
    std::vector<PatchListing> out;
    for (const auto& [file, path] : manifest_files(manifest_dirs(mods))) {
        Manifest m;
        PatchListing l;
        l.path = path;
        if (!patch_file::parse(path, &m, &l.error)) {
            l.name = file.substr(0, file.size() - 5);
        } else {
            l.name = m.name;
            l.title = m.title;
            l.description = m.description;
            l.option = m.option;
            l.eboot = m.eboot;
            l.enabled = m.enabled;
            l.gameplay = m.gameplay;
            l.live = m.live;
            l.group = m.group;
            l.order = m.order;
        }
        out.push_back(l);
    }
    // As they apply: the list they came from, in its order.
    std::stable_sort(out.begin(), out.end(),
                     [](const PatchListing& a, const PatchListing& b) { return std::tie(a.order, a.name) < std::tie(b.order, b.name); });
    return out;
}

const char* patch_manifest_owner(std::uint64_t address, std::size_t n) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    const auto it = g_written.lower_bound(address);
    if (it == g_written.end() || it->first >= address + n) return nullptr;
    // Whoever asks leaves this value to the manifest: it keeps its bytes until
    // the next start (patch_manifest_set_live), so its entries, and the string
    // returned, stay put.
    g_deferred_to.insert(it->second.owner);
    return it->second.owner.c_str();
}

std::vector<std::string> patch_manifests_gameplay() {
    std::lock_guard<std::mutex> lk(g_gameplay_mu);
    return g_gameplay;
}

bool patch_manifests_done() { return g_manifests_done.load(std::memory_order_acquire); }

std::uint64_t shadow_resolution_heap_mib() {
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        const auto it = g_written.lower_bound(kShadowSunSite);
        if (it == g_written.end() || it->first >= kShadowSunSite + 12) return 0;  // not applied (off, or refused)
    }
    const ShadowSizes z = shadow_sizes();
    // Each LOD's maps and the one target they share, 4 bytes a texel.
    const auto lods = [](std::uint64_t size, const int* count, int faces) {
        std::uint64_t b = 0;
        for (int l = 0; l < 4; ++l, size >>= 1) b += (static_cast<std::uint64_t>(count[l]) * faces + 1) * size * size * 4;
        return b;
    };
    static constexpr int kSpots[] = {2, 4, 6, 10}, kCubes[] = {2, 2, 2, 2};
    const auto over = [](std::uint64_t now, std::uint64_t was) { return now > was ? now - was : 0; };
    // Only what is larger than the game's: a smaller size leaves the heap as it is.
    const std::uint64_t more = over(lods(z.spot, kSpots, 1), lods(512, kSpots, 1)) + over(lods(z.cube, kCubes, 6), lods(256, kCubes, 6)) +
                               5ull * over(4ull * z.sun * z.sun, 4ull * 2048 * 2048);  // the sun's five maps, where an area asked for 2048
    if (!more) return 0;
    const std::uint64_t mib = (more >> 20) + 128;  // and a margin
    return (mib + 63) / 64 * 64;
}

void patch_manifests_apply(ElfImage* image) {
    struct Done {
        ~Done() { g_manifests_done.store(true, std::memory_order_release); }
    } done;
    if (const char* e = std::getenv("BBHOST_PATCHES"); e && e[0] == '0') {
        host_log("patch: manifests off (BBHOST_PATCHES=0)");
        return;
    }
    const char* mods = hle_fs_mods_root();
    const std::vector<std::string> dirs = manifest_dirs(mods ? mods : "");
    const std::map<std::string, std::string> files = manifest_files(dirs);
    if (files.empty()) {
        host_log("patch: no manifest files (looked in %s, %s, %s%s); bbhost's own still apply", dirs[0].c_str(), dirs[1].c_str(),
                 dirs[2].c_str(), dirs.size() > 3 ? ", the mods directory" : "");
    }
    std::lock_guard<std::mutex> lk(g_live_mu);
    g_mem = &image->mem;
    // The files for this eboot, then bbhost's own that no file replaces.
    std::vector<Manifest> read;
    for (const auto& [name, path] : files) {
        Manifest m;
        std::string err;
        if (!patch_file::parse(path, &m, &err)) {
            host_log("patch: %s: %s", path.c_str(), err.c_str());
            continue;
        }
        if (m.eboot != image->sha256) {
            host_log("patch: %s skipped: for eboot %.12s..., this is %.12s...", m.name.c_str(), m.eboot.c_str(),
                     image->sha256.c_str());
            continue;
        }
        read.push_back(std::move(m));
    }
    for (Manifest& b : built_in_manifests()) {
        if (b.eboot != image->sha256) continue;
        if (std::none_of(read.begin(), read.end(), [&](const Manifest& m) { return m.name == b.name; })) {
            g_built_in.insert(b.name);
            read.push_back(std::move(b));
        }
    }
    // Which are on, first: a site may step aside for another patch (yields).
    std::vector<Manifest> chosen;
    for (Manifest& m : read) {
        // Every one for this eboot, on or not, for the patch menu.
        g_loaded.push_back({m, false});
        if (!patch_manifest_on(m.name, m.enabled)) {
            // Off by the player, or off until the player turns it on (enabled = false).
            if (m.enabled) host_log("patch: %s off ([patches] in bbhost.toml: the setup window's Patches tab)", m.name.c_str());
            continue;
        }
        if (!m.option.empty()) {
            bool known = false;
            const bool on = option_on(m.option, &known);
            if (!known) {
                host_log("patch: %s skipped: unknown option %s", m.name.c_str(), m.option.c_str());
                continue;
            }
            if (!on) continue;
        }
        if (m.name == "old-hunters") {
            if (const std::string missing = old_hunters_missing(); !missing.empty()) {
                host_log("patch: old-hunters skipped - the game folder has no %s. Its files are older than the 1.09 "
                         "update's, which bring The Old Hunters: copy the update's files over the game folder, replacing "
                         "the ones there (or keep them in a CUSA00900-UPDATE folder beside it). The game stays the edition "
                         "without it.",
                         missing.c_str());
                continue;
            }
        }
        chosen.push_back(std::move(m));
    }
    // In their list's order; one of a group (variants of one patch: the Model
    // LODs, the hunter sizes, Performance Patch v3's two) - the setup window
    // and the patch menu turn the others off, and a bbhost.toml with two on
    // keeps the later.
    std::stable_sort(chosen.begin(), chosen.end(), [](const Manifest& a, const Manifest& b) { return a.order < b.order; });
    std::map<std::string, std::string> group_on;
    for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
        if (!it->group.empty()) group_on.try_emplace(it->group, it->name);
    }
    std::vector<Manifest> apply_list;
    for (Manifest& m : chosen) {
        if (!m.group.empty() && group_on[m.group] != m.name) {
            host_log("patch: %s off: %s, another of group %s, is on (one at a time)", m.name.c_str(), group_on[m.group].c_str(),
                     m.group.c_str());
            continue;
        }
        apply_list.push_back(std::move(m));
    }
    std::set<std::string> on;
    for (const Manifest& m : apply_list) on.insert(m.name);
    std::vector<std::string> gameplay;
    std::stable_sort(g_loaded.begin(), g_loaded.end(),
                     [](const Loaded& a, const Loaded& b) { return std::tie(a.m.order, a.m.name) < std::tie(b.m.order, b.m.name); });
    for (const Manifest& m : apply_list) {
        const bool applied = apply(image, m, on);
        if (applied && m.gameplay) gameplay.push_back(m.name);
        for (Loaded& l : g_loaded) {
            if (l.m.name == m.name) l.applied = applied;
        }
    }
    std::sort(gameplay.begin(), gameplay.end());
    if (!gameplay.empty()) {
        std::string names, tokens;
        for (const std::string& n : gameplay) {
            names += (names.empty() ? "" : ", ") + n;
            tokens += (tokens.empty() ? "" : ",") + patch_rule_token(n);
        }
        host_log("patch: %s change%s the game: the session's online rules carry %s, so the server keeps its bloodstains, "
                 "messages and statistics apart from vanilla ones",
                 names.c_str(), gameplay.size() == 1 ? "s" : "", tokens.c_str());
    }
    std::lock_guard<std::mutex> gk(g_gameplay_mu);
    g_gameplay = std::move(gameplay);
}

namespace {

// Whether a code site is new code where the eboot has zeros (a cave): put in
// before the jumps to it, and left in when they are taken out.
bool cave(const Site& s) {
    return s.code && std::all_of(s.expect.begin(), s.expect.end(), [](std::uint8_t b) { return b == 0; });
}

// One compare-and-swap of the aligned 8-byte word holding [p, p + n): no
// thread sees half of it. The word's other bytes are whatever is there.
void store_in_word(std::uint8_t* p, const std::uint8_t* bytes, std::size_t n) {
    auto* word = reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t{7});
    const std::size_t at = static_cast<std::size_t>(p - reinterpret_cast<std::uint8_t*>(word));
    std::uint64_t was = __atomic_load_n(word, __ATOMIC_ACQUIRE), now;
    do {
        now = was;
        std::memcpy(reinterpret_cast<std::uint8_t*>(&now) + at, bytes, n);
    } while (!__atomic_compare_exchange_n(word, &was, now, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE));
}

bool in_one_word(const std::uint8_t* p, std::size_t n) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
    return n && n <= 8 && (a & ~std::uintptr_t{7}) == ((a + n - 1) & ~std::uintptr_t{7});
}

// [p, p + n) one aligned word at a time, the last first: a thread sees each
// word old or new.
void store_words(std::uint8_t* p, const std::uint8_t* bytes, std::size_t n) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
    for (std::size_t end = n; end;) {
        const std::uintptr_t word = (a + end - 1) & ~std::uintptr_t{7};
        const std::size_t begin = word > a ? static_cast<std::size_t>(word - a) : 0;
        store_in_word(p + begin, bytes + begin, end - begin);
        end = begin;
    }
}

// An opcode a hook starts with (jmp, call, nop, ret, int3): the site starts
// an instruction, where a thread can arrive.
bool starts_insn(std::uint8_t b) { return b == 0xe9 || b == 0xe8 || b == 0xeb || b == 0x90 || b == 0xc3 || b == 0xcc; }

// Writes a site while the game runs (patch_manifest_set_live's rules). The
// page is writable for the length of it.
bool write_live(const Site& s, std::uint8_t* p, const std::uint8_t* bytes) {
    const std::size_t n = s.bytes.size();
    const std::uint64_t at = reinterpret_cast<std::uint64_t>(p);
    const std::uint64_t lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
    if (!(s.code ? guest_protect_rwx(g_mem, lo, hi - lo) : guest_protect_rw(g_mem, lo, hi - lo))) return false;
    if (in_one_word(p, n)) {
        store_in_word(p, bytes, n);
    } else if (s.code && n > 2 && in_one_word(p, 2) && (starts_insn(p[0]) || starts_insn(bytes[0]))) {
        // jmp $ while the rest goes in: a thread arriving waits on it.
        static constexpr std::uint8_t kSelf[2] = {0xeb, 0xfe};
        store_in_word(p, kSelf, 2);
        store_words(p + 2, bytes + 2, n - 2);
        store_in_word(p, bytes, 2);
    } else {
        store_words(p, bytes, n);
    }
    if (s.code && !guest_protect_rx(g_mem, lo, hi - lo)) {
        host_log("patch: cannot restore the protection of 0x%llx", static_cast<unsigned long long>(s.address));
        std::abort();
    }
    return true;
}

// Why a manifest may not change while the game runs, or "" when it may.
std::string held_locked(const Loaded& l) {
    if (!g_mem) return "the game's code is not loaded";
    if (!l.m.live) return "it takes effect as the game starts";
    if (!l.m.option.empty() && !patch_manifest_option_known(l.m.option)) return "it follows " + l.m.option + ", which this bbhost does not know";
    if (g_deferred_to.count(l.m.name)) return "bbhost's frame-rate conversion left its values to it at the start";
    return {};
}

Loaded* find_locked(const std::string& name) {
    for (Loaded& l : g_loaded) {
        if (l.m.name == name) return &l;
    }
    return nullptr;
}

bool take_out_locked(Loaded& l, std::string* why);

std::string join(const std::set<std::string>& names) {
    std::string out;
    for (const std::string& n : names) out += (out.empty() ? "" : ", ") + n;
    return out;
}

// The byte the applied manifests want at `a`, as they apply at the start: the
// last in order with a site over it that does not step aside, else the
// eboot's own. `who` is that manifest (null: the eboot's).
std::uint8_t layered_byte(std::uint64_t a, std::uint8_t now, const std::set<std::string>& on, const Loaded** who) {
    std::uint8_t b = original_byte(a, now);
    *who = nullptr;
    for (const Loaded& o : g_loaded) {
        if (!o.applied) continue;
        for (const Site& s : o.m.sites) {
            if (a < s.address || a >= s.address + s.bytes.size() || yielded(s, on)) continue;
            b = s.bytes[a - s.address];
            *who = &o;
        }
    }
    return b;
}

// A site's bytes made what the applied manifests want there now, and the
// owners noted: a manifest put in under a later one leaves the later one's
// bytes, and one taken out hands its bytes to the next under it, or the
// eboot's back. `keep_cave`: new code where the eboot had zeros stays (a
// thread may still be inside it). False when its page cannot be written.
bool relayer_locked(const Site& s, bool keep_cave, bool* wrote, const std::string& self, std::set<std::string>* others) {
    std::set<std::string> on;
    for (const Loaded& o : g_loaded) {
        if (o.applied) on.insert(o.m.name);
    }
    GuestMemory& mem = *g_mem;
    auto* p = static_cast<std::uint8_t*>(guest_ptr(mem, mem.slide + (s.address - kPreferredGuestSlide)));
    const std::size_t n = s.bytes.size();
    std::vector<std::uint8_t> want(p, p + n), original(n);
    std::vector<const Loaded*> owner(n, nullptr);
    for (std::size_t i = 0; i < n; ++i) {
        original[i] = original_byte(s.address + i, p[i]);
        want[i] = layered_byte(s.address + i, p[i], on, &owner[i]);
        if (keep_cave && cave(s) && !owner[i]) want[i] = p[i];
    }
    *wrote = std::memcmp(p, want.data(), n) != 0;
    if (*wrote && !write_live(s, p, want.data())) return false;
    for (std::size_t i = 0; i < n; ++i) {
        if (owner[i] && owner[i]->m.name != self) others->insert(owner[i]->m.name);
        if (owner[i] && want[i] != original[i]) {
            g_written[s.address + i] = Written{owner[i]->m.name, original[i]};
        } else {
            g_written.erase(s.address + i);
        }
    }
    return true;
}

// The manifest's bytes in, checked first as patch_manifests_apply checks them.
bool put_in_locked(Loaded& l, std::string* why) {
    if (l.applied) return true;
    if (std::string h = held_locked(l); !h.empty()) {
        *why = h;
        return false;
    }
    if (l.m.name == "old-hunters" && !old_hunters_missing().empty()) {
        *why = "the game folder lacks The Old Hunters' files";
        return false;
    }
    // One of a group: the one on goes first.
    if (!l.m.group.empty()) {
        for (Loaded& o : g_loaded) {
            if (&o == &l || !o.applied || o.m.group != l.m.group) continue;
            std::string w;
            if (!take_out_locked(o, &w)) {
                *why = (o.m.title.empty() ? o.m.name : o.m.title) + ", another of its kind, cannot be taken out now (" + w + ")";
                return false;
            }
        }
    }
    std::set<std::string> on;
    for (const Loaded& o : g_loaded) {
        if (o.applied) on.insert(o.m.name);
    }
    on.insert(l.m.name);
    GuestMemory& mem = *g_mem;
    for (const Site& s : l.m.sites) {
        if (yielded(s, on)) continue;
        const std::uint64_t off = s.address - kPreferredGuestSlide;
        if (s.address < kPreferredGuestSlide || off >= mem.size || s.expect.size() > mem.size - off) {
            *why = "a site is outside the image";
            return false;
        }
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, mem.slide + off));
        for (std::size_t i = 0; i < s.expect.size(); ++i) {
            if (p[i] == s.expect[i] || p[i] == s.bytes[i] || original_byte(s.address + i, p[i]) == s.expect[i]) continue;
            char at[32];
            std::snprintf(at, sizeof(at), "0x%llx", static_cast<unsigned long long>(s.address + i));
            *why = std::string("the game's code at ") + at + " is not what it expects (bbhost or another patch changed it)";
            return false;
        }
    }
    // The new code first, the jumps to it after.
    l.applied = true;
    std::vector<const Site*> order;
    for (const Site& s : l.m.sites) order.push_back(&s);
    std::stable_partition(order.begin(), order.end(), [](const Site* s) { return cave(*s); });
    int written = 0;
    std::set<std::string> others;
    for (const Site* s : order) {
        bool wrote = false;
        if (!relayer_locked(*s, false, &wrote, l.m.name, &others)) {
            *why = "a page could not be made writable";
            take_out_locked(l, why);
            return false;
        }
        written += wrote;
    }
    host_log("patch: %s: on while the game runs, %d of %zu sites written%s%s", l.m.name.c_str(), written, l.m.sites.size(),
             others.empty() ? "" : " (bytes shared with ", others.empty() ? "" : (join(others) + ", layered in the list's order)").c_str());
    return true;
}

// Its bytes out: what the applied manifests under it want there, or the
// eboot's own.
bool take_out_locked(Loaded& l, std::string* why) {
    if (!l.applied) return true;
    if (std::string h = held_locked(l); !h.empty()) {
        *why = h;
        return false;
    }
    GuestMemory& mem = *g_mem;
    // Its bytes must still be its own: one the game or bbhost changed since
    // means a writer that would find the eboot's put back under it.
    for (const Site& s : l.m.sites) {
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, mem.slide + (s.address - kPreferredGuestSlide)));
        for (std::size_t i = 0; i < s.bytes.size(); ++i) {
            const auto it = g_written.find(s.address + i);
            if (it == g_written.end() || it->second.owner != l.m.name || p[i] == s.bytes[i]) continue;
            char at[32];
            std::snprintf(at, sizeof(at), "0x%llx", static_cast<unsigned long long>(s.address + i));
            *why = std::string("its bytes at ") + at + " were changed after it wrote them";
            return false;
        }
    }
    // The jumps first; new code where the eboot had zeros stays.
    l.applied = false;
    std::vector<const Site*> order;
    for (const Site& s : l.m.sites) order.push_back(&s);
    std::stable_partition(order.begin(), order.end(), [](const Site* s) { return !cave(*s); });
    int restored = 0;
    std::set<std::string> others;
    for (const Site* s : order) {
        bool wrote = false;
        if (!relayer_locked(*s, true, &wrote, l.m.name, &others)) {
            *why = "a page could not be made writable";
            l.applied = true;
            return false;
        }
        restored += wrote;
    }
    host_log("patch: %s: off while the game runs, %d of %zu sites put back%s%s", l.m.name.c_str(), restored, l.m.sites.size(),
             others.empty() ? "" : " (bytes shared with ", others.empty() ? "" : (join(others) + ", layered in the list's order)").c_str());
    return true;
}

}  // namespace

PatchLiveState patch_manifest_live_state(const std::string& name) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    PatchLiveState st;
    const Loaded* l = find_locked(name);
    if (!l) return st;
    st.known = true;
    st.applied = l->applied;
    st.why = held_locked(*l);
    st.can_change = st.why.empty();
    return st;
}

bool patch_manifest_set_live(const std::string& name, bool on, std::string* why) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    Loaded* l = find_locked(name);
    std::string w;
    if (!l) w = "it was not read for this eboot at the start";
    const bool ok = l && (on ? put_in_locked(*l, &w) : take_out_locked(*l, &w));
    if (!ok) host_log("patch: %s: %s at the next start - %s", name.c_str(), on ? "on" : "off", w.c_str());
    if (why) *why = w;
    return ok;
}

bool patch_manifest_refresh_live(const std::string& name, bool on, bool* changed, std::string* why) {
    if (changed) *changed = false;
    // Made again from the settings now, outside the lock (the settings take their own).
    Manifest fresh;
    bool found = false;
    for (Manifest& b : built_in_manifests()) {
        if (b.name != name) continue;
        fresh = std::move(b);
        found = true;
    }
    std::lock_guard<std::mutex> lk(g_live_mu);
    Loaded* l = find_locked(name);
    std::string w;
    if (!l) w = "it was not read for this eboot at the start";
    else if (!found || !g_built_in.count(name)) w = "a patch file of that name stands in for bbhost's own";
    else if (!g_mem) w = "the game's code is not loaded";
    bool ok = w.empty();
    if (ok) {
        // The game's bytes under its old sites and its new ones, before and after.
        std::vector<std::pair<std::uint64_t, std::size_t>> spans;
        for (const Site& s : l->m.sites) spans.emplace_back(s.address, s.bytes.size());
        for (const Site& s : fresh.sites) spans.emplace_back(s.address, s.bytes.size());
        const auto bytes_now = [&] {
            std::vector<std::uint8_t> v;
            for (const auto& [at, n] : spans) {
                const auto* p = static_cast<const std::uint8_t*>(guest_ptr(*g_mem, g_mem->slide + (at - kPreferredGuestSlide)));
                v.insert(v.end(), p, p + n);
            }
            return v;
        };
        const std::vector<std::uint8_t> before = bytes_now();
        if (l->applied) ok = take_out_locked(*l, &w);
        if (ok) {
            l->m.sites = std::move(fresh.sites);
            if (on) ok = put_in_locked(*l, &w);
        }
        if (changed) *changed = bytes_now() != before;
    }
    if (!ok) host_log("patch: %s: made again %s at the next start - %s", name.c_str(), on ? "on" : "off", w.c_str());
    if (why) *why = w;
    return ok;
}
