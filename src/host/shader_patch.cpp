#include "host/shader_patch.h"

#include "host/options.h"
#include "host/settings.h"
#include "log.h"

#include <atomic>
#include <cstring>
#include <mutex>

namespace {

struct Patch {
    const char* name;                // the program's footer hash
    bool HostSettings::*setting;     // applied while this setting is off (host/settings.h)
    const char* what;
    std::uint32_t expect[2];
    std::uint32_t write[2];
};

const Patch kPatches[] = {
    // YEBIS motion blur. After decoding the velocity:
    //   v_add_f32 v10, |v1|, |v9|            ; |vx| + |vy|
    //   v_cmp_le_f32 vcc, s32, v10           ; C#[0x2f2] <= it: blur
    //   s_and_saveexec_b64 s[32:33], vcc
    // The else side writes the centre sample through untouched, so an
    // always-false compare (v_cmp_f_f32, opcode 0) leaves every pixel as it was.
    {"e0305cef", &HostSettings::motion_blur, "motion blur", {0x7c061420, 0xbea0246a}, {0x7c001420, 0xbea0246a}},
    // Its other build, which the camera blur (HD motion blur's) draws with -
    // twice a frame at the output size, after a pass that packs the velocity
    // map (7ea47480+d3c8bb21) - the same three instructions. Without it the
    // Motion blur setting left this blur on, and rebuilt motion vectors' map
    // (raw NDC motion, which it reads as 0.5 off centre) smeared every mesh.
    {"157a299c", &HostSettings::motion_blur, "motion blur (camera blur)", {0x7c061420, 0xbea0246a}, {0x7c001420, 0xbea0246a}},
    // Depth of field is not a patch: YEBIS's own enable, cleared each view
    // (engine/graphics_patch.cpp, kDofEnable). Zeroing one composite's blend
    // (d3ca03f3+111fce32) left the multi-level ones - lower resolutions,
    // cutscenes, strong blur - and every pass before them running.
};
constexpr std::size_t kCount = sizeof(kPatches) / sizeof(kPatches[0]);

std::atomic<std::uint32_t> g_mask{0};
std::atomic<std::uint32_t> g_settings_mask{0};  // what the settings call for
std::atomic<int> g_look_blur{-1};  // shader_patch_look

// The settings' mask with a plugin's look over it (the settings' thread and
// the look's call it, one at a time).
void mask_update() {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    std::uint32_t m = g_settings_mask.load(std::memory_order_acquire);
    const int blur = g_look_blur.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < kCount; ++i) {
        const int look = kPatches[i].setting == &HostSettings::motion_blur ? blur : -1;
        if (look == 1) m &= ~(1u << i);
        if (look == 0) m |= 1u << i;
    }
    g_mask.store(m, std::memory_order_release);
}

// The settings' listener: which patches they call for, kept for every draw.
void settings_changed(const HostSettings& hs) {
    std::uint32_t m = 0;
    for (std::size_t i = 0; i < kCount; ++i) {
        // Performance mode takes motion blur off here too: it leaves the
        // scene's motion blur flag on for the velocity map with an upscaler
        // (engine/patch_manifest.cpp, performance-mode).
        const bool off = !(hs.*(kPatches[i].setting)) || (kPatches[i].setting == &HostSettings::motion_blur &&
                                                          (hs.performance_mode || host_opt_motion_rebuilt_bits()));
        if (off) m |= 1u << i;
    }
    g_settings_mask.store(m, std::memory_order_release);
    mask_update();
}

}  // namespace

std::uint32_t shader_patch_mask() {
    // Listening from the first ask on: told the settings at once, then each change.
    static const bool listening = (host_settings_listen(&settings_changed), true);
    (void)listening;
    return g_mask.load(std::memory_order_acquire);
}

void shader_patch_look(int motion_blur) {
    (void)shader_patch_mask();  // the settings' listener in first
    g_look_blur.store(motion_blur, std::memory_order_relaxed);
    mask_update();
}

bool shader_patch_touches(const std::string& name) {
    for (const Patch& p : kPatches) {
        if (name == p.name) return true;
    }
    return false;
}

void shader_patch_apply(const std::string& name, std::vector<std::uint32_t>& words) {
    const std::uint32_t mask = shader_patch_mask();
    for (std::size_t k = 0; k < kCount; ++k) {
        const Patch& p = kPatches[k];
        if (name != p.name || !(mask & (1u << k))) continue;
        std::size_t at = words.size(), found = 0;
        for (std::size_t i = 0; i + 1 < words.size(); ++i) {
            if (words[i] == p.expect[0] && words[i + 1] == p.expect[1]) {
                at = i;
                ++found;
            }
        }
        static std::atomic<int> logs{0};
        if (found != 1) {
            if (logs.fetch_add(1) < 8) {
                host_log("shader patch: %s off not applied - %s holds the expected instructions %zu times, not once",
                         p.what, p.name, found);
            }
            continue;
        }
        std::memcpy(&words[at], p.write, sizeof(p.write));
        if (logs.fetch_add(1) < 16) {
            host_log("shader patch: %s off - %s patched at +0x%zx", p.what, p.name, at * 4);
        }
    }
}
