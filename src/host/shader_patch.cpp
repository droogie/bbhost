#include "host/shader_patch.h"

#include "host/options.h"
#include "host/settings.h"
#include "log.h"

#include <atomic>
#include <cstring>

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
    // YEBIS depth of field's composite (d3ca03f3+111fce32):
    //   v_mad_f32 v0, v5, s0, v0 clamp       ; t = clamp(scene.a * k + b)
    //   rgb = scene + t * (blurred - scene)
    // t = 0 - v_mov_b32 v0, 0 and an s_nop for the second word - leaves the
    // scene as it was. The game's own switch (the je at 0x25d7bbc the
    // community patch forces) stops YEBIS being fed its focus parameters,
    // which is only "off" if it never had any: turned off later, the last
    // ones stay. The blend is where the effect lands, so it is off here.
    {"111fce32", &HostSettings::depth_of_field, "depth of field", {0xd2820800, 0x04000105}, {0x7e000280, 0xbf800000}},
};
constexpr std::size_t kCount = sizeof(kPatches) / sizeof(kPatches[0]);

thread_local std::uint64_t g_seen_serial = 0;
thread_local std::uint32_t g_mask = 0;

}  // namespace

std::uint32_t shader_patch_mask() {
    const std::uint64_t serial = host_opt_serial();
    if (g_seen_serial != serial) {
        std::uint32_t m = 0;
        const HostSettings hs = host_settings();
        for (std::size_t i = 0; i < kCount; ++i) {
            if (!(hs.*(kPatches[i].setting))) m |= 1u << i;
        }
        g_mask = m;
        g_seen_serial = hs.serial;
    }
    return g_mask;
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
