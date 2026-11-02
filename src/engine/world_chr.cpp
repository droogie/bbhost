#include "engine/world_chr.h"

#include "core/elf.h"
#include "core/portable.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "log.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace {

constexpr std::uint64_t kWorldChrMan = 0x593e878;  // WorldChrMan*
constexpr std::uint64_t kCameraRoot = 0x593e860;   // the camera owner's root object

std::uint64_t g_slide = 0;
bool g_ok = false;
bool g_warp_enabled = false;  // BBHOST_TEST_WARP=1, or a gameplay plugin the player turned on
bool g_unlock = false;        // BBHOST_TEST_UNLOCK_LAMPS=1

template <typename T>
bool rd(std::uint64_t at, T* out) {
    return at && host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, sizeof(T));
}

std::uint64_t ptr(std::uint64_t at) {
    std::uint64_t p = 0;
    return rd(at, &p) ? p : 0;
}

std::uint64_t slot(std::uint64_t bn) { return g_ok ? g_slide + (bn - kPreferredGuestSlide) : 0; }

bool finite3(const float v[3]) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

}  // namespace

void world_chr_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_ok = image->sha256 == kEboot109Sha256;
    if (const char* e = std::getenv("BBHOST_TEST_WARP"); e && e[0] == '1') {
        g_warp_enabled = true;
        host_log("world: BBHOST_TEST_WARP=1 - plugins may move the player (a test switch)");
    }
    if (const char* e = std::getenv("BBHOST_TEST_UNLOCK_LAMPS"); g_ok && e && e[0] == '1') {
        g_unlock = true;
        host_log("world: BBHOST_TEST_UNLOCK_LAMPS=1 - every lamp is lit at each load (a test switch)");
    }
}

void world_warp_allow() {
    if (!g_warp_enabled) host_log("world: warps on for a gameplay plugin the player turned on");
    g_warp_enabled = true;
}

bool world_player_position(float xyz[3], float* yaw) {
    const std::uint64_t man = ptr(slot(kWorldChrMan));
    const std::uint64_t chr = man ? ptr(man + 0x60) : 0;
    const std::uint64_t modules = chr ? ptr(chr + 0x3b0) : 0;
    const std::uint64_t transform = modules ? ptr(modules + 0x68) : 0;
    float v[3] = {}, y = 0;
    if (!transform || !rd(transform + 0x1e0, &v) || !rd(transform + 0x1d4, &y) || !finite3(v)) return false;
    if (xyz) {
        xyz[0] = v[0];
        xyz[1] = v[1];
        xyz[2] = v[2];
    }
    if (yaw) *yaw = y;
    return true;
}

std::uint64_t world_camera_owner() {
    const std::uint64_t root = ptr(slot(kCameraRoot));
    return root ? ptr(root + 0x2830) : 0;
}

bool world_camera(float pos[3], float focus[3]) {
    const std::uint64_t owner = world_camera_owner();
    const std::uint64_t cam = owner ? ptr(owner + 0x60) : 0;
    float p[3] = {}, f[3] = {};
    if (!cam || !rd(cam + 0x40, &p) || !rd(cam + 0xd0, &f) || !finite3(p) || !finite3(f)) return false;
    if (f[0] == 0 && f[1] == 0 && f[2] == 0) return false;  // between areas the camera is reset
    for (int i = 0; i < 3; ++i) {
        if (pos) pos[i] = p[i];
        if (focus) focus[i] = f[i];
    }
    return true;
}

bool world_player_block(std::uint32_t* block) {
    const std::uint64_t man = ptr(slot(kWorldChrMan));
    const std::uint64_t chr = man ? ptr(man + 0x60) : 0;
    std::uint32_t b = 0;
    if (!chr || !rd(chr + 0x3f8, &b) || b == 0xffffffffu) return false;
    if (block) *block = b;
    return true;
}

namespace {
constexpr std::uint64_t kWorldRes = 0x593b148;          // the world manager the warps are called on (its +8 holds WorldRes)
constexpr std::uint64_t kPreparePlayerWarp = 0x194b110;  // (WorldRes*, const int* block, const vec4* pos, const vec4* rot)
std::mutex g_warp_mu;
bool g_warp_pending = false;  // under g_warp_mu
float g_warp_xyz[3] = {}, g_warp_yaw = 0;
std::uint32_t g_warp_block = 0;
}  // namespace

bool world_player_warp_queue(std::uint32_t block, const float xyz[3], float yaw) {
    if (!g_ok || !g_warp_enabled || !xyz || !std::isfinite(yaw)) return false;
    const float v[3] = {xyz[0], xyz[1], xyz[2]};
    if (!finite3(v) || !world_player_block(nullptr)) return false;
    std::lock_guard<std::mutex> lk(g_warp_mu);
    g_warp_pending = true;
    g_warp_xyz[0] = v[0];
    g_warp_xyz[1] = v[1];
    g_warp_xyz[2] = v[2];
    g_warp_yaw = yaw;
    g_warp_block = block;
    return true;
}

namespace {

// The lamps. Each map's event script starts
// event 7000 once per lamp with (warp object entity, lamp entity, the flag
// that must be on before it can be lit - 999 for none, flag base); the lamp
// DB (FrpgNetMan +0xc70, bbhost/engine/frpg/bonfire_db.hpp) registers the base,
// and base+10 is the lamp's lit flag, base+11 "selected" (the last one
// rested at in that map). Extracted from dvdroot_ps4/event/m*.emevd.dcx by
// tools/mapval_lamps.py; the chalice dungeons (m29) have none there.
// Map loads re-register their own lamps and clear those bits again, which
// is why the unlock runs at every load.
struct Lamp {
    std::int32_t entity, flag_base;
};
constexpr Lamp kLamps[] = {
    {2111950, 12117800},                                                                // m21_01 Abandoned Old Workshop
    {2201950, 12207800}, {2201951, 12207820},                                           // m22 Hemwick
    {2301950, 12307800}, {2301951, 12307820}, {2301952, 12307840},                      // m23 Old Yharnam
    {2401950, 12407800}, {2401951, 12407820},                                           // m24_00 Cathedral Ward
    {2411950, 12417800}, {2411951, 12417820}, {2411952, 12417840}, {2411953, 12417860},  // m24_01 Central Yharnam
    {2421950, 12427800}, {2421951, 12427820}, {2421952, 12427840},                      // m24_02 Upper Cathedral Ward
    {2501950, 12507800}, {2501951, 12507820}, {2501952, 12507840},                      // m25 Cainhurst
    {2601950, 12607800}, {2601951, 12607820}, {2601952, 12607840}, {2601953, 12607860},  // m26 Mensis
    {2701950, 12707800}, {2701951, 12707820},                                           // m27 Forbidden Woods
    {2801950, 12807800}, {2801951, 12807820}, {2801952, 12807840}, {2801953, 12807860},  // m28 Yahar'gul
    {3201950, 13207800}, {3201951, 13207820}, {3201952, 13207840}, {3201953, 13207860},  // m32 Byrgenwerth
    {3301950, 13307800}, {3301951, 13307820},                                           // m33 Nightmare Frontier
    {3401950, 13407800}, {3401951, 13407820}, {3401952, 13407840}, {3401953, 13407860},  // m34 Hunter's Nightmare
    {3501950, 13507800}, {3501951, 13507820}, {3501952, 13507840},                      // m35 Research Hall
    {3601950, 13607800}, {3601951, 13607820}, {3601952, 13607840},                      // m36 Fishing Hamlet
};
constexpr std::uint64_t kEventFlagMan = 0x593b100;       // SprjEventFlagMan*
constexpr std::uint64_t kSetEventFlag = 0x17cfcc0;       // (man, id, on)
constexpr std::uint64_t kIsEventFlag = 0x17cfc00;        // (man, id, int* exists) -> on
constexpr std::uint64_t kWarpNextStageBonfire = 0x172e050;  // (lua state, unused; return point entity id)
constexpr std::uint64_t kSetReturnPointEntity = 0x196d110;  // GameStateMan: ReturnPointParam row -> packed return point (+0x1528)
constexpr std::uint64_t kSetUseReturnPoint = 0x196d0e0;     // GameStateMan +0x1524: the next load places the player there
std::uint32_t g_unlock_block = 0;  // the block the last unlock ran in (one per load)
std::int32_t g_lamp_warp = 0;      // under g_warp_mu

void unlock_lamps(std::uint32_t block) {
    const std::uint64_t man = ptr(slot(kEventFlagMan));
    if (!man) return;
    int set = 0, already = 0, missing = 0;
    for (const Lamp& l : kLamps) {
        const std::uint32_t lit = static_cast<std::uint32_t>(l.flag_base + 10);
        std::int32_t exists = 0;
        const auto on = hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot(kIsEventFlag))), man, lit, &exists);
        if (!exists) {
            ++missing;
            continue;
        }
        if (on) {
            ++already;
            continue;
        }
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot(kSetEventFlag))), man, lit, 1);
        ++set;
    }
    host_log("world: BBHOST_TEST_UNLOCK_LAMPS in block 0x%08x: %d lamp lit flags set, %d were set, %d have no flag block in this save", block,
             set, already, missing);
}

}  // namespace

bool world_lamp_warp_queue(std::int32_t return_point) {
    if (!g_ok || !g_warp_enabled || return_point <= 0 || !world_player_block(nullptr)) return false;
    std::lock_guard<std::mutex> lk(g_warp_mu);
    g_lamp_warp = return_point;
    return true;
}

void world_chr_tick() {
    if (g_unlock) {
        std::uint32_t b = 0;
        if (world_player_block(&b) && b != g_unlock_block) {
            g_unlock_block = b;
            unlock_lamps(b);
        }
    }
    std::int32_t lamp = 0;
    {
        std::lock_guard<std::mutex> lk(g_warp_mu);
        lamp = g_lamp_warp;
        g_lamp_warp = 0;
    }
    if (lamp) {
        std::uint32_t from = 0;
        world_player_block(&from);
        // The lamp as the return point, as resting at it makes it, and the
        // load told to use it; then the stage change.
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot(kSetReturnPointEntity))), lamp);
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot(kSetUseReturnPoint))), 1);
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot(kWarpNextStageBonfire))), 0, lamp);
        host_log("world: lamp travel from block 0x%08x to return point %d (WarpNextStage_Bonfire)", from, lamp);
    }
    float xyz[3], yaw;
    std::uint32_t want = 0;
    {
        std::lock_guard<std::mutex> lk(g_warp_mu);
        if (!g_warp_pending) return;
        g_warp_pending = false;
        xyz[0] = g_warp_xyz[0];
        xyz[1] = g_warp_xyz[1];
        xyz[2] = g_warp_xyz[2];
        yaw = g_warp_yaw;
        want = g_warp_block;
    }
    // Block 0 is the player's own: the in-map path. Another block goes
    // through the same routine, which rebinds the player to it - only for a
    // block that is resident (a neighbour the world streams in).
    std::uint32_t block = 0, from = 0;
    const std::uint64_t res = ptr(slot(kWorldRes));
    if (!res || !world_player_block(&from)) {
        host_log("world: warp dropped (no player or no WorldRes)");
        return;
    }
    block = want ? want : from;
    alignas(16) std::int32_t blk[4] = {static_cast<std::int32_t>(block), 0, 0, 0};
    alignas(16) float pos[4] = {xyz[0], xyz[1], xyz[2], 0.0f};
    alignas(16) float rot[4] = {0.0f, yaw, 0.0f, 0.0f};  // pitch, yaw, roll (radians), as the transform keeps them at +0x1d0
    hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot(kPreparePlayerWarp))), res, blk, pos, rot);
    std::uint32_t now = 0;
    world_player_block(&now);
    host_log("world: warp from block 0x%08x into 0x%08x at %.3f %.3f %.3f yaw %.3f; now in 0x%08x", from, block, xyz[0], xyz[1], xyz[2],
             yaw, now);
}
