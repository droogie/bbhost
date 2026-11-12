#include "engine/world_pause.h"

#include "core/elf.h"
#include "core/memory.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "guest_abi.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace {

// sub_1d3ac10, the in-game update: push rbp; mov rbp, rsp; push r15..r12;
// push rbx; sub rsp, 0x1c8 - none of it rip-relative.
constexpr std::uint64_t kInGameUpdate = 0x1d3ac10;
constexpr std::uint8_t kInGameUpdatePrologue[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                    0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xc8, 0x01, 0x00, 0x00};
// The step's bytes: the world held, its effects and cloth held with it, and the
// game's own leaving (which holds the world itself); the camera root and its
// debug free camera, which the held update reads with no check.
constexpr std::uint64_t kHeld = 0x150, kHeldEffects = 0x151, kLeaving = 0x153;
constexpr std::uint64_t kCameraRoot = 0x110, kDebugCamera = 0xd0;

bool g_installed = false;
std::mutex g_mu;
std::map<std::string, bool> g_holds;    // under g_mu: each plugin that asks
std::atomic<bool> g_want{false};        // any of them
std::atomic<std::uint32_t> g_steps{0};  // updates still to run while held
std::atomic<bool> g_held{false};

// The update thread's own state.
std::uint64_t g_step = 0;    // the step seen last
bool g_ours = false;         // the bytes are ours: we set them
bool g_told = false;         // "held" logged, until the world runs again

std::uint64_t read64(std::uint64_t at) {
    std::uint64_t v = 0;
    return at && host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), &v, sizeof(v)) ? v : 0;
}

void want_update_locked() {
    bool any = false;
    for (const auto& [name, on] : g_holds) any = any || on;
    g_want.store(any, std::memory_order_relaxed);
    if (!any) g_steps.store(0, std::memory_order_relaxed);
}

GUEST_ABI std::int64_t in_game_update_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t step = saved[5];  // rdi
    if (!step) return 0;
    auto* bytes = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(step));
    if (step != g_step) {
        // A load made a new step: what was ours went with the old one, and
        // every hold is let go - a world never starts held.
        if (g_step && g_want.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lk(g_mu);
            g_holds.clear();
            want_update_locked();
            host_log("world pause: a new world (in-game step %llx): every hold let go", static_cast<unsigned long long>(step));
        }
        g_step = step;
        g_ours = false;
        g_told = false;
        g_held.store(false, std::memory_order_relaxed);
    }
    if (!g_want.load(std::memory_order_relaxed) && !g_ours) return 0;
    const bool leaving = bytes[kLeaving] != 0;
    bool hold = g_want.load(std::memory_order_relaxed) && !leaving;
    if (hold) {
        // A frame advance: this update runs.
        std::uint32_t left = g_steps.load(std::memory_order_relaxed);
        while (left && !g_steps.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
        }
        if (left) hold = false;
    }
    if (hold && !read64(read64(step + kCameraRoot) + kDebugCamera)) {
        static bool told = false;
        if (!told) host_log("world pause: this world's camera root has no debug camera, which the held update reads; not held");
        told = true;
        hold = false;
    }
    if (hold) {
        if (!g_told) host_log("world pause: the world held (in-game step %llx)", static_cast<unsigned long long>(step));
        g_told = true;
        bytes[kHeld] = 1;
        bytes[kHeldEffects] = 1;
        g_ours = true;
    } else if (g_ours) {
        if (!leaving) {
            bytes[kHeld] = 0;
            bytes[kHeldEffects] = 0;
        }
        g_ours = false;
    }
    if (g_told && !g_want.load(std::memory_order_relaxed)) {
        host_log("world pause: the world runs again");
        g_told = false;
    }
    g_held.store(g_ours, std::memory_order_relaxed);
    return 0;
}

}  // namespace

void world_pause_install(ElfImage* image) {
    if (g_installed || !image) return;
    const std::uint64_t at = image->mem.slide + (kInGameUpdate - kPreferredGuestSlide);
    if (!engine_prologue_hook(image, at, kInGameUpdatePrologue, sizeof(kInGameUpdatePrologue),
                              reinterpret_cast<void*>(&in_game_update_hook))) {
        host_log("world pause: 0x%llx is not the in-game update as expected; world_pause does nothing",
                 static_cast<unsigned long long>(kInGameUpdate));
        return;
    }
    g_installed = true;
}

bool world_pause_set(const std::string& owner, bool on) {
    if (!g_installed || owner.empty()) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    const bool was = g_holds.count(owner) && g_holds[owner];
    if (on) g_holds[owner] = true;
    else g_holds.erase(owner);
    if (was != on) host_log("world pause: %s %s", owner.c_str(), on ? "holds the world" : "lets the world go");
    want_update_locked();
    return true;
}

bool world_pause_step(const std::string& owner, std::uint32_t updates) {
    if (!g_installed) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_holds.find(owner);
    if (it == g_holds.end() || !it->second) return false;
    g_steps.fetch_add(updates, std::memory_order_relaxed);
    host_log("world pause: %s lets %u more update(s) run", owner.c_str(), updates);
    return true;
}

bool world_pause_held() { return g_held.load(std::memory_order_relaxed); }

void world_pause_drop(const std::string& owner) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_holds.erase(owner)) host_log("world pause: %s turned off; its hold let go", owner.c_str());
    want_update_locked();
}
