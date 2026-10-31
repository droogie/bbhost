#include "engine/lua_events.h"

#include "core/elf.h"
#include "core/portable.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "guest_abi.h"
#include "log.h"

#include <deque>
#include <mutex>
#include <string>

namespace {

constexpr std::uint64_t kLuaDispatchByName = 0x1739870;  // (context = *(SprjLuaEventMan + 8), const char* name)
constexpr std::uint64_t kSprjLuaEventMan = 0x593b0c8;    // the manager's singleton slot
constexpr std::size_t kQueueMax = 64;
// The cutscene request event scripts make (bank 2002, 0x1bc5585) and Lua's
// RequestRemo: (unused, cutscene id, playback method, -1, -1, the player's
// entity id or -1).
constexpr std::uint64_t kRequestRemo = 0x171a9f0;

std::uint64_t g_slide = 0;
bool g_ok = false;  // the 1.09 build: the addresses above mean what they say
std::mutex g_mu;
std::deque<std::string> g_queue;  // under g_mu
struct Remo {
    std::int32_t id, method;
};
std::deque<Remo> g_remos;  // under g_mu
std::uint64_t g_raised = 0, g_dropped = 0;

std::uint64_t guest_of(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }

}  // namespace

void lua_events_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_ok = image->sha256 == kEboot109Sha256;
}

bool lua_event_queue(const char* name) {
    if (!g_ok || !name || !name[0]) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_queue.size() >= kQueueMax) {
        ++g_dropped;
        return false;
    }
    g_queue.emplace_back(name);
    return true;
}

bool remo_request_queue(std::int32_t id, std::int32_t method) {
    if (!g_ok || id <= 0) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    g_remos.push_back({id, method});
    return true;
}

void lua_events_tick() {
    std::deque<std::string> now;
    std::deque<Remo> remos;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_queue.empty() && g_remos.empty()) return;
        now.swap(g_queue);
        remos.swap(g_remos);
    }
    for (const Remo& r : remos) {
        hle_call_guest6(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(kRequestRemo))), 0, r.id, r.method, -1, -1, 10000);
        host_log("lua events: cutscene %d requested (method %d)", r.id, r.method);
    }
    if (now.empty()) return;
    std::uint64_t man = 0;
    if (!host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(guest_of(kSprjLuaEventMan))), &man, 8) || !man) {
        host_log("lua events: no event manager yet; %zu event(s) dropped", now.size());
        return;
    }
    std::uint64_t ctx = 0;
    if (!host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(man + 8)), &ctx, 8) || !ctx) return;
    for (const std::string& name : now) {
        const auto r = hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(kLuaDispatchByName))), ctx,
                                                    name.c_str());
        ++g_raised;
        host_log("lua events: raised %s -> %lld", name.c_str(), static_cast<long long>(r));
    }
}
