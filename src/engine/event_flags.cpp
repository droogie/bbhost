#include "engine/play_history.h"
#include "engine/event_flags.h"

#include "core/elf.h"
#include "core/portable.h"
#include "core/thunk.h"
#include "decomp/events/flag_store.h"
#include "engine/addr.h"
#include "guest_abi.h"
#include "host/plugins.h"
#include "log.h"

#include <cstdlib>

namespace {

constexpr std::uint64_t kManagerSlot = 0x593b100;   // SprjEventFlagMan*
constexpr std::uint64_t kGuestGetValue = 0x17cfd80;  // (manager, first id, bit count) -> value

std::uint64_t g_slide = 0;
bool g_ok = false;
bool g_checked = false;
bool g_log = false;
std::uint64_t g_logged = 0, g_lost_said = 0;

// The game's memory read without trusting it: the store may not exist yet
// when a plugin asks.
struct Safe {
    template <typename T>
    bool operator()(std::uint64_t at, T* out) const {
        return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, sizeof(T));
    }
};

std::uint64_t guest_of(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }

std::uint64_t manager() {
    std::uint64_t m = 0;
    return g_ok && Safe{}(guest_of(kManagerSlot), &m) ? m : 0;
}

}  // namespace

void event_flags_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_ok = image->sha256 == kEboot109Sha256;
    if (const char* e = std::getenv("BBHOST_EVENT_FLAG_LOG"); g_ok && e && e[0] == '1') {
        g_log = true;
        event_flag_watch(true);
        host_log("event flags: every change is logged (BBHOST_EVENT_FLAG_LOG=1)");
    }
}

bool event_flag_get(std::uint32_t id, bool* value) {
    const std::uint64_t man = manager();
    std::uint64_t byte = 0;
    std::uint8_t mask = 0, v = 0;
    if (!man || !sprj_event_flag::locate(man, id, Safe{}, &byte, &mask) || !Safe{}(byte, &v)) return false;
    if (value) *value = (v & mask) != 0;
    return true;
}

bool event_flags_get(const std::uint32_t* ids, std::size_t n, bool* out) {
    const std::uint64_t man = manager();
    std::uint32_t size = 0;
    if (!man || !Safe{}(man + sprj_event_flag::kBlockSize, &size) || !size) return false;
    std::uint32_t walked = 0;
    std::uint64_t data = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // As sprj_event_flag::locate splits an id, with the block's storage
        // found once for the ids that share it.
        const std::uint32_t block = ids[i] / size, bit = ids[i] - block * size;
        if (i == 0 || block != walked) {
            data = sprj_event_flag::block_data(man, block, Safe{});
            walked = block;
        }
        std::uint8_t v = 0;
        if (!data || !Safe{}(data + (bit >> 3), &v)) return false;
        out[i] = (v & sprj_event_flag::mask_of(bit)) != 0;
    }
    return true;
}

bool event_flag_set(std::uint32_t id, bool value) {
    const std::uint64_t man = manager();
    std::uint64_t byte = 0;
    std::uint8_t mask = 0, was = 0;
    if (!man || !sprj_event_flag::locate(man, id, Safe{}, &byte, &mask) || !Safe{}(byte, &was)) return false;
    auto* p = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(byte));
    *p = value ? static_cast<std::uint8_t>(was | mask) : static_cast<std::uint8_t>(was & ~mask);
    if (((was & mask) != 0) != value) event_flag_changed(id, value);
    return true;
}

void event_flags_tick() {
    if (!g_ok) return;
    if (event_flag_watching()) {
        EventFlagChange changes[256];
        std::uint64_t lost = 0;
        std::size_t n = 0;
        while ((n = event_flag_take_changes(changes, 256, &lost)) != 0) {
            for (std::size_t i = 0; i < n; ++i) {
                if (g_log) host_log("event flag %u -> %u", changes[i].id, changes[i].value);
                plugins_event_flag(changes[i].id, changes[i].value != 0);
                play_history_event_flag(changes[i].id, changes[i].value != 0);
            }
            g_logged += n;
        }
        if (lost != g_lost_said) {
            host_log("event flags: %llu changes dropped (more than the ring holds between two frames)",
                     static_cast<unsigned long long>(lost - g_lost_said));
            g_lost_said = lost;
        }
    }
    if (g_checked) return;
    const std::uint64_t man = manager();
    if (!man) return;
    // The online-state flags the session code writes every frame (2000..2022),
    // and a spread of others: ours against the game's own getter - its
    // trampoline when decomp/ holds its entry - one flag at a time. Only flags
    // whose blocks exist are compared.
    void* game = sprj_event_flag_game_get_value();
    if (!game) game = reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(kGuestGetValue)));
    static const std::uint32_t kIds[] = {2000, 2001, 2010, 2011, 2020, 2021, 2022, 1000, 6001, 9000, 11000000, 12000000};
    int compared = 0, differ = 0;
    for (const std::uint32_t id : kIds) {
        bool ours = false;
        if (!event_flag_get(id, &ours)) continue;
        const auto theirs = hle_call_guest<std::int64_t>(game, man, static_cast<std::int64_t>(id), static_cast<std::int64_t>(1));
        ++compared;
        if ((theirs != 0) != ours) {
            ++differ;
            host_log("event flags: flag %u reads %d here, %lld through the game's getter", id, ours ? 1 : 0, static_cast<long long>(theirs));
        }
    }
    if (!compared) return;  // no block exists yet: try again next frame
    g_checked = true;
    host_log("event flags: our reader against the game's getter: %d flags compared, %d differ", compared, differ);
}
