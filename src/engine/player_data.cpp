#include "engine/player_data.h"

#include "core/elf.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "log.h"

#include <cstring>

namespace {

constexpr std::uint64_t kGameDataMan = 0x593b130;  // GameDataMan*; the local record at +0x08

struct Stat {
    const char* name;
    bool manager;  // GameDataMan's own field, else the local player's record
    std::uint32_t offset;
    bool is_signed;
};
// The record: GameDataManPlayerRecord, 32-bit. The manager's counters (from
// the saturating add helpers 0x18eb5c0..0x18eb660 and the save reader):
// unsigned 32-bit but ClearCount.
constexpr Stat kStats[] = {
    {"hp", false, 0x14, true},          {"max_hp", false, 0x18, true},   {"stamina", false, 0x30, true},
    {"max_stamina", false, 0x34, true}, {"vitality", false, 0x40, true}, {"endurance", false, 0x48, true},
    {"strength", false, 0x58, true},    {"skill", false, 0x60, true},    {"bloodtinge", false, 0x68, true},
    {"arcane", false, 0x70, true},      {"insight", false, 0x84, true},  {"level", false, 0x90, true},
    {"echoes", false, 0x94, true},
    {"ng_cycle", true, 0x68, true},     {"coop_helps", true, 0x78, false}, {"invader_kills", true, 0x7c, false},
    {"true_deaths", true, 0x84, false}, {"deaths", true, 0x88, false},     {"play_time_ms", true, 0x94, false},
};

std::uint64_t g_slide = 0;
bool g_ok = false, g_said = false;

std::uint64_t manager() {
    std::uint64_t man = 0;
    const auto at = g_slide + (kGameDataMan - kPreferredGuestSlide);
    if (!g_ok || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), &man, 8)) return 0;
    return man;
}

std::uint64_t record() {
    const std::uint64_t man = manager();
    std::uint64_t rec = 0;
    if (!man || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(man + 8)), &rec, 8)) return 0;
    return rec;
}

// Where a stat lives, or 0 when no character is loaded (the manager's
// counters are read only while there is a record, so both mean the same).
std::uint64_t base_of(const Stat& s) {
    const std::uint64_t rec = record();
    return rec ? (s.manager ? manager() : rec) : 0;
}

const Stat* find(const char* name) {
    if (!name) return nullptr;
    for (const Stat& s : kStats) {
        if (std::strcmp(s.name, name) == 0) return &s;
    }
    return nullptr;
}

}  // namespace

void player_data_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_ok = image->sha256 == kEboot109Sha256;
}

bool player_stat_get(const char* name, std::int64_t* value) {
    const Stat* s = find(name);
    const std::uint64_t base = s ? base_of(*s) : 0;
    std::uint32_t v = 0;
    if (!base || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base + s->offset)), &v, 4)) return false;
    if (value) *value = s->is_signed ? static_cast<std::int64_t>(static_cast<std::int32_t>(v)) : static_cast<std::int64_t>(v);
    return true;
}

bool player_stat_set(const char* name, std::int64_t value) {
    const Stat* s = find(name);
    const std::uint64_t base = s ? base_of(*s) : 0;
    if (!base) return false;
    const std::uint32_t v = static_cast<std::uint32_t>(value);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(base + s->offset)), &v, 4);
    return true;
}

bool player_origin(int* origin) {
    const std::uint64_t rec = record();
    std::uint8_t v = 0;
    if (!rec || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rec + 0xce)), &v, 1)) return false;
    if (origin) *origin = v;
    return true;
}

// The game's own HUD option: the byte at +9 of GameDataMan's game options
// (+0x48), 1 shown and 0 hidden, which the in-game HUD's update reads before
// it draws (sub_1ffb2a0 and its neighbours: cmp byte [options + 9], 0). A
// plugin's HUD shown (plugin API 29's hud_hide with -1) sets it to 1 while it
// asks, and the player's own value is put back when it stops.
void player_hud_option_hold(int want) {
    static int saved = -1;           // the player's value while it is held
    static std::uint64_t held_at = 0;
    const bool held = want == 0 || want == 1;
    if (!held && saved < 0) return;  // nothing asked, nothing to put back
    const std::uint64_t man = manager();
    std::uint64_t options = 0;
    if (!man || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(man + 0x48)), &options, 8)) options = 0;
    const std::uint64_t at = options ? options + 9 : 0;
    std::uint8_t v = 0;
    const bool readable = at && host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), &v, 1);
    if (held && readable) {
        if (saved < 0 || held_at != at) {
            saved = v;
            held_at = at;
            host_log("player data: the game's HUD option %s for a plugin (the player's is %s)", want ? "shown" : "hidden", v ? "on" : "off");
        }
        if (v != want) *reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(at)) = static_cast<std::uint8_t>(want);
    } else if (!held && saved >= 0) {
        if (readable && held_at == at) *reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(at)) = static_cast<std::uint8_t>(saved);
        host_log("player data: the game's HUD option back to the player's (%s)", saved ? "on" : "off");
        saved = -1;
        held_at = 0;
    }
}

void player_data_tick() {
    if (g_said || !g_ok) return;
    std::int64_t level = 0;
    if (!player_stat_get("level", &level) || level <= 0) return;  // no character yet
    g_said = true;
    char line[512];
    std::size_t n = 0;
    for (const Stat& s : kStats) {
        std::int64_t v = 0;
        if (player_stat_get(s.name, &v)) n += static_cast<std::size_t>(std::snprintf(line + n, sizeof(line) - n, " %s %lld", s.name, static_cast<long long>(v)));
        if (n >= sizeof(line)) break;
    }
    host_log("player data: the loaded character:%s", line);
}
