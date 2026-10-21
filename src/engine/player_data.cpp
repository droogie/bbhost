#include "engine/player_data.h"

#include "core/elf.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/menu_pointer.h"
#include "engine/world_chr.h"
#include "bbhost/engine/sprj/game_data_man.hpp"
#include "net/discord_presence.h"
#include "log.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

namespace {

constexpr std::uint64_t kGameDataMan = bb::GAME_DATA_MAN_SINGLETON.bn();

struct Stat {
    const char* name;
    bool manager;  // GameDataMan's own field, else the local player's record
    std::uint32_t offset;
    bool is_signed;
};
// The record: GameDataManPlayerRecord, 32-bit. The manager's counters (from
// the saturating add helpers 0x18eb5c0..0x18eb660 and the save reader):
// unsigned 32-bit but ClearCount.
constexpr auto kStatsBlock = offsetof(bb::GameDataManPlayerRecord, stats);
constexpr Stat kStats[] = {
    {"hp", false, offsetof(bb::GameDataManPlayerRecord, hp), true},
    {"max_hp", false, offsetof(bb::GameDataManPlayerRecord, max_hp), true},
    {"stamina", false, offsetof(bb::GameDataManPlayerRecord, stamina), true},
    {"max_stamina", false, offsetof(bb::GameDataManPlayerRecord, max_stamina), true},
    {"vitality", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, vitality), true},
    {"endurance", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, endurance), true},
    {"strength", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, strength), true},
    {"skill", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, skill), true},
    {"bloodtinge", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, bloodtinge), true},
    {"arcane", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, arcane), true},
    {"insight", false, kStatsBlock + offsetof(bb::GameDataManPlayerStats, insight), true},
    {"level", false, offsetof(bb::GameDataManPlayerRecord, level), true},
    {"echoes", false, offsetof(bb::GameDataManPlayerRecord, blood_echoes), true},
    {"ng_cycle", true, bb::GAME_DATA_MAN_NG_CYCLE_OFFSET, true},
    {"coop_helps", true, offsetof(bb::GameDataMan, help_white_ghost_count), false},
    {"invader_kills", true, offsetof(bb::GameDataMan, kill_black_ghost_count), false},
    {"true_deaths", true, bb::GAME_DATA_MAN_TRUE_DEATH_COUNT_OFFSET, false},
    {"deaths", true, bb::GAME_DATA_MAN_DEATH_COUNT_OFFSET, false},
    {"play_time_ms", true, bb::GAME_DATA_MAN_PLAY_TIME_OFFSET, false},
};

std::uint64_t g_slide = 0;
bool g_ok = false;
std::string g_last_logged_identity;
std::uint32_t g_pending_map = 0;
std::uint32_t g_active_map = 0;
std::chrono::steady_clock::time_point g_pending_map_since{};
std::string g_active_location;

std::uint64_t manager() {
    std::uint64_t man = 0;
    const auto at = g_slide + (kGameDataMan - kPreferredGuestSlide);
    if (!g_ok || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), &man, 8)) return 0;
    return man;
}

std::uint64_t record() {
    const std::uint64_t man = manager();
    std::uint64_t rec = 0;
    if (!man || !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(man + offsetof(bb::GameDataMan, local_player_record))), &rec, 8)) return 0;
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

namespace {

std::string utf16_to_utf8(const std::uint16_t* text, std::size_t count) {
    std::string out;
    for (std::size_t i = 0; i < count && text[i]; ++i) {
        std::uint32_t cp = text[i];
        if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < count && text[i + 1] >= 0xdc00 && text[i + 1] <= 0xdfff)
            cp = 0x10000 + ((cp - 0xd800) << 10) + (text[++i] - 0xdc00);
        else if (cp >= 0xd800 && cp <= 0xdfff)
            continue;
        if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
        else if (cp <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    return out;
}

std::string location_name(std::uint32_t block) {
    struct KnownMap { std::uint32_t prefix; const char* name; };
    static constexpr KnownMap kKnownMaps[] = {
        {0x15000000, "Hunter's Dream"}, {0x18000000, "Cathedral Ward"}, {0x18010000, "Central Yharnam"},
        {0x17000000, "Old Yharnam"}, {0x1b000000, "Forbidden Woods"}, {0x20000000, "Moonside Lake"},
        {0x1a000000, "Nightmare of Mensis"}, {0x24000000, "Fishing Hamlet"}, {0x22000000, "Hunter's Nightmare"},
    };
    const std::uint32_t prefix = block & 0xffff0000u;
    for (const KnownMap& map : kKnownMaps) if (map.prefix == prefix) return map.name;
    char label[40] = {};
    std::snprintf(label, sizeof(label), "Area %02X · Block %02X", block >> 24, (block >> 16) & 0xff);
    return label;
}

bool stable_player_location(std::string* location) {
    std::uint32_t block = 0;
    if (!world_player_block(&block) || block == 0) { g_pending_map = 0; return false; }
    const auto now = std::chrono::steady_clock::now();
    if (block != g_pending_map) { g_pending_map = block; g_pending_map_since = now; return false; }
    if (now - g_pending_map_since < std::chrono::milliseconds(1200)) return false;
    if (g_active_map != block) {
        g_active_map = block;
        g_active_location = location_name(block);
        host_log("discord presence: stable player map 0x%08x -> %s", block, g_active_location.c_str());
    }
    if (location) *location = g_active_location;
    return true;
}

}  // namespace

bool player_character_identity(std::string* name, int* level) {
    const std::uint64_t man = manager(), rec = record();
    if (!man || !rec) return false;
    std::uint32_t live_level = 0, live_play_time_ms = 0;
    std::uint64_t summaries = 0;
    if (!host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rec + bb::GAME_DATA_MAN_PLAYER_RECORD_LEVEL_OFFSET)), &live_level, sizeof(live_level)) ||
        !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(man + bb::GAME_DATA_MAN_PLAY_TIME_OFFSET)), &live_play_time_ms, sizeof(live_play_time_ms)) ||
        !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(man + bb::GAME_DATA_MAN_CHARACTER_SUMMARY_STORE_OFFSET)), &summaries, sizeof(summaries)) ||
        !summaries || live_level == 0 || live_level > bb::GAME_DATA_CHARACTER_LEVEL_MAX) return false;
    const std::uint32_t live_seconds = live_play_time_ms / 1000;
    int match = -1;
    for (std::size_t i = 0; i < bb::GAME_DATA_CHARACTER_SUMMARY_COUNT; ++i) {
        std::uint8_t occupied = 0;
        const auto occupied_offset = offsetof(bb::GameDataCharacterSummaryStore, occupied) + i;
        if (!host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(summaries + occupied_offset)), &occupied, sizeof(occupied)) || !occupied) continue;
        const std::uint64_t summary = summaries + offsetof(bb::GameDataCharacterSummaryStore, records) + i * bb::GAME_DATA_CHARACTER_SUMMARY_SIZE;
        std::uint32_t summary_level = 0, summary_time = 0;
        if (!host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(summary + offsetof(bb::GameDataCharacterSummary, level))), &summary_level, sizeof(summary_level)) ||
            !host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(summary + offsetof(bb::GameDataCharacterSummary, play_time_div_1000))), &summary_time, sizeof(summary_time))) return false;
        const std::uint32_t delta = summary_time > live_seconds ? summary_time - live_seconds : live_seconds - summary_time;
        if (summary_level == live_level && delta <= 10) { if (match >= 0) return false; match = static_cast<int>(i); }
    }
    if (match < 0) return false;
    const std::uint64_t summary = summaries + offsetof(bb::GameDataCharacterSummaryStore, records) + static_cast<std::size_t>(match) * bb::GAME_DATA_CHARACTER_SUMMARY_SIZE;
    std::array<std::uint16_t, 17> units{};
    if (!host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(summary)), units.data(), units.size() * sizeof(units[0]))) return false;
    std::string decoded = utf16_to_utf8(units.data(), units.size());
    if (decoded.empty()) return false;
    if (name) *name = std::move(decoded);
    if (level) *level = static_cast<int>(live_level);
    return true;
}

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

void player_data_tick() {
    if (!g_ok) return;
    if (!net::discord_rpc_enabled()) {
        net::ShutdownDiscordRPC();
        return;
    }
    net::InitDiscordRPC();
    if (menu_pointer_front_end()) {
        net::UpdateDiscordMainMenuPresence();
        g_pending_map = g_active_map = 0;
        g_active_location.clear();
        g_last_logged_identity.clear();
        return;
    }
    std::string location;
    if (!stable_player_location(&location)) return;
    std::string character_name;
    int level = 0;
    if (!player_character_identity(&character_name, &level)) return;
    net::UpdateDiscordGamePresence(location.c_str(), character_name.c_str(), level);
    const std::string identity = character_name + "#" + std::to_string(level);
    if (identity == g_last_logged_identity) return;
    g_last_logged_identity = identity;
    char line[512];
    std::size_t n = 0;
    for (const Stat& s : kStats) {
        std::int64_t v = 0;
        if (player_stat_get(s.name, &v)) n += static_cast<std::size_t>(std::snprintf(line + n, sizeof(line) - n, " %s %lld", s.name, static_cast<long long>(v)));
        if (n >= sizeof(line)) break;
    }
    host_log("player data: the loaded character:%s", line);
}
