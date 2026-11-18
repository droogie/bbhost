#include "engine/play_history.h"

#include "bbhost/engine/sprj/game_state_man.hpp"
#include "bbhost/engine/sprj/game_data_man.hpp"
#include "bbhost/engine/sprj/world_chr_man.hpp"
#include "core/elf.h"
#include "core/play_history.h"
#include "core/chalice_history.h"
#include "core/portable.h"
#include "decomp/events/flag_store.h"
#include "engine/addr.h"
#include "engine/event_flags.h"
#include "engine/loading.h"
#include "engine/player_data.h"
#include "engine/world_chr.h"
#include "engine/chalice_history.h"
#include "bbhost/engine/cs/dungeon_ritual_helper.hpp"
#include "log.h"

#include <chrono>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include <set>

namespace {
std::uint64_t g_slide = 0;
std::set<std::string> g_pauses;  // main thread only
std::array<play_history::FightFlags, play_history::kBossCount> g_flags{};
std::chrono::steady_clock::time_point g_polled{};
bool g_was_world = false;
int g_last_slot = -1, g_last_cycle = -1;
std::vector<chalice_events::Source> g_chalice_sources;
std::string g_chalice_key;
int g_chalice_altar = 0;
std::chrono::steady_clock::time_point g_chalice_polled{};

template<class T> bool read_at(std::uint64_t at, T* value) {
    return at && host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), value, sizeof(T));
}
std::uint64_t pointer(std::uint64_t at) { std::uint64_t value = 0; read_at(at, &value); return value; }

bool flag_bits(std::uint32_t first, std::size_t size, std::uint64_t* value) {
    std::uint32_t ids[64]{}; bool bits[64]{};
    if (!size || size > 64) return false;
    for (std::size_t i = 0; i < size; ++i) ids[i] = first + static_cast<std::uint32_t>(i);
    if (!event_flags_get(ids, size, bits)) return false;
    *value = 0;
    for (std::size_t i = 0; i < size; ++i) *value = (*value << 1) | bits[i];
    return true;
}
bool dungeon_key(std::string* key, int* altar) {
    std::uint64_t selector = 0, uid = 0; std::uint32_t bank = 0;
    if (!flag_bits(bb::DUNGEON_ALTAR_SELECTOR_FLAG, 7, &selector) ||
        !bb::dungeon_altar_bank_from_selector(static_cast<std::uint32_t>(selector), bank)) return false;
    const auto offset = bank * bb::DUNGEON_ALTAR_FLAG_STRIDE;
    if (!flag_bits(bb::DUNGEON_ALTAR_MAP_UID_FLAG + offset, 32, &uid) || uid == UINT32_MAX || (uid >> 24) != 29) return false;
    std::uint64_t hash = 14695981039346656037ull;
    const auto add = [&](std::uint64_t word) { for (int i = 0; i < 8; ++i) { hash ^= (word >> (i * 8)) & 255; hash *= 1099511628211ull; } };
    // Server database IDs can be assigned after an offline ritual. The map,
    // chalice and rites remain stable through that online/offline change.
    std::uint64_t chalice = 0;
    if (!flag_bits(bb::DUNGEON_ALTAR_HOLYGRAIL_EX_PARAM_FLAG + offset, 32, &chalice)) return false;
    add(chalice);
    for (int i = 0; i < 9; ++i) {
        std::uint64_t pair = 0;
        if (!flag_bits(bb::DUNGEON_ALTAR_FEATURE_PAIRS_FLAG + offset + 64u * i, 64, &pair)) return false;
        add(pair);
    }
    char text[64];
    std::snprintf(text, sizeof text, "%u-%08x-%016llx", bank, static_cast<unsigned>(uid), static_cast<unsigned long long>(hash));
    *key = text; *altar = static_cast<int>(bank); return true;
}
void chalice_tick(const play_history::Observation& game, bool entered) {
    chalice_history::Observation o;
    o.character = play_history::runtime().current_character_id(); o.cycle = game.cycle; o.deaths = game.deaths;
    o.in_world = game.valid && game.in_world; o.alive = game.alive; o.suspended = game.suspended || !game.own_world;
    if (o.in_world && !o.suspended && (game.block >> 24) == 29) {
        if (entered) { g_chalice_sources.clear(); g_chalice_key.clear(); }
        const auto now = std::chrono::steady_clock::now();
        if (entered || now - g_chalice_polled >= std::chrono::seconds(1)) {
            std::string key; int altar;
            if (dungeon_key(&key, &altar)) {
                // Altar selection can change before the old world finishes
                // fading out. Its still-running events belong to the old
                // dungeon. Wait for a world entry before changing identity.
                if (!entered && !g_chalice_key.empty() && key != g_chalice_key) {
                    o.in_world = false;
                    chalice_history::runtime().observe(o);
                    return;
                }
                if (key != g_chalice_key) g_chalice_sources.clear();
                g_chalice_key = key; g_chalice_altar = altar;
                const auto system = pointer(g_slide + bb::SPRJ_EMK_SYSTEM_SINGLETON_PTR.rva);
                std::vector<chalice_events::Source> sources;
                const auto safe = [](std::uint64_t at, auto* out) { return read_at(at, out); };
                if (chalice_events::sources(system, g_slide + bb::SprjEmkEventIns::VTABLE.rva, safe, sources)) {
                    // Completed events leave the active list. Keep their flag
                    // mapping until this dungeon is left, including the win.
                    for (const auto& s : sources) {
                        const auto same = [&](const auto& v) { return v.defeated == s.defeated && v.name_id == s.name_id; };
                        if (std::none_of(g_chalice_sources.begin(), g_chalice_sources.end(), same)) g_chalice_sources.push_back(s);
                    }
                    if (entered || (!sources.empty() && g_chalice_sources.size() == sources.size())) {
                        static std::string last;
                        const auto note = key + ":" + std::to_string(g_chalice_sources.size());
                        if (note != last) { last = note; host_log("chalice history: altar %d, %zu boss layers identified", altar, g_chalice_sources.size()); }
                    }
                }
            }
            g_chalice_polled = now;
        }
        o.dungeon = g_chalice_key; o.altar = g_chalice_altar;
        for (const auto& s : g_chalice_sources) {
            const std::uint32_t ids[]{s.defeated, s.begun}; bool bits[2]{};
            const bool valid = event_flags_get(ids, 2, bits);
            o.bosses.push_back({s.layer, static_cast<int>(s.name_id), {valid, bits[0], bits[1]}});
        }
    }
    chalice_history::runtime().observe(o);
}

std::string character_name(std::uint64_t data, int slot) {
    const auto store = pointer(data + offsetof(bb::GameDataMan, unknown_data_58));
    std::uint16_t name[17]{};
    if (!store || !read_at(store + 0x20 + sizeof(bb::GameDataCharacterSummary) * slot, &name)) return {};
    std::string text;
    for (int i = 0; i < 16 && name[i]; ++i) {
        std::uint32_t ch = name[i];
        if (ch >= 0xd800 && ch <= 0xdbff && i + 1 < 16 && name[i + 1] >= 0xdc00 && name[i + 1] <= 0xdfff)
            ch = 0x10000 + ((ch - 0xd800) << 10) + (name[++i] - 0xdc00);
        else if (ch >= 0xd800 && ch <= 0xdfff) ch = 0xfffd;
        if (ch < 0x20) ch = ' ';
        if (ch < 0x80) text += static_cast<char>(ch);
        else {
            if (ch >= 0x10000) { text += static_cast<char>(0xf0 | (ch >> 18)); text += static_cast<char>(0x80 | ((ch >> 12) & 63)); }
            else if (ch >= 0x800) text += static_cast<char>(0xe0 | (ch >> 12));
            if (ch < 0x800) text += static_cast<char>(0xc0 | (ch >> 6));
            else text += static_cast<char>(0x80 | ((ch >> 6) & 63));
            text += static_cast<char>(0x80 | (ch & 63));
        }
    }
    return text;
}
}

void play_history_install(ElfImage* image) {
    g_slide = image && image->sha256 == kEboot109Sha256 ? image->mem.slide : 0;
    if (g_slide) event_flag_watch(true);
}
int play_history_pause(const char* plugin, int paused) {
    if (!plugin || !plugin[0]) return 1;
    if (paused) g_pauses.insert(plugin);
    else g_pauses.erase(plugin);
    return 0;
}
void play_history_event_flag(std::uint32_t id, bool value) {
    for (int i = 0; i < play_history::kBossCount; ++i) {
        const auto& boss = play_history::bosses()[i];
        if (boss.defeated == id) g_flags[i].defeated = value;
        if (boss.begun == id) g_flags[i].begun = value;
    }
}
void play_history_tick() {
    if (!g_slide) return;
    play_history::Observation o;
    const auto absent = [&] { play_history::runtime().observe(o); chalice_tick(o, false); g_was_world = false; };
    o.in_world = loading_in_world() && !loading_screen_up();
    if (!o.in_world && !loading_screen_up()) { absent(); return; }
    if (o.in_world && !world_player_block(&o.block)) { o.in_world = false; absent(); return; }
    const auto state = pointer(g_slide + bb::GAME_STATE_MAN_SINGLETON.rva);
    const auto data = pointer(g_slide + bb::GAME_DATA_MAN_SINGLETON.rva);
    // RequestCharacterSave (RVA 0x156d570) passes this slot to the save
    // writer and summary updater. FrpgSaveLoadMan+0x38 is request state,
    // despite its historical slot_index label; it is NOT a character slot.
    if (!state || !data || !read_at(state + offsetof(bb::GameStateMan, active_slot_index), &o.slot) || o.slot < 0 || o.slot >= 10) {
        o.in_world = false; absent(); return;
    }
    std::int64_t cycle = 0, time = 0, deaths = 0, level = 0;
    if (!player_stat_get("ng_cycle", &cycle) || !player_stat_get("play_time_ms", &time) ||
        !player_stat_get("deaths", &deaths) || !player_stat_get("level", &level) || level <= 0 || cycle < 0) {
        o.in_world = false; absent(); return;
    }
    o.valid = true; o.cycle = static_cast<int>(cycle); o.play_ms = time; o.deaths = deaths;
    o.suspended = !g_pauses.empty(); o.when = std::time(nullptr);
    static bool own_world = true;
    if (!o.in_world) {
        o.own_world = own_world;
        play_history::runtime().observe(o); chalice_tick(o, false); g_was_world = false; return;
    }
    const auto world = pointer(g_slide + bb::WORLD_CHR_MAN_SINGLETON_PTR.rva); const auto chr = world ? pointer(world + 0x60) : 0;
    const auto modules = chr ? pointer(chr + 0x3b0) : 0; const auto hp_data = modules ? pointer(modules + 0x20) : 0;
    std::int32_t hp = 0; std::uint32_t type = 0;
    if (!read_at(hp_data ? hp_data + 0xf8 : 0, &hp) || !read_at(chr ? chr + 0x78 : 0, &type)) {
        o.valid = o.in_world = false; absent(); return;
    }
    o.alive = hp > 0; o.own_world = own_world = type == 0;
    static std::string name;
    const bool entered = !g_was_world || o.slot != g_last_slot || o.cycle != g_last_cycle;
    if (entered) {
        name = character_name(data, o.slot);
        host_log("play history: slot %d, %s, NG+%d, played %llu ms%s", o.slot + 1, name.c_str(), o.cycle,
                 static_cast<unsigned long long>(o.play_ms), o.own_world ? "" : ", summoned world");
    }
    o.name = name;
    const auto now = std::chrono::steady_clock::now();
    static auto last_tick = now;
    static std::chrono::steady_clock::duration remainder{};
    if (!entered) {
        const auto elapsed = now - last_tick + remainder;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        o.capped_delta_ms = ms.count(); remainder = elapsed - ms;
    } else remainder = {};
    last_tick = now;
    // Setters feed changes every frame. Polls also recover whole-store loads
    // and a dropped ring, without 44 tree walks on every frame.
    if (entered || now - g_polled >= std::chrono::seconds(1)) {
        for (int i = 0; i < play_history::kBossCount; ++i) {
            const auto& boss = play_history::bosses()[i];
            const std::uint32_t ids[] = {boss.defeated, boss.begun}; bool flags[2]{};
            g_flags[i].valid = event_flags_get(ids, 2, flags);
            if (g_flags[i].valid) { g_flags[i].defeated = flags[0]; g_flags[i].begun = flags[1]; }
        }
        g_polled = now;
    }
    o.flags = g_flags;
    play_history::runtime().observe(o);
    chalice_tick(o, entered);
    g_was_world = true; g_last_slot = o.slot; g_last_cycle = o.cycle;
}
