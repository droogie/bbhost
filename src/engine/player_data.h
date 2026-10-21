// The local player's game data by name:
// GameDataMan (singleton slot at Binary Ninja 0x593b130) keeps the local
// player's record at +0x08 - hit points and stamina, the six attributes,
// insight, level and blood echoes - the numbers the status screen shows and
// the save keeps; and counters of its own: the NG+ cycle (+0x68), co-op
// helps and invader kills (+0x78/+0x7c), true deaths and deaths (+0x84/
// +0x88), play time in milliseconds (+0x94). The offsets were confirmed by
// live 1.09 reads and the game's own debug menu, and are checked here by
// what a loaded character reads back (a line at the first frame the record
// exists).
#pragma once

#include <cstdint>
#include <string>

struct ElfImage;

void player_data_install(ElfImage* image);
// "level", "echoes", "insight", "vitality", "endurance", "strength", "skill",
// "bloodtinge", "arcane", "hp", "max_hp", "stamina", "max_stamina"; and the
// manager's own counters: "ng_cycle", "deaths", "true_deaths", "coop_helps",
// "invader_kills", "play_time_ms" - which load with the world (at the title,
// where the record already holds the save's character, they read 0; the
// seed's in the world: 3 deaths, 2,567 s played). False when the name is
// unknown or no character is loaded.
bool player_stat_get(const char* name, std::int64_t* value);
bool player_stat_set(const char* name, std::int64_t value);
// The origin chosen at character creation (the record's byte at +0xce):
// CharaInitParam row 3000 + origin holds its starting level and attributes,
// as character creation's own origin list reads it (sub_1f8e6b0). False when
// no character is loaded.
bool player_origin(int* origin);
// Returns the unique save-summary name matching the live character's level and play time.
bool player_character_identity(std::string* name, int* level);
// From the frame-time manager's hook: the one line with the loaded
// character's numbers, once.
void player_data_tick();
