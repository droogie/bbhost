// Chalice encounters stay independent of campaign clears: a dungeon can
// outlive an NG+ transition, and the same boss can appear in many dungeons.
#pragma once
#include "core/play_history.h"

namespace chalice_history {
struct Stats {
    int cycle = 0;
    play_history::BossRecord count;
};
struct Encounter {
    std::uint64_t id = 0, character = 0, last_deaths = 0;
    std::string dungeon;
    int altar = 0, layer = 1, name_id = 0;
    bool cleared = false, prior_clear = false, active = false, blocked = false;
    std::vector<Stats> stats;
};
struct Snapshot { std::vector<Encounter> encounters; std::string error; };
struct Boss {
    int layer = 1, name_id = 0;
    play_history::FightFlags flags;
};
struct Observation {
    std::uint64_t character = 0, deaths = 0;
    int cycle = 0, altar = 0;
    bool in_world = false, suspended = false, alive = true;
    std::string dungeon;
    std::vector<Boss> bosses;
};
struct Total { int name_id; play_history::BossRecord count; };
Snapshot read(const std::filesystem::path& file);
std::vector<Total> totals(const Snapshot& state, std::uint64_t character, int cycle = -1);
std::string boss_name(int name_id);
class Store {
public:
    void open(const std::filesystem::path& file);
    void observe(const Observation& observation);
    Snapshot snapshot() const;
    bool flush();
private:
    bool save();
    mutable std::mutex mutex_;
    std::filesystem::path file_;
    Snapshot state_;
    bool readable_ = true, dirty_ = false;
};
Store& runtime();
}
