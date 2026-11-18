// Local campaign records. Observations contain no guest pointers; the state
// machine and persistence are also usable by tests without a running game.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace play_history {
constexpr int kBossCount = 22;  // 17 story bosses, five DLC bosses; chalices do not reset with NG+
constexpr std::uint64_t kPlayTimeCap = 0xd693a018ull;  // native accumulator: 999:59:59
struct BossInfo {
    const char* name;
    std::uint32_t defeated, begun, block;  // block: packed map's upper 16 bits
};
const std::array<BossInfo, kBossCount>& bosses();
struct BossRecord {
    std::uint64_t attempts = 0, kills = 0;
    bool partial = false;  // attempts before tracking began are unknown
};
struct Cycle {
    int number = 0;  // NG = 0, NG+ = 1
    std::uint64_t elapsed_ms = 0;
    bool complete = false, partial_time = false;
    std::array<BossRecord, kBossCount> bosses{};
};
struct Character {
    std::uint64_t id = 0;
    int slot = 0;
    std::string name;
    std::int64_t created = 0;
    std::uint64_t last_play_ms = 0, last_deaths = 0;
    int active_boss = -1, ended_boss = -1;
    std::vector<Cycle> cycles;
};
struct Snapshot {
    std::vector<Character> characters;
    std::string error;
};
struct FightFlags { bool valid = false, defeated = false, begun = false; };
struct Observation {
    bool valid = false, in_world = false, own_world = true, suspended = false, alive = true;
    int slot = -1, cycle = 0;
    std::string name;
    std::int64_t when = 0;
    std::uint64_t play_ms = 0, deaths = 0, capped_delta_ms = 0;
    std::uint32_t block = 0;
    std::array<FightFlags, kBossCount> flags{};
};
Snapshot read(const std::filesystem::path& file);
std::array<BossRecord, kBossCount> totals(const Character& character);
std::string cycle_name(int cycle);
std::string duration(std::uint64_t ms);
// Shared atomic persistence for the two independent history files.
bool write_history_file(const std::filesystem::path& file, const std::string& text);

class Store {
public:
    void open(const std::filesystem::path& file);
    void observe(const Observation& observation);
    bool flush();
    Snapshot snapshot() const;
    std::uint64_t current_character_id() const;
private:
    bool save();  // mutex held; malformed/future files are never overwritten
    mutable std::mutex mutex_;
    std::filesystem::path file_;
    Snapshot state_;
    bool readable_ = true, dirty_ = false, connected_ = false, suspended_ = false;
    std::uint64_t selected_ = 0, pending_ms_ = 0;
};
Store& runtime();
}  // namespace play_history
