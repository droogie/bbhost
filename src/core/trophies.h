#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>

namespace trophies {
constexpr int kCount = 40;
constexpr int kBaseCount = 34;  // platinum plus 33 base-game trophies; DLC is 34–39
enum class Grade : int { Platinum = 1, Gold = 2, Silver = 3, Bronze = 4 };
struct Info { Grade grade; bool hidden; const char* name; const char* description; };
const std::array<Info, kCount>& catalog();
const char* grade_name(Grade grade);
struct Snapshot {
    std::array<std::int64_t, kCount> earned{};
    std::string error;
    int count(int begin = 0, int end = kCount) const;
};
Snapshot read(const std::filesystem::path& file);
enum class Status { Ok, AlreadyUnlocked, InvalidId, AutomaticPlatinum, IoError };
struct Unlock { Status status; int platinum = -1; };

// Preserves legacy records; backs up once and replaces atomically.
class Store {
public:
    void open(const std::filesystem::path& file);
    Snapshot snapshot() const;
    Unlock unlock(int id, std::int64_t when);
    Unlock reconcile_platinum(std::int64_t when);
    int pop_notice();  // -1 when empty; only successful new unlocks are queued
private:
    Unlock update(int id, std::int64_t when);
    mutable std::mutex mutex_;
    std::filesystem::path file_;
    Snapshot state_;
    std::deque<int> notices_;
};
Store& runtime();
}  // namespace trophies
