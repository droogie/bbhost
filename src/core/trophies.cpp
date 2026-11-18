// SPDX-License-Identifier: GPL-3.0-or-later
// Catalog reference: ayozetr/bloodborne_linux scripts/trophies.py
// (804651e4, GPL-2.0-or-later). See docs/trophies.md for provenance.
#include "core/trophies.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace trophies {
namespace {
using G = Grade;
const std::array<Info, kCount> kCatalog{{
    {G::Platinum, false, "Bloodborne", "Earn every base-game trophy."},
    {G::Gold, true, "Yharnam Sunrise", "Finish the night of the hunt and see the dawn."},
    {G::Gold, true, "Honoring Wishes", "Take over the watch of the Hunter's Dream."},
    {G::Gold, true, "Childhood's Beginning", "Awaken as a new Great One."},
    {G::Gold, true, "Yharnam, Pthumerian Queen", "Defeat Queen Yharnam in the ancient underground ruins."},
    {G::Gold, false, "Hunter's Essence", "Collect every hunter weapon from the base game."},
    {G::Gold, false, "Hunter's Craft", "Collect every special hunter tool from the base game."},
    {G::Silver, false, "Weapon Master", "Upgrade a weapon to its highest level."},
    {G::Silver, false, "Blood Gem Master", "Find a very rare blood gem."},
    {G::Silver, false, "Rune Master", "Find a very rare Caryll rune."},
    {G::Silver, true, "Cainhurst", "Reach the abandoned castle of Cainhurst."},
    {G::Silver, true, "The Choir", "Reach the domain of the Choir, high within the Healing Church."},
    {G::Silver, true, "The Source of the Dream", "Discover the old workshop on which the Hunter's Dream was based."},
    {G::Silver, true, "Nightmare Lecture Building", "Reach the lecture building within the nightmare."},
    {G::Bronze, true, "Father Gascoigne", "Defeat Father Gascoigne."},
    {G::Bronze, true, "Vicar Amelia", "Defeat Vicar Amelia."},
    {G::Bronze, true, "Shadow of Yharnam", "Defeat Shadow of Yharnam."},
    {G::Bronze, true, "Rom, the Vacuous Spider", "Defeat Rom, the Vacuous Spider."},
    {G::Bronze, true, "The One Reborn", "Defeat The One Reborn."},
    {G::Bronze, true, "Micolash, Host of the Nightmare", "Defeat Micolash, Host of the Nightmare."},
    {G::Bronze, true, "Mergo's Wet Nurse", "Defeat Mergo's Wet Nurse."},
    {G::Bronze, true, "Cleric Beast", "Defeat Cleric Beast."},
    {G::Bronze, true, "Blood-starved Beast", "Defeat Blood-starved Beast."},
    {G::Bronze, true, "The Witch of Hemwick", "Defeat The Witch of Hemwick."},
    {G::Bronze, true, "Darkbeast Paarl", "Defeat Darkbeast Paarl."},
    {G::Bronze, true, "Amygdala", "Defeat Amygdala."},
    {G::Bronze, true, "Martyr Logarius", "Defeat Martyr Logarius."},
    {G::Bronze, true, "Celestial Emissary", "Defeat Celestial Emissary."},
    {G::Bronze, true, "Ebrietas, Daughter of the Cosmos", "Defeat Ebrietas, Daughter of the Cosmos."},
    {G::Bronze, false, "Blood Gem Contact", "Find your first blood gem for strengthening a weapon."},
    {G::Bronze, false, "Rune Contact", "Find your first Caryll rune."},
    {G::Bronze, false, "Chalice of Pthumeru", "Obtain the chalice that opens the Pthumerian ruins."},
    {G::Bronze, true, "Chalice of Ailing Loran", "Obtain the chalice that opens the Loran ruins."},
    {G::Bronze, true, "Chalice of Isz", "Obtain the great chalice that opens the Isz ruins."},
    {G::Gold, false, "Old Hunter's Essence", "Collect every old hunter weapon from the expansion."},
    {G::Silver, true, "Orphan of Kos", "Defeat Orphan of Kos."},
    {G::Bronze, true, "Ludwig, the Holy Blade", "Defeat Ludwig."},
    {G::Bronze, true, "Lady Maria of the Astral Clocktower", "Defeat Lady Maria of the Astral Clocktower."},
    {G::Bronze, true, "Living Failures", "Defeat Living Failures."},
    {G::Bronze, true, "Laurence, the First Vicar", "Defeat Laurence, the First Vicar."},
}};
constexpr std::int64_t kLatestTimestamp = 253402300799ll;  // end of year 9999

Snapshot load(const std::filesystem::path& file, std::string* original) {
    Snapshot result;
    original->clear();
    std::error_code ec;
    const bool exists = std::filesystem::exists(file, ec);
    if (ec || (exists && (!std::filesystem::is_regular_file(file, ec) || ec))) {
        result.error = "Cannot read the trophy record. Check its location.";
        return result;
    }
    if (!exists) return result;
    if (std::filesystem::file_size(file, ec) > 1024 * 1024 || ec) {
        result.error = "The trophy record is too large or unreadable.";
        return result;
    }
    std::ifstream in(file, std::ios::binary);
    if (!in) { result.error = "Cannot open the trophy record."; return result; }
    *original = std::string(std::istreambuf_iterator<char>(in), {});
    if (in.bad()) { result.error = "Reading the trophy record failed."; return result; }
    std::istringstream lines(*original);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream entry(line);
        int id; std::int64_t when;
        if (entry >> id >> when && id >= 0 && id < kCount && when > 0 && when <= kLatestTimestamp && !result.earned[id])
            result.earned[id] = when;
    }
    return result;
}

bool persist(const std::filesystem::path& file, const std::string& text) {
    if (file.empty()) return false;
    std::error_code ec;
    if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) return false;
    if (std::filesystem::exists(file, ec)) {
        auto backup = file; backup += ".pre-trophies.bak";
        std::filesystem::copy_file(file, backup, std::filesystem::copy_options::skip_existing, ec);
        if (ec) return false;
    }
    auto temp = file;
    temp += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
#if defined(_WIN32)
    std::FILE* f = _wfopen(temp.c_str(), L"wb");
#else
    std::FILE* f = std::fopen(temp.c_str(), "wb");
#endif
    if (!f) return false;
    bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size() && std::fflush(f) == 0;
#if defined(_WIN32)
    if (ok) ok = _commit(_fileno(f)) == 0;
#else
    if (ok) ok = fdatasync(fileno(f)) == 0;
#endif
    ok = std::fclose(f) == 0 && ok;
    if (ok) {
#if defined(_WIN32)
        ok = MoveFileExW(temp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        std::filesystem::rename(temp, file, ec);
        ok = !ec;
#endif
    }
    if (!ok) std::filesystem::remove(temp, ec);
    return ok;
}
bool platinum_due(const Snapshot& s) {
    return !s.earned[0] && std::all_of(s.earned.begin() + 1, s.earned.begin() + kBaseCount,
                                    [](auto when) { return when > 0; });
}
}  // namespace

const std::array<Info, kCount>& catalog() { return kCatalog; }
const char* grade_name(Grade g) {
    switch (g) {
    case Grade::Platinum: return "Platinum";
    case Grade::Gold: return "Gold";
    case Grade::Silver: return "Silver";
    default: return "Bronze";
    }
}
int Snapshot::count(int begin, int end) const {
    begin = std::clamp(begin, 0, kCount);
    end = std::clamp(end, begin, kCount);
    return std::count_if(earned.begin() + begin, earned.begin() + end,
                         [](auto t) { return t > 0; });
}
Snapshot read(const std::filesystem::path& file) { std::string text; return load(file, &text); }
void Store::open(const std::filesystem::path& file) {
    std::lock_guard lock(mutex_);
    file_ = file; notices_.clear(); state_ = read(file_);
}
Snapshot Store::snapshot() const { std::lock_guard lock(mutex_); return state_; }
Unlock Store::unlock(int id, std::int64_t when) {
    if (id < 0 || id >= kCount) return {Status::InvalidId};
    if (id == 0) return {Status::AutomaticPlatinum};
    return update(id, when);
}
Unlock Store::reconcile_platinum(std::int64_t when) { return update(-1, when); }
Unlock Store::update(int id, std::int64_t when) {
    std::lock_guard lock(mutex_);
    if (when <= 0 || when > kLatestTimestamp || file_.empty()) return {Status::IoError};
    std::string text;
    state_ = load(file_, &text);
    if (!state_.error.empty()) return {Status::IoError};
    if (id >= 0 && state_.earned[id]) return {Status::AlreadyUnlocked};
    Snapshot next = state_;
    if (id >= 0) next.earned[id] = when;
    const bool platinum = platinum_due(next);
    if (id < 0 && !platinum) return {Status::Ok};
    if (platinum) next.earned[0] = when;
    if (!text.empty() && text.back() != '\n') text += '\n';
    if (id >= 0) text += std::to_string(id) + " " + std::to_string(when) + "\n";
    if (platinum) text += "0 " + std::to_string(when) + "\n";
    if (!persist(file_, text)) {
        state_.error = "Cannot save trophies. Check free space and write permissions.";
        return {Status::IoError};
    }
    state_ = next;
    if (id >= 0) notices_.push_back(id);
    if (platinum) notices_.push_back(0);
    return {Status::Ok, platinum ? 0 : -1};
}
int Store::pop_notice() {
    std::lock_guard lock(mutex_);
    if (notices_.empty()) return -1;
    const int id = notices_.front(); notices_.pop_front(); return id;
}
Store& runtime() { static Store store; return store; }
}  // namespace trophies
