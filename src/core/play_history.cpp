#include "core/play_history.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace play_history {
namespace {
// Each begun event enables the AI and boss health bar. These are event
// completion flags, reset on rest/death, rather than the persistent "met"
// flags. Verified against the 1.09 map event scripts, including the endings.
const std::array<BossInfo, kBossCount> kBosses{{
    {"Cleric Beast", 12411700, 12414702, 0x1801},
    {"Father Gascoigne", 12411800, 12414802, 0x1801},
    {"Blood-starved Beast", 12301800, 12304802, 0x1700},
    {"Vicar Amelia", 12401800, 12404802, 0x1800},
    {"The Witch of Hemwick", 12201800, 12204802, 0x1600},
    {"Darkbeast Paarl", 12301700, 12304702, 0x1700},
    {"Shadow of Yharnam", 12701800, 12704802, 0x1b00},
    {"Rom, the Vacuous Spider", 13201800, 13204802, 0x2000},
    {"The One Reborn", 12801800, 12804802, 0x1c00},
    {"Martyr Logarius", 12501800, 12504802, 0x1900},
    {"Amygdala", 13301800, 13304802, 0x2100},
    {"Celestial Emissary", 12421700, 12424702, 0x1802},
    {"Ebrietas, Daughter of the Cosmos", 12421800, 12424802, 0x1802},
    {"Micolash, Host of the Nightmare", 12601850, 12604852, 0x1a00},
    {"Mergo's Wet Nurse", 12601800, 12604802, 0x1a00},
    {"Gehrman, the First Hunter", 12101800, 12104802, 0x1500},
    {"Moon Presence", 12101850, 12104852, 0x1500},
    {"Ludwig, the Holy Blade", 13401800, 13404802, 0x2200},
    {"Living Failures", 13501850, 13504852, 0x2300},
    {"Lady Maria of the Astral Clocktower", 13501800, 13504802, 0x2300},
    {"Laurence, the First Vicar", 13401850, 13404852, 0x2200},
    {"Orphan of Kos", 13601800, 13604802, 0x2400},
}};
constexpr std::uint64_t kMaxCounter = 1ull << 50;
constexpr auto kReadError = "Cannot read play history. The original file has not been changed.";
constexpr auto kWriteError = "Cannot save play history. Check free space and write permissions.";

bool persist(const std::filesystem::path& file, const std::string& text) {
    if (file.empty()) return false;
    std::error_code ec;
    if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) return false;
    auto temp = file;
    temp += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
#if defined(_WIN32)
    auto* f = _wfopen(temp.c_str(), L"wb");
#else
    auto* f = std::fopen(temp.c_str(), "wb");
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
bool end(std::istringstream& in) { in >> std::ws; return in.eof(); }
}

const std::array<BossInfo, kBossCount>& bosses() { return kBosses; }
bool write_history_file(const std::filesystem::path& file, const std::string& text) { return persist(file, text); }
std::string cycle_name(int n) { return n == 0 ? "NG (cycle 1)" : "NG+" + std::to_string(n) + " (cycle " + std::to_string(n + 1) + ")"; }
std::string duration(std::uint64_t ms) {
    const auto s = ms / 1000;
    char text[64];
    std::snprintf(text, sizeof text, "%llu:%02u:%02u", static_cast<unsigned long long>(s / 3600),
                  static_cast<unsigned>(s / 60 % 60), static_cast<unsigned>(s % 60));
    return text;
}
std::array<BossRecord, kBossCount> totals(const Character& character) {
    std::array<BossRecord, kBossCount> result{};
    for (const auto& cycle : character.cycles) for (int i = 0; i < kBossCount; ++i) {
        result[i].attempts += cycle.bosses[i].attempts;
        result[i].kills += cycle.bosses[i].kills;
        result[i].partial = result[i].partial || cycle.bosses[i].partial;
    }
    return result;
}

Snapshot read(const std::filesystem::path& file) {
    Snapshot result;
    const auto bad = [&]() { result.error = kReadError; return result; };
    std::error_code ec;
    const bool exists = std::filesystem::exists(file, ec);
    if (ec) return bad();
    if (!exists) return result;
    if (!std::filesystem::is_regular_file(file, ec) || ec || std::filesystem::file_size(file, ec) > 8 * 1024 * 1024 || ec) return bad();
    std::ifstream in(file, std::ios::binary);
    std::string line;
    if (!in || !std::getline(in, line) || line != "BBHOST_PLAY_HISTORY 1") return bad();
    Character* character = nullptr;
    Cycle* cycle = nullptr;
    std::set<std::uint64_t> ids;
    std::set<int> rows;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string kind;
        if (!(row >> kind)) continue;
        if (kind == "character") {
            Character c;
            if (!(row >> c.id >> c.slot >> c.created >> c.last_play_ms >> c.last_deaths >> c.active_boss >> c.ended_boss >> std::quoted(c.name)) ||
                !end(row) || !c.id || c.id > kMaxCounter || !ids.insert(c.id).second || c.slot < 0 || c.slot >= 10 || c.created < 0 ||
                c.last_play_ms > UINT32_MAX || c.last_deaths > UINT32_MAX || c.active_boss < -1 || c.active_boss >= kBossCount ||
                c.ended_boss < -1 || c.ended_boss >= kBossCount || c.name.size() > 256 || result.characters.size() >= 1000 ||
                (character && (character->cycles.empty() || rows.size() != kBossCount))) return bad();
            result.characters.push_back(std::move(c));
            character = &result.characters.back(); cycle = nullptr;
        } else if (kind == "cycle") {
            Cycle c; int complete, partial;
            if (!character || !(row >> c.number >> c.elapsed_ms >> complete >> partial) || !end(row) ||
                c.number < 0 || c.number > 1000000 || c.elapsed_ms > kMaxCounter ||
                (complete != 0 && complete != 1) || (partial != 0 && partial != 1) ||
                character->cycles.size() >= 10000 || (cycle && (!cycle->complete || c.number <= cycle->number || rows.size() != kBossCount))) return bad();
            c.complete = complete; c.partial_time = partial;
            character->cycles.push_back(c); cycle = &character->cycles.back(); rows.clear();
        } else if (kind == "boss") {
            int i, partial; BossRecord b;
            if (!cycle || !(row >> i >> b.attempts >> b.kills >> partial) || !end(row) ||
                i < 0 || i >= kBossCount || !rows.insert(i).second || b.attempts > kMaxCounter || b.kills > 1 ||
                (partial != 0 && partial != 1) || (b.kills > b.attempts && !partial)) return bad();
            b.partial = partial; cycle->bosses[i] = b;
        } else return bad();
    }
    if (in.bad() || (character && (character->cycles.empty() || rows.size() != kBossCount))) return bad();
    for (const auto& c : result.characters) if (c.active_boss >= 0 &&
        (c.cycles.back().bosses[c.active_boss].kills || !c.cycles.back().bosses[c.active_boss].attempts)) return bad();
    return result;
}

void Store::open(const std::filesystem::path& file) {
    std::lock_guard lock(mutex_);
    file_ = file; state_ = read(file); readable_ = state_.error.empty();
    connected_ = suspended_ = dirty_ = false; selected_ = pending_ms_ = 0;
}
Snapshot Store::snapshot() const { std::lock_guard lock(mutex_); return state_; }
std::uint64_t Store::current_character_id() const {
    std::lock_guard lock(mutex_);
    return readable_ && connected_ && state_.error.empty() ? selected_ : 0;
}
bool Store::save() {
    if (!readable_) return false;
    if (!dirty_) return true;
    std::ostringstream out;
    out << "BBHOST_PLAY_HISTORY 1\n";
    for (const auto& c : state_.characters) {
        out << "character " << c.id << ' ' << c.slot << ' ' << c.created << ' ' << c.last_play_ms << ' ' << c.last_deaths << ' '
            << c.active_boss << ' ' << c.ended_boss << ' ' << std::quoted(c.name) << '\n';
        for (const auto& cy : c.cycles) {
            out << "cycle " << cy.number << ' ' << cy.elapsed_ms << ' ' << cy.complete << ' ' << cy.partial_time << '\n';
            for (int i = 0; i < kBossCount; ++i) {
                const auto& b = cy.bosses[i];
                out << "boss " << i << ' ' << b.attempts << ' ' << b.kills << ' ' << b.partial << '\n';
            }
        }
    }
    if (!persist(file_, out.str())) { state_.error = kWriteError; return false; }
    state_.error.clear(); dirty_ = false; pending_ms_ = 0;
    return true;
}
bool Store::flush() { std::lock_guard lock(mutex_); return save(); }

void Store::observe(const Observation& o) {
    std::lock_guard lock(mutex_);
    if (!readable_) return;
    if (!o.valid) {
        if (connected_) save();
        connected_ = false;
        return;
    }
    if (o.slot < 0 || o.slot >= 10 || o.cycle < 0 || o.cycle > 1000000 || o.play_ms > UINT32_MAX || o.deaths > UINT32_MAX) return;
    Character* c = nullptr;
    for (auto it = state_.characters.rbegin(); it != state_.characters.rend(); ++it) {
        if (it->slot == o.slot) { c = &*it; break; }
    }
    // A load may reset the managers temporarily. Only accept monotonic
    // counters from the character already connected, never a title's zeroes.
    if (!o.in_world && (!connected_ || !c || selected_ != c->id || o.cycle < c->cycles.back().number ||
        o.play_ms < c->last_play_ms || o.deaths < c->last_deaths)) { save(); return; }
    bool fresh = false, event = false;
    // A replaced character or restored older save gets its own history. The
    // old branch remains reviewable and never silently loses or gains kills.
    // Window-close/crash can precede the guest's next autosave. A small
    // rewind resumes the existing history, with a new clock baseline.
    // A fresh NG character or a substantially older save gets a new branch.
    const bool rollback = c && (o.cycle < c->cycles.back().number || o.play_ms + 300000 < c->last_play_ms ||
        (o.cycle == 0 && o.play_ms < 5000 && c->last_play_ms > 5000 &&
         !o.name.empty() && !c->name.empty() && o.name != c->name));
    if (!c || rollback) {
        Character next;
        for (const auto& old : state_.characters) next.id = std::max(next.id, old.id);
        ++next.id; next.slot = o.slot; next.name = o.name; next.created = std::max<std::int64_t>(0, o.when);
        next.last_play_ms = o.play_ms; next.last_deaths = o.deaths;
        Cycle cy; cy.number = o.cycle; cy.partial_time = o.cycle != 0 || o.play_ms >= kPlayTimeCap;
        // For NG the save's total playtime is the whole first cycle. Earlier
        // NG+ boundaries cannot be recovered from a lifetime counter.
        cy.elapsed_ms = o.cycle == 0 ? o.play_ms : 0;
        for (auto& b : cy.bosses) b.partial = o.play_ms > 5000 || o.cycle != 0;
        next.cycles.push_back(cy);
        state_.characters.push_back(std::move(next)); c = &state_.characters.back();
        fresh = event = true;
    }
    const bool attaching = !connected_ || selected_ != c->id || suspended_;
    selected_ = c->id;
    connected_ = true;
    auto delta = o.play_ms >= c->last_play_ms ? o.play_ms - c->last_play_ms : 0;
    if (attaching && !fresh && (delta > 1000 || o.deaths > c->last_deaths)) {
        // Play in another build, or a missing checkpoint, may contain failed
        // fights whose flags were reset. Those attempts cannot be recovered.
        for (auto& b : c->cycles.back().bosses) if (!b.kills) b.partial = true;
    }
    // Continue past the native limit with measured world time. A title,
    // load or offline interval must never enter this fallback.
    if (!attaching && c->last_play_ms >= kPlayTimeCap && o.play_ms >= kPlayTimeCap) delta += o.capped_delta_ms;
    const bool missed_boundary = o.cycle > c->cycles.back().number && (attaching || o.cycle != c->cycles.back().number + 1);
    if (!o.suspended && o.own_world && !suspended_ && !fresh && !missed_boundary) c->cycles.back().elapsed_ms += delta;
    pending_ms_ += delta;
    c->last_play_ms = attaching ? o.play_ms : std::max(c->last_play_ms, o.play_ms);
    if (!o.name.empty()) c->name = o.name;
    if (o.cycle > c->cycles.back().number) {
        auto& previous = c->cycles.back();
        previous.partial_time = previous.partial_time || missed_boundary;
        previous.complete = true;
        Cycle next; next.number = o.cycle;
        next.partial_time = missed_boundary || o.suspended || suspended_;
        for (auto& b : next.bosses) b.partial = next.partial_time;
        c->cycles.push_back(next); c->active_boss = c->ended_boss = -1; event = true;
    }
    auto& cy = c->cycles.back();
    if (!o.in_world) { dirty_ = true; if (event || pending_ms_ >= 10000) save(); return; }
    if (o.suspended || !o.own_world) {
        // Practice rewrites defeat/begun flags and may grant rewards. Discard
        // that interval and rebaseline flags on return to campaign play.
        c->active_boss = c->ended_boss = -1;
        c->last_deaths = o.deaths;
        suspended_ = true; dirty_ = true;
        if (event || pending_ms_ >= 10000) save();
        return;
    }
    suspended_ = false;
    const bool died = o.deaths > c->last_deaths || !o.alive;
    // A win takes priority over a simultaneous player death.
    for (int i = 0; i < kBossCount; ++i) {
        const auto& f = o.flags[i]; auto& b = cy.bosses[i];
        if (!f.valid) continue;
        if (!f.begun && c->ended_boss == i) c->ended_boss = -1;
        if (f.defeated && !b.kills) {
            if (c->active_boss != i) {
                b.partial = true;  // already defeated or the fight's start was unobserved
            }
            b.kills = 1; if (c->active_boss == i) c->active_boss = -1;
            event = true;
        }
    }
    if (c->active_boss >= 0) {
        const auto i = c->active_boss; const auto& f = o.flags[i];
        if (died || kBosses[i].block != (o.block >> 16) || (f.valid && !f.begun)) {
            c->active_boss = -1; c->ended_boss = f.valid && !f.begun ? -1 : i; event = true;
        }
    }
    if (!died && c->active_boss < 0) for (int i = 0; i < kBossCount; ++i) {
        const auto& f = o.flags[i]; auto& b = cy.bosses[i];
        if (kBosses[i].block == (o.block >> 16) && f.valid && f.begun && !f.defeated && !b.kills && c->ended_boss != i) {
            ++b.attempts; c->active_boss = i; event = true; break;
        }
    }
    c->last_deaths = o.deaths;
    dirty_ = true;
    if (event || pending_ms_ >= 10000) save();
}
Store& runtime() { static Store store; return store; }
}  // namespace play_history
