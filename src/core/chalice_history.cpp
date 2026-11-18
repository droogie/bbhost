#include "core/chalice_history.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace chalice_history {
namespace {
constexpr std::uint64_t kMaxCounter = 1ull << 50;
bool end(std::istringstream& in) { in >> std::ws; return in.eof(); }
int summary_id(int id) {
    // Different dungeon templates use separate health-bar IDs for the same
    // displayed boss. Keep encounter IDs exact; combine their boss totals.
    if (id == 209010) return 209000;
    if (id == 509010) return 509000;
    return id;
}
Stats& stats(Encounter& e, int cycle) {
    for (auto& s : e.stats) if (s.cycle == cycle) return s;
    e.stats.push_back({cycle, {}}); return e.stats.back();
}
}
std::string boss_name(int id) {
    // Health-bar name IDs, including the multi-character Watchers event;
    // NpcParam.nameId is different for several of these fights.
    static const std::pair<int, const char*> names[] = {
        {1010, "Forgotten Madman"}, {106000, "Brainsucker"}, {127000, "Maneater Boar"},
        {209000, "Blood-starved Beast"}, {209010, "Blood-starved Beast"}, {216000, "Keeper of the Old Lords"},
        {218000, "Loran Silverbeast"}, {251010, "Ebrietas, Daughter of the Cosmos"}, {257010, "Celestial Emissary"},
        {304001, "Merciless Watchers"}, {305001, "Pthumerian Descendant"}, {305002, "Pthumerian Elder"},
        {306000, "Pthumerian Bride"}, {306001, "Yharnam, Pthumerian Queen"}, {313000, "Undead Giant"},
        {501000, "Watchdog of the Old Lords"}, {504010, "Abhorrent Beast"}, {508010, "Loran Darkbeast"},
        {509000, "Bloodletting Beast"}, {509010, "Bloodletting Beast"}, {510010, "Rom, the Vacuous Spider"},
        {512010, "Amygdala"}, {750000, "Beast-possessed Soul"},
    };
    for (const auto& [key, name] : names) if (key == id) return name;
    return "Chalice boss #" + std::to_string(id);
}
Snapshot read(const std::filesystem::path& file) {
    Snapshot out;
    const auto bad = [&] { out.error = "Cannot read chalice history. The original file has not been changed."; return out; };
    std::error_code ec;
    const bool exists = std::filesystem::exists(file, ec);
    if (ec) return bad();
    if (!exists) return out;
    if (!std::filesystem::is_regular_file(file, ec) || ec || std::filesystem::file_size(file, ec) > 8 * 1024 * 1024 || ec) return bad();
    std::ifstream in(file, std::ios::binary); std::string line;
    if (!in || !std::getline(in, line) || line != "BBHOST_CHALICE_HISTORY 1") return bad();
    std::set<std::uint64_t> ids;
    Encounter* current = nullptr;
    bool finished = false;
    while (std::getline(in, line)) {
        std::istringstream row(line); std::string kind;
        if (!(row >> kind)) continue;
        if (kind == "end") { if (!end(row)) return bad(); finished = true; break; }
        if (kind == "encounter") {
            Encounter e; int cleared, prior, active, blocked;
            if (!(row >> e.id >> e.character >> e.altar >> e.layer >> e.name_id >> e.last_deaths >> cleared >> prior >> active >> blocked >> std::quoted(e.dungeon)) ||
                !end(row) || !e.id || e.id > kMaxCounter || !ids.insert(e.id).second || !e.character || e.character > kMaxCounter ||
                e.altar < 0 || e.altar > 6 || e.layer < 1 || e.layer > 4 || e.name_id <= 0 || e.name_id > 10000000 ||
                e.last_deaths > UINT32_MAX || cleared < 0 || cleared > 1 || prior < 0 || prior > 1 ||
                active < 0 || active > 1 || blocked < 0 || blocked > 1 || e.dungeon.empty() || e.dungeon.size() > 128 ||
                out.encounters.size() >= 10000 || (active && (cleared || blocked)) || (prior && !cleared)) return bad();
            e.cleared = cleared; e.prior_clear = prior; e.active = active; e.blocked = blocked;
            out.encounters.push_back(std::move(e)); current = &out.encounters.back();
        } else if (kind == "stats") {
            Stats s; int partial;
            if (!current || !(row >> s.cycle >> s.count.attempts >> s.count.kills >> partial) || !end(row) ||
                s.cycle < 0 || s.cycle > 1000000 || s.count.attempts > kMaxCounter || s.count.kills > 1 ||
                partial < 0 || partial > 1 || (s.count.kills > s.count.attempts && !partial) ||
                current->stats.size() >= 10000 || (!current->stats.empty() && s.cycle <= current->stats.back().cycle)) return bad();
            s.count.partial = partial; current->stats.push_back(s);
        } else return bad();
    }
    while (std::getline(in, line)) if (line.find_first_not_of(" \r\t") != std::string::npos) return bad();
    if (in.bad() || !finished) return bad();
    for (const auto& e : out.encounters) {
        std::uint64_t kills = 0, attempts = 0;
        for (const auto& s : e.stats) { kills += s.count.kills; attempts += s.count.attempts; }
        if (kills > 1 || (kills && !e.cleared) || (e.prior_clear && (kills || attempts)) || (e.active && !attempts) ||
            (!e.prior_clear && e.stats.empty())) return bad();
    }
    return out;
}
std::vector<Total> totals(const Snapshot& state, std::uint64_t character, int cycle) {
    std::map<int, play_history::BossRecord> sums;
    for (const auto& e : state.encounters) if (e.character == character) {
        if (cycle < 0 && e.prior_clear) sums[summary_id(e.name_id)].partial = true;
        for (const auto& s : e.stats) if (cycle < 0 || s.cycle == cycle) {
            auto& t = sums[summary_id(e.name_id)]; t.attempts += s.count.attempts; t.kills += s.count.kills;
            t.partial = t.partial || s.count.partial;
        }
    }
    std::vector<Total> out;
    for (const auto& [id, count] : sums) out.push_back({id, count});
    return out;
}
void Store::open(const std::filesystem::path& file) {
    std::lock_guard lock(mutex_); file_ = file; state_ = read(file); readable_ = state_.error.empty(); dirty_ = false;
}
Snapshot Store::snapshot() const { std::lock_guard lock(mutex_); return state_; }
bool Store::save() {
    if (!readable_) return false;
    if (!dirty_) return true;
    std::ostringstream out; out << "BBHOST_CHALICE_HISTORY 1\n";
    for (const auto& e : state_.encounters) {
        out << "encounter " << e.id << ' ' << e.character << ' ' << e.altar << ' ' << e.layer << ' ' << e.name_id << ' ' << e.last_deaths
            << ' ' << e.cleared << ' ' << e.prior_clear << ' ' << e.active << ' ' << e.blocked << ' ' << std::quoted(e.dungeon) << '\n';
        for (const auto& s : e.stats) out << "stats " << s.cycle << ' ' << s.count.attempts << ' ' << s.count.kills << ' ' << s.count.partial << '\n';
    }
    out << "end\n";
    if (!play_history::write_history_file(file_, out.str())) {
        state_.error = "Cannot save chalice history. Check free space and write permissions."; return false;
    }
    state_.error.clear(); dirty_ = false; return true;
}
bool Store::flush() { std::lock_guard lock(mutex_); return save(); }
void Store::observe(const Observation& o) {
    std::lock_guard lock(mutex_);
    if (!readable_) return;
    if (!o.character || !o.in_world) { save(); return; }
    if (o.character > kMaxCounter || o.cycle < 0 || o.cycle > 1000000 || o.deaths > UINT32_MAX || o.altar < 0 || o.altar > 6 || o.dungeon.size() > 128) return;
    if (o.suspended) {
        for (auto& e : state_.encounters) if (e.character == o.character && e.active) { e.active = false; dirty_ = true; }
        save(); return;
    }
    for (const auto& boss : o.bosses) {
        if (!boss.flags.valid || o.dungeon.empty() || boss.layer < 1 || boss.layer > 4 || boss.name_id <= 0 || boss.name_id > 10000000) continue;
        Encounter* e = nullptr;
        for (auto i = state_.encounters.rbegin(); i != state_.encounters.rend(); ++i) {
            if (i->character == o.character && i->dungeon == o.dungeon && i->layer == boss.layer && i->name_id == boss.name_id) { e = &*i; break; }
        }
        // An observed cleared dungeon becoming uncleared is a new ritual or
        // a replay of restored progress. Keep the previous clear intact.
        const bool replay = e && e->cleared && !boss.flags.defeated;
        if (!e || replay) {
            Encounter next;
            for (const auto& old : state_.encounters) next.id = std::max(next.id, old.id);
            ++next.id; next.character = o.character; next.dungeon = o.dungeon; next.altar = o.altar;
            next.layer = boss.layer; next.name_id = boss.name_id; next.last_deaths = o.deaths;
            next.cleared = next.prior_clear = boss.flags.defeated;
            if (!next.prior_clear) stats(next, o.cycle).count.partial = !replay;
            state_.encounters.push_back(std::move(next)); e = &state_.encounters.back(); dirty_ = true;
        }
        if (e->cleared) continue; // no inherited clear when NG+ changes
        const bool died = !o.alive || (e->active && o.deaths > e->last_deaths);
        if (!boss.flags.begun) e->blocked = false;
        if (boss.flags.defeated) {
            auto& s = stats(*e, o.cycle).count;
            if (!e->active || !s.attempts) s.partial = true;
            ++s.kills; e->cleared = true; e->active = e->blocked = false; dirty_ = true;
        } else if (e->active && (died || !boss.flags.begun)) {
            e->active = false; e->blocked = boss.flags.begun; dirty_ = true;
        } else if (!e->active && !e->blocked && boss.flags.begun && !died) {
            ++stats(*e, o.cycle).count.attempts; e->active = true; dirty_ = true;
        }
        e->last_deaths = o.deaths;
    }
    for (auto& e : state_.encounters) if (e.character == o.character && e.active) {
        if (o.dungeon != e.dungeon || !o.alive || o.deaths > e.last_deaths) { e.active = false; e.blocked = true; dirty_ = true; }
    }
    save();
}
Store& runtime() { static Store store; return store; }
}
