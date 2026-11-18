#undef NDEBUG
#include "core/chalice_history.h"
#include "engine/chalice_history.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using namespace chalice_history;
namespace {
std::string contents(const fs::path& file) { std::ifstream in(file); return {std::istreambuf_iterator<char>(in), {}}; }
Observation world() {
    Observation o; o.character = 1; o.cycle = 0; o.in_world = true;
    o.dungeon = "1-1d0a5a00-0123456789abcdef"; o.altar = 1;
    o.bosses.push_back({1, 313000, {true, false, false}});
    return o;
}
play_history::BossRecord total(const Store& store, int cycle = -1, std::uint64_t character = 1) {
    auto rows = totals(store.snapshot(), character, cycle);
    for (const auto& row : rows) if (row.name_id == 313000) return row.count;
    return {};
}
void event_sources() {
    // The actual live-instance layouts: a root/fixed single boss, then the
    // multi-character Watchers template in the pending chain.
    bb::SprjEmkEventIns giant{}, watchers{}, other{};
    constexpr std::uint64_t vt = 0x123400;
    giant.core.vtable = reinterpret_cast<const bb::SprjEmkEventVTablePrefix*>(vt);
    giant.core.event_id = 12906806;
    std::uint32_t ga[]{2900100, 313000, 12901800, 12905502, 12905504};
    giant.arguments = reinterpret_cast<std::uint8_t*>(ga); giant.argument_size = sizeof ga;
    giant.core.next = &other.core;
    other.core.event_id = 1000000001; // completion events are shorter and have no argument buffer
    watchers.core.vtable = giant.core.vtable; watchers.core.event_id = 12906795;
    std::uint32_t wa[]{2900105, 304001, 12901801, 12905511, 12905513, 2900101, 2900103};
    watchers.arguments = reinterpret_cast<std::uint8_t*>(wa); watchers.argument_size = sizeof wa;
    bb::SprjEmkSystem system{}; system.active = &giant.core; system.pending = &watchers.core;
    struct Region { std::uint64_t begin, size; };
    const Region regions[]{
        {reinterpret_cast<std::uint64_t>(&system), sizeof system}, {reinterpret_cast<std::uint64_t>(&giant), sizeof giant},
        {reinterpret_cast<std::uint64_t>(&watchers), sizeof watchers}, {reinterpret_cast<std::uint64_t>(&other), sizeof other},
        {reinterpret_cast<std::uint64_t>(ga), sizeof ga}, {reinterpret_cast<std::uint64_t>(wa), sizeof wa},
    };
    const auto safe = [&](std::uint64_t at, auto* out) {
        for (const auto& r : regions) if (at >= r.begin && at - r.begin <= r.size && sizeof(*out) <= r.size - (at - r.begin)) {
            std::memcpy(out, reinterpret_cast<const void*>(at), sizeof(*out)); return true;
        }
        return false;
    };
    std::vector<chalice_events::Source> rows;
    assert(chalice_events::sources(reinterpret_cast<std::uint64_t>(&system), vt, safe, rows));
    assert(rows.size() == 2 && rows[0].name_id == 313000 && rows[0].layer == 1 && rows[1].name_id == 304001 && rows[1].layer == 2);
    watchers.core.flags = 1;
    assert(chalice_events::sources(reinterpret_cast<std::uint64_t>(&system), vt, safe, rows) && rows.size() == 1);
    other.core.next = &giant.core;
    assert(!chalice_events::sources(reinterpret_cast<std::uint64_t>(&system), vt, safe, rows)); // cycle bounded
    other.core.next = nullptr; giant.arguments = reinterpret_cast<std::uint8_t*>(1);
    assert(!chalice_events::sources(reinterpret_cast<std::uint64_t>(&system), vt, safe, rows)); // bad pointer
}
}
int main() {
    event_sources();
    const auto root = fs::temp_directory_path() / ("bbhost-chalice-history-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const auto file = root / "chalice-history.txt";
    Store store; store.open(file); auto o = world(); store.observe(o);
    assert(total(store).attempts == 0 && total(store).kills == 0);
    o.bosses[0].flags.begun = true; store.observe(o); store.observe(o);
    assert(total(store).attempts == 1);
    store.flush(); store.open(file); store.observe(o);
    assert(total(store).attempts == 1); // an open fight survives process restart
    o.alive = false; ++o.deaths; store.observe(o); assert(!store.snapshot().encounters[0].active);
    o.in_world = false; store.observe(o); store.open(file);
    o.in_world = o.alive = true; store.observe(o);
    assert(total(store).attempts == 1); // stale fight flag after respawn
    o.bosses[0].flags.begun = false; store.observe(o);
    o.bosses[0].flags.begun = true; store.observe(o);
    o.bosses[0].flags.defeated = true; o.alive = false; ++o.deaths; store.observe(o);
    assert(total(store).attempts == 2 && total(store).kills == 1); // win beats simultaneous death
    o.alive = true; o.cycle = 1; store.observe(o); store.observe(o);
    assert(total(store).attempts == 2 && total(store).kills == 1);
    assert(totals(store.snapshot(), 1, 1).empty()); // NG+ does not re-credit inherited chalice clears

    // The same event flags in a different altar/dungeon are independent.
    o.dungeon = "2-1d145a00-aaaaaaaaaaaaaaaa"; o.altar = 2; o.bosses[0].flags = {true, false, false}; store.observe(o);
    o.bosses[0].flags.begun = true; store.observe(o);
    o.bosses[0].flags.defeated = true; store.observe(o);
    assert(store.snapshot().encounters.size() == 2 && total(store).attempts == 3 && total(store).kills == 2);
    assert(total(store, 0).kills == 1 && total(store, 1).kills == 1);

    // Recreating/replaying a dungeon retains the previous encounter.
    o.bosses[0].flags = {true, false, false}; store.observe(o);
    assert(store.snapshot().encounters.size() == 3);
    o.bosses[0].flags.begun = true; store.observe(o);
    o.bosses[0].flags.defeated = true; store.observe(o);
    assert(total(store).attempts == 4 && total(store).kills == 3);

    // A first-seen clear can predate this NG cycle. Keep it visible as an
    // unknown baseline, without inventing a kill in this cycle or totals.
    o.dungeon = "3-1d1e5a00-bbbbbbbbbbbbbbbb"; o.altar = 3; store.observe(o);
    assert(store.snapshot().encounters.back().prior_clear && total(store).kills == 3);
    o.cycle = 2; store.observe(o); assert(total(store, 2).kills == 0);

    // Layers of the same boss stay distinct within one dungeon.
    o.bosses.push_back({2, 313000, {true, false, false}}); store.observe(o);
    o.bosses[1].flags.begun = true; store.observe(o);
    o.bosses[1].flags.valid = false; store.observe(o);
    assert(store.snapshot().encounters.back().active); // missing flag block is not an exit
    o.bosses[1].flags = {true, true, true}; store.observe(o);
    assert(total(store).attempts == 5 && total(store).kills == 4);

    const auto before = total(store);
    o.suspended = true; o.dungeon = "practice"; o.bosses[0].flags = {true, true, true}; store.observe(o);
    assert(total(store).kills == before.kills && total(store).attempts == before.attempts);
    o.suspended = false; o.character = 2; store.observe(o);
    assert(total(store).kills == before.kills && total(store, -1, 2).kills == 0);

    // Story/DLC records are a different file, type and aggregation path.
    play_history::Store campaign; const auto campaign_file = root / "play-history.txt"; campaign.open(campaign_file);
    play_history::Observation p; p.valid = p.in_world = true; p.slot = 0; p.name = "hunter";
    campaign.observe(p); campaign.flush(); const auto campaign_text = contents(campaign_file);
    store.flush(); store.open(file);
    assert(store.snapshot().error.empty() && total(store).kills == 4 && contents(campaign_file) == campaign_text);
    assert(play_history::totals(campaign.snapshot().characters[0])[0].attempts == 0);
    assert(campaign.current_character_id() == 1);

    Store variants; variants.open(root / "variants.txt"); auto v = world();
    v.bosses[0].name_id = 209000; v.bosses[0].flags.begun = true; variants.observe(v);
    v.bosses[0].flags.defeated = true; variants.observe(v);
    v.dungeon = "different-root"; v.bosses[0].name_id = 209010; v.bosses[0].flags = {true, false, true}; variants.observe(v);
    v.bosses[0].flags.defeated = true; variants.observe(v);
    auto combined = totals(variants.snapshot(), 1);
    assert(combined.size() == 1 && combined[0].count.attempts == 2 && combined[0].count.kills == 2);

    const auto valid = contents(file);
    for (const auto& text : {std::string("BBHOST_CHALICE_HISTORY 99\n"), valid.substr(0, valid.size() - 4), valid + "unknown\n"}) {
        const auto damaged = root / "damaged.txt"; std::ofstream(damaged) << text;
        Store guarded; guarded.open(damaged); assert(!guarded.snapshot().error.empty());
        guarded.observe(world()); assert(!guarded.flush() && contents(damaged) == text);
    }
    fs::remove_all(root);
    std::cout << "PASS: chalice dungeon/layer identity, retries, clears, NG+ carry-over, cumulative separation, persistence, root/Watchers event layouts\n";
}
