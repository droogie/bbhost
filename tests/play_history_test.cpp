#undef NDEBUG
#include "core/play_history.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace play_history;
namespace fs = std::filesystem;
namespace {
Observation world(int slot = 0, int cycle = 0, std::uint64_t time = 0) {
    Observation o;
    o.valid = o.in_world = true; o.slot = slot; o.cycle = cycle; o.play_ms = time;
    o.name = "狩人 \"月\""; o.when = 1793833200;
    o.block = bosses()[0].block << 16;
    for (auto& flag : o.flags) flag.valid = true;
    return o;
}
Character last(const Store& store) { return store.snapshot().characters.back(); }
BossRecord boss(const Store& store, int cycle = 0, int i = 0) { return last(store).cycles.at(cycle).bosses[i]; }
std::string text(const fs::path& file) {
    std::ifstream in(file, std::ios::binary); return {std::istreambuf_iterator<char>(in), {}};
}
void sample(Store& store, Observation& o, std::uint64_t delta = 100) { o.play_ms += delta; store.observe(o); }
}
int main() {
    const auto root = fs::temp_directory_path() / ("bbhost-history-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const auto file = root / fs::u8path("周回記録") / "play-history.txt";
    Store store; store.open(file);
    assert(store.snapshot().characters.empty());
    auto o = world(); store.observe(o);
    o.flags[0].begun = true; sample(store, o);
    for (int i = 0; i < 50; ++i) sample(store, o);
    assert(boss(store).attempts == 1);
    o.alive = false; ++o.deaths; sample(store, o);
    assert(last(store).active_boss == -1 && boss(store).attempts == 1);
    // A loading screen or process restart must not re-count a stale begun flag.
    store.observe({}); store.open(file); o.alive = true; sample(store, o);
    assert(boss(store).attempts == 1);
    o.flags[0].begun = false; sample(store, o);
    o.flags[0].begun = true; sample(store, o);
    assert(boss(store).attempts == 2);
    // Quit out and resume the same open fight: exactly one attempt.
    store.flush(); store.open(file); sample(store, o);
    assert(boss(store).attempts == 2);
    o.flags[0].begun = false; sample(store, o); // return to fog without a death
    o.flags[0].begun = true; sample(store, o);
    o.flags[0].defeated = true; o.alive = false; ++o.deaths; sample(store, o);
    assert(boss(store).attempts == 3 && boss(store).kills == 1); // simultaneous death/win
    const auto clear_attempts = boss(store).attempts;
    o.alive = true; o.flags[0].defeated = false; sample(store, o);
    o.flags[0].begun = true; sample(store, o);
    assert(boss(store).attempts == clear_attempts && boss(store).kills == 1);
    const auto ng_end = o.play_ms + 100;
    o.cycle = 1; o.flags[0] = {true, false, false}; sample(store, o);
    auto c = last(store);
    assert(c.cycles.size() == 2 && c.cycles[0].complete && c.cycles[0].elapsed_ms == ng_end);
    assert(c.cycles[1].elapsed_ms == 0 && !c.cycles[1].partial_time);
    o.flags[0].begun = true; sample(store, o);
    o.flags[0].defeated = true; sample(store, o, 12345);
    assert(boss(store, 1).attempts == 1 && boss(store, 1).kills == 1);
    const auto sum = totals(last(store));
    assert(sum[0].attempts == 4 && sum[0].kills == 2 && !sum[0].partial);
    assert(last(store).cycles[1].elapsed_ms == 12445);
    store.flush();
    const auto loaded = read(file);
    assert(loaded.error.empty() && loaded.characters.size() == 1 && loaded.characters[0].name == o.name);
    assert(totals(loaded.characters[0])[0].attempts == 4);
    assert(duration(3600123000ull) == "1000:02:03");

    // Character switches, title-screen zero counters, old-save rollback.
    store.observe({});
    auto other = world(1, 2, 100000); store.observe(other);
    assert(store.snapshot().characters.size() == 2 && last(store).cycles[0].partial_time);
    sample(store, other, 5000); assert(last(store).cycles[0].elapsed_ms == 5000);
    store.observe({}); sample(store, o);
    assert(store.snapshot().characters.size() == 2);
    o.cycle = 0; o.play_ms = 0; o.deaths = 0; o.flags[0] = {true, false, false}; sample(store, o);
    assert(store.snapshot().characters.size() == 3 && last(store).cycles.size() == 1);
    assert(store.snapshot().characters[0].cycles.size() == 2); // original records retained

    // Existing saves: known kills, unknown earlier attempts; no fictional successes.
    Store partial; partial.open(root / "partial.txt");
    auto late = world(0, 4, 999999); late.flags[0].defeated = true; partial.observe(late);
    assert(boss(partial).attempts == 0 && boss(partial).kills == 1 && boss(partial).partial);
    assert(last(partial).cycles[0].elapsed_ms == 0 && last(partial).cycles[0].partial_time);
    late.flags[1].begun = true; sample(partial, late);
    assert(boss(partial, 0, 1).attempts == 1 && boss(partial, 0, 1).partial);
    // Missing flag blocks cannot turn an active attempt off or add a kill.
    late.flags[1].valid = false; sample(partial, late);
    assert(last(partial).active_boss == 1 && boss(partial, 0, 1).kills == 0);
    late.flags[2] = {true, true, false}; sample(partial, late);
    assert(boss(partial, 0, 2).attempts == 0 && boss(partial, 0, 2).kills == 1 && boss(partial, 0, 2).partial);

    // Boss Rush and summoned worlds must not contaminate campaign records.
    Store practice; practice.open(root / "practice.txt");
    auto p = world(); practice.observe(p);
    sample(practice, p, 5000);
    p.suspended = true; p.flags[0].begun = true; sample(practice, p, 1000);
    p.flags[0].defeated = true; ++p.deaths; sample(practice, p, 60000);
    assert(boss(practice).attempts == 0 && boss(practice).kills == 0 && last(practice).cycles[0].elapsed_ms == 5000);
    p.suspended = false; p.flags[0] = {true, false, false}; sample(practice, p, 100);
    sample(practice, p, 1000); assert(last(practice).cycles[0].elapsed_ms == 6000);
    p.own_world = false; p.flags[0] = {true, true, true}; sample(practice, p);
    assert(boss(practice).kills == 0);
    p.own_world = true; p.flags[0] = {true, false, false}; sample(practice, p);
    p.flags[0].begun = true; sample(practice, p);
    assert(boss(practice).attempts == 1);

    // A fight elsewhere in the same map must not count a persistent met flag.
    Store wrong; wrong.open(root / "wrong.txt");
    auto w = world(); w.block = 0x22000000; w.flags[0].begun = true; wrong.observe(w);
    assert(boss(wrong).attempts == 0);
    w.block = bosses()[0].block << 16; sample(wrong, w);
    assert(boss(wrong).attempts == 1);

    // The game's clock saturates, but later NG+ cycles still time.
    Store capped; capped.open(root / "capped.txt");
    auto cap = world(0, 8, kPlayTimeCap); capped.observe(cap);
    cap.capped_delta_ms = 20000; capped.observe(cap);
    assert(last(capped).cycles[0].elapsed_ms == 20000 && last(capped).cycles[0].partial_time);
    cap.cycle = 9; cap.capped_delta_ms = 1000; capped.observe(cap);
    cap.capped_delta_ms = 5000; capped.observe(cap);
    assert(last(capped).cycles[1].elapsed_ms == 5000 && !last(capped).cycles[1].partial_time);
    capped.observe({}); cap.capped_delta_ms = 999999; capped.observe(cap);
    assert(last(capped).cycles[1].elapsed_ms == 5000);

    Store boundary; const auto boundary_file = root / "boundary.txt"; boundary.open(boundary_file);
    auto bo = world(); boundary.observe(bo); sample(boundary, bo, 5000);
    bo.in_world = false; bo.cycle = 1; sample(boundary, bo, 100);
    bo.in_world = true; sample(boundary, bo, 1000);
    assert(last(boundary).cycles[0].elapsed_ms == 5100 && !last(boundary).cycles[0].partial_time);
    assert(last(boundary).cycles[1].elapsed_ms == 1000 && !last(boundary).cycles[1].partial_time);
    boundary.flush(); boundary.open(boundary_file); bo.cycle = 2; sample(boundary, bo, 80000);
    assert(last(boundary).cycles[1].partial_time && last(boundary).cycles[2].partial_time);
    assert(last(boundary).cycles[1].elapsed_ms == 1000 && last(boundary).cycles[2].elapsed_ms == 0);

    Store gap; const auto gap_file = root / "gap.txt"; gap.open(gap_file);
    auto go = world(); gap.observe(go); gap.flush(); gap.open(gap_file);
    sample(gap, go, 60000);
    assert(boss(gap).partial && boss(gap).attempts == 0);

    // Closing the window can leave the game save behind the host checkpoint.
    // A routine autosave rewind must preserve this character and its attempts.
    Store resume; const auto resume_file = root / "resume.txt"; resume.open(resume_file);
    auto ro = world(0, 1, 1000000); resume.observe(ro);
    ro.flags[0].begun = true; sample(resume, ro, 20000); resume.flush(); resume.open(resume_file);
    ro.play_ms -= 15000; resume.observe(ro);
    assert(resume.snapshot().characters.size() == 1 && boss(resume).attempts == 1);
    const auto resumed_time = last(resume).cycles[0].elapsed_ms;
    sample(resume, ro, 1000);
    assert(last(resume).cycles[0].elapsed_ms == resumed_time + 1000);
    resume.flush(); resume.open(resume_file); ro.play_ms -= 400000; resume.observe(ro);
    assert(resume.snapshot().characters.size() == 2); // substantial restore: separate branch
    Store early; const auto early_file = root / "early.txt"; early.open(early_file);
    auto eo = world(); early.observe(eo); sample(early, eo, 10000); early.flush(); early.open(early_file);
    eo.play_ms = 0; early.observe(eo);
    assert(early.snapshot().characters.size() == 1); // first autosave can still be at spawn

    // Strict loading: truncated, duplicate, malformed and future records are
    // preserved and never replaced by a fresh empty history.
    store.flush(); const auto valid = text(file);
    for (const auto& broken : {std::string("BBHOST_PLAY_HISTORY 99\n"), valid.substr(0, valid.rfind("boss ")),
                               valid + "boss 0 3 1 0\n", valid + "unexpected\n"}) {
        const auto damaged = root / "damaged.txt"; std::ofstream(damaged, std::ios::binary) << broken;
        Store guarded; guarded.open(damaged); assert(!guarded.snapshot().error.empty());
        guarded.observe(world()); assert(!guarded.flush() && text(damaged) == broken);
    }
    const auto blocker = root / "blocker"; std::ofstream(blocker) << "keep";
    Store io; io.open(blocker / "history.txt"); io.observe(world());
    assert(!io.flush() && !io.snapshot().error.empty() && text(blocker) == "keep");
    fs::remove_all(root);
    std::cout << "PASS: cycles, attempts, kills, cumulative totals, reloads, characters, partial histories, practice, rollback, persistence\n";
}
