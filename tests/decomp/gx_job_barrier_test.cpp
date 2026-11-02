// The job barrier's step (decomp/gx/job_barrier.cpp) against the game's own
// sub_25b9980 in the loaded eboot, on GX's job system laid out in test memory
// (2-8 participants of 0x530 bytes, participant 0 the caller), the deque's pop
// (sub_1247b60) and steal (sub_1247d20) and sceKernelReadTsc standing in for
// the game's, and jobs whose run records what ran where:
//   - BBHOST_GX_BARRIER_STEAL=0: the same calls in the same order with the
//     same arguments (pops, steals, the time-stamp reads, the job run with its
//     participant's index) and every deque the same afterwards, over random
//     deques - the caller's empty or not, the others empty or not, one job or
//     many;
//   - on: the game's step, then - when the caller's deque is empty - the same
//     sweep once more, from a fresh time-stamp, without a pop: the caller's
//     deque then holds jobs exactly when a sweep could steal some;
// and each of the source's mutants caught.
#define DECOMP_GX_JOB_BARRIER_MUTANTS
int g_jb_mutant = 0;
#include "decomp/gx/job_barrier.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x25b9980);
int g_failed = 0;
bool g_quiet = false;

void fail(int trial, const char* why) {
    if (++g_failed <= 30 && !g_quiet) std::printf("FAIL: case %d: %s\n", trial, why);
}

constexpr int kMaxParticipants = 8;
alignas(64) u8 g_participants[kMaxParticipants * kParticipantSize];
alignas(64) u8 g_system[64];
alignas(64) u64 g_job_objs[256 * 2];  // job objects: [vtable, id]
u64 g_job_vtable[4];

// The deques, by their address (participant + 0x18); the count the game reads
// at +0x38 (the deque's +0x20) kept with them.
std::map<u64, std::deque<u64>> g_deques;
void sync_count(u64 deque) { store<u64>(deque + 0x20, g_deques[deque].size()); }

std::vector<std::string> g_log;
std::vector<u64> g_tsc;
size_t g_tsc_at = 0;

GUEST_ABI bool stub_pop(u64 deque, u64* job) {
    std::deque<u64>& d = g_deques[deque];
    char b[96];
    std::snprintf(b, sizeof b, "pop %llx %zu", static_cast<ull>(deque), d.size());
    g_log.push_back(b);
    if (d.empty()) return false;
    *job = d.front();
    d.pop_front();
    sync_count(deque);
    return true;
}
GUEST_ABI bool stub_steal(u64 victim, u64 thief) {
    std::deque<u64>& v = g_deques[victim];
    std::deque<u64>& t = g_deques[thief];
    char b[96];
    std::snprintf(b, sizeof b, "steal %llx<-%llx %zu %zu", static_cast<ull>(thief), static_cast<ull>(victim), t.size(), v.size());
    g_log.push_back(b);
    if (!t.empty() || v.empty()) return false;
    const size_t n = (v.size() + 1) / 2;
    for (size_t i = 0; i < n; ++i) {
        t.push_back(v.front());
        v.pop_front();
    }
    sync_count(victim);
    sync_count(thief);
    return true;
}
GUEST_ABI u64 stub_tsc() {
    const u64 v = g_tsc_at < g_tsc.size() ? g_tsc[g_tsc_at] : 0;
    ++g_tsc_at;
    char b[48];
    std::snprintf(b, sizeof b, "tsc %llx", static_cast<ull>(v));
    g_log.push_back(b);
    return v;
}
GUEST_ABI void stub_run(u64 job, u32 index) {
    char b[64];
    std::snprintf(b, sizeof b, "run %llu on %u", static_cast<ull>(load<u64>(job + 8)), index);
    g_log.push_back(b);
}

struct State {
    int n;
    std::vector<std::deque<u64>> deques;
    std::vector<u32> index;
};

State random_state() {
    State s;
    s.n = g_rng.range(2, kMaxParticipants);
    s.deques.resize(s.n);
    s.index.resize(s.n);
    int next = 0;
    for (int i = 0; i < s.n; ++i) {
        s.index[i] = static_cast<u32>(g_rng.chance(0.9) ? i : g_rng.range(0, 1000));
        const int k = g_rng.chance(0.35) ? 0 : g_rng.chance(0.4) ? 1 : g_rng.range(2, 12);
        for (int j = 0; j < k && next < 256; ++j) s.deques[i].push_back(reinterpret_cast<u64>(&g_job_objs[2 * next++]));
    }
    return s;
}

void install(const State& s) {
    std::memset(g_participants, 0, sizeof g_participants);
    std::memset(g_system, 0, sizeof g_system);
    store<u64>(reinterpret_cast<u64>(g_system), reinterpret_cast<u64>(g_participants));
    store<u32>(reinterpret_cast<u64>(g_system) + 8, static_cast<u32>(s.n));
    g_deques.clear();
    for (int i = 0; i < s.n; ++i) {
        const u64 p = reinterpret_cast<u64>(g_participants) + static_cast<u64>(i) * kParticipantSize;
        store<u64>(p + 8, reinterpret_cast<u64>(g_system));
        store<u32>(p + 0xa0, s.index[i]);
        g_deques[p + 0x18] = s.deques[i];
        sync_count(p + 0x18);
    }
    g_log.clear();
    g_tsc_at = 0;
}

std::vector<std::deque<u64>> deques_now(int n) {
    std::vector<std::deque<u64>> out(n);
    for (int i = 0; i < n; ++i) out[i] = g_deques[reinterpret_cast<u64>(g_participants) + static_cast<u64>(i) * kParticipantSize + 0x18];
    return out;
}

using StepFn = GUEST_ABI void (*)(u64);

// The game's step on a state; its log and deques.
void run_game(const State& s, std::vector<std::string>* log, std::vector<std::deque<u64>>* after) {
    install(s);
    eboot_kit::fn<StepFn>(kRunOne)(reinterpret_cast<u64>(g_participants));
    *log = g_log;
    *after = deques_now(s.n);
}
void run_ours(const State& s, bool steal, std::vector<std::string>* log, std::vector<std::deque<u64>>* after) {
    install(s);
    g_steal = steal;
    gx_job_run_one(reinterpret_cast<u64>(g_participants));
    *log = g_log;
    *after = deques_now(s.n);
}

int differential(int cases) {
    int bad = 0, stole_after = 0, swept_after = 0;
    for (int t = 0; t < cases; ++t) {
        const State s = random_state();
        g_tsc.clear();
        for (int i = 0; i < 4; ++i) g_tsc.push_back(g_rng.next() >> g_rng.range(0, 40));
        std::vector<std::string> lg, lo;
        std::vector<std::deque<u64>> ag, ao;
        run_game(s, &lg, &ag);
        run_ours(s, false, &lo, &ao);
        if (lg != lo || ag != ao) {
            ++bad;
            fail(t, "off: not the game's calls or deques");
            continue;
        }
        // On: the game's calls first, then a sweep when the caller's deque is empty.
        std::vector<std::string> ls;
        std::vector<std::deque<u64>> as;
        run_ours(s, true, &ls, &as);
        if (ls.size() < lg.size() || !std::equal(lg.begin(), lg.end(), ls.begin())) {
            ++bad;
            fail(t, "on: the game's step not made first");
            continue;
        }
        const bool own_empty = ag[0].empty();
        if (!own_empty) {
            if (ls.size() != lg.size() || as != ag) {
                ++bad;
                fail(t, "on: swept although the caller's deque had jobs");
            }
            continue;
        }
        ++swept_after;
        bool any = false;
        for (int i = 1; i < s.n; ++i) any = any || !ag[i].empty();
        if (any != !as[0].empty()) {
            ++bad;
            fail(t, "on: the caller's deque after the sweep does not match what could be stolen");
            continue;
        }
        size_t total_g = 0, total_s = 0;
        for (int i = 0; i < s.n; ++i) {
            total_g += ag[i].size();
            total_s += as[i].size();
        }
        if (total_g != total_s) {
            ++bad;
            fail(t, "on: jobs lost or made by the sweep");
            continue;
        }
        for (size_t i = lg.size(); i < ls.size(); ++i) {
            if (ls[i].rfind("pop", 0) == 0 || ls[i].rfind("run", 0) == 0) {
                ++bad;
                fail(t, "on: the sweep popped or ran a job");
                break;
            }
        }
        if (!as[0].empty()) ++stole_after;
    }
    std::printf("  %d cases: %d differ; with the switch on %d swept again, %d of them took jobs\n", cases, bad, swept_after,
                stole_after);
    return bad;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_gx_job_barrier");
    if (std::memcmp(eboot_kit::at(kRunOne), kEntry, sizeof kEntry) != 0) {
        std::printf("decomp_gx_job_barrier: sub_25b9980 is not what the decomp expects\n");
        return 1;
    }
    eboot_kit::stub(kPop, reinterpret_cast<const void*>(&stub_pop));
    eboot_kit::stub(kSteal, reinterpret_cast<const void*>(&stub_steal));
    eboot_kit::stub(kReadTsc, reinterpret_cast<const void*>(&stub_tsc));
    g_job_vtable[0] = reinterpret_cast<u64>(&stub_run);
    for (int i = 0; i < 256; ++i) {
        g_job_objs[2 * i] = reinterpret_cast<u64>(g_job_vtable);
        g_job_objs[2 * i + 1] = static_cast<u64>(i);
    }
    int bad = differential(20000);
    // Each mutant must be caught.
    int missed = 0;
    for (int m = 1; m <= kMutants; ++m) {
        g_jb_mutant = m;
        g_quiet = true;
        const int was = g_failed;
        const int found = differential(4000);
        g_quiet = false;
        g_failed = was;
        if (!found) {
            std::printf("FAIL: mutant %d not caught\n", m);
            ++missed;
        }
    }
    g_jb_mutant = 0;
    std::printf("  %d of %d mutants caught\n", kMutants - missed, kMutants);
    bad += missed;
    std::printf("decomp_gx_job_barrier: %s\n", bad ? "FAILED" : "the game's step exactly with the switch off; on, one sweep more, exactly when the caller's deque is empty");
    return bad ? 1 : 0;
}
