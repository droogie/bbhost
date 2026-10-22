// The GX lights' blink (decomp/gx/blink.cpp) against the game's own update,
// sub_12a25f0, in the loaded eboot - the effects' schedule (engine/
// sfx_schedule.h) saying which frames take a whole step, as the frame-rate code
// has it:
//   - at 30, where every frame takes a whole step: every update the game's,
//     bit for bit, over the Grand Cathedral's lights (its torches and braziers,
//     the effects' light modules) and over fuzzed blinks (zeros, denormals,
//     infinities, NaNs);
//   - on a frame between whole steps, over fuzzed blinks: a light within its
//     cycle, without a blink, or with a NaN time or length is the game's, bit
//     for bit; one past its cycle's length holds the factor the game's update
//     begins its next cycle with, its time moved on by the game's add, its
//     length and generator untouched (the test's reading of "past" checked
//     against what the game's update does);
//   - at 60, 90, 120 and 144 fps, at 120 with frames up to a fifth long, at
//     uncapped's uneven frames (1/240 to 1/30 s) and at 60 with every tenth
//     frame 1/30, each light beside the same light at 30 (the game's):
//       - its cycles begin only on frames that take a whole step, at most one
//         a step, and each draws the length 30's does, the generator left as
//         30's leaves it (one draw a cycle);
//       - at 60, 90 and 120 every cycle lasts the whole steps it lasts at 30
//         (but a length within 1e-4 s of a whole 1/30 s, which the float sums
//         may put either side: counted), the cycles a game second 30's; at 144
//         and the uneven frames within a whole step of 30's, the cycles a game
//         second within 0.5% of 30's at 144, 1% at 120 and 60 with uneven
//         frames, 2.5% at uncapped's (why: kRates);
//       - within a cycle the update is the game's, bit for bit; the factor
//         held past the cycle's end is the one its next cycle begins with;
//       - the frames held a cycle (the plateau at the high), by rate;
//   - with the hold off (BBHOST_BLINK_HOLD=0) and with no schedule (every
//     frame a whole step): the game's on every frame, bit for bit, and its
//     cycles faster than 30's at 120 - what the hold is for;
// and each of the source's mutants caught. (A held light's time is read by no
// one but the blink, which begins the next cycle whatever it is: not moving it
// on would be an equivalent mutant.)
#define DECOMP_GX_BLINK_MUTANTS
int g_blink_mutant = 0;
#include "decomp/gx/blink.cpp"

#include "eboot_kit.h"

#include "engine/sfx_schedule.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {
bool g_between = false;
}  // namespace

// What the frame-rate code last told the step gate (decomp/sfx/step_gate.h).
bool decomp_sfx_between_steps() { return g_between; }

namespace {

bool g_quiet = false;
int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        if (!g_quiet) std::printf("FAIL: %s\n", what);
    }
}

// A blink as the light keeps it, with room around it.
struct Blink {
    alignas(16) u8 b[0x40];
    u64 at() { return reinterpret_cast<u64>(b); }
    float get(u64 o) const { return load<float>(b, o); }
    u32 bits(u64 o) const { return load<u32>(b, o); }
    void set(u64 o, float v) { std::memcpy(b + o, &v, 4); }
    std::array<u32, 4> state() const { return {load<u32>(b, 4), load<u32>(b, 8), load<u32>(b, 0xc), load<u32>(b, 0x10)}; }
};

bool same(const Blink& x, const Blink& y) { return std::memcmp(x.b, y.b, kBlinkBytes) == 0; }

BlinkUpdate g_game = nullptr;

// The game's update on x, and ours on y, from the same blink; whether ours held.
bool both(Blink& game, Blink& ours, float step) {
    g_game(game.at(), step);
    const u64 before = g_held.load();
    blink_update(ours.at(), step);
    return g_held.load() != before;
}

// --- the lights ------------------------------------------------------------------

struct Light {
    const char* name;
    float shortest, longest, low, high;
};
// The Grand Cathedral boss room's blinking lights (their blinks read in the game: the map's torches and braziers,
// the effects' light modules), a light faster than a frame at 30, one with its cycle's ends given the other way
// round, a factor far from the game's.
const Light kLights[] = {
    {"torch 0.13-0.2 s", 0.13f, 0.2f, 0.85f, 1.0f},     {"torch 0.13-0.3 s", 0.13f, 0.3f, 0.8f, 1.0f},
    {"brazier 0.6-1 s", 0.6f, 1.0f, 0.65f, 1.0f},       {"brazier 0.7-1 s", 0.7f, 1.0f, 0.7f, 1.0f},
    {"light module 1.5-2.5 s", 1.5f, 2.5f, 0.6f, 1.0f}, {"light module 1.25-2.5 s", 1.25f, 2.5f, 0.5f, 1.0f},
    {"light module 0.5-1 s", 0.5f, 1.0f, 0.9f, 1.0f},  {"fast 0.02-0.05 s", 0.02f, 0.05f, 0.5f, 1.0f},
    {"reversed 0.2-0.13 s", 0.2f, 0.13f, 0.85f, 1.0f},  {"wide factor 0.13-0.3 s", 0.13f, 0.3f, -0.4f, 2.5f},
};
constexpr int kLightCount = sizeof kLights / sizeof kLights[0];

// A light as made: no cycle yet (length and time 0), the factor 0, its generator seeded.
Blink fresh(const Light& l, int seed) {
    Blink x;
    std::memset(x.b, 0, sizeof x.b);
    eboot_kit::Rng r(0x12a25f0u + static_cast<u64>(seed) * 7919u);
    for (u64 o = 4; o <= 0x10; o += 4) {
        u32 v = r.u32();
        if (!v) v = 0x9e3779b9u;
        std::memcpy(x.b + o, &v, 4);
    }
    x.set(kShortest, l.shortest);
    x.set(kLongest, l.longest);
    x.set(0x1c, l.low);
    x.set(kHigh, l.high);
    return x;
}

// A cycle as it began: the frame (or whole step), the length drawn, the generator after it.
struct Cycle {
    long step;
    u32 length;
    std::array<u32, 4> state;
    int held;  // frames held past its end (before the next cycle)
};

// The light at 30, the game's update every frame: its cycles over `seconds` game seconds.
std::vector<Cycle> run30(const Light& l, int seed, double seconds) {
    Blink x = fresh(l, seed);
    std::vector<Cycle> c;
    const float step = 1.0f / 30.0f;
    const long frames = static_cast<long>(seconds * 30.0 + 0.5);
    for (long f = 0; f < frames; ++f) {
        const auto s0 = x.state();
        g_game(x.at(), step);
        if (x.state() != s0) c.push_back({f, x.bits(kLength), x.state(), 0});
    }
    return c;
}

struct Rate {
    const char* name;
    std::function<float(long, eboot_kit::Rng&)> step;
    bool aligned;      // a whole number of frames a whole step: the cycles 30's, step for step
    double tolerance;  // the cycles a game second against 30's
};

struct RunResult {
    std::vector<Cycle> cycles;
    double game_s = 0;
    long frames = 0, whole = 0, held = 0, not_game = 0, begun_between = 0, twice_a_step = 0, held_factor_wrong = 0;
    long held_state_moved = 0, max_between = 0;  // the most frames between two whole steps
};

// The light at a rate, ours with the schedule: what it did, checked frame by frame against the game's update on a
// copy. `schedule` false: no schedule (every frame a whole step).
RunResult run_rate(const Light& l, int seed, const Rate& rate, double seconds, bool schedule = true) {
    Blink x = fresh(l, seed);
    RunResult r;
    eboot_kit::Rng jitter(0x5eedu + static_cast<u64>(seed));
    double carry = 0.0;
    long whole_steps = -1, last_begin_step = -2, between_run = 0;
    u32 held_factor = 0;
    bool holding = false;
    int held_now = 0;
    double t = 0.0;
    while (t < seconds) {
        const float step = rate.step(r.frames, jitter);
        const sfx_schedule::Frame sf = sfx_schedule::next(carry, static_cast<double>(step), 30);
        g_between = schedule && sf.whole == 0;
        if (!g_between) ++whole_steps;
        if (g_between) {
            r.max_between = std::max(r.max_between, ++between_run);
        } else {
            between_run = 0;
        }
        Blink game = x;
        const auto s0 = x.state();
        const u32 length0 = x.bits(kLength);
        const float time0 = x.get(kTime);
        const bool held = both(game, x, step);
        const bool begun = x.state() != s0;
        if (held) {
            ++r.held;
            ++held_now;
            // what the game's update would have done here: begun its next cycle, whose first factor ours holds
            if (x.bits(kFactor) != game.bits(kFactor)) ++r.held_factor_wrong;
            if (holding && x.bits(kFactor) != held_factor) ++r.held_factor_wrong;
            // nothing drawn, the length kept, the time moved on by the game's add
            const float moved = time0 + step;
            std::uint32_t mb;
            std::memcpy(&mb, &moved, 4);
            if (x.state() != s0 || x.bits(kLength) != length0 || x.bits(kTime) != mb) ++r.held_state_moved;
            held_factor = x.bits(kFactor);
            holding = true;
        } else if (!same(game, x)) {
            ++r.not_game;
        }
        if (begun) {
            if (g_between) ++r.begun_between;
            if (whole_steps == last_begin_step) ++r.twice_a_step;
            last_begin_step = whole_steps;
            if (!r.cycles.empty()) r.cycles.back().held = held_now;
            // the factor held before it is the one it begins with
            if (holding && x.bits(kFactor) != held_factor) ++r.held_factor_wrong;
            r.cycles.push_back({whole_steps, x.bits(kLength), x.state(), 0});
            held_now = 0;
            holding = false;
        }
        t += static_cast<double>(step);
        ++r.frames;
    }
    r.whole = whole_steps + 1;
    r.game_s = t;
    return r;
}

float fixed(double fps) { return static_cast<float>(1.0 / fps); }

// At a steady rate whose frames make up a whole step the cycles are 30's, step for step. Otherwise a whole step's
// frame begins some way before the step's boundary, and a cycle's time counts from the start of the frame it began on;
// the frame on which a cycle ends tends to be one that begins close before its boundary (its time then reads
// highest), so the next cycle's time runs a little behind its whole steps and the cycles come out a little fewer:
// more so the more uneven the frames (a fraction of a frame a cycle, against the game's half a frame dropped at 30).
const Rate kRates[] = {
    {"60", [](long, eboot_kit::Rng&) { return fixed(60); }, true, 0.001},
    {"90", [](long, eboot_kit::Rng&) { return fixed(90); }, true, 0.001},
    {"120", [](long, eboot_kit::Rng&) { return fixed(120); }, true, 0.001},
    {"144", [](long, eboot_kit::Rng&) { return fixed(144); }, false, 0.005},
    {"120, frames up to a fifth long", [](long, eboot_kit::Rng& j) { return j.uniform(1.0f / 120.0f, 1.0f / 100.0f); }, false, 0.01},
    {"uncapped's uneven frames (1/240 to 1/30 s)", [](long, eboot_kit::Rng& j) { return j.uniform(1.0f / 240.0f, 1.0f / 30.0f); }, false,
     0.025},
    {"60, every tenth frame 1/30", [](long f, eboot_kit::Rng&) { return f % 10 == 9 ? fixed(30) : fixed(60); }, false, 0.01},
};

// --- at 30 ---------------------------------------------------------------------

void at_thirty(double seconds) {
    long frames = 0, differ = 0, cycles = 0;
    for (int i = 0; i < kLightCount; ++i) {
        Blink game = fresh(kLights[i], i), ours = game;
        double carry = 0.0;
        for (long f = 0; f < static_cast<long>(seconds * 30); ++f) {
            const float step = fixed(30);
            g_between = sfx_schedule::next(carry, static_cast<double>(step), 30).whole == 0;
            check(!g_between, "at 30 every frame takes a whole step");
            const auto s0 = game.state();
            both(game, ours, step);
            cycles += game.state() != s0;
            differ += !same(game, ours);
            ++frames;
        }
    }
    if (!g_quiet) std::printf("  at 30: %ld light updates (%ld cycles begun), %ld not the game's\n", frames, cycles, differ);
    check(differ == 0 && cycles > 0, "at 30 every update the game's");
}

// --- fuzzed blinks, on whole steps and between them --------------------------------

eboot_kit::Rng g_rng(0x12a25f0);

Blink fuzzed() {
    Blink x;
    for (u64 o = 0; o < sizeof x.b; o += 4) {
        const u32 v = g_rng.u32();
        std::memcpy(x.b + o, &v, 4);
    }
    const bool none = g_rng.chance(0.1);
    x.set(kShortest, none ? 0.0f : g_rng.value(0.0f, 3.0f));
    x.set(kLongest, none ? (g_rng.chance(0.5) ? -0.0f : 0.0f) : g_rng.value(0.0f, 3.0f));
    x.set(0x1c, g_rng.value(-2.0f, 2.0f));
    x.set(kHigh, g_rng.value(-2.0f, 2.0f));
    const float length = g_rng.value(0.0f, 3.0f);
    x.set(kLength, length);
    // the time: about the length, past it as often as not
    x.set(kTime, g_rng.chance(0.2) ? length : g_rng.value(0.0f, 3.5f));
    x.set(kFactor, g_rng.value(0.0f, 1.0f));
    return x;
}

void fuzz(int cases) {
    long held = 0, game_path = 0, wrong = 0, oracle = 0;
    for (int between = 0; between < 4; ++between) {
        g_rng.specials = between >= 2;
        for (int n = 0; n < cases / 2; ++n) {
            Blink orig = fuzzed();
            Blink game = orig, ours = orig;
            const float step = g_rng.chance(0.05) ? g_rng.value(0.0f, 0.04f) : g_rng.uniform(1.0f / 240.0f, 1.0f / 30.0f);
            g_between = (between & 1) != 0;
            const bool h = both(game, ours, step);
            // the test's reading of "past its cycle's end", as the game's update has it (under the console's MXCSR, as
            // ours runs): a blink, and the time above the length
            const bool none = orig.get(kShortest) == 0.0f && orig.get(kLongest) == 0.0f;
            const bool past = !none && orig.get(kTime) > orig.get(kLength);
            // ... which the game's own update bears out: a new cycle's time is 0 + the step
            if (past) {
                const float started = 0.0f + step;
                std::uint32_t sb;
                std::memcpy(&sb, &started, 4);
                oracle += game.bits(kTime) != sb;
            }
            if (!g_between || !past) {
                ++game_path;
                wrong += h || !same(game, ours);
                continue;
            }
            ++held;
            Blink expect = orig;
            expect.set(kTime, orig.get(kTime) + step);
            std::memcpy(expect.b + kFactor, game.b + kFactor, 4);  // the game's next cycle's first factor
            wrong += !h || !same(expect, ours);
        }
    }
    if (!g_quiet)
        std::printf("  fuzzed: %ld updates the game's, %ld held past their cycle's end between whole steps, %ld wrong; the reading of "
                    "\"past\" %s the game's\n",
                    game_path, held, wrong, oracle ? "NOT" : "is");
    check(oracle == 0, "the test reads \"past\" as the game's update does");
    check(wrong == 0, "fuzzed blinks: the game's, or held with its next cycle's first factor");
    check(held > 0, "fuzzed blinks held");
}

// --- at the rates ------------------------------------------------------------------

struct Totals {
    long cycles30 = 0, cycles = 0, compared = 0, same_steps = 0, boundary = 0, off_by_one = 0, further = 0, draws_differ = 0;
    long held = 0, not_game = 0, begun_between = 0, twice = 0, factor_wrong = 0, state_moved = 0, max_between = 0;
    std::map<int, long> plateau;
    double game_s = 0;
    std::string lights;  // the two torches' cycles a game second, for the report
};

// Within 1e-4 s of a whole 1/30 s: the float sums of 30's frames and of the rate's may fall either side of it.
bool near_whole(u32 bits) {
    float l;
    std::memcpy(&l, &bits, 4);
    const double s = static_cast<double>(l) * 30.0;
    return std::fabs(s - std::round(s)) < 3e-3;
}

Totals one_rate(const Rate& rate, double seconds, bool report) {
    Totals T;
    for (int i = 0; i < kLightCount; ++i) {
        const std::vector<Cycle> c30 = run30(kLights[i], i, seconds);
        const RunResult r = run_rate(kLights[i], i, rate, seconds);
        T.cycles30 += static_cast<long>(c30.size());
        T.cycles += static_cast<long>(r.cycles.size());
        T.held += r.held;
        T.not_game += r.not_game;
        T.begun_between += r.begun_between;
        T.twice += r.twice_a_step;
        T.factor_wrong += r.held_factor_wrong;
        T.state_moved += r.held_state_moved;
        T.max_between = std::max(T.max_between, r.max_between);
        T.game_s += r.game_s;
        const size_t n = std::min(c30.size(), r.cycles.size());
        for (size_t k = 0; k < n; ++k) {
            T.draws_differ += c30[k].length != r.cycles[k].length || c30[k].state != r.cycles[k].state;
            if (k + 1 < r.cycles.size()) ++T.plateau[r.cycles[k].held];
            if (k + 1 >= n) continue;
            ++T.compared;
            const long d30 = c30[k + 1].step - c30[k].step, d = r.cycles[k + 1].step - r.cycles[k].step;
            if (d == d30) {
                ++T.same_steps;
            } else if (near_whole(c30[k].length)) {
                ++T.boundary;
            } else if (std::labs(d - d30) == 1) {
                ++T.off_by_one;
            } else {
                ++T.further;
            }
        }
        // the cycles a game second, the light's own: 30's over the same game time
        if (report && i < 2) {
            char line[160];
            std::snprintf(line, sizeof line, "    %s: %.3f cycles a game second (30: %.3f)\n", kLights[i].name, r.cycles.size() / r.game_s,
                          c30.size() / seconds);
            T.lights += line;
        }
    }
    return T;
}

void rates(double seconds) {
    for (const Rate& rate : kRates) {
        const Totals T = one_rate(rate, seconds, true);
        long pl_n = 0, pl_sum = 0, pl_max = 0;
        for (const auto& [k, v] : T.plateau) {
            pl_n += v;
            pl_sum += static_cast<long>(k) * v;
            pl_max = std::max<long>(pl_max, k);
        }
        const double per30 = static_cast<double>(T.cycles30), per = static_cast<double>(T.cycles);
        if (!g_quiet) {
            std::printf("  %s: %ld cycles (30: %ld, %+.2f%%); of %ld cycles beside 30's, %ld the same whole steps, %ld a length at a whole "
                        "1/30 s, %ld a step either way, %ld further; %ld held frames, %.2f a cycle (most %ld; ",
                        rate.name, T.cycles, T.cycles30, 100.0 * (per / per30 - 1.0), T.compared, T.same_steps, T.boundary, T.off_by_one,
                        T.further, T.held, pl_n ? static_cast<double>(pl_sum) / pl_n : 0.0, pl_max);
            bool first = true;
            for (const auto& [k, v] : T.plateau) {
                std::printf("%s%d: %.1f%%", first ? "" : ", ", k, 100.0 * v / pl_n);
                first = false;
            }
            std::printf(")\n%s", T.lights.c_str());
        }
        check(T.not_game == 0, "within a cycle, and on whole steps, the game's update");
        check(T.begun_between == 0, "cycles begin only on whole steps");
        check(T.twice == 0, "at most one cycle begun a whole step");
        check(T.factor_wrong == 0, "the factor held is the next cycle's first, the same on every held frame");
        check(T.state_moved == 0, "nothing drawn on a held frame");
        check(T.draws_differ == 0, "each cycle draws 30's length, the generator left as 30's leaves it");
        check(T.further == 0, "every cycle within a whole step of 30's");
        if (rate.aligned) {
            check(T.off_by_one == 0, "at 60, 90 and 120 every cycle 30's whole steps (but at a whole 1/30 s)");
            check(T.boundary * 100 <= T.compared, "lengths at a whole 1/30 s are few");
        }
        check(std::fabs(per / per30 - 1.0) < rate.tolerance, "the cycles a game second 30's (within the rate's bound)");
        check(pl_max <= T.max_between, "no more frames held a cycle than lie between two whole steps");
        // the plateau at the steady rates: a cycle's end falls anywhere in its last whole step, so 0 to n-1 of the
        // step's n frames are held, (n-1)/2 on average
        const int n = rate.aligned ? (std::string(rate.name) == "60" ? 2 : std::string(rate.name) == "90" ? 3 : 4) : 0;
        if (n && pl_n) {
            const double avg = static_cast<double>(pl_sum) / pl_n;
            check(pl_max == n - 1, "the most frames held a cycle: a whole step's but one");
            check(std::fabs(avg - (n - 1) / 2.0) < 0.15 * (n - 1), "the frames held a cycle: half a whole step's but one on average");
        }
    }
}

// --- the hold off, no schedule -------------------------------------------------------

void hold_off(double seconds) {
    const Rate& r120 = kRates[2];
    long differ = 0, frames = 0;
    double fast30 = 0, fast120 = 0;
    for (int pass = 0; pass < 2; ++pass) {
        // pass 0: BBHOST_BLINK_HOLD=0; pass 1: the hold on, but no schedule (every frame a whole step)
        g_hold_on = pass == 1;
        for (int i = 0; i < kLightCount; ++i) {
            const RunResult r = run_rate(kLights[i], i, r120, seconds, pass == 0);
            differ += r.not_game + r.held;
            frames += r.frames;
            if (i == 0 && pass == 0) {
                fast120 = r.cycles.size() / r.game_s;
                fast30 = run30(kLights[i], i, seconds).size() / seconds;
            }
        }
    }
    g_hold_on = true;
    if (!g_quiet)
        std::printf("  the hold off, and no schedule: %ld light updates at 120, %ld not the game's; the torch at 120 the game's way: "
                    "%.3f cycles a game second (30: %.3f, %+.1f%%)\n",
                    frames, differ, fast120, fast30, 100.0 * (fast120 / fast30 - 1.0));
    check(differ == 0, "the hold off or no schedule: the game's every frame");
    check(fast120 > fast30 * 1.03, "the game's way, the torch blinks faster at 120 than at 30");
}

int run_all(double seconds, int fuzz_cases) {
    g_fail = 0;
    at_thirty(seconds / 4);
    fuzz(fuzz_cases);
    rates(seconds);
    hold_off(seconds / 4);
    return g_fail;
}

}  // namespace

int main() {
    eboot_kit::load("decomp_gx_blink");
    if (std::memcmp(eboot_kit::at(kBlinkUpdate), kBlinkEntry, sizeof kBlinkEntry) != 0) {
        std::printf("decomp_gx_blink: sub_12a25f0's entry is not the one the leaf expects\n");
        return 1;
    }
    g_game = eboot_kit::fn<BlinkUpdate>(kBlinkUpdate);
    g_game_blink = eboot_kit::at(kBlinkUpdate);
    std::printf("decomp_gx_blink: the GX lights' blink, its cycles begun on whole effect steps\n");
    const int fails = run_all(600.0, 200000);
    // the mutants' runs, shorter: without a mutant they pass
    g_quiet = true;
    const int control = run_all(150.0, 20000);
    g_quiet = false;
    if (control) std::printf("decomp_gx_blink: the mutants' shorter run fails without a mutant (%d failures)\n", control);
    static const char* const kMutants[] = {"",
                                           "the hold off (every update the game's)",
                                           "a light without a blink held",
                                           "a NaN time or length taken as past the cycle's end",
                                           "no cycle begun on a whole step either",
                                           "the factor left as it was",
                                           "the factor held at the high, not the next cycle's first",
                                           "the light's own generator drawn on a held frame"};
    constexpr int kCount = sizeof kMutants / sizeof kMutants[0] - 1;
    int caught = 0;
    g_quiet = true;
    for (int m = 1; m <= kCount; ++m) {
        g_blink_mutant = m;
        g_rng = eboot_kit::Rng(0x12a25f0 + m);
        const int f = run_all(150.0, 20000);
        caught += f > 0;
        std::printf("  mutant %d (%s): %s\n", m, kMutants[m], f ? "caught" : "NOT CAUGHT");
    }
    g_blink_mutant = 0;
    g_quiet = false;
    const bool pass = fails == 0 && control == 0 && caught == kCount;
    std::printf("decomp_gx_blink: %s (%d failures; %d of %d mutants caught)\n", pass ? "pass" : "FAIL", fails, caught, kCount);
    return pass ? 0 : 1;
}
