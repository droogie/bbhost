// The keyboard's movement keys on the left stick (host/bindings.cpp), frame by
// frame as the game reads them: the window's pump applies the keys held every
// 4 ms (host/window.cpp update_pad) and the game reads the last pad once a frame
// at 30, 60 or 120 fps (hle/system.cpp hle_pad_read). Each read is shown as the
// stick's magnitude (1 a full push along an axis, 1.41 the square's corner) and
// angle (0 forward, 90 right), against two models: v0.2.21's push (a full push
// the moment a key is held, in its direction at once) and v0.2.22's (the same
// direction at once, the push from rest over 66 ms again whenever no movement
// key is held - between two keys of a typed change of direction too).
//
// What it holds the code to, on both sides:
//  - No sprint turn while Dodge / Sprint is held: the stick never swings 90
//    degrees or more at a full push between two of the game's reads (with
//    nothing but rest between them) - where the character script stops a
//    sprint or spins the hunter round (f9227ee5). Not with W held and A and D
//    alternating (b), nor when W+D meets A (c), nor when W is let go and D
//    pressed a moment later (d). The stick goes round at a full push, 90
//    degrees in 70 ms (about 45 a read at 30 fps), and right round (W+A to
//    S+D) by the front.
//  - But a straight reversal - A and D in turn without W or S (a), W and S
//    without A or D - flips the stick at once, as v0.2.21's push did and a
//    thumb flicking a pad's stick does: the game's quick turn. Also when it
//    comes while the stick is still turning.
//  - No dip into the walk on a turn: between the keys' directions the push stays
//    full, so a typed W to D keeps the sprint (v0.2.22's ramp went through the
//    walk there and the sprint came back late).
//  - Without it the stick points where the keys do at once, as v0.2.21's
//    did, and letting go of it part of the way round puts it there: the game
//    dodges on that release, in the stick's direction.
//  - A start from standing ramps up as v0.2.22's does; Walk and the mouse's
//    buttons push as before.
#include "core/config.h"
#include "host/bindings.h"
#include "host/window.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// What host/bindings.cpp asks of the rest of the host.
const HostConfig& config() {
    static HostConfig c;
    return c;
}
bool debug_menu_active() { return false; }
std::uint32_t menu_confirm_button() { return 0x4000u; }
std::uint32_t menu_back_button() { return 0x2000u; }

namespace {

int g_fail = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                  \
        }                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;

// From `ms` on, these movement keys are held, and Dodge / Sprint or Walk.
struct Keys {
    int ms;
    bool f, b, l, r, roll, walk;
};
constexpr Keys kRest{0, false, false, false, false, false, false};
Keys keys(int ms, const char* which, bool roll, bool walk) {
    Keys k{ms, false, false, false, false, roll, walk};
    for (const char* c = which; *c; ++c) {
        if (*c == 'W') k.f = true;
        if (*c == 'S') k.b = true;
        if (*c == 'A') k.l = true;
        if (*c == 'D') k.r = true;
    }
    return k;
}
Keys sprinting(int ms, const char* which) { return keys(ms, which, true, false); }
Keys running(int ms, const char* which) { return keys(ms, which, false, false); }
Keys walking(int ms, const char* which) { return keys(ms, which, false, true); }

struct Stick {
    int lx, ly;
};

float magnitude(Stick s) {
    const float x = static_cast<float>(s.lx - kStickRest) / 127.0f, y = static_cast<float>(s.ly - kStickRest) / 127.0f;
    return std::sqrt(x * x + y * y);
}
float angle(Stick s) {
    if (s.lx == kStickRest && s.ly == kStickRest) return 0.0f;
    return std::atan2(static_cast<float>(s.lx - kStickRest), static_cast<float>(kStickRest - s.ly)) * 57.29578f;
}
bool at_rest(Stick s) { return s.lx == kStickRest && s.ly == kStickRest; }
// A full push: one axis all the way out (the square's edge).
bool full_push(Stick s) { return std::max(std::abs(s.lx - kStickRest), std::abs(s.ly - kStickRest)) >= 126; }
float turn_between(Stick a, Stick b) {
    float d = std::fabs(angle(a) - angle(b));
    return d > 180.0f ? 360.0f - d : d;
}

const Keys& keys_at(const std::vector<Keys>& script, int ms) {
    const Keys* k = &script.front();
    for (const Keys& s : script)
        if (s.ms <= ms) k = &s;
    return *k;
}

// The stick each model gives for the keys held, x and y each -1, 0 or 1.
Stick full(int x, int y, float push) {
    const int reach = static_cast<int>(std::lround(127.0f * push));
    return Stick{kStickRest + x * reach, kStickRest + y * reach};
}
struct Ramp22 {  // v0.2.22's move_ramp
    float ramp = 0.0f;
    int at = 0;
    float step(bool moving, int ms) {
        const float d = static_cast<float>(ms - at);
        at = ms;
        ramp = moving ? std::fmin(1.0f, ramp + std::fmin(std::fmax(d, 0.0f), 66.0f) / 66.0f) : 0.0f;
        return ramp;
    }
};

struct Run {
    std::vector<double> t;  // ms of each read
    std::vector<Stick> now, v21, v22;
};

// The pump every 4 ms from 0 to end_ms, the game's reads every 1000/fps ms
// (the first half a frame in), each taking what the last pump left.
Run run(const std::vector<Keys>& script, double fps, int end_ms, int device = 0) {
    Run out;
    PadState pad;
    Ramp22 r22;
    Stick s21{kStickRest, kStickRest}, s22{kStickRest, kStickRest};
    // Each run on a clock of its own, an hour after the last, and from half a
    // second of standing still: what the bindings keep from one run does not
    // reach the next.
    static const Clock::time_point base = Clock::now();
    static int runs = 0;
    const Clock::time_point t0 = base + std::chrono::hours(++runs);
    double next_read = 500.0 / fps;
    for (int ms = -500; ms <= end_ms; ms += 4) {
        const Keys& k = ms < 0 ? kRest : keys_at(script, ms);
        bool held[kBindCount] = {};
        held[kBindMoveF] = k.f;
        held[kBindMoveB] = k.b;
        held[kBindMoveL] = k.l;
        held[kBindMoveR] = k.r;
        held[kBindWalk] = k.walk;
        held[kBindRoll] = k.roll;
        // Opposite keys cancel, as host_bindings_keys_held does.
        if (held[kBindMoveF] && held[kBindMoveB]) held[kBindMoveF] = held[kBindMoveB] = false;
        if (held[kBindMoveL] && held[kBindMoveR]) held[kBindMoveL] = held[kBindMoveR] = false;
        pad = PadState{};
        host_bindings_apply(held, false, pad, device, t0 + std::chrono::milliseconds(ms));
        const int x = (held[kBindMoveR] ? 1 : 0) - (held[kBindMoveL] ? 1 : 0);
        const int y = (held[kBindMoveB] ? 1 : 0) - (held[kBindMoveF] ? 1 : 0);
        const bool moving = x != 0 || y != 0;
        const float walk = k.walk ? 0.6f / (x != 0 && y != 0 ? std::sqrt(2.0f) : 1.0f) : 1.0f;
        const float ramp = r22.step(moving, ms);
        s21 = moving ? full(x, y, walk) : Stick{kStickRest, kStickRest};
        s22 = moving ? full(x, y, walk * ramp) : Stick{kStickRest, kStickRest};
        while (next_read <= ms + 4 && next_read <= end_ms) {
            if (next_read >= ms) {
                out.t.push_back(next_read);
                out.now.push_back(Stick{pad.lx, pad.ly});
                out.v21.push_back(s21);
                out.v22.push_back(s22);
            }
            next_read += 1000.0 / fps;
        }
    }
    return out;
}

// The reads from `from_ms` to `to_ms`, as a table.
void show(const char* what, double fps, const Run& r, int from_ms, int to_ms) {
    std::printf("\n%s at %.0f fps (read: ms, then magnitude and angle):\n", what, fps);
    std::printf("  %7s | %13s | %13s | %13s\n", "ms", "v0.2.21", "v0.2.22-23", "now");
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        if (r.t[i] < from_ms || r.t[i] > to_ms) continue;
        std::printf("  %7.1f | %5.2f %6.1f  | %5.2f %6.1f  | %5.2f %6.1f\n", r.t[i], magnitude(r.v21[i]), angle(r.v21[i]),
                    magnitude(r.v22[i]), angle(r.v22[i]), magnitude(r.now[i]), angle(r.now[i]));
    }
}

// The sprint turns among the reads from `from_ms` on: two full pushes, with
// nothing but rest between them, 90 degrees or more apart.
int sprint_turns(const std::vector<Stick>& s, const Run& r, int from_ms) {
    int n = 0;
    int last = -1;  // the last full push
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        if (r.t[i] < from_ms) {
            if (full_push(s[i])) last = static_cast<int>(i);
            continue;
        }
        if (at_rest(s[i])) continue;
        if (!full_push(s[i])) {
            last = -1;  // through the walk: the sprint ends there, no turn
            continue;
        }
        if (last >= 0 && turn_between(s[last], s[i]) >= 89.5f) ++n;
        last = static_cast<int>(i);
    }
    return n;
}

// The largest turn between two consecutive full pushes from `from_ms` on.
float largest_step(const std::vector<Stick>& s, const Run& r, int from_ms) {
    float m = 0.0f;
    for (std::size_t i = 1; i < r.t.size(); ++i) {
        if (r.t[i] < from_ms || !full_push(s[i]) || !full_push(s[i - 1])) continue;
        m = std::fmax(m, turn_between(s[i - 1], s[i]));
    }
    return m;
}

// Reads from `from_ms` on whose push is between rest and a full push - the
// stick in the walk.
int partial(const std::vector<Stick>& s, const Run& r, int from_ms) {
    int n = 0;
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        if (r.t[i] < from_ms || at_rest(s[i])) continue;
        if (!full_push(s[i])) ++n;
    }
    return n;
}

// The two give the same stick at every read from `from_ms` on.
bool same(const std::vector<Stick>& a, const std::vector<Stick>& b, const Run& r, int from_ms) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (r.t[i] >= from_ms && (a[i].lx != b[i].lx || a[i].ly != b[i].ly)) return false;
    return true;
}

// The first read at or after `from_ms` whose angle is `deg` (within a degree).
double reaches(const std::vector<Stick>& s, const Run& r, int from_ms, float deg) {
    for (std::size_t i = 0; i < r.t.size(); ++i)
        if (r.t[i] >= from_ms && full_push(s[i]) && std::fabs(angle(s[i]) - deg) < 1.0f) return r.t[i];
    return 1e9;
}

// The reads from `from_ms` to `to_ms` that are a full push within `within`
// degrees of `deg`.
int near(const std::vector<Stick>& s, const Run& r, int from_ms, int to_ms, float deg, float within) {
    int n = 0;
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        if (r.t[i] < from_ms || r.t[i] > to_ms || !full_push(s[i])) continue;
        float d = std::fabs(angle(s[i]) - deg);
        if (d > 180.0f) d = 360.0f - d;
        if (d <= within) ++n;
    }
    return n;
}

}  // namespace

int main() {
    const double rates[] = {30.0, 60.0, 120.0};
    // Sprinting forward for a second, then right: D added to W.
    const std::vector<Keys> add_d = {sprinting(0, "W"), sprinting(1000, "WD")};
    // (d) W let go and D pressed `gap` ms later, as a change of direction is typed.
    const auto switch_gap = [](int gap) {
        return std::vector<Keys>{sprinting(0, "W"), sprinting(1000, ""), sprinting(1000 + gap, "D")};
    };
    // D pressed, W let go 40 ms later.
    const std::vector<Keys> overlap = {sprinting(0, "W"), sprinting(1000, "WD"), sprinting(1040, "D")};
    // (a) Sprinting on A, then D, A, D, typed (16 ms between) or overlapping (both held 30 ms: they cancel).
    const std::vector<Keys> alt_typed = {sprinting(0, "A"),    sprinting(1000, ""), sprinting(1016, "D"),
                                         sprinting(1400, ""),  sprinting(1416, "A"), sprinting(1800, ""),
                                         sprinting(1816, "D")};
    const std::vector<Keys> alt_overlap = {sprinting(0, "A"),     sprinting(1000, "AD"), sprinting(1030, "D"),
                                           sprinting(1400, "AD"), sprinting(1430, "A"),  sprinting(1800, "AD"),
                                           sprinting(1830, "D")};
    // (b) Sprinting on W+A, then W+D, W+A, W+D: let go and pressed in the same moment, or typed (16 ms of W alone).
    const std::vector<Keys> wad_snap = {sprinting(0, "WA"), sprinting(1000, "WD"), sprinting(1400, "WA"),
                                        sprinting(1800, "WD")};
    const std::vector<Keys> wad_typed = {sprinting(0, "WA"),   sprinting(1000, "W"),  sprinting(1016, "WD"),
                                         sprinting(1400, "W"), sprinting(1416, "WA"), sprinting(1800, "W"),
                                         sprinting(1816, "WD")};
    // (c) Sprinting on W+D, A pressed (A and D cancel: W), then D let go (W+A): f9227ee5's case.
    const std::vector<Keys> wd_a = {sprinting(0, "WD"), sprinting(1000, "WAD"), sprinting(1300, "WA")};
    // Standing, then W.
    const std::vector<Keys> start = {running(0, ""), running(1000, "W")};

    for (const double fps : rates) {
        // The largest turn a read may see at a full push: 90 degrees in 70 ms.
        const float step_max = static_cast<float>(90.0 / 70.0 * (1000.0 / fps + 4.0)) + 1.0f;
        const double frame = 1000.0 / fps;
        {
            const Run a = run(add_d, fps, 1300);
            show("Sprinting on W, then D added at 1000 ms", fps, a, 950, 1150);
            CHECK(sprint_turns(a.now, a, 500) == 0);
            CHECK(partial(a.now, a, 500) == 0);
            CHECK(largest_step(a.now, a, 500) <= step_max);
            CHECK(reaches(a.now, a, 1000, 45.0f) <= 1000 + 35 + 4 + frame);
        }
        for (const int gap : {8, 16, 33, 50}) {
            const Run b = run(switch_gap(gap), fps, 1400);
            char what[96];
            std::snprintf(what, sizeof(what), "(d) Sprinting on W, let go at 1000 ms, D pressed %d ms later", gap);
            show(what, fps, b, 950, 1150 + gap);
            // v0.2.22's ramp went through the walk after the gap; v0.2.21's push swung 90 degrees at once.
            CHECK(partial(b.v22, b, 1000 + gap) >= 1);
            if (gap >= frame) CHECK(sprint_turns(b.v21, b, 500) == 1);
            // Now: no walk, no sprint turn - round the corner at a full push.
            CHECK(partial(b.now, b, 500) == 0);
            CHECK(sprint_turns(b.now, b, 500) == 0);
            CHECK(largest_step(b.now, b, 500) <= step_max);
            CHECK(reaches(b.now, b, 1000 + gap, 90.0f) <= 1000 + gap + 70 + 4 + frame);
        }
        {
            const Run c = run(overlap, fps, 1300);
            show("Sprinting on W, D pressed at 1000 ms, W let go at 1040 ms", fps, c, 950, 1150);
            CHECK(sprint_turns(c.now, c, 500) == 0);
            CHECK(partial(c.now, c, 500) == 0);
            CHECK(largest_step(c.now, c, 500) <= step_max);
        }
        // (a) A and D in turn: a straight reversal flips at once, as v0.2.21's
        // push did - the quick turn.
        for (const auto* alt : {&alt_typed, &alt_overlap}) {
            const Run a = run(*alt, fps, 2100);
            show(alt == &alt_typed ? "(a) Sprinting on A, then D, A, D typed (16 ms gaps)"
                                   : "(a) Sprinting on A, then D, A, D overlapping (30 ms both held)",
                 fps, a, 950, 1250);
            CHECK(sprint_turns(a.v21, a, 500) >= 1);
            CHECK(same(a.now, a.v21, a, 900));
            CHECK(sprint_turns(a.now, a, 500) >= 1);
        }
        // W and S in turn, the same.
        {
            const Run w = run({sprinting(0, "W"), sprinting(1000, ""), sprinting(1016, "S"), sprinting(1400, ""),
                               sprinting(1416, "W")},
                              fps, 1700);
            show("Sprinting on W, then S, W typed (16 ms gaps)", fps, w, 950, 1100);
            CHECK(same(w.now, w.v21, w, 900));
        }
        // A reversal that comes while the stick is still turning flips too:
        // W, D, and 30 ms later A in D's place.
        {
            const Run m = run({sprinting(0, "W"), sprinting(1000, "D"), sprinting(1030, "A")}, fps, 1300);
            show("Sprinting on W, D at 1000 ms, A in its place at 1030 ms", fps, m, 950, 1100);
            CHECK(reaches(m.now, m, 1030, -90.0f) <= 1030 + 4 + frame);
            CHECK(same(m.now, m.v21, m, static_cast<int>(1034 + frame)));
        }
        // Both axes reversed - W+A to S+D, swapped or typed - is no straight
        // reversal: right round by the front (through W+D), at a full push.
        for (const bool typed : {false, true}) {
            const Run d = run(typed ? std::vector<Keys>{sprinting(0, "WA"), sprinting(1000, ""), sprinting(1016, "SD")}
                                    : std::vector<Keys>{sprinting(0, "WA"), sprinting(1000, "SD")},
                              fps, 1400);
            const int at = typed ? 1016 : 1000;
            show(typed ? "Sprinting on W+A, then S+D typed" : "Sprinting on W+A, then S+D at once", fps, d, 950, 1200);
            CHECK(sprint_turns(d.now, d, 500) == 0);
            CHECK(partial(d.now, d, 500) == 0);
            CHECK(largest_step(d.now, d, 500) <= step_max);
            CHECK(near(d.now, d, at, at + 140, 45.0f, step_max) >= 1);
            CHECK(reaches(d.now, d, at, 135.0f) <= at + 140 + 4 + frame);
        }
        // (b) W held, A and D alternating: round at a full push.
        for (const auto* wad : {&wad_snap, &wad_typed}) {
            const Run b = run(*wad, fps, 2100);
            show(wad == &wad_snap ? "(b) Sprinting on W+A, W+D, W+A, W+D (keys swapped at once)"
                                  : "(b) Sprinting on W+A, W, W+D, W, W+A ... (typed)",
                 fps, b, 950, 1150);
            if (wad == &wad_snap) {
                CHECK(sprint_turns(b.v21, b, 500) >= 1);  // forward-left to forward-right in one read
                CHECK(sprint_turns(b.v22, b, 500) >= 1);
            }
            CHECK(sprint_turns(b.now, b, 500) == 0);
            CHECK(partial(b.now, b, 500) == 0);
            CHECK(largest_step(b.now, b, 500) <= step_max);
        }
        // (c) W+D meeting A: two turns of 45 degrees, at a full push.
        {
            const Run c = run(wd_a, fps, 1600);
            show("(c) Sprinting on W+D, A pressed at 1000 ms, D let go at 1300 ms", fps, c, 950, 1400);
            CHECK(sprint_turns(c.now, c, 500) == 0);
            CHECK(partial(c.now, c, 500) == 0);
            CHECK(largest_step(c.now, c, 500) <= step_max);
        }
        // Not sprinting the keys turn the stick at once, as v0.2.21's push did -
        // and the push is kept over a typed gap.
        {
            const Run r = run({running(0, "A"), running(1000, ""), running(1016, "D"), running(1400, "WD")}, fps, 1700);
            show("Running on A, D pressed 16 ms after A let go, then W+D", fps, r, 950, 1100);
            CHECK(same(r.now, r.v21, r, 900));
        }
        // Dodge / Sprint let go part of the way round: the stick goes where the
        // keys point at once, for the dodge the game makes on that release.
        {
            const Run r =
                run({sprinting(0, "W"), sprinting(1000, ""), sprinting(1016, "D"), running(1046, "D")}, fps, 1300);
            show("Sprinting on W, D pressed at 1016 ms, Dodge / Sprint let go at 1046 ms", fps, r, 1000, 1150);
            CHECK(near(r.now, r, 1016, 1045, 45.0f, 44.0f) >= 1);  // still turning
            CHECK(reaches(r.now, r, 1046, 90.0f) <= 1046 + 4 + frame);
            CHECK(same(r.now, r.v21, r, 1050));
        }
        // From standing the ramp is kept: the first reads after the press are a
        // partial push, and the push is full within 66 ms and a frame.
        for (const bool roll : {false, true}) {
            const Run s = run(roll ? std::vector<Keys>{sprinting(0, ""), sprinting(1000, "W")} : start, fps, 1300);
            show(roll ? "Standing with Dodge / Sprint held, then W at 1000 ms" : "Standing, then W at 1000 ms", fps, s,
                 950, 1150);
            CHECK(same(s.now, s.v22, s, 0));
            int first = -1, full_at = -1;
            for (std::size_t i = 0; i < s.t.size(); ++i) {
                if (s.t[i] < 1000) continue;
                if (first < 0 && s.now[i].ly != kStickRest) first = static_cast<int>(i);
                if (full_at < 0 && s.now[i].ly == kStickLow) full_at = static_cast<int>(i);
            }
            CHECK(first >= 0 && s.now[first].ly > kStickLow);
            CHECK(full_at >= 0 && s.t[full_at] <= 1000 + 66 + 4 + frame);
        }
    }
    // A stop long enough to be one: let go for half a second, then D - a start
    // from standing in its own direction, so it ramps there.
    for (const bool roll : {false, true}) {
        const Run r =
            run({keys(0, "W", roll, false), keys(1000, "", roll, false), keys(1500, "D", roll, false)}, 60.0, 1700);
        show(roll ? "Sprinting on W, let go at 1000 ms, D at 1500 ms" : "Running on W, let go at 1000 ms, D at 1500 ms",
             60.0, r, 1450, 1600);
        CHECK(partial(r.now, r, 1500) >= 1);
        CHECK(same(r.now, r.v22, r, 1450));
    }
    // Walk held: the gentle push, its diagonal on the same reach, unchanged.
    {
        const Run r = run({walking(0, "W"), walking(1000, "WD")}, 60.0, 1300);
        CHECK(r.now.back().lx == kStickRest + 54 && r.now.back().ly == kStickRest - 54);
        const Run w = run({walking(0, "W")}, 60.0, 400);
        CHECK(w.now.back().lx == kStickRest && w.now.back().ly == kStickRest - 76);
    }
    // A full diagonal is the square's corner, as v0.2.21's.
    {
        const Run r = run({sprinting(0, "WD")}, 60.0, 400);
        CHECK(r.now.back().lx == kStickHigh && r.now.back().ly == kStickLow);
    }
    // The mouse's buttons (device 1) push in full and turn at once: no ramp, ever.
    {
        const Run r = run(start, 60.0, 1300, 1);
        CHECK(same(r.now, r.v21, r, 0));
        const Run t = run(wad_snap, 60.0, 2100, 1);
        CHECK(same(t.now, t.v21, t, 0));
    }
    if (g_fail) {
        std::printf("\nkey_move_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("\nkey_move_test: ok\n");
    return 0;
}
