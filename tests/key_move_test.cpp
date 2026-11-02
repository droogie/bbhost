// The keyboard's movement keys on the left stick (host/bindings.cpp), frame by
// frame as the game reads them: the window's pump applies the keys held every
// 4 ms (host/window.cpp update_pad) and the game reads the last pad once a frame
// at 30, 60 or 120 fps (hle/system.cpp hle_pad_read). Each read is shown as the
// stick's magnitude (1 a full push along an axis, 1.41 the square's corner) and
// angle (0 forward, 90 right), against two models: v0.2.21's push (a full push
// the moment a key is held) and v0.2.22's (from rest over 66 ms, back to rest
// whenever no movement key is held - between two keys of a typed change of
// direction too).
//
// What it holds the code to: a change of direction - D added to W, or W let go
// and D pressed a moment later, or the two overlapping - pushes in full at once,
// as v0.2.21 did (v0.2.22's ramp ran from rest again after the gap, the stick
// passed through the walk for several frames and the sprint dropped); a start
// from standing still ramps up as v0.2.22's does; Walk and the mouse's buttons
// push as before.
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

// From `ms` on, these movement keys are held (and Walk).
struct Keys {
    int ms;
    bool f, b, l, r, walk;
};

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
    const Keys rest{0, false, false, false, false, false};
    double next_read = 500.0 / fps;
    for (int ms = -500; ms <= end_ms; ms += 4) {
        const Keys& k = ms < 0 ? rest : keys_at(script, ms);
        bool held[kBindCount] = {};
        held[kBindMoveF] = k.f;
        held[kBindMoveB] = k.b;
        held[kBindMoveL] = k.l;
        held[kBindMoveR] = k.r;
        held[kBindWalk] = k.walk;
        pad = PadState{};
        host_bindings_apply(held, false, pad, device, t0 + std::chrono::milliseconds(ms));
        const int x = (k.r ? 1 : 0) - (k.l ? 1 : 0), y = (k.b ? 1 : 0) - (k.f ? 1 : 0);
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
        std::printf("  %7.1f | %5.2f %6.1f  | %5.2f %6.1f  | %5.2f %6.1f%s\n", r.t[i], magnitude(r.v21[i]), angle(r.v21[i]),
                    magnitude(r.v22[i]), angle(r.v22[i]), magnitude(r.now[i]), angle(r.now[i]),
                    r.now[i].lx != r.v21[i].lx || r.now[i].ly != r.v21[i].ly ? "   <- not v0.2.21's" : "");
    }
}

// Reads at or after `from_ms` whose push is between rest and a full push along
// its axes - the stick passing through the walk.
int partial(const std::vector<Stick>& s, const Run& r, int from_ms) {
    int n = 0;
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        if (r.t[i] < from_ms) continue;
        const int dx = std::abs(s[i].lx - kStickRest), dy = std::abs(s[i].ly - kStickRest);
        const int far = dx > dy ? dx : dy;
        if (far > 0 && far < 127) ++n;
    }
    return n;
}

// The two give the same stick at every read from `from_ms` on (each script's
// first key is a start from standing at 0 ms, which ramps; what is compared is
// what follows it).
bool same(const std::vector<Stick>& a, const std::vector<Stick>& b, const Run& r, int from_ms = 500) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (r.t[i] >= from_ms && (a[i].lx != b[i].lx || a[i].ly != b[i].ly)) return false;
    return true;
}

}  // namespace

int main() {
    const double rates[] = {30.0, 60.0, 120.0};
    // Sprinting forward for a second, then right: D added to W.
    const std::vector<Keys> add_d = {{0, true, false, false, false, false}, {1000, true, false, false, true, false}};
    // W let go and D pressed `gap` ms later, as a change of direction is typed.
    const auto switch_gap = [](int gap) {
        return std::vector<Keys>{{0, true, false, false, false, false},
                                 {1000, false, false, false, false, false},
                                 {1000 + gap, false, false, false, true, false}};
    };
    // D pressed, W let go 40 ms later.
    const std::vector<Keys> overlap = {{0, true, false, false, false, false},
                                       {1000, true, false, false, true, false},
                                       {1040, false, false, false, true, false}};
    // Standing still, then W.
    const std::vector<Keys> start = {{0, false, false, false, false, false}, {1000, true, false, false, false, false}};

    for (const double fps : rates) {
        // A warm start: one run first, so the bindings' own state is a
        // player's who has been standing (each run starts at rest anyway).
        const Run a = run(add_d, fps, 1300);
        show("W, then D added at 1000 ms", fps, a, 950, 1150);
        CHECK(same(a.now, a.v21, a));
        for (const int gap : {8, 16, 33, 50}) {
            const Run b = run(switch_gap(gap), fps, 1400);
            char what[96];
            std::snprintf(what, sizeof(what), "W, let go at 1000 ms, D pressed %d ms later", gap);
            show(what, fps, b, 950, 1150 + gap);
            // The regression: v0.2.22's ramp passes through the walk after the gap.
            CHECK(partial(b.v22, b, 1000 + gap) >= 1);
            // Now: as v0.2.21 - at rest through the gap, then the full push.
            CHECK(same(b.now, b.v21, b));
            CHECK(partial(b.now, b, 1000) == 0);
        }
        const Run c = run(overlap, fps, 1300);
        show("W, D pressed at 1000 ms, W let go at 1040 ms", fps, c, 950, 1150);
        CHECK(same(c.now, c.v21, c));
        // From standing still the ramp is kept: the first reads after the press
        // are a partial push, and the push is full within 66 ms and a frame.
        const Run s = run(start, fps, 1300);
        show("standing, then W at 1000 ms", fps, s, 950, 1150);
        CHECK(same(s.now, s.v22, s, 0));
        int first = -1, full_at = -1;
        for (std::size_t i = 0; i < s.t.size(); ++i) {
            if (s.t[i] < 1000) continue;
            if (first < 0 && s.now[i].ly != kStickRest) first = static_cast<int>(i);
            if (full_at < 0 && s.now[i].ly == kStickLow) full_at = static_cast<int>(i);
        }
        CHECK(first >= 0 && s.now[first].ly > kStickLow);  // the first read after the press: not yet full
        CHECK(full_at >= 0 && s.t[full_at] <= 1000 + 66 + 4 + 1000.0 / fps);
    }
    // A stop long enough to be one: let go for half a second, then W again - a
    // start from rest, so it ramps.
    {
        const std::vector<Keys> stop = {{0, true, false, false, false, false},
                                        {1000, false, false, false, false, false},
                                        {1500, true, false, false, false, false}};
        const Run r = run(stop, 60.0, 1700);
        show("W, let go at 1000 ms, W again at 1500 ms", 60.0, r, 1450, 1600);
        CHECK(partial(r.now, r, 1500) >= 1);
    }
    // Walk held: the gentle push, its diagonal on the same reach, unchanged.
    {
        const std::vector<Keys> walk = {{0, true, false, false, false, true}, {1000, true, false, false, true, true}};
        const Run r = run(walk, 60.0, 1300);
        CHECK(r.now.back().lx == kStickRest + 54 && r.now.back().ly == kStickRest - 54);
        const Run w = run({{0, true, false, false, false, true}}, 60.0, 400);
        CHECK(w.now.back().lx == kStickRest && w.now.back().ly == kStickRest - 76);
    }
    // The mouse's buttons (device 1) push in full at once: no ramp, ever.
    {
        const Run r = run(start, 60.0, 1300, 1);
        CHECK(same(r.now, r.v21, r, 0));
    }
    if (g_fail) {
        std::printf("\nkey_move_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("\nkey_move_test: ok\n");
    return 0;
}
