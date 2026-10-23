// The water interaction's gate (decomp/gx/water_gate.cpp), on simulated
// frames - no eboot needed:
//   - the step rule (water_gate_step) over 20 game seconds at 30, 60, 90,
//     120 and 144 fps, at uncapped's uneven frames and at 60 with long
//     frames: 30 steps a game second (every frame at 30), a static camera or
//     none; the game time standing: no step; going back or leaping (a load):
//     a step at once and the cadence after it, no burst;
//   - screen-space mode with the camera moving through negative and positive
//     coordinates: every frame on which trunc(x) or trunc(z) left the last
//     step's steps (the grid never leaves what it holds), and the steps still
//     30 a game second at a running speed; at a speed that crosses every frame,
//     every frame steps;
//   - the update and the task (water_update, water_task) over a stand-in
//     manager whose update and task do to the grids' height maps and the water
//     surfaces' impulses what the game's do (sub_2677780's turn of the maps,
//     its handing on of the impulses and a resize's swap; sub_2677e50's resize
//     copy, splats and step), at 30, 60, 90, 120 and 144 fps and uneven frames,
//     in screen-space mode (the camera still and running) and with a grid a
//     water surface:
//       - 30 steps a game second, every frame at 30;
//       - on the frame-rate code's schedule of whole 1/30 s (engine/sfx_schedule.h,
//         the effects' own), a step on every whole frame and on no other - one
//         clock with the effects - and a camera step taking the schedule's next
//         whole step; on the game time where nothing is scheduled;
//       - every step reads the maps the two steps before it wrote, in order,
//         and the frames between show the last step's (the surfaces' +0x10);
//       - every step takes the latest frame's impulses (a held frame's are
//         given up for a newer frame's: a wake's effect queues one a frame),
//         and impulses queued on the frames after steps only still reach the
//         next step - after a camera's step took its 1/30 s ahead the one
//         after; none twice;
//       - a resize swapped in on a frame between steps read as the game's
//         update at 30 hands it on;
//       - the manager off for a while (its update does nothing): nothing taken
//         twice; a surface whose two vectors come from two heaps keeps its
//         impulses and its own vectors;
//       - every frame the game's with BBHOST_WATER_GATE=0, with no GX world,
//         and while the task is not ours (its update then holds no frame);
// and each of the source's mutants caught.
#define DECOMP_WATER_GATE_MUTANTS
#define DECOMP_WATER_GATE_TEST_GX
int g_water_gate_mutant = 0;
#include "decomp/gx/water_gate.cpp"
#include "engine/sfx_schedule.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

namespace {

bool g_quiet = false;
int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        if (!g_quiet) std::printf("FAIL: %s\n", what);
    }
}

// --- the rule ------------------------------------------------------------------

// A run of frames: each frame's game step, the camera's x and z at it. Counts
// the steps, the steps a game second over [1 s, end] and whether every frame
// whose truncated camera left the last step's stepped.
struct Run {
    int steps = 0, frames = 0, camera_steps = 0;
    double game_s = 0.0;
    bool aligned = true;
};

template <class StepFn, class CameraFn>
Run simulate(int frames, StepFn dt_of, bool scroll, CameraFn camera, float t0 = 3000.0f) {
    WaterGate g;
    Run r;
    float t = t0;
    float last_cx = 0, last_cz = 0;
    bool have = false;
    for (int i = 0; i < frames; ++i) {
        float x = 0, z = 0;
        camera(static_cast<double>(t) - static_cast<double>(t0), x, z);
        const WaterStep why = water_gate_step(g, t, scroll, x, z);
        if (scroll && have && (std::trunc(x) != last_cx || std::trunc(z) != last_cz) && why == kWaterSkip) r.aligned = false;  // the game's rounding
        if (why != kWaterSkip) {
            ++r.steps;
            if (why == kWaterCamera) ++r.camera_steps;
            last_cx = std::trunc(x);
            last_cz = std::trunc(z);
            have = true;
        }
        ++r.frames;
        t += static_cast<float>(dt_of(i));
    }
    // the game time as the gate reads it: GX's clock is a float sum, which a
    // frame's step moves by its rounding (2.4e-4 s at 3000 s) - the shaders'
    // time drifts with it, and the gate keeps to it
    r.game_s = static_cast<double>(t) - static_cast<double>(t0);
    return r;
}

void still(double, float& x, float& z) { x = 12.3f; z = -45.6f; }

std::uint32_t g_rng = 0x2677e50u;
double uniform(double lo, double hi) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((g_rng >> 8) / double(1u << 24));
}

void rates() {
    struct Rate {
        const char* name;
        double fps;
    } const kRates[] = {{"30", 30}, {"60", 60}, {"90", 90}, {"120", 120}, {"144", 144}};
    for (const Rate& rt : kRates) {
        for (int scroll = 0; scroll < 2; ++scroll) {
            const int frames = static_cast<int>(20 * rt.fps);
            const Run r = simulate(frames, [&](int) { return 1.0 / rt.fps; }, scroll != 0, still);
            const double per_s = r.steps / r.game_s;
            if (!g_quiet)
                std::printf("  %s fps%s: %d steps over %d frames, %.3f game s: %.2f a game second\n", rt.name, scroll ? ", screen-space" : "",
                            r.steps, r.frames, r.game_s, per_s);
            if (rt.fps == 30)
                check(r.steps == r.frames, "at 30 every frame steps");  // the game's own, whatever its clock's rounding
            else
                check(std::fabs(r.steps - 30.0 * r.game_s) <= 1.5, "30 steps a game second at a steady rate");
        }
    }
    // uncapped's uneven frames (the game's step stays within [1/240, 1/30])
    const Run u = simulate(5000, [](int) { return uniform(1.0 / 240, 1.0 / 30); }, true, still);
    if (!g_quiet) std::printf("  uneven: %d steps over %d frames, %.3f game s: %.2f a game second\n", u.steps, u.frames, u.game_s, u.steps / u.game_s);
    check(std::fabs(u.steps / u.game_s - 30.0) < 0.2, "30 steps a game second at uneven frames");
    // 60 with every tenth frame a long one (1/30)
    const Run l = simulate(1200, [](int i) { return i % 10 == 9 ? 1.0 / 30 : 1.0 / 60; }, false, still);
    check(std::fabs(l.steps / l.game_s - 30.0) < 0.3, "30 steps a game second with long frames");
}

void standing_and_loads() {
    // the game time standing: no step after the first
    WaterGate g;
    int steps = 0;
    for (int i = 0; i < 100; ++i) steps += water_gate_step(g, 50.0f, false, 0, 0) != kWaterSkip;
    check(steps == 1, "no step while the game time stands");
    // back (a load resets the GX world's clock): a step at once, then the cadence
    WaterGate b;
    float t = 300.0f;
    for (int i = 0; i < 120; ++i, t += 1.0f / 60) water_gate_step(b, t, false, 0, 0);
    check(water_gate_step(b, 2.0f, false, 0, 0) == kWaterStart, "a step at once when the game time goes back");
    int after = 0;
    for (int i = 1; i <= 60; ++i) after += water_gate_step(b, 2.0f + i / 60.0f, false, 0, 0) != kWaterSkip;
    check(after == 30, "30 steps in the game second after it, no burst");
    // a leap (a long load): a step at once, no burst
    WaterGate j;
    t = 10.0f;
    for (int i = 0; i < 60; ++i, t += 1.0f / 60) water_gate_step(j, t, false, 0, 0);
    check(water_gate_step(j, t + 5.0f, false, 0, 0) == kWaterStart, "a step at once after a leap");
    int burst = 0;
    for (int i = 1; i <= 4; ++i) burst += water_gate_step(j, t + 5.0f + i / 120.0f, false, 0, 0) != kWaterSkip;
    check(burst <= 1, "no burst after a leap");
}

void camera() {
    // running through x = 0 and z = 0 at 6 m/s and 2 m/s, at 120: aligned, and 30 a game second
    const Run r = simulate(2400, [](int) { return 1.0 / 120; }, true, [](double s, float& x, float& z) {
        x = static_cast<float>(-60.0 + 6.0 * s);
        z = static_cast<float>(-20.0 + 2.0 * s);
    });
    if (!g_quiet)
        std::printf("  running at 120, screen-space: %d steps (%d as the camera crossed) over %.3f game s: %.2f a game second, %s\n", r.steps,
                    r.camera_steps, r.game_s, r.steps / r.game_s, r.aligned ? "aligned" : "NOT aligned");
    check(r.aligned, "every frame whose truncated camera moved stepped");
    check(r.camera_steps > 0, "the camera's crossings made steps of their own");
    check(std::fabs(r.steps / r.game_s - 30.0) < 0.4, "30 steps a game second while running");
    // at 60 too
    const Run s = simulate(1200, [](int) { return 1.0 / 60; }, true, [](double t, float& x, float& z) {
        x = static_cast<float>(0.5 - 6.0 * t);
        z = 7.25f;
    });
    check(s.aligned, "aligned at 60");
    check(std::fabs(s.steps / s.game_s - 30.0) < 0.4, "30 steps a game second while running at 60");
    // so fast every frame crosses (1.25 m a frame, away from trunc's double-width unit at 0): every frame steps
    const Run f = simulate(600, [](int) { return 1.0 / 120; }, true, [](double t, float& x, float& z) {
        x = static_cast<float>(10.0 + 150.0 * t);
        z = 0.5f;
    });
    check(f.aligned, "aligned when every frame crosses");
    check(f.steps == f.frames, "every frame steps when every frame crosses");
    // not screen-space: the camera does not matter
    const Run n = simulate(1200, [](int) { return 1.0 / 60; }, false, [](double t, float& x, float& z) {
        x = static_cast<float>(-60.0 + 6.0 * t);
        z = 0.0f;
    });
    check(n.camera_steps == 0 && std::fabs(n.steps - 30.0 * n.game_s) <= 1.5, "the camera makes no step outside screen-space mode");
}

// --- a stand-in manager ----------------------------------------------------------
//
// Its update and task do to what ours reads and writes what the game's do:
// game_update is sub_2677780's turn of each grid's maps (sub_12bbda0), a
// resize's swap (sub_12bd860) and the hand-on of each surface's impulses;
// game_task is sub_2677e50's resize copy (sub_12bd410), splats (each impulse
// in +0x98 taken) and step (into +0x10, from +0xd0 and +0xd8). A map is an
// id; what a step wrote into it is that grid's step count, so a step's read
// of +0xd0 and +0xd8 says which steps' maps it took.

struct alignas(16) Obj {
    u8 b[0x140];
};
u64 at(Obj& o) { return reinterpret_cast<u64>(o.b); }

u8 g_gx[0x8000];  // GX's world: the game time at +0x78cc
bool g_have_gx = true;
u8 g_context[0x500];  // the draw context: +0x3f8 the camera
u8 g_camera[0x40];    // +0x30 x, +0x38 z
u64 g_param[2];       // arg3: [0] the manager, [1] the camera the scroll is taken from

constexpr int kSurfaces = 3, kRoom = 512;
struct Impulse {
    u64 id;
    u64 pad[3];
};

struct World {
    Obj manager, none_holder, screen_holder, holders[kSurfaces], grids[kSurfaces], surfaces[kSurfaces], vectors[2 * kSurfaces], heaps[2];
    u64 holder_list[kSurfaces], surface_list[kSurfaces];
    Impulse room[2 * kSurfaces][kRoom];
    bool screen = true;
    std::map<u64, long> content;  // map -> the step that wrote it (absent: none)
    long written[kSurfaces] = {};  // each grid's steps
    u64 next_map = 0x1000;
    // what each grid's steps read, (+0xd0, +0xd8)
    std::vector<std::pair<long, long>> reads[kSurfaces];
    long resize_at = -1;       // the step after which grid 0 is resized
    bool swapped_in = false;   // a resize swapped in since grid 0's last step
    int updates = 0, tasks = 0, steps = 0;
    // impulses
    u64 next_impulse = 1;
    std::map<u64, double> added;   // id -> game time
    std::map<u64, int> taken;      // id -> times taken
    double worst_delay = 0.0;
    double now = 0.0;
    // each surface's latest frame's impulses (queue_impulse), the frame a step last took its impulses, and the steps
    // that took other than the latest frame's since the step before (checked while the manager stays on)
    std::vector<u64> batch[kSurfaces];
    int batch_frame[kSurfaces] = {-1, -1, -1};
    int last_take[kSurfaces] = {-1, -1, -1};
    int frame_no = 0;
    bool check_latest = true;
    int not_latest = 0;
    int rooms_swapped = 0;  // frames on which surface 2's two vectors (two heaps) no longer held their own rooms
};
World* W = nullptr;

int grid_index(u64 grid) {
    for (int i = 0; i < kSurfaces; ++i)
        if (at(W->grids[i]) == grid) return i;
    return -1;
}
long content_of(u64 map) {
    const auto it = W->content.find(map);
    return it == W->content.end() ? -1 : it->second;
}

void game_turn(u64 grid) {  // sub_12bbda0
    if (!load<u64>(grid + 0x10) || load<u8>(grid + 0x30) != 3) return;
    const u64 map = load<u64>(grid + 0x10), last = load<u64>(grid + 0xd0), before = load<u64>(grid + 0xd8);
    store<u64>(grid + 0xd8, last);
    store<u64>(grid + 0xd0, map);
    store<u64>(grid + 0x10, before);
}
void game_resize_swap(u64 grid) {  // sub_12bd860
    if (!load<u8>(grid + 0x100)) return;
    store<u8>(grid + 0x100, 0);
    store<u8>(grid + 0x101, 0);
    store<u64>(grid + 0x10, load<u64>(grid + 0xe0));
    store<u64>(grid + 0xd0, load<u64>(grid + 0xe8));
    store<u64>(grid + 0xd8, load<u64>(grid + 0xf0));
    store<u64>(grid + 0xe0, 0);
    store<u64>(grid + 0xe8, 0);
    store<u64>(grid + 0xf0, 0);
    if (grid_index(grid) == 0) W->swapped_in = true;
}
void game_resize_copy(u64 grid) {  // sub_12bd410
    if (!load<u8>(grid + 0x101) || load<u8>(grid + 0x100)) return;
    store<u8>(grid + 0x100, 1);
    const u64 slots[3] = {0x10, 0xd0, 0xd8}, into[3] = {0xe0, 0xe8, 0xf0};
    for (int k = 0; k < 3; ++k) {
        const long c = content_of(load<u64>(grid + slots[k]));
        if (c >= 0) W->content[load<u64>(grid + into[k])] = c;
    }
}

GUEST_ABI u64 game_update(u64 mgr) {
    ++W->updates;
    if (!load<u8>(mgr + 0x78) || !load<u64>(mgr + 0x68)) return 0;
    const u64 none = load<u64>(mgr + 0x68);
    for (u64 p = load<u64>(mgr + 0x18); p < load<u64>(mgr + 0x20); p += 8) {
        const u64 h = load<u64>(p);
        if (!load<u8>(h + 0x18) || h == none) continue;
        if (load<u64>(h)) game_turn(load<u64>(h));
    }
    if (const u64 sh = load<u64>(mgr + 0x60); sh && load<u64>(sh)) game_turn(load<u64>(sh));
    for (u64 p = load<u64>(mgr + 0x40); p < load<u64>(mgr + 0x48); p += 8) {
        const u64 s = load<u64>(p);
        const u64 h = load<u64>(s + 8);
        if (h != none && load<u64>(h)) game_resize_swap(load<u64>(h));
        const u64 taken = load<u64>(s + 0x98), queued = load<u64>(s + 0xa0);
        store<u64>(taken + 0x10, load<u64>(taken + 8));
        if (taken != queued) {
            if (load<u64>(taken + 0x20) == load<u64>(queued + 0x20)) {
                for (u64 f = 8; f <= 0x18; f += 8) {
                    const u64 a = load<u64>(taken + f);
                    store<u64>(taken + f, load<u64>(queued + f));
                    store<u64>(queued + f, a);
                }
            } else {  // two heaps: copied
                const u64 n = load<u64>(queued + 0x10) - load<u64>(queued + 8);
                std::memcpy(reinterpret_cast<void*>(load<u64>(taken + 8)), reinterpret_cast<void*>(load<u64>(queued + 8)), n);
                store<u64>(taken + 0x10, load<u64>(taken + 8) + n);
            }
        }
        store<u64>(queued + 0x10, load<u64>(queued + 8));
    }
    return 0;
}

void game_step(u64 grid) {  // sub_12bbe80, as its read and write go
    const int g = grid_index(grid);
    W->reads[g].push_back({content_of(load<u64>(grid + 0xd0)), content_of(load<u64>(grid + 0xd8))});
    W->content[load<u64>(grid + 0x10)] = ++W->written[g];
    if (g == 0) W->swapped_in = false;
    ++W->steps;
    if (g == 0 && W->written[0] == W->resize_at) {  // a resize asked for (sub_12bd2a0): new maps, copied at the next task
        store<u8>(grid + 0x101, 1);
        store<u64>(grid + 0xe0, W->next_map++);
        store<u64>(grid + 0xe8, W->next_map++);
        store<u64>(grid + 0xf0, W->next_map++);
    }
}

bool g_args_ok = true;
GUEST_ABI u64 game_task(u64 self, u64 context, u64 param) {
    ++W->tasks;
    g_args_ok = self == 0x1234 && context == reinterpret_cast<u64>(g_context) && param == reinterpret_cast<u64>(g_param);
    const u64 mgr = load<u64>(param);
    if (!load<u8>(mgr + 0x78) || !load<u64>(mgr + 0x68)) return 0x5eedu;
    const u64 none = load<u64>(mgr + 0x68);
    for (u64 p = load<u64>(mgr + 0x40); p < load<u64>(mgr + 0x48); p += 8) {
        const u64 s = load<u64>(p);
        const u64 h = load<u64>(s + 8);
        if (h == none) continue;
        const u64 grid = load<u64>(h);
        game_resize_copy(grid);
        const u64 taken = load<u64>(s + 0x98);
        std::vector<u64> ids;
        for (u64 q = load<u64>(taken + 8); q < load<u64>(taken + 0x10); q += sizeof(Impulse)) {
            const u64 id = load<u64>(q);
            ids.push_back(id);
            const int n = ++W->taken[id];
            if (n == 1) {
                const double d = W->now - W->added[id];
                if (d > W->worst_delay) W->worst_delay = d;
            }
        }
        const int si = static_cast<int>((p - load<u64>(mgr + 0x40)) / 8);
        if (W->check_latest && si >= 0 && si < kSurfaces) {
            // the latest frame's since this surface's last step, or none
            const std::vector<u64> none_new;
            const std::vector<u64>& expect = W->batch_frame[si] > W->last_take[si] ? W->batch[si] : none_new;
            if (ids != expect) ++W->not_latest;
            W->last_take[si] = W->frame_no;
        }
        if (!load<u8>(mgr + 0x79)) game_step(grid);
    }
    if (load<u8>(mgr + 0x79)) game_step(load<u64>(load<u64>(mgr + 0x60)));
    return 0x5eedu;
}


}  // namespace

namespace {
u64 gx_world() { return g_have_gx ? reinterpret_cast<u64>(g_gx) : 0; }
}  // namespace

namespace {

void set_time(float t) { std::memcpy(g_gx + 0x78cc, &t, 4); }
void set_camera(float x, float z) {
    std::memcpy(g_camera + 0x30, &x, 4);
    std::memcpy(g_camera + 0x38, &z, 4);
}

// A fresh stand-in: screen-space mode (one grid, every surface's holder the
// screen's) or a grid a surface. Surface 2's vectors come from two heaps.
void build(World& w, bool screen) {
    std::memset(static_cast<void*>(&w.manager), 0, sizeof w.manager);
    w.screen = screen;
    const u64 m = at(w.manager);
    store<u8>(m + 0x78, 1);
    store<u8>(m + 0x79, screen ? 1 : 0);
    store<u64>(m + 0x68, at(w.none_holder));
    for (int i = 0; i < kSurfaces; ++i) {
        const u64 g = at(w.grids[i]);
        std::memset(static_cast<void*>(&w.grids[i]), 0, sizeof w.grids[i]);
        store<u8>(g + 0x30, 3);
        store<u64>(g + 0x10, w.next_map++);
        store<u64>(g + 0xd0, w.next_map++);
        store<u64>(g + 0xd8, w.next_map++);
        const u64 h = at(w.holders[i]);
        std::memset(static_cast<void*>(&w.holders[i]), 0, sizeof w.holders[i]);
        store<u64>(h, g);
        store<u8>(h + 0x18, 1);
        w.holder_list[i] = h;
        const u64 s = at(w.surfaces[i]);
        std::memset(static_cast<void*>(&w.surfaces[i]), 0, sizeof w.surfaces[i]);
        w.surface_list[i] = s;
        for (int k = 0; k < 2; ++k) {
            const u64 v = at(w.vectors[2 * i + k]);
            std::memset(static_cast<void*>(&w.vectors[2 * i + k]), 0, sizeof w.vectors[0]);
            const u64 first = reinterpret_cast<u64>(&w.room[2 * i + k][0]);
            store<u64>(v + 8, first);
            store<u64>(v + 0x10, first);
            store<u64>(v + 0x18, first + kRoom * sizeof(Impulse));
            store<u64>(v + 0x20, at(w.heaps[i == 2 ? k : 0]));
            store<u64>(s + (k ? 0xa0 : 0x98), v);
        }
    }
    if (screen) {
        const u64 sh = at(w.screen_holder);
        std::memset(static_cast<void*>(&w.screen_holder), 0, sizeof w.screen_holder);
        store<u64>(sh, at(w.grids[0]));
        store<u8>(sh + 0x18, 1);
        store<u64>(m + 0x60, sh);
        for (int i = 0; i < kSurfaces; ++i) store<u64>(at(w.surfaces[i]) + 8, sh);
        // the holder list holds the screen's other grids' holders, not in use
        for (int i = 0; i < kSurfaces; ++i) store<u8>(w.holder_list[i] + 0x18, 0);
    } else {
        for (int i = 0; i < kSurfaces; ++i) store<u64>(at(w.surfaces[i]) + 8, w.holder_list[i]);
    }
    store<u64>(m + 0x18, reinterpret_cast<u64>(&w.holder_list[0]));
    store<u64>(m + 0x20, reinterpret_cast<u64>(&w.holder_list[kSurfaces]));
    store<u64>(m + 0x40, reinterpret_cast<u64>(&w.surface_list[0]));
    store<u64>(m + 0x48, reinterpret_cast<u64>(&w.surface_list[kSurfaces]));
}

void queue_impulse(World& w, int surface) {  // sub_2677170: nothing while the manager is off
    if (!load<u8>(at(w.manager) + 0x78)) return;
    const u64 queued = load<u64>(at(w.surfaces[surface]) + 0xa0);
    const u64 end = load<u64>(queued + 0x10);
    if (end + sizeof(Impulse) > load<u64>(queued + 0x18)) {
        check(false, "the stand-in's queue has room");
        return;
    }
    const u64 id = w.next_impulse++;
    store<u64>(end, id);
    store<u64>(queued + 0x10, end + sizeof(Impulse));
    w.added[id] = w.now;
    if (w.batch_frame[surface] != w.frame_no) {
        w.batch[surface].clear();
        w.batch_frame[surface] = w.frame_no;
    }
    w.batch[surface].push_back(id);
}

struct Frames {
    double fps = 60;
    bool uneven = false, screen = true, running = false;
    int frames = 0;
    long resize_at = -1;
    int off_from = -1, off_to = -1;  // the manager off over these frames
    bool our_task = true;            // false: the game's task in place of ours
    int impulse_every = 1;           // a frame, each surface
    bool after_steps = false;        // impulses only on the frames after a step (an effect queueing on its steps' draws)
    bool schedule = false;           // the frame-rate code's schedule of whole 1/30 s told each frame (sfx_schedule::next)
};

struct Outcome {
    int frames = 0, steps = 0, held_shown_wrong = 0, misaligned = 0, bad_answers = 0;
    int off_schedule = 0;  // frames that stepped where the schedule held, or held where it stepped (camera steps apart)
    double game_s = 0.0;
};

// The frames, each: impulses queued, the time and camera set, the update and
// the task; then what a surface would bind checked.
Outcome play(World& w, const Frames& f) {
    W = &w;
    w.resize_at = f.resize_at;
    w.check_latest = f.off_from < 0;
    for (ManagerGate& mg : g_managers) {
        mg.manager.store(0);
        mg.gate = WaterGate{};
        mg.decided = false;
        mg.waiting = false;
        mg.scheduled = 0;
        mg.turned = 0;
    }
    g_schedule_frames.store(0);
    g_schedule_between.store(0);
    double carry = 0.0;  // the schedule's, as engine/frame_rate.cpp keeps it
    g_task_ours.store(false);
    const u64 self = 0x1234, context = reinterpret_cast<u64>(g_context), param = reinterpret_cast<u64>(g_param);
    const u64 cam = reinterpret_cast<u64>(g_camera);
    std::memcpy(g_context + 0x3f8, &cam, 8);
    g_param[0] = at(w.manager);
    g_param[1] = cam;
    g_game_water_update = reinterpret_cast<void*>(&game_update);
    g_game_water_task = reinterpret_cast<void*>(&game_task);
    Outcome o;
    float t = 500.0f;
    const float t0 = t;
    bool have_step = false, stepped_before = true;
    float step_x = 0, step_z = 0;
    for (int i = 0; i < f.frames; ++i) {
        w.frame_no = i;
        w.now = static_cast<double>(t) - static_cast<double>(t0);
        set_time(t);
        const double dt = f.uneven ? uniform(1.0 / 240, 1.0 / 30) : 1.0 / f.fps;
        bool whole = true;
        if (f.schedule) {  // each_frame: the schedule for the frame about to run, from its own time
            whole = sfx_schedule::next(carry, dt, 30).whole > 0;
            decomp_water_step_frame(!whole);
        }
        float x = 10.5f, z = -20.5f;
        if (f.running) {
            x = static_cast<float>(-3.0 + 6.5 * w.now);
            z = static_cast<float>(-1.0 + 2.5 * w.now);
        }
        set_camera(x, z);
        const bool on = !(i >= f.off_from && i < f.off_to);
        store<u8>(at(w.manager) + 0x78, on ? 1 : 0);
        if (f.after_steps ? stepped_before : i % f.impulse_every == 0)
            for (int s = 0; s < kSurfaces; ++s) queue_impulse(w, s);
        const int steps_before = w.steps;
        water_update(at(w.manager));
        g_args_ok = true;
        const u64 r = f.our_task ? water_task(self, context, param) : game_task(self, context, param);
        const bool stepped = w.steps != steps_before;
        stepped_before = stepped;
        if (on && f.our_task && (stepped ? r != 0x5eedu || !g_args_ok : r != 0)) ++o.bad_answers;
        // on the schedule, without the camera: a step on every whole frame and on no other (the first two frames the
        // game's and the start)
        if (f.schedule && !f.running && on && i >= 2 && stepped != whole) ++o.off_schedule;
        // the grid follows the camera: every frame whose truncated camera left the last step's steps
        if (w.screen && on) {
            if (stepped) {
                have_step = true;
                step_x = std::trunc(x);
                step_z = std::trunc(z);
            } else if (have_step && (std::trunc(x) != step_x || std::trunc(z) != step_z)) {
                ++o.misaligned;
            }
        }
        // surface 2's vectors come from two heaps: each keeps its own room, every frame
        if (load<u64>(load<u64>(at(w.surfaces[2]) + 0x98) + 8) != reinterpret_cast<u64>(&w.room[4][0]) ||
            load<u64>(load<u64>(at(w.surfaces[2]) + 0xa0) + 8) != reinterpret_cast<u64>(&w.room[5][0]))
            ++w.rooms_swapped;
        // what the water surfaces bind: the last step's heights
        for (int g = 0; g < (w.screen ? 1 : kSurfaces); ++g) {
            const u64 grid = at(w.grids[g]);
            const bool shown = content_of(load<u64>(grid + 0x10)) == w.written[g];
            if (!shown && !(g == 0 && w.swapped_in)) ++o.held_shown_wrong;
        }
        t += static_cast<float>(dt);
        ++o.frames;
    }
    o.game_s = static_cast<double>(t) - static_cast<double>(t0);
    o.steps = w.steps;
    return o;
}

// Every step of grid g read the two before it in order: step n (from 1) read
// n-1 and n-2 - outside a resize, the reference's.
int out_of_order(const World& w, int g, const std::vector<std::pair<long, long>>* reference = nullptr) {
    int bad = 0;
    const auto& r = w.reads[g];
    for (std::size_t n = 0; n < r.size(); ++n) {
        if (reference) {
            if (n < reference->size() && r[n] != (*reference)[n]) ++bad;
            continue;
        }
        const long step = static_cast<long>(n) + 1;
        if (step >= 3 && (r[n].first != step - 1 || r[n].second != step - 2)) ++bad;
    }
    return bad;
}

// Every step took the latest frame's impulses since the step before (or none,
// none queued since); none taken twice.
void impulses_latest(const World& w, const char* what) {
    int twice = 0;
    for (const auto& [id, n] : w.taken) twice += n > 1;
    if (!g_quiet && (w.not_latest || twice))
        std::printf("    %s: %d steps took other than the latest frame's impulses, %d taken twice\n", what, w.not_latest, twice);
    check(w.not_latest == 0, what);
    check(twice == 0, "no impulse taken twice");
}

void frames_and_maps() {
    struct Rate {
        const char* name;
        double fps;
        bool uneven;
    } const kRates[] = {{"30", 30, false}, {"60", 60, false}, {"90", 90, false}, {"120", 120, false}, {"144", 144, false}, {"uneven", 0, true}};
    for (int sched = 1; sched >= 0; --sched)
    for (int screen = 1; screen >= 0; --screen) {
        for (int running = 0; running < (screen ? 2 : 1); ++running) {
            for (const Rate& rt : kRates) {
                static World w;
                w = World{};
                build(w, screen != 0);
                Frames f;
                f.fps = rt.fps;
                f.uneven = rt.uneven;
                f.screen = screen != 0;
                f.running = running != 0;
                f.schedule = sched != 0;
                f.frames = rt.uneven ? 2400 : static_cast<int>(rt.fps * 10);
                const Outcome o = play(w, f);
                if (sched && !running) check(o.off_schedule == 0, "on the schedule, a step on every whole 1/30 s frame and no other");
                int bad = 0;
                for (int g = 0; g < (screen ? 1 : kSurfaces); ++g) bad += out_of_order(w, g);
                const double per_s = (screen ? o.steps : o.steps / kSurfaces) / o.game_s;
                if (!g_quiet)
                    std::printf("  %s, %s%s%s: %d frames, %.2f steps a game second, %d steps read out of order, %d frames showing other "
                                "than the last step, worst impulse delay %.1f ms%s\n",
                                rt.name, screen ? "screen-space" : "a grid a surface", running ? ", running" : "",
                                sched ? ", on the schedule" : ", on the game time", o.frames, per_s, bad, o.held_shown_wrong,
                                w.worst_delay * 1000.0, sched && !running ? (o.off_schedule ? ", OFF the schedule" : ", on its frames") : "");
                check(bad == 0, "every step reads the two before it in order");
                check(o.held_shown_wrong == 0, "the frames between steps show the last step's heights");
                check(o.misaligned == 0, "every frame whose truncated camera left the last step's steps");
                check(o.bad_answers == 0, "the game's task with the same arguments and its answer on a step, 0 on the others");
                if (rt.fps == 30)
                    check((screen ? o.steps : o.steps / kSurfaces) == o.frames, "at 30 every frame steps");
                else
                    check(std::fabs(per_s - 30.0) < (running ? 0.6 : 0.4), "30 steps a game second");
                impulses_latest(w, "a step takes the latest frame's impulses, none older beside them");
                // the next step at the latest - after a camera's step took its 1/30 s ahead, up to one more
                const double frame = rt.uneven ? 1.0 / 30 : 1.0 / rt.fps;
                check(w.worst_delay <= (running ? 2 * kWaterStep : kWaterStep) + frame + 1e-4, "an impulse waits for the next step at most");
                // an effect that queues its impulses on the frames after the steps only (the ones its steps drew): each
                // reaches the next step, whatever frames are held between
                static World ws;
                ws = World{};
                build(ws, screen != 0);
                f.after_steps = true;
                play(ws, f);
                impulses_latest(ws, "impulses queued on the frames after steps only reach the next step");
                check(ws.worst_delay <= (running ? 2 * kWaterStep : kWaterStep) + frame + 1e-4,
                      "an impulse queued after a step waits for the next step at most");
                // and one that queues every 7th frame, whatever the steps: a camera's step takes what waits, and nothing
                // it took comes back
                ws = World{};
                build(ws, screen != 0);
                f.after_steps = false;
                f.impulse_every = 7;
                play(ws, f);
                impulses_latest(ws, "impulses queued every 7th frame: each step the latest since the step before");
                check(ws.rooms_swapped == 0 && w.rooms_swapped == 0, "two heaps' vectors never swapped");
            }
        }
    }
}

void resize() {
    // the game's own order, at 30 (every frame steps), then 60 and 120 with
    // the resize's swap on a frame between steps
    static World ref;
    ref = World{};
    build(ref, true);
    Frames f;
    f.fps = 30;
    f.frames = 120;
    f.resize_at = 40;
    play(ref, f);
    for (double fps : {60.0, 120.0}) {
        static World w;
        w = World{};
        build(w, true);
        f.fps = fps;
        f.frames = static_cast<int>(fps * 4);
        const Outcome o = play(w, f);
        const int bad = out_of_order(w, 0, &ref.reads[0]);
        if (!g_quiet)
            std::printf("  a resize at %.0f: %zu steps, %d read other than at 30, %d frames showing other than the last step\n", fps, w.reads[0].size(),
                        bad, o.held_shown_wrong);
        check(w.reads[0].size() >= 100 && bad == 0, "a resize on a frame between steps read as at 30");
        check(o.held_shown_wrong == 0, "the frames between steps show the last step's heights around a resize");
    }
}

void off_and_passes() {
    // the manager off for a while (its update does nothing, nothing is queued): nothing taken twice - from a frame that
    // stepped (what it took still in +0x98) and from one between
    static World w;
    Frames f;
    int twice = 0;
    for (int from : {201, 200})
        for (int len : {60, 61}) {
            w = World{};
            build(w, true);
            f = Frames{};
            f.fps = 60;
            f.frames = 600;
            f.off_from = from;
            f.off_to = from + len;
            play(w, f);
            twice = 0;
            for (const auto& [id, n] : w.taken) twice += n > 1;
            check(twice == 0, "with the manager off for a while no impulse is taken twice");
            check(out_of_order(w, 0) == 0, "with the manager off for a while the steps read in order");
            check(w.rooms_swapped == 0, "two heaps' vectors never swapped, the manager off for a while");
        }
    // surface 2's vectors come from two heaps: each keeps its own room
    const u64 room_taken = reinterpret_cast<u64>(&w.room[4][0]), room_queued = reinterpret_cast<u64>(&w.room[5][0]);
    auto own_rooms = [&](World& x) {
        const u64 t = load<u64>(at(x.surfaces[2]) + 0x98), q = load<u64>(at(x.surfaces[2]) + 0xa0);
        return load<u64>(t + 8) == room_taken && load<u64>(q + 8) == room_queued;
    };
    check(own_rooms(w), "two heaps' vectors never swapped");
    // the same with impulses every 7th frame: a held frame's then wait for the next step with nothing newer beside them
    // (they move to the queue, in its room), and what a step took never comes back, the manager off or on
    w = World{};
    build(w, true);
    f.impulse_every = 7;
    play(w, f);
    twice = 0;
    for (const auto& [id, n] : w.taken) twice += n > 1;
    check(twice == 0, "impulses every 7th frame, the manager off for a while: none taken twice");
    check(own_rooms(w) && w.rooms_swapped == 0, "impulses every 7th frame: two heaps' vectors never swapped");
    f.impulse_every = 1;
    // BBHOST_WATER_GATE=0: every frame the game's
    w = World{};
    build(w, true);
    g_gate_on = false;
    f = Frames{};
    f.fps = 60;
    f.frames = 300;
    Outcome o = play(w, f);
    g_gate_on = true;
    check(o.steps == 300 && w.updates == 300, "BBHOST_WATER_GATE=0: every frame the game's");
    // no GX world: every frame the game's
    w = World{};
    build(w, true);
    g_have_gx = false;
    o = play(w, f);
    g_have_gx = true;
    check(o.steps == 300, "no GX world: every frame the game's");
    // the task not ours: its update holds no frame, every step in order
    w = World{};
    build(w, true);
    f.our_task = false;
    o = play(w, f);
    check(o.steps == 300 && out_of_order(w, 0) == 0, "while the task is not ours the update holds no frame");
}

int run_all() {
    g_fail = 0;
    g_rng = 0x2677e50u;
    rates();
    standing_and_loads();
    camera();
    frames_and_maps();
    resize();
    off_and_passes();
    return g_fail;
}

}  // namespace

int main() {
    std::printf("decomp_gx_water_gate: the water interaction's steps at 30 a game second\n");
    const int fails = run_all();
    static const char* const kMutants[] = {"",
                                           "the gate never on (every frame the game's)",
                                           "screen-space mode ignored",
                                           "never holding a frame",
                                           "floor for trunc",
                                           "a camera step not taken from the next",
                                           "no start over when the game time goes back",
                                           "a whole 1/30 s to step (no room for the clock's rounding)",
                                           "the maps not turned back on a held frame",
                                           "the maps put back as they were (a resize undone)",
                                           "a held frame's impulses dropped",
                                           "frames held before the task is ours",
                                           "impulses kept waiting after an update that did nothing",
                                           "impulses swapped between two heaps' vectors (and no room used)",
                                           "a camera step without turning the maps round again",
                                           "a camera step's impulses kept waiting as well",
                                           "the schedule ignored (the game time decides)",
                                           "a camera step's whole step not taken from the schedule's next",
                                           "a held frame's impulses kept beside a newer frame's (every frame's taken)"};
    constexpr int kCount = sizeof kMutants / sizeof kMutants[0] - 1;
    int caught = 0, mutants = 0;
    g_quiet = true;
    for (int m = 1; m <= kCount; ++m) {
        g_water_gate_mutant = m;
        const int f = run_all();
        ++mutants;
        caught += f > 0;
        std::printf("  mutant %d (%s): %s\n", m, kMutants[m], f ? "caught" : "NOT CAUGHT");
    }
    g_water_gate_mutant = 0;
    g_quiet = false;
    std::printf("decomp_gx_water_gate: %s (%d failures; %d of %d mutants caught)\n", fails == 0 && caught == mutants ? "pass" : "FAIL", fails, caught,
                mutants);
    return fails == 0 && caught == mutants ? 0 : 1;
}
