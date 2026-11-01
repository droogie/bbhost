// ReplayManipulator's update by game time (decomp/chr/replay_manipulator.cpp)
// beside the game's own, sub_192c390, in the loaded eboot. A ghost here is the
// manipulator (its real vtable, whose move is the game's store), a recording
// of samples, and the character it moves - the position and angle the update
// reads, which a played-back ghost moves by what the update leaves, as the
// character's own update does - with the samples' events (sub_1e1f7d0, and
// sub_21e0ee0 / sub_21f0e70 for each sub-event) recorders.
//   - At the game's own 30 (the clock unpublished): on generated ghosts -
//     countdowns, samples, positions near, far and past the jump, the step
//     1/30, 0, -0, NaN, a denormal; the recording missing or empty; the
//     fade's window - ours is the game's update: every byte of the ghost and
//     the character, and every event call, the same.
//   - At frames of other lengths (60 to 1000 fps and any between), a frame
//     that begins a 30 fps frame or not: ours is the game's update with the
//     countdown held where none begins (and the countdown as it was after),
//     a new ghost's first sample at once, and the move and turn it left times
//     the frame in 30 fps frames - the jump past 10 units whole; a ghost of
//     another vtable the game's update untouched.
//   - Played whole: recordings of a path (and of jumps past 10 units) at 30,
//     and by ours at 60, 90, 120, 144, 165, 240, 360 and 1000 fps and under a
//     jitter of 30-240: the samples taken at the game seconds 30 takes them
//     (a frame early at most where the frames do not divide 1/30), 3 a game
//     second, the character on 30's path - where it is at each 1/30 s of game
//     time within a frame's move of 30's - and the replay at its end at 30's
//     time; the game's own update unconverted at 120 runs 4 times as fast.
// Each of the source's mutants caught; the function's bytes the ones ours
// runs around; and the list takes it where the frame clock is published - by
// default at a converted rate, at every rate with the clock on - unless
// BBHOST_REPLAY_STEP=0. Skips without the 1.09 eboot.
#define DECOMP_CHR_REPLAY_MANIPULATOR_MUTANTS
int g_replay_manipulator_mutant = 0;
#include "decomp/chr/replay_manipulator.cpp"

#include "eboot_kit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// What the source's add function put on the list (the kit's weak decomp_add
// stands aside for this one).
int g_added = 0;
const char* g_added_name = nullptr;
void decomp_add(const DecompFunction& fn) {
    ++g_added;
    g_added_name = fn.name;
}

namespace {

eboot_kit::Rng g_rng(0x192c390);

constexpr u64 kEvents = 0x1e1f7d0, kSubLookup = 0x21e0ee0, kSubAdd = 0x21f0e70;
constexpr u64 kMoveStore = 0x192cbe0;  // the vtable's +0x130: the move into +0x70
constexpr int kMaxSamples = 48;
constexpr u64 kSampleSize = 0x7c;

// The events the update hands on, in order.
struct Seen {
    int events = 0, lookups = 0, adds = 0;
    u64 sum = 0;  // of what they were handed
    bool operator==(const Seen&) const = default;
};
Seen g_seen;
GUEST_ABI u64 events_recorder(u64 module, u64 events, u64 a, u64 b) {
    ++g_seen.events;
    g_seen.sum = g_seen.sum * 31 + module + events * 3 + (a & 0xffffffff) * 5 + (b & 0xff) * 7;
    return 0;
}
GUEST_ABI u64 lookup_recorder(u64 out, u64 table, u64 id) {
    ++g_seen.lookups;
    g_seen.sum = g_seen.sum * 31 + table + (id & 0xffffffff);
    store<s32>(out, (id & 1) ? -1 : static_cast<s32>(id & 0xfff));
    return 0;
}
GUEST_ABI u64 add_recorder(u64 table, u64 entry) {
    ++g_seen.adds;
    g_seen.sum = g_seen.sum * 31 + table + load<u32>(entry) + load<u32>(entry + 8);
    return 0;
}

// The objects, at fixed addresses so both runs see the same pointers.
struct alignas(16) Memory {
    u8 ghost[0x120];
    u8 recording[0xa0];
    u8 samples[kMaxSamples * kSampleSize + 16];
    u8 owner[0x40];      // the update's second argument: +8 the character
    u8 chr[0x400];       // +0x3b0 its modules; +0x1e5 and +0x20c the fade's
    u8 modules[0x80];    // +0x38 the event module (or none), +0x68 the position's
    u8 position[0x200];  // +0x1d0 the angle, +0x1e0 the position
    u8 events[0x40];     // +0x10 the sub-events' table (or none)
    u8 table[0x40];
    alignas(16) u8 vtable[0x200];  // another class's (a copy of the replay's)
};
Memory g_mem, g_start, g_after_game;

u64 ghost() { return reinterpret_cast<u64>(g_mem.ghost); }
u64 owner() { return reinterpret_cast<u64>(g_mem.owner); }
template <class T>
void put(u8* base, u64 o, T v) {
    std::memcpy(base + o, &v, sizeof v);
}
template <class T>
T get(const u8* base, u64 o) {
    T v;
    std::memcpy(&v, base + o, sizeof v);
    return v;
}

Update g_game = nullptr;

int g_failed = 0;
bool g_quiet = false;
void fail(int c, const char* what) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: case %d: %s\n", c, what);
}

// A sample's packed position and yaw, as the recorder writes them
// (sub_1d09b40): x and z in 18 bits of 1/50, y in 17 of 1/25 (its low three
// bits at 32), the yaw in the top 7 bits.
u64 pack(float x, float y, float z, float yaw) {
    const auto field = [](float v, float scale, int bits) {
        long n = std::lround(static_cast<double>(v) * scale);
        const long lim = (1L << (bits - 1)) - 1;
        if (n > lim) n = lim;
        if (n < -lim) n = -lim;
        return static_cast<u64>(n) & ((1ull << bits) - 1);
    };
    const u64 xs = field(x, 50.0f, 18), zs = field(z, 50.0f, 18), ys = field(y, 25.0f, 17);
    long q = std::lround((static_cast<double>(yaw) / 3.14159265358979 + 1.0) * 64.0);
    q = q < 0 ? 0 : q > 127 ? 127 : q;
    return xs | ((ys >> 3) << 18) | ((ys & 7) << 32) | (zs << 35) | (static_cast<u64>(q) << 57);
}

float special(float lo, float hi) {
    g_rng.specials = true;
    const float v = g_rng.value(lo, hi);
    g_rng.specials = false;
    return v;
}

// A generated ghost: garbage where the update reads nothing, its fields often
// what the game builds.
void generate() {
    for (u8& b : g_start.ghost) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_start.chr) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_start.position) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_start.samples) b = static_cast<u8>(g_rng.u32());
    std::memset(g_start.recording, 0, sizeof g_start.recording);
    std::memset(g_start.owner, 0, sizeof g_start.owner);
    std::memset(g_start.modules, 0, sizeof g_start.modules);
    std::memset(g_start.events, 0, sizeof g_start.events);
    std::memcpy(g_start.vtable, eboot_kit::at(kVtable), sizeof g_start.vtable);
    // The ghost: the replay's vtable mostly, another class's now and then.
    put<u64>(g_start.ghost, 0, g_rng.chance(0.9) ? kVtable : reinterpret_cast<u64>(g_mem.vtable));
    const int count = g_rng.chance(0.08) ? 0 : g_rng.range(1, kMaxSamples);
    put<u64>(g_start.ghost, kRecording, g_rng.chance(0.05) ? 0 : reinterpret_cast<u64>(g_mem.recording));
    put<u64>(g_start.recording, 0x88, g_rng.chance(0.03) ? 0 : reinterpret_cast<u64>(g_mem.samples));
    put<s32>(g_start.recording, kSampleCount, g_rng.chance(0.03) ? -1 : count);
    put<s32>(g_start.ghost, kCountdown, g_rng.chance(0.1) ? g_rng.range(-3, 0) : g_rng.range(1, 10));
    put<s32>(g_start.ghost, kSample, g_rng.range(0, count + 1));
    // The character and where it stands.
    put<u64>(g_start.owner, 8, reinterpret_cast<u64>(g_mem.chr));
    put<u64>(g_start.chr, 0x3b0, reinterpret_cast<u64>(g_mem.modules));
    put<u64>(g_start.modules, 0x68, reinterpret_cast<u64>(g_mem.position));
    if (g_rng.chance(0.4)) {
        put<u64>(g_start.modules, 0x38, reinterpret_cast<u64>(g_mem.events));
        if (g_rng.chance(0.6)) put<u64>(g_start.events, 0x10, reinterpret_cast<u64>(g_mem.table));
    }
    const float px = g_rng.uniform(-300.0f, 300.0f), py = g_rng.uniform(-50.0f, 50.0f), pz = g_rng.uniform(-300.0f, 300.0f);
    const float pos[4] = {px, py, pz, 1.0f};
    std::memcpy(g_start.position + 0x1e0, pos, 16);
    const float ang[4] = {g_rng.chance(0.8) ? 0.0f : g_rng.uniform(-1.0f, 1.0f), g_rng.uniform(-3.2f, 3.2f), 0.0f, 0.0f};
    std::memcpy(g_start.position + 0x1d0, ang, 16);
    // The samples: near the character, within 1/1000, past 10 units.
    for (int i = 0; i < kMaxSamples; ++i) {
        u8* s = g_start.samples + i * kSampleSize;
        const float r = g_rng.chance(0.1) ? 0.0f : g_rng.chance(0.15) ? g_rng.uniform(10.5f, 40.0f) : g_rng.uniform(0.0f, 6.0f);
        const float a = g_rng.uniform(-3.14f, 3.14f);
        put<u64>(s, 0, pack(px + r * std::cos(a), py + g_rng.uniform(-1.0f, 1.0f), pz + r * std::sin(a), g_rng.uniform(-3.14f, 3.14f)));
        put<s8>(s, 0x79, static_cast<s8>(g_rng.range(-1, 3)));
    }
    // The tenths, the elapsed time, the end: what a ghost had, specials too.
    for (u64 o = 0xd0; o < 0xf0; o += 4) put<float>(g_start.ghost, o, g_rng.chance(0.2) ? special(-0.5f, 0.5f) : g_rng.uniform(-0.5f, 0.5f));
    put<float>(g_start.ghost, 0x104, g_rng.chance(0.2) ? 0.0f : g_rng.uniform(0.0f, 12.0f));
    put<float>(g_start.ghost, 0x108, g_rng.uniform(0.0f, 14.0f));
}

float a_step() {
    const int k = g_rng.range(0, 9);
    if (k == 0) return 0.0f;
    if (k == 1) return -0.0f;
    if (k == 2) return special(-1.0f, 1.0f);
    if (k <= 5) return frame_clock::kThirtieth;
    return g_rng.uniform(0.0005f, frame_clock::kThirtieth);
}

// The game's update on the start state, then ours: what each left.
void both(float step, Seen& game, Seen& ours) {
    g_mem = g_start;
    g_seen = Seen{};
    g_game(ghost(), owner(), step);
    game = g_seen;
    g_after_game = g_mem;
    g_mem = g_start;
    g_seen = Seen{};
    replay_manipulator_update(ghost(), owner(), step);
    ours = g_seen;
}

void clock_at(float dt, int ticks) {
    frame_clock::g_dt.store(dt);
    frame_clock::g_frames30.store(frame_clock::frames30_of(dt));
    frame_clock::g_ticks30.store(ticks);
}
void clock_unpublished() { clock_at(frame_clock::kThirtieth, 1); }

// At the game's own 30: the game's update, byte for byte.
long at_thirty(int cases) {
    clock_unpublished();
    long n = 0;
    for (int c = 0; c < cases; ++c, ++n) {
        generate();
        Seen game, ours;
        both(a_step(), game, ours);
        if (!(game == ours)) fail(c, "at 30: the events handed on");
        if (std::memcmp(&g_mem, &g_after_game, sizeof g_mem) != 0) fail(c, "at 30: what the update left");
    }
    return n;
}

// The update as ours is to run it at a converted frame: the game's, with the
// countdown held where no 30 fps frame begins (not a new ghost's), then the
// move and the turn scaled to the frame.
void expected(float step, int ticks, float frames) {
    g_mem = g_start;
    const bool replay = get<u64>(g_start.ghost, 0) == kVtable;
    const s32 countdown = get<s32>(g_start.ghost, kCountdown);
    const bool hold = replay && ticks == 0 && countdown > 0;
    if (hold) put<s32>(g_mem.ghost, kCountdown, 2);
    g_seen = Seen{};
    g_game(ghost(), owner(), step);
    if (hold) put<s32>(g_mem.ghost, kCountdown, countdown);
    const u64 rec = get<u64>(g_start.ghost, kRecording);
    const bool plays = rec != 0 && get<s32>(g_start.recording, kSampleCount) > 0;
    if (!replay || !plays || frames == 1.0f) return;
    if (std::memcmp(g_mem.ghost + kMove, g_mem.ghost + kMoveTenth, 16) == 0)
        for (u64 k = 0; k < 16; k += 4) put<float>(g_mem.ghost, kMove + k, get<float>(g_mem.ghost, kMoveTenth + k) * frames);
    for (u64 k = 0; k < 16; k += 4) put<float>(g_mem.ghost, kTurn + k, get<float>(g_mem.ghost, kTurnTenth + k) * frames);
}

constexpr float kSteps[] = {1.0f / 60.0f, 1.0f / 90.0f, 1.0f / 120.0f, 1.0f / 144.0f, 1.0f / 165.0f, 1.0f / 240.0f, 1.0f / 360.0f, 1.0f / 1000.0f};

long converted(int cases) {
    long n = 0;
    for (int c = 0; c < cases; ++c, ++n) {
        const int k = g_rng.range(-1, static_cast<int>(sizeof kSteps / sizeof kSteps[0]));
        const float dt = k < 0 ? frame_clock::kThirtieth
                         : k < static_cast<int>(sizeof kSteps / sizeof kSteps[0]) ? kSteps[k]
                                                                                  : g_rng.uniform(0.001f, frame_clock::kThirtieth);
        const int ticks = g_rng.chance(0.5) ? 1 : 0;
        clock_at(dt, ticks);
        generate();
        const float step = g_rng.chance(0.9) ? dt : a_step();
        expected(step, ticks, frame_clock::frames30());
        const Memory want = g_mem;
        const Seen want_seen = g_seen;
        g_mem = g_start;
        g_seen = Seen{};
        replay_manipulator_update(ghost(), owner(), step);
        if (!(g_seen == want_seen)) fail(c, "converted: the events handed on");
        if (std::memcmp(&g_mem, &want, sizeof g_mem) != 0) fail(c, "converted: what the update left");
        // What holding means, whatever the rest: no sample taken, the countdown kept.
        if (ticks == 0 && get<u64>(g_start.ghost, 0) == kVtable && get<s32>(g_start.ghost, kCountdown) > 0 &&
            (get<s32>(g_mem.ghost, kSample) != get<s32>(g_start.ghost, kSample) ||
             get<s32>(g_mem.ghost, kCountdown) != get<s32>(g_start.ghost, kCountdown)))
            fail(c, "a frame that begins no 30 fps frame took a sample or moved the countdown");
    }
    clock_unpublished();
    return n;
}

// ---- Played whole -------------------------------------------------------------

struct Track {
    std::vector<double> t;        // game seconds at each frame's end
    std::vector<float> x, z, yaw; // the character there
    std::vector<double> samples;  // game seconds at the start of each frame that took a sample
    double done = -1.0;           // when the replay ended (+0x10c)
};

// A recording of `count` samples along a path (with jumps past 10 units when
// asked), and a fresh ghost at its start, as the factory makes one.
void recording(int count, bool jumps) {
    std::memset(&g_mem, 0, sizeof g_mem);
    put<u64>(g_mem.ghost, 0, kVtable);
    put<u64>(g_mem.ghost, kRecording, reinterpret_cast<u64>(g_mem.recording));
    put<u64>(g_mem.recording, 0x88, reinterpret_cast<u64>(g_mem.samples));
    put<s32>(g_mem.recording, kSampleCount, count);
    put<s32>(g_mem.ghost, 0x100, -1);
    put<float>(g_mem.ghost, 0x108, 1e9f);  // no fade
    put<u64>(g_mem.owner, 8, reinterpret_cast<u64>(g_mem.chr));
    put<u64>(g_mem.chr, 0x3b0, reinterpret_cast<u64>(g_mem.modules));
    put<u64>(g_mem.modules, 0x68, reinterpret_cast<u64>(g_mem.position));
    const float start[4] = {10.0f, 2.0f, -20.0f, 1.0f};
    std::memcpy(g_mem.position + 0x1e0, start, 16);
    for (int i = 0; i < count; ++i) {
        const float s = static_cast<float>(i);
        float x = 10.0f + 1.3f * s + std::sin(s * 0.7f), z = -20.0f + 0.8f * s * std::cos(s * 0.3f);
        if (jumps && i % 9 == 8) x += 25.0f;
        put<u64>(g_mem.samples + i * kSampleSize, 0, pack(x, 2.0f, z, std::sin(s * 0.5f) * 3.0f));
    }
}

// Plays the recording with `step(i)` the i-th frame's step; `ours` runs ours
// (the clock published every frame), else the game's update (the clock at
// 30). The character moves by the move and the turn left after each update.
template <class Step>
Track play(Step step, bool ours, int frames) {
    Track tr;
    double t = 0.0;
    frame_clock::g_tick_carry = 0.0;
    for (int i = 0; i < frames; ++i) {
        const float dt = step(i);
        if (ours) frame_clock::publish(dt);
        const s32 sample = get<s32>(g_mem.ghost, kSample);
        if (ours) replay_manipulator_update(ghost(), owner(), dt);
        else g_game(ghost(), owner(), dt);
        if (get<s32>(g_mem.ghost, kSample) != sample) tr.samples.push_back(t);
        if (tr.done < 0 && g_mem.ghost[0x10c]) tr.done = t;
        float* pos = reinterpret_cast<float*>(g_mem.position + 0x1e0);
        float* ang = reinterpret_cast<float*>(g_mem.position + 0x1d0);
        for (int k = 0; k < 4; ++k) {
            pos[k] += get<float>(g_mem.ghost, kMove + 4 * k);
            ang[k] += get<float>(g_mem.ghost, kTurn + 4 * k);
        }
        t += static_cast<double>(dt);
        tr.t.push_back(t);
        tr.x.push_back(pos[0]);
        tr.z.push_back(pos[2]);
        tr.yaw.push_back(ang[1]);
    }
    clock_unpublished();
    frame_clock::g_tick_carry = 0.0;
    return tr;
}

// Where `tr` is at game time `t`: at the end of its last frame that ends by
// then (a jump is never spread over the frames around it).
void at(const Track& tr, double t, float& x, float& z) {
    std::size_t j = 0;
    while (j + 1 < tr.t.size() && tr.t[j + 1] <= t + 1e-9) ++j;
    x = tr.x[j];
    z = tr.z[j];
}

// The largest move a frame makes in `tr` (bar the jumps).
float largest_move(const Track& tr) {
    float m = 0.0f;
    for (std::size_t j = 1; j < tr.x.size(); ++j) {
        const float d = std::hypot(tr.x[j] - tr.x[j - 1], tr.z[j] - tr.z[j - 1]);
        if (d < 10.0f && d > m) m = d;
    }
    return m;
}

struct Rate {
    const char* name;
    double fps;  // 0: jitter
};

void played(int& checked) {
    constexpr int kCount = 30;     // 10 game seconds of samples
    constexpr double kSpan = 12.0;  // played for 12 game seconds (the end at 10)
    for (const bool jumps : {false, true}) {
        recording(kCount, jumps);
        const Track ref = play([](int) { return frame_clock::kThirtieth; }, false, static_cast<int>(kSpan * 30));
        // The game's own cadence at 30: a sample each 10 updates, from the first.
        bool cadence = ref.samples.size() >= static_cast<std::size_t>(kCount);
        for (std::size_t i = 0; cadence && i < ref.samples.size(); ++i)
            cadence = std::fabs(ref.samples[i] - static_cast<double>(i) / 3.0) < 1e-4;
        if (!cadence) fail(-2, "the game's update at 30 did not take a sample each 10 updates");
        for (const Rate r : {Rate{"60", 60}, Rate{"90", 90}, Rate{"120", 120}, Rate{"144", 144}, Rate{"165", 165}, Rate{"240", 240},
                             Rate{"360", 360}, Rate{"1000", 1000}, Rate{"jitter 30-240", 0}}) {
            recording(kCount, jumps);
            eboot_kit::Rng jit(0x5eed);
            std::vector<float> steps;
            double total = 0.0;
            while (total < kSpan - 1e-6) {
                const float st = r.fps > 0 ? static_cast<float>(1.0 / r.fps) : jit.uniform(1.0f / 240.0f, frame_clock::kThirtieth);
                steps.push_back(st);
                total += static_cast<double>(st);
            }
            const Track run = play([&](int i) { return steps[static_cast<std::size_t>(i)]; }, true, static_cast<int>(steps.size()));
            ++checked;
            // The samples at 30's game seconds, each in the frame in which 30's begins
            // (within a frame early), over the time both cover.
            std::vector<double> mine, theirs;
            for (const double t : run.samples)
                if (t < kSpan - 0.05) mine.push_back(t);
            for (const double t : ref.samples)
                if (t < kSpan - 0.05) theirs.push_back(t);
            const double frame = r.fps > 0 ? 1.0 / r.fps : 1.0 / 30.0;
            if (mine.size() != theirs.size()) {
                if (!g_quiet) std::printf("FAIL: %s%s: %zu samples, 30 takes %zu\n", r.name, jumps ? " (jumps)" : "", mine.size(), theirs.size());
                ++g_failed;
                continue;
            }
            double worst_t = 0.0;
            for (std::size_t i = 0; i < theirs.size(); ++i) worst_t = std::max(worst_t, std::fabs(mine[i] - theirs[i]));
            if (worst_t > frame + 1e-6) {
                if (!g_quiet) std::printf("FAIL: %s%s: a sample %.4f s from 30's\n", r.name, jumps ? " (jumps)" : "", worst_t);
                ++g_failed;
            }
            // On 30's path at each 1/30 s of game time: within a 30 fps frame's move
            // for the sample a frame early, and one for the frame that ends before.
            // (While the replay plays: past its end the ghost keeps its last tenth and
            // jumps back each time it is 10 units past, in either run.)
            const float tol = 2.0f * largest_move(ref) + 1e-3f;
            float worst = 0.0f;
            double worst_at = 0.0;
            const auto jumped = [&](std::size_t j) {  // 30's frame j moved past 10 units (bar the first)
                return j > 0 && j < ref.x.size() && std::hypot(ref.x[j] - ref.x[j - 1], ref.z[j] - ref.z[j - 1]) > 10.0f;
            };
            for (std::size_t k = 0; k < ref.t.size() && ref.t[k] <= run.t.back() && ref.t[k] <= ref.done; ++k) {
                // At the moment a jump is made, one frame either side: the sample
                // that makes it may come a frame early.
                if (jumped(k) || jumped(k + 1)) continue;
                float x, z;
                at(run, ref.t[k], x, z);
                const float d = std::hypot(x - ref.x[k], z - ref.z[k]);
                if (d > worst) {
                    worst = d;
                    worst_at = ref.t[k];
                }
            }
            if (worst > tol) {
                if (!g_quiet)
                    std::printf("FAIL: %s%s: %.3f units from 30's path at %.4f s (two 30 fps frames' moves there: %.3f)\n", r.name,
                                jumps ? " (jumps)" : "", worst, worst_at, tol);
                ++g_failed;
            }
            if (std::fabs(run.done - ref.done) > frame + 1e-6) {
                if (!g_quiet)
                    std::printf("FAIL: %s%s: the replay ended at %.4f s, at 30 %.4f s\n", r.name, jumps ? " (jumps)" : "", run.done, ref.done);
                ++g_failed;
            }
            if (!g_quiet && !jumps)
                std::printf("  %-14s %zu samples (30: %zu), the latest %.4f s from 30's; the path within %.4f units of 30's (a 30 fps "
                            "frame's move %.4f); the end at %.3f s (30: %.3f s)\n",
                            r.name, mine.size(), theirs.size(), worst_t, worst, tol / 2.0f, run.done, ref.done);
        }
    }
    // The game's own update unconverted at 120 plays 4 times as fast.
    recording(kCount, false);
    const Track fast = play([](int) { return 1.0f / 120.0f; }, false, static_cast<int>(kSpan * 120));
    if (!(fast.done > 0 && fast.done < 3.0)) fail(-3, "the game's update unconverted at 120 did not end the replay early");
    else if (!g_quiet)
        std::printf("  the game's own update unconverted at 120: the replay ends at %.3f s of game time (at 30 at %.3f s)\n", fast.done,
                    10.0);
}

bool caught(int mutant) {
    const int before = g_failed;
    g_replay_manipulator_mutant = mutant;
    g_quiet = true;
    at_thirty(2000);
    converted(4000);
    int checked = 0;
    played(checked);
    g_quiet = false;
    g_replay_manipulator_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("chr_replay_manipulator_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 100000;
    const u8* at_file = eboot_kit::file_at("chr_replay_manipulator_test", kUpdate);
    if (!at_file || std::memcmp(at_file, kEntry, sizeof kEntry) != 0) fail(-1, "the entry's instructions");
    if (!at_file || !replay_manipulator_body_ok(at_file)) fail(-1, "the update's bytes are not the ones ours runs around");
    if (load<u64>(kVtable + 0x48) != kUpdate || load<u64>(kVtable + 0x130) != kMoveStore)
        fail(-1, "the replay's vtable (slot 9 the update, +0x130 the move's store)");
    eboot_kit::stub(kEvents, reinterpret_cast<const void*>(&events_recorder));
    eboot_kit::stub(kSubLookup, reinterpret_cast<const void*>(&lookup_recorder));
    eboot_kit::stub(kSubAdd, reinterpret_cast<const void*>(&add_recorder));
    g_game = eboot_kit::fn<Update>(kUpdate);
    // On the list where the clock is published - by default at a converted
    // rate (publishing), at every rate with the clock on - unless its own
    // switch is 0; the game's own at 30 with the clock off.
    for (int clock = 0; clock < 2; ++clock) {
        for (int pub = 0; pub < 2; ++pub) {
            for (int sw = 0; sw < 3; ++sw) {
                frame_clock::set_on(clock != 0);
                frame_clock::set_publishing(pub != 0);
                if (sw == 0) unsetenv("BBHOST_REPLAY_STEP");
                else setenv("BBHOST_REPLAY_STEP", sw == 1 ? "0" : "1", 1);
                g_added = 0;
                decomp_chr_replay_manipulator_add();
                const bool want = (clock || pub) && sw != 1;
                if (g_added != (want ? 1 : 0) || (want && std::strcmp(g_added_name, "sub_192c390") != 0))
                    fail(-1, "placed where it should not be, or not where it should");
            }
        }
    }
    frame_clock::set_on(false);
    frame_clock::set_publishing(false);
    unsetenv("BBHOST_REPLAY_STEP");
    g_game_update = reinterpret_cast<void*>(g_game);

    const long thirty = at_thirty(cases);
    const long conv = converted(cases);
    int checked = 0;
    std::printf("chr_replay_manipulator_test: recordings played whole:\n");
    played(checked);
    if (!g_held.load() || !g_moves.load() || !g_turns.load() || !g_samples.load()) fail(-1, "a kind of update never ran (held, moved, turned, sampled)");
    for (const int mutant : {1, 2, 3, 4, 5, 6, 7}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("chr_replay_manipulator_test: %d failures\n", g_failed);
        return 1;
    }
    std::printf("chr_replay_manipulator_test: ok - %ld updates at 30 the game's to the byte, %ld at converted frames as specified, %d "
                "recordings played whole on 30's path and pace, 7 mutants caught\n",
                thirty, conv, checked);
    return 0;
}
