// The wakes characters leave in water - GX's water interaction - at 30 steps a
// second of game time, at any frame rate.
//
// The water interaction's manager (vtable 0x5760c60, made by sub_2675f50
// inside the scene renderer at +0x1cd0) keeps a grid of heights: one that
// follows the camera in screen-space mode (the manager's +0x79, which its
// constructor sets), else one a water surface. A grid holds three height
// maps: +0x10, the one a step writes and the water surfaces read (sub_2675e30
// binds it for their vertex heights, beside the normal map built from it),
// +0xd0 the last step's and +0xd8 the one before. Two functions move it on,
// each once a frame:
//   - the manager's update, sub_2677780, which the scene renderer's prepare
//     (sub_26c53e0) runs before the task: it turns each grid's maps round
//     (sub_12bbda0: +0x10 <- +0xd8, +0xd0 <- +0x10, +0xd8 <- +0xd0), so the
//     oldest becomes the one to write, and hands each water surface the
//     impulses queued for it (the characters' splashes, which sub_2677170
//     adds to the vector at its +0xa0) in the vector at its +0x98, dropping
//     the ones there;
//   - its render task, sub_2677e50 (the manager's slot 0): it splats the +0x98
//     impulses into the last two maps (sub_12bce30), runs one wave step into
//     +0x10 (sub_12bbe80: the game's GXWaterInteractionHeightMap shader
//     integrates a damped wave equation over a fixed time, the grid's +0x84,
//     0.003, whatever the frame's) and builds the normal map (sub_2679ef0).
// So the waves spread at the frame rate: right at 30, twice as fast at 60,
// four times at 120 (measured on Moonside Lake in game time: 30.0 steps a game
// second at 30, 60.0 at 60, one a frame at both).
//
// Ours steps on the frames that close a whole 1/30 s of game time - the GX
// world's clock, *(0x5940dd8)+0x78cc, the one the shaders' time is read from
// (sub_2651f10) - decided once a frame at the update, before the task and the
// water surfaces' draws. On the frames between, the update runs as the game's,
// and then:
//   - each grid it turned round is turned back, so +0x10 holds the last
//     step's heights again and the next step reads its two maps before in
//     order (new maps a resize swapped in during the update, sub_12bd860,
//     are turned back with them, as the game's next update expects them);
//   - the task returns at once, and the impulses the update handed on stay
//     where the task reads them: the next update, before the game's, moves
//     them to the end of each surface's queue (in the room the queue has; into
//     an empty queue's place when it has none - the swap the update does
//     itself), so the next step takes them with the ones added meanwhile.
//     The update hands the impulses on while nothing adds to the queues; the
//     task may run while the game adds the next frame's, so ours touches the
//     two vectors only in the update, as the game's does.
// The water surfaces show the last step's water until the next, as they do
// between frames at 30. At 30 every frame steps: the game's.
//
// In screen-space mode the grid scrolls with the camera in whole units: a
// step moves it by trunc(the draw context's camera, +0x3f8 -> +0x30) less
// trunc(the camera arg3[1] holds, the frame before's), in x and z. On a frame
// between steps on which the camera's truncated x or z left the last step's,
// the task turns the grids round again and steps too, whatever its time, so
// the grid and what it holds never part, and the step is taken from the next
// ones (at most one ahead), which keeps 30 a second while the camera crosses;
// it takes that frame's impulses as any step does.
//
// BBHOST_WATER_GATE=0: the game's update and task, every frame.
#include "decomp/gx/water_gate.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "log.h"

using namespace decomp;

#ifdef DECOMP_WATER_GATE_MUTANTS
extern int g_water_gate_mutant;
#define MUTANT(n) (g_water_gate_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kWaterUpdate = 0x2677780;
constexpr u64 kWaterTask = 0x2677e50;
constexpr u64 kGxWorld = 0x5940dd8;   // GX's world singleton; +0x78c8 its timer, whose +4 is the game time
constexpr u64 kGameTime = 0x78cc;
constexpr u64 kContextCamera = 0x3f8;  // the draw context's camera: +0x30 x, +0x38 z

// The manager.
constexpr u64 kManagerHolders = 0x18;      // +0x18 .. +0x20: the grids' holders, a pointer each
constexpr u64 kManagerSurfaces = 0x40;     // +0x40 .. +0x48: the water surfaces' records, a pointer each
constexpr u64 kManagerScreenHolder = 0x60; // screen-space mode's holder
constexpr u64 kManagerNoHolder = 0x68;     // the holder that stands for none; without it the update does nothing
constexpr u64 kManagerOn = 0x78;           // byte; 0: the update does nothing
constexpr u64 kManagerScreenSpace = 0x79;  // byte
// A holder: +0 its grid, +0x18 a byte, set while it is in use.
constexpr u64 kHolderInUse = 0x18;
// A grid: its three height maps, and +0x30 a byte, its kind (3: the waves the
// characters make; the update turns only those round).
constexpr u64 kGridMap = 0x10, kGridLast = 0xd0, kGridBefore = 0xd8, kGridKind = 0x30;
// A water surface's record: its impulses, two vectors, each +8 its first, +0x10
// its end, +0x18 its room's end, +0x20 its heap.
constexpr u64 kSurfaceTaken = 0x98, kSurfaceQueued = 0xa0;
constexpr u64 kImpulseBytes = 0x20;

using WaterUpdate = GUEST_ABI u64 (*)(u64 manager);
using WaterTask = GUEST_ABI u64 (*)(u64 self, u64 context, u64 param);

void* g_game_water_update = nullptr;
void* g_game_water_task = nullptr;
bool g_gate_on = true;
// Our task has run. Before that the update holds no frame: the game's task
// would step the maps the update turned back.
std::atomic<bool> g_task_ours{false};

// A manager's gate (one in practice; a few kept). The scene renderer runs a
// manager's update and then its task, one at a time.
constexpr int kManagers = 4, kGrids = 32;
struct ManagerGate {
    std::atomic<u64> manager{0};
    WaterGate gate;
    bool decided = false;  // its update ran since its task last did
    bool waiting = false;  // a held frame's update handed impulses on that no task took
    WaterStep why = kWaterSkip;
    int turned = 0;        // the grids that update turned round, in its order
    u64 grids[kGrids];
    u64 maps[kGrids][3];   // their maps before it (for a mutant)
};
ManagerGate g_managers[kManagers];

std::atomic<u64> g_frames{0}, g_steps{0}, g_camera_steps{0}, g_held{0}, g_turned_back{0}, g_impulses_kept{0}, g_impulses_lost{0}, g_passed{0};

#ifndef DECOMP_WATER_GATE_TEST_GX
inline u64 gx_world() { return load<u64>(game_address(kGxWorld)); }
#else
u64 gx_world();
#endif

ManagerGate* gate_of(u64 manager, bool claim) {
    for (ManagerGate& m : g_managers) {
        u64 have = m.manager.load(std::memory_order_acquire);
        if (have == manager) return &m;
        if (!have && claim && (m.manager.compare_exchange_strong(have, manager, std::memory_order_acq_rel) || have == manager)) return &m;
    }
    return nullptr;
}

// sub_12bbda0's turn: the oldest map becomes the one to write.
void turn_round(u64 grid) {
    const u64 map = load<u64>(grid + kGridMap), last = load<u64>(grid + kGridLast), before = load<u64>(grid + kGridBefore);
    store<u64>(grid + kGridMap, before);
    store<u64>(grid + kGridLast, map);
    store<u64>(grid + kGridBefore, last);
}

// And back: the map a step wrote is +0x10 again.
void turn_back(u64 grid) {
    const u64 map = load<u64>(grid + kGridMap), last = load<u64>(grid + kGridLast), before = load<u64>(grid + kGridBefore);
    store<u64>(grid + kGridMap, last);
    store<u64>(grid + kGridLast, before);
    store<u64>(grid + kGridBefore, map);
}

// Whether the manager's update does anything: without these it returns at
// once.
bool update_runs(u64 manager) { return load<u8>(manager + kManagerOn) != 0 && load<u64>(manager + kManagerNoHolder) != 0; }

// The grids the update is about to turn round, in its order: the grid of
// each holder in use (but the one for none), then screen-space mode's - each
// one with its maps and of the kind that steps (sub_12bbda0's test). False
// when there are more than kGrids.
bool grids_to_turn(u64 manager, ManagerGate& m) {
    m.turned = 0;
    if (!update_runs(manager)) return true;
    auto add = [&](u64 grid) {
        if (!grid || !load<u64>(grid + kGridMap) || load<u8>(grid + kGridKind) != 3) return true;
        if (m.turned == kGrids) return false;
        m.grids[m.turned] = grid;
        m.maps[m.turned][0] = load<u64>(grid + kGridMap);
        m.maps[m.turned][1] = load<u64>(grid + kGridLast);
        m.maps[m.turned][2] = load<u64>(grid + kGridBefore);
        ++m.turned;
        return true;
    };
    const u64 none = load<u64>(manager + kManagerNoHolder);
    const u64 first = load<u64>(manager + kManagerHolders), end = load<u64>(manager + kManagerHolders + 8);
    for (u64 at = first; at < end; at += 8) {
        const u64 holder = load<u64>(at);
        if (!holder || !load<u8>(holder + kHolderInUse) || holder == none) continue;
        if (!add(load<u64>(holder))) return false;
    }
    const u64 screen = load<u64>(manager + kManagerScreenHolder);
    return !screen || add(load<u64>(screen));
}

// A held frame's impulses back in their queues, before the update hands the
// queues on: each water surface's +0x98 vector still holds what the held
// frame's update handed it; they go to the end of its +0xa0 queue, in the room
// the queue has (whatever the two vectors' heaps: the queue keeps its own
// room), or - the queue empty and without the room - the two vectors swap
// places, as the update's own swap does, when they come from one heap. Else
// they are dropped. The +0x98 vector is left empty either way. Adds to the
// counts kept and lost.
void requeue_impulses(u64 manager) {
    const u64 first = load<u64>(manager + kManagerSurfaces), end = load<u64>(manager + kManagerSurfaces + 8);
    for (u64 at = first; at < end; at += 8) {
        const u64 surface = load<u64>(at);
        if (!surface) continue;
        const u64 taken = load<u64>(surface + kSurfaceTaken), queued = load<u64>(surface + kSurfaceQueued);
        if (!taken || !queued || taken == queued) continue;
        const u64 t_first = load<u64>(taken + 8), t_end = load<u64>(taken + 0x10);
        if (t_end <= t_first) continue;
        const u64 bytes = t_end - t_first;
        const u64 q_first = load<u64>(queued + 8), q_end = load<u64>(queued + 0x10), q_room = load<u64>(queued + 0x18);
        if (q_end >= q_first && q_room >= q_end && q_room - q_end >= bytes && !MUTANT(13)) {
            std::memmove(reinterpret_cast<void*>(static_cast<std::uintptr_t>(q_end)), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(t_first)),
                         bytes);
            store<u64>(queued + 0x10, q_end + bytes);
            store<u64>(taken + 0x10, t_first);
            g_impulses_kept.fetch_add(bytes / kImpulseBytes, std::memory_order_relaxed);
        } else if (q_end == q_first && (load<u64>(taken + 0x20) == load<u64>(queued + 0x20) || MUTANT(13))) {
            for (u64 f = 8; f <= 0x18; f += 8) {
                const u64 a = load<u64>(taken + f);
                store<u64>(taken + f, load<u64>(queued + f));
                store<u64>(queued + f, a);
            }
            g_impulses_kept.fetch_add(bytes / kImpulseBytes, std::memory_order_relaxed);
        } else {
            store<u64>(taken + 0x10, t_first);
            g_impulses_lost.fetch_add(bytes / kImpulseBytes, std::memory_order_relaxed);
        }
    }
}

}  // namespace

DECOMP_LEAF u64 water_update(u64 manager) {
    const auto game = reinterpret_cast<WaterUpdate>(g_game_water_update);
    const u64 gx = gx_world();
    const bool ours = g_task_ours.load(std::memory_order_relaxed) || MUTANT(11);
    ManagerGate* m = g_gate_on && gx && manager && ours && !MUTANT(1) ? gate_of(manager, true) : nullptr;
    if (!m) return game(manager);
    if (m->waiting && !MUTANT(10)) requeue_impulses(manager);
    m->waiting = false;
    m->decided = true;
    m->turned = 0;
    m->why = MUTANT(3) ? kWaterTime : water_gate_time(m->gate, load<float>(gx + kGameTime));
    if (m->why != kWaterSkip) return game(manager);
    if (!grids_to_turn(manager, *m)) {
        // more grids than kept: this frame steps
        m->why = kWaterTime;
        m->turned = 0;
        return game(manager);
    }
    // the impulses it hands on wait where the task reads them, for this frame's camera step or the next update
    m->waiting = update_runs(manager) || MUTANT(12);
    const u64 r = game(manager);
    for (int i = m->turned - 1; i >= 0 && !MUTANT(8); --i) {
        if (MUTANT(9)) {  // the maps put back as they were
            store<u64>(m->grids[i] + kGridMap, m->maps[i][0]);
            store<u64>(m->grids[i] + kGridLast, m->maps[i][1]);
            store<u64>(m->grids[i] + kGridBefore, m->maps[i][2]);
        } else {
            turn_back(m->grids[i]);
        }
    }
    g_held.fetch_add(1, std::memory_order_relaxed);
    g_turned_back.fetch_add(static_cast<u64>(m->turned), std::memory_order_relaxed);
    return r;
}

DECOMP_LEAF u64 water_task(u64 self, u64 context, u64 param) {
    const auto game = reinterpret_cast<WaterTask>(g_game_water_task);
    g_task_ours.store(true, std::memory_order_relaxed);
    const u64 manager = param ? load<u64>(param) : 0;
    ManagerGate* m = g_gate_on && manager ? gate_of(manager, false) : nullptr;
    if (!m || !m->decided) {
        g_passed.fetch_add(1, std::memory_order_relaxed);
        return game(self, context, param);
    }
    m->decided = false;
    bool scroll = load<u8>(manager + kManagerScreenSpace) != 0 && !MUTANT(2);
    float x = 0.0f, z = 0.0f;
    const u64 camera = context ? load<u64>(context + kContextCamera) : 0;
    if (scroll && camera) {
        x = load<float>(camera + 0x30);
        z = load<float>(camera + 0x38);
    } else {
        scroll = false;
    }
    g_frames.fetch_add(1, std::memory_order_relaxed);
    const WaterStep why = water_gate_camera(m->gate, m->why, scroll, x, z);
    if (why == kWaterSkip) return 0;
    if (m->why == kWaterSkip) {
        // the camera crossed on a held frame: its grids as the update left them, and its impulses this step's
        for (int i = 0; i < m->turned && !MUTANT(14); ++i) turn_round(m->grids[i]);
        if (!MUTANT(15)) m->waiting = false;
    }
    m->turned = 0;
    if (why == kWaterCamera) g_camera_steps.fetch_add(1, std::memory_order_relaxed);
    g_steps.fetch_add(1, std::memory_order_relaxed);
    return game(self, context, param);
}

void decomp_water_gate_report() {
    if (decomp_placed("sub_2677780") != 1 || decomp_placed("sub_2677e50") != 1) return;
    const u64 frames = g_frames.load(), steps = g_steps.load();
    host_log("decomp: the water interaction's wave steps: %llu frames, %llu stepped (%llu of them as the camera crossed the grid's unit), "
             "%llu held between whole 1/30 s (their update's turn of the height maps undone on %llu grids; %llu impulses kept for the next "
             "step, %llu dropped for want of room), %llu task calls with no update before them the game's%s",
             static_cast<ull>(frames), static_cast<ull>(steps), static_cast<ull>(g_camera_steps.load()), static_cast<ull>(g_held.load()),
             static_cast<ull>(g_turned_back.load()), static_cast<ull>(g_impulses_kept.load()), static_cast<ull>(g_impulses_lost.load()),
             static_cast<ull>(g_passed.load()), g_gate_on ? "" : " (BBHOST_WATER_GATE=0: every frame the game's)");
}

namespace {
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x68; mov r13, rdi.
constexpr std::uint8_t kUpdateEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                         0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x68, 0x49, 0x89, 0xfd};
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x328.
constexpr std::uint8_t kTaskEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                       0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x28, 0x03, 0x00, 0x00};
}  // namespace

void decomp_water_gate_add() {
    const char* e = std::getenv("BBHOST_WATER_GATE");
    g_gate_on = !(e && e[0] == '0');
    DecompFunction update{"sub_2677780", "gx", kWaterUpdate, kUpdateEntry, sizeof(kUpdateEntry), reinterpret_cast<void*>(&water_update),
                          DecompKind::Leaf, &g_game_water_update};
    decomp_add(update);
    DecompFunction task{"sub_2677e50", "gx", kWaterTask, kTaskEntry, sizeof(kTaskEntry), reinterpret_cast<void*>(&water_task),
                        DecompKind::Leaf, &g_game_water_task};
    task.report = &decomp_water_gate_report;
    decomp_add(task);
}
