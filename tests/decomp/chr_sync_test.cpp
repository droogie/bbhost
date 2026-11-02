// The local player's character sync by game time (decomp/chr/chr_sync.cpp)
// beside the game's own, sub_1929470, in the loaded eboot. What the sync
// reaches - the character's network module (sub_1e1e270), its packing of
// state (sub_190a8e0, sub_1e73390), the send (sub_1e72220), the setting
// ChrSync.FrameSkipCount (std::string's initialize and sub_24eba00) and the
// character's three virtuals - are recorders or stand-ins here, and the
// character's modules test memory.
//   - At the game's own 30 (the clock unpublished): on generated states -
//     countdowns from -2 to 7, the packet there or not, the network module
//     there or not, the setting unset (the default 5) or 0 to 8 - ours is the
//     game's sync: every byte, every call and its arguments the same.
//   - At frames of other lengths, a frame that begins a 30 fps frame or not:
//     ours is the game's sync with the countdown held where none begins (2
//     for the call, as it was after).
//   - Played over 10 game seconds at 30 and by ours at 60, 90, 120, 144, 165,
//     240, 360 and 1000 fps and under a jitter of 30-240: the packets sent at
//     30's game seconds (a frame early at most where the frames do not divide
//     1/30), 6 a second with the default and 30 with a setting of 0 or 1, the
//     module handed the packet every frame; the game's own sync unconverted
//     at 120 sends 4 times as often.
// Each of the source's mutants caught; the function's bytes the ones ours
// runs around; and the list takes it where the frame clock is published - by
// default at a converted rate, at every rate with the clock on - unless
// BBHOST_CHR_SYNC_STEP=0. Skips without the 1.09 eboot.
#define DECOMP_CHR_SYNC_MUTANTS
int g_chr_sync_mutant = 0;
#include "decomp/chr/chr_sync.cpp"

#include "eboot_kit.h"

#include <immintrin.h>

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

eboot_kit::Rng g_rng(0x1929470);

constexpr u64 kModuleSync = 0x1e1e270, kPackState = 0x190a8e0, kPackAction = 0x1e73390, kSend = 0x1e72220;
constexpr u64 kStringInit = 0x2bc0e00, kSetting = 0x24eba00;

// What the sync reached, in order, folded.
struct Seen {
    int module = 0, module_sends = 0, pack = 0, action = 0, sends = 0, inits = 0, settings = 0;
    u64 sum = 0;
    bool operator==(const Seen&) const = default;
};
Seen g_seen;
int g_setting = -1;  // ChrSync.FrameSkipCount: -1 unset, the default stands
s32 g_hp_action = 0;  // what the character's +0x480 answers

GUEST_ABI u64 module_recorder(u64 module, u64 send, u64 packet) {
    ++g_seen.module;
    if (send & 0xff) ++g_seen.module_sends;
    g_seen.sum = g_seen.sum * 31 + module + (send & 0xff) * 3 + packet * 5;
    return 0;
}
GUEST_ABI u64 pack_recorder(u64 a) {
    ++g_seen.pack;
    g_seen.sum = g_seen.sum * 31 + a;
    return 0x1234567u;
}
GUEST_ABI u64 action_recorder(u64 packet, u64 action, u64 v, u64 w) {
    ++g_seen.action;
    g_seen.sum = g_seen.sum * 31 + packet + (action & 0xffffffff) * 3 + load<u64>(v) * 5 + load<u32>(w) * 7;
    return 0;
}
GUEST_ABI u64 send_recorder(u64 packet) {
    ++g_seen.sends;
    g_seen.sum = g_seen.sum * 31 + packet + load<u64>(packet) + load<u64>(packet + 0x10) + load<u64>(packet + 0x28);
    return 1;
}
// std::string's initialize, for the setting's name: an empty short string
// (its capacity at +0x20 below 8, so the game's sync frees nothing).
GUEST_ABI u64 string_init(u64 s, u64 text) {
    ++g_seen.inits;
    std::memset(reinterpret_cast<void*>(s), 0, 0x28);
    store<u64>(s + 0x20, 7);
    g_seen.sum = g_seen.sum * 31 + (text ? 1 : 0);
    return s;
}
GUEST_ABI u64 setting_lookup(u64 props, u64 name, u64 fallback) {
    ++g_seen.settings;
    g_seen.sum = g_seen.sum * 31 + (fallback & 0xffffffff);
    return g_setting < 0 ? (fallback & 0xffffffff) : static_cast<u64>(g_setting);
}
GUEST_ABI u64 chr_480(u64) { return static_cast<u64>(static_cast<s64>(g_hp_action)); }
GUEST_ABI __m128 chr_490(u64) { return _mm_setr_ps(1.5f, -2.0f, 3.25f, 0.0f); }
GUEST_ABI u64 chr_498(u64) { return 0x77; }

struct alignas(16) Memory {
    u8 sync[0x120];
    u8 packet[0x200];
    u8 owner[0x40];      // the sync's second argument: +8 the character, +0x38 what sub_190a8e0 is handed
    u8 chr[0x400];       // +0 its vtable, +0x3b0 its modules
    u8 modules[0x80];    // +0x20 the HP's, +0x38 the network module, +0x68 the position's
    u8 hp[0x200];        // +0xf8 HP, +0xfc its most
    u8 position[0x200];
    u8 module[0x40];
    alignas(16) u8 vtable[0x500];
};
Memory g_mem, g_start, g_after_game;

u64 self() { return reinterpret_cast<u64>(g_mem.sync); }
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

Sync g_game = nullptr;

int g_failed = 0;
bool g_quiet = false;
void fail(int c, const char* what) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: case %d: %s\n", c, what);
}

void generate() {
    for (u8& b : g_start.sync) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_start.packet) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_start.position) b = static_cast<u8>(g_rng.u32());
    std::memset(g_start.owner, 0, sizeof g_start.owner);
    std::memset(g_start.chr, 0, sizeof g_start.chr);
    std::memset(g_start.modules, 0, sizeof g_start.modules);
    std::memset(g_start.hp, 0, sizeof g_start.hp);
    std::memset(g_start.vtable, 0, sizeof g_start.vtable);
    put<u64>(g_start.vtable, 0x480, reinterpret_cast<u64>(&chr_480));
    put<u64>(g_start.vtable, 0x490, reinterpret_cast<u64>(&chr_490));
    put<u64>(g_start.vtable, 0x498, reinterpret_cast<u64>(&chr_498));
    put<s32>(g_start.sync, kCountdown, g_rng.range(-2, 7));
    put<u64>(g_start.sync, 0xc0, g_rng.chance(0.85) ? reinterpret_cast<u64>(g_mem.packet) : 0);
    put<u64>(g_start.owner, 8, reinterpret_cast<u64>(g_mem.chr));
    put<u64>(g_start.owner, 0x38, g_rng.next());
    put<u64>(g_start.chr, 0, reinterpret_cast<u64>(g_mem.vtable));
    put<u64>(g_start.chr, 0x3b0, reinterpret_cast<u64>(g_mem.modules));
    put<u64>(g_start.modules, 0x20, reinterpret_cast<u64>(g_mem.hp));
    put<u64>(g_start.modules, 0x38, g_rng.chance(0.8) ? reinterpret_cast<u64>(g_mem.module) : 0);
    put<u64>(g_start.modules, 0x68, reinterpret_cast<u64>(g_mem.position));
    put<s32>(g_start.hp, 0xf8, g_rng.range(-10, 70000));
    put<s32>(g_start.hp, 0xfc, g_rng.range(1, 70000));
    g_setting = g_rng.chance(0.6) ? -1 : g_rng.range(0, 8);
    g_hp_action = g_rng.chance(0.5) ? -1 : g_rng.range(0, 1000);
}

void both(Seen& game, Seen& ours) {
    g_mem = g_start;
    g_seen = Seen{};
    g_game(self(), owner());
    game = g_seen;
    g_after_game = g_mem;
    g_mem = g_start;
    g_seen = Seen{};
    chr_sync_update(self(), owner());
    ours = g_seen;
}

void clock_at(float dt, int ticks) {
    frame_clock::g_dt.store(dt);
    frame_clock::g_frames30.store(frame_clock::frames30_of(dt));
    frame_clock::g_ticks30.store(ticks);
}
void clock_unpublished() { clock_at(frame_clock::kThirtieth, 1); }

long at_thirty(int cases) {
    clock_unpublished();
    long n = 0;
    for (int c = 0; c < cases; ++c, ++n) {
        generate();
        Seen game, ours;
        both(game, ours);
        if (!(game == ours)) fail(c, "at 30: what the sync reached");
        if (std::memcmp(&g_mem, &g_after_game, sizeof g_mem) != 0) fail(c, "at 30: what the sync left");
    }
    return n;
}

long converted(int cases) {
    long n = 0;
    for (int c = 0; c < cases; ++c, ++n) {
        const int ticks = g_rng.chance(0.5) ? 1 : 0;
        clock_at(g_rng.uniform(0.001f, frame_clock::kThirtieth), ticks);
        generate();
        // The game's sync with the countdown held where no 30 fps frame begins.
        g_mem = g_start;
        const s32 countdown = get<s32>(g_start.sync, kCountdown);
        if (ticks == 0) put<s32>(g_mem.sync, kCountdown, 2);
        g_seen = Seen{};
        g_game(self(), owner());
        if (ticks == 0) put<s32>(g_mem.sync, kCountdown, countdown);
        const Memory want = g_mem;
        const Seen want_seen = g_seen;
        g_mem = g_start;
        g_seen = Seen{};
        chr_sync_update(self(), owner());
        if (!(g_seen == want_seen)) fail(c, "converted: what the sync reached");
        if (std::memcmp(&g_mem, &want, sizeof g_mem) != 0) fail(c, "converted: what the sync left");
        if (ticks == 0 && (g_seen.sends || g_seen.module_sends || get<s32>(g_mem.sync, kCountdown) != countdown))
            fail(c, "a frame that begins no 30 fps frame sent, or moved the countdown");
        if (g_seen.module != (get<u64>(g_start.modules, 0x38) ? 1 : 0)) fail(c, "the module not handed the packet this frame");
    }
    clock_unpublished();
    return n;
}

// Played: the game seconds at the start of each frame that sent a packet.
template <class Step>
std::vector<double> play(Step step, bool ours, int frames, int setting, int* module_calls) {
    std::memset(&g_mem, 0, sizeof g_mem);
    g_start = g_mem;
    g_rng.specials = false;
    put<u64>(g_mem.vtable, 0x480, reinterpret_cast<u64>(&chr_480));
    put<u64>(g_mem.vtable, 0x490, reinterpret_cast<u64>(&chr_490));
    put<u64>(g_mem.vtable, 0x498, reinterpret_cast<u64>(&chr_498));
    put<u64>(g_mem.sync, 0xc0, reinterpret_cast<u64>(g_mem.packet));
    put<u64>(g_mem.owner, 8, reinterpret_cast<u64>(g_mem.chr));
    put<u64>(g_mem.chr, 0, reinterpret_cast<u64>(g_mem.vtable));
    put<u64>(g_mem.chr, 0x3b0, reinterpret_cast<u64>(g_mem.modules));
    put<u64>(g_mem.modules, 0x20, reinterpret_cast<u64>(g_mem.hp));
    put<u64>(g_mem.modules, 0x38, reinterpret_cast<u64>(g_mem.module));
    put<u64>(g_mem.modules, 0x68, reinterpret_cast<u64>(g_mem.position));
    put<s32>(g_mem.hp, 0xfc, 100);
    g_setting = setting;
    std::vector<double> sends;
    double t = 0.0;
    frame_clock::g_tick_carry = 0.0;
    g_seen = Seen{};
    for (int i = 0; i < frames; ++i) {
        const float dt = step(i);
        if (ours) frame_clock::publish(dt);
        const int before = g_seen.sends;
        if (ours) chr_sync_update(self(), owner());
        else g_game(self(), owner());
        if (g_seen.sends != before) sends.push_back(t);
        t += static_cast<double>(dt);
    }
    *module_calls = g_seen.module;
    clock_unpublished();
    frame_clock::g_tick_carry = 0.0;
    return sends;
}

struct Rate {
    const char* name;
    double fps;  // 0: jitter
};

void played(int& checked) {
    for (const int setting : {-1, 0, 1, 3}) {
        const int per_second = setting < 0 ? 6 : 30 / std::max(setting, 1);
        int calls = 0;
        const std::vector<double> ref = play([](int) { return frame_clock::kThirtieth; }, false, 300, setting, &calls);
        if (static_cast<int>(ref.size()) != 10 * per_second) fail(-2, "the game's sync at 30 did not send at its rate");
        for (const Rate r : {Rate{"60", 60}, Rate{"90", 90}, Rate{"120", 120}, Rate{"144", 144}, Rate{"165", 165}, Rate{"240", 240},
                             Rate{"360", 360}, Rate{"1000", 1000}, Rate{"jitter 30-240", 0}}) {
            eboot_kit::Rng jit(0x5eed);
            const float fixed = r.fps > 0 ? static_cast<float>(1.0 / r.fps) : 0.0f;
            // 10 game seconds of frames (under jitter, to a little past them).
            std::vector<float> steps;
            double total = 0.0;
            while (total < 10.0 - 1e-6) {
                const float s = r.fps > 0 ? fixed : jit.uniform(1.0f / 240.0f, frame_clock::kThirtieth);
                steps.push_back(s);
                total += static_cast<double>(s);
            }
            int module_calls = 0;
            const std::vector<double> run = play([&](int i) { return steps[static_cast<std::size_t>(i)]; }, true, static_cast<int>(steps.size()),
                                                 setting, &module_calls);
            ++checked;
            const double frame = r.fps > 0 ? 1.0 / r.fps : 1.0 / 30.0;
            const bool count_ok = std::abs(static_cast<int>(run.size()) - static_cast<int>(ref.size())) <= 1;
            double worst = 0.0;
            for (std::size_t i = 0; i < std::min(run.size(), ref.size()); ++i) worst = std::max(worst, std::fabs(run[i] - ref[i]));
            if (!count_ok || worst > frame + 1e-6) {
                if (!g_quiet) std::printf("FAIL: %s, setting %d: %zu packets (30: %zu), the latest %.4f s from 30's\n", r.name, setting, run.size(), ref.size(), worst);
                ++g_failed;
            }
            if (module_calls != static_cast<int>(steps.size())) {
                if (!g_quiet) std::printf("FAIL: %s, setting %d: the module handed the packet %d times in %zu frames\n", r.name, setting, module_calls, steps.size());
                ++g_failed;
            }
            if (!g_quiet && setting < 0)
                std::printf("  %-14s %zu packets in 10 game seconds (30: %zu), the latest %.4f s from 30's; the module handed the packet "
                            "every frame (%d)\n",
                            r.name, run.size(), ref.size(), worst, module_calls);
        }
    }
    int calls = 0;
    const std::vector<double> fast = play([](int) { return 1.0f / 120.0f; }, false, 1200, -1, &calls);
    if (fast.size() != 240) fail(-3, "the game's sync unconverted at 120 did not send 24 a second");
    else if (!g_quiet)
        std::printf("  the game's own sync unconverted at 120: %zu packets in 10 game seconds (24 a second, 4 times 30's)\n", fast.size());
}

bool caught(int mutant) {
    const int before = g_failed;
    g_chr_sync_mutant = mutant;
    g_quiet = true;
    at_thirty(1000);
    converted(2000);
    int checked = 0;
    played(checked);
    g_quiet = false;
    g_chr_sync_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("chr_sync_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 100000;
    const u8* at_file = eboot_kit::file_at("chr_sync_test", kSync);
    if (!at_file || std::memcmp(at_file, kEntry, sizeof kEntry) != 0) fail(-1, "the entry's instructions");
    if (!at_file || !chr_sync_body_ok(at_file)) fail(-1, "the sync's bytes are not the ones ours runs around");
    if (load<u64>(0x5770430 + 0x50) != kSync) fail(-1, "PadManipulator's vtable: +0x50 the sync");
    eboot_kit::stub(kModuleSync, reinterpret_cast<const void*>(&module_recorder));
    eboot_kit::stub(kPackState, reinterpret_cast<const void*>(&pack_recorder));
    eboot_kit::stub(kPackAction, reinterpret_cast<const void*>(&action_recorder));
    eboot_kit::stub(kSend, reinterpret_cast<const void*>(&send_recorder));
    eboot_kit::stub(kStringInit, reinterpret_cast<const void*>(&string_init));
    eboot_kit::stub(kSetting, reinterpret_cast<const void*>(&setting_lookup));
    g_game = eboot_kit::fn<Sync>(kSync);
    // On the list where the clock is published - by default at a converted
    // rate (publishing), at every rate with the clock on - unless its own
    // switch is 0; the game's own at 30 with the clock off.
    for (int clock = 0; clock < 2; ++clock) {
        for (int pub = 0; pub < 2; ++pub) {
            for (int sw = 0; sw < 3; ++sw) {
                frame_clock::set_on(clock != 0);
                frame_clock::set_publishing(pub != 0);
                if (sw == 0) unsetenv("BBHOST_CHR_SYNC_STEP");
                else setenv("BBHOST_CHR_SYNC_STEP", sw == 1 ? "0" : "1", 1);
                g_added = 0;
                decomp_chr_sync_add();
                const bool want = (clock || pub) && sw != 1;
                if (g_added != (want ? 1 : 0) || (want && std::strcmp(g_added_name, "sub_1929470") != 0))
                    fail(-1, "placed where it should not be, or not where it should");
            }
        }
    }
    frame_clock::set_on(false);
    frame_clock::set_publishing(false);
    unsetenv("BBHOST_CHR_SYNC_STEP");
    g_game_sync = reinterpret_cast<void*>(g_game);

    const long thirty = at_thirty(cases);
    const long conv = converted(cases);
    int checked = 0;
    std::printf("chr_sync_test: played over 10 game seconds:\n");
    played(checked);
    if (!g_held.load() || !g_packets.load()) fail(-1, "a kind of sync never ran (held, a packet)");
    for (const int mutant : {1, 2, 3}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("chr_sync_test: %d failures\n", g_failed);
        return 1;
    }
    std::printf("chr_sync_test: ok - %ld syncs at 30 the game's to the byte, %ld at converted frames as specified, %d runs over 10 "
                "game seconds at 30's packets, 3 mutants caught\n",
                thirty, conv, checked);
    return 0;
}
