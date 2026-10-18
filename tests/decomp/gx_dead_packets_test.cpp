// The Gnm packet writers nothing reads (decomp/gx/dead_packets.cpp) beside the
// game's own (tests/decomp/eboot_kit.h). For each of the 21, on generated
// command buffers and arguments - room to spare, exactly enough, one dword
// short, none, a refill that gives a chunk and one that gives nothing, every
// argument's bits (floats as raw bit patterns: NaNs, denormals, infinities),
// the pixel-shader input table's counts past its jump table and its 32-bit
// wrap - ours writing the game's packet must leave everything as the game's
// leaves it: the return value, the command buffer object, both chunks' bytes
// and the refill's calls with their arguments. Then the modes bbhost runs
// them in: writing nothing (no room asked for, nothing written, the write
// pointer returned) and the poisoned address; the modes each writer is
// configured with (twenty dead, prepareFlip's reservation writing its NOP,
// BBHOST_CP_DEAD_KEEP's places writing). And the entries the decomp list
// places them at hold the instructions written against. Skips without the
// 1.09 eboot.
#include "decomp/gx/dead_packets.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x6e75fe11);

// A command buffer as Gnm's are: an object {+0x8 the chunk's end, +0x10 the
// write pointer, +0x18 the refill, +0x20 its argument} over a chunk, and the
// chunk the refill hands over.
struct alignas(64) Memory {
    u8 object[0x40];
    u8 chunk[4096];
    u8 next[4096];
};
Memory g_mem, g_saved, g_game;

struct RefillCall {
    u64 cb;
    u32 dwords;
    u64 arg, room;
    bool operator==(const RefillCall&) const = default;
};
std::vector<RefillCall> g_calls;
bool g_refill_gives = true;
u64 g_refill_result = 0;

u64 cb() { return reinterpret_cast<u64>(g_mem.object); }

GUEST_ABI u64 stub_refill(u64 c, u32 dwords, u64 arg, u64 room) {
    g_calls.push_back({c, dwords, arg, room});
    if (!g_refill_gives) return g_refill_result;
    const u64 next = reinterpret_cast<u64>(g_mem.next);
    store<u64>(c + 0x8, next + sizeof g_mem.next);
    store<u64>(c + 0x10, next);
    return g_refill_result;
}

// The packet's dwords as the game reserves them (the input table's: count + 2).
u32 need_of(std::size_t i, const Args& a) { return room_for(i, a); }

u32 random_u32() {
    switch (g_rng.range(0, 9)) {
        case 0: return 0;
        case 1: return 0xffffffffu;
        case 2: return static_cast<u32>(g_rng.range(0, 63));
        case 3: return 0x80000000u | static_cast<u32>(g_rng.range(0, 7));
        default: return g_rng.u32();
    }
}

float random_float() {
    if (g_rng.chance(0.5)) return std::bit_cast<float>(g_rng.u32());
    static const u32 kSpecial[] = {0x00000000, 0x80000000, 0x3f800000, 0x00000001, 0x807fffff, 0x7f800000, 0xff800000,
                                   0x7fc00000, 0x7fa00001, 0xffc12345, 0x7f7fffff, 0x3f000000, 0x40800000};
    return std::bit_cast<float>(kSpecial[g_rng.range(0, static_cast<int>(std::size(kSpecial)) - 1)]);
}

alignas(16) u8 g_data[256];  // what the pointer arguments point at

Args random_args(std::size_t i) {
    Args a{};
    a.r[0] = cb();
    for (int k = 1; k < 6; ++k) a.r[k] = (static_cast<u64>(g_rng.u32()) << 32) | random_u32();
    a.s0 = (static_cast<u64>(g_rng.u32()) << 32) | random_u32();
    for (float& x : a.x) x = random_float();
    for (u8& b : g_data) b = static_cast<u8>(g_rng.u32());
    switch (i) {
        case kSampleLocations:
        case kSampleLocationsAndCentroid:
        case kPsInputCntl:
            a.r[1] = reinterpret_cast<u64>(g_data) + static_cast<u64>(g_rng.range(0, 7));  // unaligned too
            break;
        default: break;
    }
    if (i == kPsInputCntl) {
        static const u32 kCounts[] = {0, 1, 2, 31, 32, 33, 40, 0x3fff, 0x4000, 0xfffffffeu, 0xffffffffu};
        const u32 c = g_rng.chance(0.7) ? static_cast<u32>(g_rng.range(0, 34))
                                        : kCounts[g_rng.range(0, static_cast<int>(std::size(kCounts)) - 1)];
        a.r[2] = (static_cast<u64>(g_rng.u32()) << 32) | c;  // garbage above the u32
    }
    if (i == kAaConfig) {
        // An index into the game's byte table (0x4b2bf4c), as far as the image goes.
        const u32 n = g_rng.chance(0.8) ? static_cast<u32>(g_rng.range(0, 15)) : static_cast<u32>(g_rng.range(0, 0xffff));
        a.r[1] = (static_cast<u64>(g_rng.u32()) << 32) | n;
    }
    return a;
}

// The command buffer for one case: random bytes everywhere, the write pointer
// somewhere in the chunk with room for what any writer writes, and the room
// around the packet's size - or none - with up to three bytes over a dword.
void random_buffer(u32 need) {
    for (u8& b : g_mem.object) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.chunk) b = static_cast<u8>(g_rng.u32());
    for (u8& b : g_mem.next) b = static_cast<u8>(g_rng.u32());
    const u32 most = need > 34 ? 34 : need;  // the most a writer writes past the write pointer, in dwords (64 for the flip room)
    const u64 span = (need == 64 ? 64 : most + 2) * 4ull;
    const u64 off = static_cast<u64>(g_rng.range(0, static_cast<int>(sizeof g_mem.chunk - span))) & ~3ull;
    const u64 wp = reinterpret_cast<u64>(g_mem.chunk) + off;
    u64 room;
    switch (g_rng.range(0, 6)) {
        case 0: room = 0; break;
        case 1: room = need ? need - 1 : 0; break;
        case 2: room = need; break;
        case 3: room = static_cast<u64>(need) + 1; break;
        case 4: room = static_cast<u64>(g_rng.range(0, 300)); break;
        case 5: room = static_cast<u64>(g_rng.u32()) + 0x100000000ull * g_rng.range(0, 1); break;  // past 32 bits
        default:  // past 32 bits with the low 32 short: the game asks the refill, and passes all 64
            room = 0x100000000ull * static_cast<u64>(g_rng.range(1, 3)) + static_cast<u64>(g_rng.range(0, need > 1 ? static_cast<int>(need) - 1 : 0));
            break;
    }
    store<u64>(cb() + 0x8, wp + room * 4 + static_cast<u64>(g_rng.range(0, 3)));
    store<u64>(cb() + 0x10, wp);
    store<u64>(cb() + 0x18, reinterpret_cast<u64>(&stub_refill));
    store<u64>(cb() + 0x20, (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32());
    g_refill_gives = g_rng.chance(0.75);
    g_refill_result = g_refill_gives ? ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu) | 1u)
                                     : ((static_cast<u64>(g_rng.u32()) << 32) | (g_rng.u32() & ~0xffu));
}

// The game's function (`game`) or ours, with the arguments in their registers.
u64 call(std::size_t i, bool game, const Args& a) {
    const u64 c = a.r[0];
    const auto d = [&](int k) { return static_cast<u32>(a.r[k]); };
    const u64 bn = kTable[i].bn;
#define CALL(F, ours, ...) (game ? eboot_kit::fn<u64(GUEST_ABI*) F>(bn)(__VA_ARGS__) : ours(__VA_ARGS__))
    switch (i) {
        case kAcquireMemRange:
            return CALL((u64, u32, u32, u32, u32, u32, u32), gnm_acquire_mem_range, c, d(1), d(2), d(3), d(4), d(5), static_cast<u32>(a.s0));
        case kAcquireMem: return CALL((u64, u32, u32, u32), dcb_acquire_mem, c, d(1), d(2), d(3));
        case kComputeAcquireMemRange:
            return CALL((u64, u32, u32, u32, u32, u32), gnm_compute_acquire_mem_range, c, d(1), d(2), d(3), d(4), d(5));
        case kComputeAcquireMem: return CALL((u64, u32, u32), gnm_compute_acquire_mem, c, d(1), d(2));
        case kSampleLocations: return CALL((u64, u64), gnm_set_aa_sample_locations, c, a.r[1]);
        case kSampleLocationsAndCentroid: return CALL((u64, u64), gnm_set_aa_sample_locations_and_centroid, c, a.r[1]);
        case kCentroidPriority: return CALL((u64, u64), gnm_set_centroid_priority, c, a.r[1]);
        case kAaConfig: return CALL((u64, u32), gnm_set_aa_config, c, d(1));
        case kDepthStencilDisable: return CALL((u64), gnm_set_depth_stencil_disable, c);
        case kBlendColor: return CALL((u64, float, float, float, float), gnm_set_blend_color, c, a.x[0], a.x[1], a.x[2], a.x[3]);
        case kAlphaToMask: return CALL((u64, u32), gnm_set_alpha_to_mask, c, d(1));
        case kTmpringSize: return CALL((u64, u32, u32), gnm_set_spi_tmpring_size, c, d(1), d(2));
        case kScreenOffset: return CALL((u64, u32, u32), gnm_set_hardware_screen_offset, c, d(1), d(2));
        case kGuardBandClip: return CALL((u64, float, float), gnm_set_guard_band_clip_adjust, c, a.x[0], a.x[1]);
        case kGuardBandDiscard: return CALL((u64, float, float), gnm_set_guard_band_discard_adjust, c, a.x[0], a.x[1]);
        case kInstanceStepRate: return CALL((u64, u32, u32), gnm_set_instance_step_rate, c, d(1), d(2));
        case kPsInputCntl: return CALL((u64, u64, u32), gnm_set_ps_input_cntl, c, a.r[1], d(2));
        case kVtxCntl: return CALL((u64, u32, u32, u32), gnm_set_su_vtx_cntl, c, d(1), d(2), d(3));
        case kLineCntl: return CALL((u64, u32), gnm_set_su_line_cntl, c, d(1));
        case kShaderPrefetch: return CALL((u64, u64, u32), gnm_draw_shader_prefetch_dma, c, a.r[1], d(2));
        case kFlipRoom: return CALL((u64), gnm_reserve_flip_packets, c);
        default: return 0;
    }
#undef CALL
}

int g_failed = 0;
void fail(std::size_t i, int trial, const char* what) {
    if (++g_failed <= 40) std::printf("FAIL: %s, case %d: %s\n", kTable[i].name, trial, what);
}

int run(std::size_t i, int cases) {
    int compared = 0, refilled = 0, refused = 0;
    for (int t = 0; t < cases; ++t) {
        const Args a = random_args(i);
        random_buffer(need_of(i, a));
        g_saved = g_mem;
        const u64 wp0 = load<u64>(cb() + 0x10);

        // The game's.
        g_calls.clear();
        const u64 game_ret = call(i, true, a);
        g_game = g_mem;
        const std::vector<RefillCall> game_calls = g_calls;
        if (!game_calls.empty()) ++(g_refill_gives ? refilled : refused);

        // Ours, writing the game's packet.
        g_mem = g_saved;
        g_calls.clear();
        g_mode[i] = static_cast<u8>(GnmDeadMode::kWrite);
        const u64 ours_ret = call(i, false, a);
        if (ours_ret != game_ret) fail(i, t, "the return value");
        if (std::memcmp(g_mem.object, g_game.object, sizeof g_mem.object) != 0) fail(i, t, "the command buffer object");
        if (std::memcmp(g_mem.chunk, g_game.chunk, sizeof g_mem.chunk) != 0) fail(i, t, "the chunk's bytes");
        if (std::memcmp(g_mem.next, g_game.next, sizeof g_mem.next) != 0) fail(i, t, "the refill's chunk's bytes");
        if (g_calls != game_calls) fail(i, t, "the refill's calls");
        ++compared;

        // Ours as bbhost runs it: nothing, the write pointer back.
        for (const GnmDeadMode m : {GnmDeadMode::kNothing, GnmDeadMode::kPoison}) {
            g_mem = g_saved;
            g_calls.clear();
            g_mode[i] = static_cast<u8>(m);
            const u64 r = call(i, false, a);
            if (r != (m == GnmDeadMode::kNothing ? wp0 : kPoison)) fail(i, t, "the address returned writing nothing");
            if (std::memcmp(&g_mem, &g_saved, sizeof g_mem) != 0) fail(i, t, "memory written writing nothing");
            if (!g_calls.empty()) fail(i, t, "a refill writing nothing");
        }
        g_mode[i] = static_cast<u8>(GnmDeadMode::kWrite);
    }
    std::printf("  %-42s %d cases against the game's (%d with a refill, %d the refill refused)\n", kTable[i].name, compared, refilled,
                refused);
    return compared;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_dead_packets_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 20000;

    // The decomp list places each at its entry only when these bytes are there.
    for (std::size_t i = 0; i < kWriters; ++i) {
        const u8* at = eboot_kit::file_at("gx_dead_packets_test", kTable[i].bn);
        if (!at || std::memcmp(at, kTable[i].bytes, kTable[i].n) != 0) fail(i, -1, "the entry's instructions");
        if (kTable[i].n < 14) fail(i, -1, "an entry too short for a census hook");
    }
    // The modes as hle_gx_trace_arm sets them.
    // The modes as hle_gx_trace_arm sets them: twenty dead, prepareFlip's
    // reservation writing its NOP whatever the mode, the context-register
    // writers by their own mode; BBHOST_CP_DEAD_KEEP's places writing.
    std::size_t dead = 0;
    for (std::size_t i = 0; i < kWriters; ++i) dead += kTable[i].dead;
    if (dead != 20 || kTable[kFlipRoom].dead) fail(kFlipRoom, -1, "twenty dead writers, the flip's reservation kept");
    for (const auto& [others, context] : {std::pair{GnmDeadMode::kNothing, GnmDeadMode::kWrite}, std::pair{GnmDeadMode::kPoison, GnmDeadMode::kPoison},
                                          std::pair{GnmDeadMode::kNothing, GnmDeadMode::kNothing}}) {
        gnm_dead_writers_configure(others, context);
        for (std::size_t i = 0; i < kWriters; ++i) {
            const GnmDeadMode want = !kTable[i].dead ? GnmDeadMode::kWrite : kTable[i].context ? context : others;
            if (g_mode[i] != static_cast<u8>(want) || gnm_dead_writer_mode(i) != want) fail(i, -1, "configure");
        }
    }
    struct KeepCase {
        const char* list;
        std::uint64_t mask;
    };
    for (const KeepCase& k : {KeepCase{nullptr, 0}, KeepCase{"", 0}, KeepCase{"0,1,2,3", 0xf}, KeepCase{"20", 1ull << 20},
                              KeepCase{"4,5,6,7,8,9,10,11,12,13,14,15,16,17,18", 0x7fff0}, KeepCase{"19,x,3", 1ull << 19},
                              KeepCase{"1,,2", 0x2}, KeepCase{"63,64,-1", 1ull << 63}}) {
        if (gnm_dead_keep_mask(k.list) != k.mask) fail(0, -1, "BBHOST_CP_DEAD_KEEP's list");
    }
    gnm_dead_writers_configure(GnmDeadMode::kNothing, GnmDeadMode::kNothing, gnm_dead_keep_mask("0,3,20"));
    for (std::size_t i = 0; i < kWriters; ++i) {
        const bool writes = i == 0 || i == 3 || i == kFlipRoom;
        if (gnm_dead_writer_mode(i) != (writes ? GnmDeadMode::kWrite : GnmDeadMode::kNothing)) fail(i, -1, "configure with places kept");
    }
    gnm_dead_writers_configure(GnmDeadMode::kWrite, GnmDeadMode::kWrite);

    long total = 0;
    for (std::size_t i = 0; i < kWriters; ++i) total += run(i, cases);
    std::printf("gx_dead_packets: %ld cases over %zu writers: %s\n", total, static_cast<std::size_t>(kWriters), g_failed ? "FAILED" : "ok");
    return g_failed ? 1 : 0;
}
