// GX's command lists (decomp/gx/command_list.cpp) beside the game's own
// (tests/decomp/eboot_kit.h): FinishCommandList's wrapper and body,
// ExecuteCommandList's and ExecuteCommandLists' wrappers and bodies, the
// close of a command buffer's segment, and the frame's last segment going out
// with the flip.
//
// Every callee is a recorder (eboot_kit::stub - the game's code and ours both
// reach it), and so are the allocator's free (+0x70 of a vtable the test
// makes) and sceGnmSubmitCommandBuffers' import entry, which keeps the
// arrays' contents. On generated contexts - the pending-flags byte set or
// not, either record set selected for colour and for depth - lists of up to
// four pages of up to seven spans with an allocator or none, batches of up
// to five lists with gaps, span appends and recording failures, and compute
// rings with their counters on the last value, past it, or wrapping, flips
// with room or without, refills that give some or none - ours
// must make the game's calls with the game's arguments, return what it
// returns and leave the memory as it leaves it. The game's one-line helpers
// run as themselves. Six deliberate mistakes (MUTANT) must each be caught,
// and the entries the decomp list places them at hold the instructions
// written against. Skips without the 1.09 eboot.
#define DECOMP_GX_COMMAND_LIST_MUTANTS
int g_list_mutant = 0;
#include "decomp/gx/command_list.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

#include <sys/mman.h>

namespace {

eboot_kit::Rng g_rng(0x1157);

struct Call {
    int who;
    u64 a[6];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;

struct Callee {
    u64 bn;
    int args;  // how many to compare
    const char* name;
};
// The order is the recorders' ids.
const Callee kCallees[] = {
    {kFinishBody, 4, "finish_body"},
    {kExecuteBody, 4, "execute_body"},
    {kExecuteListsBody, 5, "execute_lists_body"},
    {kCommandBufferFlush, 1, "command_buffer_flush"},
    {kFinishRecording, 2, "finish_recording"},
    {kSnapshotOutputs, 3, "snapshot_outputs"},
    {kResetInputAssembly, 1, "reset_input_assembly"},
    {kResetRasterizer, 1, "reset_rasterizer"},
    {kResetOutputMerger, 1, "reset_output_merger"},
    {kResetCommit, 1, "reset_commit"},
    {kRenameResolveB, 2, "rename_resolve_b"},
    {kRenameResolve, 2, "rename_resolve"},
    {kOutputTransition, 3, "output_transition"},
    {kApplyPendingFlags, 2, "apply_pending_flags"},
    {kCloseCurrentSegment, 1, "close_current_segment"},
    {kSubmitListSpans, 1, "submit_list_spans"},
    {kSpanStreamBegin, 1, "span_stream_begin"},
    {kAppendSpan, 2, "append_span"},
    {kRetireChunkList, 1, "retire_chunk_list"},
    {kWaitRegMem, 5, "wait_reg_mem"},
};
constexpr int kNumCallees = static_cast<int>(std::size(kCallees));
constexpr int kFinishBodyId = 0, kExecuteBodyId = 1, kExecuteListsBodyId = 2, kFinishRecordingId = 4, kAppendSpanId = 17;
constexpr int kSubmitId = 100, kFreeId = 101, kRefillId = 102, kReserveId = 103, kCloseFlipId = 104, kSubmitFlipBodyId = 105, kSubmitFlipId = 106,
              kSubmitDoneId = 107;

u32 g_body_result = 0, g_recording_result = 0;
std::vector<u32> g_append_results;  // one per append, then 0
std::size_t g_appends = 0;
long g_append_failed = 0, g_game_frees = 0, g_game_appends = 0, g_game_submits = 0;

void record(int who, const u64* a) {
    Call c{who, {}};
    for (int k = 0; k < kCallees[who].args; ++k) c.a[k] = a[k];
    g_rec.push_back(c);
}

template <int W>
GUEST_ABI u64 recorder(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 a[6] = {a0, a1, a2, a3, a4, a5};
    record(W, a);
    if (W == kFinishBodyId) return g_body_result;
    if (W == kFinishRecordingId) return g_recording_result;
    if (W == kAppendSpanId) {
        const u32 r = g_appends < g_append_results.size() ? g_append_results[g_appends++] : 0;
        g_append_failed += r != 0;
        return r;
    }
    return 0x5eed;
}

// sceGnmSubmitCommandBuffers: what the arrays hold, not where they are.
GUEST_ABI s32 submit_recorder(u64 count, u64 dcb, u64 dcb_bytes, u64 ccb, u64 ccb_bytes) {
    Call c{kSubmitId, {count & 0xffffffffull, 0, 0, ccb, ccb_bytes, 0}};
    const u32 n = static_cast<u32>(count);
    if (n >= 1) {
        c.a[1] = load<u64>(dcb);
        c.a[5] = load<u32>(dcb_bytes);
    }
    if (n >= 2) {
        c.a[2] = load<u64>(dcb + 8);
        c.a[5] |= static_cast<u64>(load<u32>(dcb_bytes + 4)) << 32;
    }
    g_rec.push_back(c);
    return 0;
}
GUEST_ABI u64 free_recorder(u64 self, u64 page) {
    g_rec.push_back({kFreeId, {self, page, 0, 0, 0, 0}});
    return 0;
}
// The flip's callees. The refill gives room or not (the end moved).
bool g_refill_gives = false;
GUEST_ABI u64 refill_recorder(u64 cb, u64 dwords, u64 arg3, u64 arg4) {
    g_rec.push_back({kRefillId, {cb, dwords & 0xffffffffull, arg3, arg4, 0, 0}});
    if (g_refill_gives) store<u64>(cb + 0x8, load<u64>(cb + 0x10) + 0x1000);
    return 1;
}
// The system's flip packets: the 64 dwords' room taken, as the game's writer does.
GUEST_ABI u64 reserve_recorder(u64 cb) {
    g_rec.push_back({kReserveId, {cb, 0, 0, 0, 0, 0}});
    store<u64>(cb + 0x10, load<u64>(cb + 0x10) + 0x100);
    return 0;
}
u64 header_sum(u64 h) {
    u64 sum = 0;
    for (int k = 0; k < 8; ++k) sum = sum * 0x100000001b3ull ^ load<u64>(h + 8ull * k);
    return sum;
}
// What the close with the flip is handed: the flip's fields it reads, not the
// stack's leftovers between them.
GUEST_ABI u64 close_flip_recorder(u64 cb, u64 fresh, u64 header, GxFlip flip) {
    g_rec.push_back({kCloseFlipId, {cb, fresh, header_sum(header), flip.target, flip.mode & 0xffffffffull, flip.arg ^ (flip.kind & 0xff) << 56}});
    return 0;
}
GUEST_ABI u64 submit_flip_body_recorder(u64 flip, u64 header) {
    g_rec.push_back({kSubmitFlipBodyId, {load<u64>(flip), load<u32>(flip + 8), load<u64>(flip + 0x10), header, 0, 0}});
    return 0;
}
GUEST_ABI s32 submit_flip_recorder(u64 count, u64 dcb, u64 dcb_bytes, u64 ccb, u64 ccb_bytes, u64 handle, u64 index, u64 mode, u64 arg) {
    Call c{kSubmitFlipId, {count & 0xffffffffull, 0, 0, ccb | ccb_bytes << 1, handle, 0}};
    const u32 n = static_cast<u32>(count);
    if (n >= 1) c.a[1] = load<u64>(dcb) ^ static_cast<u64>(load<u32>(dcb_bytes)) << 40;
    if (n >= 2) c.a[2] = load<u64>(dcb + 8) ^ static_cast<u64>(load<u32>(dcb_bytes + 4)) << 40;
    c.a[5] = (index & 0xffffffffull) ^ (mode & 0xffffffffull) << 32 ^ arg * 31;
    g_rec.push_back(c);
    return 0;
}
GUEST_ABI s32 submit_done_recorder() {
    g_rec.push_back({kSubmitDoneId, {0, 0, 0, 0, 0, 0}});
    return 0;
}

const char* who_name(int who) {
    if (who == kSubmitId) return "submit";
    if (who == kFreeId) return "free";
    if (who == kRefillId) return "refill";
    if (who == kReserveId) return "reserve_flip_packets";
    if (who == kCloseFlipId) return "close_segment_and_flip";
    if (who == kSubmitFlipBodyId) return "submit_segment_and_flip";
    if (who == kSubmitFlipId) return "submit_and_flip";
    if (who == kSubmitDoneId) return "submit_done";
    return who >= 0 && who < kNumCallees ? kCallees[who].name : "-";
}

// The three bodies are stubbed for their wrappers and the game's again for
// themselves: their first bytes kept, put back.
constexpr int kShared[3] = {kFinishBodyId, kExecuteBodyId, kExecuteListsBodyId};
u8 g_original_bytes[5][16];  // the three bodies, then the close and the submit with the flip
template <int... W>
void stub_all(std::integer_sequence<int, W...>) {
    for (int k = 0; k < 3; ++k) std::memcpy(g_original_bytes[k], eboot_kit::at(kCallees[kShared[k]].bn), 16);
    std::memcpy(g_original_bytes[3], eboot_kit::at(kCloseSegmentAndFlip), 16);
    std::memcpy(g_original_bytes[4], eboot_kit::at(kSubmitSegmentAndFlip), 16);
    ((W == kFinishBodyId || W == kExecuteBodyId || W == kExecuteListsBodyId
          ? void()
          : eboot_kit::stub(kCallees[W].bn, reinterpret_cast<const void*>(&recorder<W>))),
     ...);
}
void put_bytes(u64 bn, const u8* bytes) {
    const u64 page = bn & ~0xfffull, end = (bn + 16 + 0xfff) & ~0xfffull;
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::memcpy(eboot_kit::at(bn), bytes, 16);
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_EXEC);
}
void bodies_stubbed(bool on) {
    if (on) {
        eboot_kit::stub(kFinishBody, reinterpret_cast<const void*>(&recorder<kFinishBodyId>));
        eboot_kit::stub(kExecuteBody, reinterpret_cast<const void*>(&recorder<kExecuteBodyId>));
        eboot_kit::stub(kExecuteListsBody, reinterpret_cast<const void*>(&recorder<kExecuteListsBodyId>));
    } else {
        for (int k = 0; k < 3; ++k) put_bytes(kCallees[kShared[k]].bn, g_original_bytes[k]);
    }
}
// The close with the flip and the submit it hands to, as recorders or the game's.
void flip_bodies_stubbed(bool close, bool submit) {
    if (close) eboot_kit::stub(kCloseSegmentAndFlip, reinterpret_cast<const void*>(&close_flip_recorder));
    else put_bytes(kCloseSegmentAndFlip, g_original_bytes[3]);
    if (submit) eboot_kit::stub(kSubmitSegmentAndFlip, reinterpret_cast<const void*>(&submit_flip_body_recorder));
    else put_bytes(kSubmitSegmentAndFlip, g_original_bytes[4]);
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}
void calls_differ(const char* what, int trial, const std::vector<Call>& ours, const std::vector<Call>& game) {
    std::size_t at = 0;
    while (at < ours.size() && at < game.size() && ours[at] == game[at]) ++at;
    char why[256];
    std::snprintf(why, sizeof why, "the calls (%zu, the game's %zu): #%zu ours %s, the game's %s", ours.size(), game.size(), at,
                  at < ours.size() ? who_name(ours[at].who) : "-", at < game.size() ? who_name(game[at].who) : "-");
    fail(what, trial, why);
}

// ---- The memory ----
alignas(64) u8 g_mem[0x20000];
constexpr u64 kCtx = 0x0, kLists = 0xe000, kPages = 0xf000, kListArray = 0x10800, kAllocator = 0x10900, kVtable = 0x10a00, kHolder = 0x11000,
              kRing = 0x1d000, kCounters = 0x1d100, kRingWord = 0x1d200, kHeader = 0x1d300;
constexpr int kMaxPages = 12, kPageBytes = 0x200;
u64 base() { return reinterpret_cast<u64>(g_mem); }
u64 ctx() { return base() + kCtx; }
u64 list_at(int k) { return base() + kLists + 0xc0ull * static_cast<u64>(k); }

u64 garbage() { return (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32(); }

void fill_memory() {
    for (u8& b : g_mem) b = static_cast<u8>(g_rng.u32());
    const u64 c = ctx();
    store<u8>(c + 0xb478 + 0x24, g_rng.chance(0.5) ? 0 : static_cast<u8>(g_rng.range(1, 255)));
    // The allocator: an object whose vtable's +0x70 frees.
    store<u64>(base() + kAllocator, base() + kVtable);
    store<u64>(base() + kVtable + 0x70, reinterpret_cast<u64>(&free_recorder));
}

int g_next_page = 0;
// A list of up to four pages of `per` spans, an allocator or none.
void make_list(u64 list) {
    const u32 per = static_cast<u32>(g_rng.range(1, 7));
    store<u64>(list, g_rng.chance(0.8) ? base() + kAllocator : 0);
    store<u32>(list + 0x18, per);
    store<u32>(list + 0x1c, static_cast<u32>(g_rng.chance(0.8) ? g_rng.range(0, static_cast<int>(per)) : g_rng.range(0, 9)));
    const int pages = g_rng.chance(0.15) ? 0 : g_rng.range(1, 4);
    u64 first = 0, prev = 0;
    for (int p = 0; p < pages && g_next_page < kMaxPages; ++p) {
        const u64 page = base() + kPages + static_cast<u64>(kPageBytes) * static_cast<u64>(g_next_page++);
        store<u64>(page + (static_cast<u64>(per) << 6), 0);
        if (prev) store<u64>(prev + (static_cast<u64>(per) << 6), page);
        else first = page;
        prev = page;
    }
    store<u64>(list + 8, first);
    store<u64>(list + 0x10, prev);
}

// ---- Running both ----
template <class Game, class Ours>
void both(const char* what, int trial, Game game, Ours ours) {
    static u8 saved[sizeof g_mem], game_mem[sizeof g_mem];
    std::memcpy(saved, g_mem, sizeof g_mem);
    const std::vector<u32> appends = g_append_results;
    g_rec.clear();
    g_appends = 0;
    const long failed_before = g_append_failed;
    const u64 game_ret = game();
    const std::vector<Call> game_calls = g_rec;
    const long game_failed = g_append_failed - failed_before;
    for (const Call& c : game_calls) {
        g_game_frees += c.who == kFreeId;
        g_game_appends += c.who == kAppendSpanId;
        g_game_submits += c.who == kSubmitId;
    }
    std::memcpy(game_mem, g_mem, sizeof g_mem);
    std::memcpy(g_mem, saved, sizeof g_mem);
    g_rec.clear();
    g_appends = 0;
    g_append_results = appends;
    const u64 our_ret = ours();
    g_append_failed = failed_before + game_failed;  // counted once, as the game's run met them
    if (our_ret != game_ret) fail(what, trial, "the return value");
    if (std::memcmp(g_mem, game_mem, sizeof g_mem) != 0) {
        std::size_t at = 0;
        while (at < sizeof g_mem && g_mem[at] == game_mem[at]) ++at;
        char why[96];
        std::snprintf(why, sizeof why, "the memory, from byte 0x%zx", at);
        fail(what, trial, why);
    }
    if (g_rec != game_calls) calls_differ(what, trial, g_rec, game_calls);
}

int run_finish(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        g_next_page = 0;
        const u64 list = g_rng.chance(0.1) ? 0 : list_at(0);
        if (list) make_list(list);
        const u64 restore = g_rng.chance(0.5) ? garbage() & ~0xffull : garbage();
        g_body_result = g_rng.chance(0.5) ? 0 : g_rng.u32();
        using W = GUEST_ABI u64 (*)(u64, u64, u64);
        both("gx_finish_command_list", t, [&] { return eboot_kit::fn<W>(0x2ad5850)(ctx(), list, restore) & 0xffffffffull; },
             [&] { return gx_finish_command_list(ctx(), list, restore) & 0xffffffffull; });
    }
    return cases;
}

int run_finish_body(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        g_next_page = 0;
        const u64 list = g_rng.chance(0.1) ? 0 : list_at(0);
        if (list) make_list(list);
        g_recording_result = g_rng.chance(0.7) ? 0 : g_rng.chance(0.5) ? 9 : g_rng.u32();
        const u64 state = garbage(), restore = garbage();
        using B = GUEST_ABI u64 (*)(u64, u64, u64, u64);
        both("gx_finish_command_list_impl", t, [&] { return eboot_kit::fn<B>(kFinishBody)(ctx(), state, list, restore) & 0xffffffffull; },
             [&] { return gx_finish_command_list_impl(ctx(), state, list, restore) & 0xffffffffull; });
    }
    return cases;
}

int run_execute(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        g_next_page = 0;
        const u64 list = list_at(0);
        make_list(list);
        const u64 restore = g_rng.chance(0.5) ? garbage() & ~0xffull : garbage();
        using W = GUEST_ABI u64 (*)(u64, u64, u64);
        both("gx_execute_command_list", t, [&] { eboot_kit::fn<W>(0x25679a0)(ctx(), list, restore); return 0ull; },
             [&] { gx_execute_command_list(ctx(), list, restore); return 0ull; });
    }
    return cases;
}

int run_execute_body(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        g_next_page = 0;
        const u64 list = list_at(0);
        make_list(list);
        const u64 unused = garbage();
        using B = GUEST_ABI u64 (*)(u64, u64, u64);
        both("gx_context_execute_command_list", t, [&] { return eboot_kit::fn<B>(kExecuteBody)(ctx(), unused, list); },
             [&] { return gx_context_execute_command_list(ctx(), unused, list); });
    }
    return cases;
}

int run_execute_lists(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        const u64 lists = g_rng.chance(0.1) ? 0 : base() + kListArray;
        const u64 count = g_rng.chance(0.1) ? garbage() & ~0xffffffffull : (garbage() & ~0xffffffffull) | static_cast<u64>(g_rng.range(1, 5));
        const u64 restore = g_rng.chance(0.5) ? garbage() & ~0xffull : garbage();
        using W = GUEST_ABI u64 (*)(u64, u64, u64, u64);
        both("gx_execute_command_lists", t, [&] { eboot_kit::fn<W>(0x25679e0)(ctx(), lists, count, restore); return 0ull; },
             [&] { gx_execute_command_lists(ctx(), lists, count, restore); return 0ull; });
    }
    return cases;
}

int run_execute_lists_body(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        g_next_page = 0;
        const int n = g_rng.range(0, 5);
        for (int k = 0; k < 5; ++k) {
            const bool present = k < n && g_rng.chance(0.85);
            store<u64>(base() + kListArray + 8ull * static_cast<u64>(k), present ? list_at(k) : 0);
            if (present) make_list(list_at(k));
        }
        const u64 lists = g_rng.chance(0.05) ? 0 : base() + kListArray;
        const u64 count = (garbage() & ~0xffffffffull) | static_cast<u64>(n);
        g_append_results.clear();
        for (int k = 0; k < 40; ++k) g_append_results.push_back(g_rng.chance(0.06) ? (g_rng.chance(0.5) ? 9 : g_rng.u32() | 1) : 0);
        const u64 unused = garbage();
        using B = GUEST_ABI u64 (*)(u64, u64, u64, u64);
        both("gx_context_execute_command_lists", t, [&] { eboot_kit::fn<B>(kExecuteListsBody)(ctx(), unused, lists, count); return 0ull; },
             [&] { gx_context_execute_command_lists(ctx(), unused, lists, count); return 0ull; });
        g_append_results.clear();
    }
    return cases;
}

int g_close_waits = 0, g_close_wraps = 0, g_close_two = 0;
int run_close_segment(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        const u64 cb = ctx() + 0x10;
        // The compute ring: its word and its two counters.
        store<u64>(cb + 0xb490, base() + kHolder);
        store<u64>(base() + kHolder + 0xb728, base() + kRing);
        const u64 ring = base() + kRing;
        store<u64>(ring + 0x10, base() + kRingWord);
        store<u64>(ring + 0x18, base() + kCounters);
        store<u64>(ring + 0x20, base() + kCounters + 4);
        const u32 total = g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, 2)) : g_rng.u32();
        u32 waited = total - 1;
        if (g_rng.chance(0.5)) waited = g_rng.chance(0.4) ? 0x80000000u | g_rng.u32() : g_rng.u32();
        store<u32>(base() + kCounters, total);
        store<u32>(base() + kCounters + 4, waited);
        if (waited != total - 1) {
            ++g_close_waits;
            if (static_cast<s32>(waited) < 0 && static_cast<s32>(total - 1) >= 0) ++g_close_wraps;
        }
        // The fresh start (the buffer itself) or none, and the segment.
        const u64 fresh = g_rng.chance(0.6) ? cb : 0;
        const auto span = [&](u64 at) {
            const u64 begin = (garbage() & ~0xfull);
            store<u64>(at, begin);
            const u32 bytes = g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, 3)) : static_cast<u32>(g_rng.range(0, 1 << 20));
            store<u64>(at + 0x10, begin + bytes);
        };
        span(cb);
        span(base() + kHeader);
        using F = GUEST_ABI u64 (*)(u64, u64, u64, u64);
        const u64 header = base() + kHeader, kind = garbage();
        both("gx_command_buffer_close_segment", t, [&] { eboot_kit::fn<F>(0x2ab0b00)(cb, fresh, header, kind); return 0ull; },
             [&] { gx_command_buffer_close_segment(cb, fresh, header, kind); return 0ull; });
        for (const Call& c : g_rec) g_close_two += c.who == kSubmitId && c.a[0] == 2;
    }
    return cases;
}

// The compute ring and a command buffer's header, for the close and the flip.
void make_ring(u64 cb) {
    store<u64>(cb + 0xb490, base() + kHolder);
    store<u64>(base() + kHolder + 0xb728, base() + kRing);
    const u64 ring = base() + kRing;
    store<u64>(ring + 0x10, base() + kRingWord);
    store<u64>(ring + 0x18, base() + kCounters);
    store<u64>(ring + 0x20, base() + kCounters + 4);
    const u32 total = g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, 2)) : g_rng.u32();
    u32 waited = total - 1;
    if (g_rng.chance(0.5)) waited = g_rng.chance(0.4) ? 0x80000000u | g_rng.u32() : g_rng.u32();
    store<u32>(base() + kCounters, total);
    store<u32>(base() + kCounters + 4, waited);
}
int g_flip_refused = 0, g_flip_refilled = 0, g_flip_fresh = 0, g_flip_last = 0;
int run_flip(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        const u64 cb = ctx() + 0x10;
        make_ring(cb);
        const u64 wp = base() + 0x2000;
        store<u64>(cb, wp - 8 * static_cast<u64>(g_rng.range(0, 64)));
        store<u64>(cb + 0x10, wp);
        // Room in dwords: none to plenty, the 64 and 7 edges often.
        static const int kRooms[] = {0, 6, 7, 8, 63, 64, 65, 70, 71, 72, 1000};
        const int room = g_rng.chance(0.7) ? kRooms[g_rng.range(0, 10)] : g_rng.range(0, 4000);
        store<u64>(cb + 0x8, wp + 4ull * static_cast<u64>(room));
        store<u64>(cb + 0x18, reinterpret_cast<u64>(&refill_recorder));
        g_refill_gives = g_rng.chance(0.5);
        const u64 handle = garbage(), index = garbage(), mode = garbage(), arg = garbage();
        using F = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
        both("gx_command_buffer_flip", t, [&] { return eboot_kit::fn<F>(0x2aafb30)(cb, handle, index, mode, arg) & 0xffffffffull; },
             [&] { return gx_command_buffer_flip(cb, handle, index, mode, arg) & 0xffffffffull; });
        bool refused = false, refilled = false, closed = false;
        u64 fresh = 0;
        for (const Call& c : g_rec) {
            refilled |= c.who == kRefillId;
            if (c.who == kCloseFlipId) closed = true, fresh = c.a[1];
        }
        refused = !closed;
        g_flip_refused += refused;
        g_flip_refilled += refilled;
        g_flip_fresh += closed && fresh;
        g_flip_last += closed && !fresh;
    }
    return cases;
}
int run_close_and_flip(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        const u64 cb = ctx() + 0x10;
        make_ring(cb);
        const u64 fresh = g_rng.chance(0.7) ? cb : 0;
        const auto span = [&](u64 at) {
            const u64 begin = garbage() & ~0xfull;
            store<u64>(at, begin);
            const u32 bytes = g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, 3)) : static_cast<u32>(g_rng.range(0, 1 << 20));
            store<u64>(at + 0x10, begin + bytes);
        };
        span(cb);
        span(base() + kHeader);
        const GxFlip flip{garbage(), garbage(), garbage(), garbage()};
        using F = GUEST_ABI void (*)(u64, u64, u64, GxFlip);
        const u64 header = base() + kHeader;
        both("gx_command_buffer_close_segment_and_flip", t, [&] { eboot_kit::fn<F>(kCloseSegmentAndFlip)(cb, fresh, header, flip); return 0ull; },
             [&] { gx_command_buffer_close_segment_and_flip(cb, fresh, header, flip); return 0ull; });
    }
    return cases;
}
int run_submit_and_flip(int cases) {
    for (int t = 0; t < cases; ++t) {
        fill_memory();
        const u64 header = base() + kHeader, flip = base() + kHeader + 0x80;
        const u64 begin = garbage() & ~0xfull;
        store<u64>(header, begin);
        store<u64>(header + 0x10, begin + (g_rng.chance(0.2) ? static_cast<u32>(g_rng.range(0, 3)) : static_cast<u32>(g_rng.range(0, 1 << 20))));
        using F = GUEST_ABI void (*)(u64, u64);
        both("gx_submit_segment_and_flip", t, [&] { eboot_kit::fn<F>(kSubmitSegmentAndFlip)(flip, header); return 0ull; },
             [&] { gx_submit_segment_and_flip(flip, header); return 0ull; });
    }
    return cases;
}

long run_all(int cases) {
    long total = 0;
    bodies_stubbed(true);
    total += run_finish(cases);
    total += run_execute(cases);
    total += run_execute_lists(cases);
    bodies_stubbed(false);
    total += run_finish_body(cases);
    total += run_execute_body(cases);
    total += run_execute_lists_body(cases);
    total += run_close_segment(cases);
    flip_bodies_stubbed(true, true);
    total += run_flip(cases);
    flip_bodies_stubbed(false, true);
    total += run_close_and_flip(cases);
    flip_bodies_stubbed(false, false);
    total += run_submit_and_flip(cases);
    return total;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_list_mutant = mutant;
    g_quiet = true;
    run_all(300);
    g_quiet = false;
    g_list_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_command_list_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 4000;
    stub_all(std::make_integer_sequence<int, kNumCallees>{});
    eboot_kit::stub(kSubmitCommandBuffers, reinterpret_cast<const void*>(&submit_recorder));
    eboot_kit::stub(kSubmitAndFlip, reinterpret_cast<const void*>(&submit_flip_recorder));
    eboot_kit::stub(kSubmitDone, reinterpret_cast<const void*>(&submit_done_recorder));
    eboot_kit::stub(kReserveFlipPackets, reinterpret_cast<const void*>(&reserve_recorder));

    struct Entry {
        u64 bn;
        const u8* bytes;
        std::size_t n;
        const char* name;
    };
    const Entry kEntries[] = {
        {0x2ad5850, kEntryFinish, sizeof kEntryFinish, "gx_finish_command_list"},
        {kFinishBody, kEntryFinishBody, sizeof kEntryFinishBody, "gx_finish_command_list_impl"},
        {0x25679a0, kEntryExecuteWrapper, sizeof kEntryExecuteWrapper, "gx_execute_command_list"},
        {kExecuteBody, kEntryExecute, sizeof kEntryExecute, "gx_context_execute_command_list"},
        {0x25679e0, kEntryExecuteListsWrapper, sizeof kEntryExecuteListsWrapper, "gx_execute_command_lists"},
        {kExecuteListsBody, kEntryExecuteLists, sizeof kEntryExecuteLists, "gx_context_execute_command_lists"},
        {0x2ab0b00, kEntryCloseSegment, sizeof kEntryCloseSegment, "gx_command_buffer_close_segment"},
        {0x2aafb30, kEntryFlip, sizeof kEntryFlip, "gx_command_buffer_flip"},
        {kCloseSegmentAndFlip, kEntryCloseSegmentAndFlip, sizeof kEntryCloseSegmentAndFlip, "gx_command_buffer_close_segment_and_flip"},
        {kSubmitSegmentAndFlip, kEntrySubmitSegmentAndFlip, sizeof kEntrySubmitSegmentAndFlip, "gx_submit_segment_and_flip"},
    };
    for (const Entry& e : kEntries) {
        const u8* at = eboot_kit::file_at("gx_command_list_test", e.bn);
        if (!at || std::memcmp(at, e.bytes, e.n) != 0) fail(e.name, -1, "the entry's instructions");
    }

    const long total = run_all(cases);
    std::printf("  %ld cases against the game's: %ld spans appended (%ld of them refused, a page not had), %ld pages freed, %ld submissions; "
                "close_segment %d with the ring's waits (%d for the wrap), %d submitting two buffers; flips: %d refused (no room after the "
                "refill), %d asked the refill, %d closed with a fresh start, %d the ring's last\n",
                total, g_game_appends, g_append_failed, g_game_frees, g_game_submits, g_close_waits, g_close_wraps, g_close_two, g_flip_refused,
                g_flip_refilled, g_flip_fresh, g_flip_last);
    for (const int mutant : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("gx_command_list_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("gx_command_list_test: ok - %ld cases, 9 mutants caught\n", total);
    return 0;
}
