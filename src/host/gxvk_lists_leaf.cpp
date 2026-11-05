// Stage 4's census (host/gxvk_lists.h): the half the decomp leaves call, on
// the game's threads with the game's FS - no locks, no allocation, no host
// thread-locals, no stack protector (CMakeLists.txt) - and the switch.
#include "host/gxvk_lists.h"
#include "host/gxvk_lists_internal.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <x86intrin.h>

namespace gxvk_lists {

// BBHOST_GXVK_LISTS's pool modes: 1 and 2 (stage 4b.1a's lookups, and their
// compare), splice (3) and splice-compare (4) (stage 4b.1b's commits),
// segments (5) and segments-compare (6) (stage 4b.2's); -1 unset (the homes
// decide: settle); 0 for anything else (off, or the census).
int pool_mode_from_env() {
    const char* e = std::getenv("BBHOST_GXVK_LISTS");
    if (!e || !*e) return -1;
    if (std::strcmp(e, "splice") == 0) return 3;
    if (std::strcmp(e, "splice-compare") == 0) return 4;
    if (std::strcmp(e, "segments") == 0) return 5;
    if (std::strcmp(e, "segments-compare") == 0) return 6;
    return (e[0] == '1' || e[0] == '2') && !e[1] ? e[0] - '0' : 0;
}

// Stage 4b.1a's pool (host/gxvk_lists.h): its threads, 0 when it is off or
// cannot run (with the switch unset: whether it may, once settled).
int pool_threads() {
    static const int n = [] {
        if (!pool_mode_from_env()) return 0;
        if (const char* t = std::getenv("BBHOST_GXVK_LIST_THREADS")) {
            const int k = std::atoi(t);
            return k < 0 ? 0 : k > 4 ? 4 : k;
        }
        const unsigned hw = std::thread::hardware_concurrency();
        return hw >= 16 ? 2 : hw >= 8 ? 1 : 0;
    }();
    return n;
}

const bool g_census = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS");
    return e && std::strcmp(e, "census") == 0;
}();
const bool g_times = [] {
    const char* e = std::getenv("BBHOST_GXVK_LISTS_TIMES");
    return e && e[0] == '1';
}();
std::atomic<int> g_pool_mode{0};

namespace detail {

Feed g_feed[kFeed];
std::atomic<std::uint64_t> g_feed_pos{0};
std::atomic<std::uint64_t> g_feed_next{0};
ListRec g_recs[kRecs];
std::atomic<std::uint64_t> g_rec_pos{0};
std::atomic<std::uint64_t> g_rec_none{0};
std::atomic<std::uint64_t> g_immediate{0};

DrawTime g_draw_times[kTimeSlots];

std::uint64_t feed_push(std::uint64_t id, std::uint32_t kind) {
    const std::uint64_t pos = g_feed_pos.fetch_add(1, std::memory_order_seq_cst);
    Feed& f = g_feed[pos & (kFeed - 1)];
    f.id.store(id, std::memory_order_relaxed);
    f.kind.store(kind, std::memory_order_relaxed);
    f.seq.store(pos + 1, std::memory_order_seq_cst);
    return pos;
}

void list_closed(CtxSlot& s, std::uint64_t list) {
    const std::uint32_t open = s.open.exchange(0, std::memory_order_acq_rel);
    if (!open || open > kRecs) return;  // none (or none to be had for this list: kRecs + 1)
    ListRec& r = g_recs[open - 1];
    r.list = list;
    if (g_times) {
        const std::uint64_t now = __rdtsc();
        const std::uint32_t n = r.n < kRecMost ? r.n : kRecMost;
        for (std::uint32_t k = 0; k < n; ++k) g_draw_times[r.id[k] & (kTimeSlots - 1)].closed.store(now, std::memory_order_relaxed);
    }
    r.state.store(kRecClosed, std::memory_order_release);
    feed_push(open - 1, kFeedList);
}

CtxSlot g_ctx[kCtxSlots];
std::atomic<std::uint64_t> g_ctx_full{0};
ListSlot g_list_map[kListSlots];
FinishRec g_finish[kRing];
ExecRec g_exec[kRing];
std::atomic<std::uint64_t> g_finish_pos{0}, g_exec_pos{0};
std::atomic<std::uint64_t> g_exec_unknown{0};

// The slot of context `ctx` (open addressing by its address), claimed at the
// first sight; null when the probe finds no room (counted).
CtxSlot* ctx_slot(std::uint64_t ctx) {
    const std::uint32_t h = static_cast<std::uint32_t>(((ctx >> 4) * 0x9e3779b97f4a7c15ull) >> 54);
    for (std::uint32_t k = 0; k < 32; ++k) {
        CtxSlot& s = g_ctx[(h + k) & (kCtxSlots - 1)];
        std::uint64_t v = s.ctx.load(std::memory_order_acquire);
        if (v == ctx) return &s;
        if (v == 0) {
            if (s.ctx.compare_exchange_strong(v, ctx, std::memory_order_acq_rel) || v == ctx) return &s;
        }
    }
    g_ctx_full.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

}  // namespace detail

using namespace detail;

void finished(std::uint64_t ctx, std::uint64_t list) {
    if (segments()) {
        // Stage 4b.2: the list's draws, in its context's open record, to the pool.
        if (CtxSlot* s = ctx_slot(ctx)) list_closed(*s, list ? list : 1);
    }
    if (!g_census) return;
    CtxSlot* s = ctx_slot(ctx);
    if (!s) return;
    const std::uint32_t seq = s->finishes.fetch_add(1, std::memory_order_acq_rel) + 1;
    const std::uint32_t tokens = s->tokens.exchange(0, std::memory_order_acq_rel);
    // The list object's last finish, for its execution to name.
    ListSlot& m = g_list_map[((list >> 4) * 0x9e3779b97f4a7c15ull) >> 52 & (kListSlots - 1)];
    m.seq.store(0, std::memory_order_relaxed);
    m.list.store(list, std::memory_order_relaxed);
    m.ctx.store(ctx, std::memory_order_relaxed);
    m.seq.store(seq, std::memory_order_release);
    const std::uint64_t pos = g_finish_pos.fetch_add(1, std::memory_order_relaxed);
    FinishRec& r = g_finish[pos & (kRing - 1)];
    r.ctx = ctx;
    r.list = list;
    r.tsc = __rdtsc();
    r.seq = seq;
    r.tokens = tokens;
    r.done.store(pos + 1, std::memory_order_release);
}

void executed(std::uint64_t list, std::uint32_t index, std::uint32_t count) {
    if (!g_census) return;
    const ListSlot& m = g_list_map[((list >> 4) * 0x9e3779b97f4a7c15ull) >> 52 & (kListSlots - 1)];
    const std::uint32_t seq = m.seq.load(std::memory_order_acquire);
    const std::uint64_t ctx = m.ctx.load(std::memory_order_relaxed);
    const bool known = seq && m.list.load(std::memory_order_relaxed) == list;
    if (!known) g_exec_unknown.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t pos = g_exec_pos.fetch_add(1, std::memory_order_relaxed);
    ExecRec& r = g_exec[pos & (kRing - 1)];
    r.list = list;
    r.ctx = known ? ctx : 0;
    r.tsc = __rdtsc();
    r.seq = known ? seq : 0;
    r.index = index;
    r.count = count;
    r.done.store(pos + 1, std::memory_order_release);
}

void immediate_context(std::uint64_t ctx) {
    if (g_immediate.load(std::memory_order_relaxed) != ctx) g_immediate.store(ctx, std::memory_order_relaxed);
}

std::uint32_t draw_recorded(std::uint64_t ctx) {
    CtxSlot* s = ctx_slot(ctx);
    if (!s) return 0;
    s->tokens.fetch_add(1, std::memory_order_relaxed);
    return s->finishes.load(std::memory_order_acquire) + 1;
}

}  // namespace gxvk_lists
