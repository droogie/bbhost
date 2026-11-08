// Stage 4's census (host/gxvk_lists.h): what its leaf half writes and its
// host half reads.
#pragma once

#include <atomic>
#include <cstdint>

namespace gxvk_lists {
// BBHOST_GXVK_LISTS as given: the pool mode it asks for, -1 unset.
int pool_mode_from_env();
}  // namespace gxvk_lists

namespace gxvk_lists::detail {

// A GX context: its finished lists so far, and the GX-native draw tokens
// recorded on it since its last finish; with stage 4b.2's segments, the list
// record its draws go into (its index plus one, 0: none open).
struct CtxSlot {
    std::atomic<std::uint64_t> ctx{0};
    std::atomic<std::uint32_t> finishes{0};
    std::atomic<std::uint32_t> tokens{0};
    std::atomic<std::uint32_t> open{0};
};
CtxSlot* ctx_slot(std::uint64_t ctx);
constexpr std::uint32_t kCtxSlots = 1024;
extern CtxSlot g_ctx[kCtxSlots];
extern std::atomic<std::uint64_t> g_ctx_full;

// A list object's last finish: which context finished it, as its seq-th.
struct ListSlot {
    std::atomic<std::uint64_t> list{0};
    std::atomic<std::uint64_t> ctx{0};
    std::atomic<std::uint32_t> seq{0};
};
constexpr std::uint32_t kListSlots = 4096;
extern ListSlot g_list_map[kListSlots];

// The rings the leaves write: a record is written once `done` is its place
// plus one.
struct FinishRec {
    std::uint64_t ctx, list, tsc;
    std::uint32_t seq, tokens;
    std::atomic<std::uint64_t> done{0};
};
struct ExecRec {
    std::uint64_t list, ctx, tsc;
    std::uint32_t seq, index, count, pad;
    std::atomic<std::uint64_t> done{0};
};
constexpr std::uint64_t kRing = 1u << 16;
extern FinishRec g_finish[kRing];
extern ExecRec g_exec[kRing];
extern std::atomic<std::uint64_t> g_finish_pos, g_exec_pos;
extern std::atomic<std::uint64_t> g_exec_unknown;

// ---- Stage 4b.1a-4b.2's pool feed (host/gxvk_lists_pool.cpp takes from it) ----
// The recording threads put each stored draw's token id on it (draw_stored), and
// with the segments each list record a context closed (FinishCommandList's leaf,
// after the list's draws: list_closed). An entry is written once `seq` is its
// place plus one. Leaf-safe: atomics only.
enum FeedKind : std::uint32_t { kFeedDraw = 0, kFeedList = 1 };
struct Feed {
    std::atomic<std::uint64_t> seq{0};
    std::atomic<std::uint64_t> id{0};  // a draw's token id, or a list record's index
    std::atomic<std::uint32_t> kind{kFeedDraw};
};
constexpr std::uint64_t kFeed = 1u << 15;
extern Feed g_feed[kFeed];
extern std::atomic<std::uint64_t> g_feed_pos;   // the recording threads' next place
extern std::atomic<std::uint64_t> g_feed_next;  // the pool's next place
std::uint64_t feed_push(std::uint64_t id, std::uint32_t kind);  // its place

// BBHOST_GXVK_LISTS_TIMES (host/gxvk_lists.h times): a stored draw's times by
// its token slot (the id's low bits, as the commits' records): the draw's id,
// when it was stored (rdtsc), its place in the feed, and when its list record
// closed (FinishCommandList's leaf). Written by the recording threads, read by
// the command processor at the token; a diagnostic, so relaxed throughout.
struct DrawTime {
    std::atomic<std::uint64_t> id{0}, stored{0}, feed{0}, closed{0};
};
constexpr std::uint64_t kTimeSlots = 16384;
extern DrawTime g_draw_times[kTimeSlots];

// ---- Stage 4b.2: the draws of a list as a context recorded them ----
// A context's open record takes each draw the recording thread stored on it,
// in order (a deferred context is one thread's at a time: one writer); the
// list's FinishCommandList closes it and feeds it to the pool, which cuts it
// into segments. A record that fills is closed as it is (its list goes on in a
// new one). States: free, open (the context's), closed (fed), the pool's.
enum RecState : std::uint32_t { kRecFree, kRecOpen, kRecClosed, kRecPool };
constexpr std::uint32_t kRecMost = 256;   // draws a record holds
constexpr std::uint32_t kRecs = 1024;     // records in the ring (~30 lists a flip)
struct ListRec {
    std::atomic<std::uint32_t> state{kRecFree};
    std::uint32_t n = 0;
    std::uint64_t ctx = 0, list = 0;     // list: 0 while open, or for a record closed full
    std::uint64_t id[kRecMost];          // the draws' token ids
    std::uint64_t va[kRecMost];          // and where each token lies in the context's command buffer
};
extern ListRec g_recs[kRecs];
extern std::atomic<std::uint64_t> g_rec_pos;      // where the next record is looked for
extern std::atomic<std::uint64_t> g_rec_none;     // draws that found no free record
extern std::atomic<std::uint64_t> g_immediate;    // the immediate context (its draws go into no record)
// The context's open record closed and fed (its list `list`, 0 when it filled).
void list_closed(CtxSlot& s, std::uint64_t list);

}  // namespace gxvk_lists::detail
