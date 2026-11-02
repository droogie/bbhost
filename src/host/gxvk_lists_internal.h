// Stage 4's census (host/gxvk_lists.h): what its leaf half writes and its
// host half reads.
#pragma once

#include <atomic>
#include <cstdint>

namespace gxvk_lists::detail {

// A GX context: its finished lists so far, and the GX-native draw tokens
// recorded on it since its last finish.
struct CtxSlot {
    std::atomic<std::uint64_t> ctx{0};
    std::atomic<std::uint32_t> finishes{0};
    std::atomic<std::uint32_t> tokens{0};
};
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

}  // namespace gxvk_lists::detail
