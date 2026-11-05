// Stage 4b.3c: GX's constant buffers in device-local homes made at their
// creation, bound by handle (host/gxvk_homes.h).
//
// Two sides share the table here. GX's threads make a home as GX makes the
// buffer (host_gpu_home_made: an entry, a range of a pool block, the handle in
// the holder) and end it with the buffer (host_gpu_home_end); they take this
// file's lock (g_mu) for that alone, never the renderer's. The command
// processor (under the renderer's lock, g.mu) learns of both from a queue of
// events it drains before any lookup by address, fills a home at its first use,
// puts the copy-backs in place, binds homes by handle, answers the lookups by
// address, and takes an ended entry back once nothing still to be recorded can
// bind it.
#include "host/gxvk_homes.h"
#include "host/gxvk_homes_index.h"

#include "core/portable.h"
#include "engine/frame_rate.h"
#include "hle/modules.h"
#include "host/cb_capture.h"
#include "host/gpu_internal.h"
#include "host/list_resolve.h"
#include "host/memory_policy.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gpu {

namespace {

using ull = unsigned long long;
using namespace gxvk_homes;

enum State : std::uint32_t { kFree = 0, kLive = 1, kEnded = 2, kMapped = 3 };

// The pool: blocks of device-local memory (kBlockBytes: host/gxvk_homes.h),
// cut into ranges by size class, each class with its free list; a block is
// carved from its start as the classes need more. Blocks stay for the session.
// The classes are 256-byte steps up to 4 KiB - a world's ~54,000 constant
// buffers average ~800 bytes, which powers of two rounded up by half again -
// then 8, 16, 32 and 64 KiB.
constexpr int kSmallClasses = 16, kClasses = kSmallClasses + 4;
int class_of(std::uint32_t bytes) {
    if (bytes <= 4096) return static_cast<int>((std::max<std::uint32_t>(bytes, 1) + 255) / 256) - 1;
    int c = kSmallClasses;
    while (c + 1 < kClasses && (VkDeviceSize{4096} << (c - kSmallClasses + 1)) < bytes) ++c;
    return c;
}
VkDeviceSize class_bytes(int c) { return c < kSmallClasses ? VkDeviceSize(c + 1) * 256 : VkDeviceSize{4096} << (c - kSmallClasses + 1); }

struct Home {
    std::atomic<std::uint32_t> word{0};  // generation << 8 | State
    std::atomic<std::uint8_t> end_why{0};
    // Written by the thread that makes it, before its word.
    std::uint64_t holder = 0, memory = 0;
    std::uint32_t bytes = 0;
    std::uint8_t cls = 0;
    std::int32_t block = -1;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress address = 0;
    // Whether its fill is recorded: written by the command processor (under g.mu) with release, read with acquire -
    // also by stage 4b's pool, whose list-mode binding (homes_bind_locked) runs outside g.mu and leaves a home not
    // filled yet to the command processor's draw in order.
    std::atomic<bool> filled{false};
    // The command processor's (under g.mu).
    bool indexed = false;      // in the index by address
    bool end_queued = false;   // on its way to being taken back
    std::uint32_t pending = 0;  // copies into it still to be put in place
    std::uint32_t pend_at = 0;      // ... the newest one's place in the pending list, valid in pend_epoch's batch
    std::uint64_t pend_epoch = 0;
    std::uint8_t* mirror = nullptr;  // compare: what the home holds at this point of the walk
};
inline std::uint32_t state_of(std::uint32_t w) { return w & 0xff; }
inline std::uint32_t gen_of(std::uint32_t w) { return (w >> 8) & 0xff; }

constexpr std::uint32_t kChunkBits = 12, kChunks = 1u << (kIndexBits - kChunkBits);
std::atomic<Home*> g_chunks[kChunks] = {};
Home* home_at(std::uint32_t index) {
    if (index > kIndexMask) return nullptr;
    Home* c = g_chunks[index >> kChunkBits].load(std::memory_order_acquire);
    return c ? &c[index & ((1u << kChunkBits) - 1)] : nullptr;
}

// GX's threads and the command processor: the table's growth and free
// indices, the pool's blocks and free lists, the events. Never the renderer's
// lock inside it.
std::mutex g_mu;
std::uint32_t g_next = 0;
std::vector<std::uint32_t> g_free_index;
struct Block {
    DevBuffer buf;
    VkDeviceSize carved = 0;
};
std::vector<Block> g_blocks;
std::vector<std::pair<std::int32_t, VkDeviceSize>> g_class_free[kClasses];
std::uint64_t g_budget = 0;          // bytes the pool may hold in ranges taken; 0 until decided
std::uint64_t g_held = 0, g_held_peak = 0;
// Events for the command processor's index, in the order GX's threads made
// them: a home made, a home ended, a range covered (unmapped, or given to a new
// resource - the homes over it end, whatever the registry saw of their buffers).
struct Event {
    enum Kind : std::uint8_t { kMade, kEnd, kCovered } kind;
    std::uint8_t why = 0;  // kCovered: gxvk_homes::End
    std::uint32_t index = 0;
    std::uint64_t va = 0, bytes = 0;
};
std::vector<Event> g_events;
std::atomic<std::uint64_t> g_ev_n{0};

// Counters.
std::atomic<std::uint64_t> g_made_n{0}, g_made_bytes{0}, g_no_home[kNoHomes] = {}, g_ended[kEnds] = {}, g_blocks_made{0},
    g_blocks_refused{0};
std::atomic<std::uint64_t> g_bound{0}, g_bound_pending{0}, g_bound_addr{0}, g_bound_ended{0}, g_fall[kFalls] = {}, g_fills{0},
    g_fill_skipped{0}, g_copies_in{0}, g_copies_staged{0}, g_copies_refused{0}, g_flushes{0}, g_flush_copies{0},
    g_flush_overwritten{0}, g_syncs{0}, g_sync_walk{0}, g_sync_edge{0}, g_sync_map{0}, g_taken_back{0}, g_ranges_freed{0},
    g_cmp_same{0}, g_cmp_diff{0}, g_cmp_none{0}, g_cmp_gpu{0}, g_bound_past{0}, g_cmp_past{0}, g_copies_replaced{0}, g_bound_flushed{0},
    g_addr_pending{0};
std::atomic<int> g_cmp_logged{0};

// ---- The command processor's (under g.mu) ----

std::uint64_t g_ev_seen = 0;
// The live homes by memory, a page filter in front (host/gxvk_homes_index.h).
HomeIndex g_ix;

struct Ending {
    std::uint32_t index;
    std::uint64_t flip;  // taken back at this flip
};
std::deque<Ending> g_ending;
struct Dead {
    std::uint64_t serial;  // free once this recording has completed
    std::int32_t block;
    VkDeviceSize offset;
    std::uint8_t cls;
    std::uint32_t index;
};
std::deque<Dead> g_dead;
constexpr std::uint64_t kEndFlips = 8;

// Copies into homes still to be put in place, oldest first, and by destination.
struct Pending {
    std::uint64_t dst = 0, bytes = 0;
    VkBuffer src = VK_NULL_HANDLE;
    VkDeviceSize src_off = 0;
    VkBuffer to = VK_NULL_HANDLE;
    VkDeviceSize to_off = 0;
    std::uint32_t index = 0;
};
std::vector<Pending> g_pending;
std::uint64_t g_pend_lo = ~0ull, g_pend_hi = 0;
// The same by destination, for the lookups by address (a vertex record, a constant buffer without a handle, any
// binding copy_version_lookup_locked is asked about): a binary search, not a walk of the list - homes lie all over
// GX's heaps, so the pending copies' bounds take in nearly every binding (OY: 16% of vertex records have no handle).
std::vector<std::pair<std::uint64_t, std::uint32_t>> g_pend_sorted;  // (dst, index into g_pending), by dst
std::uint64_t g_pend_maxlen = 0;
std::vector<VkBuffer> g_pend_bufs;  // the buffers the pending copies read or write (a few): a transfer's test
std::uint64_t g_pend_epoch = 1;  // moves at each flush: a home's pend_at is good in its own batch only
bool g_pend_overlap = false;     // two pending copies may share bytes: the flush keeps the newest's bytes

void index_erase(std::uint32_t index, Home& e) {
    g_ix.erase(index, e.memory);
    e.indexed = false;
}

// The index's answer for [va, va + bytes), kept per thread while the index stands (its version): a vertex record without
// a handle asks it three times a draw (the straddle test, locate's two), and the same records draw after draw, so the
// map is walked once per record and index change. The page filter's "no" comes first, uncached.
struct MeetMemo {
    std::uint64_t va = ~0ull, bytes = 0, version = ~0ull;
    bool hit = false;
    HomeIndex::Hit h;
};
thread_local MeetMemo t_meet[64];
bool ix_meets(std::uint64_t va, std::uint64_t bytes, HomeIndex::Hit* out) {
    if (!g_ix.may_meet(va, bytes)) return false;
    MeetMemo& m = t_meet[((va >> 4) ^ (va >> 10) ^ bytes) & 63];
    if (m.va != va || m.bytes != bytes || m.version != g_ix.version()) {
        m.hit = g_ix.meets(va, bytes, &m.h);
        m.va = va;
        m.bytes = bytes;
        m.version = g_ix.version();
    }
    if (m.hit && out) *out = m.h;
    return m.hit;
}

// A live home whose range meets [va, va + bytes): the one at or below va, else the first above it.
Home* index_meets(std::uint64_t va, std::uint64_t bytes, std::uint32_t* out_index) {
    HomeIndex::Hit hit;
    if (!ix_meets(va, bytes, &hit)) return nullptr;
    if (out_index) *out_index = hit.index;
    return home_at(hit.index);
}

void end_locked(std::uint32_t index, Home& e, int why);

void drain_locked() {
    // (A run part's draw - BBHOST_RUN_COMMIT - reads the index as the fork left it: homes_settled_locked refused the
    // fork while events were waiting.)
    if (g_ev_n.load(std::memory_order_acquire) == g_ev_seen || run_part()) return;
    std::vector<Event> events;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        events.swap(g_events);
        g_ev_seen = g_ev_n.load(std::memory_order_relaxed);
    }
    // The live homes meeting [va, +bytes) ended for `why` (their buffers' memory is another's now).
    const auto end_over = [](std::uint64_t va, std::uint64_t bytes, int why) {
        std::uint32_t oi = 0;
        while (Home* o = index_meets(va, bytes, &oi)) {
            std::uint32_t w = o->word.load(std::memory_order_acquire);
            if (state_of(w) == kLive && o->word.compare_exchange_strong(w, (w & ~0xffu) | kEnded, std::memory_order_acq_rel)) {
                o->end_why.store(static_cast<std::uint8_t>(why), std::memory_order_relaxed);
                g_ended[why].fetch_add(1, std::memory_order_relaxed);
            }
            end_locked(oi, *o, why);
        }
    };
    for (const Event& ev : events) {
        if (ev.kind == Event::kCovered) {
            end_over(ev.va, ev.bytes, ev.why);
        } else if (ev.kind == Event::kEnd) {
            if (Home* e = home_at(ev.index)) end_locked(ev.index, *e, e->end_why.load(std::memory_order_relaxed));
        } else {
            Home* e = home_at(ev.index);
            if (!e || state_of(e->word.load(std::memory_order_acquire)) != kLive || e->indexed) continue;
            end_over(e->memory, e->bytes, kEndReplaced);  // (an older home over its memory: its buffer's end unheard of)
            g_ix.insert(ev.index, e->memory, e->bytes);
            e->indexed = true;
        }
    }
}

void sync_one_locked(Home& e);

// Out of the index; taken back some flips on (a draw recorded before the end may still bind it by handle). A home a
// Map handed its own memory to is copied back to it first (the CPU reads or writes it there from now on).
void end_locked(std::uint32_t index, Home& e, int why) {
    if (e.end_queued) return;  // (twice: its end event after a newer home over its memory ended it)
    if ((why == kEndCpuMapped || why == kEndEdge) && e.filled.load(std::memory_order_acquire)) sync_one_locked(e);
    index_erase(index, e);
    e.end_queued = true;
    g_ending.push_back({index, hle_video_flip_count() + kEndFlips});
}

void free_range(std::int32_t block, VkDeviceSize offset, std::uint8_t cls) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_class_free[cls].emplace_back(block, offset);
    g_held -= std::min<std::uint64_t>(g_held, class_bytes(cls));
}

void recycle(std::uint32_t index) {
    Home* e = home_at(index);
    if (!e) return;
    delete[] e->mirror;
    e->mirror = nullptr;
    e->filled.store(false, std::memory_order_relaxed);  // (made again, it is published by its word's release)
    e->pending = 0;
    e->end_queued = false;
    e->indexed = false;
    std::lock_guard<std::mutex> lk(g_mu);
    const std::uint32_t gen = gen_of(e->word.load(std::memory_order_relaxed));
    e->word.store(((gen + 1) & 0xff) << 8 | kFree, std::memory_order_release);
    g_free_index.push_back(index);
}

// ---- Copies into homes and from them ----

// The newest pending copy meeting [base, +bytes): kCopyVersionInside with its source in `out` when it holds the range
// whole, kCopyVersionStraddles when it holds part of it.
int pending_lookup(std::uint64_t base, std::uint64_t bytes, Located* out) {
    if (g_pending.empty() || base >= g_pend_hi || base + bytes <= g_pend_lo) return kCopyVersionNone;
    // The newest pending copy meeting the range: those starting within the longest's reach below it.
    const std::uint64_t from = base > g_pend_maxlen ? base - g_pend_maxlen + 1 : 0;
    auto it = std::lower_bound(g_pend_sorted.begin(), g_pend_sorted.end(), std::make_pair(from, std::uint32_t{0}));
    const Pending* best = nullptr;
    std::uint32_t best_i = 0;
    for (; it != g_pend_sorted.end() && it->first < base + bytes; ++it) {
        const Pending& p = g_pending[it->second];
        if (p.dst + p.bytes <= base) continue;
        if (!best || it->second > best_i) {
            best = &p;
            best_i = it->second;
        }
    }
    if (!best) return kCopyVersionNone;
    if (base < best->dst || base + bytes > best->dst + best->bytes) return kCopyVersionStraddles;
    if (out) *out = {best->src, best->src_off + (base - best->dst), best->bytes - (base - best->dst)};
    return kCopyVersionInside;
}

// A home's bytes from its buffer's memory, in this recording's uploads (run
// before the recording's own work): its initial data, which nothing writes
// after the creation. Through staging, copied now - never from the import,
// where a GPU write of the memory's last user could still land.
bool fill_locked(Home& e) {
    if (!hle_kernel_va_mapped(e.memory, e.bytes)) return false;
    begin_recording_locked();
    const VkCommandBuffer cmd = shadow_upload_cmd_locked();
    if (!cmd) return false;
    DevBuffer sb;
    VkDeviceSize so = 0;
    if (!acquire_staging_locked(sb, e.bytes, so)) return false;
    std::memcpy(sb.map, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(e.memory)), e.bytes);
    const VkBufferCopy region{so, e.offset, e.bytes};
    vkCmdCopyBuffer(cmd, sb.buffer, e.buffer, 1, &region);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;  // before every use of the home in the recording
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (env_mode() == Mode::kCompare) {
        if (!e.mirror) e.mirror = new std::uint8_t[e.bytes];
        std::memcpy(e.mirror, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(e.memory)), e.bytes);
    }
    e.filled.store(true, std::memory_order_release);
    g_fills.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// The home ready for a copy of [dst, +bytes) into it: filled first, unless the copy writes the whole of it.
bool ready_for_copy(Home& e, std::uint64_t dst, std::uint64_t bytes) {
    if (e.filled.load(std::memory_order_acquire)) return true;
    if (dst == e.memory && bytes >= e.bytes) {
        e.filled.store(true, std::memory_order_release);
        g_fill_skipped.fetch_add(1, std::memory_order_relaxed);
        if (env_mode() == Mode::kCompare && !e.mirror) e.mirror = new std::uint8_t[e.bytes]();
        return true;
    }
    return fill_locked(e);
}

// The home's bytes copied back to its buffer's memory, at this point of the
// stream (pending copies into it first): what a reader of guest memory there
// reads from now on. The home stays the home.
void sync_one_locked(Home& e) {
    if (!e.filled.load(std::memory_order_acquire)) return;  // guest memory holds what the home would
    if (e.pending) {
        render_end_pass_locked();
        homes_flush_locked();
    }
    const Located guest = locate_guest(e.memory, e.bytes);
    if (!guest.buffer || guest.avail < e.bytes || !copy_in_stream_locked(e.buffer, e.offset, guest.buffer, guest.offset, e.bytes)) return;
    shadow_written_locked(e.memory, e.bytes, true);
    g_syncs.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

// ---- GX's threads ----

void homes_decide(bool integrated, bool small_vram, bool tight, std::uint64_t free_at_start) {
    Mode m = env_mode();
    const char* budget_env = std::getenv("BBHOST_GXVK_HOMES_BUDGET_MB");
    const bool budget_given = budget_env && *budget_env;
    const ull free_mib = free_at_start >> 20;
    // Small video memory: the homes from kSmallHomesFreeMiB free at the start, with the pool's budget; under it off
    // unless a budget is given (below).
    const bool small_off = small_refused(small_vram, free_mib, budget_given);
    // The default's rule (host/gxvk_homes.h default_off), asked with BBHOST_GXVK_HOMES=1 too: where it holds, the run
    // commit on by Windows' default alone gives way to the homes and the pool's segments (run_commit_gives_way), before
    // any draw. The frame rate was chosen as the game's code was patched, before the device.
    const unsigned threads = host_cpu_threads();
    const int fps = frame_rate_game_fps();
    const unsigned off = default_off(integrated, tight, small_off, threads, fps, list_resolve::commit_by_setting());
    const bool gave_way = run_commit_gives_way(off, list_resolve::commit_by_default(), m);
    if (gave_way) list_resolve::commit_give_way();
    // The default (BBHOST_GXVK_HOMES unset): on as 1 turns them on, or off with its reasons in one line.
    char by_default[448] = "";
    if (m == Mode::kAuto) {
        if (off) {
            std::string why;
            char b[160];
            const auto add = [&why](const char* w) {
                if (!why.empty()) why += ", ";
                why += w;
            };
            if (off & kDefaultOffIntegrated) add("an integrated GPU");
            if (off & kDefaultOffTight) add("tight memory");
            if (off & kDefaultOffSmall) {
                std::snprintf(b, sizeof(b), "small video memory, %llu MiB free at the start", free_mib);
                add(b);
            }
            if (off & kDefaultOffThreads) {
                std::snprintf(b, sizeof(b), "%u hardware threads", threads);
                add(b);
            }
            if (off & kDefaultOffRate) {
                std::snprintf(b, sizeof(b), "%d fps", fps);
                add(b);
            }
            if (off & kDefaultOffRunCommit) add("the run commit on by BBHOST_RUN_COMMIT, which the pool's segments give way to");
            host_log("gxvk homes: off by default here (%s) - the constant buffers' homes, and with them the draw segments prepared on "
                     "the pool's threads, are on by default only with a graphics card with room or %llu MiB of video memory free at the "
                     "start, %u or more hardware threads and a frame rate above %d (where the run commit is on by Windows' default, it "
                     "gives way to them); at %d and below the pool would take about a core more for nothing. Decided at start: a frame "
                     "rate changed in play applies at the next start. BBHOST_GXVK_HOMES=1 turns them on",
                     why.c_str(), kSmallHomesFreeMiB, kDefaultThreads, kDefaultAboveFps, kDefaultAboveFps);
            return;
        }
        const char* rc = std::getenv("BBHOST_RUN_COMMIT");
        char card[128];
        if (small_vram)
            std::snprintf(card, sizeof(card), "a graphics card with %llu MiB of video memory free at the start (%llu or more)", free_mib,
                          kSmallHomesFreeMiB);
        else
            std::snprintf(card, sizeof(card), "a graphics card with room");
        std::snprintf(by_default, sizeof(by_default),
                      "on by default: %s, %u hardware threads, %d fps, the run commit %s; decided at start, a frame rate changed in play "
                      "applies at the next start; BBHOST_GXVK_HOMES=0: off",
                      card, threads, fps,
                      gave_way    ? "off (on by Windows' default, it gives way to them; BBHOST_RUN_COMMIT=on keeps it on)"
                      : rc && *rc ? "off (BBHOST_RUN_COMMIT)"
                                  : "off");
        m = Mode::kHandle;
        cb_capture::set_on();          // the copy-backs captured at the resolve, as 1 has them
        list_resolve::follow_homes();  // and the run batching and the part machinery the segments need
    }
    if (m != Mode::kHandle && m != Mode::kCompare) return;
    if (integrated || tight) {
        host_log("gxvk homes: off on this GPU (%s: its video memory is the system's) - BBHOST_GXVK_HOMES takes today's path",
                 integrated ? "an integrated GPU" : "tight memory");
        return;
    }
    // Small video memory under kSmallHomesFreeMiB free: off unless a budget is given. Acting out a 4 GB card
    // (BBHOST_TEST_VRAM_MB=2950) the pool's 80 MiB left the GX copies' pool under the mirrors' cushion, the mirrors grew
    // and images went to host memory: OY's GPU time 4.7 -> 11.1 ms a flip, 99 -> 80 fps. A budget under the ~60 MiB a
    // world's constant buffers take would leave most of them on today's path anyway. An 8 GB card (7,249 MiB free under
    // Windows) took the 128 MiB budget with its frame rate up and its shared GPU memory unchanged (host/gxvk_homes.h).
    if (small_off) {
        host_log("gxvk homes: off with small video memory (%llu MiB free at the start, under %llu: their pool would take the room the "
                 "images and the mirrors need) - BBHOST_GXVK_HOMES takes today's path; BBHOST_GXVK_HOMES_BUDGET_MB=N turns them on with "
                 "that budget",
                 free_mib, kSmallHomesFreeMiB);
        return;
    }
    std::uint64_t mib = 128;
    if (budget_given) mib = std::strtoull(budget_env, nullptr, 10);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_budget = mib << 20;
    }
    g_bind.store(m == Mode::kHandle, std::memory_order_relaxed);
    g_made.store(true, std::memory_order_release);
    char small[96] = "";
    if (small_vram) std::snprintf(small, sizeof(small), " (small video memory, %llu MiB free at the start)", free_mib);
    host_log("gxvk homes: GX's constant buffers in device-local homes made at their creation, %s (%s); the pool's budget %llu MiB%s",
             m == Mode::kHandle ? "bound by handle" : "kept beside today's path and checked against it",
             by_default[0] ? by_default : m == Mode::kHandle ? "BBHOST_GXVK_HOMES=1" : "BBHOST_GXVK_HOMES=compare", static_cast<ull>(mib),
             small);
}

}  // namespace gpu

void host_gpu_home_made(std::uint64_t holder, std::uint64_t memory, std::uint32_t bytes, std::uint8_t usage, std::uint8_t cpu, std::uint8_t bind,
                        std::uint32_t spare) {
    using namespace gpu;
    if (!holder) return;
    // The holder's +0x14 is the creator's, just made, and ours to write: a home's handle or 0. A holder reused before
    // its buffer's end was heard of carried the old one's (`spare`), which ends here.
    const auto put = [holder](std::uint32_t w) { std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(holder + 0x14)), &w, 4); };
    if (gxvk_homes::is_handle(spare)) {
        put(spare);
        host_gpu_home_end(holder, gxvk_homes::kEndReplaced);
        put(0);
    }
    if (!gxvk_homes::made_on()) {
        const gxvk_homes::Mode m = gxvk_homes::env_mode();
        if (m == gxvk_homes::Mode::kHandle || m == gxvk_homes::Mode::kCompare || m == gxvk_homes::Mode::kAuto)
            g_no_home[gxvk_homes::kNoHomeNoDevice].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    using namespace gxvk_homes;
    int why = -1;
    // (A constant buffer bound as anything else too - vertex data, a view - is read by address, past its end: not a home.)
    if (bind != 4 || (usage != 0 && usage != 2)) why = (bind & 4) && bind != 4 ? kNoHomeOtherBind : kNoHomeNotConstant;
    else if (usage == 0 && cpu) why = kNoHomeCpu;
    else if (!memory || !bytes || bytes > (64u << 10)) why = kNoHomeSize;
    if (why >= 0) {
        g_no_home[why].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const int cls = class_of(bytes);
    const VkDeviceSize need = class_bytes(cls);
    std::uint32_t index = 0;
    std::int32_t block = -1;
    VkDeviceSize offset = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_held + need > g_budget) {
            why = kNoHomeBudget;
        } else if (!g_free_index.empty()) {
            index = g_free_index.back();
        } else if (g_next > kIndexMask) {
            why = kNoHomeTable;
        } else {
            index = g_next;  // taken below, once the range is
        }
        if (why < 0) {
            if (!g_class_free[cls].empty()) {
                block = g_class_free[cls].back().first;
                offset = g_class_free[cls].back().second;
                g_class_free[cls].pop_back();
            } else {
                if (g_blocks.empty() || g_blocks.back().carved + need > g_blocks.back().buf.size) {
                    Block nb;
                    MemOwner own_mem("gxvk homes");
                    if (!create_dev_buffer(nb.buf, kBlockBytes, false) || nb.buf.in_host) {
                        if (nb.buf.buffer) {
                            vkDestroyBuffer(g.device, nb.buf.buffer, nullptr);
                            vkFreeMemory(g.device, nb.buf.memory, nullptr);
                            g_blocks_refused.fetch_add(1, std::memory_order_relaxed);
                        }
                        why = kNoHomeNoRoom;
                    } else {
                        g_blocks.push_back(std::move(nb));
                        g_blocks_made.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (why < 0) {
                    block = static_cast<std::int32_t>(g_blocks.size() - 1);
                    offset = g_blocks.back().carved;
                    g_blocks.back().carved += need;
                }
            }
        }
        if (why < 0) {
            if (!g_free_index.empty() && g_free_index.back() == index) {
                g_free_index.pop_back();
            } else {
                const std::uint32_t c = g_next >> kChunkBits;
                if (!g_chunks[c].load(std::memory_order_relaxed)) g_chunks[c].store(new Home[1u << kChunkBits], std::memory_order_release);
                ++g_next;
            }
            g_held += need;
            g_held_peak = std::max(g_held_peak, g_held);
            Home& e = *home_at(index);
            e.holder = holder;
            e.memory = memory;
            e.bytes = bytes;
            e.cls = static_cast<std::uint8_t>(cls);
            e.block = block;
            e.buffer = g_blocks[static_cast<std::size_t>(block)].buf.buffer;
            e.offset = offset;
            e.address = g_blocks[static_cast<std::size_t>(block)].buf.address + offset;
            e.end_why.store(0, std::memory_order_relaxed);
            const std::uint32_t gen = gen_of(e.word.load(std::memory_order_relaxed));
            e.word.store(gen << 8 | kLive, std::memory_order_release);
            g_events.push_back({Event::kMade, 0, index, 0, 0});
            g_ev_n.fetch_add(1, std::memory_order_release);
            put(handle_of(index, gen));
        }
    }
    if (why >= 0) {
        g_no_home[why].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_made_n.fetch_add(1, std::memory_order_relaxed);
    g_made_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void host_gpu_home_end(std::uint64_t holder, int why) {
    using namespace gpu;
    if (!holder) return;
    std::uint32_t h = 0;
    std::memcpy(&h, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(holder + 0x14)), 4);
    if (!gxvk_homes::is_handle(h)) return;
    Home* e = home_at(gxvk_homes::handle_index(h));
    if (!e || e->holder != holder) return;
    std::uint32_t w = e->word.load(std::memory_order_acquire);
    const std::uint32_t to = why == gxvk_homes::kEndCpuMapped ? kMapped : kEnded;
    if (gen_of(w) != gxvk_homes::handle_gen(h) || state_of(w) != kLive || !e->word.compare_exchange_strong(w, (w & ~0xffu) | to, std::memory_order_acq_rel))
        return;
    e->end_why.store(static_cast<std::uint8_t>(why), std::memory_order_relaxed);
    g_ended[why].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_mu);
    g_events.push_back({Event::kEnd, 0, gxvk_homes::handle_index(h), 0, 0});
    g_ev_n.fetch_add(1, std::memory_order_release);
}

void host_gpu_home_covered(std::uint64_t va, std::uint64_t bytes, int why) {
    using namespace gpu;
    if (!gxvk_homes::made_on() || !bytes) return;
    std::lock_guard<std::mutex> lk(g_mu);
    g_events.push_back({Event::kCovered, static_cast<std::uint8_t>(why), 0, va, bytes});
    g_ev_n.fetch_add(1, std::memory_order_release);
}

namespace gpu {

// ---- The command processor's ----

void homes_begin_recording_locked() {
    if (!gxvk_homes::made_on()) return;
    drain_locked();
    const std::uint64_t flip = hle_video_flip_count();
    while (!g_ending.empty() && g_ending.front().flip <= flip) {
        const std::uint32_t index = g_ending.front().index;
        g_ending.pop_front();
        Home* e = home_at(index);
        if (!e) continue;
        // (A copy still pending into it goes into place first: its range is free only after the work that reads it.)
        if (e->pending) {
            render_end_pass_locked();
            homes_flush_locked();
        }
        g_dead.push_back({g.flushes, e->block, e->offset, e->cls, index});
        g_taken_back.fetch_add(1, std::memory_order_relaxed);
    }
    while (!g_dead.empty() && g_dead.front().serial < g.completed_submits) {
        const Dead d = g_dead.front();
        g_dead.pop_front();
        free_range(d.block, d.offset, d.cls);
        recycle(d.index);
        g_ranges_freed.fetch_add(1, std::memory_order_relaxed);
    }
}

bool homes_holding_locked(std::uint64_t va, std::uint64_t bytes, Located* out, VkDeviceAddress* address) {
    if (!gxvk_homes::made_on() || !bytes) return false;
    drain_locked();
    if (!g_ix.may_meet(va, bytes)) return false;
    std::uint32_t index = 0;
    Home* e = index_meets(va, bytes, &index);
    if (!e || va < e->memory || va + bytes > e->memory + e->bytes) return false;
    if (!bind_on()) return false;  // compare: today's path is bound
    if (!e->filled.load(std::memory_order_acquire) && (run_part_stop(kRunStopRefresh) || !fill_locked(*e))) return false;
    if (e->pending) bump(g_addr_pending);  // (a copy into it pending: the callers that bind ask the copies first - counted)
    if (out) *out = {e->buffer, e->offset + (va - e->memory), class_bytes(e->cls) - (va - e->memory)};
    if (address) *address = e->address + (va - e->memory);
    g_bound_addr.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool homes_meet_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_bind.load(std::memory_order_relaxed) || !bytes) return false;
    drain_locked();
    return ix_meets(va, bytes, nullptr);
}

bool homes_straddle_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_bind.load(std::memory_order_relaxed) || !bytes) return false;
    drain_locked();
    HomeIndex::Hit h;
    return ix_meets(va, bytes, &h) && (va < h.memory || va + bytes > h.memory + h.bytes);
}

void homes_sync_locked(std::uint64_t va, std::uint64_t bytes, int why) {
    if (!g_bind.load(std::memory_order_relaxed) || !bytes) return;
    drain_locked();
    if (!g_ix.may_meet(va, bytes)) return;
    std::uint64_t at = va;
    for (int n = 0; n < 4096 && at < va + bytes; ++n) {
        Home* e = index_meets(at, va + bytes - at, nullptr);
        if (!e) break;
        sync_one_locked(*e);
        (why == 0 ? g_sync_walk : why == 1 ? g_sync_edge : g_sync_map).fetch_add(1, std::memory_order_relaxed);
        at = e->memory + e->bytes;
    }
}

void homes_end_meeting_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_bind.load(std::memory_order_relaxed) || !bytes || run_part()) return;
    drain_locked();
    if (!g_ix.may_meet(va, bytes)) return;
    std::uint32_t index = 0;
    for (int n = 0; n < 4096; ++n) {
        Home* e = index_meets(va, bytes, &index);
        if (!e) break;
        std::uint32_t w = e->word.load(std::memory_order_acquire);
        if (state_of(w) == kLive && e->word.compare_exchange_strong(w, (w & ~0xffu) | kMapped, std::memory_order_acq_rel)) {
            e->end_why.store(kEndEdge, std::memory_order_relaxed);
            g_ended[kEndEdge].fetch_add(1, std::memory_order_relaxed);
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 12)
                host_log("gxvk homes: %s at 0x%llx + %llu meets the home of 0x%llx (%u bytes) across its edge: copied back, the buffer on today's "
                         "path from now on",
                         t_import_site ? t_import_site : "a GPU access", static_cast<ull>(va), static_cast<ull>(bytes), static_cast<ull>(e->memory),
                         e->bytes);
        }
        end_locked(index, *e, kEndEdge);  // (copied back first; out of the index)
    }
}

int homes_bind_locked(std::uint32_t handle, std::uint64_t base, std::uint64_t bytes, Located& out) {
    Home* e = home_at(gxvk_homes::handle_index(handle));
    if (!e) return kHomeFall + static_cast<int>(kFallStale);
    const std::uint32_t w = e->word.load(std::memory_order_acquire);
    const std::uint32_t st = state_of(w);
    if (gen_of(w) != gxvk_homes::handle_gen(handle) || (st != kLive && st != kEnded)) {
        bump(g_fall[st == kMapped ? kFallRetired : kFallStale]);
        return kHomeFall + static_cast<int>(st == kMapped ? kFallRetired : kFallStale);
    }
    if (base < e->memory || base - e->memory >= e->bytes || bytes > class_bytes(e->cls) - (base - e->memory)) {
        bump(g_fall[kFallRange]);
        return kHomeFall + static_cast<int>(kFallRange);
    }
    if (!e->filled.load(std::memory_order_acquire)) {
        if (run_part_stop(kRunStopRefresh)) {  // (stage 4's census notes a fill as before-work a stitcher does)
            bump(g_fall[kFallPart]);
            return kHomeFall + static_cast<int>(kFallPart);
        }
        if (!fill_locked(*e)) {
            bump(g_fall[kFallFill]);
            return kHomeFall + static_cast<int>(kFallFill);
        }
    }
    if (t_run_draw && t_run_draw->list) {
        // Stage 4b.1b's list mode: bound by handle, with no copy pending into
        // it at its splice either (homes_splice_ok); one pending now is in order.
        if (e->pending) {
            run_part_stop(kRunStopOrder);
            bump(g_fall[kFallPart]);
            return kHomeFall + static_cast<int>(kFallPart);
        }
        t_run_draw->list->note_home(handle);
    } else if (e->pending) {
        // A copy into it pending in this pass: the draw reads what the copy writes, from its source.
        if (pending_lookup(base, bytes, &out) == kCopyVersionInside) {
            bump(g_bound_pending);
            return kHomePending;
        }
        if (run_part_stop(kRunStopOrder)) {  // (one across a pending copy's edge: in place first, in order)
            bump(g_fall[kFallPart]);
            return kHomeFall + static_cast<int>(kFallPart);
        }
        render_end_pass_locked();
        homes_flush_locked();
        bump(g_bound_flushed);
    }
    out = {e->buffer, e->offset + (base - e->memory), class_bytes(e->cls) - (base - e->memory)};
    bump(st == kEnded ? g_bound_ended : g_bound);
    if (bytes > e->bytes - (base - e->memory)) bump(g_bound_past);  // (a shader reading past the buffer's end: its class's room)
    return kHomeBound;
}

bool homes_splice_ok(std::uint32_t handle) {
    const Home* e = home_at(gxvk_homes::handle_index(handle));
    if (!e) return false;
    const std::uint32_t w = e->word.load(std::memory_order_acquire);
    const std::uint32_t st = state_of(w);
    return gen_of(w) == gxvk_homes::handle_gen(handle) && (st == kLive || st == kEnded) && e->filled.load(std::memory_order_acquire) &&
           !e->pending;
}

int homes_copy_in_locked(std::uint64_t dst, std::uint64_t bytes, VkBuffer src, VkDeviceSize src_off, const void* src_map) {
    if (!gxvk_homes::made_on() || !bytes) return kHomeCopyNotHome;
    drain_locked();
    if (!g_ix.may_meet(dst, bytes)) return kHomeCopyNotHome;
    std::uint32_t index = 0;
    Home* e = index_meets(dst, bytes, &index);
    if (!e) return kHomeCopyNotHome;
    if (dst < e->memory || dst + bytes > e->memory + e->bytes) {
        // Across a home's edge (none expected: a copy-back writes its buffer whole): the home copied back and ended, the
        // copy as today.
        homes_end_meeting_locked(dst, bytes);
        bump(g_copies_refused);
        return kHomeCopyRefused;
    }
    if (!ready_for_copy(*e, dst, bytes)) {
        bump(g_copies_refused);
        return kHomeCopyRefused;
    }
    if (e->mirror && src_map) std::memcpy(e->mirror + (dst - e->memory), src_map, bytes);
    bump(g_copies_in);
    // A newer copy over the same bytes takes the older one's place, as its batch would leave only it (a copy-back writes
    // its buffer whole: a constant buffer copied back twice between two passes is one copy). A draw that bound the older
    // one's source keeps it.
    if (e->pending && e->pend_epoch == g_pend_epoch && e->pend_at < g_pending.size()) {
        Pending& p = g_pending[e->pend_at];
        if (p.index == index && p.dst == dst && p.bytes == bytes) {
            p.src = src;
            p.src_off = src_off;
            if (src && std::find(g_pend_bufs.begin(), g_pend_bufs.end(), src) == g_pend_bufs.end()) g_pend_bufs.push_back(src);
            bump(g_copies_replaced);
            return kHomeCopyTaken;
        }
        g_pend_overlap = true;
    }
    e->pend_at = static_cast<std::uint32_t>(g_pending.size());
    e->pend_epoch = g_pend_epoch;
    g_pending.push_back({dst, bytes, src, src_off, e->buffer, e->offset + (dst - e->memory), index});
    {
        const std::pair<std::uint64_t, std::uint32_t> key{dst, static_cast<std::uint32_t>(g_pending.size() - 1)};
        g_pend_sorted.insert(std::upper_bound(g_pend_sorted.begin(), g_pend_sorted.end(), key), key);
        g_pend_maxlen = std::max(g_pend_maxlen, bytes);
        for (const VkBuffer b : {src, e->buffer}) {
            if (b && std::find(g_pend_bufs.begin(), g_pend_bufs.end(), b) == g_pend_bufs.end()) g_pend_bufs.push_back(b);
        }
    }
    g_pend_lo = std::min(g_pend_lo, dst);
    g_pend_hi = std::max(g_pend_hi, dst + bytes);
    ++e->pending;
    return kHomeCopyTaken;
}

int homes_copy_in_staged_locked(std::uint64_t dst, std::uint64_t bytes, const void* from) {
    if (!gxvk_homes::made_on() || !bytes) return kHomeCopyNotHome;
    drain_locked();
    if (!g_ix.may_meet(dst, bytes)) return kHomeCopyNotHome;
    Home* e = index_meets(dst, bytes, nullptr);
    if (!e) return kHomeCopyNotHome;
    DevBuffer sb;
    VkDeviceSize so = 0;
    if (dst >= e->memory && dst + bytes <= e->memory + e->bytes && acquire_staging_locked(sb, bytes, so)) {
        std::memcpy(sb.map, from, bytes);
        const int r = homes_copy_in_locked(dst, bytes, sb.buffer, so, sb.map);
        if (r == kHomeCopyTaken) bump(g_copies_staged);
        return r;
    }
    return homes_copy_in_locked(dst, bytes, VK_NULL_HANDLE, 0, nullptr) == kHomeCopyNotHome ? kHomeCopyNotHome : kHomeCopyRefused;
}

int homes_pending_lookup_locked(std::uint64_t base, std::uint64_t bytes, Located* out) {
    if (!g_bind.load(std::memory_order_relaxed)) return kCopyVersionNone;
    return pending_lookup(base, bytes, out);
}

bool homes_pending_locked() { return !g_pending.empty(); }

bool homes_settled_locked() {
    if (!g_bind.load(std::memory_order_relaxed)) return true;
    drain_locked();
    return g_ev_n.load(std::memory_order_acquire) == g_ev_seen;
}

bool homes_before_transfer_locked(VkBuffer buffer, VkDeviceSize off, VkDeviceSize bytes) {
    if (std::find(g_pend_bufs.begin(), g_pend_bufs.end(), buffer) == g_pend_bufs.end()) return false;
    for (const Pending& p : g_pending) {
        if ((p.to == buffer && p.to_off < off + bytes && off < p.to_off + p.bytes) ||
            (p.src == buffer && p.src_off < off + bytes && off < p.src_off + p.bytes)) {
            homes_flush_locked();
            return true;
        }
    }
    return false;
}

// The pending copies into place: a batch of their own (no pass is open; the callers see to it), newest first, each
// byte from the newest copy that has it - copies in one batch run in any order.
void homes_flush_locked() {
    if (g_pending.empty()) return;
    begin_recording_locked();
    static std::vector<Pending> pending;
    pending.clear();
    pending.swap(g_pending);
    g_pend_sorted.clear();
    g_pend_bufs.clear();
    g_pend_maxlen = 0;
    g_pend_lo = ~0ull;
    g_pend_hi = 0;
    ++g_pend_epoch;
    const bool overlap = g_pend_overlap;
    g_pend_overlap = false;
    std::optional<DrawCmds> own;
    if (!DrawCmds::open() && !stream_on()) own.emplace(!g.profile && !g.has_checkpoints);
    if (DrawCmds* c = DrawCmds::open(); !c || !c->transfer_begin()) rec().transfer_barrier(true);
    std::size_t copies = 0;
    if (!overlap) {
        // One pending copy a home (the usual batch): no two share bytes, so they run in any order.
        for (const Pending& p : pending) {
            if (Home* e = home_at(p.index); e && e->pending) --e->pending;
            const VkBufferCopy region{p.src_off, p.to_off, p.bytes};
            DrawCmds* c = DrawCmds::open();
            if (!c || !c->copy_buffer(p.src, p.to, region)) rec().copy_buffer(p.src, p.to, 1, &region);
            g.transfers.fetch_add(1);
            ++copies;
        }
        pending.clear();
        if (DrawCmds* c = DrawCmds::open(); !c || !c->transfer_end()) rec().transfer_barrier(false);
        g_flushes.fetch_add(1, std::memory_order_relaxed);
        g_flush_copies.fetch_add(copies, std::memory_order_relaxed);
        return;
    }
    static std::map<std::uint64_t, std::uint64_t> covered;  // start -> end, disjoint, of the newer copies
    covered.clear();
    for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
        const Pending& p = *it;
        if (Home* e = home_at(p.index); e && e->pending) --e->pending;
        const std::uint64_t end = p.dst + p.bytes;
        auto copy_piece = [&](std::uint64_t lo, std::uint64_t hi) {
            const VkBufferCopy region{p.src_off + (lo - p.dst), p.to_off + (lo - p.dst), hi - lo};
            DrawCmds* c = DrawCmds::open();
            if (!c || !c->copy_buffer(p.src, p.to, region)) rec().copy_buffer(p.src, p.to, 1, &region);
            g.transfers.fetch_add(1);
            ++copies;
        };
        auto c = covered.upper_bound(p.dst);
        if (c != covered.begin() && std::prev(c)->second > p.dst) --c;
        for (std::uint64_t at = p.dst; at < end;) {
            if (c == covered.end() || c->first >= end) {
                copy_piece(at, end);
                break;
            }
            if (c->first > at) copy_piece(at, c->first);
            at = std::max(at, c->second);
            ++c;
        }
        std::uint64_t lo = p.dst, hi = end;
        auto m = covered.upper_bound(lo);
        if (m != covered.begin() && std::prev(m)->second >= lo) --m;
        while (m != covered.end() && m->first <= hi) {
            lo = std::min(lo, m->first);
            hi = std::max(hi, m->second);
            m = covered.erase(m);
        }
        covered.emplace(lo, hi);
    }
    if (DrawCmds* c = DrawCmds::open(); !c || !c->transfer_end()) rec().transfer_barrier(false);
    g_flushes.fetch_add(1, std::memory_order_relaxed);
    g_flush_copies.fetch_add(copies, std::memory_order_relaxed);
    g_flush_overwritten.fetch_add(pending.size() > copies ? pending.size() - copies : 0, std::memory_order_relaxed);
    pending.clear();
}

void homes_compare_locked(std::uint32_t handle, std::uint64_t base, std::uint64_t bytes, const void* today) {
    Home* e = home_at(gxvk_homes::handle_index(handle));
    if (!e) return;
    const std::uint32_t w = e->word.load(std::memory_order_acquire);
    if (gen_of(w) != gxvk_homes::handle_gen(handle) || state_of(w) == kFree || base < e->memory || base >= e->memory + e->bytes) {
        bump(g_cmp_none);
        return;
    }
    if (base + bytes > e->memory + e->bytes) {  // past the buffer's end: the bytes it holds compared
        bump(g_cmp_past);
        bytes = e->memory + e->bytes - base;
    }
    if (!e->filled.load(std::memory_order_acquire)) fill_locked(*e);
    if (!today || !e->mirror) {
        bump(today ? g_cmp_none : g_cmp_gpu);
        return;
    }
    if (std::memcmp(e->mirror + (base - e->memory), today, bytes) == 0) {
        bump(g_cmp_same);
        return;
    }
    bump(g_cmp_diff);
    if (g_cmp_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
        std::size_t at = 0;
        const auto* a = e->mirror + (base - e->memory);
        const auto* b = static_cast<const std::uint8_t*>(today);
        while (at < bytes && a[at] == b[at]) ++at;
        host_log("gxvk homes compare: the home of 0x%llx (%u bytes) would bind other bytes than today's path at 0x%llx + %llu: first at "
                 "+%zu (home %02x, today %02x)%s",
                 static_cast<ull>(e->memory), e->bytes, static_cast<ull>(base), static_cast<ull>(bytes), at, a[at], b[at],
                 e->pending ? ", a copy into it pending" : "");
    }
}

std::string homes_report() {
    const Mode m = env_mode() == Mode::kAuto && made_on() ? Mode::kHandle : env_mode();
    if (m != Mode::kHandle && m != Mode::kCompare) return {};
    static const char* const kNoName[kNoHomes] = {"not a constant buffer of default or dynamic usage", "bound as something else too", "CPU access", "size",
                                                  "the pool at its budget", "no device memory", "the table full", "before the device"};
    static const char* const kEndName[kEnds] = {"released", "replaced", "unmapped", "mapped by the CPU", "a GPU access across its edge or a walk's stores"};
    static const char* const kFallName[kFalls] = {"stale handle", "retired", "outside the home", "a run part's fill", "no bytes to fill from"};
    std::string nh, en, fl;
    char b[160];
    for (int k = 0; k < kNoHomes; ++k) {
        if (const std::uint64_t n = g_no_home[k].load()) {
            std::snprintf(b, sizeof(b), "%s%s %llu", nh.empty() ? "" : ", ", kNoName[k], static_cast<ull>(n));
            nh += b;
        }
    }
    for (int k = 0; k < kEnds; ++k) {
        if (const std::uint64_t n = g_ended[k].load()) {
            std::snprintf(b, sizeof(b), "%s%s %llu", en.empty() ? "" : ", ", kEndName[k], static_cast<ull>(n));
            en += b;
        }
    }
    for (int k = 0; k < kFalls; ++k) {
        if (const std::uint64_t n = g_fall[k].load()) {
            std::snprintf(b, sizeof(b), "%s%s %llu", fl.empty() ? "" : ", ", kFallName[k], static_cast<ull>(n));
            fl += b;
        }
    }
    std::uint64_t held = 0, peak = 0, blocks = 0;
    std::size_t live = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        held = g_held;
        peak = g_held_peak;
        blocks = g_blocks.size();
    }
    {
        std::lock_guard<GpuMutex> lock(g.mu);
        live = g_ix.size();
    }
    char out[1800];
    std::snprintf(out, sizeof(out),
                  "gxvk homes: %s; made %llu (%llu KiB of constants), none for: %s; %zu live; the pool %llu blocks (%llu MiB), ranges held %llu KiB "
                  "now, %llu KiB at most; ended: %s; taken back %llu, ranges freed %llu\n"
                  "gxvk homes: bindings by handle %llu (%llu after their buffer's end, %llu reading past the buffer's bytes), of a pending "
                  "copy's source %llu, after putting the copies in place first %llu, by address %llu (%llu with a copy pending); by handle to "
                  "today's path: %s; filled from their memory %llu, a whole copy-back first "
                  "%llu; copies in %llu (%llu staged, %llu refused, %llu taking a pending one's place) in %llu batches (%llu copies, %llu "
                  "overwritten); copied back to their memory %llu (walks %llu, edges %llu, CPU maps %llu)",
                  m == Mode::kCompare                  ? "compared beside today's path (BBHOST_GXVK_HOMES=compare)"
                  : env_mode() == Mode::kAuto ? "bound by handle (on by default)"
                                                       : "bound by handle (BBHOST_GXVK_HOMES=1)",
                  static_cast<ull>(g_made_n.load()), static_cast<ull>(g_made_bytes.load() >> 10), nh.empty() ? "none" : nh.c_str(), live,
                  static_cast<ull>(blocks), static_cast<ull>((blocks * kBlockBytes) >> 20), static_cast<ull>(held >> 10),
                  static_cast<ull>(peak >> 10), en.empty() ? "none" : en.c_str(), static_cast<ull>(g_taken_back.load()),
                  static_cast<ull>(g_ranges_freed.load()), static_cast<ull>(g_bound.load()), static_cast<ull>(g_bound_ended.load()),
                  static_cast<ull>(g_bound_past.load()), static_cast<ull>(g_bound_pending.load()), static_cast<ull>(g_bound_flushed.load()),
                  static_cast<ull>(g_bound_addr.load()), static_cast<ull>(g_addr_pending.load()), fl.empty() ? "none" : fl.c_str(),
                  static_cast<ull>(g_fills.load()), static_cast<ull>(g_fill_skipped.load()), static_cast<ull>(g_copies_in.load()),
                  static_cast<ull>(g_copies_staged.load()), static_cast<ull>(g_copies_refused.load()), static_cast<ull>(g_copies_replaced.load()),
                  static_cast<ull>(g_flushes.load()),
                  static_cast<ull>(g_flush_copies.load()), static_cast<ull>(g_flush_overwritten.load()), static_cast<ull>(g_syncs.load()),
                  static_cast<ull>(g_sync_walk.load()), static_cast<ull>(g_sync_edge.load()), static_cast<ull>(g_sync_map.load()));
    std::string s = out;
    if (m == Mode::kCompare) {
        std::snprintf(out, sizeof(out),
                      "\ngxvk homes compare: bindings of a home today's path made: %llu the same bytes as the home's, %llu different; not "
                      "compared: today's bytes still to land on the GPU (a GPU copy at a token, a version across an edge) %llu, no live home "
                      "%llu; %llu of them reading past the buffer's end (its bytes compared)",
                      static_cast<ull>(g_cmp_same.load()), static_cast<ull>(g_cmp_diff.load()), static_cast<ull>(g_cmp_gpu.load()),
                      static_cast<ull>(g_cmp_none.load()), static_cast<ull>(g_cmp_past.load()));
        s += out;
    }
    return s;
}

}  // namespace gpu
