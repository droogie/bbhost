// GX on Vulkan, stage 1: the command processor's side of GX buffers by
// handle (host/gxvk.h, host/gxvk_table.h).
//
// An entry's copy is placed at the first draw that binds it: a range of a
// device-local pool block (64 MiB blocks, first fit at 256-byte steps; a
// larger buffer a block of its own) while video memory is above the reserve
// kept for render targets and device buffers. Short of room the entry is
// retired and its buffer keeps today's path - resolved by address, as the
// buffer shadow serves it, which gives its mirrors back and takes them again
// as memory comes and goes - never a copy in host memory on a discrete GPU:
// one placed there during a dip stayed for the session, and every draw of it
// read across the bus (Nightmare of Mensis with 6 GB: the GPU 18.3 ms a flip
// against 5.7). On an integrated GPU host memory is the same RAM, and a
// host-visible block is still taken there once video memory is short.
// Its bytes come from the buffer's memory in the recording's uploads - the
// buffer shadow's command buffer, which runs before the recording's own work
// (buffer_shadow.cpp) - from the import, or with window imports through
// upload staging. Nothing earlier in the recording read the range: it is
// new, or its last user's work has run (a retired entry's range is free
// again only once the recording that retired it has completed).
//
// With BBHOST_GXVK_BUFFERS=2 every binding takes today's path, and the
// handle's is checked beside it: that the handle names the holder's live
// entry and covers the range, that today's path finds the bytes in guest
// memory as they stand (no copy version, draw window or snapshot over them),
// that the vertex input planned from the handles is today's, that the
// buffer's bytes still hash as they did when copied (the first bindings,
// then one in 64), and that the device copy holds those bytes (read back
// when the recording has run). A disagreement is counted by kind, logged
// the first times, and retires the entry.
#include "host/gpu_internal.h"
#include "host/gxvk_table.h"
#include "hle/modules.h"
#include "log.h"

#include <utility>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace gpu {

namespace {

using ull = unsigned long long;
using gxvk::BufferEntry;

constexpr VkDeviceSize kBlockBytes = 64ull << 20, kAlign = 256;

struct Block {
    DevBuffer buf;
    std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size: disjoint, merged with their neighbours
    VkDeviceSize free_bytes = 0;
    bool host = false;       // host-visible memory (video memory was down to the reserve, or full)
    bool dedicated = false;  // one buffer's, larger than a block: let go once free
};
std::vector<Block> g_blocks;  // under g.mu
struct Dead {
    std::uint64_t serial;  // the recording that let it go: free once that has completed
    int block;
    VkDeviceSize offset, size;
};
std::deque<Dead> g_dead;  // under g.mu, oldest first

// Counters (the command processor writes them, the report reads them).
std::atomic<std::uint64_t> g_placed[2], g_placed_bytes[2], g_blocks_made[2], g_block_bytes[2], g_uploads{0}, g_upload_bytes{0},
    g_staged{0}, g_ranges_freed{0}, g_taken_back{0};
std::atomic<std::uint64_t> g_blocks_refused{0};  // device-local blocks that went to host memory, given back (discrete GPU)
std::atomic<std::uint64_t> g_disagree[gxvk_kDisagrees], g_checks{0}, g_readbacks{0}, g_readback_bytes{0}, g_readbacks_skipped{0};
std::atomic<int> g_disagree_logged{0};
std::uint64_t g_barrier_serial = ~0ull;  // the recording whose uploads have the source barrier

const char* const kDisagreeName[gxvk_kDisagrees] = {"a copy version over the bytes", "the bytes in a draw window", "the bytes snapshotted",
                                                    "another vertex input", "the guest bytes changed since the copy",
                                                    "the device copy differs", "a draw reading past the buffer's end"};
std::atomic<std::uint64_t> g_overrun{0};  // vertex records whose V# counts past the buffer's end, bound to its copy's end
const char* const kFallName[gxvk::kFalls] = {"no GX object", "not kept", "a handle of no entry", "taken back since the call",
                                             "retired by a write", "past the buffer's end", "no memory", "bytes out of reach",
                                             "indirect", "a binding shared with another buffer"};
// Entries retired by their end, kept bindable for the draws recorded before
// it, until the flip given (8 on from the retirement's taking).
constexpr std::uint64_t kEndFlips = 8;
std::deque<std::pair<std::uint32_t, std::uint64_t>> g_ending;  // under g.mu: index, the flip it is taken back at

std::uint64_t hash_bytes(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    std::uint64_t h = 0x9e3779b97f4a7c15ull ^ n;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t v;
        std::memcpy(&v, b + i, 8);
        h = (h ^ v) * 0xff51afd7ed558ccdull;
        h ^= h >> 29;
    }
    for (; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

bool compare_mode() { return gxvk::buffers_mode() == gxvk::Buffers::kCompare; }

void range_free_locked(int block, VkDeviceSize offset, VkDeviceSize size) {
    Block& bl = g_blocks[static_cast<std::size_t>(block)];
    bl.free_bytes += size;
    auto it = bl.free.emplace(offset, size).first;
    if (const auto nx = std::next(it); nx != bl.free.end() && it->first + it->second == nx->first) {
        it->second += nx->second;
        bl.free.erase(nx);
    }
    if (it != bl.free.begin()) {
        if (const auto pv = std::prev(it); pv->first + pv->second == it->first) {
            pv->second += it->second;
            bl.free.erase(it);
        }
    }
    g_ranges_freed.fetch_add(1, std::memory_order_relaxed);
    // A block of one buffer's goes once that buffer's range is free.
    if (bl.dedicated && bl.free_bytes == bl.buf.size) {
        defer_destroy(bl.buf);
        g_block_bytes[bl.host].fetch_sub(bl.buf.size, std::memory_order_relaxed);
        bl = Block{};
    }
}

void reclaim_locked() {
    while (!g_dead.empty() && g_dead.front().serial < g.completed_submits) {
        const Dead d = g_dead.front();
        g_dead.pop_front();
        range_free_locked(d.block, d.offset, d.size);
    }
}

bool take_from_locked(int b, VkDeviceSize need, BufferEntry& e) {
    Block& bl = g_blocks[static_cast<std::size_t>(b)];
    for (auto it = bl.free.begin(); it != bl.free.end(); ++it) {
        if (it->second < need) continue;
        const VkDeviceSize off = it->first, rest = it->second - need;
        bl.free.erase(it);
        if (rest) bl.free.emplace(off + need, rest);
        bl.free_bytes -= need;
        e.vk_buffer = reinterpret_cast<std::uint64_t>(bl.buf.buffer);
        e.vk_offset = off;
        e.block = b;
        return true;
    }
    return false;
}

int new_block_locked(VkDeviceSize size, bool host, bool dedicated) {
    Block nb;
    MemOwner own_mem("gxvk buffer copies");
    if (!create_dev_buffer(nb.buf, size, host)) return -1;
    // A device-local block that found video memory full went to host memory
    // (create_dev_buffer's fallback): not on a discrete GPU (above).
    if (!host && nb.buf.in_host && !g.integrated) {
        vkDestroyBuffer(g.device, nb.buf.buffer, nullptr);
        vkFreeMemory(g.device, nb.buf.memory, nullptr);
        g_blocks_refused.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    nb.host = host || nb.buf.in_host;
    nb.dedicated = dedicated;
    nb.free.emplace(0, size);
    nb.free_bytes = size;
    g_blocks_made[nb.host].fetch_add(1, std::memory_order_relaxed);
    g_block_bytes[nb.host].fetch_add(size, std::memory_order_relaxed);
    for (std::size_t b = 0; b < g_blocks.size(); ++b) {
        if (!g_blocks[b].buf.buffer) {
            g_blocks[b] = std::move(nb);
            return static_cast<int>(b);
        }
    }
    g_blocks.push_back(std::move(nb));
    return static_cast<int>(g_blocks.size() - 1);
}

// Whether placing `need` more would reach the reserve. BBHOST_TEST_GXVK_NO_ROOM=N
// (tests): once N copies are placed, it would, as in a dip.
bool pool_short_locked(VkDeviceSize need) {
    static const long long after = [] {
        const char* e = std::getenv("BBHOST_TEST_GXVK_NO_ROOM");
        return e && *e ? std::atoll(e) : -1;
    }();
    if (after >= 0 && static_cast<long long>(g_placed[0].load(std::memory_order_relaxed) + g_placed[1].load(std::memory_order_relaxed)) >= after)
        return true;
    return vram_within_reserve(need);
}

// A range for `bytes`: video memory while it is above the reserve; host-visible
// memory after it only on an integrated GPU (above). False: no room.
bool pool_place_locked(VkDeviceSize bytes, BufferEntry& e) {
    reclaim_locked();
    const bool host_ok = g.integrated;
    const VkDeviceSize need = (std::max<VkDeviceSize>(bytes, 1) + kAlign - 1) & ~(kAlign - 1);
    if (need > kBlockBytes) {
        const bool room = !pool_short_locked(need);
        int b = room ? new_block_locked(need, false, true) : -1;
        if (b < 0 && host_ok) b = new_block_locked(need, true, true);
        return b >= 0 && take_from_locked(b, need, e);
    }
    for (int host = 0; host < (host_ok ? 2 : 1); ++host) {
        for (std::size_t b = 0; b < g_blocks.size(); ++b) {
            const Block& bl = g_blocks[b];
            if (!bl.buf.buffer || bl.dedicated || bl.host != (host != 0) || bl.free_bytes < need) continue;
            if (take_from_locked(static_cast<int>(b), need, e)) return true;
        }
        if (host == 0 && !pool_short_locked(kBlockBytes)) {
            const int b = new_block_locked(kBlockBytes, false, false);
            if (b >= 0 && take_from_locked(b, need, e)) return true;
        }
    }
    if (!host_ok) return false;
    const int b = new_block_locked(kBlockBytes, true, false);
    return b >= 0 && take_from_locked(b, need, e);
}

void let_go_locked(BufferEntry& e) {
    if (e.block >= 0) {
        g_dead.push_back({g.flushes, e.block, e.vk_offset, (std::max<VkDeviceSize>(e.bytes, 1) + kAlign - 1) & ~(kAlign - 1)});
    }
    e.block = -1;
    e.vk_buffer = 0;
    e.vk_offset = 0;
}

// Compare mode's read-backs of device copies, by the slot that made them.
struct Readback {
    DevBuffer buf;
    std::uint64_t hash = 0, bytes = 0;
    std::uint32_t handle = 0;
};
std::vector<Readback> g_readback_slots[kSlots];
std::uint64_t g_readback_serial = ~0ull, g_readback_serial_bytes = 0;
constexpr std::uint64_t kReadbackBytesPerRecording = 64ull << 20;

// The buffer's bytes into its copy, in this recording's uploads.
bool fill_locked(BufferEntry& e, std::uint32_t handle) {
    begin_recording_locked();
    const VkCommandBuffer cmd = shadow_upload_cmd_locked();
    if (!cmd) return false;
    VkBuffer src = VK_NULL_HANDLE;
    VkDeviceSize so = 0;
    if (!imports_whole()) {
        if (!hle_kernel_va_mapped(e.memory, e.bytes)) return false;
        DevBuffer sb;
        if (!acquire_staging_locked(sb, e.bytes, so)) return false;
        std::memcpy(sb.map, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(e.memory)), e.bytes);  // sb.map: the placement's own
        src = sb.buffer;
        g_staged.fetch_add(1, std::memory_order_relaxed);
    } else {
        const Located guest = locate_guest(e.memory, e.bytes);
        if (!guest.buffer || guest.avail < e.bytes) return false;
        src = guest.buffer;
        so = guest.offset;
    }
    if (g_barrier_serial != g.flushes) {
        // Earlier submissions' writes of the sources before these reads.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        g_barrier_serial = g.flushes;
    }
    const VkBuffer dst = reinterpret_cast<VkBuffer>(e.vk_buffer);
    const VkBufferCopy region{so, e.vk_offset, e.bytes};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    g_uploads.fetch_add(1, std::memory_order_relaxed);
    g_upload_bytes.fetch_add(e.bytes, std::memory_order_relaxed);
    if (compare_mode()) {
        e.hash = hash_bytes(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(e.memory)), e.bytes);
        // The copy read back once the recording has run (a capped share of
        // each recording's bytes).
        if (g_readback_serial != g.flushes) g_readback_serial = g.flushes, g_readback_serial_bytes = 0;
        Readback rb;
        MemOwner own_mem("gxvk readbacks");
        if (g_readback_serial_bytes + e.bytes <= kReadbackBytesPerRecording && create_dev_buffer(rb.buf, e.bytes, true, true)) {
            g_readback_serial_bytes += e.bytes;
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            const VkBufferCopy back{e.vk_offset, 0, e.bytes};
            vkCmdCopyBuffer(cmd, dst, rb.buf.buffer, 1, &back);
            // The read-back made available to the host, which reads it once
            // the slot's fence has signalled (the fence alone does not).
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            rb.hash = e.hash;
            rb.bytes = e.bytes;
            rb.handle = handle;
            g_readback_slots[g.slot].push_back(rb);
        } else {
            g_readbacks_skipped.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return true;
}

// The entry's copy placed and filled: gxvk::kFalls, else why not (no
// memory, its bytes out of reach - the entry retired - or retired meanwhile).
int place_locked(BufferEntry& e, std::uint32_t handle) {
    const std::uint32_t index = gxvk::handle_index(handle), gen = gxvk::handle_gen(handle);
    if (!pool_place_locked(e.bytes, e)) {
        gxvk::retire(index, gen, gxvk::kRetireNoRoom);
        return gxvk::kFallNoRoom;
    }
    if (!fill_locked(e, handle)) {
        let_go_locked(e);
        gxvk::retire(index, gen, gxvk::kRetireNoSource);
        return gxvk::kFallNoSource;
    }
    e.placed_serial = g.flushes;
    std::uint32_t want = (gen << 8) | gxvk::kEntryMade;
    if (!e.word.compare_exchange_strong(want, (gen << 8) | gxvk::kEntryPlaced, std::memory_order_acq_rel)) {
        let_go_locked(e);  // retired while it was placed: the range goes back once this recording has run
        return gxvk::kFallRetired;
    }
    const Block& bl = g_blocks[static_cast<std::size_t>(e.block)];
    g_placed[bl.host].fetch_add(1, std::memory_order_relaxed);
    g_placed_bytes[bl.host].fetch_add(e.bytes, std::memory_order_relaxed);
    return gxvk::kFalls;
}

// A compare run's disagreement: counted, the first logged, and the entry
// retired - but for another vertex input, which binds the same bytes
// through other bindings (today's path put two records in one buffer: a
// staging chunk, a snapshot ring) and costs only a pipeline of its own.
void disagree_locked(int kind, std::uint32_t handle, const BufferEntry* e) {
    g_disagree[kind].fetch_add(1, std::memory_order_relaxed);
    if (g_disagree_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
        host_log("gxvk buffers: compare: %s - buffer 0x%llx (memory 0x%llx, %u bytes, usage %u, bind 0x%x)", kDisagreeName[kind],
                 static_cast<ull>(e ? e->holder : 0), static_cast<ull>(e ? e->memory : 0), e ? e->bytes : 0u, e ? e->usage : 0u,
                 e ? e->bind : 0u);
    }
    if (kind != gxvk_kLayout) gxvk::retire(gxvk::handle_index(handle), gxvk::handle_gen(handle), gxvk::kRetireCompare);
}

}  // namespace

bool gxvk_buffers_bind_mode() { return gxvk::buffers_mode() == gxvk::Buffers::kOn; }

namespace {
thread_local bool g_last_ended = false;  // the last resolve on this thread bound an entry retired by its end
}  // namespace
bool gxvk_last_resolve_ended() { return g_last_ended; }

int gxvk_resolve_locked(int use, std::uint32_t handle, std::uint64_t holder, std::uint64_t va, std::uint64_t bytes, Located& out) {
    g_last_ended = false;
    BufferEntry* e = gxvk::entry_at(gxvk::handle_index(handle));
    if (!e) return gxvk::kFallStale;
    const std::uint32_t w = e->word.load(std::memory_order_acquire);
    if (gxvk::entry_gen(w) != gxvk::handle_gen(handle)) return gxvk::kFallRecycled;
    if (e->holder != holder) return gxvk::kFallStale;
    const std::uint32_t st = gxvk::entry_state(w);
    // Retired by its end: the copy, if there is one, is what the draw reads.
    const bool ended = st == gxvk::kEntryRetired && gxvk::retired_by_end(e->retired_why.load(std::memory_order_relaxed)) && e->vk_buffer;
    if (st == gxvk::kEntryRetired && !ended) return gxvk::kFallRetired;
    if (st != gxvk::kEntryMade && st != gxvk::kEntryPlaced && !ended) return gxvk::kFallStale;
    if (va < e->memory || va - e->memory >= e->bytes) return gxvk::kFallRange;
    if (bytes > e->bytes - (va - e->memory)) {
        if (use != gxvk::kUseVertex) return gxvk::kFallRange;
        bump(g_overrun);
    }
    if (st == gxvk::kEntryMade) {
        // A run part's draw places no copy (BBHOST_RUN_COMMIT): placed in order.
        if (run_part_stop(kRunStopMirror)) return gxvk::kFallStale;
        if (const int r = place_locked(*e, handle); r != gxvk::kFalls) return r;
    }
    out.buffer = reinterpret_cast<VkBuffer>(e->vk_buffer);
    out.offset = e->vk_offset + (va - e->memory);
    out.avail = e->bytes - (va - e->memory);
    if (ended) {
        g_last_ended = true;
        bump(gxvk::g_bound_after_end[use]);
        return gxvk::kFalls;  // its memory may hold another's bytes by now: no hash check
    }
    if (compare_mode()) {
        // The buffer's bytes as they were copied: the first bindings, then one in 64.
        const std::uint32_t n = e->binds++;
        if (n < 2 || (n & 63) == 0) {
            g_checks.fetch_add(1, std::memory_order_relaxed);
            if (hle_kernel_va_mapped(e->memory, e->bytes) &&
                hash_bytes(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(e->memory)), e->bytes) != e->hash) {
                disagree_locked(gxvk_kGuestChanged, handle, e);
                return gxvk::kFallRetired;
            }
        }
    }
    return gxvk::kFalls;
}

void gxvk_count_locked(int use, int result) {
    if (result == gxvk::kFalls) bump(gxvk::g_bound[use]);
    else if (result >= 0 && result < gxvk::kFalls) bump(gxvk::g_fallback[use][result]);
}

void gxvk_disagree_locked(int kind, std::uint32_t handle) {
    disagree_locked(kind, handle, gxvk::entry_at(gxvk::handle_index(handle)));
}

void gxvk_begin_recording_locked() {
    if (!gxvk::buffers_on()) return;
    const std::uint64_t flip = hle_video_flip_count();
    const auto take_back = [](std::uint32_t index) {
        if (BufferEntry* e = gxvk::entry_at(index)) let_go_locked(*e);
        gxvk::recycle(index);
        g_taken_back.fetch_add(1, std::memory_order_relaxed);
    };
    std::uint32_t taken[256];
    for (;;) {
        const std::size_t n = gxvk::take_retired(taken, 256);
        for (std::size_t k = 0; k < n; ++k) {
            BufferEntry* e = gxvk::entry_at(taken[k]);
            // One retired by its end, copied: bound for the draws before its
            // end that are still to come, taken back some flips on.
            if (e && e->vk_buffer && gxvk::retired_by_end(e->retired_why.load(std::memory_order_relaxed))) {
                g_ending.emplace_back(taken[k], flip + kEndFlips);
            } else {
                take_back(taken[k]);
            }
        }
        if (n < 256) break;
    }
    while (!g_ending.empty() && g_ending.front().second <= flip) {
        take_back(g_ending.front().first);
        g_ending.pop_front();
    }
    reclaim_locked();
}

void gxvk_slot_retired_locked(int slot) {
    if (slot < 0 || slot >= kSlots || g_readback_slots[slot].empty()) return;
    for (Readback& rb : g_readback_slots[slot]) {
        g_readbacks.fetch_add(1, std::memory_order_relaxed);
        g_readback_bytes.fetch_add(rb.bytes, std::memory_order_relaxed);
        if (rb.buf.map && hash_bytes(rb.buf.map, rb.bytes) != rb.hash) {
            disagree_locked(gxvk_kDeviceCopy, rb.handle, gxvk::entry_at(gxvk::handle_index(rb.handle)));
        }
        vkDestroyBuffer(g.device, rb.buf.buffer, nullptr);
        vkFreeMemory(g.device, rb.buf.memory, nullptr);
    }
    g_readback_slots[slot].clear();
}

std::string gxvk_report() {
    if (!gxvk::buffers_on()) return {};
    std::string s;
    char b[512];
    const auto bound = [](int use) { return gxvk::g_bound[use].load(); };
    const auto falls = [](int use, bool gx_only) {
        std::uint64_t n = 0;
        for (int k = gx_only ? 1 : 0; k < gxvk::kFalls; ++k) n += gxvk::g_fallback[use][k].load();
        return n;
    };
    for (int use = gxvk::kUseVertex; use <= gxvk::kUseIndex; ++use) {
        const std::uint64_t by = bound(use), all = by + falls(use, false), gx = by + falls(use, true);
        std::snprintf(b, sizeof b,
                      "%sgxvk buffers: %s by handle %llu (%.1f%% of all, %.1f%% of those with a GX object; %llu after their buffer's end); "
                      "today's path:",
                      s.empty() ? "" : "\n", use == gxvk::kUseVertex ? "vertex records" : "index ranges", static_cast<ull>(by),
                      all ? 100.0 * static_cast<double>(by) / static_cast<double>(all) : 0.0,
                      gx ? 100.0 * static_cast<double>(by) / static_cast<double>(gx) : 0.0,
                      static_cast<ull>(gxvk::g_bound_after_end[use].load()));
        s += b;
        bool any = false;
        for (int k = 0; k < gxvk::kFalls; ++k) {
            const std::uint64_t n = gxvk::g_fallback[use][k].load();
            if (!n) continue;
            std::snprintf(b, sizeof b, "%s %s %llu", any ? "," : "", kFallName[k], static_cast<ull>(n));
            s += b;
            any = true;
        }
        if (!any) s += " none";
    }
    std::snprintf(b, sizeof b,
                  "\ngxvk buffers: copies placed %llu in video memory (%llu MiB), %llu in host memory (%llu MiB); pool blocks made %llu + %llu, "
                  "holding %llu + %llu MiB now; %llu uploads (%llu MiB, %llu through staging); ranges freed %llu, entries taken back %llu; "
                  "vertex bindings whose V# counts past the buffer's end %llu; %s",
                  static_cast<ull>(g_placed[0].load()), static_cast<ull>(g_placed_bytes[0].load() >> 20), static_cast<ull>(g_placed[1].load()),
                  static_cast<ull>(g_placed_bytes[1].load() >> 20), static_cast<ull>(g_blocks_made[0].load()), static_cast<ull>(g_blocks_made[1].load()),
                  static_cast<ull>(g_block_bytes[0].load() >> 20), static_cast<ull>(g_block_bytes[1].load() >> 20), static_cast<ull>(g_uploads.load()),
                  static_cast<ull>(g_upload_bytes.load() >> 20), static_cast<ull>(g_staged.load()), static_cast<ull>(g_ranges_freed.load()),
                  static_cast<ull>(g_taken_back.load()), static_cast<ull>(g_overrun.load()),
                  g.integrated ? "host memory once video memory is short (an integrated GPU)"
                               : ("no copy in host memory (a discrete GPU): blocks refused there " + std::to_string(g_blocks_refused.load())).c_str());
    s += b;
    if (compare_mode()) {
        std::uint64_t total = 0;
        std::string kinds;
        for (int k = 0; k < gxvk_kDisagrees; ++k) {
            const std::uint64_t n = g_disagree[k].load();
            total += n;
            if (!n) continue;
            std::snprintf(b, sizeof b, "%s%s %llu", kinds.empty() ? "" : ", ", kDisagreeName[k], static_cast<ull>(n));
            kinds += b;
        }
        std::snprintf(b, sizeof b,
                      "\ngxvk buffers: compare: %llu disagreements%s%s%s; %llu hash checks of guest bytes, %llu device copies read back (%llu MiB, "
                      "%llu not read back: the per-recording cap)",
                      static_cast<ull>(total), kinds.empty() ? "" : " (", kinds.c_str(), kinds.empty() ? "" : ")",
                      static_cast<ull>(g_checks.load()), static_cast<ull>(g_readbacks.load()), static_cast<ull>(g_readback_bytes.load() >> 20),
                      static_cast<ull>(g_readbacks_skipped.load()));
        s += b;
    }
    return s;
}

}  // namespace gpu
