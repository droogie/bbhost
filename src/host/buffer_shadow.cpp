// Device-local copies of the guest memory draws read as vertex and index data.
//
// Guest direct memory reaches the GPU as imported host memory (gpu.cpp's 1 GiB
// chunks), so every vertex fetch and index read crossed PCIe: in the world at
// a 60 fps target the GPU pulled ~7 GB/s from the host, ~120 MB a frame, with
// its shader units 98% busy and its own memory 8% - waiting on system-memory
// latency. The same meshes are read every frame and almost never change.
//
// Here each dmem chunk gets a device-local mirror, allocated when a draw first
// reads from it, holding guest memory at the same offset as the import, kept
// in 64 KiB pages. A page a draw reads (from a vertex record's extent or an
// index range) is copied into the mirror once and read from there. It stays
// valid while:
//
//   - it takes no CPU write through the guest's view (core/write_watch, as the
//     texture cache does, checked once per recording): a write marks it dirty
//     and the next draw that reads it copies it again;
//   - the command processor writes nothing over it (DMA, WRITE_DATA: on the
//     CPU, through record() in gnm_exec.cpp): the same;
//   - the GPU writes nothing over it (copy and fill dispatches, copy tokens):
//     that write lands later in the command stream than any copy made now
//     could, so such a page is never mirrored again.
//
// A page copied more than kMaxUploads times in kUploadWindowMs holds data
// the game keeps rewriting (rings of dynamic geometry): it, and every draw
// that reads it, stays in host memory from then on. The window is eight
// seconds, not one: a ring the game rewrites every half second to two - the
// vertex data it builds on the CPU (0x2cdd6b9), say - never reached four
// copies in a second, and each pass over it cost a copy and a write-watch
// fault per 4 KiB written (~98% of a soak's ~200k faults). All the state is per
// page, so a draw's check is an array index per page it touches; ranges keyed
// by address grew without bound (ring allocations are new addresses every
// frame) and walking them cost ~1 us a draw.
//
// Copies go into a command buffer of their own that is submitted just before
// the draws' (submit_locked): a barrier at its start keeps them after the
// reads earlier submissions make, one at its end before this submission's
// reads. Copying a dirty range there makes the draws recorded before the write
// read the new contents too - which is also what reading the host memory at
// execution time gave them.
//
// BBHOST_BUFFER_SHADOW=0 binds everything from the imports, as before, and so
// do an integrated GPU and tight memory unless BBHOST_BUFFER_SHADOW=1
// (buffer_shadow_on).

#include "core/write_watch.h"
#include "host/gpu_internal.h"
#include "log.h"

#include <chrono>
#include <cstring>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <utility>

namespace gpu {

namespace {

constexpr std::uint32_t kMaxUploads = 4;
constexpr std::uint64_t kUploadWindowMs = 8000;
constexpr unsigned kPageShift = 16;  // 64 KiB: validity, write watches and copies go by these
constexpr std::uint64_t kPageBytes = 1ull << kPageShift;
constexpr std::uint32_t kPagesPerChunk = static_cast<std::uint32_t>(kChunkBytes >> kPageShift);

// BBHOST_SHADOW_VERIFY=1: every submission re-hashes a batch of valid pages
// against their contents when copied, and reports (and stops mirroring) any
// that changed without a write this cache saw - a write it cannot track,
// such as a compute shader's store through the GPU page table.
const bool g_verify = [] {
    const char* e = std::getenv("BBHOST_SHADOW_VERIFY");
    return e && e[0] == '1';
}();
std::uint64_t g_verify_cursor = 0, g_verify_checked = 0, g_verify_stale = 0;
// Verify mode: each page's contents when copied, so a stale page's report
// says which bytes changed and to what.
std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> g_verify_copies;

const int g_shadow_env = [] {
    const char* e = std::getenv("BBHOST_BUFFER_SHADOW");
    return !e || !*e ? -1 : e[0] == '0' ? 0 : 1;
}();

// A 64 KiB page of a chunk's mirror: everything is tracked here, so a draw's
// check is an array index per page. Ranges that overlap share their pages, so
// a vertex pool read from a thousand bases is copied once.
struct Page {
    WriteWatch watch;
    WriteWatch alias_watch;           // over the GPU alias, which GX writes through (unarmed when there is none)
    std::uint64_t va = 0, bytes = 0;  // the guest view the watch is armed on (the page within one mapping)
    std::uint64_t alias_va = 0;       // where alias_watch is armed
    std::uint64_t checked = 0;        // the recording (record_serial + 1) it was last checked for writes in
    std::uint64_t window_ms = 0;
    std::uint32_t window_uploads = 0;
    std::uint64_t gpu_written = 0;    // the recording a GPU write over it was recorded in (0: none)
    std::uint64_t hash = 0;           // BBHOST_SHADOW_VERIFY: its contents when copied
    bool valid = false;               // the mirror holds it (or will, once this submission's copies run)
    bool dynamic = false;             // bound from the import from now on (rewritten, unwatchable)
};

struct Mirror {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    bool failed = false;
    std::vector<Page> pages;
    // A bit per page: checked in the recording `ok_stamp` and valid. Vertex
    // records span up to hundreds of pages (pools), and a draw whose pages
    // are all set costs a few word tests instead of a walk.
    std::vector<std::uint64_t> ok;
    std::uint64_t ok_stamp = 0;
    // A bit per page for what sends a draw to host memory, so that draw
    // takes a few word tests too instead of a walk over the pages' records:
    // bound from host memory for good (Page::dynamic), and written by the
    // GPU in the recording `ok_stamp` (Page::gpu_written; cleared with ok).
    std::vector<std::uint64_t> dyn, gpu;
    // Sparse: a 64 KiB device page is bound the first time the page is
    // copied, from 64 MiB blocks; the binds queue up until the submission.
    bool sparse = false;
    std::uint32_t mem_type_bits = 0;
    std::vector<VkDeviceMemory> blocks;
    std::uint32_t block_pages_used = 0;  // in blocks.back()
    std::vector<std::uint64_t> resident;  // a bit per page
    std::vector<VkSparseMemoryBind> pending;
    std::uint64_t resident_pages = 0;
};
constexpr std::uint32_t kBlockPages = 1024;  // 64 MiB blocks
VkSemaphore g_bind_sem[kSlots] = {};

bool all_ok(const std::vector<std::uint64_t>& ok, std::uint32_t p0, std::uint32_t p1) {
    for (std::uint32_t w = p0 >> 6; w <= p1 >> 6; ++w) {
        std::uint64_t mask = ~0ull;
        if (w == p0 >> 6) mask &= ~0ull << (p0 & 63);
        if (w == p1 >> 6) mask &= ~0ull >> (63 - (p1 & 63));
        if ((ok[w] & mask) != mask) return false;
    }
    return true;
}

bool any_set(const std::vector<std::uint64_t>& bits, std::uint32_t p0, std::uint32_t p1) {
    for (std::uint32_t w = p0 >> 6; w <= p1 >> 6; ++w) {
        std::uint64_t mask = ~0ull;
        if (w == p0 >> 6) mask &= ~0ull << (p0 & 63);
        if (w == p1 >> 6) mask &= ~0ull >> (63 - (p1 & 63));
        if (bits[w] & mask) return true;
    }
    return false;
}

Mirror g_mirrors[gcn::kDmemChunks];
// The video memory the mirrors may hold: g.local_budget less kImageRoom, the
// room kept for images and targets - what the longest sessions' took in the
// world at 1080p (5.8 GiB) and some to spare. A full
// mirror is a whole chunk (sparse ones run only on NVIDIA's driver): on an
// 8 GB AMD card, about 7 GiB of budget, four of them left the world's targets
// no memory and the world drew black. BBHOST_SHADOW_VRAM_MB sets the share
// instead; once the mirrors are given back (shadow_give_back_locked) it is 0.
constexpr std::uint64_t kImageRoom = 6656ull << 20;
std::uint64_t g_mirror_cap = 0;    // shadow_budget_locked
std::uint64_t g_mirror_bytes = 0;  // video memory the mirrors hold: whole mirrors and sparse blocks
bool g_given_back = false;
std::uint64_t g_initial_cap = 0;     // the share at the start, which mirrors that come back get again
std::uint64_t g_given_back_ms = 0;   // when video memory last ran short (shadow_give_back_locked)
std::uint64_t g_gives = 0, g_returns = 0;
// The live share (shadow_live): a mirror block or a whole mirror is made only
// while video memory keeps the reserve and this cushion above it for the
// next area's textures (BBHOST_SHADOW_CUSHION_MB, 256); a page refused for
// want of room is read from host memory by that draw and tried again later.
bool g_live = false;
std::uint64_t g_cushion = 256ull << 20;
std::uint64_t g_room_refusals = 0;     // copies refused for want of room, counted per draw
std::uint64_t g_room_retry_ms = 0;     // no room until then: refused without asking (now_ms)
std::uint64_t g_room_skips = 0;        // draws read from host memory while refused, before arming anything
std::uint64_t g_first_refusal_ms = 0;  // the first refusal since the mirrors last started over (0: none)
std::uint64_t g_last_flush_ms = 0;     // when they last started over
std::uint64_t g_flushes = 0;           // the mirrors started over at a load
std::uint64_t g_second_chances = 0;    // dynamic pages given another try at a load (shadow_second_chance_locked)
std::uint64_t g_flush_refusals = 0, g_flush_held_back = 0, g_flush_gives = 0;  // the counts at the last flush
VkCommandBuffer g_upload_cmd[kSlots] = {};
int g_upload_slot = -1;  // the slot whose upload buffer is recording, or -1
std::size_t g_last_map = 0;  // g.maps index of the last hit

std::atomic<std::uint64_t> g_hits{0}, g_uploads{0}, g_upload_bytes{0}, g_dynamic{0}, g_imports{0}, g_gpu_written{0};
std::uint64_t g_page_checks = 0, g_invalid_cp = 0, g_invalid_gpu = 0;
std::uint64_t g_span_hist[6] = {};  // calls by pages spanned: 1, 2-4, 5-16, 17-64, 65-256, more

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Guest VA -> (chunk, offset) through the snapshot of direct-memory mappings,
// with the mapping's bounds; false outside one mapping and one chunk.
bool phys_of(std::uint64_t va, std::uint64_t bytes, std::uint32_t& chunk, std::uint64_t& off, std::uint64_t* map_lo = nullptr,
             std::uint64_t* map_hi = nullptr) {
    // The last hit first, with no walk: draws come in runs, and nearly all
    // of them land in the one big direct mapping.
    const std::size_t n = g.maps.size();
    std::size_t i = g_last_map;
    if (i >= n || va < g.maps[i].va || va >= g.maps[i].va + g.maps[i].len) {
        for (i = 0; i < n; ++i) {
            if (va >= g.maps[i].va && va < g.maps[i].va + g.maps[i].len) break;
        }
        if (i == n) return false;
        g_last_map = i;
    }
    const GuestMapInfo& mi = g.maps[i];
    if (g_import_audit) {
        import_audit(mi, 2);
        audit_range(va, bytes, kUseMirror);
    }
    if (!mi.dmem || va + bytes > mi.va + mi.len) return false;
    const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + (va - mi.va);
    chunk = static_cast<std::uint32_t>(phys >> gcn::kDmemChunkShift);
    off = phys & (kChunkBytes - 1);
    if (map_lo) *map_lo = mi.va;
    if (map_hi) *map_hi = mi.va + mi.len;
    return chunk < gcn::kDmemChunks && dmem_import(chunk, off, bytes) != nullptr;
}

// Whether the mirrors may take another `bytes` of video memory now: within
// the share, and with the live share only while video memory keeps the
// reserve and the cushion above it. A refusal holds for 250 ms, so the draws
// in between pay no budget reading.
bool mirror_room(std::uint64_t bytes) {
    if (g_mirror_bytes + bytes > g_mirror_cap) return false;
    if (!g_live) return true;
    const std::uint64_t now = now_ms();
    if (now < g_room_retry_ms) return false;
    if (vram_room_for(bytes, g_cushion)) return true;
    g_room_retry_ms = now + 250;
    if (!g_first_refusal_ms) g_first_refusal_ms = now;
    return false;
}

Mirror* mirror_for(std::uint32_t chunk) {
    Mirror& m = g_mirrors[chunk];
    if (m.buffer || m.failed) return m.failed ? nullptr : &m;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kChunkBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    VkMemoryRequirements req{};
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    if (g.has_sparse) {
        // A sparse buffer the size of the chunk, no memory yet: pages are
        // bound as they are copied. The sparse block must divide our page.
        bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
        if (vkCreateBuffer(g.device, &bci, nullptr, &m.buffer) == VK_SUCCESS) {
            vkGetBufferMemoryRequirements(g.device, m.buffer, &req);
            if (req.alignment == 0 || (kPageBytes % req.alignment) != 0 ||
                find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == UINT32_MAX) {
                host_log("buffer shadow: sparse block %llu does not fit a %llu-byte page; full mirrors instead",
                         static_cast<unsigned long long>(req.alignment), static_cast<unsigned long long>(kPageBytes));
                vkDestroyBuffer(g.device, m.buffer, nullptr);
                m.buffer = VK_NULL_HANDLE;
            } else {
                m.sparse = true;
                m.mem_type_bits = req.memoryTypeBits;
                m.resident.assign(kPagesPerChunk / 64, 0);
                m.pages.resize(kPagesPerChunk);
                m.ok.assign(kPagesPerChunk / 64, 0);
                m.dyn.assign(kPagesPerChunk / 64, 0);
                m.gpu.assign(kPagesPerChunk / 64, 0);
                host_log("buffer shadow: sparse mirror for chunk %u (pages bound as copied)", chunk);
                return &m;
            }
        } else {
            m.buffer = VK_NULL_HANDLE;
        }
        bci.flags = 0;
    }
    if (g_live && g_mirror_bytes + kChunkBytes <= g_mirror_cap && !mirror_room(kChunkBytes)) {
        // No room for a whole mirror now: this draw reads host memory, and
        // a later one asks again.
        ++g_room_refusals;
        return nullptr;
    }
    if (g_mirror_bytes + kChunkBytes > g_mirror_cap) {
        m.failed = true;
        static bool said = false;
        if (!said && !g_given_back) {
            said = true;
            host_log("buffer shadow: no mirror for chunk %u: another %llu MiB would pass the mirrors' share of video memory (%llu MiB, "
                     "%llu held); a chunk without one is read from host memory",
                     chunk, static_cast<unsigned long long>(kChunkBytes >> 20), static_cast<unsigned long long>(g_mirror_cap >> 20),
                     static_cast<unsigned long long>(g_mirror_bytes >> 20));
        }
        return nullptr;
    }
    if (vkCreateBuffer(g.device, &bci, nullptr, &m.buffer) != VK_SUCCESS) {
        m.buffer = VK_NULL_HANDLE;
    } else {
        vkGetBufferMemoryRequirements(g.device, m.buffer, &req);
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &m.memory) != VK_SUCCESS) {
            vkDestroyBuffer(g.device, m.buffer, nullptr);
            m.buffer = VK_NULL_HANDLE;
        } else {
            vkBindBufferMemory(g.device, m.buffer, m.memory, 0);
            g_mirror_bytes += req.size;
        }
    }
    if (!m.buffer) {
        m.failed = true;
        host_log("buffer shadow: no device-local mirror for chunk %u; its vertex and index data stay in host memory", chunk);
        return nullptr;
    }
    m.pages.resize(kPagesPerChunk);
    m.ok.assign(kPagesPerChunk / 64, 0);
    m.dyn.assign(kPagesPerChunk / 64, 0);
    m.gpu.assign(kPagesPerChunk / 64, 0);
    host_log("buffer shadow: device-local mirror for chunk %u (%llu MiB)", chunk,
             static_cast<unsigned long long>(kChunkBytes >> 20));
    return &m;
}

// ---- per-draw copies of vertex ranges left in host memory ----------------------
// A page the game keeps rewriting is read from host memory for good
// (go_dynamic), and a draw on vertex input then fetches it across PCIe vertex
// by vertex: two Scaleform pipelines went from 72 to 183 and from 64 to 97 us a
// draw on an RTX 4070 when their vertex tables first came from memory
// (render.cpp vertex_table_from_memory). shadow_stage_locked copies such a
// range for its draw alone into device memory, in the uploads that run just
// ahead of the draw's submission: the copy holds what the draw used to read
// from host memory as it ran, and the uploads' closing barrier orders it before
// the vertex fetch. The copies are packed into device-local chunks, which a
// slot holds from its first upload until its upload buffer is begun again -
// by then every submission that read them has retired.
struct StageChunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};
constexpr VkDeviceSize kStageChunkBytes = 4ull << 20;
constexpr VkDeviceSize kStageMaxBytes = 1ull << 20;  // a larger range stays in host memory
constexpr std::size_t kStageMaxChunks = 32;          // 128 MiB of video memory at most
std::vector<StageChunk> g_stage_free;
std::vector<StageChunk> g_stage_held[kSlots];  // each slot's chunks, the last the one being filled
VkDeviceSize g_stage_fill[kSlots] = {};        // bytes used in the slot's last chunk
std::size_t g_stage_chunks = 0;                // chunks made
std::atomic<std::uint64_t> g_staged{0}, g_staged_bytes{0}, g_stage_refused{0};
// One copy of a range for every draw of a submission that reads it. The
// uploads all run before the submission's first draw, so two copies of one
// range made for two of its draws read the same bytes; the second draw reads
// the first one's. An RTX 5080's log copied ~160,000 ranges, 2.7 GB, every
// 300 flips standing still (9-37 MiB a frame) - on the 3D queue, ahead of the
// draws, and outside the GPU busy meter's spans (Task Manager's 3D engine at
// 100% where the meter said 64%). Keyed by the range's place in the import
// (buffer, offset); a range of the same start no longer than one copied is
// read from it. BBHOST_STAGE_DEDUP=0: a copy per draw, as before.
struct StagedRange {
    VkBuffer buffer = VK_NULL_HANDLE;  // the stage chunk holding it
    VkDeviceSize offset = 0, bytes = 0;
};
struct StageKey {
    VkBuffer src;
    VkDeviceSize offset;
    bool operator==(const StageKey& o) const { return src == o.src && offset == o.offset; }
};
struct StageKeyHash {
    std::size_t operator()(const StageKey& k) const {
        return std::hash<std::uint64_t>()(reinterpret_cast<std::uint64_t>(k.src) * 0x9e3779b97f4a7c15ull ^ k.offset);
    }
};
std::unordered_map<StageKey, StagedRange, StageKeyHash> g_stage_seen[kSlots];
std::atomic<std::uint64_t> g_stage_shared{0}, g_stage_shared_bytes{0};
const bool g_stage_dedup = [] {
    const char* e = std::getenv("BBHOST_STAGE_DEDUP");
    return !(e && e[0] == '0');
}();

void stage_release_slot(int slot) {
    for (const StageChunk& c : g_stage_held[slot]) g_stage_free.push_back(c);
    g_stage_held[slot].clear();
    g_stage_fill[slot] = 0;
    g_stage_seen[slot].clear();
}

bool new_stage_chunk(StageChunk& c) {
    if (g_stage_chunks >= kStageMaxChunks || g_given_back) return false;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kStageChunkBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &c.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, c.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &c.memory) != VK_SUCCESS) {
        vkDestroyBuffer(g.device, c.buffer, nullptr);
        c = StageChunk{};
        return false;
    }
    vkBindBufferMemory(g.device, c.buffer, c.memory, 0);
    ++g_stage_chunks;
    return true;
}

VkCommandBuffer upload_cmd_locked() {
    if (g_upload_slot == g.slot) return g_upload_cmd[g.slot];
    VkCommandBuffer& cmd = g_upload_cmd[g.slot];
    if (!cmd) {
        // A pool of its own: these are recorded on the command processor
        // while the draw recorder (recorder.cpp) records g.cmd_, and a pool's
        // command buffers are recorded one thread at a time.
        static VkCommandPool pool = [] {
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pci.queueFamilyIndex = g.family;
            VkCommandPool p = VK_NULL_HANDLE;
            if (vkCreateCommandPool(g.device, &pci, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
            return p;
        }();
        if (!pool) return VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(g.device, &cai, &cmd) != VK_SUCCESS) {
            cmd = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    gpu_checkpoint_cmd(cmd, kGpuMarkSubmission, g.flushes * 4 + 0);
    busy_upload_begin_locked(g.slots[g.slot], cmd);  // the busy meter counts the uploads too
    stage_release_slot(g.slot);  // the slot's last uploads have run: their staged copies are free
    // Earlier submissions' reads of the mirror finish before these writes.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    g_upload_slot = g.slot;
    return cmd;
}

// Written since its copy, through the CPU view or the GPU alias.
bool written(Page& pg) {
    bool w = write_watch_dirty(pg.va, pg.bytes, pg.watch) != 0;
    if (pg.alias_va) w |= write_watch_dirty(pg.alias_va, pg.bytes, pg.alias_watch) != 0;
    return w;
}

void go_dynamic(Mirror& m, std::uint32_t p, std::uint64_t va, const char* why) {
    Page& pg = m.pages[p];
    pg.dynamic = true;
    pg.valid = false;
    m.dyn[p >> 6] |= 1ull << (p & 63);
    if (g_dynamic.fetch_add(1, std::memory_order_relaxed) < 8) {
        host_log("buffer shadow: the page at 0x%llx %s; bound from host memory from now on", static_cast<unsigned long long>(va), why);
    }
}

// The per-recording bits start empty in each recording: whichever of a
// draw's check or a GPU write comes first in it clears them.
void begin_recording(Mirror& m, std::uint64_t stamp) {
    if (m.ok_stamp == stamp) return;
    std::fill(m.ok.begin(), m.ok.end(), 0);
    std::fill(m.gpu.begin(), m.gpu.end(), 0);
    m.ok_stamp = stamp;
}

}  // namespace

// Off with BBHOST_BUFFER_SHADOW=0 and, unless it is 1, on an integrated GPU -
// its device-local memory is the RAM the imports already are, so a mirror
// saves no bus - and with tight memory (gpu.cpp, memory_tight): a mirror is a
// gigabyte a chunk, and on a Steam Deck (1 GiB of VRAM, 8 GiB of GTT, which
// every submission's buffers have to fit together) the second one made the
// submissions fail.
//
// An integrated GPU with 8 GiB or more of video memory (a Strix Halo's
// carve-out) takes the mirrors too: only their page tracking says which
// vertex data the game rewrites, which is copied as its draws record
// (shadow_snapshot_locked) - read in place, it was the effects flickering
// whenever the GPU was behind. The meshes then come from the carve-out
// rather than the imports: on the Radeon 8060S in Kyo's room GPU-bound, 102
// fps with the mirrors against 105 without, back to back (2026-10-15) - no
// faster, about 1.7 GiB more of the carve-out in use.
bool integrated_with_room() {
    static int known = -1;
    if (known < 0 && g.mem_props.memoryHeapCount) {
        VkDeviceSize local = 0;
        for (std::uint32_t i = 0; i < g.mem_props.memoryHeapCount; ++i) {
            if (g.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                local = std::max(local, g.mem_props.memoryHeaps[i].size);
        }
        known = local >= (8ull << 30) ? 1 : 0;
    }
    return known == 1;
}
bool buffer_shadow_on() {
    return g_shadow_env >= 0 ? g_shadow_env == 1 : !g.tight && (!g.integrated || integrated_with_room());
}

void shadow_locate_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    if (!buffer_shadow_on() || !bytes || !loc.buffer) return;
    std::uint32_t chunk = 0;
    std::uint64_t off = 0, map_lo = 0, map_hi = 0;
    if (!phys_of(va, bytes, chunk, off, &map_lo, &map_hi)) return;
    Mirror* m = mirror_for(chunk);
    if (!m) return;
    const std::uint64_t stamp = g.record_serial + 1;
    const std::uint32_t p0 = static_cast<std::uint32_t>(off >> kPageShift);
    const std::uint32_t p1 = static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift);
    {
        const std::uint32_t span = p1 - p0 + 1;
        ++g_span_hist[span == 1 ? 0 : span <= 4 ? 1 : span <= 16 ? 2 : span <= 64 ? 3 : span <= 256 ? 4 : 5];
    }
    begin_recording(*m, stamp);
    if (all_ok(m->ok, p0, p1)) {
        bump(g_hits);
        loc.buffer = m->buffer;
        loc.offset = off;
        loc.avail = bytes;
        return;
    }
    // A page bound from host memory for good, or one a GPU write in this
    // recording left uncopied (a copy now would run before that write): the
    // walk below would stop at it, having looked at every page before it.
    if (any_set(m->dyn, p0, p1) || any_set(m->gpu, p0, p1)) {
        bump(g_imports);
        return;
    }
    // The live share refused a block less than 250 ms ago, this mirror's
    // block is full and the range has pages with no device page yet: the
    // range is read from host memory, as the refusal below would leave it,
    // before any page's write watch is armed. Each draw used to arm its pages
    // (an mprotect a run) and drop them at the refusal; on a 2.5 GB card that
    // was 21 million refusals over six places and up to three times the
    // frame time at the same GPU time.
    if (g_live && m->sparse && (m->blocks.empty() || m->block_pages_used == kBlockPages) && !all_ok(m->resident, p0, p1) &&
        now_ms() < g_room_retry_ms) {
        ++g_room_skips;
        bump(g_imports);
        return;
    }
    // The pages this call will copy, armed a run at a time: one protection of
    // the guest view and one of its GPU alias per run of consecutive pages,
    // instead of two per 64 KiB page - each is an mprotect, and the new
    // buffers of a fight's first frame armed hundreds of pages (~9% of the
    // command processor there). The walk stops where the loop below would;
    // a run that cannot be armed whole is left to the loop, a page at a time.
    thread_local std::vector<std::uint8_t> prearmed;  // per page of [p0, p1]: 1 view armed, 2 alias armed
    prearmed.assign(p1 - p0 + 1, 0);
    {
        const auto arm_run = [&](std::uint32_t a, std::uint32_t b) {  // pages [a, b)
            if (b - a < 2) return;  // one page: the loop's own arming is the same work
            const std::uint64_t first_va = va - (off - (static_cast<std::uint64_t>(a) << kPageShift));
            const std::uint64_t lo = std::max(first_va, map_lo);
            const std::uint64_t hi = std::min(first_va + (static_cast<std::uint64_t>(b - a) << kPageShift), map_hi);
            WriteWatch w, aw;
            if (!hle_kernel_write_watch(lo, hi - lo, w)) return;
            const std::uint64_t alias = hle_kernel_write_watch_alias(lo, hi - lo, aw);
            for (std::uint32_t p = a; p < b; ++p) {
                Page& pg = m->pages[p];
                pg.watch = w;
                prearmed[p - p0] = 1;
                if (alias) {
                    const std::uint64_t page_va = va - (off - (static_cast<std::uint64_t>(p) << kPageShift));
                    pg.alias_watch = aw;
                    pg.alias_va = alias + (std::max(page_va, map_lo) - lo);
                    prearmed[p - p0] |= 2;
                }
            }
        };
        const std::uint64_t t = now_ms();
        std::uint32_t run = p0, p = p0;
        for (; p <= p1; ++p) {
            Page& pg = m->pages[p];
            // The loop's checks in its order, with only its write check's
            // effects (once per recording, as the loop makes it).
            if (pg.dynamic || (!pg.valid && pg.gpu_written == stamp)) break;
            if (pg.valid && pg.checked != stamp) {
                pg.checked = stamp;
                ++g_page_checks;
                if (written(pg)) pg.valid = false;
            }
            if (pg.valid) {  // nothing to copy: the run ends before it
                arm_run(run, p);
                run = p + 1;
                continue;
            }
            const std::uint32_t uploads = t - pg.window_ms >= kUploadWindowMs ? 0 : pg.window_uploads;
            if (uploads + 1 > kMaxUploads) break;  // the loop sends it to host memory and stops there
        }
        arm_run(run, p);
    }
    // Pages to copy, as runs, from the import that holds them all.
    const Chunk* source = dmem_import(chunk, static_cast<std::uint64_t>(p0) << kPageShift, static_cast<std::uint64_t>(p1 - p0 + 1) << kPageShift);
    if (!source) {
        bump(g_imports);
        return;
    }
    VkBufferCopy regions[16];
    std::uint32_t n_regions = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    auto flush = [&] {
        if (n_regions) vkCmdCopyBuffer(cmd, source->buffer, m->buffer, n_regions, regions);
        n_regions = 0;
    };
    for (std::uint32_t p = p0; p <= p1; ++p) {
        Page& pg = m->pages[p];
        const std::uint64_t page_va = va - (off - (static_cast<std::uint64_t>(p) << kPageShift));
        if (pg.dynamic) {
            flush();
            bump(g_imports);
            return;
        }
        // Written by the GPU in this recording: a copy now would run before
        // that write. In a later recording the copy is ordered after it.
        if (!pg.valid && pg.gpu_written == stamp) {
            flush();
            bump(g_imports);
            return;
        }
        if (pg.valid) {
            if (pg.checked != stamp) {
                pg.checked = stamp;
                ++g_page_checks;
                if (written(pg)) pg.valid = false;
            }
            if (pg.valid) {
                m->ok[p >> 6] |= 1ull << (p & 63);
                continue;
            }
        }
        // Copy it. A page copied more than kMaxUploads times in the window
        // keeps being rewritten: host memory serves it better.
        const std::uint64_t t = now_ms();
        if (t - pg.window_ms >= kUploadWindowMs) {
            pg.window_ms = t;
            pg.window_uploads = 0;
        }
        if (++pg.window_uploads > kMaxUploads) {
            go_dynamic(*m, p, page_va, "keeps being rewritten");
            flush();
            bump(g_imports);
            return;
        }
        // Watched from before the copy reads it: a write after that marks it
        // again rather than going unseen. Only the part in this mapping.
        const std::uint64_t lo = std::max(page_va, map_lo), hi = std::min(page_va + kPageBytes, map_hi);
        if (!(prearmed[p - p0] & 1) && !hle_kernel_write_watch(lo, hi - lo, pg.watch)) {
            go_dynamic(*m, p, page_va, "cannot be write-watched");
            flush();
            bump(g_imports);
            return;
        }
        if (!(prearmed[p - p0] & 2)) pg.alias_va = hle_kernel_write_watch_alias(lo, hi - lo, pg.alias_watch);
        if (!cmd && !(cmd = upload_cmd_locked())) {
            pg.watch = WriteWatch{};
            pg.alias_watch = WriteWatch{};
            pg.alias_va = 0;
            return;
        }
        pg.va = lo;
        pg.bytes = hi - lo;
        pg.valid = true;
        pg.checked = stamp;
        if (g_verify) {
            const auto* src = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(lo));
            pg.hash = fnv1a(src, hi - lo);
            g_verify_copies[(std::uint64_t{chunk} << 32) | p].assign(src, src + (hi - lo));
        }
        m->ok[p >> 6] |= 1ull << (p & 63);
        const VkDeviceSize at = static_cast<VkDeviceSize>(p) << kPageShift;
        if (m->sparse && !(m->resident[p >> 6] & (1ull << (p & 63)))) {
            // Bind a device page for it; the bind runs on the queue before
            // this recording's copies (shadow_bind_sparse_locked).
            if (m->blocks.empty() || m->block_pages_used == kBlockPages) {
                VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                bai.allocationSize = static_cast<VkDeviceSize>(kBlockPages) * kPageBytes;
                bai.memoryTypeIndex = find_memory_type(m->mem_type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VkDeviceMemory mem = VK_NULL_HANDLE;
                const bool room = mirror_room(bai.allocationSize);
                if (!room && g_live) {
                    // No room now is not for good: this draw reads host
                    // memory, a later one tries again (after the mirrors
                    // start over at a load, say). Not a copy toward 'keeps
                    // being rewritten' either.
                    pg.valid = false;
                    m->ok[p >> 6] &= ~(1ull << (p & 63));
                    if (pg.window_uploads) --pg.window_uploads;
                    pg.watch = WriteWatch{};
                    pg.alias_watch = WriteWatch{};
                    pg.alias_va = 0;
                    ++g_room_refusals;
                    flush();
                    bump(g_imports);
                    return;
                }
                if (!room || vkAllocateMemory(g.device, &bai, nullptr, &mem) != VK_SUCCESS) {
                    go_dynamic(*m, p, page_va, !room ? "has no room in the mirrors' share of video memory"
                                                     : "has no device memory for its mirror page");
                    flush();
                    bump(g_imports);
                    return;
                }
                g_mirror_bytes += bai.allocationSize;
                m->blocks.push_back(mem);
                m->block_pages_used = 0;
            }
            VkSparseMemoryBind b{};
            b.resourceOffset = at;
            b.size = kPageBytes;
            b.memory = m->blocks.back();
            b.memoryOffset = static_cast<VkDeviceSize>(m->block_pages_used++) * kPageBytes;
            m->pending.push_back(b);
            m->resident[p >> 6] |= 1ull << (p & 63);
            ++m->resident_pages;
        }
        // The source is that import, which starts at the chunk's byte lo.
        const VkDeviceSize src = at - source->lo;
        if (n_regions && regions[n_regions - 1].srcOffset + regions[n_regions - 1].size == src) {
            regions[n_regions - 1].size += kPageBytes;
        } else {
            if (n_regions == 16) flush();
            regions[n_regions++] = {src, at, kPageBytes};
        }
        bump(g_uploads);
        bump(g_upload_bytes, kPageBytes);
    }
    flush();
    bump(g_hits);
    loc.buffer = m->buffer;
    loc.offset = off;
    loc.avail = bytes;
}

// For a diagnostic: what the mirror would do with this range right now, and -
// the question that matters - whether a page it is still serving has been
// written since the recording's one check of it. A valid page is checked once
// per recording; a guest write after that check is not seen again until the
// next one, so `valid + checked-this-recording + written` is a page whose
// mirror the GPU will read stale. Pure: `write_watch_dirty` only memoises a
// clean answer, and a written watch stays written until it is armed again.
bool shadow_stage_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    if (!buffer_shadow_on() || !bytes || !loc.buffer || loc.avail < bytes) return false;
    if (bytes > kStageMaxBytes) {
        bump(g_stage_refused);
        return false;
    }
    // A page the GPU writes earlier in this recording: the copy would run
    // ahead of that write. The mirror's page state is what tells; without a
    // mirror for the chunk there is none, and the range stays where it is.
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off)) return false;
    const Mirror& m = g_mirrors[chunk];
    if (m.pages.empty()) return false;
    const std::uint64_t stamp = g.record_serial + 1;
    for (std::uint64_t p = off >> kPageShift; p <= (off + bytes - 1) >> kPageShift; ++p) {
        if (m.pages[p].gpu_written == stamp) {
            bump(g_stage_refused);
            return false;
        }
    }
    const VkCommandBuffer cmd = upload_cmd_locked();
    if (!cmd) return false;
    const int slot = g.slot;
    if (g_stage_dedup) {
        const auto it = g_stage_seen[slot].find(StageKey{loc.buffer, loc.offset});
        if (it != g_stage_seen[slot].end() && it->second.bytes >= bytes) {
            loc.buffer = it->second.buffer;
            loc.offset = it->second.offset;
            loc.avail = bytes;
            bump(g_stage_shared);
            g_stage_shared_bytes.fetch_add(bytes, std::memory_order_relaxed);
            return true;
        }
    }
    const VkDeviceSize need = (bytes + 255) & ~VkDeviceSize{255};
    if (g_stage_held[slot].empty() || g_stage_fill[slot] + need > kStageChunkBytes) {
        StageChunk c;
        if (!g_stage_free.empty()) {
            c = g_stage_free.back();
            g_stage_free.pop_back();
        } else if (!new_stage_chunk(c)) {
            bump(g_stage_refused);
            return false;
        }
        g_stage_held[slot].push_back(c);
        g_stage_fill[slot] = 0;
    }
    const StageChunk& c = g_stage_held[slot].back();
    const VkBufferCopy region{loc.offset, g_stage_fill[slot], bytes};
    vkCmdCopyBuffer(cmd, loc.buffer, c.buffer, 1, &region);
    if (g_stage_dedup) {
        StagedRange& seen = g_stage_seen[slot][StageKey{loc.buffer, loc.offset}];
        if (bytes >= seen.bytes) seen = StagedRange{c.buffer, g_stage_fill[slot], bytes};
    }
    loc.buffer = c.buffer;
    loc.offset = g_stage_fill[slot];
    loc.avail = bytes;
    g_stage_fill[slot] += need;
    bump(g_staged);
    g_staged_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

// ---- copies, taken as the draw records, of vertex data the game rewrites ------
// A dynamic page holds a ring the game writes on the CPU every frame (effects'
// and particles' quads, YEBIS's full-screen triangles). Read in place - or
// staged above, which copies it as the GPU reaches the submission - a draw
// read it when the GPU got to it: up to 16 submissions on, a frame and a half
// GPU-bound, by when the game had written later draws' vertices over it. In
// Kyo's room GPU-bound (BBHOST_VB_RACE=1, 2026-10-15), YEBIS's triangles and
// the 4 MiB rings of two effect pipelines changed under queued draws - the
// effects flickering whenever the GPU was behind, on an RTX 2060 as on the
// 8060S, as the constants did before they were copied as recorded (render.cpp,
// BBHOST_CB_SNAPSHOT). The bytes the draw reads are copied here on the CPU as
// it records, into a ring the GPU reads (in video memory the CPU writes when
// there is some: an integrated GPU's, or a discrete one's BAR), and staged
// into device memory from there when the ring is in host memory. Memory a GPU
// write is still to land on is left alone (gpu_writes_pending_locked): its
// draws read it in the GPU's order. BBHOST_VERTEX_SNAPSHOT=0: as before.
namespace {
const bool g_snapshot_on = [] {
    const char* e = std::getenv("BBHOST_VERTEX_SNAPSHOT");
    return !(e && e[0] == '0');
}();
constexpr VkDeviceSize kSnapRing = 64ull << 20, kSnapSegment = 4ull << 20;
constexpr VkDeviceSize kSnapMaxBytes = 1ull << 20;
struct SnapRing {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint8_t* map = nullptr;
    bool device_local = false, failed = false;
    VkDeviceSize next = 0;
    std::uint64_t lap = 1;
    // The recording that last copied into each segment, and in which lap.
    std::uint64_t segment_serial[kSnapRing / kSnapSegment] = {};
    std::uint64_t segment_lap[kSnapRing / kSnapSegment] = {};
};
SnapRing g_snap;
std::atomic<std::uint64_t> g_snapshots{0}, g_snapshot_bytes{0}, g_snapshot_staged{0}, g_snapshot_pending{0};

bool snap_ring_make() {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kSnapRing;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &g_snap.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, g_snap.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // Video memory the CPU writes on an integrated GPU, and on a discrete one
    // where the rings go there (rings_in_vram); else host memory.
    std::uint32_t type = UINT32_MAX;
    if (g.integrated || rings_in_vram()) type = find_memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type != UINT32_MAX) {
        mai.memoryTypeIndex = type;
        if (vkAllocateMemory(g.device, &mai, nullptr, &g_snap.memory) == VK_SUCCESS) g_snap.device_local = true;
    }
    if (!g_snap.memory) {
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, visible);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &g_snap.memory) != VK_SUCCESS) {
            vkDestroyBuffer(g.device, g_snap.buffer, nullptr);
            g_snap.buffer = VK_NULL_HANDLE;
            return false;
        }
    }
    vkBindBufferMemory(g.device, g_snap.buffer, g_snap.memory, 0);
    void* p = nullptr;
    if (vkMapMemory(g.device, g_snap.memory, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return false;
    g_snap.map = static_cast<std::uint8_t*>(p);
    host_log("buffer shadow: vertex data the game rewrites is copied as its draws record, into a %llu MiB ring in %s "
             "(BBHOST_VERTEX_SNAPSHOT=0: read as the GPU reaches it)",
             static_cast<unsigned long long>(kSnapRing >> 20),
             g_snap.device_local ? "video memory the CPU writes" : "host memory, staged into video memory per draw");
    return true;
}
}  // namespace

bool shadow_snapshot_maybe_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_snapshot_on || !buffer_shadow_on() || !bytes) return false;
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off) || off + bytes > kChunkBytes) return true;  // not told this cheaply
    const Mirror& m = g_mirrors[chunk];
    return !m.pages.empty() && any_set(m.dyn, static_cast<std::uint32_t>(off >> kPageShift), static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift));
}

bool shadow_snapshot_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    if (!g_snapshot_on || !buffer_shadow_on() || !bytes || bytes > kSnapMaxBytes) return false;
    // Only a range on a page the game keeps rewriting: a static page the
    // mirrors had no room for is staged as before, not copied by the CPU at
    // every draw.
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off)) return false;
    const Mirror& m = g_mirrors[chunk];
    if (m.pages.empty() || !any_set(m.dyn, static_cast<std::uint32_t>(off >> kPageShift), static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift)))
        return false;
    if (gpu_writes_pending_locked(va, bytes)) {
        bump(g_snapshot_pending);
        return false;
    }
    if (g_snap.failed || (!g_snap.buffer && !snap_ring_make())) {
        g_snap.failed = true;
        return false;
    }
    if (g_snap.next + bytes + 256 > kSnapRing) {
        g_snap.next = 0;
        ++g_snap.lap;
    }
    const VkDeviceSize at = (g_snap.next + 255) & ~VkDeviceSize{255};
    // A lap round the ring: the draws that read a segment in the lap before
    // have run before it is written again (nearly always long since).
    for (VkDeviceSize s = at / kSnapSegment; s <= (at + bytes - 1) / kSnapSegment; ++s) {
        if (g_snap.segment_lap[s] != g_snap.lap) {
            if (g_snap.segment_lap[s]) wait_submission_locked(g_snap.segment_serial[s]);
            g_snap.segment_lap[s] = g_snap.lap;
        }
        g_snap.segment_serial[s] = g.flushes;
    }
    std::memcpy(g_snap.map + at, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), bytes);
    g_snap.next = at + bytes;
    bump(g_snapshots);
    bump(g_snapshot_bytes, bytes);  // not a locked add: it would wait for the copy to drain
    loc.buffer = g_snap.buffer;
    loc.offset = at;
    loc.avail = bytes;
    // A ring in host memory: the copy staged into device memory for the draw,
    // from the ring - what the draw read when it recorded.
    if (!g_snap.device_local && shadow_stage_locked(va, bytes, loc)) bump(g_snapshot_staged);
    return true;
}

void shadow_probe_locked(std::uint64_t va, std::uint64_t bytes, char* out, std::size_t n) {
    if (!out || !n) return;
    out[0] = 0;
    if (!buffer_shadow_on() || !bytes) {
        std::snprintf(out, n, "shadow off");
        return;
    }
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off, nullptr, nullptr)) {
        std::snprintf(out, n, "not in a mirrored chunk");
        return;
    }
    Mirror* m = mirror_for(chunk);
    if (!m) {
        std::snprintf(out, n, "no mirror");
        return;
    }
    const std::uint64_t stamp = g.record_serial + 1;
    const std::uint32_t p0 = static_cast<std::uint32_t>(off >> kPageShift);
    const std::uint32_t p1 = static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift);
    std::size_t at = 0;
    std::uint32_t stale = 0;
    for (std::uint32_t p = p0; p <= p1 && at + 48 < n; ++p) {
        Page& pg = m->pages[p];
        const bool w = pg.valid && written(pg);
        if (pg.valid && pg.checked == stamp && w) ++stale;
        at += static_cast<std::size_t>(std::snprintf(out + at, n - at, "%s%u:%s%s%s%s", at ? " " : "", p,
                                                     pg.dynamic ? "host" : pg.valid ? "mirror" : "uncopied",
                                                     pg.valid && pg.checked == stamp ? "+checked" : "",
                                                     w ? "+WRITTEN" : "",
                                                     pg.gpu_written == stamp ? "+gpuwrite" : ""));
    }
    if (stale && at + 32 < n) std::snprintf(out + at, n - at, " <= %u STALE", stale);
}

// GPU writes by 64 KiB page, with the buffer shadow off too: the recording
// the newest went into. render.cpp copies a constant buffer as its draw is
// recorded only when no GPU write to its memory is still to land
// (gpu_writes_pending_locked); landed pages leave the table as it grows.
// Asked at every constant buffer a draw binds (~3,400 a frame), so both are
// flat tables - open addressing, one probe as a rule: the std::map of lines
// and the node map of pages they replace cost the command processor 0.6 ms a
// frame, the frame rate wherever it rather than the GPU set the pace (Kyo,
// 2026-10-14: "it feels like we're heavily cpubound"). A small write - a
// label or a counter, ~280 a frame - goes by 4 KiB page: by 64 KiB, each
// label's page (beside the game's constants) read as GPU-written for good.
namespace {
constexpr unsigned kSmallShift = 12;
constexpr std::uint64_t kSmallWrite = 256;

// Page number -> the newest recording a GPU write to it went into. Rebuilt
// with only the writes still to land when it is half full.
class PageSerials {
public:
    void note(std::uint64_t page, std::uint64_t serial) {
        if ((used_ + 1) * 2 > slots_.size()) rebuild();
        Slot& s = slots_[find(page + 1)];
        if (!s.key) {
            s.key = page + 1;  // 0: an empty slot
            ++used_;
        }
        if (serial > s.serial) s.serial = serial;
    }
    bool pending(std::uint64_t page) const {
        if (!used_) return false;
        const Slot& s = slots_[find(page + 1)];
        return s.key && s.serial >= g.completed_submits;
    }
    // The latest recording a GPU write over the page was made in (0: none).
    std::uint64_t latest(std::uint64_t page) const {
        if (!used_) return 0;
        const Slot& s = slots_[find(page + 1)];
        return s.key ? s.serial : 0;
    }

private:
    struct Slot {
        std::uint64_t key = 0, serial = 0;
    };
    std::vector<Slot> slots_;
    std::size_t used_ = 0;
    unsigned bits_ = 10;
    std::size_t find(std::uint64_t key) const {
        const std::size_t mask = slots_.size() - 1;
        std::size_t i = static_cast<std::size_t>((key * 0x9e3779b97f4a7c15ull) >> (64 - bits_));
        while (slots_[i].key && slots_[i].key != key) i = (i + 1) & mask;
        return i;
    }
    void rebuild() {
        std::vector<Slot> old;
        old.swap(slots_);
        std::size_t keep = 0;
        for (const Slot& s : old) keep += s.key && s.serial >= g.completed_submits;
        while ((keep + 1) * 4 > (std::size_t{1} << bits_)) ++bits_;
        slots_.assign(std::size_t{1} << bits_, Slot{});
        used_ = 0;
        for (const Slot& s : old) {
            if (!s.key || s.serial < g.completed_submits) continue;
            slots_[find(s.key)] = s;
            ++used_;
        }
    }
};
PageSerials g_gpu_write_pages, g_gpu_write_small;
}  // namespace

void gpu_writes_note_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!bytes) return;
    const std::uint64_t serial = g.flushes;  // the recording under way
    if (bytes <= kSmallWrite) {
        for (std::uint64_t p = va >> kSmallShift; p <= (va + bytes - 1) >> kSmallShift; ++p) g_gpu_write_small.note(p, serial);
        return;
    }
    const std::uint64_t first = va >> kPageShift, last = (va + bytes - 1) >> kPageShift;
    if (last - first > (1u << 16)) return;  // 4 GiB: not a write anything makes
    for (std::uint64_t p = first; p <= last; ++p) g_gpu_write_pages.note(p, serial);
}

bool gpu_writes_pending_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!bytes) return false;
    for (std::uint64_t p = va >> kPageShift; p <= (va + bytes - 1) >> kPageShift; ++p) {
        if (g_gpu_write_pages.pending(p)) return true;
    }
    for (std::uint64_t p = va >> kSmallShift; p <= (va + bytes - 1) >> kSmallShift; ++p) {
        if (g_gpu_write_small.pending(p)) return true;
    }
    return false;
}

bool gpu_wrote_since_locked(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial) {
    if (!bytes) return false;
    for (std::uint64_t p = va >> kPageShift; p <= (va + bytes - 1) >> kPageShift; ++p) {
        if (g_gpu_write_pages.latest(p) >= serial) return true;
    }
    for (std::uint64_t p = va >> kSmallShift; p <= (va + bytes - 1) >> kSmallShift; ++p) {
        if (g_gpu_write_small.latest(p) >= serial) return true;
    }
    return false;
}

void shadow_written_locked(std::uint64_t va, std::uint64_t bytes, bool by_gpu) {
    if (by_gpu) gpu_writes_note_locked(va, bytes);
    if (!buffer_shadow_on() || !bytes) return;
    std::uint32_t chunk = 0;
    std::uint64_t off = 0, map_lo = 0, map_hi = 0;
    if (!phys_of(va, 1, chunk, off, &map_lo, &map_hi)) return;
    Mirror& m = g_mirrors[chunk];
    if (m.pages.empty()) return;
    const std::uint64_t stamp = g.record_serial + 1;
    begin_recording(m, stamp);
    const std::uint64_t end = std::min<std::uint64_t>(off + std::min(bytes, map_hi - va), kChunkBytes);
    std::uint64_t gpu_pages = 0;
    for (std::uint64_t p = off >> kPageShift; p <= (end - 1) >> kPageShift; ++p) {
        Page& pg = m.pages[p];
        m.ok[p >> 6] &= ~(1ull << (p & 63));
        if (pg.valid) {
            pg.valid = false;
            ++(by_gpu ? g_invalid_gpu : g_invalid_cp);
        }
        if (by_gpu) {
            pg.gpu_written = stamp;
            m.gpu[p >> 6] |= 1ull << (p & 63);
            ++gpu_pages;
        }
    }
    if (gpu_pages) g_gpu_written.fetch_add(gpu_pages, std::memory_order_relaxed);
}

namespace {
// Pages valid for a while (checked in an earlier recording, so their copy has
// run) whose memory no longer hashes as it did when copied.
void verify_batch_locked() {
    const std::uint64_t stamp = g.record_serial + 1;
    for (std::uint32_t chunk = 0; chunk < gcn::kDmemChunks; ++chunk) {
        Mirror& m = g_mirrors[chunk];
        if (m.pages.empty()) continue;
        for (int n = 0; n < 256; ++n) {
            const std::uint32_t p = static_cast<std::uint32_t>(g_verify_cursor++ % kPagesPerChunk);
            Page& pg = m.pages[p];
            if (!pg.valid || pg.checked == stamp || !pg.va) continue;
            const std::uint64_t page_va = pg.va & ~(kPageBytes - 1);
            if (!hle_kernel_va_mapped(pg.va, pg.bytes)) continue;
            ++g_verify_checked;
            if (fnv1a(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(pg.va)), pg.bytes) == pg.hash) continue;
            if (written(pg)) continue;  // a write the next check will see anyway
            if (g_verify_stale++ < 32) {
                host_log("buffer shadow: VERIFY: page 0x%llx changed with no write this cache saw (chunk %u)",
                         static_cast<unsigned long long>(page_va), chunk);
                hle_kernel_mappings_of(pg.va);  // logs each address the page is mapped at
                auto it = g_verify_copies.find((std::uint64_t{chunk} << 32) | p);
                if (it != g_verify_copies.end() && it->second.size() == pg.bytes) {
                    const auto* now = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(pg.va));
                    const auto* was = reinterpret_cast<const std::uint32_t*>(it->second.data());
                    std::uint32_t changed = 0, first = ~0u, last = 0;
                    std::string sample;
                    for (std::uint32_t k = 0; k < pg.bytes / 4; ++k) {
                        if (now[k] == was[k]) continue;
                        ++changed;
                        first = std::min(first, k * 4);
                        last = k * 4;
                        if (changed <= 6) {
                            char b[96];
                            float fw, fn;
                            std::memcpy(&fw, &was[k], 4);
                            std::memcpy(&fn, &now[k], 4);
                            std::snprintf(b, sizeof(b), " +0x%x: %08x->%08x (%g->%g)", k * 4, was[k], now[k], fw, fn);
                            sample += b;
                        }
                    }
                    host_log("buffer shadow: VERIFY:   %u dwords changed in [+0x%x, +0x%x]:%s", changed, first, last, sample.c_str());
                }
                char d[512];
                write_watch_describe(pg.va, pg.bytes, pg.watch, d, sizeof(d));
                host_log("buffer shadow: VERIFY:   CPU view watch: %s", d);
                if (pg.alias_va) {
                    write_watch_describe(pg.alias_va, pg.bytes, pg.alias_watch, d, sizeof(d));
                    host_log("buffer shadow: VERIFY:   alias 0x%llx watch: %s", static_cast<unsigned long long>(pg.alias_va), d);
                }
                if (std::FILE* f = std::fopen("/proc/self/maps", "r")) {
                    char line[256];
                    while (std::fgets(line, sizeof(line), f)) {
                        unsigned long long a = 0, b = 0;
                        if (std::sscanf(line, "%llx-%llx", &a, &b) != 2) continue;
                        const bool cpu = a < pg.va + pg.bytes && b > pg.va;
                        const bool alias = pg.alias_va && a < pg.alias_va + pg.bytes && b > pg.alias_va;
                        if (cpu || alias) host_log("buffer shadow: VERIFY:   maps %s", line);
                    }
                    std::fclose(f);
                }
            }
            go_dynamic(m, p, page_va, "changed without a tracked write");
        }
    }
}
}  // namespace

// The copies this recording made, ended; submit_locked runs them first.
VkCommandBuffer shadow_end_uploads_locked() {
    if (g_verify) verify_batch_locked();
    if (g_upload_slot != g.slot) return VK_NULL_HANDLE;
    const VkCommandBuffer cmd = g_upload_cmd[g.slot];
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(cmd);
    g_upload_slot = -1;
    return cmd;
}

VkSemaphore shadow_bind_sparse_locked() {
    VkSparseBufferMemoryBindInfo infos[gcn::kDmemChunks];
    std::uint32_t n = 0;
    for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) {
        Mirror& m = g_mirrors[c];
        if (!m.sparse || m.pending.empty()) continue;
        infos[n].buffer = m.buffer;
        infos[n].bindCount = static_cast<std::uint32_t>(m.pending.size());
        infos[n].pBinds = m.pending.data();
        ++n;
    }
    if (!n) return VK_NULL_HANDLE;
    VkSemaphore& sem = g_bind_sem[g.slot];
    if (!sem) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(g.device, &sci, nullptr, &sem) != VK_SUCCESS) sem = VK_NULL_HANDLE;
    }
    VkBindSparseInfo bsi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    bsi.bufferBindCount = n;
    bsi.pBufferBinds = infos;
    bsi.signalSemaphoreCount = sem ? 1 : 0;
    bsi.pSignalSemaphores = sem ? &sem : nullptr;
    VkResult r;
    {
        QueueGuard queue;
        r = vkQueueBindSparse(g.queue, 1, &bsi, VK_NULL_HANDLE);
    }
    static int logs = 0;
    if (r != VK_SUCCESS && logs++ < 5) host_log("buffer shadow: vkQueueBindSparse failed (%d)", r);
    for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) g_mirrors[c].pending.clear();
    return r == VK_SUCCESS ? sem : VK_NULL_HANDLE;
}

void shadow_budget_locked() {
    const char* e = std::getenv("BBHOST_SHADOW_VRAM_MB");
    // The live share: with the video memory reserve on and no share given,
    // the mirrors grow while video memory keeps the reserve and a cushion
    // free, instead of taking only what lies above 6656 MiB of the budget at
    // the start - which left any card with less than about 6.7 GB free no
    // mirrors at all, every vertex and index read across the bus (18-27 ms
    // of GPU a frame at 1080p where the mirrors give 6). BBHOST_SHADOW_LIVE=0
    // keeps the fixed share.
    const char* live = std::getenv("BBHOST_SHADOW_LIVE");
    g_live = !(e && *e) && vram_reserve_bytes() && !(live && live[0] == '0');
    if (const char* c = std::getenv("BBHOST_SHADOW_CUSHION_MB"); c && *c) g_cushion = std::strtoull(c, nullptr, 10) << 20;
    g_mirror_cap = e && *e ? static_cast<std::uint64_t>(std::strtoull(e, nullptr, 10)) << 20
                   : g_live ? ~0ull
                   : g.local_budget > kImageRoom ? g.local_budget - kImageRoom : 0;
    g_initial_cap = g_mirror_cap;
    if (!buffer_shadow_on()) return;
    if (g_live) {
        host_log("buffer shadow: its mirrors may hold what video memory has free above the reserve and a %llu MiB cushion (%s)",
                 static_cast<unsigned long long>(g_cushion >> 20), g.has_sparse ? "64 MiB blocks" : "whole 1 GiB mirrors: no sparse binding");
        return;
    }
    if (!g.has_sparse && g_mirror_cap < kChunkBytes) {
        host_log("buffer shadow: no mirrors - one is a whole %llu MiB chunk here (no sparse binding), and the mirrors' share of video "
                 "memory is %llu MiB (%s); vertex and index data are read from host memory",
                 static_cast<unsigned long long>(kChunkBytes >> 20), static_cast<unsigned long long>(g_mirror_cap >> 20),
                 e && *e ? "BBHOST_SHADOW_VRAM_MB" : "what is free above the 6656 MiB kept for images and targets");
    } else {
        host_log("buffer shadow: its mirrors may hold %llu MiB of video memory (%s)", static_cast<unsigned long long>(g_mirror_cap >> 20),
                 e && *e ? "BBHOST_SHADOW_VRAM_MB" : "what is free above the 6656 MiB kept for images and targets");
    }
}

void shadow_give_back_locked(Gpu::Slot& slot) {
    std::uint32_t mirrors = 0;
    for (Mirror& m : g_mirrors) {
        if (!m.buffer) continue;
        slot.garbage.push_back(DevBuffer{m.memory, m.buffer});
        for (VkDeviceMemory b : m.blocks) slot.garbage.push_back(DevBuffer{b, VK_NULL_HANDLE});
        m = Mirror{};
        m.failed = true;
        ++mirrors;
    }
    const std::uint64_t held = g_mirror_bytes;
    g_mirror_bytes = 0;
    g_mirror_cap = 0;
    g_given_back_ms = now_ms();
    if (!g_given_back) {
        g_given_back = true;
        if (g_gives++ < 8)
            host_log("buffer shadow: video memory ran short - %u mirror(s), %llu MiB, given back to the images; vertex and index data are "
                     "read from host memory %s", mirrors, static_cast<unsigned long long>(held >> 20),
                     vram_return_on() ? "until video memory has room for them again" : "from now on");
    }
}

// Mirrors given back for a shortage come back once video memory has had room
// for them for 15 s: the share they started with, or what is free above the
// reserve and a margin if less. A card that ran short once read every vertex
// and index across the bus for the rest of the session - one 64 MiB scratch
// chunk at a load took a 5 GB card from 60 fps to 44.8 for good.
void shadow_return_locked() {
    if (!g_given_back || !vram_return_on() || !buffer_shadow_on()) return;
    static std::uint64_t roomy_since = 0;
    const std::uint64_t now = now_ms();
    const std::int64_t room = vram_headroom() - static_cast<std::int64_t>(vram_reserve_bytes()) - (256ll << 20);
    if (room < (512ll << 20) || now - g_given_back_ms < 15000) {
        roomy_since = 0;
        return;
    }
    if (!roomy_since) roomy_since = now;
    if (now - roomy_since < 15000) return;
    roomy_since = 0;
    std::uint32_t reset = 0;
    for (Mirror& m : g_mirrors) {
        if (m.failed && !m.buffer) {
            m = Mirror{};
            ++reset;
        }
    }
    g_mirror_cap = g_live ? g_initial_cap : std::min<std::uint64_t>(g_initial_cap, static_cast<std::uint64_t>(room));
    g_given_back = false;
    if (g_returns++ < 8)
        host_log("buffer shadow: video memory has room again - the mirrors come back (%u chunk(s)%s)", reset,
                 g_live ? ", the live share" : (", a share of " + std::to_string(g_mirror_cap >> 20) + " MiB").c_str());
}

namespace {
// Every mirror goes into `slot`, freed once that submission - the last that
// reads them - has run; the pages are copied again as draws read them.
void flush_mirrors_locked(Gpu::Slot& slot, const char* why) {
    std::uint32_t n = 0;
    const std::uint64_t held = g_mirror_bytes;
    for (Mirror& m : g_mirrors) {
        if (!m.buffer) {
            if (!m.failed) m = Mirror{};
            continue;
        }
        slot.garbage.push_back(DevBuffer{m.memory, m.buffer});
        for (VkDeviceMemory b : m.blocks) slot.garbage.push_back(DevBuffer{b, VK_NULL_HANDLE});
        m = Mirror{};
        ++n;
    }
    g_mirror_bytes = 0;
    g_room_retry_ms = 0;
    g_first_refusal_ms = 0;
    g_last_flush_ms = now_ms();
    if (g_flushes++ < 12)
        host_log("buffer shadow: %s - %u mirror(s), %llu MiB, start over", why, n, static_cast<unsigned long long>(held >> 20));
}
}  // namespace

// A page goes dynamic for good (go_dynamic), but a load hands the game's
// memory to the next area: a page that held one area's vertex ring can hold
// the next one's meshes, which then stayed in host memory - every draw of them
// reading across the bus, or (vertex input) copying its whole record into
// device memory for itself (shadow_stage_locked). An RTX 4060 laptop's log
// over ten lamp travels: 143 -> 405 pages dynamic, draws bound from host
// memory 15-27% -> 44%, the copy a staged draw made ~10 KiB -> ~100 KiB
// (37 MiB a frame), and the frame rate standing still 77 -> 47 with the GPU
// less busy (the command processor waiting on the uploads' labels). So at
// each load every dynamic page is uncopied and unwatched again: a ring is
// found again within its first few rewrites (kMaxUploads in kUploadWindowMs),
// a mesh stays in the mirror. BBHOST_SHADOW_SECOND_CHANCE=0: never.
namespace {
void shadow_second_chance_locked() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_SHADOW_SECOND_CHANCE");
        return !(e && e[0] == '0');
    }();
    if (!on) return;
    std::uint64_t pages = 0;
    for (Mirror& m : g_mirrors) {
        if (m.pages.empty()) continue;
        for (std::size_t w = 0; w < m.dyn.size(); ++w) {
            std::uint64_t bits = m.dyn[w];
            m.dyn[w] = 0;
            while (bits) {
                const std::size_t p = w * 64 + static_cast<std::size_t>(__builtin_ctzll(bits));
                bits &= bits - 1;
                if (p >= m.pages.size()) continue;
                Page& pg = m.pages[p];
                pg.dynamic = false;
                pg.valid = false;
                pg.window_ms = 0;
                pg.window_uploads = 0;
                pg.watch = WriteWatch{};
                pg.alias_watch = WriteWatch{};
                pg.alias_va = 0;
                ++pages;
            }
        }
    }
    if (!pages) return;
    g_second_chances += pages;
    static int logs = 0;
    if (logs++ < 8)
        host_log("buffer shadow: a load - %llu dynamic page(s) (%llu MiB) copied again when next read, a ring found again by its rewrites "
                 "(BBHOST_SHADOW_SECOND_CHANCE=0: dynamic for good)",
                 static_cast<unsigned long long>(pages), static_cast<unsigned long long>((pages * kPageBytes) >> 20));
}
}  // namespace

// The mirrors start over at a load that follows a shortage of room: when a
// loading screen comes up and, since the last one, the live share refused
// pages, the reserve kept textures out of video memory or the mirrors were
// given back - or the share is all but shut right then - every mirror goes
// and the coming area's pages are copied as its draws read them; and once in
// play between two loads, when pages have been refused for 5 s. A share that
// has stopped growing otherwise keeps the areas already left and reads the
// new one's geometry across the bus. BBHOST_SHADOW_FLUSH_AT_LOAD=0: never;
// on with the live share.

void shadow_flush_at_load_locked(Gpu::Slot& slot) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_SHADOW_FLUSH_AT_LOAD");
        return !(e && e[0] == '0');
    }();
    static bool was_loading = false;
    static bool mid_flushed = false;  // one start-over in play between two loads
    const bool loading = loading_screen_now();
    const bool starts = loading && !was_loading;
    was_loading = loading;
    if (starts && buffer_shadow_on()) shadow_second_chance_locked();
    if (!on || !g_live || !buffer_shadow_on() || g_given_back || !g_mirror_bytes) {
        if (starts) mid_flushed = false;
        return;
    }
    const std::uint64_t block = static_cast<std::uint64_t>(kBlockPages) * kPageBytes;
    if (starts) {
        mid_flushed = false;
        const std::uint64_t held_back = vram_reserve_refusals();
        // Or the share is all but shut as the load starts - less than a
        // cushion's worth of blocks would fit above the reserve and the
        // cushion: the coming area would find the mirrors full of the areas
        // already left and read its own geometry across the bus until the
        // load after it (a 4 GB card entered the Nightmare of Mensis that way,
        // at 15.5 ms of GPU a frame against 6.3).
        const bool shut = !vram_room_for(block, 2 * g_cushion);
        if (!shut && g_room_refusals == g_flush_refusals && held_back == g_flush_held_back && g_gives == g_flush_gives) return;
        g_flush_refusals = g_room_refusals;
        g_flush_held_back = held_back;
        g_flush_gives = g_gives;
        flush_mirrors_locked(slot, "a load after a shortage of room");
        return;
    }
    // In play: pages refused for 5 s on end (the area moved on without a
    // load - most of the game's are seamless - and the mirrors hold the one
    // left), 10 s or more after the last start-over: once until the next load.
    const std::uint64_t now = now_ms();
    if (!loading && !mid_flushed && g_first_refusal_ms && now - g_first_refusal_ms >= 5000 && g_room_retry_ms + 750 >= now &&
        now - g_last_flush_ms >= 10000) {
        mid_flushed = true;
        g_flush_refusals = g_room_refusals;
        flush_mirrors_locked(slot, "pages refused for want of room in play");
    }
}

std::string shadow_report() {
    std::uint64_t resident = 0;
    for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) resident += g_mirrors[c].resident_pages;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "buffer shadow: %llu binds from device memory, %llu page copies (%.1f MiB), %llu from host memory; %llu pages "
                  "dynamic, %llu GPU writes over them; %llu page checks; invalidations %llu by the CP and %llu by the GPU",
                  static_cast<unsigned long long>(g_hits.load()), static_cast<unsigned long long>(g_uploads.load()),
                  static_cast<double>(g_upload_bytes.load()) / (1 << 20), static_cast<unsigned long long>(g_imports.load()),
                  static_cast<unsigned long long>(g_dynamic.load()), static_cast<unsigned long long>(g_gpu_written.load()),
                  static_cast<unsigned long long>(g_page_checks), static_cast<unsigned long long>(g_invalid_cp),
                  static_cast<unsigned long long>(g_invalid_gpu));
    if (g_verify) {
        char v[96];
        std::snprintf(v, sizeof(v), "; verified %llu, stale %llu", static_cast<unsigned long long>(g_verify_checked),
                      static_cast<unsigned long long>(g_verify_stale));
        std::strncat(buf, v, sizeof(buf) - std::strlen(buf) - 1);
    }
    char h[160];
    std::snprintf(h, sizeof(h), "; spans 1:%llu 2-4:%llu 5-16:%llu 17-64:%llu 65-256:%llu more:%llu",
                  static_cast<unsigned long long>(g_span_hist[0]), static_cast<unsigned long long>(g_span_hist[1]),
                  static_cast<unsigned long long>(g_span_hist[2]), static_cast<unsigned long long>(g_span_hist[3]),
                  static_cast<unsigned long long>(g_span_hist[4]), static_cast<unsigned long long>(g_span_hist[5]));
    char rr[192];
    char share[32];
    if (g_mirror_cap == ~0ull) std::snprintf(share, sizeof(share), "live");
    else std::snprintf(share, sizeof(share), "%llu MiB", static_cast<unsigned long long>(g_mirror_cap >> 20));
    std::snprintf(rr, sizeof(rr), "; mirror pages resident %llu (%llu MiB); mirrors hold %llu MiB of video memory, share %s%s",
                  static_cast<unsigned long long>(resident), static_cast<unsigned long long>((resident * kPageBytes) >> 20),
                  static_cast<unsigned long long>(g_mirror_bytes >> 20), share, g_given_back ? " (given back: video memory ran short)" : "");
    std::string out = std::string(buf) + h + rr;
    if (g_second_chances) {
        char sc[96];
        std::snprintf(sc, sizeof(sc), "; dynamic pages given another try at a load %llu", static_cast<unsigned long long>(g_second_chances));
        out += sc;
    }
    if (g_room_refusals || g_flushes || g_room_skips) {
        char lr[160];
        std::snprintf(lr, sizeof(lr), "; copies refused for want of room %llu (%llu more draws not tried), started over at a load %llu",
                      static_cast<unsigned long long>(g_room_refusals), static_cast<unsigned long long>(g_room_skips),
                      static_cast<unsigned long long>(g_flushes));
        out += lr;
    }
    if (g_gives || g_returns) {
        char gr[96];
        std::snprintf(gr, sizeof(gr), "; given back %llu time(s), back %llu", static_cast<unsigned long long>(g_gives),
                      static_cast<unsigned long long>(g_returns));
        out += gr;
    }
    char st[288];
    std::snprintf(st, sizeof(st),
                  "; vertex ranges copied for their draw %llu (%.1f MiB), read from another draw's copy %llu (%.1f MiB), left in host "
                  "memory %llu, staging chunks %zu (%zu MiB)",
                  static_cast<unsigned long long>(g_staged.load()), static_cast<double>(g_staged_bytes.load()) / (1 << 20),
                  static_cast<unsigned long long>(g_stage_shared.load()), static_cast<double>(g_stage_shared_bytes.load()) / (1 << 20),
                  static_cast<unsigned long long>(g_stage_refused.load()), g_stage_chunks,
                  static_cast<std::size_t>((g_stage_chunks * kStageChunkBytes) >> 20));
    char sn[192];
    std::snprintf(sn, sizeof(sn),
                  "; rewritten vertex ranges copied as recorded %llu (%.1f MiB, %llu staged from host memory), left for a GPU write "
                  "still to land %llu",
                  static_cast<unsigned long long>(g_snapshots.load()), static_cast<double>(g_snapshot_bytes.load()) / (1 << 20),
                  static_cast<unsigned long long>(g_snapshot_staged.load()), static_cast<unsigned long long>(g_snapshot_pending.load()));
    return out + st + sn;
}

}  // namespace gpu

std::string host_gpu_shadow_report() {
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    return gpu::shadow_report();
}

void host_gpu_shadow_cp_write(std::uint64_t va, std::size_t bytes) {
    using namespace gpu;
    std::lock_guard<GpuMutex> lock(g.mu);
    shadow_written_locked(va, bytes, false);
}
