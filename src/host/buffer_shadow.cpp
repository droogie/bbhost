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
#include "engine/gx_resources.h"
#include "host/apu_memory.h"
#include "host/import_fallback.h"
#include "host/copy_runs.h"
#include "host/gpu_internal.h"
#include "host/gxvk.h"
#include "host/list_resolve.h"
#include "host/shadow_place.h"
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
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include "core/portable.h"

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
    return e && (e[0] == '1' || e[0] == '2');
}();
// BBHOST_SHADOW_VERIFY=2: told pages are write-watched as well (their faults
// change nothing but the census's - BBHOST_RESOURCE_CENSUS=1 - which classes
// them as told and lists their writers): who writes the pages GX's calls
// tell, the writes the calls tell and those they do not.
const bool g_verify_watch_told = [] {
    const char* e = std::getenv("BBHOST_SHADOW_VERIFY");
    return e && e[0] == '2';
}();
std::uint64_t g_verify_cursor = 0, g_verify_checked = 0, g_verify_stale = 0;
// Verify mode: each page's contents when copied, so a stale page's report
// says which bytes changed and to what.
std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> g_verify_copies;

const int g_shadow_env = [] {
    const char* e = std::getenv("BBHOST_BUFFER_SHADOW");
    return !e || !*e ? -1 : e[0] == '0' ? 0 : 1;
}();
// BBHOST_SHADOW_LITE=0: the mirrors on an integrated GPU with room, as before
// (buffer_shadow_on); otherwise the lite path stands in for them there
// (shadow_lite_on).
const bool g_lite_wanted = [] {
    const char* e = std::getenv("BBHOST_SHADOW_LITE");
    return !(e && e[0] == '0');
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
    std::uint64_t gx_seq = 0;         // BBHOST_RESOURCE_CENSUS: the GX registry's write sequence when copied
    // Told (BBHOST_SHADOW_GX): GX buffers cover the page whole and GX never
    // handed any of it out, so the CPU writes it only in a creation, which
    // tells the mirror (host_gpu_shadow_gx_write) - the page is not
    // write-watched. told_seq is g_told_seq when it was copied; gx_written
    // the sequence of the newest such write over it, gx_submit the
    // submissions made when that was asked.
    bool told = false;
    std::uint64_t told_seq = 0, gx_written = 0, gx_submit = 0;
    // Told and copied while a renamed copy over it was still outstanding
    // (handed out, not yet written back by GX): its CPU writes may not have
    // ended, so the copy holds for its recording only.
    bool provisional = false;
    std::uint16_t provisional_run = 0;  // copies in a row made so
    // BBHOST_SHADOW_VERIFY: the bytes of it draws bound since its copy, as
    // offsets in the page - a change outside them is no draw's.
    std::uint32_t read_lo = ~0u, read_hi = 0;
    bool valid = false;               // the mirror holds it (or will, once this submission's copies run)
    bool dynamic = false;             // bound from the import from now on (rewritten, unwatchable)
    // In place (BBHOST_SHADOW_IN_PLACE): read from the import with no copy -
    // GX tells every write to it. placed_seq is g_told_seq when it was placed,
    // placed_checked the recording its placement was last checked in.
    bool placed = false;
    std::uint64_t placed_seq = 0, placed_checked = 0;
    // place_range_locked's question to GX's registry for the page (whether
    // GX buffers cover [cover_va, +cover_bytes)), answered once a recording:
    // kept while the recording, the page's newest told write (gx_written) and
    // the told epoch (g_told_epoch) are those it was asked under. A buffer
    // made or let go over the page is a told write there; a range of pages
    // one of which GX does not cover asked the registry, under its lock, for
    // every draw that read it (5.5% of bb-cp0 in Kyo's Central Yharnam save).
    std::uint64_t cover_stamp = 0, cover_gx = 0, cover_epoch = 0, cover_va = 0;
    std::uint32_t cover_bytes = 0;
    bool cover = false;
};

struct Mirror {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    bool failed = false;
    bool making = false;  // a whole mirror the memory keeper is making (shadow_keeper_work)
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
    // A bit per page: placed in place and checked in the recording `ok_stamp`
    // (BBHOST_SHADOW_IN_PLACE) - a draw on such pages binds the import.
    std::vector<std::uint64_t> placed_ok;
    // A bit per page: placed in place with no told write over it since
    // (Page::placed, shadow_place::rewritten false) - what each recording's
    // first check of a placed page finds, kept across recordings: set where a
    // page is placed, cleared wherever placement ends or a told write lands.
    // A range all of whose pages have it is placed_ok for the recording with
    // no walk over its pages' records (place_range_locked): the command
    // processor submits after every job, and re-checking every placed page
    // of every mesh pool once a recording was ~4% of bb-cp0 in Kyo's Central
    // Yharnam save.
    std::vector<std::uint64_t> placed_valid;
    // Sparse: a 64 KiB device page is bound the first time the page is
    // copied, from 64 MiB blocks; the binds queue up until the submission.
    bool sparse = false;
    std::uint32_t mem_type_bits = 0;
    std::vector<VkDeviceMemory> blocks;
    std::uint32_t block_pages_used = 0;  // in blocks.back()
    std::vector<std::uint64_t> resident;  // a bit per page
    std::vector<VkSparseMemoryBind> pending;
    std::uint64_t resident_pages = 0;
    // Windowed (no sparse binding; BBHOST_SHADOW_WINDOWS): the chunk's mirror
    // as buffers of a 64 MiB window each, made with their memory when a page
    // in the window is first copied, instead of one buffer for the whole
    // chunk made at its first draw.
    bool windowed = false;
    struct Window {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        bool making = false;  // the memory keeper is making it (shadow_keeper_work)
    };
    std::vector<Window> windows;
};
constexpr std::uint32_t kBlockPages = 1024;  // 64 MiB blocks
// A window's pages (BBHOST_SHADOW_WINDOWS): 64 MiB on a discrete GPU, 4 on an
// integrated one, 1 on an integrated GPU under Windows, or
// BBHOST_SHADOW_WINDOW_MB (a power of two from 1 to 512; host/apu_memory.h).
// Larger windows hold more pages no draw reads, smaller ones send more ranges
// across a window's edge to host memory - on an integrated GPU the same RAM:
// on the Radeon 680M at a lamp 16 MiB windows held 1,248 MiB where 64 MiB ones
// held 1,728, at the same frame rate, with 6.8 M ranges read in place instead
// of 1.6 M. With the told pages read in place (BBHOST_SHADOW_IN_PLACE) the
// copies left are scattered: 228 MiB of them kept 63 windows of 16 MiB
// (1,008 MiB), and in 4 MiB windows 464 MiB, at the same frame rate and with
// no range read in place changing under its draws. Read when the first
// mirror is made, after the device is known.
std::uint32_t window_pages() {
    static const std::uint32_t pages = [] {
#if defined(_WIN32)
        constexpr bool kWindowsHost = true;
#else
        constexpr bool kWindowsHost = false;
#endif
        const std::uint64_t mb = apu_memory::shadow_window_mib(std::getenv("BBHOST_SHADOW_WINDOW_MB"), g.integrated ? 1 : 0, kWindowsHost);
        return static_cast<std::uint32_t>((mb << 20) / kPageBytes);
    }();
    return pages;
}
std::uint64_t window_bytes() { return static_cast<std::uint64_t>(window_pages()) * kPageBytes; }
VkSemaphore g_bind_sem[kSlots] = {};

// Without sparse binding a mirror was a whole chunk: 1 GiB of video memory for
// the first vertex range a draw read in it. Over an area's places the game's
// meshes lie in four to six of the guest's chunks, so the mirrors took 4-6 GiB
// of an integrated GPU's carve-out (a Radeon 8060S, where the fixed share is
// its budget less 6.5 GiB), and on an 8 GB AMD card, where a whole mirror is
// made only while 1.8 GiB is free above the reserve, two fitted and the other
// chunks' vertices crossed the bus. Made 64 MiB window by window as their
// pages are first copied, they hold the windows the draws read; a range
// across two windows is read from host memory, as one in a chunk with no
// mirror was. BBHOST_SHADOW_WINDOWS=0: whole mirrors, as before.
const bool g_windows_on = [] {
    const char* e = std::getenv("BBHOST_SHADOW_WINDOWS");
    return !(e && e[0] == '0');
}();
std::uint64_t g_window_crossings = 0;  // ranges across two windows, read from host memory
std::uint64_t g_windows_made = 0;

// In place (BBHOST_SHADOW_IN_PLACE; on for an integrated GPU, =1 anywhere, =0
// nowhere): a range whose every page GX tells the writes of - GX buffers
// cover it whole and GX never handed it out to be rewritten - binds the
// import where it lies, with no copy and no mirror memory, its page tracking
// kept (shadow_place.h). On an integrated GPU the mirror was the same RAM
// again: on the Radeon 680M at a lamp 1,264 MiB of 16 MiB windows held 531
// MiB of page copies, ~84% of them told pages. With window imports
// (BBHOST_IMPORT_WINDOWS=1) the import is the range's window, which the
// binding's locate() makes when the import thread has not (GX's meshes are
// imported as GX makes them, host_gpu_import_prefetch);
// BBHOST_SHADOW_IN_PLACE_WINDOWS=0 keeps in place to whole imports, the
// mirrors serving every mesh with window imports, as before.
const int g_in_place_env = [] {
    const char* e = std::getenv("BBHOST_SHADOW_IN_PLACE");
    return !e || !*e ? -1 : e[0] == '0' ? 0 : 1;
}();
const bool g_in_place_windows = [] {
    const char* e = std::getenv("BBHOST_SHADOW_IN_PLACE_WINDOWS");
    return !(e && e[0] == '0');
}();
// (Decided at the first use, a draw's or the import thread's, by when the
// device - and whether it is integrated - is known.)
bool in_place_on() {
    static const bool on = shadow_place::in_place_wanted(imports_whole(), g_in_place_env, g_in_place_windows, g.integrated);
    return on;
}
const bool g_placed_valid_on = [] {
    const char* e = std::getenv("BBHOST_SHADOW_PLACED_VALID");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_placed_binds{0};    // ranges bound in place
std::uint64_t g_placed = 0, g_placed_again = 0;  // pages placed, and placed again after a told write
std::uint64_t g_placed_refused = 0;              // ranges that had a page GX does not tell (to the mirrors)

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
// next area's textures (BBHOST_SHADOW_CUSHION_MB; a 24th of the budget at the
// start, from 128 to 256 MiB, as the reserve scales); a page refused for want
// of room is read from host memory by that draw and tried again later.
bool g_live = false;
std::uint64_t g_cushion = 256ull << 20;
std::uint64_t g_room_refusals = 0;     // copies refused for want of room, counted per draw
// Made off the command processor (shadow_keeper_work, on the memory keeper's
// thread): a mirror's memory - a whole one's 1 GiB, a window's (64 MiB on a
// discrete GPU), a sparse mirror's 64 MiB blocks - was vkCreateBuffer and
// vkAllocateMemory on the command processor at the draw that first needed
// it: 29-35 ms for a whole mirror on the Radeon 8060S, four of them as the
// world first drew and again after each start-over at a load (4508e7e3,
// before the windows). Now that draw - and its range's until the memory is in
// place - reads host memory, as at a refusal for want of room, and asks the
// keeper, which makes it and hands it over under the lock if it is still
// wanted (freed if the mirrors started over or were given back meanwhile); a
// sparse mirror's next block comes from a spare the keeper keeps ready while
// the share has room for it and the next one. BBHOST_SHADOW_ASYNC=0 (or
// BBHOST_MEMORY_KEEPER=0): made on the command processor, as before.
bool g_async = false;  // shadow_keeper_started, unless BBHOST_SHADOW_ASYNC=0
std::uint64_t g_mirrors_async = 0, g_windows_async = 0;  // made off the command processor and handed over
double g_mirror_async_ms = 0.0, g_window_async_ms = 0.0;  // the time they took the keeper
std::uint64_t g_async_waits = 0;  // ranges read from host memory while their window or mirror was being made
std::uint64_t g_blocks_made = 0, g_blocks_spare = 0;  // sparse blocks made on the command processor, taken from the spare
double g_block_ms_total = 0.0;                        // the time the command processor's took
VkDeviceMemory g_spare_block = VK_NULL_HANDLE;
std::uint32_t g_spare_type = UINT32_MAX;  // its memory type: the sparse mirrors' (g_block_type)
std::uint32_t g_block_type = UINT32_MAX;  // the memory type the last sparse block was made of
std::uint64_t g_room_retry_ms = 0;     // no room until then: refused without asking (now_ms)
std::uint64_t g_room_skips = 0;        // draws read from host memory while refused, before arming anything
std::uint64_t g_first_refusal_ms = 0;  // the first refusal since the mirrors last started over (0: none)
std::uint64_t g_last_flush_ms = 0;     // when they last started over
std::uint64_t g_flushes = 0;           // the mirrors started over at a load
std::uint64_t g_second_chances = 0;    // dynamic pages given another try at a load (shadow_second_chance_locked)
std::uint64_t g_flush_refusals = 0, g_flush_held_back = 0, g_flush_gives = 0;  // the counts at the last flush
VkCommandBuffer g_upload_cmd[kSlots] = {};
int g_upload_slot = -1;  // the slot whose upload buffer is recording, or -1
thread_local std::size_t g_last_map = 0;  // g.maps index of this thread's last hit (a run part's too)

std::atomic<std::uint64_t> g_hits{0}, g_uploads{0}, g_upload_bytes{0}, g_dynamic{0}, g_imports{0}, g_gpu_written{0};
std::atomic<std::uint64_t> g_staged_pages{0};  // window imports: page copies taken through upload staging

// ---- Window imports: the staged page copies on threads of their own ----
// A mirror page copied through upload staging is a 64 KiB memcpy, and an
// area's arrival takes thousands of them in its first recordings (8,552 at
// Mensis, 534 MiB): done inline that was ~100 ms of the command processor in
// the first world frames, the arrival stalls window imports added (whole
// imports have the GPU copy from the import instead). Now the command
// processor records the staging -> mirror copy as before and hands the
// memcpy to these threads, which run it while it records on;
// shadow_end_uploads_locked, which ends the recording's upload commands
// before their submission, takes whatever no thread has yet and waits for
// the rest. BBHOST_SHADOW_COPY_THREADS=N (default 3); 0 copies inline.
class StagedCopies {
public:
    static StagedCopies& get() {
        static StagedCopies pool;
        return pool;
    }
    bool on() const { return threads_ > 0; }
    void push(void* dst, const void* src, std::size_t bytes) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back({dst, src, bytes});
            ++pending_;
        }
        cv_.notify_one();
    }
    // Every copy handed over so far is done (the command processor, before a
    // submission reads the staging).
    void finish() {
        std::uint64_t ran = 0;
        for (;;) {
            Job j;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (queue_.empty()) break;
                j = queue_.front();
                queue_.pop_front();
            }
            std::memcpy(j.dst, j.src, j.bytes);
            ++ran;
            done_one();
        }
        if (ran) inline_runs_.fetch_add(ran, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lk(mu_);
        if (pending_) waits_.fetch_add(1, std::memory_order_relaxed);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
    }
    std::uint64_t inline_runs() const { return inline_runs_.load(std::memory_order_relaxed); }
    std::uint64_t waits() const { return waits_.load(std::memory_order_relaxed); }
    unsigned threads() const { return threads_; }

private:
    struct Job {
        void* dst = nullptr;
        const void* src = nullptr;
        std::size_t bytes = 0;
    };
    StagedCopies() {
        const char* e = std::getenv("BBHOST_SHADOW_COPY_THREADS");
        threads_ = e && *e ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 3u;
        for (unsigned i = 0; i < threads_; ++i) {
            std::thread([this, i] {
                char name[16];
                std::snprintf(name, sizeof(name), "bb-pagecopy%u", i);
                host_thread_set_name(name);
                for (;;) {
                    Job j;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [this] { return !queue_.empty(); });
                        j = queue_.front();
                        queue_.pop_front();
                    }
                    std::memcpy(j.dst, j.src, j.bytes);
                    done_one();
                }
            }).detach();
        }
    }
    void done_one() {
        std::lock_guard<std::mutex> lk(mu_);
        if (--pending_ == 0) done_cv_.notify_all();
    }
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> queue_;
    std::size_t pending_ = 0;
    unsigned threads_ = 0;
    std::atomic<std::uint64_t> inline_runs_{0}, waits_{0};
};
std::atomic<std::uint64_t> g_staged_pending{0};  // ... and pages left in host memory for a GPU write still to land
std::uint64_t g_page_checks = 0, g_invalid_cp = 0, g_invalid_gpu = 0;
std::uint64_t g_span_hist[6] = {};  // calls by pages spanned: 1, 2-4, 5-16, 17-64, 65-256, more

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Guest VA -> (chunk, offset) through the snapshot of direct-memory mappings,
// with the mapping's bounds; false outside one mapping and one chunk.
// Set by shadow_import_fallback_locked: the mirror's pages copied through
// upload staging even with whole imports, since the range's import failed.
thread_local bool t_force_staged = false;

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
    // One chunk's mirror holds its pages and no others: a range past the
    // chunk's end is no mirror's (its pages would index past the mirror's).
    if (chunk >= gcn::kDmemChunks || off + bytes > kChunkBytes) return false;
    // Window imports: the mirrors' copies come from upload staging (below), so
    // the range needs no import, only to lie in a GPU-visible span - and the
    // same with whole imports for a range whose import failed (t_force_staged).
    if (!imports_whole() || t_force_staged) return off >= g.dmem[chunk].span_lo && off + bytes <= g.dmem[chunk].span_hi;
    return dmem_import(chunk, off, bytes) != nullptr;
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

// A windowed mirror's window w, made now if it is not: false when the
// mirrors' share has no room for it or video memory none.
bool window_ready(Mirror& m, std::uint32_t w) {
    Mirror::Window& win = m.windows[w];
    if (win.buffer) return true;
    if (!mirror_room(window_bytes())) return false;
    if (g_async) {
        // The keeper makes it (shadow_keeper_work); until it is in place the
        // window's ranges read host memory.
        if (!win.making) {
            win.making = true;
            keeper_wake();
        }
        return false;
    }
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = window_bytes();
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &win.buffer) != VK_SUCCESS) {
        win.buffer = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, win.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    MemOwner own_mem("mirror windows");
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &win.memory) != VK_SUCCESS) {
        vkDestroyBuffer(g.device, win.buffer, nullptr);
        win = Mirror::Window{};
        return false;
    }
    vkBindBufferMemory(g.device, win.buffer, win.memory, 0);
    g_mirror_bytes += req.size;
    ++g_windows_made;
    return true;
}

// A whole mirror's page records, as it starts.
void mirror_pages(Mirror& m) {
    m.pages.resize(kPagesPerChunk);
    m.ok.assign(kPagesPerChunk / 64, 0);
    m.dyn.assign(kPagesPerChunk / 64, 0);
    m.gpu.assign(kPagesPerChunk / 64, 0);
}

Mirror* mirror_for(std::uint32_t chunk) {
    Mirror& m = g_mirrors[chunk];
    if (m.buffer || m.windowed || m.failed) return m.failed ? nullptr : &m;
    if (m.making) {  // the keeper is making it: its ranges read host memory meanwhile
        ++g_async_waits;
        return nullptr;
    }
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
    if (g_windows_on) {
        m.windowed = true;
        m.windows.assign(kChunkBytes / window_bytes(), Mirror::Window{});
        m.pages.resize(kPagesPerChunk);
        m.ok.assign(kPagesPerChunk / 64, 0);
        m.dyn.assign(kPagesPerChunk / 64, 0);
        m.gpu.assign(kPagesPerChunk / 64, 0);
        host_log("buffer shadow: mirror for chunk %u made %llu MiB window by window as its pages are copied (no sparse binding; "
                 "BBHOST_SHADOW_WINDOWS=0: whole)",
                 chunk, static_cast<unsigned long long>(window_bytes() >> 20));
        return &m;
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
    if (g_async) {
        // The keeper makes it (shadow_keeper_work); this draw, and the chunk's
        // until it is in place, read host memory.
        m.making = true;
        keeper_wake();
        return nullptr;
    }
    if (vkCreateBuffer(g.device, &bci, nullptr, &m.buffer) != VK_SUCCESS) {
        m.buffer = VK_NULL_HANDLE;
    } else {
        vkGetBufferMemoryRequirements(g.device, m.buffer, &req);
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        MemOwner own_mem("mirrors");
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
    mirror_pages(m);
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
// (buffer, offset) and ordered, so a range inside one copied - of the same
// start, or a later one (a mesh whose vertex elements are records of their
// own, their bases a few bytes apart over the same bytes: four to eight
// copies of one range a draw at Astral Clocktower) - is read from it.
// BBHOST_STAGE_DEDUP=0: a copy per draw, as before.
struct StagedRange {
    VkBuffer buffer = VK_NULL_HANDLE;  // the stage chunk holding it
    VkDeviceSize offset = 0, bytes = 0;
};
struct StageKey {
    VkBuffer src;
    VkDeviceSize offset;
    bool operator==(const StageKey& o) const { return src == o.src && offset == o.offset; }
    bool operator<(const StageKey& o) const {
        return src != o.src ? reinterpret_cast<std::uintptr_t>(src) < reinterpret_cast<std::uintptr_t>(o.src) : offset < o.offset;
    }
};
std::map<StageKey, StagedRange> g_stage_seen[kSlots];
std::atomic<std::uint64_t> g_stage_inside{0};  // of the shared, ranges read from a copy that starts before them
std::atomic<std::uint64_t> g_stage_shared{0}, g_stage_shared_bytes{0};
const bool g_stage_dedup = [] {
    const char* e = std::getenv("BBHOST_STAGE_DEDUP");
    return !(e && e[0] == '0');
}();
// BBHOST_STAGE_CENSUS=1: the ranges staged for their draw since the last
// report, by bytes, with why their pages are in host memory - [0] a page
// bound from host memory for good (dynamic), [1] a page a GPU write left
// unmirrored, [2] a page not valid otherwise (refused, never copied) - and
// the GX resource they lie in.
const bool g_stage_census = [] {
    const char* e = std::getenv("BBHOST_STAGE_CENSUS");
    return e && e[0] == '1';
}();
struct StageStat {
    std::uint64_t bytes = 0, copies = 0, record = 0;
    int why = -1;
};
std::unordered_map<std::uint64_t, StageStat> g_stage_stats;  // by the range's start, under g.mu
std::uint64_t g_stage_why_bytes[4] = {};

void stage_release_slot(int slot) {
    for (const StageChunk& c : g_stage_held[slot]) g_stage_free.push_back(c);
    g_stage_held[slot].clear();
    g_stage_fill[slot] = 0;
    g_stage_seen[slot].clear();
}

// A chunk is made in a local and handed over only once its buffer, its memory
// and their binding have all succeeded. A failed bind used to leave an unbound
// buffer in the pool, and a failed create could leave the driver's handle in `c`.
bool new_stage_chunk(StageChunk& c) {
    if (g_stage_chunks >= kStageMaxChunks || g_given_back) return false;
    StageChunk made;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kStageChunkBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &made.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, made.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    MemOwner own_mem("vertex stage chunks");
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &made.memory) != VK_SUCCESS) {
        vkDestroyBuffer(g.device, made.buffer, nullptr);
        return false;
    }
    if (vkBindBufferMemory(g.device, made.buffer, made.memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(g.device, made.buffer, nullptr);
        vkFreeMemory(g.device, made.memory, nullptr);
        return false;
    }
    c = made;
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

// BBHOST_RESOURCE_CENSUS=1: of the pages the write watch found written,
// those a GX write (a Map, a creation with data) of a resource over them
// explains since their copy, those on a GX resource it does not, and those
// on none - whether GX's own calls could tell the mirrors what changed.
const bool g_map_census = [] {
    const char* e = std::getenv("BBHOST_RESOURCE_CENSUS");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_written_by_gx{0}, g_written_not_gx{0}, g_written_no_resource{0};

// BBHOST_SHADOW_GX=0: every page write-watched, as before. Otherwise the
// mirrors go by what GX's own calls say of the memory (the watch's faults
// land on the game's threads, a fault per 4 KiB page each time a mirrored
// ring or pool is written - 97.6% of them, the census found, on the memory of
// GX's renamed copies):
//   - GX's dynamic buffers are written through renamed copies from the
//     context's upload pool: a Map hands one out (types 3 and 4) and the CPU
//     writes it until the context's map resolves, which copies it back to
//     the buffer on the GPU (0x2aadf50 / 0x2ad1090 through 0x2ab2810 and
//     0x2ab2670), or until a later discard Map of the buffer in the same
//     context frees it. A page with renamed copies on it is told
//     (host_gpu_shadow_gx_handed, _settled), not watched: a hand-out
//     invalidates it, and while a copy over it is outstanding - handed out,
//     neither written back nor freed - a copy of the page holds for its
//     recording only. Once the copies are done, the page's copy holds until
//     the next hand-out (BBHOST_SHADOW_HANDED=0: watched as before). What
//     else the pool holds - constant buffers' copies (their Maps and every
//     dynamic UpdateSubresource), the draw tables the commits copy in - is
//     written without a word, and no draw reads it as vertex data;
//   - a page GX buffers cover whole changes only when a creation writes it
//     (host_gpu_shadow_gx_write): told, not watched;
//   - the rest - memory no GX buffer covers whole: the engine's own rings,
//     YEBIS's, Scaleform's - stays watched.
const bool g_shadow_gx = [] {
    const char* e = std::getenv("BBHOST_SHADOW_GX");
    return !(e && e[0] == '0');
}();
const bool g_shadow_handed = [] {
    const char* e = std::getenv("BBHOST_SHADOW_HANDED");
    return g_shadow_gx && !(e && e[0] == '0');
}();
struct GxWrite {
    std::uint64_t va, bytes, submit;
    int kind;  // kGxWrite, or 1 + a hand-out's kind (host_gpu_shadow_gx_handed)
};
// A write; a GX resource made over the range (a write, and no longer handed
// out); a renamed copy GX wrote back or freed (no longer outstanding).
constexpr int kGxWrite = 0, kGxCreate = 8, kGxSettle = 10;
// What GX handed the CPU to write (host_gpu_shadow_gx_handed): a bit per 4 KiB
// of direct memory by physical address - [kHandMap] the renamed copies Maps
// hand out and, with the census, [kHandPool] the upload pool's blocks,
// [kHandRing] ring allocations and [kHandYebis] the memory YEBIS's discard
// Maps rename its resources into.
enum : unsigned { kHandMap = 0, kHandPool = 1, kHandRing = 2, kHandYebis = 3, kHandKinds = 4 };
constexpr std::uint64_t kHand4k = (std::uint64_t{gcn::kDmemChunks} << gcn::kDmemChunkShift) >> 12;
std::vector<std::uint64_t> g_hand_bits[kHandKinds];  // under g.mu, made at the first hand-out
std::uint64_t g_hand_pages[kHandKinds] = {};         // 4 KiB pages marked
// The lite path's (shadow_lite_on): the flip, plus one, at which a Map last
// handed out each 64 KiB of direct memory (kHandMap), by physical page; 0 for
// never, or for memory a GX resource was made over since. Under g.mu.
std::vector<std::uint32_t> g_handed_flip;
void note_handed_flip_locked(std::uint64_t phys, std::uint64_t bytes, bool handed) {
    if (g_handed_flip.empty()) {
        if (!handed) return;
        g_handed_flip.assign(static_cast<std::size_t>((std::uint64_t{gcn::kDmemChunks} << gcn::kDmemChunkShift) >> kPageShift), 0);
    }
    const std::uint32_t flip = handed ? static_cast<std::uint32_t>(hle_video_flip_count()) + 1 : 0;
    for (std::uint64_t p = phys >> kPageShift; p <= (phys + bytes - 1) >> kPageShift && p < g_handed_flip.size(); ++p) g_handed_flip[p] = flip;
}
std::uint64_t g_map_in_pool = 0, g_map_in_ring = 0;  // Map hand-outs whose first page the pool / ring had handed out
std::uint64_t g_handed_dynamic = 0;                  // told pages that stayed unsettled too long, to host memory
// The renamed copies handed out and not yet written back, by physical start
// (under g.mu): the pages over them are unsettled.
struct Outstanding {
    std::uint64_t end = 0, since_ms = 0;
};
std::map<std::uint64_t, Outstanding> g_outstanding;
std::uint64_t g_outstanding_most = 0, g_settled = 0, g_settle_unmatched = 0, g_aged_out = 0, g_provisional_copies = 0, g_handed_invalid = 0;
constexpr std::uint64_t kOutstandingAgeMs = 10000;  // a copy left out longer was written back unseen (a resolve not hooked)
constexpr std::uint16_t kMaxProvisionalRun = 64;     // copies in a row a page may take unsettled before host memory serves it
// A page under GX's renamed copies is copied again at every hand-out over its
// copy - a no-overwrite Map's notice spans the whole copy, though the CPU
// appends to part of it - without the faults and arming a watched page's
// copy costs; on the snapshot path every draw of it would cost the command
// processor a copy as recorded. So it goes to host memory only past
// kMaxUploadsHanded copies in the window (3 a second), not 4.
constexpr std::uint32_t kMaxUploadsHanded = 24;
std::uint64_t g_made_over_handed = 0;                // GX resources made over memory GX had handed out
// Whether renamed copies were ever on the 64 KiB page p of chunk `chunk` (the
// 16 bits of one word). The census's kinds do not count.
bool page_handed(std::uint32_t chunk, std::uint32_t p) {
    const std::uint64_t i = ((std::uint64_t{chunk} << gcn::kDmemChunkShift) | (std::uint64_t{p} << kPageShift)) >> 12;
    if (i >= kHand4k || g_hand_bits[kHandMap].empty()) return false;
    return ((g_hand_bits[kHandMap][i >> 6] >> (i & 63)) & 0xffff) != 0;
}
// Whether a renamed copy over the page is outstanding.
bool page_unsettled(std::uint32_t chunk, std::uint32_t p) {
    if (g_outstanding.empty()) return false;
    const std::uint64_t lo = (std::uint64_t{chunk} << gcn::kDmemChunkShift) | (std::uint64_t{p} << kPageShift), hi = lo + kPageBytes;
    auto it = g_outstanding.lower_bound(hi);
    if (it == g_outstanding.begin()) return false;
    --it;
    return it->second.end > lo;
}
std::atomic<std::uint64_t> g_hand_dropped{0};  // hand-outs the full queue had no room for
bool handed(unsigned k, std::uint64_t phys) {
    const std::uint64_t i = phys >> 12;
    return i < kHand4k && !g_hand_bits[k].empty() && (g_hand_bits[k][i >> 6] >> (i & 63) & 1);
}
std::mutex g_gx_writes_mu;
std::vector<GxWrite> g_gx_writes;  // under g_gx_writes_mu
std::atomic<bool> g_gx_writes_pending{false}, g_gx_writes_overflow{false};
constexpr std::size_t kGxWritesMax = 1u << 16;
std::uint64_t g_told_seq = 0;  // under g.mu: one a drained write
std::uint64_t g_told_epoch = 0;  // under g.mu: one an invalidate_all_told (Page::cover's answers stand within one)
std::atomic<std::uint64_t> g_gx_records{0};
std::uint64_t g_told_copies = 0, g_watched_partial = 0, g_watched_none = 0, g_told_invalid = 0, g_told_held = 0, g_told_overflows = 0,
              g_told_unmapped = 0;
std::uint64_t g_verify_stale_told = 0, g_verify_stale_told_read = 0;

// The newest GX write over a told page is one its copy predates, and the
// submission walked now was made after that write was asked for (the CPU has
// written by the time a submission that draws it is made; one made before
// the Map draws what was there, and keeps the copy it has).
bool told_written(Page& pg, bool count) {
    if (pg.gx_written <= pg.told_seq) return false;
    const std::uint64_t job = hle_gnm_cp_job_seq();
    const std::uint64_t now = job ? job : hle_gnm_submitted_total();
    if (now > pg.gx_submit) {
        if (count) ++g_told_invalid;
        return true;
    }
    if (count) ++g_told_held;
    return false;
}

// Written since its copy, through the CPU view or the GPU alias.
bool written(Page& pg) {
    bool w = write_watch_dirty(pg.va, pg.bytes, pg.watch) != 0;
    if (pg.alias_va) w |= write_watch_dirty(pg.alias_va, pg.bytes, pg.alias_watch) != 0;
    if (w && g_map_census && pg.bytes) {
        bool any = false;
        const std::uint64_t seq = gx_write_seq_overlapping(pg.va, pg.bytes, &any);
        (!any ? g_written_no_resource : seq > pg.gx_seq ? g_written_by_gx : g_written_not_gx).fetch_add(1, std::memory_order_relaxed);
    }
    return w;
}

bool page_written(Page& pg) {
    if (!pg.told) return written(pg);
    if (pg.provisional) return true;  // its copy held for the recording it was made in
    return told_written(pg, true);
}

void go_dynamic(Mirror& m, std::uint32_t p, std::uint64_t va, const char* why) {
    Page& pg = m.pages[p];
    pg.dynamic = true;
    pg.valid = false;
    pg.placed = false;
    if (!m.placed_ok.empty()) m.placed_ok[p >> 6] &= ~(1ull << (p & 63));
    if (!m.placed_valid.empty()) m.placed_valid[p >> 6] &= ~(1ull << (p & 63));
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
    std::fill(m.placed_ok.begin(), m.placed_ok.end(), 0);
    m.ok_stamp = stamp;
}

// Direct memory's GPU alias (hle/kernel.cpp map_gpu_alias): kGpuDmemBase +
// the physical address, which GX may hand out from a Map.
constexpr std::uint64_t kGpuDmemBase = 0xFF00000000ull;

// The GX writes asked for since the last drain (host_gpu_shadow_gx_write),
// onto the pages of the mirrors under them: each page takes the write's
// sequence and submission count, and leaves this recording's checked set.
void invalidate_all_told(std::uint64_t seq, std::uint64_t submit) {
    ++g_told_epoch;
    for (Mirror& m : g_mirrors) {
        if (m.pages.empty()) continue;
        for (Page& pg : m.pages) {
            if (!pg.told && !pg.placed) continue;
            pg.gx_written = seq;
            pg.gx_submit = submit;
        }
        std::fill(m.ok.begin(), m.ok.end(), 0);
        std::fill(m.placed_ok.begin(), m.placed_ok.end(), 0);
        std::fill(m.placed_valid.begin(), m.placed_valid.end(), 0);
    }
}

// Guest VA -> physical address of direct memory, through the GPU alias or the
// snapshot of mappings, with how far the mapping goes on from there.
bool phys_at(std::uint64_t va, std::uint64_t& phys, std::uint64_t& span) {
    if (va >= kGpuDmemBase && va < kGpuDmemBase + (1ull << 32)) {
        phys = va - kGpuDmemBase;
        span = kGpuDmemBase + (1ull << 32) - va;
        return true;
    }
    for (const GuestMapInfo& x : g.maps) {
        if (va >= x.va && va < x.va + x.len) {
            if (!x.dmem || x.phys < 0) return false;
            phys = static_cast<std::uint64_t>(x.phys) + (va - x.va);
            span = x.va + x.len - va;
            return true;
        }
    }
    return false;
}

// A hand-out: its 4 KiB pages marked. A renamed copy a Map hands out also
// invalidates the mirror pages over it - told pages for the submissions
// after the hand-out, watched ones at their next check - and stays
// outstanding until GX writes it back or frees it (settle_locked).
// The 4 KiB bits [i0, i1] of `bits` set (`set`) or cleared, a word at a time;
// returns how many changed. A hand-out spans its whole buffer - megabytes for
// a particle ring, thousands of hand-outs a second on the command processor.
std::uint64_t mark_bits(std::vector<std::uint64_t>& bits, std::uint64_t i0, std::uint64_t i1, bool set) {
    std::uint64_t changed = 0;
    i1 = std::min(i1, kHand4k - 1);
    for (std::uint64_t w = i0 >> 6; i0 <= i1 && w <= i1 >> 6; ++w) {
        std::uint64_t mask = ~0ull;
        if (w == i0 >> 6) mask &= ~0ull << (i0 & 63);
        if (w == i1 >> 6) mask &= ~0ull >> (63 - (i1 & 63));
        std::uint64_t& word = bits[w];
        const std::uint64_t flip = (set ? ~word : word) & mask;
        changed += static_cast<std::uint64_t>(__builtin_popcountll(flip));
        word = set ? word | mask : word & ~mask;
    }
    return changed;
}

void mark_handed_locked(const GxWrite& w) {
    const unsigned k = static_cast<unsigned>(w.kind - 1);
    if (k >= kHandKinds) return;
    if (g_hand_bits[k].empty()) g_hand_bits[k].assign(kHand4k / 64, 0);
    if (k == kHandMap && g_map_census) {
        std::uint64_t phys = 0, span = 0;
        if (phys_at(w.va, phys, span)) {
            g_map_in_pool += handed(kHandPool, phys);
            g_map_in_ring += handed(kHandRing, phys);
        }
    }
    // (The lite path keeps no mirror pages and no outstanding copies.)
    const bool act = k == kHandMap && g_shadow_handed && !shadow_lite_on();
    const std::uint64_t seq = act ? ++g_told_seq : 0;
    std::uint64_t first_phys = ~0ull, first_span = 0;
    std::uint64_t va = w.va;
    const std::uint64_t end = w.va + w.bytes;
    while (va < end) {
        std::uint64_t phys = 0, span = 0;
        if (!phys_at(va, phys, span)) {
            va = (va | 0xfff) + 1;
            continue;
        }
        const std::uint64_t seg = std::min(end - va, span);
        if (first_phys == ~0ull) {
            first_phys = phys;
            first_span = seg;
        }
        g_hand_pages[k] += mark_bits(g_hand_bits[k], phys >> 12, (phys + seg - 1) >> 12, true);
        if (k == kHandMap) note_handed_flip_locked(phys, seg, true);
        if (k == kHandMap) {
            // Renamed copies are rewritten: no page under them is read in
            // place from now on (shadow_place::eligible).
            for (std::uint64_t at = phys & ~(kPageBytes - 1); at < phys + seg; at += kPageBytes) {
                Mirror& m = g_mirrors[at >> gcn::kDmemChunkShift];
                if (m.pages.empty()) continue;
                const std::uint32_t p = static_cast<std::uint32_t>((at & (kChunkBytes - 1)) >> kPageShift);
                m.pages[p].placed = false;
                if (!m.placed_ok.empty()) m.placed_ok[p >> 6] &= ~(1ull << (p & 63));
                if (!m.placed_valid.empty()) m.placed_valid[p >> 6] &= ~(1ull << (p & 63));
            }
        }
        if (act) {
            for (std::uint64_t at = phys & ~(kPageBytes - 1); at < phys + seg; at += kPageBytes) {
                Mirror& m = g_mirrors[at >> gcn::kDmemChunkShift];
                if (m.pages.empty()) continue;
                const std::uint32_t p = static_cast<std::uint32_t>((at & (kChunkBytes - 1)) >> kPageShift);
                Page& pg = m.pages[p];
                if (pg.valid) ++g_handed_invalid;
                pg.gx_written = seq;
                pg.gx_submit = w.submit;
                m.ok[p >> 6] &= ~(1ull << (p & 63));
            }
        }
        va += seg;
    }
    if (!act || first_phys == ~0ull) return;
    Outstanding& o = g_outstanding[first_phys];
    o.end = std::max(o.end, first_phys + first_span);
    o.since_ms = now_ms();
    g_outstanding_most = std::max<std::uint64_t>(g_outstanding_most, g_outstanding.size());
}

// GX wrote renamed copies in [w.va, +w.bytes) back to their buffers, or freed
// them: their CPU writes are over, and the pages under them settle. Copies left outstanding
// past kOutstandingAgeMs were written back unseen (by a resolve these hooks
// miss) and settle too, counted.
void settle_locked(const GxWrite& w) {
    std::uint64_t phys = 0, span = 0;
    bool matched = false;
    if (!g_outstanding.empty() && phys_at(w.va, phys, span)) {
        const std::uint64_t lo = phys, hi = phys + std::min<std::uint64_t>(w.bytes, span);
        auto it = g_outstanding.lower_bound(lo);
        if (it != g_outstanding.begin() && std::prev(it)->second.end > lo) --it;
        while (it != g_outstanding.end() && it->first < hi) {
            it = g_outstanding.erase(it);
            matched = true;
        }
    }
    ++(matched ? g_settled : g_settle_unmatched);
}

// Once a second: renamed copies outstanding past kOutstandingAgeMs settle.
void age_outstanding_locked() {
    static std::uint64_t swept_ms = 0;
    static std::uint32_t calls = 0;
    if (g_outstanding.empty() || (++calls & 4095) != 0) return;  // a clock read a draw would be the CP's
    const std::uint64_t now = now_ms();
    if (now - swept_ms < 1000) return;
    swept_ms = now;
    for (auto it = g_outstanding.begin(); it != g_outstanding.end();) {
        if (now - it->second.since_ms > kOutstandingAgeMs) {
            it = g_outstanding.erase(it);
            ++g_aged_out;
        } else {
            ++it;
        }
    }
}

// A GX resource made over [w.va, +w.bytes): the memory is the resource's, no
// longer what GX hands out to be rewritten (the allocators may share it).
void clear_handed_locked(const GxWrite& w) {
    bool any = false;
    std::uint64_t va = w.va;
    const std::uint64_t end = w.va + w.bytes;
    while (va < end) {
        std::uint64_t phys = 0, span = 0;
        if (!phys_at(va, phys, span)) {
            va = (va | 0xfff) + 1;
            continue;
        }
        const std::uint64_t seg = std::min(end - va, span);
        for (unsigned k = 0; k < kHandKinds; ++k) {
            if (g_hand_bits[k].empty()) continue;
            const std::uint64_t n = mark_bits(g_hand_bits[k], phys >> 12, (phys + seg - 1) >> 12, false);
            g_hand_pages[k] -= n;
            any |= n != 0;
        }
        note_handed_flip_locked(phys, seg, false);
        va += seg;
    }
    if (any && g_made_over_handed++ < 4)
        host_log("buffer shadow: a GX resource made over memory GX had handed out to be rewritten: [0x%llx, +0x%llx)",
                 static_cast<unsigned long long>(w.va), static_cast<unsigned long long>(w.bytes));
}

void drain_gx_writes_locked() {
    age_outstanding_locked();
    if (!g_gx_writes_pending.load(std::memory_order_acquire)) return;
    std::vector<GxWrite> writes;
    {
        std::lock_guard<std::mutex> lk(g_gx_writes_mu);
        writes.swap(g_gx_writes);
        g_gx_writes_pending.store(false, std::memory_order_relaxed);
    }
    if (g_gx_writes_overflow.exchange(false, std::memory_order_acq_rel)) {
        // More writes than the queue keeps came between two drains: every
        // told page is copied again before it is read.
        ++g_told_overflows;
        invalidate_all_told(++g_told_seq, hle_gnm_submitted_total());
    }
    for (const GxWrite& w : writes) {
        if (w.kind == kGxCreate) {
            clear_handed_locked(w);
            if (!g_shadow_gx) continue;
        } else if (w.kind == kGxSettle) {
            settle_locked(w);
            continue;
        } else if (w.kind != kGxWrite) {
            mark_handed_locked(w);
            continue;
        }
        const std::uint64_t seq = ++g_told_seq;
        std::uint64_t va = w.va;
        const std::uint64_t end = w.va + w.bytes;
        while (va < end) {
            // The physical span of [va, end) in one mapping and one chunk.
            std::uint64_t phys = 0, seg = 0;
            if (va >= kGpuDmemBase && va < kGpuDmemBase + (1ull << 32)) {
                phys = va - kGpuDmemBase;
                seg = end - va;
            } else {
                const GuestMapInfo* mi = nullptr;
                for (const GuestMapInfo& x : g.maps) {
                    if (va >= x.va && va < x.va + x.len) {
                        mi = &x;
                        break;
                    }
                }
                if (!mi) {
                    // No longer mapped (an unmap's range) or never in the
                    // snapshot: the physical pages it named are unknown, so
                    // every told page is copied again before it is read.
                    ++g_told_unmapped;
                    invalidate_all_told(seq, w.submit);
                    break;
                }
                if (!mi->dmem || mi->phys < 0) {
                    // Not direct memory: nothing a mirror holds; on to the next page.
                    va = (va | (kPageBytes - 1)) + 1;
                    continue;
                }
                phys = static_cast<std::uint64_t>(mi->phys) + (va - mi->va);
                seg = std::min(end, mi->va + mi->len) - va;
            }
            const std::uint64_t chunk = phys >> gcn::kDmemChunkShift;
            const std::uint64_t off = phys & (kChunkBytes - 1);
            seg = std::min<std::uint64_t>(seg, kChunkBytes - off);
            if (chunk < gcn::kDmemChunks && !g_mirrors[chunk].pages.empty()) {
                Mirror& m = g_mirrors[chunk];
                for (std::uint64_t p = off >> kPageShift; p <= (off + seg - 1) >> kPageShift; ++p) {
                    Page& pg = m.pages[p];
                    pg.gx_written = seq;
                    pg.gx_submit = w.submit;
                    m.ok[p >> 6] &= ~(1ull << (p & 63));
                    if (!m.placed_ok.empty()) m.placed_ok[p >> 6] &= ~(1ull << (p & 63));
                    if (!m.placed_valid.empty()) m.placed_valid[p >> 6] &= ~(1ull << (p & 63));
                }
            }
            va += seg;
        }
    }
}

// BBHOST_RESOURCE_CENSUS=1: the write faults on mirror pages, by what GX says
// of the 4 KiB page each hit, and the writers behind each kind - whether GX's
// calls could tell the mirrors of the writes the watch now catches. Taken at
// each report from the faults since the last (the ring holds 16384).
// A fault's class: what covers its 4 KiB (GX buffers whole, a GX resource in
// part, a texture or buffer of YEBIS's own device, a Scaleform texture,
// nothing) times what handed it out (a Map, the upload pool, the ring,
// nothing GX said), as owner * (kHandKinds + 1) + handed.
constexpr unsigned kFaultHanders = kHandKinds + 1, kFaultOwners = 6, kFaultClasses = kFaultOwners * kFaultHanders;
const char* const kFaultGxName[kFaultOwners] = {"GX buffers'", "a GX resource's in part", "YEBIS's", "Scaleform's", "nobody's",
                                                "a told page's (BBHOST_SHADOW_VERIFY=2)"};
const char* const kFaultHandName[kFaultHanders] = {"Maps hand out", "the upload pool hands out", "the ring hands out", "YEBIS's discard Maps hand out",
                                                    "nothing GX hands out"};
std::uint64_t g_fault_census_seq = 0, g_fault_census_other = 0;
std::uint64_t g_fault_class[kFaultClasses] = {};
std::unordered_map<std::uint64_t, std::uint64_t> g_fault_writers[kFaultClasses];

void census_faults_locked() {
    if (!g_map_census) return;
    constexpr int kMax = 16384;
    static std::vector<std::uint64_t> addrs(kMax), pcs(kMax);
    static std::vector<std::uint8_t> owners(kMax);
    std::uint64_t newest = g_fault_census_seq;
    const int n = write_watch_faults_since(g_fault_census_seq, addrs.data(), pcs.data(), owners.data(), kMax, &newest);
    g_fault_census_seq = newest;
    for (int i = 0; i < n; ++i) {
        if (!(owners[i] & kWatchOwnerMirror)) {
            ++g_fault_census_other;
            continue;
        }
        const std::uint64_t page = addrs[i] & ~0xfffull;
        std::uint64_t phys = 0, span = 0;
        const bool known = phys_at(page, phys, span);
        unsigned by = kHandKinds;
        for (unsigned k : {kHandMap, kHandPool, kHandYebis, kHandRing}) {
            if (known && handed(k, phys)) {
                by = k;
                break;
            }
        }
        const int mw = gx_middleware_at(page, 0x1000);
        bool told_page = false;
        if (known && g_verify_watch_told) {
            const Mirror& tm = g_mirrors[phys >> gcn::kDmemChunkShift];
            if (!tm.pages.empty()) told_page = tm.pages[(phys & (kChunkBytes - 1)) >> kPageShift].told;
        }
        const unsigned gx = told_page                                      ? 5
                            : gx_buffers_cover(page, 0x1000)                 ? 0
                            : gx_resource_overlapping(page, 0x1000, nullptr) ? 1
                            : mw                                             ? 1 + static_cast<unsigned>(mw)
                                                                             : 4;
        const unsigned c = gx * kFaultHanders + by;
        ++g_fault_class[c];
        auto& w = g_fault_writers[c];
        if (w.size() < 4096 || w.count(pcs[i])) ++w[pcs[i]];
    }
}

// The census on lines of its own (a log line holds 1024 bytes, which the
// shadow's report fills by itself): the faults by class at every report, and
// every twelfth (about a minute) each class with its four first writers.
const char* const kFaultGxCode[kFaultOwners] = {"gx-buffers", "gx-part", "yebis", "scaleform", "nobody", "told"};
const char* const kFaultHandCode[kFaultHanders] = {"map", "pool", "ring", "yebis-rename", "none"};
void log_fault_census() {
    static unsigned calls = 0;
    std::string line;
    for (unsigned c = 0; c < kFaultClasses; ++c) {
        if (!g_fault_class[c]) continue;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s %s/%s %llu", line.empty() ? "" : ",", kFaultGxCode[c / kFaultHanders], kFaultHandCode[c % kFaultHanders],
                      static_cast<unsigned long long>(g_fault_class[c]));
        line += buf;
    }
    host_log("  buffer shadow census: mirror write faults by memory owner/hander:%s; other pages' faults %llu; 4 KiB pages handed out: by "
             "Maps %llu (in pool memory %llu, in ring memory %llu), by the upload pool %llu, by the ring %llu, by YEBIS's renames %llu; GX "
             "resources made over handed-out memory %llu",
             line.empty() ? " none" : line.c_str(), static_cast<unsigned long long>(g_fault_census_other),
             static_cast<unsigned long long>(g_hand_pages[kHandMap]), static_cast<unsigned long long>(g_map_in_pool),
             static_cast<unsigned long long>(g_map_in_ring), static_cast<unsigned long long>(g_hand_pages[kHandPool]),
             static_cast<unsigned long long>(g_hand_pages[kHandRing]), static_cast<unsigned long long>(g_hand_pages[kHandYebis]),
             static_cast<unsigned long long>(g_made_over_handed));
    if (calls++ % 12 != 0) return;
    for (unsigned c = 0; c < kFaultClasses; ++c) {
        if (!g_fault_class[c]) continue;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> top(g_fault_writers[c].begin(), g_fault_writers[c].end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string w;
        for (std::size_t k = 0; k < top.size() && k < (c / kFaultHanders == 5 ? 16u : 6u); ++k) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s0x%llx x%llu", k ? ", " : "", static_cast<unsigned long long>(top[k].first ? gx_guest_to_bn(top[k].first) : 0),
                          static_cast<unsigned long long>(top[k].second));
            w += buf;
        }
        host_log("  buffer shadow census: %s/%s (%s memory %s): %llu faults; writers %s", kFaultGxCode[c / kFaultHanders],
                 kFaultHandCode[c % kFaultHanders], kFaultGxName[c / kFaultHanders], kFaultHandName[c % kFaultHanders],
                 static_cast<unsigned long long>(g_fault_class[c]), w.c_str());
    }
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
    return g_shadow_env >= 0 ? g_shadow_env == 1 : !g.tight && (!g.integrated || (integrated_with_room() && !g_lite_wanted));
}

// Lite: where the mirrors were on only for their page tracking - an
// integrated GPU with room - they stay off, and the vertex data the game
// rewrites is told by GX instead. In Kyo's Central Yharnam save on the Radeon
// 8060S (1080p, the GPU behind), 77,437 of the 77,454 draws whose vertices the
// CPU rewrote before their submission ran were in memory a Map had handed out
// (the other 17 under a loading screen) - and the mirrors let 79,218 of 9.2
// million race there too, in ranges across two of their windows. Their
// tracking cost the command processor about a third of its time (176 against
// 115-135 fps at 640x360 without it). Off with BBHOST_SHADOW_LITE=0, or any
// BBHOST_BUFFER_SHADOW.
bool shadow_lite_on() { return g_shadow_env < 0 && g_lite_wanted && !g.tight && g.integrated && integrated_with_room(); }
bool shadow_in_place_on() { return buffer_shadow_on() && in_place_on(); }

namespace {
// Why the last shadow_locate_locked on this thread left its range where it
// was, or that a mirror serves it (shadow_locate_why).
thread_local std::uint8_t t_locate_why = kLocNotAsked;
// In place (in_place_on): true when [p0, p1] binds the import - each page
// placed already, checked once a recording for a told write since (which
// counts as a rewrite, as a copy did, and places it again), or placed now.
// False for a range with a page GX does not tell the writes of, or one the
// game keeps rewriting (host memory for good, its ranges copied as their
// draws record): the mirror path takes those as before.
bool place_range_locked(Mirror& m, std::uint32_t chunk, std::uint64_t va, std::uint64_t off, std::uint32_t p0, std::uint32_t p1,
                        std::uint64_t map_lo, std::uint64_t map_hi, std::uint64_t stamp) {
    begin_recording(m, stamp);
    if (m.placed_ok.empty()) m.placed_ok.assign(kPagesPerChunk / 64, 0);
    if (all_ok(m.placed_ok, p0, p1)) return true;
    if (m.placed_valid.empty()) m.placed_valid.assign(kPagesPerChunk / 64, 0);
    // Every page placed and told of no write since: what the walk below
    // would find and leave (placed, eligible, placed_ok for the recording).
    // BBHOST_SHADOW_PLACED_VALID=0 walks them as before.
    if (g_placed_valid_on && all_ok(m.placed_valid, p0, p1)) {
        mark_bits(m.placed_ok, p0, p1, true);
        return true;
    }
    // Each page's facts: a placed page with no told write since stands as
    // eligible without asking GX again; any other is asked (shadow_place.h).
    thread_local std::vector<shadow_place::PageFacts> facts;
    facts.assign(p1 - p0 + 1, shadow_place::PageFacts{});
    const auto page_va = [&](std::uint32_t p) { return va - (off - (static_cast<std::uint64_t>(p) << kPageShift)); };
    bool out = false;  // a page is not eligible: no more of GX's lookups, only whether a later one is rewritten
    for (std::uint32_t p = p0; p <= p1; ++p) {
        Page& pg = m.pages[p];
        shadow_place::PageFacts& f = facts[p - p0];
        f.dynamic = pg.dynamic;
        if (pg.placed && pg.placed_checked != stamp) {
            pg.placed_checked = stamp;
            if (shadow_place::rewritten(pg.placed_seq, pg.gx_written)) pg.placed = false;
        }
        if (pg.placed || f.dynamic || out) {
            f.told_cover = pg.placed;
            continue;
        }
        const std::uint64_t lo = std::max(page_va(p), map_lo), hi = std::min(page_va(p) + kPageBytes, map_hi);
        f.handed = page_handed(chunk, p);
        if (!g_shadow_gx || f.handed || hi <= lo) {
            f.told_cover = false;
        } else if (pg.cover_stamp == stamp && pg.cover_gx == pg.gx_written && pg.cover_epoch == g_told_epoch && pg.cover_va == lo &&
                   pg.cover_bytes == hi - lo) {
            f.told_cover = pg.cover;
        } else {
            f.told_cover = gx_buffers_cover(lo, hi - lo);
            pg.cover_stamp = stamp;
            pg.cover_gx = pg.gx_written;
            pg.cover_epoch = g_told_epoch;
            pg.cover_va = lo;
            pg.cover_bytes = static_cast<std::uint32_t>(hi - lo);
            pg.cover = f.told_cover;
        }
        out = !shadow_place::eligible(f);
    }
    const shadow_place::Range r = shadow_place::choose(facts.data(), p1 - p0 + 1);
    if (r != shadow_place::Range::kInPlace) {
        if (r == shadow_place::Range::kMirror) ++g_placed_refused;
        return false;
    }
    // Every page may be read in place: those not placed yet are placed now,
    // each placement a rewrite as a copy was.
    const std::uint64_t t = now_ms();
    for (std::uint32_t p = p0; p <= p1; ++p) {
        Page& pg = m.pages[p];
        if (!pg.placed) {
            shadow_place::RewriteWindow w{pg.window_ms, pg.window_uploads};
            const bool too_many = shadow_place::rewrite_too_many(w, t, kMaxUploads, kUploadWindowMs);
            pg.window_ms = w.start_ms;
            pg.window_uploads = w.count;
            if (too_many) {
                go_dynamic(m, p, page_va(p), "keeps being rewritten (in place)");
                return false;
            }
            if (pg.placed_seq) ++g_placed_again;
            pg.placed = true;
            pg.placed_seq = std::max<std::uint64_t>(g_told_seq, 1);
            pg.placed_checked = stamp;
            ++g_placed;
        }
        m.placed_ok[p >> 6] |= 1ull << (p & 63);
        m.placed_valid[p >> 6] |= 1ull << (p & 63);
    }
    return true;
}
}  // namespace

namespace {
// BBHOST_RUN_COMMIT: a run part's draw asks the mirrors (shadow_locate_locked)
// without changing what they hold for anyone else. A range whose pages the
// recording has checked (ok) is served, and one with a page bound from host
// memory for good or written by the GPU in this recording stays in host
// memory - as in order. A valid page's check for the recording - the first
// question about it since the recording began, a look at its write watch or
// its told writes - is made here as in order, under a lock of the parts' own,
// and the ok bits the parts read while another sets them are atomics. Anything
// else - the mirror or a window to make, a page to copy, the room for more -
// stops the draw (kRunStopMirror, a local stop: a draw that reads the page
// stops there too), and the command processor asks again in order.
HostLock g_part_mirror_mu;
bool part_all_ok(std::vector<std::uint64_t>& ok, std::uint32_t p0, std::uint32_t p1) {
    for (std::uint32_t w = p0 >> 6; w <= p1 >> 6; ++w) {
        std::uint64_t mask = ~0ull;
        if (w == p0 >> 6) mask &= ~0ull << (p0 & 63);
        if (w == p1 >> 6) mask &= ~0ull >> (63 - (p1 & 63));
        if ((std::atomic_ref<std::uint64_t>(ok[w]).load(std::memory_order_acquire) & mask) != mask) return false;
    }
    return true;
}
void part_locate_locked(std::uint32_t chunk, std::uint64_t va, std::uint64_t off, std::uint64_t bytes, Located& loc) {
    Mirror& m = g_mirrors[chunk];
    if (m.failed) { t_locate_why = kLocNoMirror; return; }  // (mirror_for's answer)
    if (!m.buffer && !m.windowed) {
        // Being made by the keeper: host memory meanwhile, as in order (the
        // waits are the command processor's count). Else asked for in order.
        if (!m.making) run_part_stop(kRunStopMirror);
        t_locate_why = m.making ? kLocNoMirror : kLocNotAsked;
        return;
    }
    const std::uint64_t stamp = g.record_serial + 1;
    const std::uint32_t p0 = static_cast<std::uint32_t>(off >> kPageShift);
    const std::uint32_t p1 = static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift);
    if (p1 >= m.pages.size()) { t_locate_why = kLocNoMirror; return; }
    const std::uint32_t win = m.windowed ? p0 / window_pages() : 0;
    if (m.windowed && p1 / window_pages() != win) {  // (the crossings are the command processor's count)
        bump(g_imports);
        t_locate_why = kLocCrossing;
        return;
    }
    const auto serve = [&] {
        bump(g_hits);
        loc.buffer = m.windowed ? m.windows[win].buffer : m.buffer;
        loc.offset = off - static_cast<VkDeviceSize>(win) * window_bytes();
        loc.avail = bytes;
        t_locate_why = kLocServed;
    };
    if (std::atomic_ref<std::uint64_t>(m.ok_stamp).load(std::memory_order_acquire) == stamp && part_all_ok(m.ok, p0, p1)) {
        serve();
        return;
    }
    std::lock_guard<HostLock> lk(g_part_mirror_mu);
    if (m.ok_stamp != stamp) {  // the recording's first question to this mirror (begin_recording)
        std::fill(m.ok.begin(), m.ok.end(), 0);
        std::fill(m.gpu.begin(), m.gpu.end(), 0);
        std::fill(m.placed_ok.begin(), m.placed_ok.end(), 0);
        std::atomic_ref<std::uint64_t>(m.ok_stamp).store(stamp, std::memory_order_release);
    }
    if (part_all_ok(m.ok, p0, p1)) {  // (checked meanwhile by another part)
        serve();
        return;
    }
    if (any_set(m.dyn, p0, p1) || any_set(m.gpu, p0, p1)) {
        bump(g_imports);
        t_locate_why = kLocHost;
        return;
    }
    if (g_live && m.sparse && (m.blocks.empty() || m.block_pages_used == kBlockPages) && !all_ok(m.resident, p0, p1) &&
        now_ms() < g_room_retry_ms) {  // (the skips are the command processor's count)
        bump(g_imports);
        t_locate_why = kLocRoom;
        return;
    }
    if (m.windowed && !m.windows[win].buffer) {
        // Being made by the keeper: host memory meanwhile, as in order. Else
        // the window asked for, or the range left in host memory, in order.
        if (m.windows[win].making) {
            bump(g_imports);
            t_locate_why = kLocWindow;
        } else {
            run_part_stop(kRunStopMirror);
            t_locate_why = kLocNotAsked;
        }
        return;
    }
    const bool staged = !imports_whole();
    for (std::uint32_t p = p0; p <= p1; ++p) {
        Page& pg = m.pages[p];
        if (pg.dynamic || (!pg.valid && pg.gpu_written == stamp) ||
            (staged && !pg.valid && gpu_writes_pending_locked(va - (off - (static_cast<std::uint64_t>(p) << kPageShift)), kPageBytes))) {
            bump(g_imports);
            t_locate_why = kLocHost;
            return;
        }
        if (pg.valid && pg.checked != stamp) {
            pg.checked = stamp;
            ++g_page_checks;
            if (page_written(pg)) pg.valid = false;
        }
        if (!pg.valid) {
            run_part_stop(kRunStopMirror);  // copied in order
            t_locate_why = kLocNotAsked;
            return;
        }
        std::atomic_ref<std::uint64_t>(m.ok[p >> 6]).fetch_or(1ull << (p & 63), std::memory_order_release);
    }
    serve();
}
}  // namespace

void shadow_locate_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    if (!buffer_shadow_on() || !bytes || !loc.buffer) { t_locate_why = kLocNotAsked; return; }
    if (twin_covers_locked(va, bytes)) { t_locate_why = kLocTwin; return; }  // a GX buffer's device-local twin: already device memory, its guest bytes stale
    std::uint32_t chunk = 0;
    std::uint64_t off = 0, map_lo = 0, map_hi = 0;
    if (!phys_of(va, bytes, chunk, off, &map_lo, &map_hi)) { t_locate_why = kLocNoMirror; return; }
    if (run_part()) {
        if (g_verify || in_place_on()) {  // (refused at the fork: BBHOST_SHADOW_VERIFY, the mirrors read in place)
            run_part_stop(kRunStopMirror);
            t_locate_why = kLocNotAsked;
            return;
        }
        part_locate_locked(chunk, va, off, bytes, loc);
        return;
    }
    Mirror* m = mirror_for(chunk);
    if (!m) { t_locate_why = kLocNoMirror; return; }
    drain_gx_writes_locked();
    const std::uint64_t stamp = g.record_serial + 1;
    const std::uint32_t p0 = static_cast<std::uint32_t>(off >> kPageShift);
    const std::uint32_t p1 = static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift);
    // (Never past the mirror's pages: a copy or a sparse bind there runs past
    // the end of its buffer.)
    if (p1 >= m->pages.size()) { t_locate_why = kLocNoMirror; return; }
    // In place: the import is the binding, and `loc` already holds it - or,
    // with window imports, holds the stand-in of shadow_mirror_locked's
    // question, whose caller then binds the import locate() finds or makes
    // (and asks again, which counts the bind).
    if (in_place_on() && place_range_locked(*m, chunk, va, off, p0, p1, map_lo, map_hi, stamp)) {
        if (loc.buffer != g.sink.buffer) bump(g_placed_binds);
        t_locate_why = kLocInPlace;
        return;
    }
    // A windowed mirror serves a range from one window's buffer.
    const std::uint32_t win = m->windowed ? p0 / window_pages() : 0;
    if (m->windowed && p1 / window_pages() != win) {
        ++g_window_crossings;
        bump(g_imports);
        t_locate_why = kLocCrossing;
        return;
    }
    const VkDeviceSize win_base = static_cast<VkDeviceSize>(win) * window_bytes();
    const auto mirror_buffer = [&] { return m->windowed ? m->windows[win].buffer : m->buffer; };
    if (g_verify) {
        for (std::uint32_t p = p0; p <= p1; ++p) {
            Page& pg = m->pages[p];
            const std::uint64_t page_lo = static_cast<std::uint64_t>(p) << kPageShift;
            const std::uint32_t lo = static_cast<std::uint32_t>(std::max(off, page_lo) - page_lo);
            const std::uint32_t hi = static_cast<std::uint32_t>(std::min(off + bytes, page_lo + kPageBytes) - page_lo);
            pg.read_lo = std::min(pg.read_lo, lo);
            pg.read_hi = std::max(pg.read_hi, hi);
        }
    }
    {
        const std::uint32_t span = p1 - p0 + 1;
        ++g_span_hist[span == 1 ? 0 : span <= 4 ? 1 : span <= 16 ? 2 : span <= 64 ? 3 : span <= 256 ? 4 : 5];
    }
    begin_recording(*m, stamp);
    if (all_ok(m->ok, p0, p1)) {
        bump(g_hits);
        loc.buffer = mirror_buffer();
        loc.offset = off - win_base;
        loc.avail = bytes;
        t_locate_why = kLocServed;
        return;
    }
    // A page bound from host memory for good, or one a GPU write in this
    // recording left uncopied (a copy now would run before that write): the
    // walk below would stop at it, having looked at every page before it.
    if (any_set(m->dyn, p0, p1) || any_set(m->gpu, p0, p1)) {
        bump(g_imports);
        t_locate_why = kLocHost;
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
        t_locate_why = kLocRoom;
        return;
    }
    // A windowed mirror's window is made before any page's write watch is
    // armed; with no room for it the range is read from host memory, and a
    // later draw asks again (the live share) - with the fixed share it stays
    // there, as a chunk with no whole mirror did.
    if (m->windowed && !m->windows[win].buffer) {
        const bool retry_wait = g_live && now_ms() < g_room_retry_ms;
        if (retry_wait || !window_ready(*m, win)) {
            if (m->windows[win].making) ++g_async_waits;  // (the keeper making it, not a refusal)
            else if (g_live) ++(retry_wait ? g_room_skips : g_room_refusals);
            bump(g_imports);
            t_locate_why = kLocWindow;
            return;
        }
    }
    // The pages this call will copy, armed a run at a time: one protection of
    // the guest view and one of its GPU alias per run of consecutive pages,
    // instead of two per 64 KiB page - each is an mprotect, and the new
    // buffers of a fight's first frame armed hundreds of pages (~9% of the
    // command processor there). The walk stops where the loop below would;
    // a run that cannot be armed whole is left to the loop, a page at a time.
    thread_local std::vector<std::uint8_t> prearmed;  // per page of [p0, p1]: 1 view armed, 2 alias armed
    prearmed.assign(p1 - p0 + 1, 0);
    // Per page of [p0, p1] that this call copies: 1 told (GX buffers cover
    // it, not watched), 2 watched; 0 not decided here.
    thread_local std::vector<std::uint8_t> told_class;
    told_class.assign(p1 - p0 + 1, 0);
    const auto page_told = [&](std::uint32_t p) {
        const std::uint64_t page_va = va - (off - (static_cast<std::uint64_t>(p) << kPageShift));
        const std::uint64_t lo = std::max(page_va, map_lo), hi = std::min(page_va + kPageBytes, map_hi);
        if (!g_shadow_gx) return false;
        if (page_handed(chunk, p)) return g_shadow_handed;
        return gx_buffers_cover(lo, hi - lo);
    };
    {
        const auto arm_run = [&](std::uint32_t a, std::uint32_t b) {  // pages [a, b)
            if (b - a < 2) return;  // one page: the loop's own arming is the same work
            const std::uint64_t first_va = va - (off - (static_cast<std::uint64_t>(a) << kPageShift));
            const std::uint64_t lo = std::max(first_va, map_lo);
            const std::uint64_t hi = std::min(first_va + (static_cast<std::uint64_t>(b - a) << kPageShift), map_hi);
            WriteWatch w, aw;
            w.owner = aw.owner = kWatchOwnerMirror;
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
                if (page_written(pg)) pg.valid = false;
            }
            if (pg.valid) {  // nothing to copy: the run ends before it
                arm_run(run, p);
                run = p + 1;
                continue;
            }
            // Told of its writes by GX: not armed, and it ends a run of watched pages.
            told_class[p - p0] = page_told(p) ? 1 : 2;
            const std::uint32_t uploads = t - pg.window_ms >= kUploadWindowMs ? 0 : pg.window_uploads;
            const std::uint32_t limit = told_class[p - p0] == 1 && page_handed(chunk, p) ? kMaxUploadsHanded : kMaxUploads;
            if (!pg.provisional && uploads + 1 > limit) break;  // the loop sends it to host memory and stops there
            if (pg.provisional && pg.provisional_run + 1 >= kMaxProvisionalRun) break;
            if (told_class[p - p0] == 1) {
                arm_run(run, p);
                run = p + 1;
            }
        }
        arm_run(run, p);
    }
    // Pages to copy, as runs, from the import that holds them all - or, with
    // window imports, each page copied into upload staging here (after its
    // write watch is armed) and from there into the mirror: no import of it.
    const bool staged = !imports_whole() || t_force_staged;
    const Chunk* source = staged ? nullptr
                                 : dmem_import(chunk, static_cast<std::uint64_t>(p0) << kPageShift,
                                               static_cast<std::uint64_t>(p1 - p0 + 1) << kPageShift);
    if (!staged && !source) {
        bump(g_imports);
        t_locate_why = kLocNoImport;
        return;
    }
    VkBuffer src_buffer = source ? source->buffer : VK_NULL_HANDLE;
    VkBufferCopy regions[16];
    std::uint32_t n_regions = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    auto flush = [&] {
        if (n_regions) vkCmdCopyBuffer(cmd, src_buffer, mirror_buffer(), n_regions, regions);
        n_regions = 0;
    };
    for (std::uint32_t p = p0; p <= p1; ++p) {
        Page& pg = m->pages[p];
        const std::uint64_t page_va = va - (off - (static_cast<std::uint64_t>(p) << kPageShift));
        if (pg.dynamic) {
            flush();
            bump(g_imports);
            t_locate_why = kLocHost;
            return;
        }
        // Written by the GPU in this recording: a copy now would run before
        // that write. In a later recording the copy is ordered after it.
        if (!pg.valid && pg.gpu_written == stamp) {
            flush();
            bump(g_imports);
            t_locate_why = kLocHost;
            return;
        }
        // With window imports the page's copy is the CPU's, taken now: a GPU
        // write still to land on it - an earlier submission's copy-back, fill
        // or stores the GPU has not run yet - would be missed, and the page
        // kept as valid with what it held before. From an import the GPU's
        // copy runs after that write, in queue order; staged, the page stays
        // in host memory until the write has landed.
        if (staged && !pg.valid && gpu_writes_pending_locked(page_va, kPageBytes)) {
            bump(g_staged_pending);
            flush();
            bump(g_imports);
            t_locate_why = kLocGpuPending;
            return;
        }
        if (pg.valid) {
            if (pg.checked != stamp) {
                pg.checked = stamp;
                ++g_page_checks;
                if (page_written(pg)) pg.valid = false;
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
        // A copy again for an outstanding renamed copy is not a rewrite; a page
        // that stays unsettled copy after copy goes to host memory too.
        if (pg.provisional) {
            if (++pg.provisional_run >= kMaxProvisionalRun) {
                go_dynamic(*m, p, page_va, "stays under renamed copies GX does not write back");
                ++g_handed_dynamic;
                flush();
                bump(g_imports);
                t_locate_why = kLocHost;
                return;
            }
        } else if (++pg.window_uploads > (g_shadow_handed && page_handed(chunk, p) ? kMaxUploadsHanded : kMaxUploads)) {
            go_dynamic(*m, p, page_va, "keeps being rewritten");
            flush();
            bump(g_imports);
            t_locate_why = kLocHost;
            return;
        }
        // Watched from before the copy reads it: a write after that marks it
        // again rather than going unseen. Only the part in this mapping. A
        // page GX buffers cover is told of its writes instead (told_written).
        const std::uint64_t lo = std::max(page_va, map_lo), hi = std::min(page_va + kPageBytes, map_hi);
        const bool told = told_class[p - p0] ? told_class[p - p0] == 1 : page_told(p);
        if (told && !g_verify_watch_told) {
            pg.watch = WriteWatch{};
            pg.alias_watch = WriteWatch{};
            pg.alias_va = 0;
            ++g_told_copies;
        } else if (told) {
            // Watched for the census only (BBHOST_SHADOW_VERIFY=2): told all the same.
            ++g_told_copies;
            pg.watch = WriteWatch{};
            pg.watch.owner = kWatchOwnerMirror;
            if (!(prearmed[p - p0] & 1)) hle_kernel_write_watch(lo, hi - lo, pg.watch);
            pg.alias_watch = WriteWatch{};
            pg.alias_va = 0;
        } else {
            if (g_shadow_gx) ++(gx_resource_overlapping(lo, hi - lo, nullptr) ? g_watched_partial : g_watched_none);
            pg.watch.owner = pg.alias_watch.owner = kWatchOwnerMirror;
            if (!(prearmed[p - p0] & 1) && !hle_kernel_write_watch(lo, hi - lo, pg.watch)) {
                go_dynamic(*m, p, page_va, "cannot be write-watched");
                flush();
                bump(g_imports);
                t_locate_why = kLocHost;
                return;
            }
            if (!(prearmed[p - p0] & 2)) pg.alias_va = hle_kernel_write_watch_alias(lo, hi - lo, pg.alias_watch);
        }
        pg.told = told;
        pg.told_seq = g_told_seq;
        const bool provisional = told && g_shadow_handed && page_unsettled(chunk, p);
        if (!provisional) pg.provisional_run = 0;
        pg.provisional = provisional;
        if (provisional) ++g_provisional_copies;
        if (!cmd && !(cmd = upload_cmd_locked())) {
            pg.watch = WriteWatch{};
            pg.alias_watch = WriteWatch{};
            pg.alias_va = 0;
            t_locate_why = kLocNoCopy;
            return;
        }
        VkDeviceSize staged_at = 0;  // the page's copy in staging (window imports)
        if (staged) {
            DevBuffer sb;
            if (!acquire_staging_locked(sb, kPageBytes, staged_at)) {
                pg.watch = WriteWatch{};
                pg.alias_watch = WriteWatch{};
                pg.alias_va = 0;
                if (pg.window_uploads) --pg.window_uploads;
                flush();
                bump(g_imports);
                t_locate_why = kLocNoCopy;
                return;
            }
            // The placement's own map (staged_at is its offset in the buffer, for the copy).
            const void* page_src = static_cast<const std::uint8_t*>(g.mirror) + static_cast<std::uint64_t>(chunk) * kChunkBytes +
                                   (static_cast<std::uint64_t>(p) << kPageShift);
            if (StagedCopies::get().on()) {
                StagedCopies::get().push(sb.map, page_src, kPageBytes);  // done before the submission (shadow_end_uploads_locked)
            } else {
                std::memcpy(sb.map, page_src, kPageBytes);
            }
            if (sb.buffer != src_buffer) {
                flush();
                src_buffer = sb.buffer;
            }
            bump(g_staged_pages);
        }
        pg.va = lo;
        pg.bytes = hi - lo;
        pg.valid = true;
        pg.checked = stamp;
        if (g_map_census) pg.gx_seq = gx_write_seq_now();
        if (g_verify) {
            const auto* src = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(lo));
            pg.hash = fnv1a(src, hi - lo);
            // This draw's bytes in the page; later draws add theirs.
            const std::uint64_t page_lo = static_cast<std::uint64_t>(p) << kPageShift;
            pg.read_lo = static_cast<std::uint32_t>(std::max(off, page_lo) - page_lo);
            pg.read_hi = static_cast<std::uint32_t>(std::min(off + bytes, page_lo + kPageBytes) - page_lo);
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
                    t_locate_why = kLocRoom;
                    return;
                }
                MemOwner own_mem("mirror blocks");
                const auto block_t0 = std::chrono::steady_clock::now();
                g_block_type = bai.memoryTypeIndex;
                const bool from_spare = room && g_spare_block != VK_NULL_HANDLE && g_spare_type == bai.memoryTypeIndex;
                if (from_spare) {
                    mem = g_spare_block;  // made ahead by the keeper
                    g_spare_block = VK_NULL_HANDLE;
                    ++g_blocks_spare;
                    if (g_async) keeper_wake();  // the next spare
                } else if (!room || vkAllocateMemory(g.device, &bai, nullptr, &mem) != VK_SUCCESS) {
                    go_dynamic(*m, p, page_va, !room ? "has no room in the mirrors' share of video memory"
                                                     : "has no device memory for its mirror page");
                    flush();
                    bump(g_imports);
                    t_locate_why = kLocHost;
                    return;
                } else {
                    g_block_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - block_t0).count();
                    ++g_blocks_made;
                    if (g_async) keeper_wake();  // a spare for the next one
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
        // The source is that import, which starts at the chunk's byte lo, or
        // the page's copy in staging. A run grows only where both ends meet
        // (copy_runs.h): the staged copies follow one another in staging
        // across the valid pages skipped between them, and grown on the
        // source alone a page's copy landed on the page after the last one
        // copied - a valid page overwritten, its own left as it was: the
        // Dream's meshes drew from the wrong bytes.
        const VkDeviceSize src = staged ? staged_at : at - source->lo;
        const VkDeviceSize dst = at - win_base;  // in the window's buffer, for a windowed mirror
        if (n_regions && copy_run_extends(regions[n_regions - 1], src, dst, kPageBytes)) {
        } else {
            if (n_regions == 16) flush();
            regions[n_regions++] = {src, dst, kPageBytes};
        }
        bump(g_uploads);
        bump(g_upload_bytes, kPageBytes);
    }
    flush();
    bump(g_hits);
    loc.buffer = mirror_buffer();
    loc.offset = off - win_base;
    loc.avail = bytes;
    t_locate_why = kLocServed;
}

std::uint8_t shadow_locate_why() { return t_locate_why; }
void shadow_count_placed_bind() { bump(g_placed_binds); }

unsigned shadow_hand_kinds_locked(std::uint64_t va) {
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!va || !phys_of(va, 1, chunk, off)) return 0;
    drain_gx_writes_locked();
    const std::uint64_t phys = (std::uint64_t{chunk} << gcn::kDmemChunkShift) | off;
    unsigned bits = 0;
    for (unsigned k = 0; k < kHandKinds; ++k) {
        if (handed(k, phys)) bits |= 1u << k;
    }
    if (page_unsettled(chunk, static_cast<std::uint32_t>(off >> kPageShift))) bits |= 1u << kHandKinds;
    return bits;
}

// For a diagnostic: what the mirror would do with this range right now, and -
// the question that matters - whether a page it is still serving has been
// written since the recording's one check of it. A valid page is checked once
// per recording; a guest write after that check is not seen again until the
// next one, so `valid + checked-this-recording + written` is a page whose
// mirror the GPU will read stale. Pure: `write_watch_dirty` only memoises a
// clean answer, and a written watch stays written until it is armed again.
bool shadow_mirror_locked(std::uint64_t va, std::uint64_t bytes, Located& out) {
    t_locate_why = kLocNotAsked;  // (shadow_locate_why: not asked, unless the question below is)
    if (!buffer_shadow_on() || !bytes || !g.sink.buffer || twin_covers_locked(va, bytes)) return false;
    // A stand-in host location: shadow_locate_locked replaces it with the
    // mirror when the mirror serves the range, and leaves it otherwise.
    Located probe{g.sink.buffer, 0, bytes};
    shadow_locate_locked(va, bytes, probe);
    if (probe.buffer == g.sink.buffer) return false;
    out = probe;
    return true;
}

bool shadow_stage_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    if (!buffer_shadow_on() || !bytes || !loc.buffer || loc.avail < bytes || twin_covers_locked(va, bytes)) return false;
    if (run_part_stop(kRunStopMirror)) return false;  // staged in order (BBHOST_RUN_COMMIT)
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
    // Read in place on purpose (BBHOST_SHADOW_IN_PLACE): on an integrated GPU
    // a copy for the draw saves no bus either.
    if (!m.placed_ok.empty() && m.ok_stamp == stamp &&
        all_ok(m.placed_ok, static_cast<std::uint32_t>(off >> kPageShift), static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift)))
        return false;
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
        // The copy that starts at or before the range, in the same import:
        // read from it when it reaches the range's end.
        auto& seen_map = g_stage_seen[slot];
        auto it = seen_map.upper_bound(StageKey{loc.buffer, loc.offset});
        if (it != seen_map.begin()) {
            --it;
            const VkDeviceSize into = loc.offset - it->first.offset;
            if (it->first.src == loc.buffer && it->second.bytes >= into + bytes) {
                loc.buffer = it->second.buffer;
                loc.offset = it->second.offset + into;
                loc.avail = bytes;
                bump(g_stage_shared);
                if (into) bump(g_stage_inside);
                g_stage_shared_bytes.fetch_add(bytes, std::memory_order_relaxed);
                return true;
            }
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
    if (g_stage_census) {
        StageStat& st = g_stage_stats[va];
        if (st.why < 0) {
            st.why = 3;
            for (std::uint64_t p = off >> kPageShift; p <= (off + bytes - 1) >> kPageShift; ++p) {
                const Page& pg = m.pages[p];
                const int w = pg.dynamic ? 0 : (!pg.valid && pg.gpu_written) ? 1 : !pg.valid ? 2 : 3;
                st.why = std::min(st.why, w);
            }
        }
        st.bytes += bytes;
        ++st.copies;
        st.record = std::max<std::uint64_t>(st.record, bytes);
        g_stage_why_bytes[st.why] += bytes;
    }
    if (g_stage_dedup) {
        StagedRange& seen = g_stage_seen[slot][StageKey{loc.buffer, loc.offset}];
        if (bytes >= seen.bytes) seen = StagedRange{c.buffer, g_stage_fill[slot], bytes};
    }
    loc.buffer = c.buffer;
    loc.offset = g_stage_fill[slot];
    loc.avail = bytes;
    g_stage_fill[slot] += need;
    bump(g_staged);
    bump(g_staged_bytes, bytes);
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

bool snap_copy_locked(std::uint64_t va, std::uint64_t bytes, Located& loc);  // below

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
    // Either only while the heap's budget leaves room (ring_fits_budget).
    std::uint32_t type = UINT32_MAX;
    if (g.integrated || rings_in_vram()) type = find_memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::string note;
    if (type != UINT32_MAX && !ring_fits_budget(type, req.size, "the rewritten vertex data's ring", &note)) type = UINT32_MAX;
    if (type != UINT32_MAX) {
        mai.memoryTypeIndex = type;
        MemOwner own_mem("vertex copies ring");
        if (vkAllocateMemory(g.device, &mai, nullptr, &g_snap.memory) == VK_SUCCESS) g_snap.device_local = true;
        note_ring_placed(g_snap.device_local, req.size);
    }
    if (!g_snap.memory) {
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, visible);
        MemOwner own_mem("vertex copies ring");
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
    host_log("buffer shadow: vertex data the game rewrites is copied as its draws record, into a %llu MiB ring in %s%s%s "
             "(BBHOST_VERTEX_SNAPSHOT=0: read as the GPU reaches it)",
             static_cast<unsigned long long>(kSnapRing >> 20),
             g_snap.device_local ? "video memory the CPU writes" : "host memory, staged into video memory per draw",
             g_snap.device_local && !note.empty() ? " - " : "", g_snap.device_local ? note.c_str() : "");
    return true;
}
}  // namespace

thread_local constinit std::uint64_t t_lite_draw = 0;
thread_local constinit DrawCtx* t_draw_ctx = nullptr;
thread_local constinit BumpSink* t_bump_sink = nullptr;  // BBHOST_RUN_COMMIT: a run worker's counters
thread_local constinit RunDraw* t_run_draw = nullptr;    // BBHOST_RUN_COMMIT: a forked run's draw
thread_local constinit RunDigest t_run_digest;           // BBHOST_RUN_COMMIT=compare: the drawing thread's last draw's
HostLock g_part_ring_mu;                       // BBHOST_RUN_COMMIT: the parts' chunk takes from the copy rings

namespace {
// The lite path's question (shadow_lite_on): whether a Map handed out a page of
// [va, va + bytes) within the last kLiteFlips flips. A ring's or a renamed
// copy's draws come right after the Map that handed their memory out, and the
// GPU is a few flips behind at most; memory a Map filled once - a mesh
// streamed in - is read in place from then on.
// BBHOST_SHADOW_LITE_FLIPS=<n> sets the window (a test aid).
const std::uint32_t kLiteFlips = [] {
    const char* e = std::getenv("BBHOST_SHADOW_LITE_FLIPS");
    const long v = e && *e ? std::strtol(e, nullptr, 10) : 0;
    return v > 0 ? static_cast<std::uint32_t>(v) : 120u;
}();
std::uint64_t g_lite_asked = 0, g_lite_rewritten = 0;
thread_local std::uint64_t t_lite_drained = 0;  // the draw (t_lite_draw) the notices were last drained for
bool lite_rewritten_locked(std::uint64_t va, std::uint64_t bytes) {
    static bool said = false;
    if (!said) {
        said = true;
        host_log("buffer shadow: lite - no mirrors on this integrated GPU; vertex data in memory a Map handed out within the last %u "
                 "flips is copied as its draws record (BBHOST_SHADOW_LITE=0: the mirrors)",
                 kLiteFlips);
    }
    // BBHOST_RUN_BATCH: once a draw - its vertex records ask one after another.
    // A run part's draw (BBHOST_RUN_COMMIT): drained at the fork.
    const bool part = run_part();
    const std::uint64_t lite_draw = list_resolve::batch_shares() ? t_lite_draw : 0;  // (no thread-local read with batching off)
    if (!part && (!lite_draw || t_lite_drained != lite_draw)) {
        drain_gx_writes_locked();
        if (lite_draw) {
            t_lite_drained = lite_draw;
            list_resolve::count(list_resolve::kBatchDrains);
        }
    }
    if (g_handed_flip.empty()) return false;
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off)) return false;
    if (!part) ++g_lite_asked;
    const std::uint64_t phys = (std::uint64_t{chunk} << gcn::kDmemChunkShift) | off;
    const std::uint32_t now = static_cast<std::uint32_t>(hle_video_flip_count()) + 1;
    for (std::uint64_t p = phys >> kPageShift; p <= (phys + bytes - 1) >> kPageShift && p < g_handed_flip.size(); ++p) {
        const std::uint32_t at = g_handed_flip[p];
        if (at && now - at <= kLiteFlips) {
            if (!part) ++g_lite_rewritten;
            return true;
        }
    }
    return false;
}
}  // namespace

void shadow_lite_drain_locked() { drain_gx_writes_locked(); }
VkBuffer shadow_snap_buffer() { return g_snap.buffer; }

bool shadow_snapshot_maybe_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_snapshot_on || !bytes) return false;
    if (shadow_lite_on()) return !twin_covers_locked(va, bytes) && lite_rewritten_locked(va, bytes);
    if (!buffer_shadow_on() || twin_covers_locked(va, bytes)) return false;
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off) || off + bytes > kChunkBytes) return true;  // not told this cheaply
    const Mirror& m = g_mirrors[chunk];
    return !m.pages.empty() && any_set(m.dyn, static_cast<std::uint32_t>(off >> kPageShift), static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift));
}

bool shadow_snapshot_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    if (!g_snapshot_on || !bytes || bytes > kSnapMaxBytes || twin_covers_locked(va, bytes)) return false;
    if (shadow_lite_on()) {
        if (!lite_rewritten_locked(va, bytes)) return false;
    } else {
        if (!buffer_shadow_on()) return false;
        // Only a range on a page the game keeps rewriting: a static page the
        // mirrors had no room for is staged as before, not copied by the CPU
        // at every draw.
        std::uint32_t chunk = 0;
        std::uint64_t off = 0;
        if (!phys_of(va, bytes, chunk, off)) return false;
        const Mirror& m = g_mirrors[chunk];
        if (m.pages.empty() ||
            !any_set(m.dyn, static_cast<std::uint32_t>(off >> kPageShift), static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift)))
            return false;
    }
    if (gpu_writes_pending_locked(va, bytes)) {
        bump(g_snapshot_pending);
        return false;
    }
    if (!g_snap.buffer && !g_snap.failed && run_part_stop(kRunStopRing)) return false;  // the ring made in order (BBHOST_RUN_COMMIT)
    if (g_snap.failed || (!g_snap.buffer && !snap_ring_make())) {
        g_snap.failed = true;
        return false;
    }
    return snap_copy_locked(va, bytes, loc);
}

namespace {
// [va, va+bytes) copied into the snapshot ring now, `loc` the copy (the ring
// made already).
std::uint64_t g_snap_full = 0;  // copies refused: the ring's next segment held the open recording's copies
// The ring's next `bytes` for the open recording, round the ring when it runs
// out; false when that would write over what the open recording read.
bool snap_ring_take_locked(VkDeviceSize bytes, VkDeviceSize& at_out) {
    // A run part's draw (BBHOST_RUN_COMMIT): under the parts' lock, inside the
    // lap's segments (g_part_ring_mu).
    const bool part = run_part();
    std::unique_lock<HostLock> lk(g_part_ring_mu, std::defer_lock);
    if (part) lk.lock();
    VkDeviceSize next = g_snap.next;
    std::uint64_t lap = g_snap.lap;
    if (next + bytes + 256 > kSnapRing) {
        if (run_part_stop(kRunStopRing)) return false;
        next = 0;
        ++lap;
    }
    const VkDeviceSize at = (next + 255) & ~VkDeviceSize{255};
    if (part) {
        // A segment new to this lap whose last recording has run needs no wait
        // (wait_submission_locked's own test); the open recording's own is
        // refused below as in order; one still running is waited for in order.
        for (VkDeviceSize s = at / kSnapSegment; s <= (at + bytes - 1) / kSnapSegment; ++s) {
            const std::uint64_t ser = g_snap.segment_serial[s];
            if (g_snap.segment_lap[s] != lap && g_snap.segment_lap[s] && ser < g.flushes && g.completed_submits <= ser) {
                run_part_stop(kRunStopRing);
                return false;
            }
        }
    }
    // A segment the open recording's own draws read in the lap before cannot
    // be written until that recording has run, which it has not: one
    // recording that copies a whole ring's worth - every mesh of an area
    // copied as it records, its import failed - would write over the bytes
    // its earlier draws read. Such a copy is refused.
    for (VkDeviceSize s = at / kSnapSegment; s <= (at + bytes - 1) / kSnapSegment; ++s) {
        if (g_snap.segment_lap[s] && g_snap.segment_lap[s] != lap && g_snap.segment_serial[s] >= g.flushes) {
            if (!part) ++g_snap_full;
            return false;
        }
    }
    g_snap.next = next;
    g_snap.lap = lap;
    // A lap round the ring: the draws that read a segment in the lap before
    // have run before it is written again (nearly always long since).
    for (VkDeviceSize s = at / kSnapSegment; s <= (at + bytes - 1) / kSnapSegment; ++s) {
        if (g_snap.segment_lap[s] != g_snap.lap) {
            if (g_snap.segment_lap[s]) wait_submission_locked(g_snap.segment_serial[s]);
            g_snap.segment_lap[s] = g_snap.lap;
        }
        g_snap.segment_serial[s] = g.flushes;
    }
    g_snap.next = at + bytes;
    at_out = at;
    return true;
}
bool snap_copy_locked(std::uint64_t va, std::uint64_t bytes, Located& loc) {
    // A ring in host memory stages each copy into device memory (below):
    // in order, for a run part's draw (BBHOST_RUN_COMMIT).
    if (!g_snap.device_local && run_part_stop(kRunStopBuffers)) return false;
    VkDeviceSize at = 0;
    if (DrawCtx* const c = draw_ctx()) {
        // BBHOST_DRAW_CONTEXT: from the context's chunk of the ring, taken
        // as a copy of the chunk's size would be (refused the same way).
        constexpr VkDeviceSize kSnapCtxChunk = 256u << 10;
        c->take_recording(g.flushes);
        at = (c->snap_next + 255) & ~VkDeviceSize{255};
        if (!c->snap_end || at + bytes > c->snap_end) {
            const VkDeviceSize n = std::max<VkDeviceSize>(kSnapCtxChunk, bytes);
            if (!snap_ring_take_locked(n, at)) return false;
            c->snap_end = at + n;
            list_resolve::count(list_resolve::kCtxSnapChunks);
            if (list_resolve::draw_ctx_compare()) list_resolve::check_chunk(2, g.flushes, at, at + n, 0, kSnapRing);
        }
        c->snap_next = at + bytes;
        list_resolve::count(list_resolve::kCtxSnapBytes, bytes);
    } else if (!snap_ring_take_locked(bytes, at)) {
        return false;
    }
    std::memcpy(g_snap.map + at, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), bytes);
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

std::atomic<std::uint64_t> g_fallback[import_fallback::kChoices], g_fallback_bytes{0};
std::atomic<std::uint64_t> g_fallback_mirror_why[kLocWhys];  // why the mirrors did not serve a range in no import
}  // namespace

int shadow_import_fallback_locked(std::uint64_t va, std::uint64_t bytes, std::uint64_t read_bytes, Located& out) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_IMPORT_COPY");
        return !(e && e[0] == '0');
    }();
    bool mirror = false;
    if (on && buffer_shadow_on() && g.sink.buffer && bytes && !twin_covers_locked(va, bytes)) {
        // The mirrors, their pages copied through staging (no import needed).
        Located probe{g.sink.buffer, 0, bytes};
        t_force_staged = true;
        shadow_locate_locked(va, bytes, probe);
        t_force_staged = false;
        if (probe.buffer != g.sink.buffer) {
            out = probe;
            mirror = true;
        } else {
            g_fallback_mirror_why[t_locate_why < kLocWhys ? t_locate_why : 0].fetch_add(1, std::memory_order_relaxed);
        }
    }
    const std::uint64_t n = read_bytes ? std::min(read_bytes, bytes ? bytes : read_bytes) : bytes;
    const bool pending = on && !mirror && n && gpu_writes_pending_locked(va, n);
    const bool need_ring = on && !mirror && !pending && n && n <= kSnapMaxBytes;
    const bool ring_ok = need_ring && !g_snap.failed && (g_snap.buffer || snap_ring_make());
    if (need_ring && !ring_ok) g_snap.failed = true;
    import_fallback::Choice c = import_fallback::choose(on, mirror, n, kSnapMaxBytes, pending, ring_ok);
    if (c == import_fallback::kCopy) {
        if (snap_copy_locked(va, n, out))
            bump(g_fallback_bytes, n);
        else
            c = import_fallback::kNoRing;  // full with the open recording's copies (g_snap_full)
    }
    bump(g_fallback[c]);
    return c;
}

std::string shadow_import_fallback_report() {
    using import_fallback::Choice;
    std::uint64_t any = 0;
    for (const auto& v : g_fallback) any += v.load();
    if (!any) return {};
    static const char* const kWhy[kLocWhys] = {"not asked", "a twin", "no mirror", "in place", "across windows", "served", "host memory",
                                               "no room", "no window", "no import", "a GPU write pending", "no copy"};
    std::string why;
    for (int k = 0; k < kLocWhys; ++k) {
        if (const std::uint64_t n = g_fallback_mirror_why[k].load()) why += std::string(why.empty() ? "" : ", ") + kWhy[k] + " " + std::to_string(n);
    }
    char b[480];
    std::snprintf(b, sizeof(b),
                  "; draws' ranges in no import (the mirrors refused: %s): served by the mirrors through staging %llu, copied as recorded %llu (%.1f MiB), not copied: "
                  "BBHOST_IMPORT_COPY=0 %llu, past the ring's 1 MiB %llu, a GPU write still to land %llu, no ring or the ring full %llu",
                  why.empty() ? "none" : why.c_str(), static_cast<unsigned long long>(g_fallback[import_fallback::kMirror].load()),
                  static_cast<unsigned long long>(g_fallback[import_fallback::kCopy].load()), g_fallback_bytes.load() / 1048576.0,
                  static_cast<unsigned long long>(g_fallback[import_fallback::kOff].load()),
                  static_cast<unsigned long long>(g_fallback[import_fallback::kTooBig].load()),
                  static_cast<unsigned long long>(g_fallback[import_fallback::kGpuPending].load()),
                  static_cast<unsigned long long>(g_fallback[import_fallback::kNoRing].load()));
    return b;
}

int shadow_place_class_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!buffer_shadow_on() || !bytes) return kPlaceNone;
    std::uint32_t chunk = 0;
    std::uint64_t off = 0;
    if (!phys_of(va, bytes, chunk, off)) return kPlaceNone;
    const Mirror& m = g_mirrors[chunk];
    const std::uint32_t p0 = static_cast<std::uint32_t>(off >> kPageShift), p1 = static_cast<std::uint32_t>((off + bytes - 1) >> kPageShift);
    if (m.pages.empty() || p1 >= m.pages.size()) return kPlaceNone;
    // The least placeable page decides; a long range by its first 256 pages.
    int cls = kPlacePlaced;
    for (std::uint32_t p = p0; p <= p1 && p < p0 + 256; ++p) {
        const Page& pg = m.pages[p];
        const int c = pg.dynamic               ? kPlaceDynamic
                      : page_handed(chunk, p)  ? kPlaceHanded
                      : pg.placed              ? kPlacePlaced
                      : pg.valid && pg.told    ? kPlaceTold
                      : pg.valid               ? kPlaceWatched
                                               : kPlaceUntracked;
        cls = std::max(cls, c);
    }
    return cls;
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
        const bool w = pg.valid && (pg.told ? pg.gx_written > pg.told_seq : written(pg));
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

// BBHOST_GPU_WRITE_FILTER (on unless 0): the region filter below.
const bool g_gpu_write_filter = [] {
    const char* e = std::getenv("BBHOST_GPU_WRITE_FILTER");
    return !(e && e[0] == '0');
}();

// Page number -> the newest recording a GPU write to it went into. Rebuilt
// with only the writes still to land when it is half full. In front of the
// table, a filter of the 2 MiB regions its writes lie in: a bit of 4,096 (a
// hash of the region, 512 bytes that stay in the L1 cache) set as a write is
// noted and set again from the kept ones at a rebuild, so a clear bit says no
// write in the table lies there and the table is not probed. Constant buffers
// lie where GPU writes do not (GX's upload heap) and are asked about at every
// binding - ~10,400 a frame at the Nightmare of Mensis on a Steam Deck, where
// the probes were among the command processor's cache misses.
class PageSerials {
public:
    explicit PageSerials(unsigned region_shift) : region_shift_(region_shift) {}
    void note(std::uint64_t page, std::uint64_t serial) {
        if ((used_ + 1) * 2 > slots_.size()) rebuild();
        Slot& s = slots_[find(page + 1)];
        if (!s.key) {
            s.key = page + 1;  // 0: an empty slot
            ++used_;
        }
        if (serial > s.serial) s.serial = serial;
        mark(page);
    }
    bool pending(std::uint64_t page) const {
        if (!used_) return false;
        if (g_gpu_write_filter && !maybe(page)) return false;
        const Slot& s = slots_[find(page + 1)];
        return s.key && s.serial >= g.completed_submits;
    }
    // The latest recording a GPU write over the page was made in (0: none).
    std::uint64_t latest(std::uint64_t page) const {
        if (!used_) return 0;
        const Slot& s = slots_[find(page + 1)];
        return s.key ? s.serial : 0;
    }
    // The newest serial a rebuild let go of (landed writes leave the table):
    // latest() cannot answer for a recording at or before it.
    std::uint64_t dropped() const { return dropped_; }

private:
    struct Slot {
        std::uint64_t key = 0, serial = 0;
    };
    unsigned filter_bit(std::uint64_t page) const {
        return static_cast<unsigned>(((page >> region_shift_) * 0x9e3779b97f4a7c15ull) >> 52);
    }
    void mark(std::uint64_t page) {
        const unsigned b = filter_bit(page);
        filter_[b >> 6] |= std::uint64_t{1} << (b & 63);
    }
    bool maybe(std::uint64_t page) const {
        const unsigned b = filter_bit(page);
        return (filter_[b >> 6] >> (b & 63)) & 1;
    }
    std::uint64_t filter_[64] = {};
    unsigned region_shift_;
    std::vector<Slot> slots_;
    std::size_t used_ = 0;
    std::uint64_t dropped_ = 0;
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
        std::fill(std::begin(filter_), std::end(filter_), std::uint64_t{0});
        for (const Slot& s : old) {
            if (!s.key) continue;
            if (s.serial < g.completed_submits) {
                dropped_ = std::max(dropped_, s.serial);
                continue;
            }
            slots_[find(s.key)] = s;
            ++used_;
            mark(s.key - 1);
        }
    }
};
// (The region filter by 2 MiB: 64 KiB pages 5 bits under it, 4 KiB pages 9.)
PageSerials g_gpu_write_pages{21 - kPageShift}, g_gpu_write_small{21 - kSmallShift};
}  // namespace

void gpu_writes_note_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!bytes) return;
    gxvk::heap_written(gxvk::kHeapGuardGpuWrite, va, bytes);  // GX's upload heap holds none (stage 1d.1 rests on it)
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

std::uint64_t gpu_writes_dropped_locked() { return std::max(g_gpu_write_pages.dropped(), g_gpu_write_small.dropped()); }

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
    drain_gx_writes_locked();
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
            if (pg.told ? pg.gx_written > pg.told_seq : written(pg)) continue;  // a write the next check will see anyway
            if (pg.told && pg.provisional) continue;  // copied again at the next recording, outstanding renamed copies over it
            bool read_changed = !pg.told;  // a watched page's change is one either way
            if (pg.told) {
                ++g_verify_stale_told;
                // Of the told page's bytes, did a draw bind any that changed?
                auto vc = g_verify_copies.find((std::uint64_t{chunk} << 32) | p);
                if (vc != g_verify_copies.end() && vc->second.size() == pg.bytes && pg.read_lo < pg.read_hi) {
                    const auto* now = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(pg.va));
                    const std::uint32_t lo = std::min<std::uint32_t>(pg.read_lo, static_cast<std::uint32_t>(pg.bytes));
                    const std::uint32_t hi = std::min<std::uint32_t>(pg.read_hi, static_cast<std::uint32_t>(pg.bytes));
                    read_changed = lo < hi && std::memcmp(now + lo, vc->second.data() + lo, hi - lo) != 0;
                }
                if (read_changed) ++g_verify_stale_told_read;
            }
            if (pg.told && !read_changed) continue;  // changed in bytes no draw bound since the copy (the pool's draw tables)
            if (g_verify_stale++ < 32) {
                host_log("buffer shadow: VERIFY: %s page 0x%llx changed with no write this cache saw (chunk %u)%s", pg.told ? "told" : "watched",
                         static_cast<unsigned long long>(page_va), chunk, pg.told ? ", in bytes draws bound" : "");
                if (pg.told)
                    host_log("buffer shadow: VERIFY:   draws bound [+0x%x, +0x%x) of it since its copy", pg.read_lo, pg.read_hi);
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

VkCommandBuffer shadow_upload_cmd_locked() { return upload_cmd_locked(); }

// The copies this recording made, ended; submit_locked runs them first.
VkCommandBuffer shadow_end_uploads_locked() {
    // The staged pages' copies this recording handed to the copy threads are
    // in staging before its submission reads them.
    if (StagedCopies::get().on()) StagedCopies::get().finish();
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
    g_cushion = std::clamp<std::uint64_t>((g.local_budget >> 20) / 24, 128, 256) << 20;
    if (const char* c = std::getenv("BBHOST_SHADOW_CUSHION_MB"); c && *c) g_cushion = std::strtoull(c, nullptr, 10) << 20;
    g_mirror_cap = e && *e ? static_cast<std::uint64_t>(std::strtoull(e, nullptr, 10)) << 20
                   : g_live ? ~0ull
                   : g.local_budget > kImageRoom ? g.local_budget - kImageRoom : 0;
    g_initial_cap = g_mirror_cap;
    if (!buffer_shadow_on()) return;
    if (g_live) {
        char how[64];
        std::snprintf(how, sizeof(how), "%llu MiB windows", static_cast<unsigned long long>(window_bytes() >> 20));
        host_log("buffer shadow: its mirrors may hold what video memory has free above the reserve and a %llu MiB cushion (%s)",
                 static_cast<unsigned long long>(g_cushion >> 20),
                 g.has_sparse ? "64 MiB blocks" : g_windows_on ? how : "whole 1 GiB mirrors: no sparse binding");
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
        if (!m.buffer && !m.windowed) continue;
        if (m.buffer) slot.garbage.push_back(DevBuffer{m.memory, m.buffer});
        for (VkDeviceMemory b : m.blocks) slot.garbage.push_back(DevBuffer{b, VK_NULL_HANDLE});
        for (const Mirror::Window& w : m.windows) {
            if (w.buffer) slot.garbage.push_back(DevBuffer{w.memory, w.buffer});
        }
        m = Mirror{};
        m.failed = true;
        ++mirrors;
    }
    if (g_spare_block != VK_NULL_HANDLE) {
        slot.garbage.push_back(DevBuffer{g_spare_block, VK_NULL_HANDLE});
        g_spare_block = VK_NULL_HANDLE;
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
        if (!m.buffer && !m.windowed) {
            if (!m.failed) m = Mirror{};
            continue;
        }
        if (m.buffer) slot.garbage.push_back(DevBuffer{m.memory, m.buffer});
        for (VkDeviceMemory b : m.blocks) slot.garbage.push_back(DevBuffer{b, VK_NULL_HANDLE});
        for (const Mirror::Window& w : m.windows) {
            if (w.buffer) slot.garbage.push_back(DevBuffer{w.memory, w.buffer});
        }
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
    for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) {
        Mirror& m = g_mirrors[c];
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

std::uint64_t shadow_room_refusals_locked() { return g_room_refusals + g_room_skips; }

void shadow_vram_census_locked(ShadowCensus& c) {
    c = ShadowCensus{};
    c.live = g_live;
    c.sparse = g.has_sparse;
    c.cap = g_live ? 0 : g_mirror_cap;
    c.given_back = g_given_back;
    for (const Mirror& m : g_mirrors) {
        if (!m.buffer && !m.windowed) continue;
        std::uint64_t valid = 0;
        // The 64 MiB windows of the chunk that hold a copied page: what a
        // mirror made window by window (no sparse binding) would hold.
        std::uint32_t windows = 0;
        for (std::uint32_t w = 0; w * kBlockPages < m.pages.size(); ++w) {
            bool any = false;
            for (std::uint32_t p = w * kBlockPages; p < (w + 1) * kBlockPages && p < m.pages.size(); ++p) {
                const bool held = m.pages[p].valid || (m.sparse && (m.resident[p >> 6] >> (p & 63)) & 1);
                any |= held;
                valid += m.pages[p].valid && !m.pages[p].dynamic;
            }
            windows += any;
        }
        c.windows += windows;
        if (m.sparse) {
            c.block_bytes += static_cast<std::uint64_t>(m.blocks.size()) * kBlockPages * kPageBytes;
            c.blocks += static_cast<std::uint32_t>(m.blocks.size());
        } else if (m.windowed) {
            for (const Mirror::Window& w : m.windows) {
                if (!w.buffer) continue;
                c.block_bytes += window_bytes();  // a window is a block of its own
                ++c.blocks;
            }
        } else if (m.memory) {
            c.whole_bytes += kChunkBytes;
            ++c.whole;
        }
        c.data_bytes += valid * kPageBytes;
    }
}

void shadow_keeper_started() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_SHADOW_ASYNC");
        return !(e && e[0] == '0');
    }();
    std::lock_guard<GpuMutex> lk(g.mu);
    g_async = on;
}

namespace {
// A buffer of `bytes` in device-local memory, bound: what mirror_for makes for
// a whole mirror and window_ready for a window, made here off the lock.
bool keeper_make(VkDeviceSize bytes, const char* owner, VkBuffer& buf, VkDeviceMemory& mem, VkDeviceSize& size) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    buf = VK_NULL_HANDLE;
    mem = VK_NULL_HANDLE;
    if (vkCreateBuffer(g.device, &bci, nullptr, &buf) != VK_SUCCESS) {
        buf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, buf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    MemOwner own_mem(owner);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS ||
        vkBindBufferMemory(g.device, buf, mem, 0) != VK_SUCCESS) {
        if (mem) vkFreeMemory(g.device, mem, nullptr);
        vkDestroyBuffer(g.device, buf, nullptr);
        buf = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
        return false;
    }
    size = req.size;
    return true;
}
}  // namespace

void shadow_keeper_work() {
    std::vector<std::uint32_t> whole;                         // chunks whose whole mirror is wanted
    std::vector<std::pair<std::uint32_t, std::uint32_t>> wins;  // (chunk, window) wanted
    bool spare = false;
    std::uint32_t spare_type = UINT32_MAX;
    const VkDeviceSize block = static_cast<VkDeviceSize>(kBlockPages) * kPageBytes;
    {
        std::lock_guard<GpuMutex> lk(g.mu);
        if (!g_async) return;
        for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) {
            const Mirror& m = g_mirrors[c];
            if (m.making && !m.buffer) whole.push_back(c);
            for (std::uint32_t w = 0; m.windowed && w < m.windows.size(); ++w) {
                if (m.windows[w].making && !m.windows[w].buffer) wins.emplace_back(c, w);
            }
        }
        // A spare block while the sparse mirrors use blocks and the share
        // has room for it and the next one.
        spare = g.has_sparse && g_spare_block == VK_NULL_HANDLE && g_block_type != UINT32_MAX && !g_given_back &&
                g_mirror_bytes + 2 * block <= g_mirror_cap && (!g_live || vram_room_for(2 * block, g_cushion));
        spare_type = g_block_type;
    }
    for (const std::uint32_t c : whole) {
        const auto t0 = std::chrono::steady_clock::now();
        VkBuffer buf;
        VkDeviceMemory mem;
        VkDeviceSize size = 0;
        const bool made = keeper_make(kChunkBytes, "mirrors", buf, mem, size);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        bool placed = false;
        {
            std::lock_guard<GpuMutex> lk(g.mu);
            Mirror& m = g_mirrors[c];
            if (made && m.making && !m.buffer && !m.windowed && !m.failed && !g_given_back && g_mirror_bytes + size <= g_mirror_cap) {
                // Still wanted: in place for the next draw that reads the chunk.
                m.buffer = buf;
                m.memory = mem;
                mirror_pages(m);
                g_mirror_bytes += size;
                ++g_mirrors_async;
                g_mirror_async_ms += ms;
                placed = true;
            }
            if (m.making) {
                m.making = false;
                if (!made) m.failed = true;
            }
        }
        if (placed) {
            host_log("buffer shadow: device-local mirror for chunk %u (%llu MiB), made in %.1f ms off the command processor", c,
                     static_cast<unsigned long long>(kChunkBytes >> 20), ms);
        } else if (made) {
            // Not wanted any more: the mirrors started over or were given back.
            vkDestroyBuffer(g.device, buf, nullptr);
            vkFreeMemory(g.device, mem, nullptr);
        } else {
            host_log("buffer shadow: no device-local mirror for chunk %u; its vertex and index data stay in host memory", c);
        }
    }
    for (const auto& [c, w] : wins) {
        const auto t0 = std::chrono::steady_clock::now();
        VkBuffer buf;
        VkDeviceMemory mem;
        VkDeviceSize size = 0;
        const bool made = keeper_make(window_bytes(), "mirror windows", buf, mem, size);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        bool placed = false;
        {
            std::lock_guard<GpuMutex> lk(g.mu);
            Mirror& m = g_mirrors[c];
            Mirror::Window* win = m.windowed && w < m.windows.size() ? &m.windows[w] : nullptr;
            if (made && win && win->making && !win->buffer && !m.failed && !g_given_back && g_mirror_bytes + size <= g_mirror_cap) {
                win->buffer = buf;
                win->memory = mem;
                g_mirror_bytes += size;
                ++g_windows_made;
                ++g_windows_async;
                g_window_async_ms += ms;
                placed = true;
            }
            if (win && win->making) win->making = false;  // (not made: a later draw asks again)
        }
        if (!placed && made) {
            vkDestroyBuffer(g.device, buf, nullptr);
            vkFreeMemory(g.device, mem, nullptr);
        }
    }
    if (spare && spare_type != UINT32_MAX) {
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = block;
        mai.memoryTypeIndex = spare_type;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        MemOwner own_mem("mirror blocks");
        if (vkAllocateMemory(g.device, &mai, nullptr, &mem) == VK_SUCCESS) {
            bool kept = false;
            {
                std::lock_guard<GpuMutex> lk(g.mu);
                if (g_spare_block == VK_NULL_HANDLE && !g_given_back) {
                    g_spare_block = mem;
                    g_spare_type = spare_type;
                    kept = true;
                }
            }
            if (!kept) vkFreeMemory(g.device, mem, nullptr);
        }
    }
}

std::string shadow_report() {
    std::uint64_t resident = 0;
    for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) resident += g_mirrors[c].resident_pages;
    char buf[768];
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
    if (const std::uint64_t staged = g_staged_pages.load()) {
        char v[256];
        const StagedCopies& sc = StagedCopies::get();
        std::snprintf(v, sizeof(v),
                      "; %llu page copies through staging (window imports; %u copy threads, %llu copied by the command processor at a "
                      "submission, %llu submissions waited), %llu pages waiting for a GPU write",
                      static_cast<unsigned long long>(staged), sc.threads(), static_cast<unsigned long long>(sc.inline_runs()),
                      static_cast<unsigned long long>(sc.waits()), static_cast<unsigned long long>(g_staged_pending.load()));
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
    std::string in_place_line;
    if (in_place_on() || g_placed) {
        std::uint64_t placed_now = 0;
        // The pages the mirrors still hold copies of, by what covers them -
        // GX buffers whole (handed out or not: in a range with another kind
        // of page), GX resources in part, nothing GX made - each with the
        // windows they lie in: what keeps the windows on an integrated GPU.
        std::uint64_t held[4] = {}, wins[4] = {};
        for (std::uint32_t ch = 0; ch < gcn::kDmemChunks; ++ch) {
            const Mirror& m = g_mirrors[ch];
            for (std::uint32_t w = 0; w * window_pages() < m.pages.size(); ++w) {
                bool in[4] = {};
                for (std::uint32_t p = w * window_pages(); p < (w + 1) * window_pages() && p < m.pages.size(); ++p) {
                    const Page& pg = m.pages[p];
                    placed_now += pg.placed;
                    if (!pg.valid || pg.dynamic || !pg.bytes) continue;
                    const int k = page_handed(ch, p)                           ? 1
                                  : gx_buffers_cover(pg.va, pg.bytes)            ? 0
                                  : gx_resource_overlapping(pg.va, pg.bytes, nullptr) ? 2
                                                                                 : 3;
                    ++held[k];
                    in[k] = true;
                }
                for (int k = 0; k < 4; ++k) wins[k] += in[k];
            }
        }
        char ip[512];
        std::snprintf(ip, sizeof(ip),
                      "buffer shadow in place (BBHOST_SHADOW_IN_PLACE): ranges bound %llu, pages placed %llu (%llu MiB now), placed again after a "
                      "told write %llu, ranges with a page GX does not tell %llu; pages the mirrors hold copies of (MiB, in windows): GX "
                      "buffers' whole %llu in %llu, under renamed copies %llu in %llu, GX resources' in part %llu in %llu, nothing GX made "
                      "%llu in %llu",
                      static_cast<unsigned long long>(g_placed_binds.load()), static_cast<unsigned long long>(g_placed),
                      static_cast<unsigned long long>((placed_now * kPageBytes) >> 20), static_cast<unsigned long long>(g_placed_again),
                      static_cast<unsigned long long>(g_placed_refused), static_cast<unsigned long long>((held[0] * kPageBytes) >> 20),
                      static_cast<unsigned long long>(wins[0]), static_cast<unsigned long long>((held[1] * kPageBytes) >> 20),
                      static_cast<unsigned long long>(wins[1]), static_cast<unsigned long long>((held[2] * kPageBytes) >> 20),
                      static_cast<unsigned long long>(wins[2]), static_cast<unsigned long long>((held[3] * kPageBytes) >> 20),
                      static_cast<unsigned long long>(wins[3]));
        in_place_line = ip;
    }
    std::string out = std::string(buf) + h + rr;
    if (g_windows_made || g_window_crossings) {
        char wn[160];
        std::snprintf(wn, sizeof(wn), "; mirror windows of %llu MiB made %llu, ranges across two windows read from host memory %llu",
                      static_cast<unsigned long long>(window_bytes() >> 20), static_cast<unsigned long long>(g_windows_made),
                      static_cast<unsigned long long>(g_window_crossings));
        out += wn;
    }
    if (g_mirrors_async || g_windows_async || g_blocks_made || g_blocks_spare || g_async_waits) {
        char mk[320];
        std::snprintf(mk, sizeof(mk),
                      "; made off the command processor: mirrors %llu (%.0f ms there), windows %llu (%.0f ms), ranges read from host "
                      "memory meanwhile %llu; sparse blocks from the spare %llu, made on the command processor %llu (%.0f ms)",
                      static_cast<unsigned long long>(g_mirrors_async), g_mirror_async_ms, static_cast<unsigned long long>(g_windows_async),
                      g_window_async_ms, static_cast<unsigned long long>(g_async_waits), static_cast<unsigned long long>(g_blocks_spare),
                      static_cast<unsigned long long>(g_blocks_made), g_block_ms_total);
        out += mk;
    }
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
    char st[384];
    std::snprintf(st, sizeof(st),
                  "; vertex ranges copied for their draw %llu (%.1f MiB), read from another draw's copy %llu (%.1f MiB; %llu inside one "
                  "starting before them), left in host memory %llu, staging chunks %zu (%zu MiB)",
                  static_cast<unsigned long long>(g_staged.load()), static_cast<double>(g_staged_bytes.load()) / (1 << 20),
                  static_cast<unsigned long long>(g_stage_shared.load()), static_cast<double>(g_stage_shared_bytes.load()) / (1 << 20),
                  static_cast<unsigned long long>(g_stage_inside.load()),
                  static_cast<unsigned long long>(g_stage_refused.load()), g_stage_chunks,
                  static_cast<std::size_t>((g_stage_chunks * kStageChunkBytes) >> 20));
    char sn[256];
    std::snprintf(sn, sizeof(sn),
                  "; rewritten vertex ranges copied as recorded %llu (%.1f MiB, %llu staged from host memory), left for a GPU write "
                  "still to land %llu, refused with the ring full of the open recording's copies %llu",
                  static_cast<unsigned long long>(g_snapshots.load()), static_cast<double>(g_snapshot_bytes.load()) / (1 << 20),
                  static_cast<unsigned long long>(g_snapshot_staged.load()), static_cast<unsigned long long>(g_snapshot_pending.load()),
                  static_cast<unsigned long long>(g_snap_full));
    out += st;
    out += sn;
    if (g_stage_census && !g_stage_stats.empty()) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> top;  // bytes, va
        for (const auto& [va, stt] : g_stage_stats) top.push_back({stt.bytes, va});
        std::sort(top.rbegin(), top.rend());
        static const char* const kWhy[4] = {"dynamic", "GPU-written", "not mirrored", "mirrored?"};
        host_log("buffer shadow: stage census since the last report: %zu ranges; MiB by why: dynamic %.0f, GPU-written %.0f, not mirrored %.0f, "
                 "mirrored? %.0f",
                 g_stage_stats.size(), g_stage_why_bytes[0] / 1048576.0, g_stage_why_bytes[1] / 1048576.0, g_stage_why_bytes[2] / 1048576.0,
                 g_stage_why_bytes[3] / 1048576.0);
        for (std::size_t i = 0; i < top.size() && i < 12; ++i) {
            const StageStat& stt = g_stage_stats[top[i].second];
            GxResourceInfo info;
            const bool res = gx_resource_at(top[i].second, 1, &info);
            host_log("buffer shadow: stage census:   0x%llx %llu B x%llu = %.0f MiB (%s)%s%llx +%u bind 0x%x usage %u", static_cast<unsigned long long>(top[i].second),
                     static_cast<unsigned long long>(stt.record), static_cast<unsigned long long>(stt.copies), stt.bytes / 1048576.0,
                     kWhy[stt.why < 0 ? 3 : stt.why], res ? (info.type == 1 ? " in GX buffer 0x" : " in GX texture 0x") : " in no GX resource 0x",
                     static_cast<unsigned long long>(res ? info.memory : 0), res ? info.bytes : 0, res ? info.bind : 0, res ? info.usage : 0);
        }
        g_stage_stats.clear();
        std::fill(std::begin(g_stage_why_bytes), std::end(g_stage_why_bytes), 0);
    }
    // What GX's calls told the mirrors, on a line of its own (the report
    // fills one).
    if (g_shadow_gx) {
        char gx[900];
        std::snprintf(gx, sizeof(gx),
                      "told by GX's calls (BBHOST_SHADOW_GX): renamed copies%s outstanding now %zu (most %llu), written back %llu (no "
                      "outstanding copy there %llu), aged out %llu; told copies made while one was outstanding %llu, valid pages a hand-out "
                      "invalidated %llu, pages that stayed unsettled to host memory %llu; resources made over handed-out memory %llu; "
                      "%llu page copies told and not watched, watched instead %llu (GX buffers cover them in part) + %llu (none), GX writes "
                      "%llu, told pages invalidated by them %llu (kept for a submission made before the write %llu)%s",
                      g_shadow_handed ? "" : " (BBHOST_SHADOW_HANDED=0: watched)", g_outstanding.size(),
                      static_cast<unsigned long long>(g_outstanding_most), static_cast<unsigned long long>(g_settled),
                      static_cast<unsigned long long>(g_settle_unmatched), static_cast<unsigned long long>(g_aged_out),
                      static_cast<unsigned long long>(g_provisional_copies), static_cast<unsigned long long>(g_handed_invalid),
                      static_cast<unsigned long long>(g_handed_dynamic), static_cast<unsigned long long>(g_made_over_handed),
                      static_cast<unsigned long long>(g_told_copies), static_cast<unsigned long long>(g_watched_partial),
                      static_cast<unsigned long long>(g_watched_none), static_cast<unsigned long long>(g_gx_records.load()),
                      static_cast<unsigned long long>(g_told_invalid), static_cast<unsigned long long>(g_told_held),
                      g_told_overflows || g_told_unmapped ? "; every told page copied again after a queue overflow or an unmapped range" : "");
        std::string told = gx;
        if (const std::uint64_t dropped = g_hand_dropped.load()) {
            char hd[96];
            std::snprintf(hd, sizeof(hd), "; hand-outs the full queue dropped %llu", static_cast<unsigned long long>(dropped));
            told += hd;
        }
        if (g_verify) {
            char vt[96];
            std::snprintf(vt, sizeof(vt), "; stale told pages %llu (in bytes draws bound %llu)", static_cast<unsigned long long>(g_verify_stale_told),
                          static_cast<unsigned long long>(g_verify_stale_told_read));
            told += vt;
        }
        host_log("  buffer shadow, %s", told.c_str());
    }
    if (g_map_census) {
        drain_gx_writes_locked();
        census_faults_locked();
        log_fault_census();
        host_log("  buffer shadow census: pages the write watch found written: a GX write explains %llu, on a GX resource it does not %llu, on "
                 "none %llu",
                 static_cast<unsigned long long>(g_written_by_gx.load()), static_cast<unsigned long long>(g_written_not_gx.load()),
                 static_cast<unsigned long long>(g_written_no_resource.load()));
    }
    if (!in_place_line.empty()) out += "\n" + in_place_line;  // a line of its own (gnm.cpp logs a line a call)
    if (shadow_lite_on()) {
        char lite[192];
        std::snprintf(lite, sizeof(lite), "buffer shadow lite: vertex ranges asked %llu, in memory a Map handed out in the last %u flips %llu",
                      static_cast<unsigned long long>(g_lite_asked), kLiteFlips, static_cast<unsigned long long>(g_lite_rewritten));
        out += "\n";
        out += lite;
    }
    return out;
}

}  // namespace gpu

std::string host_gpu_shadow_report() {
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    return gpu::shadow_report();
}

void host_gpu_shadow_gx_write(std::uint64_t va, std::size_t bytes) {
    using namespace gpu;
    if (!g_shadow_gx || !va || !bytes) return;
    g_gx_records.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_gx_writes_mu);
    if (g_gx_writes.size() >= kGxWritesMax) {
        g_gx_writes_overflow.store(true, std::memory_order_release);
    } else {
        g_gx_writes.push_back({va, bytes, hle_gnm_submitted_total(), kGxWrite});
    }
    g_gx_writes_pending.store(true, std::memory_order_release);
}

void host_gpu_shadow_gx_created(std::uint64_t va, std::size_t bytes) {
    using namespace gpu;
    if (!(g_shadow_gx || g_map_census) || !va || !bytes) return;
    if (g_shadow_gx) g_gx_records.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_gx_writes_mu);
    if (g_gx_writes.size() >= kGxWritesMax) {
        g_gx_writes_overflow.store(true, std::memory_order_release);
    } else {
        g_gx_writes.push_back({va, bytes, hle_gnm_submitted_total(), kGxCreate});
    }
    g_gx_writes_pending.store(true, std::memory_order_release);
}

bool host_gpu_shadow_hands_out() { return gpu::g_shadow_handed; }

void host_gpu_shadow_gx_handed(std::uint64_t va, std::size_t bytes, int kind) {
    using namespace gpu;
    if (!(g_shadow_gx || g_map_census) || !va || !bytes) return;
    const unsigned k = static_cast<unsigned>(kind);
    if (k >= kHandKinds) return;
    // The census's hand-outs come in runs through the same pages (a ring, a
    // pool's blocks): one record per span of 4 KiB pages a thread repeats.
    // A renamed copy's is never skipped - the same memory handed out again
    // after a write-back is outstanding again.
    thread_local std::uint64_t last_lo[kHandKinds] = {~0ull, ~0ull, ~0ull, ~0ull}, last_hi[kHandKinds] = {};
    const std::uint64_t lo = va & ~0xfffull, hi = (va + bytes + 0xfff) & ~0xfffull;
    if (k != kHandMap && lo >= last_lo[k] && hi <= last_hi[k]) return;
    std::lock_guard<std::mutex> lk(g_gx_writes_mu);
    if (g_gx_writes.size() >= kGxWritesMax) {
        // No room: told pages under it may be written unseen, so every told
        // page is copied again before it is read (and the pages go to host
        // memory at a later hand-out of the same memory).
        g_hand_dropped.fetch_add(1, std::memory_order_relaxed);
        if (g_shadow_handed) g_gx_writes_overflow.store(true, std::memory_order_release);
    } else {
        g_gx_writes.push_back({va, bytes, hle_gnm_submitted_total(), static_cast<int>(k) + 1});
        last_lo[k] = lo;
        last_hi[k] = hi;
    }
    g_gx_writes_pending.store(true, std::memory_order_release);
}

void host_gpu_shadow_gx_settled(std::uint64_t va, std::size_t bytes) {
    using namespace gpu;
    if (!g_shadow_handed || !va || !bytes) return;
    std::lock_guard<std::mutex> lk(g_gx_writes_mu);
    if (g_gx_writes.size() >= kGxWritesMax) {
        // No room: the copy stays outstanding until it ages out - its pages
        // are copied again at each recording until then, which is safe.
        g_hand_dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_gx_writes.push_back({va, bytes, hle_gnm_submitted_total(), kGxSettle});
    g_gx_writes_pending.store(true, std::memory_order_release);
}

void host_gpu_shadow_cp_write(std::uint64_t va, std::size_t bytes) {
    using namespace gpu;
    std::lock_guard<GpuMutex> lock(g.mu);
    shadow_written_locked(va, bytes, false);
}
