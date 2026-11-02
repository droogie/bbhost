// GX on Vulkan, stage 1c: GX's upload heap bound in place (host/gxvk.h).
//
// The heap is the one region GX's device gives its block pool, which every
// renamed constant buffer, Map rename, ring allocation and draw table comes
// from (decomp/gx/upload_heap.cpp tells us of it). The pool hands out from
// both ends - the deferred contexts from the heap's start, the immediate
// context from its top - and both ends' hand-outs are records here
// (engine/gx_resources.cpp's pool hooks). GX hands a range of it out
// again only after an end-of-pipe label behind the work that used it has
// landed, and a label lands only once the host's submission has completed;
// so a range the command processor binds holds the same bytes until the GPU
// is done with it, and can be bound where it is - no snapshot, no lookup.
//
// That rests on the game's code and ours, so it is proven again every run
// while it is used (the owner's condition for constant buffers in place):
// - the release order, exactly: each hand-out is a record; a binding stamps
//   the record it lies in with the submission it is recorded into; a
//   hand-out over a record's memory needs that record's stamp completed;
// - the bytes, sampled: one in-place binding in kSampleEvery has its bytes
//   hashed as it is bound and again once its submission has completed, and
//   they must not have changed while its range was not handed out again.
// A violation is counted, said once, and every binding of the heap takes
// today's path for the rest of the run. The arena then stops its own work
// too: its records are dropped, and it records no more hand-outs and looks up
// no more bindings, so the run costs what today's path costs.
//
// BBHOST_GXVK_ARENA=1 binds in place; =compare keeps today's path and checks
// every binding of the heap as 1 would bind it (the records and the bytes,
// every one). Off (0, the default) nothing here runs.
#include "host/gxvk.h"

#include "host/gpu.h"
#include "host/label_landing.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// tests/gxvk_arena_test.cpp builds this file with mutants: small wrong turns
// it switches on one at a time and must catch.
#ifdef GXVK_ARENA_MUTANTS
extern int g_arena_mutant;
#define MUTANT(n) (g_arena_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace gxvk {
namespace {

using ull = unsigned long long;

#ifdef GXVK_ARENA_TESTING
Arena g_mode = Arena::kOff;  // tests/gxvk_arena_test.cpp sets it (arena_test_reset)
#else
const Arena g_mode = [] {
    const char* e = std::getenv("BBHOST_GXVK_ARENA");
    if (!e || !*e) return Arena::kOff;
    if (std::strcmp(e, "compare") == 0) return Arena::kCompare;
    return e[0] == '1' ? Arena::kInPlace : Arena::kOff;  // (2 is gone: arena_region says so)
}();
#endif

// One in-place binding in this many has its bytes hashed (compare: all).
constexpr std::uint32_t kSampleEvery = 64;
constexpr std::uint64_t kHashMax = 4096;  // a longer range's first bytes
constexpr std::uint64_t kPageShift = 16;

std::atomic<std::uint64_t> g_lo{0}, g_hi{0}, g_pool{0};
std::atomic<std::uint64_t> g_heap_lo{0}, g_heap_hi{0};  // the first heap's bounds, whatever the mode (upload_heap)

// The heap's hand-outs: their ranges, the newest submission (+1) that bound
// each, and a generation that changes when one is handed out over. Indexed
// by the heap's 64 KiB pages each overlaps.
struct Record {
    std::uint64_t start = 0, end = 0;
    std::uint64_t stamp = 0;  // the newest binding's submission + 1; 0: none
    std::uint32_t gen = 0;
    bool live = false;
    bool held = false;  // a copy-back's source held in it (arena_hold) since its hand-out
    // BBHOST_GXVK_COPYBACK_CHECK: the pool's node (its first dword is the status a retire's label writes), the
    // hand-out's number, and the submissions that must have completed for that label to have landed (0: not written).
    std::uint64_t node = 0, seq = 0, retire = 0;
    std::uint32_t tid = 0;  // (and the tag of the thread GX handed it out on: arena_thread_tag)
};
std::mutex g_mu;
std::vector<Record> g_records;            // under g_mu
// BBHOST_GXVK_COPYBACK_CHECK: the live records by their node (under g_mu), the nodes' span (a cheap test for the
// command processor's label writes) and the hand-outs so far, published for the GX threads.
const bool g_check = [] {
    const char* e = std::getenv("BBHOST_GXVK_COPYBACK_CHECK");
    return e && e[0] == '1';
}();
std::unordered_map<std::uint64_t, std::uint32_t> g_node_rec;
std::atomic<std::uint64_t> g_node_lo{~0ull}, g_node_hi{0}, g_seq_pub{0};
enum CheckClass : int { kChkReused, kChkLanded, kChkWalked, kChkOpen, kChkNoRecord, kChkClasses };
const char* const kChkName[kChkClasses] = {"its block handed out again since GX issued the copy-back",
                                           "its block's retire label landed (GX may take it back)", "its block retired, the label not landed",
                                           "its block not retired", "no hand-out holds it"};
std::atomic<std::uint64_t> g_chk[kChkClasses] = {}, g_chk_labels{0};
std::vector<std::uint32_t> g_free;        // under g_mu: records not live
std::vector<std::vector<std::uint32_t>> g_pages;  // under g_mu: by page, the live records over it
std::uint64_t g_live = 0, g_live_max = 0;         // under g_mu
// The last records handed out over (BBHOST_GXVK_ARENA_TRACE=1): where a binding
// no hand-out holds was, for the log.
struct Dropped {
    std::uint64_t start = 0, end = 0, stamp = 0, by_start = 0, by_end = 0, seq = 0;
};
constexpr std::size_t kDropped = 4096;
Dropped g_dropped[kDropped];  // under g_mu: a ring
std::uint64_t g_dropped_n = 0, g_handout_seq = 0;  // under g_mu
const bool g_trace = [] {
    const char* e = std::getenv("BBHOST_GXVK_ARENA_TRACE");
    return e && e[0] == '1';
}();
int g_traced = 0;  // under g_mu
// BBHOST_GXVK_ARENA_TRACE=1: bindings no hand-out holds, by MiB of the heap; hand-outs of other pools.
std::atomic<std::uint64_t> g_norecord_mib[256] = {};
std::atomic<std::uint64_t> g_other_pool{0}, g_other_pool_in_heap{0};
std::atomic<int> g_handout_said{0};

// A sampled binding, waiting for its submission.
struct Sample {
    std::uint64_t va = 0, bytes = 0, hash = 0, serial = 0;
    std::uint32_t record = 0, gen = 0;
};
constexpr std::size_t kSamples = 8192;
Sample g_samples[kSamples];  // under g_mu: a ring
std::size_t g_sample_head = 0, g_sample_n = 0;
std::uint32_t g_sample_tick = 0;

std::atomic<bool> g_in_place{true};  // false after a violation
std::atomic<int> g_said{0};
// BBHOST_TEST_ARENA_FAIL_AFTER=N: a violation after the N-th recorded hand-out,
// for testing what the arena does once its proof has failed.
#ifdef GXVK_ARENA_TESTING
const std::uint64_t g_fail_after = 0;
#else
const std::uint64_t g_fail_after = [] {
    const char* e = std::getenv("BBHOST_TEST_ARENA_FAIL_AFTER");
    return e && *e ? std::strtoull(e, nullptr, 10) : 0ull;
}();
#endif

std::atomic<std::uint64_t> g_handouts{0}, g_outside{0}, g_bound[kUses] = {}, g_bound_bytes[kUses] = {};
std::atomic<std::uint64_t> g_handouts_top{0};   // of g_handouts: from the pool's top end
std::atomic<std::uint64_t> g_unrecorded{0};     // hand-outs after the proof failed, left alone
enum Why : int { kWhyNoRecord, kWhyStraddle, kWhyStored, kWhyProofOff, kWhyNoBacking, kWhys };
const char* const kWhyName[kWhys] = {"no hand-out holds it", "it runs past its hand-out", "the shader stores to it", "the proof failed earlier",
                                     "the heap is not in one buffer"};
std::atomic<std::uint64_t> g_fell[kWhys] = {};
std::atomic<std::uint64_t> g_violations_order{0}, g_violations_bytes{0}, g_samples_checked{0}, g_samples_skipped{0};
std::atomic<std::uint64_t> g_noted[kArenaNotes] = {}, g_noted_bytes[kArenaNotes] = {};
// Copy-backs' sources held (arena_hold, stage 1d.2a), and those refused by reason (kWhyNoRecord, kWhyStraddle, kWhyProofOff).
std::atomic<std::uint64_t> g_holds{0}, g_hold_bytes{0}, g_hold_fell[kWhys] = {};
// Hand-outs over a held source whose copy had not run, which waited for the GPU first (arena_handout); this many
// within a minute stop the proof for the run (a rate no run has come near: each wait stalls a GX thread).
constexpr std::uint64_t kHoldWaitsMax = 16;
std::atomic<std::uint64_t> g_hold_waits{0};
std::atomic<std::int64_t> g_wait_minute{0};       // the steady clock's minute the count below is of
std::atomic<std::uint64_t> g_waits_in_minute{0};

// The records are kept for in-place bindings (BBHOST_GXVK_ARENA) or copy-backs from the heap (BBHOST_GXVK_HEAP_COPY).
bool proof_wanted() { return g_mode != Arena::kOff || heap_copy_wanted() || g_check; }

// The submissions whose work is done: the landing's count or the retirements',
// whichever is further. The landing's count raised before their labels are in
// memory (landing(), not landed()): GX hands a range out again as soon as it
// reads its release label, which can be before landed() counts the submission.
std::uint64_t done_submissions() {
    return std::max<std::uint64_t>(MUTANT(7) ? label_landing::landed() : label_landing::landing(), host_gpu_completed_submits_seen());
}

std::uint64_t hash_of(std::uint64_t va, std::uint64_t bytes) {
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va));
    std::uint64_t h = 1469598103934665603ull;
    const std::uint64_t n = std::min(bytes, kHashMax);
    std::uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    for (; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

void violation(const char* what, std::uint64_t va, std::uint64_t bytes, std::uint64_t serial, std::uint64_t done) {
    const bool was = g_in_place.exchange(false, std::memory_order_relaxed);
    if (g_said.fetch_add(1, std::memory_order_relaxed) < 4)
        host_log("gxvk arena: %s: 0x%llx, %llu bytes, bound in submission %llu with %llu completed%s", what, static_cast<ull>(va),
                 static_cast<ull>(bytes), static_cast<ull>(serial), static_cast<ull>(done),
                 was ? " - the heap's bindings take today's path from here on" : "");
}

void page_range(std::uint64_t start, std::uint64_t end, std::uint64_t& p0, std::uint64_t& p1) {
    const std::uint64_t lo = g_lo.load(std::memory_order_relaxed);
    p0 = (start - lo) >> kPageShift;
    p1 = (end - 1 - lo) >> kPageShift;
    if (p1 >= g_pages.size()) p1 = g_pages.size() - 1;
}

void drop_locked(std::uint32_t r) {
    Record& rec = g_records[r];
    std::uint64_t p0, p1;
    page_range(rec.start, rec.end, p0, p1);
    for (std::uint64_t p = p0; p <= p1; ++p) {
        auto& v = g_pages[p];
        v.erase(std::remove(v.begin(), v.end(), r), v.end());
    }
    rec.live = false;
    ++rec.gen;
    if (rec.node) {
        const auto it = g_node_rec.find(rec.node);
        if (it != g_node_rec.end() && it->second == r) g_node_rec.erase(it);
    }
    g_free.push_back(r);
    --g_live;
}

// The proof has failed: the records, their pages and the samples waiting go,
// once (under g_mu; nothing here is used again this run).
bool g_released = false;  // under g_mu
void release_locked() {
    if (g_released || MUTANT(6)) return;
    g_released = true;
    std::vector<Record>().swap(g_records);
    std::vector<std::uint32_t>().swap(g_free);
    std::vector<std::vector<std::uint32_t>>().swap(g_pages);
    g_live = 0;
    g_sample_head = g_sample_n = 0;
}

// The oldest samples whose submission has completed: hashed again.
void check_samples_locked(std::uint64_t done) {
    while (g_sample_n) {
        Sample& s = g_samples[g_sample_head];
        if (s.serial >= done) return;
        const Record& rec = g_records[s.record];
        if (rec.gen != s.gen && !MUTANT(4)) {
            g_samples_skipped.fetch_add(1, std::memory_order_relaxed);  // handed out again since: its bytes may change
        } else {
            g_samples_checked.fetch_add(1, std::memory_order_relaxed);
            if (hash_of(s.va, s.bytes) != s.hash) {
                g_violations_bytes.fetch_add(1, std::memory_order_relaxed);
                violation("bytes a queued draw read changed while their range was not handed out again", s.va, s.bytes, s.serial, done);
            }
        }
        g_sample_head = (g_sample_head + 1) % kSamples;
        --g_sample_n;
    }
}

}  // namespace

Arena arena_mode() { return g_mode; }

bool arena_proof_wanted() { return proof_wanted(); }

bool arena_recording() { return proof_wanted() && g_in_place.load(std::memory_order_relaxed); }

bool copyback_check_wanted() { return g_check; }

std::uint64_t arena_handout_seq() { return g_seq_pub.load(std::memory_order_acquire); }

void arena_label(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial) {
    if (!g_check || va + bytes <= g_node_lo.load(std::memory_order_relaxed) || va > g_node_hi.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_released) return;
    const auto it = g_node_rec.find(va);
    if (it == g_node_rec.end()) return;
    Record& rec = g_records[it->second];
    if (!rec.live || rec.node != va) return;
    rec.retire = serial + 1;
    g_chk_labels.fetch_add(1, std::memory_order_relaxed);
}

void copyback_check(std::uint64_t src, std::uint64_t bytes, std::uint64_t seq) {
    if (!g_check || !bytes) return;
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire), hi = g_hi.load(std::memory_order_relaxed);
    if (!lo || src < lo || src >= hi) return;  // not the heap's
    const std::uint64_t done = done_submissions();
    CheckClass c = kChkNoRecord;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_released) return;
        const std::uint64_t p = (src - lo) >> kPageShift;
        if (p < g_pages.size()) {
            for (const std::uint32_t r : g_pages[p]) {
                const Record& rec = g_records[r];
                if (!rec.live || src < rec.start || src >= rec.end) continue;
                c = rec.seq > seq ? kChkReused : !rec.retire ? kChkOpen : done >= rec.retire ? kChkLanded : kChkWalked;
                break;
            }
        }
    }
    g_chk[c].fetch_add(1, std::memory_order_relaxed);
}

std::uint32_t arena_thread_tag() {
    static std::atomic<std::uint32_t> next{1};
    thread_local std::uint32_t tag = 0;
    if (!tag) tag = next.fetch_add(1, std::memory_order_relaxed);
    return tag;
}

int capture_class(std::uint64_t src, std::uint64_t bytes, std::uint64_t seq, std::uint32_t tid, bool* landed) {
    *landed = false;
    if (!g_check || !bytes) return kCapNoRecord;
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire), hi = g_hi.load(std::memory_order_relaxed);
    if (!lo || src < lo || src >= hi) return kCapNotHeap;
    const std::uint64_t done = done_submissions();
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_released) return kCapNoRecord;
    const std::uint64_t p = (src - lo) >> kPageShift;
    if (p >= g_pages.size()) return kCapNoRecord;
    for (const std::uint32_t r : g_pages[p]) {
        const Record& rec = g_records[r];
        if (!rec.live || src < rec.start || src >= rec.end) continue;
        *landed = rec.retire && done >= rec.retire;
        return rec.seq <= seq ? kCapUpdateBlock : rec.tid != tid ? kCapSinceOtherThread : src == rec.start ? kCapNewBlock : kCapSinceSameThread;
    }
    return kCapNoRecord;
}

void arena_unrecorded() {
    if (proof_wanted()) g_unrecorded.fetch_add(1, std::memory_order_relaxed);
}

void arena_region(std::uint64_t pool, std::uint64_t region, std::uint64_t bytes) {
    // =2 moved the heap into video memory the CPU writes: two thirds of the
    // frame rate gone, the CPU reading the heap at ~30 MB/s there - removed.
    if (const char* e = std::getenv("BBHOST_GXVK_ARENA"); e && e[0] == '2')
        host_log("gxvk arena: BBHOST_GXVK_ARENA=2 is gone - the heap in video memory the CPU writes was measured slower "
                 "(docs/rendering.md); today's path");
    // The pool's bounds as its init sets them: 16-byte aligned.
    const std::uint64_t pad = (0u - static_cast<std::uint32_t>(region)) & 0xfu;
    const std::uint64_t lo = region + pad, hi = lo + ((bytes - pad) & ~std::uint64_t{0xf});
    if (!g_heap_lo.load(std::memory_order_relaxed)) {
        g_heap_hi.store(hi, std::memory_order_relaxed);
        g_heap_lo.store(lo, std::memory_order_release);
    }
    heap_snap_region(lo, hi);  // stage 1d.1's snapshots with the lookups' answers known (host/gxvk_heapsnap.cpp), when on
    if (!proof_wanted() && !census()) return;  // the census counts today's path over the heap
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_lo.load(std::memory_order_relaxed)) {
            if (g_said.fetch_add(1, std::memory_order_relaxed) < 4)
                host_log("gxvk arena: a second upload heap (0x%llx, %llu MiB) - only the first is bound in place", static_cast<ull>(lo),
                         static_cast<ull>((hi - lo) >> 20));
            return;
        }
        g_pages.assign(((hi - lo) >> kPageShift) + 1, {});
        g_pool.store(pool, std::memory_order_relaxed);
        g_hi.store(hi, std::memory_order_relaxed);
        g_lo.store(lo, std::memory_order_release);
    }
    if (g_mode != Arena::kOff)
        host_log("gxvk arena: GX's upload heap is 0x%llx, %llu MiB: its bindings %s, under the release proof (BBHOST_GXVK_ARENA=0: today's path)",
                 static_cast<ull>(lo), static_cast<ull>((hi - lo) >> 20),
                 g_mode == Arena::kCompare ? "checked as they would be bound in place, and bound as today" : "in place");
    else if (heap_copy_wanted())
        host_log("gxvk arena: GX's upload heap is 0x%llx, %llu MiB: its hand-outs recorded for the copy-backs' release proof "
                 "(BBHOST_GXVK_HEAP_COPY), its bindings as today", static_cast<ull>(lo), static_cast<ull>((hi - lo) >> 20));
}

bool upload_heap(std::uint64_t* lo, std::uint64_t* hi) {
    const std::uint64_t l = g_heap_lo.load(std::memory_order_acquire);
    if (!l) return false;
    *lo = l;
    *hi = g_heap_hi.load(std::memory_order_relaxed);
    return true;
}

bool arena_heap(std::uint64_t* lo, std::uint64_t* hi) {
    const std::uint64_t l = g_lo.load(std::memory_order_acquire);
    if (!l) return false;
    *lo = l;
    *hi = g_hi.load(std::memory_order_relaxed);
    return true;
}

// Whether a live record over [start, end) holds a copy-back's source whose stamp `done` has not reached (under g_mu).
bool held_pending_locked(std::uint64_t start, std::uint64_t end, std::uint64_t done) {
    if (g_released) return false;
    std::uint64_t p0, p1;
    page_range(start, end, p0, p1);
    for (std::uint64_t p = p0; p <= p1; ++p) {
        for (const std::uint32_t r : g_pages[p]) {
            const Record& rec = g_records[r];
            if (rec.live && rec.held && rec.end > start && rec.start < end && rec.stamp > done) return true;
        }
    }
    return false;
}

void arena_handout(std::uint64_t pool, std::uint64_t node, bool top) {
    if (!proof_wanted() || !node) return;
    if (!g_in_place.load(std::memory_order_relaxed) && !MUTANT(6)) {  // the proof failed: nothing more to record
        g_unrecorded.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire), hi = g_hi.load(std::memory_order_relaxed);
    std::uint64_t start = 0, next = 0, end = 0;
    std::memcpy(&start, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(node + 8)), 8);
    std::memcpy(&next, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(node + 0x18)), 8);
    if (next) std::memcpy(&end, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(next + 8)), 8);
    if (g_trace && (hi && start >= hi - (1u << 20) && start < hi) && g_handout_said.fetch_add(1) < 16)
        host_log("gxvk arena: a hand-out at the heap's top: [0x%llx, 0x%llx) by pool 0x%llx (the heap's 0x%llx)", static_cast<ull>(start),
                 static_cast<ull>(end), static_cast<ull>(pool), static_cast<ull>(g_pool.load()));
    if (pool != g_pool.load(std::memory_order_relaxed)) {
        g_other_pool.fetch_add(1, std::memory_order_relaxed);
        if (lo && start >= lo && start < hi) g_other_pool_in_heap.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (!lo || start < lo || end > hi || end <= start) {
        g_outside.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::uint64_t done = done_submissions();
    std::unique_lock<std::mutex> lk(g_mu);
    // A held copy-back's source (stage 1d.2a) whose copy has not run yet. GX takes a block back once the end-of-pipe
    // label it writes when it retires the block has landed - but it retires a context's blocks when UpdateSubresource
    // kicks the context's command buffer (sub_2aaf700), before the copy-backs that read renames in them are resolved
    // (sub_2aadf50, a later submission). On the console that copy runs right after the retire; here the version's GPU
    // copy runs with its own submission, which GX's reclaim can beat. The new owner has written nothing yet: the
    // hand-out waits until what is recorded has been submitted and has run, as GX's own block reclaim does
    // (decomp/gx/block_reclaim.cpp), so the source has been read first. kHoldWaitsMax such waits within a minute stop
    // the proof for the run (copy-backs as today). The wait is outside g_mu: it takes the renderer's lock, which
    // arena_hold's callers hold.
    if (heap_copy_wanted() && !MUTANT(11) && held_pending_locked(start, end, done)) {
        lk.unlock();
        g_hold_waits.fetch_add(1, std::memory_order_relaxed);
#ifdef GXVK_ARENA_TESTING
        const std::int64_t minute = 1;  // (a test's sequences take milliseconds: one minute, never a boundary)
#else
        const std::int64_t minute =
            std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
        if (g_wait_minute.exchange(minute, std::memory_order_relaxed) != minute) g_waits_in_minute.store(0, std::memory_order_relaxed);
        const std::uint64_t in_minute = g_waits_in_minute.fetch_add(1, std::memory_order_relaxed) + 1;
        host_gpu_flush();
        if (!MUTANT(12)) done = done_submissions();
        if (in_minute >= kHoldWaitsMax && !MUTANT(13))
            violation("the hand-outs' waits for held copy-backs' sources reached their limit for a minute", start, end - start, 0, done);
        lk.lock();
    }
    if (g_released) {  // the proof failed while this hand-out came in
        g_unrecorded.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const std::uint64_t recorded = g_handouts.fetch_add(1, std::memory_order_relaxed) + 1;
    if (top) g_handouts_top.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t p0, p1;
    page_range(start, end, p0, p1);
    // Every record the new range overlaps is over: its bindings must have completed.
    for (std::uint64_t p = p0; p <= p1; ++p) {
        const std::vector<std::uint32_t> over = g_pages[p];
        for (const std::uint32_t r : over) {
            Record& rec = g_records[r];
            if (!rec.live || rec.end <= start || rec.start >= end) continue;
            if (MUTANT(2) && (rec.start < start || rec.end > end)) continue;  // only records the new range holds whole
            if (rec.stamp > done) {
                g_violations_order.fetch_add(1, std::memory_order_relaxed);
                violation("a range handed out again before the submission that bound it completed", rec.start, rec.end - rec.start, rec.stamp - 1,
                          done);
            }
            if (g_trace) g_dropped[g_dropped_n++ % kDropped] = {rec.start, rec.end, rec.stamp, start, end, g_handout_seq};
            drop_locked(r);
        }
    }
    std::uint32_t r;
    if (!g_free.empty()) {
        r = g_free.back();
        g_free.pop_back();
    } else {
        r = static_cast<std::uint32_t>(g_records.size());
        g_records.emplace_back();
    }
    Record& rec = g_records[r];
    rec.start = start;
    rec.end = end;
    rec.stamp = 0;
    rec.live = true;
    rec.held = false;
    ++g_handout_seq;
    if (g_check) {
        rec.node = node;
        rec.seq = g_handout_seq;
        rec.retire = 0;
        rec.tid = arena_thread_tag();
        g_node_rec[node] = r;
        if (node < g_node_lo.load(std::memory_order_relaxed)) g_node_lo.store(node, std::memory_order_relaxed);
        if (node > g_node_hi.load(std::memory_order_relaxed)) g_node_hi.store(node, std::memory_order_relaxed);
        g_seq_pub.store(g_handout_seq, std::memory_order_release);
    }
    for (std::uint64_t p = p0; p <= p1; ++p) g_pages[p].push_back(r);
    if (++g_live > g_live_max) g_live_max = g_live;
    if (g_fail_after && recorded == g_fail_after)
        violation("a violation BBHOST_TEST_ARENA_FAIL_AFTER asked for", start, end - start, 0, done);
    if (!g_in_place.load(std::memory_order_relaxed)) release_locked();
}

bool arena_bind(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial, Use use, bool stored) {
    if (g_mode == Arena::kOff) return false;
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire), hi = g_hi.load(std::memory_order_relaxed);
    if (!lo || va < lo || (va >= hi && !MUTANT(5))) return false;  // not the heap's: nothing to count
    if (!g_in_place.load(std::memory_order_relaxed) && !MUTANT(6)) {  // the proof failed: today's path, no lookup
        g_fell[kWhyProofOff].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (bytes > hi - va) {
        g_fell[kWhyStraddle].fetch_add(1, std::memory_order_relaxed);  // runs past the heap's end
        return false;
    }
    if (stored) {
        g_fell[kWhyStored].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint64_t end = va + std::max<std::uint64_t>(bytes, 1);
    bool found = false, straddle = false, failed = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const std::uint64_t p = (va - lo) >> kPageShift;
        if (g_released) {
            failed = true;  // the proof failed while this binding came in
        } else if (p < g_pages.size()) {
            for (const std::uint32_t r : g_pages[p]) {
                Record& rec = g_records[r];
                if (va < rec.start || va >= rec.end) continue;
                if (end > rec.end && !MUTANT(3)) {
                    straddle = true;
                    break;
                }
                rec.stamp = std::max(rec.stamp, MUTANT(1) ? serial : serial + 1);
                found = true;
                // The bytes, sampled (compare: every one).
                if (g_mode == Arena::kCompare || ++g_sample_tick % kSampleEvery == 0) {
                    const std::uint64_t done = done_submissions();
                    check_samples_locked(done);
                    if (g_sample_n == kSamples) {  // full: the oldest goes unchecked
                        g_samples_skipped.fetch_add(1, std::memory_order_relaxed);
                        g_sample_head = (g_sample_head + 1) % kSamples;
                        --g_sample_n;
                    }
                    Sample& s = g_samples[(g_sample_head + g_sample_n) % kSamples];
                    s = {va, bytes, hash_of(va, bytes), serial, r, rec.gen};
                    ++g_sample_n;
                }
                break;
            }
        }
        if (!g_in_place.load(std::memory_order_relaxed)) {  // a sample just checked found changed bytes
            release_locked();
            failed = true;
        }
    }
    if (failed) {
        g_fell[kWhyProofOff].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!found) {
        g_fell[straddle ? kWhyStraddle : kWhyNoRecord].fetch_add(1, std::memory_order_relaxed);
        if (g_trace && !straddle) g_norecord_mib[std::min<std::uint64_t>((va - lo) >> 20, 255)].fetch_add(1, std::memory_order_relaxed);
        if (g_trace && !straddle) {
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_traced < 24) {
                ++g_traced;
                const std::uint64_t p = (va - lo) >> kPageShift;
                std::uint64_t near_lo = 0, near_hi = 0;
                std::size_t n = p < g_pages.size() ? g_pages[p].size() : 0;
                for (std::size_t k = 0; k < n; ++k) {
                    const Record& rec = g_records[g_pages[p][k]];
                    if (rec.end <= va && rec.end > near_lo) near_lo = rec.end;
                    if (rec.start > va && (!near_hi || rec.start < near_hi)) near_hi = rec.start;
                }
                const Dropped* was = nullptr;
                for (std::uint64_t k = 0; k < std::min<std::uint64_t>(g_dropped_n, kDropped); ++k) {
                    const Dropped& d = g_dropped[(g_dropped_n - 1 - k) % kDropped];
                    if (va >= d.start && va < d.end) {
                        was = &d;
                        break;
                    }
                }
                char drop[200] = "";
                if (was)
                    std::snprintf(drop, sizeof drop, "; it was in [0x%llx, 0x%llx), stamp %llu, handed out over by [0x%llx, 0x%llx) %llu hand-outs ago",
                                  static_cast<ull>(was->start), static_cast<ull>(was->end), static_cast<ull>(was->stamp), static_cast<ull>(was->by_start),
                                  static_cast<ull>(was->by_end), static_cast<ull>(g_handout_seq - was->seq));
                host_log("gxvk arena: a binding no hand-out holds: 0x%llx, %llu bytes, submission %llu (%llu completed); its page has %zu records, the "
                         "nearest ending 0x%llx before it, the nearest starting 0x%llx after it%s",
                         static_cast<ull>(va), static_cast<ull>(bytes), static_cast<ull>(serial), static_cast<ull>(done_submissions()), n,
                         static_cast<ull>(near_lo), static_cast<ull>(near_hi), drop);
            }
        }
        return false;
    }
    if (!g_in_place.load(std::memory_order_relaxed)) {
        g_fell[kWhyProofOff].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_bound[use].fetch_add(1, std::memory_order_relaxed);
    g_bound_bytes[use].fetch_add(bytes, std::memory_order_relaxed);
    return g_mode == Arena::kInPlace;
}

bool arena_hold(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial) {
    if (!proof_wanted()) return false;
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire), hi = g_hi.load(std::memory_order_relaxed);
    if (!lo || va < lo || va >= hi) return false;  // not the heap's: nothing to count
    if (!g_in_place.load(std::memory_order_relaxed)) {  // the proof failed: no lookup
        g_hold_fell[kWhyProofOff].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (bytes > hi - va) {
        g_hold_fell[kWhyStraddle].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint64_t end = va + std::max<std::uint64_t>(bytes, 1);
    bool found = false, straddle = false, failed = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const std::uint64_t p = (va - lo) >> kPageShift;
        if (g_released) {
            failed = true;
        } else if (p < g_pages.size()) {
            for (const std::uint32_t r : g_pages[p]) {
                Record& rec = g_records[r];
                if (va < rec.start || va >= rec.end) continue;
                if (end > rec.end && !MUTANT(8)) {
                    straddle = true;
                    break;
                }
                if (!MUTANT(9)) rec.stamp = std::max(rec.stamp, serial + 1);
                rec.held = true;
                found = true;
                // The bytes, sampled (BBHOST_GXVK_HEAP_COPY=compare: every one).
                if (heap_copy_mode() == HeapCopy::kCompare || ++g_sample_tick % kSampleEvery == 0) {
                    const std::uint64_t done = done_submissions();
                    check_samples_locked(done);
                    if (g_sample_n == kSamples) {
                        g_samples_skipped.fetch_add(1, std::memory_order_relaxed);
                        g_sample_head = (g_sample_head + 1) % kSamples;
                        --g_sample_n;
                    }
                    Sample& smp = g_samples[(g_sample_head + g_sample_n) % kSamples];
                    smp = {va, bytes, hash_of(va, bytes), serial, r, rec.gen};
                    ++g_sample_n;
                }
                break;
            }
        }
        if (!g_in_place.load(std::memory_order_relaxed)) {  // a sample just checked found changed bytes
            release_locked();
            failed = true;
        }
    }
    if (failed) {
        g_hold_fell[kWhyProofOff].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!found && !MUTANT(10)) {
        g_hold_fell[straddle ? kWhyStraddle : kWhyNoRecord].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_holds.fetch_add(1, std::memory_order_relaxed);
    g_hold_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

void arena_no_backing() { g_fell[kWhyNoBacking].fetch_add(1, std::memory_order_relaxed); }

void arena_note(ArenaNote what, std::uint64_t bytes) {
    g_noted[what].fetch_add(1, std::memory_order_relaxed);
    g_noted_bytes[what].fetch_add(bytes, std::memory_order_relaxed);
}

#ifdef GXVK_ARENA_TESTING
void arena_test_reset(Arena mode) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_mode = mode;
    g_lo.store(0);
    g_hi.store(0);
    g_pool.store(0);
    g_records.clear();
    g_free.clear();
    g_pages.clear();
    g_live = g_live_max = 0;
    g_sample_head = g_sample_n = 0;
    g_sample_tick = 0;
    g_in_place.store(true);
    g_said.store(0);
    g_handouts.store(0);
    g_handouts_top.store(0);
    g_unrecorded.store(0);
    g_released = false;
    g_outside.store(0);
    for (auto& a : g_bound) a.store(0);
    for (auto& a : g_bound_bytes) a.store(0);
    for (auto& a : g_fell) a.store(0);
    g_violations_order.store(0);
    g_violations_bytes.store(0);
    g_samples_checked.store(0);
    g_samples_skipped.store(0);
    g_holds.store(0);
    g_hold_bytes.store(0);
    for (auto& a : g_hold_fell) a.store(0);
    g_hold_waits.store(0);
    g_wait_minute.store(0);
    g_waits_in_minute.store(0);
}
ArenaTestCounts arena_test_counts() {
    ArenaTestCounts c;
    c.violations_order = g_violations_order.load();
    c.violations_bytes = g_violations_bytes.load();
    c.samples_checked = g_samples_checked.load();
    c.in_place = g_in_place.load();
    for (int w = 0; w < kWhys; ++w) c.fell += g_fell[w].load();
    c.proof_off = g_fell[kWhyProofOff].load();
    c.holds = g_holds.load();
    for (int w = 0; w < kWhys; ++w) c.hold_fell += g_hold_fell[w].load();
    c.hold_waits = g_hold_waits.load();
    c.unrecorded = g_unrecorded.load();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        c.live = g_live;
    }
    return c;
}
#endif

void arena_report() {
    if (g_check) {
        std::string by;
        std::uint64_t all = 0;
        for (int c = 0; c < kChkClasses; ++c) all += g_chk[c].load();
        for (int c = 0; c < kChkClasses; ++c) {
            char b[160];
            std::snprintf(b, sizeof b, "%s%s %llu", c ? "; " : "", kChkName[c], static_cast<ull>(g_chk[c].load()));
            by += b;
        }
        host_log("gxvk copy-back check (BBHOST_GXVK_COPYBACK_CHECK): %llu copy-backs from the heap as the command processor reached "
                 "their tokens: %s. Retire labels the command processor wrote on a hand-out's node: %llu; hand-outs recorded %llu%s",
                 static_cast<ull>(all), by.c_str(), static_cast<ull>(g_chk_labels.load()), static_cast<ull>(g_handouts.load()),
                 g_released ? " (the records stopped early)" : "");
    }
    if (!proof_wanted() && !census()) return;
    const auto v = [](const std::atomic<std::uint64_t>& a) { return static_cast<ull>(a.load(std::memory_order_relaxed)); };
    if (!proof_wanted()) {
        std::string notes;
        static const char* const kNoteName[kArenaNotes] = {"constant-buffer snapshots", "vertex snapshots", "GPU writes pending", "descriptor-table reads"};
        for (int n = 0; n < kArenaNotes; ++n) {
            char b[96];
            std::snprintf(b, sizeof b, "%s%s %llu (%llu MiB)", n ? ", " : "", kNoteName[n], v(g_noted[n]), v(g_noted_bytes[n]) >> 20);
            notes += b;
        }
        host_log("gxvk census: today's path over GX's upload heap: %s", notes.c_str());
        return;
    }
    std::uint64_t live_max = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_released) check_samples_locked(done_submissions());
        if (!g_in_place.load(std::memory_order_relaxed)) release_locked();
        live_max = g_live_max;
    }
    if (g_trace) {
        std::string mib;
        for (int m = 0; m < 256; ++m) {
            const std::uint64_t n = g_norecord_mib[m].load();
            if (!n) continue;
            char b[48];
            std::snprintf(b, sizeof b, " %d:%llu", m, static_cast<ull>(n));
            mib += b;
        }
        host_log("gxvk arena: bindings no hand-out holds, by MiB from the heap's start:%s; hand-outs of other pools %llu (%llu in the heap)",
                 mib.c_str(), v(g_other_pool), v(g_other_pool_in_heap));
    }
    static const char* const kUseLabel[kUses] = {"vertex records", "index buffers", "constant buffers", "unordered-access buffers"};
    std::string bound;
    for (int u = 0; u < kUses; ++u) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu (%llu MiB)", u ? ", " : "", kUseLabel[u], v(g_bound[u]), v(g_bound_bytes[u]) >> 20);
        bound += b;
    }
    std::string fell;
    for (int w = 0; w < kWhys; ++w) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu", w ? ", " : "", kWhyName[w], v(g_fell[w]));
        fell += b;
    }
    const bool placing = g_mode == Arena::kInPlace;
    if (g_mode != Arena::kOff)
        host_log("gxvk arena: %s; the heap's bindings %s: %s; today's path because %s", placing ? "bound in place" : "compared",
                 placing ? "in place" : "that would be in place", bound.c_str(), fell.c_str());
    if (heap_copy_wanted())
        host_log("gxvk arena: copy-backs' sources held %llu (%llu MiB); not held because no hand-out holds it %llu, it runs past its "
                 "hand-out %llu, the proof failed earlier %llu; hand-outs over a held source whose copy had not run, which waited for "
                 "the GPU first %llu",
                 v(g_holds), v(g_hold_bytes) >> 20, v(g_hold_fell[kWhyNoRecord]), v(g_hold_fell[kWhyStraddle]), v(g_hold_fell[kWhyProofOff]),
                 v(g_hold_waits));
    char after[160] = "";
    if (!g_in_place.load())
        std::snprintf(after, sizeof after, " - in place stopped at the first, and the arena with it: %llu hand-outs and %llu bindings left alone after",
                      v(g_unrecorded), v(g_fell[kWhyProofOff]));
    host_log("gxvk arena: the release proof: %llu hand-outs recorded (%llu from the pool's top end, %llu outside the heap), %llu alive at most; "
             "%llu handed out again before their binding's submission completed; bytes checked %llu (%llu skipped: handed out again first, or "
             "the ring full), %llu changed under a queued draw%s",
             v(g_handouts), v(g_handouts_top), v(g_outside), static_cast<ull>(live_max), v(g_violations_order), v(g_samples_checked),
             v(g_samples_skipped), v(g_violations_bytes), after);
    static const char* const kNoteName[kArenaNotes] = {"constant-buffer snapshots", "vertex snapshots", "GPU writes pending",
                                                       "descriptor-table reads"};
    std::string notes;
    for (int n = 0; n < kArenaNotes; ++n) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu (%llu MiB)", n ? ", " : "", kNoteName[n], v(g_noted[n]), v(g_noted_bytes[n]) >> 20);
        notes += b;
    }
    if (g_mode != Arena::kOff) host_log("gxvk arena: today's path over the heap: %s", notes.c_str());
}

}  // namespace gxvk
