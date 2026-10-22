// GX on Vulkan, stage 1c: GX's upload heap bound in place (host/gxvk.h).
//
// The heap is the one region GX's device gives its block pool, which every
// renamed constant buffer, Map rename, ring allocation and draw table comes
// from (decomp/gx/upload_heap.cpp tells us of it). GX hands a range of it out
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
// today's path for the rest of the run.
//
// BBHOST_GXVK_ARENA=1 binds in place; =compare keeps today's path and checks
// every binding of the heap as 1 would bind it (the records and the bytes,
// every one). Off (0, the default) nothing here runs.
#include "host/gxvk.h"

#include "hle/modules.h"
#include "host/arena_move.h"
#include "host/gpu.h"
#include "host/label_landing.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
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
    return e[0] == '1' ? Arena::kInPlace : e[0] == '2' ? Arena::kMoved : Arena::kOff;
}();
#endif

// One in-place binding in this many has its bytes hashed (compare: all).
constexpr std::uint32_t kSampleEvery = 64;
constexpr std::uint64_t kHashMax = 4096;  // a longer range's first bytes
// The heap moved into video memory the CPU writes (=2): the CPU reads it at
// tens of MB/s (an RTX 4070's: 64 MiB took 2.1 s), under the arena's lock on
// the command processor, so its bytes are sampled one binding in 1024 and
// hashed over their first 256 bytes.
constexpr std::uint32_t kSampleEveryMoved = 1024;
constexpr std::uint64_t kHashMaxMoved = 256;
constexpr std::uint64_t kPageShift = 16;

std::atomic<std::uint64_t> g_lo{0}, g_hi{0}, g_pool{0};

// The heap's hand-outs: their ranges, the newest submission (+1) that bound
// each, and a generation that changes when one is handed out over. Indexed
// by the heap's 64 KiB pages each overlaps.
struct Record {
    std::uint64_t start = 0, end = 0;
    std::uint64_t stamp = 0;  // the newest binding's submission + 1; 0: none
    std::uint32_t gen = 0;
    bool live = false;
};
std::mutex g_mu;
std::vector<Record> g_records;            // under g_mu
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
// BBHOST_GXVK_ARENA=2: the host mapping the pool was given, to move at the
// GPU's start; and while it moves, its range (the fault handler's).
std::atomic<std::uint64_t> g_move_va{0}, g_move_bytes{0};
// The pieces the host mapping is recorded as (each its own mapping, import and
// move: host/arena_move.h); 0 while the heap is not ours.
std::atomic<std::uint64_t> g_piece{0};
std::atomic<std::uint64_t> g_moving_lo{0}, g_moving_hi{0};
std::atomic<int> g_moved{-1};  // -1 not tried, 0 failed, 1 moved
std::atomic<int> g_said{0};

std::atomic<std::uint64_t> g_handouts{0}, g_outside{0}, g_bound[kUses] = {}, g_bound_bytes[kUses] = {};
enum Why : int { kWhyNoRecord, kWhyStraddle, kWhyStored, kWhyProofOff, kWhyNoBacking, kWhyPieces, kWhys };
const char* const kWhyName[kWhys] = {"no hand-out holds it", "it runs past its hand-out", "the shader stores to it", "the proof failed earlier",
                                     "the heap is not in one buffer", "it crosses two of the heap's pieces"};
std::atomic<std::uint64_t> g_fell[kWhys] = {};
std::atomic<std::uint64_t> g_violations_order{0}, g_violations_bytes{0}, g_samples_checked{0}, g_samples_skipped{0};
std::atomic<std::uint64_t> g_noted[kArenaNotes] = {}, g_noted_bytes[kArenaNotes] = {};

std::uint64_t done_submissions() {
    return std::max<std::uint64_t>(label_landing::landed(), host_gpu_completed_submits_seen());
}

std::uint64_t hash_of(std::uint64_t va, std::uint64_t bytes) {
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va));
    std::uint64_t h = 1469598103934665603ull;
    const std::uint64_t n = std::min(bytes, g_moved.load(std::memory_order_relaxed) == 1 ? kHashMaxMoved : kHashMax);
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
    g_free.push_back(r);
    --g_live;
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

std::uint64_t arena_region(std::uint64_t pool, std::uint64_t region, std::uint64_t bytes, std::uint64_t* out_bytes) {
    if (g_mode == Arena::kOff && !census()) return 0;  // the census counts today's path over the heap
#if !defined(_WIN32)
    // BBHOST_GXVK_ARENA=2: the pool gets a host mapping of our own, low enough
    // for a V# (40 bits) and GPU-visible as the kernel's own host mappings
    // are; at the GPU's start its bytes move into video memory placed over it
    // (gpu.cpp). The game's region stays allocated, unused.
    // In pieces (BBHOST_GXVK_ARENA_PIECE_MB, default 32): mappings side by side,
    // each moved and imported on its own - NVIDIA's driver places at most 124
    // MiB of that memory at a time (host/arena_move.h).
    if (g_mode == Arena::kMoved && !g_move_va.load(std::memory_order_relaxed)) {
        const std::uint64_t len = (bytes + 0xffff) & ~std::uint64_t{0xffff};
        const std::uint64_t piece = arena_move::piece_bytes(std::getenv("BBHOST_GXVK_ARENA_PIECE_MB"));
        if (const std::uint64_t va = hle_kernel_map_host_pieces(len, piece)) {
            g_move_bytes.store(len, std::memory_order_relaxed);
            g_piece.store(piece, std::memory_order_relaxed);
            g_move_va.store(va, std::memory_order_release);
            host_log("gxvk arena: GX's upload heap goes to 0x%llx (%llu MiB, %llu pieces of up to %llu MiB) to move at the GPU's start, "
                     "not the game's region 0x%llx",
                     static_cast<ull>(va), static_cast<ull>(len >> 20), static_cast<ull>((len + piece - 1) / piece), static_cast<ull>(piece >> 20),
                     static_cast<ull>(region));
            region = va;
            *out_bytes = bytes;
        }
    }
#endif
    // The pool's bounds as its init sets them: 16-byte aligned.
    const std::uint64_t pad = (0u - static_cast<std::uint32_t>(region)) & 0xfu;
    const std::uint64_t lo = region + pad, hi = lo + ((bytes - pad) & ~std::uint64_t{0xf});
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_lo.load(std::memory_order_relaxed)) {
            if (g_said.fetch_add(1, std::memory_order_relaxed) < 4)
                host_log("gxvk arena: a second upload heap (0x%llx, %llu MiB) - only the first is bound in place", static_cast<ull>(lo),
                         static_cast<ull>((hi - lo) >> 20));
            return 0;
        }
        g_pages.assign(((hi - lo) >> kPageShift) + 1, {});
        g_pool.store(pool, std::memory_order_relaxed);
        g_hi.store(hi, std::memory_order_relaxed);
        g_lo.store(lo, std::memory_order_release);
    }
    if (g_move_va.load(std::memory_order_relaxed) == region) return region;  // BBHOST_GXVK_ARENA=2: ours
    if (g_mode != Arena::kOff)
        host_log("gxvk arena: GX's upload heap is 0x%llx, %llu MiB: its bindings %s, under the release proof (BBHOST_GXVK_ARENA=0: today's path)",
                 static_cast<ull>(lo), static_cast<ull>((hi - lo) >> 20),
                 g_mode == Arena::kCompare ? "checked as they would be bound in place, and bound as today" : "in place");
    return 0;  // the game's region stays: stage 1c binds it where it is
}

bool arena_heap(std::uint64_t* lo, std::uint64_t* hi) {
    const std::uint64_t l = g_lo.load(std::memory_order_acquire);
    if (!l) return false;
    *lo = l;
    *hi = g_hi.load(std::memory_order_relaxed);
    return true;
}

bool arena_pieces(std::uint64_t* va, std::uint64_t* bytes, std::uint64_t* piece) {
    const std::uint64_t p = g_piece.load(std::memory_order_relaxed);
    const std::uint64_t v = g_move_va.load(std::memory_order_acquire);
    if (!p || !v) return false;
    *va = v;
    *bytes = g_move_bytes.load(std::memory_order_relaxed);
    *piece = p;
    return true;
}

bool arena_moved_ok() { return g_moved.load(std::memory_order_relaxed) == 1; }

void arena_handout(std::uint64_t pool, std::uint64_t node) {
    if (g_mode == Arena::kOff || !node) return;
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
    g_handouts.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t done = done_submissions();
    std::lock_guard<std::mutex> lk(g_mu);
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
    ++g_handout_seq;
    for (std::uint64_t p = p0; p <= p1; ++p) g_pages[p].push_back(r);
    if (++g_live > g_live_max) g_live_max = g_live;
}

bool arena_bind(std::uint64_t va, std::uint64_t bytes, std::uint64_t serial, Use use, bool stored) {
    if (g_mode == Arena::kOff) return false;
    const std::uint64_t lo = g_lo.load(std::memory_order_acquire), hi = g_hi.load(std::memory_order_relaxed);
    if (!lo || va < lo || (va >= hi && !MUTANT(5))) return false;  // not the heap's: nothing to count
    if (bytes > hi - va) {
        g_fell[kWhyStraddle].fetch_add(1, std::memory_order_relaxed);  // runs past the heap's end
        return false;
    }
    if (stored) {
        g_fell[kWhyStored].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint64_t end = va + std::max<std::uint64_t>(bytes, 1);
    bool found = false, straddle = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const std::uint64_t p = (va - lo) >> kPageShift;
        if (p < g_pages.size()) {
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
                if (g_mode == Arena::kCompare ||
                    ++g_sample_tick % (g_moved.load(std::memory_order_relaxed) == 1 ? kSampleEveryMoved : kSampleEvery) == 0) {
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
    return g_mode == Arena::kInPlace || g_mode == Arena::kMoved;
}

bool arena_move_pending(std::uint64_t* va, std::uint64_t* bytes) {
    if (g_moved.load(std::memory_order_relaxed) >= 0) return false;
    const std::uint64_t v = g_move_va.load(std::memory_order_acquire);
    if (!v) return false;
    *va = v;
    *bytes = g_move_bytes.load(std::memory_order_relaxed);
    return true;
}

void arena_moving(bool on) {
    if (on) {
        g_moving_hi.store(g_move_va.load() + g_move_bytes.load(), std::memory_order_relaxed);
        g_moving_lo.store(g_move_va.load(), std::memory_order_release);
    } else {
        g_moving_lo.store(0, std::memory_order_release);
    }
}

void arena_moved(bool ok, const char* what) {
    g_moved.store(ok ? 1 : 0, std::memory_order_relaxed);
    host_log("gxvk arena: GX's upload heap %s %s", ok ? "moved into" : "stays in host memory:", what);
}

__attribute__((no_stack_protector)) bool arena_move_fault(std::uint64_t addr) {
    const std::uint64_t lo = g_moving_lo.load(std::memory_order_acquire);
    if (!lo || addr < lo || addr >= g_moving_hi.load(std::memory_order_relaxed)) return false;
    while (g_moving_lo.load(std::memory_order_acquire)) __builtin_ia32_pause();
    return true;
}

void arena_no_backing() { g_fell[kWhyNoBacking].fetch_add(1, std::memory_order_relaxed); }
void arena_crossing() { g_fell[kWhyPieces].fetch_add(1, std::memory_order_relaxed); }

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
    g_outside.store(0);
    for (auto& a : g_bound) a.store(0);
    for (auto& a : g_bound_bytes) a.store(0);
    for (auto& a : g_fell) a.store(0);
    g_violations_order.store(0);
    g_violations_bytes.store(0);
    g_samples_checked.store(0);
    g_samples_skipped.store(0);
}
ArenaTestCounts arena_test_counts() {
    ArenaTestCounts c;
    c.violations_order = g_violations_order.load();
    c.violations_bytes = g_violations_bytes.load();
    c.samples_checked = g_samples_checked.load();
    c.in_place = g_in_place.load();
    for (int w = 0; w < kWhys; ++w) c.fell += g_fell[w].load();
    return c;
}
#endif

void arena_report() {
    if (g_mode == Arena::kOff && !census()) return;
    const auto v = [](const std::atomic<std::uint64_t>& a) { return static_cast<ull>(a.load(std::memory_order_relaxed)); };
    if (g_mode == Arena::kOff) {
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
        check_samples_locked(done_submissions());
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
        if (w == kWhyPieces && !g_piece.load(std::memory_order_relaxed)) continue;  // =2's: the heap in pieces
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu", w ? ", " : "", kWhyName[w], v(g_fell[w]));
        fell += b;
    }
    const bool placing = g_mode == Arena::kInPlace || g_mode == Arena::kMoved;
    host_log("gxvk arena: %s; the heap's bindings %s: %s; today's path because %s",
             g_mode == Arena::kCompare ? "compared" : g_moved.load() == 1 ? "bound in place, the heap moved" : "bound in place",
             placing ? "in place" : "that would be in place", bound.c_str(), fell.c_str());
    host_log("gxvk arena: the release proof: %llu hand-outs recorded (%llu outside the heap), %llu alive at most; %llu handed out again before "
             "their binding's submission completed; bytes checked %llu (%llu skipped: handed out again first, or the ring full), %llu changed "
             "under a queued draw%s%s",
             v(g_handouts), v(g_outside), static_cast<ull>(live_max), v(g_violations_order), v(g_samples_checked), v(g_samples_skipped),
             v(g_violations_bytes), g_moved.load() == 1 ? " (one binding in 1024, its first 256 bytes: the heap is video memory)" : "",
             g_in_place.load() ? "" : " - in place stopped at the first");
    static const char* const kNoteName[kArenaNotes] = {"constant-buffer snapshots", "vertex snapshots", "GPU writes pending",
                                                       "descriptor-table reads"};
    std::string notes;
    for (int n = 0; n < kArenaNotes; ++n) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu (%llu MiB)", n ? ", " : "", kNoteName[n], v(g_noted[n]), v(g_noted_bytes[n]) >> 20);
        notes += b;
    }
    host_log("gxvk arena: today's path over the heap: %s", notes.c_str());
}

}  // namespace gxvk
