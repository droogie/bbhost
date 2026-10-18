// Host command lists (host/host_lists.h).
#include "host/host_lists.h"

#include "core/portable.h"
#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace host_lists {

const int g_mode = [] {
    const char* e = std::getenv("BBHOST_HOST_LISTS");
    return e && e[0] ? std::atoi(e) : 1;
}();

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using ull = unsigned long long;

// A chunk's list, found by the chunk's end - the command buffer object's
// +0x8, which stays put while its +0x0 moves to each segment's start (GX moves
// it at every segment it submits), so all of a chunk's segments share one
// list. Entries are the packet's place in dwords from 64 MiB below the end
// (kMaxChunkBytes, the place's 24 bits) << 8 | its opcode, in the order
// written - which is the chunk's order, as the write pointer only moves
// forward through a chunk. A packet written at or before the last one
// recorded is the first of the chunk written over again (a command buffer
// reset in place, without a new chunk): every packet before it in this round
// would have come first, so all the list held is gone.
constexpr u64 kMaxChunkBytes = 64ull << 20;
constexpr u32 kOverflow = 1, kReady = 2, kDead = 4;
struct Region {
    u64 end = 0;                 // the key
    std::atomic<u64> lo{~0ull};  // the lowest segment start seen: as far down as the chunk is known
    u32* entry = nullptr;
    u32 cap = 0;
    std::atomic<u32> count{0};
    std::atomic<u32> flags{0};
};
u64 base_of(u64 end) { return end > kMaxChunkBytes ? end - kMaxChunkBytes : 0; }
constexpr u32 kRegions = 8192;
Region g_regions[kRegions];
std::atomic<u32> g_region_count{0};

// The chunk's end -> its region (index + 1; kNoRegion when it could not be
// made), open addressing.
constexpr u32 kSlots = 16384, kProbes = 64, kNoRegion = ~0u;
struct Slot {
    std::atomic<u64> key{0};
    std::atomic<u32> region{0};
};
Slot g_slots[kSlots];

// Every list's entries come out of one block, taken when the mode is on
// (before any of the game's code runs): a chunk's list holds one entry per
// four of its dwords (a token's size) - at least 4096 - and one that fills up
// stops recording until the chunk starts over: its ranges are not covered.
constexpr u64 kPoolEntries = 16ull << 20;  // 64 MiB
u32* const g_pool = g_mode ? static_cast<u32*>(host_page_alloc(kPoolEntries * 4, false)) : nullptr;
std::atomic<u64> g_pool_used{0};

std::atomic<u64> g_unkeyed{0}, g_unrecorded{0}, g_overflowed{0}, g_stale{0};

// GX's command arenas, as the arena's acquire meets them (note_arena): a
// packet in one belongs to the chunk the arena's geometry puts it in - its
// command buffer's +0x0 moves to each segment's start and its +0x8 down by
// each label GX carves from the chunk's end, so neither names the chunk.
struct Arena {
    u64 base = 0, end = 0, chunk = 0;
};
constexpr u32 kArenas = 8;
Arena g_arenas[kArenas];
std::atomic<u32> g_arena_count{0};
std::mutex g_arena_mu;

u32 slot_of(u64 key) { return static_cast<u32>(((key >> 3) * 0x9E3779B97F4A7C15ull) >> 50) & (kSlots - 1); }

// Leaf-safe: a new chunk's region, or kNoRegion when there is no room left.
u32 make_region(u64 end, u64 begin) {
    const u32 i = g_region_count.fetch_add(1, std::memory_order_acq_rel);
    if (i >= kRegions) return kNoRegion;
    Region& r = g_regions[i];
    const u64 cap = std::min<u64>(std::max<u64>((end - begin) / 16 + 64, 4096), 1u << 20);
    const u64 at = g_pool ? g_pool_used.load(std::memory_order_relaxed) : kPoolEntries;
    r.end = end;
    r.lo.store(begin, std::memory_order_relaxed);
    if (at + cap > kPoolEntries || g_pool_used.fetch_add(cap, std::memory_order_relaxed) + cap > kPoolEntries) {
        r.flags.store(kReady | kDead, std::memory_order_release);
        return kNoRegion;
    }
    r.entry = g_pool + (g_pool_used.load(std::memory_order_relaxed) - cap);
    r.cap = static_cast<u32>(cap);
    r.flags.store(kReady, std::memory_order_release);
    return i + 1;
}

// Leaf-safe: the region of the chunk ending at `end`, made on its first packet.
Region* region_of(u64 end, u64 begin) {
    const u32 h = slot_of(end);
    for (u32 p = 0; p < kProbes; ++p) {
        Slot& s = g_slots[(h + p) & (kSlots - 1)];
        u64 k = s.key.load(std::memory_order_acquire);
        if (k == 0) {
            if (s.key.compare_exchange_strong(k, end, std::memory_order_acq_rel)) {
                const u32 r = make_region(end, begin);
                s.region.store(r, std::memory_order_release);
                return r == kNoRegion ? nullptr : &g_regions[r - 1];
            }
        }
        if (k != end) continue;
        u32 r;
        while ((r = s.region.load(std::memory_order_acquire)) == 0) {
            // Another thread is making it (a chunk's first two packets on
            // two threads: not one command buffer at a time).
        }
        if (r == kNoRegion) return nullptr;
        Region* region = &g_regions[r - 1];
        for (u64 lo = region->lo.load(std::memory_order_relaxed); begin < lo;) {
            if (region->lo.compare_exchange_weak(lo, begin, std::memory_order_relaxed)) break;
        }
        return region;
    }
    return nullptr;
}

// The submitting side's index of the regions by their end (hosted).
std::mutex g_index_mu;
std::map<u64, u32> g_index;
u32 g_indexed = 0;

void sync_index_locked() {
    const u32 n = std::min(g_region_count.load(std::memory_order_acquire), kRegions);
    while (g_indexed < n) {
        Region& r = g_regions[g_indexed];
        const u32 f = r.flags.load(std::memory_order_acquire);
        if (!(f & kReady)) break;  // still being made; the next take indexes it
        if (!(f & kDead)) g_index[r.end] = g_indexed;
        ++g_indexed;
    }
}

// The region whose chunk holds [va, va + bytes): the first ends at or past
// its end, the innermost that reaches down to va.
Region* containing(u64 va, std::size_t bytes) {
    std::lock_guard<std::mutex> lk(g_index_mu);
    sync_index_locked();
    auto it = g_index.lower_bound(va + bytes);
    for (int k = 0; k < 4 && it != g_index.end(); ++k, ++it) {
        Region& r = g_regions[it->second];
        if (r.lo.load(std::memory_order_relaxed) <= va && va >= base_of(r.end)) return &r;
    }
    return nullptr;
}

// What the command processor acts on (gnm_exec.cpp execute_packet), with
// every draw and dispatch a token and no register file kept: our tokens, the
// labels, the waits, predication, the indirect buffers, data written inline
// or moved by DMA, the occlusion counters' dumps, the constant engine, and
// draw and dispatch packets were there any. Not: data and markers (NOPs of no
// token), the register, base and index-state packets only packet draws and
// fallback dispatches read, and what it has no case for (cache actions,
// context control, and the like).
bool carried(u32 op, const u32* body, std::size_t n) {
    switch (op) {
        case 0x10:
            if (!n) return false;
            switch (body[0]) {
                case kGxNativeMagic: case kGxCopyMagic: case kGxFillMagic: case kGxUploadMagic: case kGxImageCopyMagic:
                case kGxClearMagic: case kGxCountCopyMagic: case kGxScaleformMagic: case kGxDispatchMagic: case kGxYebisMagic:
                case kGxBarrierMagic: case kGxQueryMagic:
                    return true;
                default: return false;
            }
        case 0x46:  // EVENT_WRITE: only the occlusion counters' dump (PixelPipeStatDump, index ZpassDone) does anything
            return n >= 1 && (body[0] & 0x3f) == 57 && ((body[0] >> 8) & 0xf) == 1;
        case 0x15: case 0x16:                        // DISPATCH_DIRECT, DISPATCH_INDIRECT
        case 0x20:                                   // SET_PREDICATION
        case 0x24: case 0x25: case 0x27: case 0x2d: case 0x35:  // the draw packets
        case 0x33: case 0x3f:                        // INDIRECT_BUFFER_CONST, INDIRECT_BUFFER
        case 0x37:                                   // WRITE_DATA
        case 0x3c:                                   // WAIT_REG_MEM
        case 0x47: case 0x48: case 0x49:             // EVENT_WRITE_EOP, _EOS, RELEASE_MEM
        case 0x50:                                   // DMA_DATA
        case 0x81: case 0x83: case 0x84: case 0x85: case 0x86: case 0x88:  // the constant engine
            return true;
        default: return false;
    }
}

// What the shadow found (shadow_*), by opcode where it says.
struct Counters {
    std::atomic<u64> ranges{0}, covered{0}, covered_empty{0}, no_list{0}, overflowed{0}, differed{0}, bytes{0}, bytes_covered{0};
    std::atomic<u64> nested{0}, nested_covered{0};
    std::atomic<u64> matched[256] = {}, walked_not_listed[256] = {}, listed_not_walked[256] = {}, listed_not_carried[256] = {};
};
Counters g;
std::atomic<int> g_logs{0};

struct Frame {
    std::vector<u32> own;  // a nested walk's entries, taken there
    const u32* e = nullptr;
    std::size_t count = 0, cursor = 0;
    u64 base = 0, guest = 0;
    u32 hi = 0;  // dwords
    bool bad = false, nested = false, no_list = false;
};
constexpr int kFrames = 6;
thread_local Frame t_frames[kFrames];
thread_local int t_depth = -1;

std::atomic<int> g_logs_no_list{0};
void differ(Frame& f, const char* what, u32 op, u32 at) {
    f.bad = true;
    std::atomic<int>& logs = f.no_list ? g_logs_no_list : g_logs;
    if (logs.fetch_add(1) < (f.no_list ? 12 : 24)) {
        host_log("host-lists (shadow): %s: opcode 0x%02x at +0x%x of the %s range 0x%llx (%u bytes); %s", what, op, at * 4,
                 f.nested ? "indirect" : "submitted", static_cast<ull>(f.guest), f.hi * 4,
                 f.no_list ? "no list holds it" : (std::string("its list has ") + std::to_string(f.count) + " entries" +
                                                   (f.cursor < f.count ? "" : ", none left")).c_str());
    }
}

}  // namespace

void note_slow(u64 cb, u64 va, u32 op) {
    u64 begin, end;
    std::memcpy(&begin, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cb)), 8);
    std::memcpy(&end, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cb + 8)), 8);
    if (va < begin || va >= end || end - begin > kMaxChunkBytes) {
        g_unkeyed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // The chunk: an arena's by its geometry, any other by its end.
    u64 key = end, lo = begin;
    for (u32 i = 0, n = g_arena_count.load(std::memory_order_acquire); i < n; ++i) {
        const Arena& a = g_arenas[i];
        if (va >= a.base && va < a.end) {
            lo = a.base + (va - a.base) / a.chunk * a.chunk;
            key = lo + a.chunk;
            break;
        }
    }
    Region* r = region_of(key, lo);
    if (!r) {
        g_unrecorded.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u32 at = static_cast<u32>((va - base_of(key)) >> 2);
    u32 n = r->count.load(std::memory_order_relaxed);
    if (n && (r->entry[n - 1] >> 8) >= at) {
        n = 0;
        r->flags.fetch_and(~kOverflow, std::memory_order_relaxed);
    }
    if (n >= r->cap) {
        if (!(r->flags.fetch_or(kOverflow, std::memory_order_relaxed) & kOverflow)) g_overflowed.fetch_add(1, std::memory_order_relaxed);
        r->count.store(n, std::memory_order_release);
        return;
    }
    r->entry[n] = (at << 8) | (op & 0xff);
    r->count.store(n + 1, std::memory_order_release);
}

void note_arena(u64 base, u64 chunk_bytes, u64 chunks) {
    if (!g_mode || !chunk_bytes || !chunks) return;
    std::lock_guard<std::mutex> lk(g_arena_mu);
    const u32 n = g_arena_count.load(std::memory_order_relaxed);
    for (u32 i = 0; i < n; ++i)
        if (g_arenas[i].base == base) return;
    if (n == kArenas) return;
    g_arenas[n] = Arena{base, base + chunk_bytes * chunks, chunk_bytes};
    g_arena_count.store(n + 1, std::memory_order_release);
}

void chunk_handed_out(u64 chunk_va, std::size_t chunk_bytes) {
    if (!g_mode) return;
    std::lock_guard<std::mutex> lk(g_index_mu);
    sync_index_locked();
    // The chunk's list: the region whose end lies inside the chunk.
    for (auto it = g_index.upper_bound(chunk_va); it != g_index.end() && it->first <= chunk_va + chunk_bytes; ++it) {
        Region& r = g_regions[it->second];
        r.flags.fetch_and(~kOverflow, std::memory_order_relaxed);
        r.count.store(0, std::memory_order_release);
        r.lo.store(chunk_va, std::memory_order_relaxed);
    }
}

void take(u64 va, std::size_t bytes, Taken* out) {
    out->entries.clear();
    out->state = Taken::kNoList;
    Region* r = containing(va, bytes);
    if (!r) return;
    if (r->flags.load(std::memory_order_acquire) & kOverflow) {
        out->state = Taken::kOverflowed;
        return;
    }
    const u32 n = r->count.load(std::memory_order_acquire);
    const u64 base = base_of(r->end);
    const u32 lo = static_cast<u32>((va - base) >> 2), hi = lo + static_cast<u32>(bytes >> 2);
    const u32* e = r->entry;
    std::size_t i = static_cast<std::size_t>(std::lower_bound(e, e + n, lo << 8) - e);
    for (; i < n && (e[i] >> 8) < hi; ++i) {
        // The packet still there - its header, and for a NOP a token's magic:
        // an entry left from a chunk's earlier round that its new round wrote
        // no such packet over is not.
        u32 head[2];
        std::memcpy(head, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(base + static_cast<u64>(e[i] >> 8) * 4)), 8);
        const u32 hdr = head[0];
        if ((hdr >> 30) != 3 || ((hdr >> 8) & 0xff) != (e[i] & 0xff) || !carried(e[i] & 0xff, head + 1, ((hdr >> 16) & 0x3fff) + 1)) {
            g_stale.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        out->entries.push_back((((e[i] >> 8) - lo) << 8) | (e[i] & 0xff));
    }
    out->state = Taken::kTaken;
}

namespace {

// The handoff's counts (hand_over), and why a range was walked instead.
enum HandRefusal { kHrNoList, kHrOverflowed, kHrPastEnd, kHrProof, kHrRefusals };
const char* const kHrName[kHrRefusals] = {"no list (a chunk with no packet recorded)", "a list that filled up", "a packet past the range's end",
                                          "the proof found a packet the list does not account for"};
std::atomic<u64> g_hand_ranges{0}, g_hand_bytes_in{0}, g_hand_bytes_out{0}, g_hand_refused[kHrRefusals] = {}, g_hand_proof_op[256] = {};

bool g_prove = [] {
    const char* e = std::getenv("BBHOST_HOST_LISTS_PROVE");
    return e && e[0] == '1';
}();

u32 dword_at(u64 va) {
    u32 v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 4);
    return v;
}

// The proof: the range's headers, walked as the command processor walks them
// (type-3 packets by their counts, anything else a dword), against the list:
// each packet it acts on listed at its place, nothing else listed.
bool proven(u64 va, std::size_t bytes, const Taken& t) {
    const std::size_t dwords = bytes / 4;
    std::size_t e = 0;
    for (std::size_t k = 0; k < dwords;) {
        const u32 hdr = dword_at(va + k * 4);
        if ((hdr >> 30) != 3) {
            ++k;
            continue;
        }
        const u32 op = (hdr >> 8) & 0xff;
        const std::size_t len = 2 + ((hdr >> 16) & 0x3fff);
        if (k + len > dwords) return false;
        u32 body[1] = {len > 1 ? dword_at(va + k * 4 + 4) : 0};
        const bool listed = e < t.entries.size() && (t.entries[e] >> 8) == k;
        if (carried(op, body, len - 1)) {
            if (!listed || (t.entries[e] & 0xff) != op) {
                g_hand_proof_op[op].fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            ++e;
        } else if (listed) {
            g_hand_proof_op[op].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        k += len;
    }
    return e == t.entries.size();
}

}  // namespace

bool hand_over(u64 va, std::size_t bytes, const Taken& t, std::vector<std::uint8_t>* out) {
    const auto refuse = [](HandRefusal why) {
        g_hand_refused[why].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (t.state == Taken::kNoList) return refuse(kHrNoList);
    if (t.state == Taken::kOverflowed) return refuse(kHrOverflowed);
    if (g_prove && !proven(va, bytes, t)) return refuse(kHrProof);
    out->clear();
    const u64 end = va + bytes;
    for (const u32 e : t.entries) {
        const u64 at = va + static_cast<u64>(e >> 8) * 4;
        const u32 hdr = dword_at(at);
        const std::size_t len = 4 * (2 + ((hdr >> 16) & 0x3fff));
        if (at + len > end) return refuse(kHrPastEnd);
        const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(at));
        out->insert(out->end(), p, p + len);
    }
    g_hand_ranges.fetch_add(1, std::memory_order_relaxed);
    g_hand_bytes_in.fetch_add(bytes, std::memory_order_relaxed);
    g_hand_bytes_out.fetch_add(out->size(), std::memory_order_relaxed);
    return true;
}

void shadow_begin(const Taken* taken, u64 walk_va, std::size_t bytes, u64 guest_va) {
    t_depth = 0;
    Frame& f = t_frames[0];
    f.own.clear();
    f.e = taken && taken->state == Taken::kTaken ? taken->entries.data() : nullptr;
    f.count = f.e ? taken->entries.size() : 0;
    f.cursor = 0;
    f.base = walk_va;
    f.guest = guest_va ? guest_va : walk_va;
    f.hi = static_cast<u32>(bytes >> 2);
    f.bad = false;
    f.nested = false;
    f.no_list = !taken || taken->state == Taken::kNoList;
    g.ranges.fetch_add(1, std::memory_order_relaxed);
    g.bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (taken && taken->state == Taken::kOverflowed) {
        g.overflowed.fetch_add(1, std::memory_order_relaxed);
        t_depth = -1;  // not compared: counted
    }
    // With no list it is compared with an empty one: covered if the walk
    // meets nothing it would have to carry.
}

void shadow_push(u64 va, std::size_t bytes) {
    if (t_depth < 0) return;
    if (t_depth + 1 >= kFrames) {
        t_frames[t_depth].bad = true;
        return;
    }
    Frame& f = t_frames[++t_depth];
    Taken t;
    take(va, bytes, &t);
    f.own = std::move(t.entries);
    f.e = f.own.data();
    f.count = f.own.size();
    f.cursor = 0;
    f.base = va;
    f.guest = va;
    f.hi = static_cast<u32>(bytes >> 2);
    f.bad = t.state == Taken::kOverflowed;
    f.no_list = t.state == Taken::kNoList;
    f.nested = true;
    g.nested.fetch_add(1, std::memory_order_relaxed);
}

void shadow_packet(u64 va, u32 op, const u32* body, std::size_t n) {
    if (t_depth < 0) return;
    Frame& f = t_frames[t_depth];
    const u32 at = static_cast<u32>((va - f.base) >> 2);
    while (f.cursor < f.count && (f.e[f.cursor] >> 8) < at) {
        g.listed_not_walked[f.e[f.cursor] & 0xff].fetch_add(1, std::memory_order_relaxed);
        differ(f, "listed, but not where the walk found a packet", f.e[f.cursor] & 0xff, f.e[f.cursor] >> 8);
        ++f.cursor;
    }
    const bool here = f.cursor < f.count && (f.e[f.cursor] >> 8) == at;
    if (carried(op & 0xff, body, n)) {
        if (here && (f.e[f.cursor] & 0xff) == (op & 0xff)) {
            g.matched[op & 0xff].fetch_add(1, std::memory_order_relaxed);
        } else {
            g.walked_not_listed[op & 0xff].fetch_add(1, std::memory_order_relaxed);
            differ(f, here ? "listed as another packet" : "walked, not listed", op & 0xff, at);
        }
    } else if (here) {
        g.listed_not_carried[op & 0xff].fetch_add(1, std::memory_order_relaxed);
        differ(f, "listed, but a packet nothing reads", op & 0xff, at);
    }
    if (here) ++f.cursor;
}

void shadow_pop() {
    if (t_depth <= 0) return;
    Frame& f = t_frames[t_depth];
    for (; f.cursor < f.count && (f.e[f.cursor] >> 8) < f.hi; ++f.cursor) {
        g.listed_not_walked[f.e[f.cursor] & 0xff].fetch_add(1, std::memory_order_relaxed);
        differ(f, "listed past the walk", f.e[f.cursor] & 0xff, f.e[f.cursor] >> 8);
    }
    if (!f.bad) g.nested_covered.fetch_add(1, std::memory_order_relaxed);
    const bool bad = f.bad;
    --t_depth;
    if (bad) t_frames[t_depth].bad = true;
}

void shadow_end() {
    if (t_depth < 0) return;
    while (t_depth > 0) shadow_pop();
    Frame& f = t_frames[0];
    for (; f.cursor < f.count && (f.e[f.cursor] >> 8) < f.hi; ++f.cursor) {
        g.listed_not_walked[f.e[f.cursor] & 0xff].fetch_add(1, std::memory_order_relaxed);
        differ(f, "listed past the walk", f.e[f.cursor] & 0xff, f.e[f.cursor] >> 8);
    }
    if (f.bad) {
        (f.no_list ? g.no_list : g.differed).fetch_add(1, std::memory_order_relaxed);
    } else {
        g.covered.fetch_add(1, std::memory_order_relaxed);
        if (f.no_list) g.covered_empty.fetch_add(1, std::memory_order_relaxed);
        g.bytes_covered.fetch_add(static_cast<u64>(f.hi) * 4, std::memory_order_relaxed);
    }
    t_depth = -1;
}

void report() {
    if (!g_mode) return;
    const u32 regions = std::min(g_region_count.load(), kRegions);
    host_log("host-lists: %u chunks' lists (%.1f MiB of entries set aside); packets not recorded: %llu outside their command "
             "buffer's chunk, %llu with no room for another list, %llu lists filled up; %llu entries left from a chunk's earlier round "
             "dropped when taken (BBHOST_HOST_LISTS=%d)",
             regions, static_cast<double>(g_pool_used.load()) * 4 / (1 << 20), static_cast<ull>(g_unkeyed.load()),
             static_cast<ull>(g_unrecorded.load()), static_cast<ull>(g_overflowed.load()), static_cast<ull>(g_stale.load()), g_mode);
    const auto by_op = [](const std::atomic<u64>* v) {
        std::string s;
        for (int op = 0; op < 256; ++op) {
            if (const u64 k = v[op].load()) {
                char b[32];
                std::snprintf(b, sizeof b, "%s 0x%02x %llu", s.empty() ? "" : ",", op, static_cast<ull>(k));
                s += b;
            }
        }
        return s.empty() ? std::string(" none") : s;
    };
    if (g_mode == 1) {
        std::string why;
        for (int k = 0; k < kHrRefusals; ++k) {
            if (const u64 v = g_hand_refused[k].load()) why += std::string(why.empty() ? " " : "; ") + kHrName[k] + " " + std::to_string(v);
        }
        host_log("host-lists (handoff): %llu submitted ranges went to the command processor as their lists' packets (%.1f MiB of command "
                 "buffer as %.1f MiB); walked as before:%s%s%s",
                 static_cast<ull>(g_hand_ranges.load()), static_cast<double>(g_hand_bytes_in.load()) / (1 << 20),
                 static_cast<double>(g_hand_bytes_out.load()) / (1 << 20), why.empty() ? " none" : why.c_str(),
                 g_prove ? "; the packets the proof refused ranges by:" : "", g_prove ? by_op(g_hand_proof_op).c_str() : "");
        return;
    }
    if (g_mode != 2) return;
    host_log("host-lists (shadow): %llu submitted ranges (%.1f MiB) walked, %llu covered by their lists (%.1f MiB; %llu of them with no "
             "list and nothing to carry); not: %llu with no list and something to carry, %llu over a full list, %llu where the list and the "
             "walk differ; indirect buffers inside them %llu, %llu covered",
             static_cast<ull>(g.ranges.load()), static_cast<double>(g.bytes.load()) / (1 << 20), static_cast<ull>(g.covered.load()),
             static_cast<double>(g.bytes_covered.load()) / (1 << 20), static_cast<ull>(g.covered_empty.load()), static_cast<ull>(g.no_list.load()),
             static_cast<ull>(g.overflowed.load()), static_cast<ull>(g.differed.load()), static_cast<ull>(g.nested.load()),
             static_cast<ull>(g.nested_covered.load()));
    host_log("host-lists (shadow): met in place and in order, by opcode:%s", by_op(g.matched).c_str());
    host_log("host-lists (shadow): walked but not listed:%s; listed but not walked:%s; listed but nothing reads them:%s",
             by_op(g.walked_not_listed).c_str(), by_op(g.listed_not_walked).c_str(), by_op(g.listed_not_carried).c_str());
}

}  // namespace host_lists
