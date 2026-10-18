// BBHOST_CB_RECORD_PROOF (host/cb_proof.h): copies taken at the GX call,
// compared where the command processor binds the buffer.
#include "host/cb_proof.h"

#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

namespace cb_proof {

const bool g_on = [] {
    const char* e = std::getenv("BBHOST_CB_RECORD_PROOF");
    return e && e[0] == '1';
}();

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// A copy in a ring: this header, then the bytes, to a 16-byte boundary.
struct Header {
    u64 addr;
    u32 bytes;
    u8 kind;
    u8 pad[3];
    u64 flip;
    u64 pos;  // where it was written, in the ring's running count of bytes
    u64 completed;
    u64 object;
};
static_assert(sizeof(Header) == 48);
constexpr u64 kMaxEntry = sizeof(Header) + kMaxCopy;

u64 g_ring_bytes = 32ull << 20;
constexpr int kMaxRings = 64;

// One recording thread's ring. `reserved` is the running count up to the end
// of the copy being written; it moves before the bytes are written, so a
// reader that finished comparing and then finds a copy at least a ring's
// length minus one entry behind it read bytes no later copy had reached.
struct Ring {
    std::atomic<u64> reserved{0};
    u8* mem = nullptr;
    u64 id = 0;
    u64 copies = 0, bytes = 0, cycles = 0, truncated = 0;  // the recording thread's own
};
Ring* g_rings[kMaxRings] = {};
std::atomic<int> g_nrings{0};
std::atomic<u64> g_no_ring{0};
thread_local Ring* t_ring = nullptr;
thread_local bool t_ring_failed = false;

Ring* ring_for_thread() {
    if (t_ring) return t_ring;
    if (t_ring_failed) return nullptr;
    const int id = g_nrings.fetch_add(1, std::memory_order_relaxed);
    if (id >= kMaxRings) {
        t_ring_failed = true;
        return nullptr;
    }
    Ring* r = new Ring;
    r->mem = static_cast<u8*>(::operator new(g_ring_bytes, std::align_val_t{64}));
    r->id = static_cast<u64>(id);
    g_rings[id] = r;  // read by the command processor only through handles made after this
    t_ring = r;
    if (id == 0)
        host_log("cb-proof: on (BBHOST_CB_RECORD_PROOF=1): every GX draw's constant buffers copied at the call, %llu MiB a "
                 "recording thread, and compared where the command processor binds them",
                 static_cast<unsigned long long>(g_ring_bytes >> 20));
    return r;
}

u64 rdtsc() {
#if defined(__x86_64__) || defined(_M_X64)
    return __rdtsc();
#else
    return 0;
#endif
}

// The counts, written by the command processor's threads.
std::atomic<u64> g_outcomes[kPaths][kOutcomes] = {};
std::atomic<u64> g_outcome_bytes[kPaths][kOutcomes] = {};
std::atomic<u64> g_uncopied[kPaths][3] = {};  // [0] no records, [1] no copy for the slot, [2] a Scaleform draw's, no records
// Same / changed bindings by whether the rule leaves them to today's copy (a
// GPU write not complete at the call noted on the range since, or the tables
// having let go of writes that recent): [path][0 same, 1 changed][0 kept, 1 left],
// and of those left, the ones left for the tables alone.
std::atomic<u64> g_rule[kPaths][2][2] = {};
std::atomic<u64> g_rule_dropped[kPaths][2] = {};
// Changed bindings by path, kind and how many flips after the call (0, 1, 2, 3+).
std::atomic<u64> g_changed[kPaths][kKinds][4] = {};
std::atomic<u64> g_changed_gpu_any[kPaths] = {};  // of them, a GPU write noted on the range at all
std::atomic<u64> g_changed_dwords[kPaths] = {};

struct PipelineChanges {
    std::string name;
    u64 changed = 0, uncaught = 0, by_kind[kKinds] = {}, lag0 = 0;
};
std::mutex g_pipelines_mu;
std::map<std::string, PipelineChanges>* const g_pipelines = new std::map<std::string, PipelineChanges>;  // under g_pipelines_mu
std::atomic<int> g_samples_uncaught{0}, g_samples_caught{0};

const char* const kKindName[kKinds] = {"static", "renamed", "dynamic at home", "other"};
const char* const kPathName[kPaths] = {"snapshot", "realigned", "gpu-pending", "copy-back", "window", "stored", "in place"};
const char* const kOutcomeName[kOutcomes] = {"same", "changed", "not covered", "lapped", "no copy"};

}  // namespace

void set_ring_bytes(std::uint64_t bytes) {
    g_ring_bytes = std::max<u64>(bytes, 4 * kMaxEntry);
}

std::uint64_t record(std::uint64_t addr, std::uint32_t bytes, Kind kind, std::uint64_t flip, std::uint64_t completed,
                     std::uint64_t object) {
    Ring* r = ring_for_thread();
    if (!r) {
        g_no_ring.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const u64 t0 = rdtsc();
    if (bytes > kMaxCopy) {
        bytes = kMaxCopy;
        ++r->truncated;
    }
    const u64 need = (sizeof(Header) + bytes + 15) & ~u64{15};
    u64 pos = r->reserved.load(std::memory_order_relaxed);
    // A copy never wraps: one that would run past the ring's end starts the
    // next round instead.
    if (pos % g_ring_bytes + need > g_ring_bytes) pos += g_ring_bytes - pos % g_ring_bytes;
    r->reserved.store(pos + need, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // the reservation before the bytes it covers
    u8* p = r->mem + pos % g_ring_bytes;
    Header h{addr, bytes, static_cast<u8>(kind), {}, flip, pos, completed, object};
    std::memcpy(p, &h, sizeof(h));
    std::memcpy(p + sizeof(h), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(addr)), bytes);
    ++r->copies;
    r->bytes += bytes;
    r->cycles += rdtsc() - t0;
    return ((r->id + 1) << 56) | pos;
}

Outcome check(std::uint64_t handle, std::uint64_t base, std::uint64_t bytes, Path path, Difference* diff) {
    const auto count = [&](Outcome o) {
        g_outcomes[path][o].fetch_add(1, std::memory_order_relaxed);
        g_outcome_bytes[path][o].fetch_add(bytes, std::memory_order_relaxed);
        return o;
    };
    const u64 id = handle >> 56;
    if (!id || id > static_cast<u64>(kMaxRings) || !g_rings[id - 1]) return count(kNoCopy);
    const Ring* r = g_rings[id - 1];
    const u64 pos = handle & ((u64{1} << 56) - 1);
    // Intact while the writer has reserved no further than a ring's length
    // past the copy (less an entry, to spare): read after the bytes.
    const auto intact = [&] {
        std::atomic_thread_fence(std::memory_order_seq_cst);  // what was read, before the reservation is
        const u64 reserved = r->reserved.load(std::memory_order_relaxed);
        return reserved + kMaxEntry <= pos + g_ring_bytes;
    };
    if (!intact()) return count(kLapped);
    const u8* p = r->mem + pos % g_ring_bytes;
    Header h;
    std::memcpy(&h, p, sizeof(h));
    if (h.pos != pos || !intact()) return count(kLapped);
    if (base < h.addr || base - h.addr + bytes > h.bytes) return count(kNotCovered);
    Difference d;
    d.kind = h.kind < kKinds ? static_cast<Kind>(h.kind) : kOtherObject;
    d.addr = h.addr;
    d.copied = h.bytes;
    d.flip = h.flip;
    d.completed = h.completed;
    d.object = h.object;
    const u8* was = p + sizeof(Header) + (base - h.addr);
    const u8* now = reinterpret_cast<const u8*>(static_cast<std::uintptr_t>(base));
    if (std::memcmp(was, now, bytes) == 0) {
        if (!intact()) return count(kLapped);
        if (diff) *diff = d;
        return count(kSame);
    }
    // Which dwords differ, and the first few from the first that does.
    u32 first = ~0u, differing = 0;
    for (u64 off = 0; off + 4 <= bytes; off += 4) {
        if (std::memcmp(was + off, now + off, 4) == 0) continue;
        if (first == ~0u) first = static_cast<u32>(off / 4);
        ++differing;
    }
    if (first == ~0u) first = 0;  // only a tail shorter than a dword
    d.first_dword = first;
    d.dwords = differing;
    const u64 left = bytes > first * 4ull ? std::min<u64>(16, bytes - first * 4ull) : 0;
    std::memcpy(d.was, was + first * 4ull, left);
    std::memcpy(d.now, now + first * 4ull, left);
    if (!intact()) return count(kLapped);
    if (diff) *diff = d;
    return count(kChanged);
}

void note_result(const Difference& d, Outcome outcome, Path path, const char* pipeline, std::uint64_t flip_now, bool gpu_since_call,
                 bool dropped_since, bool gpu_any, std::uint64_t base, std::uint64_t bytes) {
    if (outcome != kSame && outcome != kChanged) return;
    const bool changed = outcome == kChanged;
    const bool left = gpu_since_call || dropped_since;
    g_rule[path][changed][left].fetch_add(1, std::memory_order_relaxed);
    if (!gpu_since_call && dropped_since) g_rule_dropped[path][changed].fetch_add(1, std::memory_order_relaxed);
    if (!changed) return;
    const u64 lag = flip_now >= d.flip ? flip_now - d.flip : 0;
    g_changed[path][d.kind][std::min<u64>(lag, 3)].fetch_add(1, std::memory_order_relaxed);
    if (gpu_any) g_changed_gpu_any[path].fetch_add(1, std::memory_order_relaxed);
    g_changed_dwords[path].fetch_add(d.dwords, std::memory_order_relaxed);
    // The ones that matter most: a copied binding that changed with nothing
    // to send it to today's copy.
    const bool uncaught = copied(path) && !left;
    const std::string name = pipeline ? pipeline : "?";
    {
        std::lock_guard<std::mutex> lk(g_pipelines_mu);
        if (g_pipelines->size() < 4096 || g_pipelines->count(name)) {
            PipelineChanges& p = (*g_pipelines)[name];
            ++p.changed;
            p.uncaught += uncaught;
            ++p.by_kind[d.kind];
            p.lag0 += lag == 0;
        }
    }
    std::atomic<int>& samples = uncaught ? g_samples_uncaught : g_samples_caught;
    if (samples.fetch_add(1, std::memory_order_relaxed) < (uncaught ? 32 : 8)) {
        float fw[4], fn[4];
        std::memcpy(fw, d.was, sizeof(fw));
        std::memcpy(fn, d.now, sizeof(fn));
        host_log("cb-proof: changed since the call (%s, %s%s): pipeline %s, 0x%llx +%llu bytes (copied 0x%llx, %u bytes; object 0x%llx) "
                 "at flip %llu, bound at flip %llu%s; %u dwords differ from dword %u: %08x %08x %08x %08x (%g %g %g %g) -> "
                 "%08x %08x %08x %08x (%g %g %g %g)",
                 kPathName[path], kKindName[d.kind], uncaught ? ", no GPU write since the call" : "", name.c_str(),
                 static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(d.addr),
                 d.copied, static_cast<unsigned long long>(d.object), static_cast<unsigned long long>(d.flip),
                 static_cast<unsigned long long>(flip_now),
                 gpu_since_call  ? ", a GPU write since the call"
                 : dropped_since ? ", the write tables let go of writes since the call"
                 : gpu_any       ? ", a GPU write noted before the call"
                                 : "",
                 d.dwords,
                 d.first_dword, d.was[0], d.was[1], d.was[2], d.was[3], fw[0], fw[1], fw[2], fw[3], d.now[0], d.now[1], d.now[2],
                 d.now[3], fn[0], fn[1], fn[2], fn[3]);
    }
}

void note_uncopied(Path path, bool records, bool scaleform) {
    g_uncopied[path][records ? 1 : scaleform ? 2 : 0].fetch_add(1, std::memory_order_relaxed);
}

void report() {
    if (!g_on) return;
    u64 copies = 0, bytes = 0, cycles = 0, truncated = 0;
    int rings = 0;
    for (const Ring* r : g_rings) {
        if (!r) continue;
        ++rings;
        copies += r->copies;
        bytes += r->bytes;
        cycles += r->cycles;
        truncated += r->truncated;
    }
    host_log("cb-proof: copied at the GX call: %llu buffers, %llu MiB, %llu Mcycles on %d recording threads (%.0f cycles a copy); "
             "%llu cut at %u KiB, %llu without a ring",
             static_cast<unsigned long long>(copies), static_cast<unsigned long long>(bytes >> 20),
             static_cast<unsigned long long>(cycles / 1000000), rings, copies ? static_cast<double>(cycles) / static_cast<double>(copies) : 0.0,
             static_cast<unsigned long long>(truncated), kMaxCopy >> 10, static_cast<unsigned long long>(g_no_ring.load()));
    for (int p = 0; p < kPaths; ++p) {
        u64 total = 0;
        for (int o = 0; o < kOutcomes; ++o) total += g_outcomes[p][o].load();
        const u64 unc = g_uncopied[p][0].load() + g_uncopied[p][1].load() + g_uncopied[p][2].load();
        if (!total && !unc) continue;
        std::string line;
        for (int o = 0; o < kOutcomes; ++o) {
            char part[96];
            std::snprintf(part, sizeof(part), "%s%s %llu (%llu MiB)", o ? ", " : "", kOutcomeName[o],
                          static_cast<unsigned long long>(g_outcomes[p][o].load()),
                          static_cast<unsigned long long>(g_outcome_bytes[p][o].load() >> 20));
            line += part;
        }
        host_log("cb-proof: bound %s: %llu with a copy from the call: %s; no copy: %llu with no records (%llu of them a Scaleform "
                 "draw's), %llu whose slot had none",
                 kPathName[p], static_cast<unsigned long long>(total), line.c_str(),
                 static_cast<unsigned long long>(g_uncopied[p][0].load() + g_uncopied[p][2].load()),
                 static_cast<unsigned long long>(g_uncopied[p][2].load()), static_cast<unsigned long long>(g_uncopied[p][1].load()));
    }
    // The rule a copy at the call needs: a copied binding with a GPU write
    // not complete at the call noted on its range since takes today's copy.
    for (int p = 0; p < kPaths; ++p) {
        if (!copied(static_cast<Path>(p))) continue;
        const u64 same_kept = g_rule[p][0][0].load(), same_left = g_rule[p][0][1].load();
        const u64 changed_left = g_rule[p][1][1].load(), uncaught = g_rule[p][1][0].load();
        if (!same_kept && !same_left && !changed_left && !uncaught) continue;
        host_log("cb-proof: bound %s, with the rule (a GPU write since the call, or tables that let go of one, leaves the binding to "
                 "today's copy): %llu would bind the call's copy, the same bytes; left to today's copy %llu that had the same bytes "
                 "(%llu for the tables alone) and %llu that had changed (%llu for the tables alone); %llu changed with nothing to "
                 "leave them (a copy at the call would have bound stale bytes)",
                 kPathName[p], static_cast<unsigned long long>(same_kept), static_cast<unsigned long long>(same_left),
                 static_cast<unsigned long long>(g_rule_dropped[p][0].load()), static_cast<unsigned long long>(changed_left),
                 static_cast<unsigned long long>(g_rule_dropped[p][1].load()), static_cast<unsigned long long>(uncaught));
    }
    for (int p = 0; p < kPaths; ++p) {
        if (!g_outcomes[p][kChanged].load()) continue;
        std::string line;
        for (int k = 0; k < kKinds; ++k) {
            const u64 n = g_changed[p][k][0].load() + g_changed[p][k][1].load() + g_changed[p][k][2].load() + g_changed[p][k][3].load();
            if (!n) continue;
            char part[160];
            std::snprintf(part, sizeof(part), "%s%s %llu (bound 0/1/2/3+ flips after the call: %llu/%llu/%llu/%llu)", line.empty() ? "" : "; ",
                          kKindName[k], static_cast<unsigned long long>(n), static_cast<unsigned long long>(g_changed[p][k][0].load()),
                          static_cast<unsigned long long>(g_changed[p][k][1].load()),
                          static_cast<unsigned long long>(g_changed[p][k][2].load()),
                          static_cast<unsigned long long>(g_changed[p][k][3].load()));
            line += part;
        }
        host_log("cb-proof: changed since the call, bound %s: %s; %llu with a GPU write noted on the range at all; %llu dwords differed",
                 kPathName[p], line.empty() ? "none described" : line.c_str(), static_cast<unsigned long long>(g_changed_gpu_any[p].load()),
                 static_cast<unsigned long long>(g_changed_dwords[p].load()));
    }
    std::vector<PipelineChanges> top;
    {
        std::lock_guard<std::mutex> lk(g_pipelines_mu);
        for (const auto& [name, c] : *g_pipelines) {
            top.push_back(c);
            top.back().name = name;
        }
    }
    std::sort(top.begin(), top.end(), [](const PipelineChanges& a, const PipelineChanges& b) {
        return a.uncaught != b.uncaught ? a.uncaught > b.uncaught : a.changed > b.changed;
    });
    for (std::size_t i = 0; i < top.size() && i < 16; ++i) {
        const PipelineChanges& c = top[i];
        host_log("cb-proof:   pipeline %s: %llu changed, %llu of them copied with nothing to leave them (static %llu, renamed %llu, "
                 "dynamic at home %llu, other %llu; bound in the call's flip %llu)",
                 c.name.c_str(), static_cast<unsigned long long>(c.changed), static_cast<unsigned long long>(c.uncaught),
                 static_cast<unsigned long long>(c.by_kind[kStatic]), static_cast<unsigned long long>(c.by_kind[kRenamed]),
                 static_cast<unsigned long long>(c.by_kind[kDynamicOwn]), static_cast<unsigned long long>(c.by_kind[kOtherObject]),
                 static_cast<unsigned long long>(c.lag0));
    }
    if (top.size() > 16) host_log("cb-proof:   ... %zu pipelines with a change in all", top.size());
}

}  // namespace cb_proof
