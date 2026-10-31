// GX on Vulkan, stage 1's census (host/gxvk.h): nothing it does changes what
// the game or the renderer does, but the canaries it writes into bytes GX
// leaves unwritten.
#include "host/gxvk.h"

#include "core/portable.h"
#include "engine/gx_resources.h"
#include "host/gpu.h"
#include "host/label_landing.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gxvk {
namespace {

using ull = unsigned long long;

struct Flags {
    bool given = false;  // BBHOST_GXVK_BUFFERS set: its mode on any device
    Buffers mode = Buffers::kOff;
    bool census = false;
};
const Flags g_flags = [] {
    Flags f;
    if (const char* e = std::getenv("BBHOST_GXVK_BUFFERS"); e && *e) {
        f.given = true;
        f.mode = buffers_mode_for(e, BuffersDevice::kRoomy);
        if (std::strcmp(e, "census") == 0) f.census = true;
    }
    if (const char* e = std::getenv("BBHOST_GXVK_CENSUS"); e && e[0] == '1') f.census = true;
    return f;
}();

template <typename T>
bool rd(std::uint64_t va, T* out) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), out, sizeof(T));
}
void wr32(std::uint64_t va, std::uint32_t v) { std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va)), &v, 4); }

// The class word the census keeps in a buffer holder's +0x44, under a magic
// that says the census wrote it: the creation's usage, initial data, CPU
// access and bind flags, then what the buffer was seen to take since.
constexpr std::uint32_t kMagic = 0xc0de0000u, kMagicMask = 0xffff0000u;
constexpr std::uint32_t kUsage = 0x3, kData = 1u << 2, kCpuShift = 3, kBindShift = 5;
constexpr std::uint32_t kMappedOwn = 1u << 13;  // a Map of type 1 or 2 handed out its own memory
constexpr std::uint32_t kRenamed = 1u << 14;    // a Map of type 3 or 4 (a renamed copy)
constexpr std::uint32_t kUpdated = 1u << 15;    // UpdateSubresource
constexpr std::uint32_t kBindUav = 0x80, kBindCb = 0x4;
// The canary in the other spare dword: the object's own, so a stale copy of
// another object's bytes does not pass.
std::uint32_t canary(std::uint64_t object, std::uint32_t salt) {
    return (0x5a5a0000u ^ salt) ^ static_cast<std::uint32_t>((object >> 4) & 0xffff);
}

// What holds the bytes a binding reads.
enum Class : int {
    kImmutable,      // default or immutable usage, made with data, no CPU access, not unordered-access, never mapped or updated
    kDefaultCopied,  // default usage, no data or updated by UpdateSubresource: GX writes it with GPU copies
    kDynamicHome,    // dynamic, bound at its own memory (what the map resolves copy back)
    kRenameInHeap,   // a renamed copy in the upload heap
    kRenameElse,     // a renamed copy outside it
    kUav,            // unordered-access bind: the GPU writes it
    kCpuShared,      // a Map of type 1/2 handed out its own memory: the CPU writes it in place
    kStaging,        // staging usage
    kClobbered,      // a holder the census tagged whose word is not its own any more
    kUntagged,       // a holder the census never saw made
    kNoObject,       // no GX holder: Scaleform, YEBIS, a table the game binds itself
    kClasses
};
const char* const kClassName[kClasses] = {"immutable",      "default (GPU-copied)", "dynamic at home", "renamed, in the upload heap",
                                          "renamed, elsewhere", "unordered-access", "CPU-mapped own memory", "staging",
                                          "tag clobbered",  "never tagged",         "no GX object"};
const char* const kUseName[kUses] = {"vertex records", "index buffers", "constant buffers", "unordered-access buffers"};

std::atomic<std::uint64_t> g_bindings[kUses][kClasses] = {};
std::atomic<std::uint64_t> g_binding_bytes[kUses][kClasses] = {};
std::atomic<int> g_odd_logged{0};

// Buffers made, by (usage, data, CPU access, bind) - creation is rare.
std::mutex g_kinds_mu;
struct KindCount {
    std::uint64_t n = 0, bytes = 0;
};
std::map<std::uint32_t, KindCount> g_kinds;  // under g_kinds_mu
std::atomic<std::uint64_t> g_buffers_made{0}, g_textures_made{0}, g_views_made[2] = {};
std::atomic<std::uint64_t> g_maps[5] = {}, g_maps_own[5] = {}, g_maps_heap[5] = {}, g_maps_failed[5] = {};
std::atomic<std::uint64_t> g_updates_cb{0}, g_updates_other{0}, g_updates_untagged{0};
// Canaries: [buffer +0x14, buffer +0x44, texture +0x14, view] x [checked, broken].
enum Spare : int { kSpareBufA, kSpareBufB, kSpareTex, kSpareSrv, kSpareUav, kSpares };
const char* const kSpareName[kSpares] = {"buffer holder +0x14", "buffer holder +0x44", "texture holder +0x14", "shader-resource view +0x48",
                                         "unordered-access view +0x44"};
std::atomic<std::uint64_t> g_spare_checked[kSpares] = {}, g_spare_broken[kSpares] = {};
std::atomic<int> g_spare_logged{0};
std::atomic<std::uint64_t> g_gone_buffers{0}, g_gone_textures{0};

// The upload heap: the first block pool's region. Two stamps per 16-byte
// granule (the pool's alignment, so no two hand-outs share one): the last
// hand-out's sequence, and the last submission that bound it (+1) with what
// bound it (bound_stamp: the use in the low two bits).
struct Heap {
    std::uint64_t pool = 0, base = 0, bytes = 0;
    std::atomic<std::uint32_t>* handed = nullptr;
    std::atomic<std::uint32_t>* bound = nullptr;
    std::size_t granules = 0;
};
Heap g_heap;
std::atomic<bool> g_heap_ready{false};
std::atomic<std::uint64_t> g_heap_pools{0};
std::atomic<std::uint32_t> g_seq{0};
std::atomic<std::uint64_t> g_handouts{0}, g_handout_bytes{0}, g_handouts_other_pool{0}, g_handouts_outside{0};
std::atomic<std::uint64_t> g_min_free{~0ull}, g_max_in_use{0};
std::atomic<std::uint64_t> g_ring_allocs{0}, g_ring_in_heap{0};
// The release proof, by what bound the range (Use): (i) hand-outs over a
// granule a submission still running had bound, and those within one
// submission of the count (a count that lagged rather than a reuse); (ii)
// bindings of a range handed out again after their call; and the heap
// bindings, and those that carried no call stamp. A vertex record's range is
// what its draw reads (to its last vertex), not its V#'s extent, which
// counts records from the buffer's start and runs on past a renamed copy.
std::atomic<std::uint64_t> g_proof_i[kUses] = {}, g_proof_i_near[kUses] = {}, g_proof_ii[kUses] = {}, g_heap_bindings[kUses] = {},
                           g_heap_unstamped[kUses] = {};
std::atomic<std::uint64_t> g_heap_granules_checked{0};
std::atomic<int> g_proof_logged{0};

constexpr std::uint64_t kGranule = 16;
inline std::uint32_t bound_stamp(std::uint64_t serial, Use use) { return static_cast<std::uint32_t>(((serial + 1) << 2) | static_cast<unsigned>(use)); }
inline std::uint64_t stamp_serial(std::uint32_t stamp) { return (stamp >> 2) - 1; }  // the submission (stamp != 0)
inline Use stamp_use(std::uint32_t stamp) { return static_cast<Use>(stamp & 3); }

bool in_heap(std::uint64_t va, std::uint64_t bytes) {
    return g_heap_ready.load(std::memory_order_acquire) && va >= g_heap.base && va - g_heap.base < g_heap.bytes &&
           bytes <= g_heap.base + g_heap.bytes - va;
}

// The submissions whose work is done: the landing's count or the retirements',
// whichever is further. The landing's count raised before their labels are in
// memory (landing(), not landed()): GX hands a range out again as soon as it
// reads its release label, which can be before landed() counts the submission.
std::uint64_t done_submissions() {
    return std::max<std::uint64_t>(label_landing::landing(), host_gpu_completed_submits_seen());
}

// The class of a binding at `va` of a holder's buffer, from its class word.
Class class_of(std::uint64_t holder, std::uint64_t va, std::uint64_t bytes, Use use) {
    if (!holder) return kNoObject;
    std::uint32_t word = 0, a = 0;
    if (!rd(holder + kHolderSpareB, &word)) return kNoObject;
    g_spare_checked[kSpareBufB].fetch_add(1, std::memory_order_relaxed);
    if ((word & kMagicMask) != kMagic) {
        GxResourceInfo info;
        if (gx_resource_by_holder(holder, &info)) {
            g_spare_broken[kSpareBufB].fetch_add(1, std::memory_order_relaxed);
            if (g_spare_logged.fetch_add(1, std::memory_order_relaxed) < 8)
                host_log("gxvk census: buffer holder 0x%llx (resource %u, %u bytes) has 0x%08x at +0x44, not its class word", static_cast<ull>(holder),
                         info.id, info.bytes, word);
            return kClobbered;
        }
        return kUntagged;
    }
    // (With GX buffers by handle, +0x14 holds the handle: no canary there.)
    if (buffers_mode() == Buffers::kOff && rd(holder + kHolderSpareA, &a)) {
        g_spare_checked[kSpareBufA].fetch_add(1, std::memory_order_relaxed);
        if (a != canary(holder, 0x14)) {
            g_spare_broken[kSpareBufA].fetch_add(1, std::memory_order_relaxed);
            if (g_spare_logged.fetch_add(1, std::memory_order_relaxed) < 8)
                host_log("gxvk census: buffer holder 0x%llx has 0x%08x at +0x14, not its canary", static_cast<ull>(holder), a);
        }
    }
    std::uint64_t home = 0;
    std::uint32_t size = 0;
    rd(holder + 0x20, &home);
    rd(holder + 0x2c, &size);
    const bool at_home = va >= home && va - home < std::max<std::uint32_t>(size, 1);
    const std::uint32_t usage = word & kUsage, bind = (word >> kBindShift) & 0xff;
    if (!at_home) {
        const Class c = in_heap(va, bytes) ? kRenameInHeap : kRenameElse;
        if (c == kRenameElse && g_odd_logged.fetch_add(1, std::memory_order_relaxed) < 6)
            host_log("gxvk census: a %s binding at 0x%llx (+%llu) of buffer 0x%llx (home 0x%llx, %u bytes, word 0x%08x) is neither its own "
                     "memory nor the upload heap",
                     kUseName[use], static_cast<ull>(va), static_cast<ull>(bytes), static_cast<ull>(holder), static_cast<ull>(home), size, word);
        return c;
    }
    if (word & kMappedOwn) return kCpuShared;
    if (usage == 3) return kStaging;
    if (bind & kBindUav) return kUav;
    if (usage == 2) return kDynamicHome;
    if ((word & kData) && !(word & (kUpdated | kRenamed)) && ((word >> kCpuShift) & 3) == 0) return kImmutable;
    return kDefaultCopied;
}

void count(Use use, Class c, std::uint64_t bytes) {
    g_bindings[use][c].fetch_add(1, std::memory_order_relaxed);
    g_binding_bytes[use][c].fetch_add(bytes, std::memory_order_relaxed);
}

// The heap's two checks for a binding in it, and its stamps.
void heap_bound(Use use, std::uint64_t va, std::uint64_t bytes, std::uint64_t call_seq, std::uint64_t serial) {
    g_heap_bindings[use].fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t g0 = (va - g_heap.base) / kGranule, g1 = (va + std::max<std::uint64_t>(bytes, 1) - 1 - g_heap.base) / kGranule;
    if (!call_seq) g_heap_unstamped[use].fetch_add(1, std::memory_order_relaxed);
    bool reused = false;
    for (std::uint64_t g = g0; g <= g1 && g < g_heap.granules; ++g) {
        if (call_seq && g_heap.handed[g].load(std::memory_order_relaxed) > static_cast<std::uint32_t>(call_seq)) reused = true;
        g_heap.bound[g].store(bound_stamp(serial, use), std::memory_order_relaxed);
    }
    if (reused) {
        g_proof_ii[use].fetch_add(1, std::memory_order_relaxed);
        if (g_proof_logged.fetch_add(1, std::memory_order_relaxed) < 12)
            host_log("gxvk census: release proof (ii): %s [0x%llx, +%llu) bound in submission %llu was handed out again after its call (stamp %llu)",
                     kUseName[use], static_cast<ull>(va), static_cast<ull>(bytes), static_cast<ull>(serial), static_cast<ull>(call_seq));
    }
}

}  // namespace

Buffers buffers_mode() {
    // Decided at the first ask - the GX hooks' install, on the host's thread
    // before the game runs - and said once.
    static const Buffers mode = [] {
        if (g_flags.given) return g_flags.mode;
        const BuffersDevice d = host_gpu_integrated()     ? (host_gpu_integrated_with_room() && !host_gpu_memory_tight() ? BuffersDevice::kIntegratedRoomy
                                                                                                                 : BuffersDevice::kIntegrated)
                                : host_gpu_memory_tight() ? BuffersDevice::kTight
                                : host_gpu_small_vram()   ? BuffersDevice::kSmallVram
                                                          : BuffersDevice::kRoomy;
        const Buffers m = buffers_mode_for(nullptr, d);
        if (d == BuffersDevice::kSmallVram)
            host_log("gxvk: GX's buffers on today's path: small video memory, where their copies compete with the targets and textures "
                     "(BBHOST_GXVK_BUFFERS=1 binds them by handle)");
        else if (m == Buffers::kOff)
            host_log("gxvk: GX's buffers on today's path: %s, whose draws read them where they are (BBHOST_GXVK_BUFFERS=1 binds them by handle)",
                     d == BuffersDevice::kIntegrated ? "an integrated GPU" : "tight memory");
        else
            host_log("gxvk: GX's buffers bound by handle%s (BBHOST_GXVK_BUFFERS=0: today's path)",
                     d == BuffersDevice::kIntegratedRoomy ? ", from their copies in an integrated GPU's video memory" : "");
        return m;
    }();
    return mode;
}
bool census() { return g_flags.census; }

void census_buffer_made(std::uint64_t holder, bool initial_data) {
    if (!g_flags.census || !holder) return;
    std::uint8_t usage = 0, cpu = 0, bind = 0;
    std::uint32_t bytes = 0;
    if (!rd(holder + 0x39, &usage) || !rd(holder + 0x3a, &cpu) || !rd(holder + 0x3b, &bind) || !rd(holder + 0x2c, &bytes)) return;
    const std::uint32_t word = kMagic | (usage & kUsage) | (initial_data ? kData : 0u) | (static_cast<std::uint32_t>(cpu & 3) << kCpuShift) |
                               (static_cast<std::uint32_t>(bind) << kBindShift);
    wr32(holder + kHolderSpareB, word);
    if (buffers_mode() == Buffers::kOff) wr32(holder + kHolderSpareA, canary(holder, 0x14));
    g_buffers_made.fetch_add(1, std::memory_order_relaxed);
    const std::uint32_t key = (static_cast<std::uint32_t>(usage & 3) << 16) | (initial_data ? 0x8000u : 0u) |
                              (static_cast<std::uint32_t>(cpu & 3) << 8) | bind;
    std::lock_guard<std::mutex> lk(g_kinds_mu);
    KindCount& k = g_kinds[key];
    ++k.n;
    k.bytes += bytes;
}

void census_buffer_mapped(std::uint64_t holder, std::uint32_t type, std::uint64_t mapped, std::uint32_t bytes, bool ok, bool own) {
    if (!g_flags.census || !holder) return;
    const unsigned t = type < 5 ? type : 0;
    g_maps[t].fetch_add(1, std::memory_order_relaxed);
    if (!ok) {
        g_maps_failed[t].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::uint32_t word = 0;
    if (!rd(holder + kHolderSpareB, &word) || (word & kMagicMask) != kMagic) return;
    if (own) {
        g_maps_own[t].fetch_add(1, std::memory_order_relaxed);
        if (t == 1 || t == 2) word |= kMappedOwn;
    } else if (t == 3 || t == 4) {
        word |= kRenamed;
        if (mapped && in_heap(mapped, bytes)) g_maps_heap[t].fetch_add(1, std::memory_order_relaxed);
    }
    wr32(holder + kHolderSpareB, word);
}

void census_buffer_updated(std::uint64_t holder) {
    if (!g_flags.census || !holder) return;
    std::uint32_t word = 0;
    if (!rd(holder + kHolderSpareB, &word) || (word & kMagicMask) != kMagic) {
        g_updates_untagged.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    (((word >> kBindShift) & kBindCb) ? g_updates_cb : g_updates_other).fetch_add(1, std::memory_order_relaxed);
    if (!(word & kUpdated)) wr32(holder + kHolderSpareB, word | kUpdated);
}

void census_buffer_gone(std::uint64_t holder) {
    if (!g_flags.census || !holder) return;
    g_gone_buffers.fetch_add(1, std::memory_order_relaxed);
    std::uint32_t a = 0, b = 0;
    if (g_flags.mode == Buffers::kOff && rd(holder + kHolderSpareA, &a)) {
        g_spare_checked[kSpareBufA].fetch_add(1, std::memory_order_relaxed);
        if (a != canary(holder, 0x14)) g_spare_broken[kSpareBufA].fetch_add(1, std::memory_order_relaxed);
    }
    if (rd(holder + kHolderSpareB, &b)) {
        g_spare_checked[kSpareBufB].fetch_add(1, std::memory_order_relaxed);
        if ((b & kMagicMask) != kMagic) g_spare_broken[kSpareBufB].fetch_add(1, std::memory_order_relaxed);
    }
}

void census_texture_made(std::uint64_t holder) {
    if (!g_flags.census || !holder) return;
    g_textures_made.fetch_add(1, std::memory_order_relaxed);
    wr32(holder + kHolderSpareA, canary(holder, 0x7e));
}

void census_texture_gone(std::uint64_t holder) {
    if (!g_flags.census || !holder) return;
    g_gone_textures.fetch_add(1, std::memory_order_relaxed);
    std::uint32_t a = 0;
    if (!rd(holder + kHolderSpareA, &a)) return;
    g_spare_checked[kSpareTex].fetch_add(1, std::memory_order_relaxed);
    if (a != canary(holder, 0x7e)) {
        g_spare_broken[kSpareTex].fetch_add(1, std::memory_order_relaxed);
        if (g_spare_logged.fetch_add(1, std::memory_order_relaxed) < 8)
            host_log("gxvk census: texture holder 0x%llx dies with 0x%08x at +0x14, not its canary", static_cast<ull>(holder), a);
    }
}

void census_view_made(std::uint64_t view, int kind) {
    if (!g_flags.census || !view) return;
    if (kind == 3) {
        g_views_made[0].fetch_add(1, std::memory_order_relaxed);
        wr32(view + kSrvSpare, canary(view, 0x48));
        wr32(view + kSrvSpare + 4, canary(view, 0x4c));
    } else if (kind == 5) {
        g_views_made[1].fetch_add(1, std::memory_order_relaxed);
        for (std::uint64_t off = kUavSpare; off < 0x50; off += 4) wr32(view + off, canary(view, static_cast<std::uint32_t>(off)));
    }
}

void census_heap(std::uint64_t pool, std::uint64_t base, std::uint64_t bytes) {
    if (!g_flags.census || !pool || !bytes) return;
    if (g_heap_pools.fetch_add(1, std::memory_order_relaxed) != 0) {
        host_log("gxvk census: another block pool [0x%llx, +0x%llx) (pool 0x%llx); the census follows the first", static_cast<ull>(base),
                 static_cast<ull>(bytes), static_cast<ull>(pool));
        return;
    }
    g_heap.pool = pool;
    g_heap.base = base;
    g_heap.bytes = bytes;
    g_heap.granules = static_cast<std::size_t>((bytes + kGranule - 1) / kGranule);
    g_heap.handed = new std::atomic<std::uint32_t>[g_heap.granules]();
    g_heap.bound = new std::atomic<std::uint32_t>[g_heap.granules]();
    g_heap_ready.store(true, std::memory_order_release);
    host_log("gxvk census: GX's upload heap [0x%llx, +0x%llx) (%llu MiB), its block pool 0x%llx", static_cast<ull>(base), static_cast<ull>(bytes),
             static_cast<ull>(bytes >> 20), static_cast<ull>(pool));
}

void census_handout(std::uint64_t pool, std::uint64_t node, std::uint64_t bytes) {
    if (!g_flags.census || !node) return;
    if (!g_heap_ready.load(std::memory_order_acquire) || pool != g_heap.pool) {
        g_handouts_other_pool.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::uint64_t start = 0;
    if (!rd(node + 8, &start)) return;
    const std::uint64_t size = (bytes + kGranule - 1) & ~(kGranule - 1);
    g_handouts.fetch_add(1, std::memory_order_relaxed);
    g_handout_bytes.fetch_add(size, std::memory_order_relaxed);
    std::uint64_t free_bytes = 0, in_use = 0;
    if (rd(pool + 0xd0, &free_bytes)) {
        for (std::uint64_t m = g_min_free.load(std::memory_order_relaxed); free_bytes < m && !g_min_free.compare_exchange_weak(m, free_bytes);) {
        }
    }
    if (rd(pool + 0xc8, &in_use)) {
        for (std::uint64_t m = g_max_in_use.load(std::memory_order_relaxed); in_use > m && !g_max_in_use.compare_exchange_weak(m, in_use);) {
        }
    }
    if (!in_heap(start, size)) {
        g_handouts_outside.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // Under the pool's lock: one hand-out at a time.
    const std::uint32_t seq = g_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::uint64_t done = done_submissions();
    const std::uint64_t g0 = (start - g_heap.base) / kGranule, g1 = (start + size - 1 - g_heap.base) / kGranule;
    std::uint32_t worst = 0;  // the latest still-running binding's stamp
    for (std::uint64_t g = g0; g <= g1 && g < g_heap.granules; ++g) {
        const std::uint32_t b = g_heap.bound[g].load(std::memory_order_relaxed);
        if (b && stamp_serial(b) >= done && (!worst || stamp_serial(b) > stamp_serial(worst))) worst = b;
        g_heap.handed[g].store(seq, std::memory_order_relaxed);
    }
    g_heap_granules_checked.fetch_add(g1 - g0 + 1, std::memory_order_relaxed);
    if (worst) {
        // done counts the submissions run; the binding's was not.
        const Use use = stamp_use(worst);
        const bool near = stamp_serial(worst) == done;
        (near ? g_proof_i_near : g_proof_i)[use].fetch_add(1, std::memory_order_relaxed);
        if (g_proof_logged.fetch_add(1, std::memory_order_relaxed) < 12)
            host_log("gxvk census: release proof (i): [0x%llx, +%llu) handed out while submission %llu, which bound it as %s, had not run (%llu "
                     "done)",
                     static_cast<ull>(start), static_cast<ull>(size), static_cast<ull>(stamp_serial(worst)), kUseName[use], static_cast<ull>(done));
    }
}

void census_ring(std::uint64_t va, std::uint64_t bytes) {
    if (!g_flags.census) return;
    g_ring_allocs.fetch_add(1, std::memory_order_relaxed);
    if (in_heap(va, bytes)) g_ring_in_heap.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t handout_seq() { return g_flags.census ? g_seq.load(std::memory_order_relaxed) : 0; }

void census_constant(std::uint64_t holder, std::uint64_t va, std::uint32_t bytes) {
    if (!g_flags.census) return;
    count(kUseConstant, class_of(holder, va, bytes, kUseConstant), bytes);
}

void census_view_bound(std::uint64_t view, int kind) {
    if (!g_flags.census || !view) return;
    if (kind == 5) {
        std::uint32_t w[3] = {};
        if (rd(view + kUavSpare, &w)) {
            g_spare_checked[kSpareUav].fetch_add(1, std::memory_order_relaxed);
            if (w[0] != canary(view, 0x44) || w[1] != canary(view, 0x48) || w[2] != canary(view, 0x4c)) {
                g_spare_broken[kSpareUav].fetch_add(1, std::memory_order_relaxed);
                if (g_spare_logged.fetch_add(1, std::memory_order_relaxed) < 8)
                    host_log("gxvk census: unordered-access view 0x%llx is bound with %08x %08x %08x at +0x44, not its canary", static_cast<ull>(view),
                             w[0], w[1], w[2]);
            }
        }
        // Its buffer, by what holds its bytes (a texture's: not a buffer, passed over).
        std::uint64_t holder = 0, vtable = 0, home = 0;
        std::uint32_t size = 0;
        if (rd(view + 8, &holder) && holder && rd(holder, &vtable) && gx_guest_to_bn(vtable) == 0x575f3a0 && rd(holder + 0x20, &home) &&
            rd(holder + 0x2c, &size))
            count(kUseUav, class_of(holder, home, size, kUseUav), size);
        return;
    }
    std::uint32_t w[2] = {};
    if (rd(view + kSrvSpare, &w)) {
        g_spare_checked[kSpareSrv].fetch_add(1, std::memory_order_relaxed);
        if (w[0] != canary(view, 0x48) || w[1] != canary(view, 0x4c)) {
            g_spare_broken[kSpareSrv].fetch_add(1, std::memory_order_relaxed);
            if (g_spare_logged.fetch_add(1, std::memory_order_relaxed) < 8)
                host_log("gxvk census: shader-resource view 0x%llx is bound with %08x %08x at +0x48, not its canary", static_cast<ull>(view), w[0], w[1]);
        }
    }
    // Its texture holder's +0x14: a 2D or 3D texture the registry tagged.
    std::uint64_t holder = 0, vtable = 0;
    if (!rd(view + 8, &holder) || !holder || !rd(holder, &vtable)) return;
    const std::uint64_t vt = gx_guest_to_bn(vtable);
    if (vt != 0x575f3e0 && vt != 0x575f400) return;
    std::uint32_t a = 0;
    if (!rd(holder + kHolderSpareA, &a)) return;
    g_spare_checked[kSpareTex].fetch_add(1, std::memory_order_relaxed);
    if (a != canary(holder, 0x7e)) g_spare_broken[kSpareTex].fetch_add(1, std::memory_order_relaxed);
}

void census_binding(Use use, std::uint64_t holder, std::uint64_t va, std::uint64_t bytes, std::uint64_t call_seq, std::uint64_t serial) {
    if (!g_flags.census) return;
    count(use, class_of(holder, va, bytes, use), bytes);
    if (in_heap(va, bytes)) heap_bound(use, va, bytes, call_seq, serial);
}

void census_constant_bound(std::uint64_t va, std::uint64_t bytes, std::uint64_t call_seq, std::uint64_t serial, bool gx_records) {
    if (!g_flags.census) return;
    if (!gx_records) count(kUseConstant, kNoObject, bytes);
    if (va && in_heap(va, bytes)) heap_bound(kUseConstant, va, bytes, call_seq, serial);
}

void census_report() {
    if (!g_flags.census) return;
    const auto v = [](const std::atomic<std::uint64_t>& a) { return static_cast<ull>(a.load(std::memory_order_relaxed)); };
    host_log("gxvk census: %llu GX buffers made, %llu 2D/3D textures, %llu shader-resource and %llu unordered-access views; %llu buffers and %llu "
             "textures died through their destructors",
             v(g_buffers_made), v(g_textures_made), v(g_views_made[0]), v(g_views_made[1]), v(g_gone_buffers), v(g_gone_textures));
    {
        std::vector<std::pair<std::uint32_t, KindCount>> kinds;
        {
            std::lock_guard<std::mutex> lk(g_kinds_mu);
            kinds.assign(g_kinds.begin(), g_kinds.end());
        }
        std::sort(kinds.begin(), kinds.end(), [](const auto& a, const auto& b) { return a.second.n > b.second.n; });
        static const char* const kUsageName[4] = {"default", "immutable", "dynamic", "staging"};
        std::string line;
        for (std::size_t i = 0; i < kinds.size() && i < 24; ++i) {
            const std::uint32_t key = kinds[i].first;
            char b[160];
            std::snprintf(b, sizeof(b), " %s%s cpu %u bind 0x%02x: %llu (%llu KiB);", kUsageName[(key >> 16) & 3], (key & 0x8000) ? "+data" : "",
                          (key >> 8) & 3, key & 0xff, static_cast<ull>(kinds[i].second.n), static_cast<ull>(kinds[i].second.bytes >> 10));
            line += b;
        }
        host_log("gxvk census: buffers by usage, CPU access (1 write, 2 read) and bind flags (1 vertex, 2 index, 4 constant, 8 shader resource, 0x80 "
                 "unordered access):%s",
                 line.empty() ? " none" : line.c_str());
    }
    host_log("gxvk census: Maps by type 1-4 (handed out the buffer's own memory; a renamed copy in the upload heap; failed): "
             "1 %llu (%llu own), 2 %llu (%llu own), 3 %llu (%llu in the heap, %llu failed), 4 %llu (%llu in the heap, %llu failed); "
             "UpdateSubresource of constant buffers %llu, of other buffers %llu, of untagged ones %llu",
             v(g_maps[1]), v(g_maps_own[1]), v(g_maps[2]), v(g_maps_own[2]), v(g_maps[3]), v(g_maps_heap[3]), v(g_maps_failed[3]), v(g_maps[4]),
             v(g_maps_heap[4]), v(g_maps_failed[4]), v(g_updates_cb), v(g_updates_other), v(g_updates_untagged));
    for (int u = 0; u < kUses; ++u) {
        std::uint64_t total = 0;
        for (int c = 0; c < kClasses; ++c) total += g_bindings[u][c].load(std::memory_order_relaxed);
        std::string line;
        for (int c = 0; c < kClasses; ++c) {
            const std::uint64_t n = g_bindings[u][c].load(std::memory_order_relaxed);
            if (!n) continue;
            char b[160];
            std::snprintf(b, sizeof(b), " %s %llu (%.1f%%, %llu MiB);", kClassName[c], static_cast<ull>(n), total ? 100.0 * static_cast<double>(n) / static_cast<double>(total) : 0.0,
                          static_cast<ull>(g_binding_bytes[u][c].load(std::memory_order_relaxed) >> 20));
            line += b;
        }
        host_log("gxvk census: %s bound, %llu:%s", kUseName[u], static_cast<ull>(total), line.empty() ? " none" : line.c_str());
    }
    if (g_heap_ready.load(std::memory_order_acquire)) {
        const std::uint64_t min_free = g_min_free.load(std::memory_order_relaxed);
        host_log("gxvk census: the upload heap [0x%llx, +0x%llx): %llu hand-outs (%llu MiB), peak use %llu KiB of %llu KiB (least free %llu KiB), "
                 "at most %llu ranges out at once; %llu hand-outs outside it, %llu from other pools; the ring allocator's %llu allocations, %llu "
                 "in the heap",
                 static_cast<ull>(g_heap.base), static_cast<ull>(g_heap.bytes), v(g_handouts), v(g_handout_bytes) >> 20,
                 static_cast<ull>(min_free != ~0ull ? (g_heap.bytes - std::min(min_free, g_heap.bytes)) >> 10 : 0), static_cast<ull>(g_heap.bytes >> 10),
                 static_cast<ull>(min_free != ~0ull ? min_free >> 10 : 0), v(g_max_in_use), v(g_handouts_outside), v(g_handouts_other_pool),
                 v(g_ring_allocs), v(g_ring_in_heap));
        for (int u = 0; u < kUses; ++u) {
            host_log("gxvk census: release proof, %s: (i) %llu hand-outs over a range a submission still running had bound (%llu more within "
                     "one submission of the count); (ii) %llu bindings of a range handed out again after their call; %llu bindings in the heap "
                     "checked (%llu without a call stamp)",
                     kUseName[u], v(g_proof_i[u]), v(g_proof_i_near[u]), v(g_proof_ii[u]), v(g_heap_bindings[u]), v(g_heap_unstamped[u]));
        }
        host_log("gxvk census: release proof: %llu granules handed out", v(g_heap_granules_checked));
    } else {
        host_log("gxvk census: GX's upload heap was never seen made");
    }
    std::string spares;
    for (int s = 0; s < kSpares; ++s) {
        char b[160];
        std::snprintf(b, sizeof(b), " %s %llu checked, %llu not the canary;", kSpareName[s], v(g_spare_checked[s]), v(g_spare_broken[s]));
        spares += b;
    }
    host_log("gxvk census: spare bytes:%s", spares.c_str());
}

}  // namespace gxvk
