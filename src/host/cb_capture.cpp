// Stage 4b.3a(i): copy-backs captured at the GX resolve (host/cb_capture.h).
#include "host/cb_capture.h"
#include "host/gxvk_homes.h"

#include "hle/modules.h"
#include "host/gpu.h"
#include "host/gxvk.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

namespace cb_capture {

// (1 and twins capture and put the captures in place; compare captures beside today's path: host/gxvk_homes.h.)
const Mode g_mode = [] {
    const char* e = std::getenv("BBHOST_GXVK_HOMES");
    if (!e || !*e) return Mode::kOff;
    if (std::strcmp(e, "compare") == 0) return Mode::kCompare;
    if (std::strcmp(e, "twins") == 0) return Mode::kOn;
    return e[0] == '1' ? Mode::kOn : Mode::kOff;
}();

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using ull = unsigned long long;

constexpr u64 kReclaimFlips = 600;

// A piece's state: its lap in the high half, the tokens written into it this
// lap and not yet walked in the low half; `serial` the newest submission (+1)
// that read one of its captures; `flip` when it was last written.
struct Piece {
    std::atomic<u64> state{0};
    std::atomic<u64> serial{0};
    std::atomic<u64> flip{0};
};
Piece g_pieces[kSegments * kPieces];
std::atomic<u8*> g_ring{nullptr};
std::atomic<u32> g_segments{0};

struct Writer {
    int segment = -1;
    bool failed = false;
    u64 off = 0;     // the next byte in the segment
    int owned = -1;  // the piece this thread is writing (its lap begun)
    u64 mapped_lo = 0, mapped_hi = 0;
};
thread_local Writer t_writer;

enum Skip : int { kSkipNoRing, kSkipNoSegment, kSkipBusy, kSkipLong, kSkipUnmapped, kSkipBatch, kSkips };
const char* const kSkipName[kSkips] = {"before the ring was made", "on a thread with no segment", "the piece still in use",
                                       "longer than 64 KiB", "unmapped", "a batch past a piece"};
std::atomic<u64> g_skips[kSkips] = {};
std::atomic<u64> g_batches{0}, g_copies{0}, g_bytes{0}, g_cycles{0}, g_laps{0}, g_reclaimed{0}, g_stale_releases{0};
std::atomic<u64> g_uses[kUses] = {};
const char* const kUseName[kUses] = {"a twin home's GPU copy (token outside a pass)", "a twin home's GPU copy (token in a pass)",
                                     "a copy version of the capture", "the version refused", "a GPU copy from the capture",
                                     "the piece taken back (as today)", "not captured (as today)"};
std::atomic<u64> g_cmp_same{0}, g_cmp_diff{0};

u64 rdtsc() {
#if defined(__x86_64__) || defined(_M_X64)
    return __rdtsc();
#else
    return 0;
#endif
}

u64 align256(u64 v) { return (v + 255) & ~u64{255}; }

bool reusable(Piece& p, u64 flip) {
    const u64 state = p.state.load(std::memory_order_acquire);
    const u64 serial = p.serial.load(std::memory_order_acquire);
    if (static_cast<u32>(state) == 0 && serial <= host_gpu_completed_submits_seen()) return true;
    if (static_cast<u32>(state) != 0 && p.flip.load(std::memory_order_relaxed) + kReclaimFlips < flip) {
        if (g_reclaimed.fetch_add(1, std::memory_order_relaxed) < 4)
            host_log("gxvk homes: a piece of the copy-back captures held for %llu flips by %u tokens never walked, taken back",
                     static_cast<ull>(flip - p.flip.load(std::memory_order_relaxed)), static_cast<u32>(state));
        return true;
    }
    return false;
}

// ---- BBHOST_GXVK_COPYBACK_CHECK: what UpdateSubresource wrote into each rename ----
const bool g_check = [] {
    const char* e = std::getenv("BBHOST_GXVK_COPYBACK_CHECK");
    return g_mode != Mode::kOff && e && e[0] == '1';
}();
struct Update {
    u64 hash = 0, seq = 0;
    u32 tid = 0;
};
struct KeyHash {
    std::size_t operator()(const std::pair<u64, u64>& k) const { return std::hash<u64>()(k.first * 0x9e3779b97f4a7c15ull ^ k.second); }
};
std::mutex g_upd_mu;
std::unordered_map<std::pair<u64, u64>, Update, KeyHash> g_updates;  // (command buffer, home) -> the newest update's
enum CheckBytes : int { kBytesSame, kBytesDiffer, kBytesNoUpdate, kBytesKinds };
std::atomic<u64> g_chk_bytes[kBytesKinds] = {}, g_chk_class[gxvk::kCapClasses] = {}, g_chk_landed{0}, g_chk_updates{0};
std::atomic<u64> g_chk_diff_class[gxvk::kCapClasses] = {};  // the different ones, by their block's class
std::atomic<u64> g_chk_said{0};

u64 fnv(const void* p, u64 n) {
    const u8* b = static_cast<const u8*>(p);
    u64 h = 0xcbf29ce484222325ull;
    for (u64 i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

void check_capture(u64 cb, u64 dst, u64 src, u64 bytes, const void* captured) {
    Update u;
    bool have = false;
    {
        std::lock_guard<std::mutex> lk(g_upd_mu);
        const auto it = g_updates.find({cb, dst});
        if (it != g_updates.end()) {
            u = it->second;
            have = true;
            g_updates.erase(it);
        }
    }
    if (!have) {
        g_chk_bytes[kBytesNoUpdate].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool same = fnv(captured, bytes) == u.hash;
    g_chk_bytes[same ? kBytesSame : kBytesDiffer].fetch_add(1, std::memory_order_relaxed);
    bool landed = false;
    const int c = gxvk::capture_class(src, bytes, u.seq, u.tid, &landed);
    g_chk_class[c].fetch_add(1, std::memory_order_relaxed);
    if (!same) g_chk_diff_class[c].fetch_add(1, std::memory_order_relaxed);
    if (landed) g_chk_landed.fetch_add(1, std::memory_order_relaxed);
    const bool again = c == gxvk::kCapSinceOtherThread || c == gxvk::kCapSinceSameThread;
    if ((!same || again) && g_chk_said.fetch_add(1, std::memory_order_relaxed) < 8)
        host_log("gxvk homes check: a copy-back's capture 0x%llx -> 0x%llx (%llu bytes) %s; its block %s%s", static_cast<ull>(src),
                 static_cast<ull>(dst), static_cast<ull>(bytes), same ? "holds the update's bytes" : "DIFFERS from the update's bytes",
                 again ? "was HANDED OUT AGAIN before the capture" : "is the update's", landed ? " (its retire label had landed)" : "");
}

}  // namespace

void set_ring(void* map) { g_ring.store(static_cast<u8*>(map), std::memory_order_release); }
bool ring_made() { return g_ring.load(std::memory_order_acquire) != nullptr; }
bool check_on() { return g_check; }

const void* bytes_at(std::uint32_t at) {
    u8* const ring = g_ring.load(std::memory_order_acquire);
    return ring && at != kNone ? ring + offset(at) : nullptr;
}

std::uint32_t capture(std::uint64_t cb, const std::uint64_t* dst, const std::uint64_t* src, const std::uint64_t* bytes, std::uint32_t n,
                      std::uint32_t* at, std::uint64_t flip) {
    for (u32 i = 0; i < n; ++i) at[i] = kNone;
    u8* const ring = g_ring.load(std::memory_order_acquire);
    if (!ring) {
        g_skips[kSkipNoRing].fetch_add(n, std::memory_order_relaxed);
        return kNone;
    }
    Writer& w = t_writer;
    if (w.segment < 0) {
        if (!w.failed) {
            const u32 s = g_segments.fetch_add(1, std::memory_order_relaxed);
            if (s < kSegments) {
                w.segment = static_cast<int>(s);
            } else {
                w.failed = true;
            }
        }
        if (w.segment < 0) {
            g_skips[kSkipNoSegment].fetch_add(n, std::memory_order_relaxed);
            return kNone;
        }
    }
    u64 total = 0;
    for (u32 i = 0; i < n; ++i) {
        if (bytes[i] && bytes[i] <= kMaxCopy) total += align256(bytes[i]);
    }
    if (!total) {
        g_skips[kSkipLong].fetch_add(n, std::memory_order_relaxed);
        return kNone;
    }
    if (total > kPieceBytes) {
        g_skips[kSkipBatch].fetch_add(n, std::memory_order_relaxed);
        return kNone;
    }
    const u64 t0 = rdtsc();
    u64 off = w.off;
    u32 piece = static_cast<u32>(off / kPieceBytes) % kPieces;
    if (w.owned == static_cast<int>(piece) && off % kPieceBytes + total > kPieceBytes) {
        piece = (piece + 1) % kPieces;
        off = static_cast<u64>(piece) * kPieceBytes;
    } else if (w.owned != static_cast<int>(piece)) {
        off = static_cast<u64>(piece) * kPieceBytes;
    }
    const u32 index = static_cast<u32>(w.segment) * kPieces + piece;
    Piece& p = g_pieces[index];
    if (w.owned != static_cast<int>(piece)) {
        if (!reusable(p, flip)) {
            g_skips[kSkipBusy].fetch_add(n, std::memory_order_relaxed);
            return kNone;
        }
        const u64 lap = (p.state.load(std::memory_order_relaxed) >> 32) + 1;
        p.serial.store(0, std::memory_order_relaxed);
        p.state.store(lap << 32, std::memory_order_release);
        w.owned = static_cast<int>(piece);
        g_laps.fetch_add(1, std::memory_order_relaxed);
    }
    p.state.fetch_add(1, std::memory_order_acq_rel);
    p.flip.store(flip, std::memory_order_relaxed);
    const u64 seg_base = static_cast<u64>(w.segment) * kSegmentBytes;
    u64 copies = 0, copied = 0;
    for (u32 i = 0; i < n; ++i) {
        const u64 b = bytes[i];
        if (!b) continue;
        if (b > kMaxCopy) {
            g_skips[kSkipLong].fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!((src[i] >= w.mapped_lo && src[i] + b <= w.mapped_hi) || hle_kernel_va_mapped_in(src[i], b, &w.mapped_lo, &w.mapped_hi))) {
            g_skips[kSkipUnmapped].fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        u8* const to = ring + seg_base + off;
        std::memcpy(to, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src[i])), b);
        if (g_check) check_capture(cb, dst[i], src[i], b, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src[i])));
        at[i] = static_cast<u32>((seg_base + off) / 256);
        off += align256(b);
        ++copies;
        copied += b;
    }
    w.off = off;
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();  // the copies out of the write-combining buffers before the token is handed over
#endif
    g_batches.fetch_add(1, std::memory_order_relaxed);
    g_copies.fetch_add(copies, std::memory_order_relaxed);
    g_bytes.fetch_add(copied, std::memory_order_relaxed);
    g_cycles.fetch_add(rdtsc() - t0, std::memory_order_relaxed);
    const u64 lap = (p.state.load(std::memory_order_relaxed) >> 32) & 0xffff;
    return static_cast<std::uint32_t>(lap << 16) | index;
}

bool piece_live(std::uint32_t piece) {
    if (piece == kNone) return false;
    const Piece& p = g_pieces[piece & 0xffff];
    return ((p.state.load(std::memory_order_acquire) >> 32) & 0xffff) == (piece >> 16);
}

void bound(std::uint32_t piece, std::uint64_t serial) {
    if (piece == kNone) return;
    Piece& p = g_pieces[piece & 0xffff];
    if (((p.state.load(std::memory_order_relaxed) >> 32) & 0xffff) != (piece >> 16)) return;  // taken back
    if (serial + 1 > p.serial.load(std::memory_order_relaxed)) p.serial.store(serial + 1, std::memory_order_release);
}

void release(std::uint32_t piece) {
    if (piece == kNone) return;
    Piece& p = g_pieces[piece & 0xffff];
    u64 s = p.state.load(std::memory_order_acquire);
    for (;;) {
        if (((s >> 32) & 0xffff) != (piece >> 16) || static_cast<u32>(s) == 0) {
            g_stale_releases.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (p.state.compare_exchange_weak(s, s - 1, std::memory_order_acq_rel, std::memory_order_acquire)) return;
    }
}

void note_update(std::uint64_t cb, std::uint64_t dst, const void* data, std::uint64_t bytes) {
    if (!g_check || !data || !bytes || bytes > kMaxCopy) return;
    Update u;
    u.hash = fnv(data, bytes);
    u.seq = gxvk::arena_handout_seq();
    u.tid = gxvk::arena_thread_tag();
    std::lock_guard<std::mutex> lk(g_upd_mu);
    if (g_updates.size() > (1u << 20)) g_updates.clear();
    g_updates[{cb, dst}] = u;
    g_chk_updates.fetch_add(1, std::memory_order_relaxed);
}

void count(Use u, std::uint64_t n) { g_uses[u].fetch_add(n, std::memory_order_relaxed); }

void compared(bool same, std::uint64_t dst, std::uint64_t src, std::uint64_t bytes) {
    (same ? g_cmp_same : g_cmp_diff).fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> said{0};
    if (!same && said.fetch_add(1, std::memory_order_relaxed) < 8)
        host_log("gxvk homes compare: a copy-back's capture 0x%llx -> 0x%llx (%llu bytes) differs from the bytes today's path copies at the token",
                 static_cast<ull>(src), static_cast<ull>(dst), static_cast<ull>(bytes));
}

void report() {
    if (g_mode == Mode::kOff) return;
    const u64 batches = g_batches.load(), copies = g_copies.load();
    std::string skips, uses;
    for (int i = 0; i < kSkips; ++i) {
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu", i ? ", " : "", kSkipName[i], static_cast<ull>(g_skips[i].load()));
        skips += b;
    }
    for (int i = 0; i < kUses; ++i) {
        char b[120];
        std::snprintf(b, sizeof b, "%s%s %llu", i ? ", " : "", kUseName[i], static_cast<ull>(g_uses[i].load()));
        uses += b;
    }
    host_log("gxvk homes (BBHOST_GXVK_HOMES=%s): copy-backs captured at the GX resolve: %llu in %llu batches, %.1f MiB, %.0f cycles a batch "
             "on the GX threads (%u segments of %llu MiB); %llu laps of a piece, %llu taken back from tokens never walked, %llu releases of a "
             "lap taken back; not captured: %s",
             g_mode == Mode::kCompare ? "compare" : "1", static_cast<ull>(copies), static_cast<ull>(batches),
             static_cast<double>(g_bytes.load()) / (1 << 20), batches ? static_cast<double>(g_cycles.load()) / static_cast<double>(batches) : 0.0,
             std::min<u32>(g_segments.load(), kSegments), static_cast<ull>(kSegmentBytes >> 20), static_cast<ull>(g_laps.load()),
             static_cast<ull>(g_reclaimed.load()), static_cast<ull>(g_stale_releases.load()), skips.c_str());
    host_log("gxvk homes: at the command processor: %s", uses.c_str());
    if (g_mode == Mode::kCompare)
        host_log("gxvk homes compare: captures against the bytes today's path copied at the token: %llu the same, %llu different",
                 static_cast<ull>(g_cmp_same.load()), static_cast<ull>(g_cmp_diff.load()));
    if (g_check) {
        static const char* const kClass[gxvk::kCapClasses] = {"the update's block (handed out before it)",
                                                              "the update's own new block (handed out on its thread since, the rename its first bytes)",
                                                              "a block HANDED OUT AGAIN on the same thread since the update",
                                                              "a block HANDED OUT AGAIN to another thread since the update", "no hand-out holds it",
                                                              "not in the heap"};
        std::string by;
        for (int c = 0; c < gxvk::kCapClasses; ++c) {
            char b[120];
            std::snprintf(b, sizeof b, "%s%s %llu", c ? ", " : "", kClass[c], static_cast<ull>(g_chk_class[c].load()));
            by += b;
        }
        std::string diff;
        for (int c = 0; c < gxvk::kCapClasses; ++c) {
            char b[40];
            std::snprintf(b, sizeof b, "%s%llu", c ? "/" : "", static_cast<ull>(g_chk_diff_class[c].load()));
            diff += b;
        }
        host_log("gxvk homes check: the different captures by their block's class (as listed below, in order): %s", diff.c_str());
        host_log("gxvk homes check (BBHOST_GXVK_COPYBACK_CHECK): %llu dynamic constant-buffer updates noted; captures against the update's "
                 "bytes: %llu the same, %llu DIFFERENT, %llu with no update noted; the source's block at the capture: %s; its retire label "
                 "landed before the capture: %llu",
                 static_cast<ull>(g_chk_updates.load()), static_cast<ull>(g_chk_bytes[kBytesSame].load()),
                 static_cast<ull>(g_chk_bytes[kBytesDiffer].load()), static_cast<ull>(g_chk_bytes[kBytesNoUpdate].load()), by.c_str(),
                 static_cast<ull>(g_chk_landed.load()));
    }
}

}  // namespace cb_capture

namespace gxvk_homes {

std::atomic<bool> g_made{false}, g_bind{false};

Mode env_mode() {
    static const Mode m = [] {
        const char* e = std::getenv("BBHOST_GXVK_HOMES");
        if (!e || !*e) return Mode::kOff;
        if (std::strcmp(e, "compare") == 0) return Mode::kCompare;
        if (std::strcmp(e, "twins") == 0) return Mode::kTwins;
        return e[0] == '1' ? Mode::kHandle : Mode::kOff;
    }();
    return m;
}

}  // namespace gxvk_homes
