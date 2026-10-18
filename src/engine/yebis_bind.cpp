#include "engine/yebis_bind.h"

#include "host/gpu.h"

#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace {

using ull = unsigned long long;

std::uint32_t rd32(std::uint64_t va) {
    std::uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 4);
    return v;
}
std::uint64_t rd64(std::uint64_t va) {
    std::uint64_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 8);
    return v;
}
std::uint8_t rd8(std::uint64_t va) { return *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va)); }
void wr8(std::uint64_t va, std::uint8_t v) { *reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(va)) = v; }
void wr64(std::uint64_t va, std::uint64_t v) { std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va)), &v, 8); }

// YEBIS's per-stage resource array is 0x641 dwords a stage.
constexpr std::uint32_t kStageDwords = 0x641;
// The staging dwords for user-data slots past 15 sit in the same array.
constexpr std::uint32_t kStagingDwords = 0x5b1;

// The plain descriptor types: a record in the resource array, copied straight
// into the shader's user data (or, when the stage uses the extended table, into
// the staging dwords the block window carries).
struct Plain {
    std::uint32_t base;    // dword offset of slot 0 in the stage's array
    std::uint32_t stride;  // dwords a slot
    std::uint32_t dwords;  // dwords the descriptor takes
    std::uint32_t wide;    // dwords when the descriptor's flag bit is set (0: never)
};
bool plain_of(std::uint32_t type, Plain* out) {
    switch (type) {
    case 0: *out = {0x000, 8, 4, 8}; return true;  // texture: 16 bytes, or 32 with the flag
    case 4: *out = {0x400, 8, 4, 8}; return true;
    case 1: *out = {0x480, 4, 4, 0}; return true;
    case 3: *out = {0x4c0, 4, 4, 0}; return true;
    case 2: *out = {0x540, 4, 4, 0}; return true;  // sampler
    case 6: *out = {0x590, 1, 1, 0}; return true;
    case 5: *out = {0x5a0, 1, 1, 0}; return true;
    case 7: *out = {0x5b0, 1, 1, 0}; return true;
    default: return false;
    }
}

// The extended types: the record is a window too big for user data, so it goes
// to a ring of its own and the shader gets the slot's address. `summary` and
// `marks` are byte arrays indexed by the stage; `ce_base` is a dword offset
// past stage * 0x641.
struct Extended {
    std::uint32_t entry;       // + slot: which ring entry
    std::uint32_t summary;     // + stage: the byte whose bit says the slot has work
    std::uint32_t marks;       // + stage * mark_bytes: the bitmap, a bit a record
    std::uint32_t mark_bytes;  // how many bytes of it a stage has
    std::uint32_t ce_base;     // dword offset of the window, past slot * ce_slot
    std::uint32_t ce_slot;     // dwords a slot adds to that offset
    std::uint32_t records;     // records in the window
    std::uint32_t record;      // bytes a record
};
bool extended_of(std::uint32_t type, Extended* out) {
    switch (type) {
    case 0x13: *out = {0, 0x10, 0x41, 16, 0x000, 0x400, 0x80, 0x20}; return true;
    case 0x19: *out = {1, 0x17, 0xb1, 2, 0x400, 0x080, 0x10, 0x20}; return true;
    case 0x15: *out = {2, 0x1e, 0xbf, 2, 0x480, 0x040, 0x10, 0x10}; return true;
    case 0x17: *out = {3, 0x25, 0xcd, 4, 0x4c0, 0x080, 0x20, 0x10}; return true;
    case 0x16: *out = {4, 0x2c, 0xe9, 3, 0x540, 0x050, 0x14, 0x10}; return true;
    case 0x18: *out = {5, 0x33, 0xfe, 1, 0x5b1, 0x010, 0x04, 0x10}; return true;
    default: return false;
    }
}

// A ring's state, 40 bytes an entry, at state + stage * 0x118 + 0x268.
struct Ring {
    std::uint64_t at;
    std::uint64_t index() const { return rd64(at); }
    std::uint64_t base() const { return rd64(at + 8); }
    std::uint32_t stride() const { return rd32(at + 0x10); }
    std::uint32_t count() const { return rd32(at + 0x14); }
    std::uint64_t mark_a() const { return rd64(at + 0x18); }
    std::uint64_t mark_b() const { return rd64(at + 0x20); }
    std::uint64_t slot(std::uint64_t i) const { return base() + ((static_cast<std::uint64_t>(stride()) * i) << 2); }
};
Ring ring_at(std::uint64_t state, int stage, std::uint64_t entry) {
    return {state + static_cast<std::uint64_t>(stage) * 0x118 + entry * 40 + 0x268};
}
// The ring the extended user-data block goes in, one a stage, whose fields sit
// in the same order at + 0x358.
Ring block_ring(std::uint64_t state, int stage) {
    return {state + static_cast<std::uint64_t>(stage) * 0x118 + 0x358};
}

// Advancing a ring also tells the game when it has come round to a slot the GPU
// may still be reading; the flag it sets is the game's, so we set it too.
std::uint64_t advance(const Ring& r, std::uint64_t state, bool dry) {
    const std::uint32_t n = r.count();
    if (!n) return r.index();
    const std::uint64_t next = (r.index() + 1) % n;
    if (dry) return next;
    wr64(r.at, next);
    if (next == r.mark_a() || next == r.mark_b()) wr8(state + 0x187, static_cast<std::uint8_t>(rd8(state + 0x187) | 1));
    return next;
}

// Our constant RAM: the same 48 KiB, addressed by the same offsets, so a window
// handed to the GPU carries a record the game has not marked at the value it
// last uploaded and not the one it is editing now.
constexpr std::uint32_t kMirrorBytes = 48 * 1024;
std::uint8_t g_mirror[kMirrorBytes];
std::uint64_t g_mirror_res = 0;

// The windows in transit, from the draw's call to its token. A world frame
// stashes about 90 KiB, and the command processor is a frame or two behind, so
// a megabyte is ten frames of headroom - and small enough to stay in cache,
// which an eight-megabyte ring was not. The pad lets a window straddle the
// wrap rather than be split, and `seq` counts bytes ever taken, so a window
// the ring has lapped can be caught instead of read as someone else's.
constexpr std::uint32_t kArenaBytes = 1u << 20;
constexpr std::uint32_t kMaxWindow = 4096;
std::uint8_t* g_arena = nullptr;
std::atomic<std::uint64_t> g_arena_seq{0};

std::atomic<std::uint64_t> g_stages{0}, g_fallback{0}, g_copies{0}, g_copy_bytes{0}, g_overflow{0};
std::atomic<std::uint64_t> g_refreshed{0}, g_refresh_bytes{0};
std::atomic<std::uint64_t> g_mirror_resets{0}, g_arena_lapped{0};
// The hardware's constant RAM belongs to a queue, and this mirror is one
// global: builds that hand over from one thread to another are what one queue
// sees (YEBIS's first compute pass runs on the thread that starts it, every
// later one and the draws on the render thread, one state and one array), but
// two at once would interleave. Counted rather than assumed.
std::atomic<int> g_building{0};
std::atomic<std::thread::id> g_last_builder{};
std::atomic<std::uint64_t> g_handoffs{0}, g_overlapped{0};
struct Building {
    Building() {
        if (g_building.fetch_add(1, std::memory_order_acq_rel) != 0) g_overlapped.fetch_add(1, std::memory_order_relaxed);
        const std::thread::id me = std::this_thread::get_id();
        if (g_last_builder.load(std::memory_order_relaxed) != me) {
            const std::thread::id was = g_last_builder.exchange(me, std::memory_order_relaxed);
            if (was != std::thread::id{} && was != me) g_handoffs.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ~Building() { g_building.fetch_sub(1, std::memory_order_acq_rel); }
};
std::atomic<std::uint64_t> g_cmp_checked{0}, g_cmp_bad{0};
std::atomic<std::uint64_t> g_verify_checked{0}, g_verify_bad{0}, g_verify_bytes{0}, g_verify_seen{0}, g_verify_stashed{0};
std::atomic<std::uint64_t> g_unknown_type[32] = {};
std::atomic<int> g_cmp_logged{0}, g_verify_logged{0};
std::atomic<std::uint32_t> g_ring_depth[32] = {}, g_ring_stride[32] = {};
void note_ring(std::uint32_t type, std::uint32_t count, std::uint32_t stride) {
    g_ring_depth[type & 31].store(count, std::memory_order_relaxed);
    g_ring_stride[type & 31].store(stride, std::memory_order_relaxed);
}

int g_mode = -1;
bool g_verify = false;

// The stashed windows, oldest first, for the verify check.
struct Stashed {
    std::uint32_t ce, at, bytes;
    bool taken;
};
constexpr int kStashed = 256;
Stashed g_stashed[kStashed];
unsigned g_stash_head = 0, g_stash_tail = 0;  // under g_stash_mu
std::mutex g_stash_mu;

// sub_15f8f70: the records the game has marked, into the mirror at the offset
// they have in the array, and the marks cleared. `first` is where this slot's
// records start in the bitmap, `ce` the window's byte offset.
void refresh(std::uint64_t res, std::uint64_t marks, std::uint32_t first, std::uint32_t records, std::uint32_t ce,
             std::uint32_t record, bool dry) {
    std::uint32_t run_from = 0;
    bool in_run = false;
    for (std::uint32_t i = 0; i <= records; ++i) {
        const std::uint32_t e = first + i;
        const bool set = i < records && ((rd8(marks + (e >> 3)) >> (e & 7)) & 1) != 0;
        if (set && !in_run) {
            run_from = e;
            in_run = true;
        } else if (!set && in_run) {
            const std::uint32_t off = run_from * record + ce;
            const std::uint32_t bytes = (e - run_from) * record;
            if (off + bytes <= kMirrorBytes) {
                std::memcpy(g_mirror + off, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(res + off)), bytes);
                g_refreshed.fetch_add(1, std::memory_order_relaxed);
                g_refresh_bytes.fetch_add(bytes, std::memory_order_relaxed);
            }
            in_run = false;
        }
        if (!dry && set) wr8(marks + (e >> 3), static_cast<std::uint8_t>(rd8(marks + (e >> 3)) & ~(1u << (e & 7))));
    }
}

// Room in the arena, where a copy waits for the token.
std::uint32_t arena_room(std::uint32_t bytes, std::uint64_t* seq) {
    const std::uint64_t was = g_arena_seq.fetch_add((bytes + 63) & ~63u, std::memory_order_relaxed);
    *seq = was;
    return static_cast<std::uint32_t>(was & (kArenaBytes - 1));
}

// The window, into the arena, where it waits for the token.
std::uint32_t stash(std::uint32_t ce, std::uint32_t bytes, std::uint64_t* seq) {
    const std::uint32_t at = arena_room(bytes, seq);
    std::memcpy(g_arena + at, g_mirror + ce, bytes);
    if (g_verify) {
        std::lock_guard<std::mutex> lk(g_stash_mu);
        g_verify_stashed.fetch_add(1, std::memory_order_relaxed);
        g_stashed[g_stash_tail % kStashed] = {ce, at, bytes, false};
        ++g_stash_tail;
        if (g_stash_tail - g_stash_head > kStashed) g_stash_head = g_stash_tail - kStashed;
    }
    return at;
}

}  // namespace

namespace yebis_bind {

int mode() {
    if (g_mode < 0) {
        const char* e = std::getenv("BBHOST_YEBIS_BIND");
        g_mode = e ? std::atoi(e) : 1;
        const char* v = std::getenv("BBHOST_YEBIS_BIND_VERIFY");
        g_verify = v && v[0] == '1';
        if (g_mode) g_arena = new std::uint8_t[kArenaBytes + kMaxWindow]();
    }
    return g_mode;
}
bool verifying() { return g_verify; }

bool table_known(std::uint64_t table, std::uint32_t count) {
    if (count > 64) return false;
    int windows = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t type = rd32(table + 4ull * i) & 0xff;
        Plain p{};
        Extended x{};
        if (plain_of(type, &p) || type == 0x1a || type == 0x1b || type == 0x12) continue;
        if (extended_of(type, &x)) {
            ++windows;
            continue;
        }
        g_unknown_type[type & 31].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return windows + 1 <= kMaxCopies;  // + the extended block's own window
}

void note_fallback() { g_fallback.fetch_add(1, std::memory_order_relaxed); }

bool build_stage(std::uint64_t state, int stage, std::uint64_t table, std::uint32_t count, std::uint64_t* block, Work& work,
                 bool dry) {
    const std::uint64_t res = rd64(state + 8);
    if (!res || count > 64) return false;
    const Building building;
    if (res != g_mirror_res) {  // a new array starts from a blank constant RAM, as the hardware's does
        g_mirror_res = res;
        std::memset(g_mirror, 0, sizeof(g_mirror));
        g_mirror_resets.fetch_add(1, std::memory_order_relaxed);
    }
    const std::uint32_t stage_dw = static_cast<std::uint32_t>(stage) * kStageDwords;

    // Does the stage use the extended table? Only the types the builder counts
    // (0..7 and all the extended ones but 0x18) put a descriptor there.
    std::uint8_t ext = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t d = rd32(table + 4ull * i);
        const std::uint32_t type = d & 0xff, ud = (d >> 16) & 0xff;
        if (type > 0x19) continue;
        const bool counts = type < 8 || ((0x2e80000u >> type) & 1) != 0;
        if (counts && ud > 0xf) ext = 1;
    }
    // Set even when predicting, for the same reason the staging dwords are:
    // the guest's builder writes these two bytes the same way at the top of
    // its own run, and the low bit of the first is what lets the block window
    // refresh at all - its own upload clears it again.
    wr8(state + static_cast<std::uint64_t>(stage) + 0x105, ext);
    wr8(state + static_cast<std::uint64_t>(stage) + 0x3a, ext);

    // The staging dwords live in the resource array too, at 0x5b1 + the
    // user-data slot, and the stage remembers the range it has used: the low
    // end resets every draw, the high end does not.
    const std::uint64_t lo_at = state + static_cast<std::uint64_t>(stage) * 2 + 0x10c;
    const std::uint64_t hi_at = lo_at + 1;
    std::uint8_t lo = rd8(lo_at), hi = rd8(hi_at);
    const std::uint64_t staging = res + 4ull * (stage_dw + kStagingDwords);

    int block_desc = -1;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t d = rd32(table + 4ull * i);
        const std::uint32_t type = d & 0xff, slot = (d >> 8) & 0xff;
        std::uint8_t ud = static_cast<std::uint8_t>((d >> 16) & 0xff);
        const bool wide = ((d >> 24) & 1) != 0;
        if (type > 0x1b) continue;

        std::uint64_t addr = 0;
        std::uint32_t dwords = 0;
        const std::uint32_t* from = nullptr;

        Plain p{};
        Extended x{};
        if (plain_of(type, &p)) {
            dwords = wide && p.wide ? p.wide : p.dwords;
            from = reinterpret_cast<const std::uint32_t*>(
                static_cast<std::uintptr_t>(res + 4ull * (stage_dw + p.base + slot * p.stride)));
        } else if (extended_of(type, &x)) {
            const Ring r = ring_at(state, stage, x.entry + slot);
            if (!r.base() || !r.count()) continue;
            note_ring(type, r.count(), r.stride());
            // The summary bit is the game's, and the builder never clears it:
            // once a slot is in use its window is re-dumped every draw, so each
            // draw reads a snapshot of its own.
            const std::uint64_t summary = state + x.summary + static_cast<std::uint64_t>(stage) + (slot >> 3);
            std::uint64_t index = r.index();
            if ((rd8(summary) >> (slot & 7)) & 1) {
                const std::uint32_t ce = 4u * (stage_dw + x.ce_base + slot * x.ce_slot);
                const std::uint32_t bytes = x.records * x.record;
                // Checked before the ring moves: giving up here would leave the
                // stage half built, which is the one thing it must not do.
                if (work.n >= kMaxCopies || ce + bytes > kMirrorBytes || bytes > kMaxWindow) {
                    g_overflow.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                index = advance(r, state, dry);
                refresh(res, state + x.marks + static_cast<std::uint64_t>(stage) * x.mark_bytes, slot * x.records, x.records,
                        ce, x.record, dry);
                std::uint64_t seq = 0;
                const std::uint32_t at = stash(ce, bytes, &seq);
                work.copies[work.n++] = {r.slot(index), seq, at, bytes};
            }
            addr = r.slot(index);
            dwords = 2;
        } else if (type == 0x1a) {
            addr = rd64(state + 0x198);
            dwords = 2;
        } else if (type == 0x1b) {
            block_desc = static_cast<int>(i);
            continue;
        } else if (type == 0x12) {
            continue;  // the builder skips it; slot 0 comes from the state
        } else {
            g_unknown_type[type & 31].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (addr) block[i] = addr;
        if (ud < 0x10 || !ext) continue;

        if (lo > ud) lo = ud;
        // Written even when predicting: the guest's builder is about to write
        // the same dwords to the same place, so this costs nothing and lets
        // compare mode weigh our staging against its own rather than against
        // the last draw's.
        {
            const std::uint64_t at = staging + 4ull * ud;
            if (from) {
                std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at)), from, 4ull * dwords);
            } else {
                const std::uint32_t pair[2] = {static_cast<std::uint32_t>(addr), static_cast<std::uint32_t>(addr >> 32)};
                std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at)), pair, 8);
            }
        }
        ud = static_cast<std::uint8_t>(ud + dwords);
        if (hi < ud) hi = ud;
    }

    if (ext) {
        // One more window: the staging range the shader reads as a table. Its
        // one mark is the flag byte, always set, so it refreshes every draw.
        const Ring r = block_ring(state, stage);
        if (r.base() && r.count()) {
            const std::uint32_t ce = 4u * (stage_dw + kStagingDwords + lo);
            const std::uint32_t bytes = 4u * static_cast<std::uint32_t>(hi > lo ? hi - lo : 0);
            if (work.n >= kMaxCopies || ce + bytes > kMirrorBytes || bytes > kMaxWindow) {
                g_overflow.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            // The ring moves and the descriptor gets its address even when the
            // range is empty, which is what the builder does.
            const std::uint64_t index = advance(r, state, dry);
            if (bytes) {
                refresh(res, state + static_cast<std::uint64_t>(stage) + 0x105, 0, 1, ce, bytes, dry);
                std::uint64_t seq = 0;
                const std::uint32_t at = stash(ce, bytes, &seq);
                work.copies[work.n++] = {r.slot(index), seq, at, bytes};
            }
            if (block_desc >= 0) block[block_desc] = r.slot(index);
        }
    }
    if (!dry) {
        wr8(lo_at, 0x80);
        wr8(hi_at, hi);
    }
    g_stages.fetch_add(1, std::memory_order_relaxed);
    return true;
}

std::uint16_t user_data(std::uint64_t state, int stage, std::uint64_t table, std::uint32_t count, const std::uint64_t* block,
                        std::uint32_t* user) {
    const std::uint64_t res = rd64(state + 8);
    if (!res || count > 64) return 0;
    const std::uint32_t stage_dw = static_cast<std::uint32_t>(stage) * kStageDwords;
    std::uint16_t written = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t d = rd32(table + 4ull * i);
        const std::uint32_t type = d & 0xff, slot = (d >> 8) & 0xff, ud = (d >> 16) & 0xff;
        const bool wide = ((d >> 24) & 1) != 0;
        if (ud > 0xf || type > 0x1b) continue;
        Plain p{};
        if (plain_of(type, &p)) {
            const std::uint32_t dwords = std::min<std::uint32_t>(wide && p.wide ? p.wide : p.dwords, 16 - ud);
            std::memcpy(user + ud, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(res + 4ull * (stage_dw + p.base + slot * p.stride))),
                        4ull * dwords);
            written = static_cast<std::uint16_t>(written | (((1u << dwords) - 1) << ud));
            continue;
        }
        if (type == 0x12 || type < 0x13 || ud + 1 > 0xf || !block[i]) continue;  // 0x12: the builder skips it
        user[ud] = static_cast<std::uint32_t>(block[i]);
        user[ud + 1] = static_cast<std::uint32_t>(block[i] >> 32);
        written = static_cast<std::uint16_t>(written | (3u << ud));
    }
    return written;
}

bool add_copy(Work& work, std::uint64_t dst, const void* src, std::uint32_t bytes) {
    if (work.n >= kMaxCopies || bytes > kMaxWindow) return false;
    std::uint64_t seq = 0;
    const std::uint32_t at = arena_room(bytes, &seq);
    std::memcpy(g_arena + at, src, bytes);
    work.copies[work.n++] = {dst, seq, at, bytes};
    return true;
}

void run(const Work& work) {
    const std::uint64_t now = g_arena_seq.load(std::memory_order_relaxed);
    for (int i = 0; i < work.n; ++i) {
        if (!g_arena || now - work.copies[i].seq >= kArenaBytes) {  // lapped: the bytes are another draw's
            g_arena_lapped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!host_gpu_draw_window(work.copies[i].dst, g_arena + work.copies[i].at, work.copies[i].bytes)) {
            hle_gnm_cp_write(work.copies[i].dst, g_arena + work.copies[i].at, work.copies[i].bytes);
        }
        g_copy_bytes.fetch_add(work.copies[i].bytes, std::memory_order_relaxed);
    }
    g_copies.fetch_add(static_cast<std::uint64_t>(work.n), std::memory_order_relaxed);
}

std::atomic<std::uint64_t> g_ordered{0}, g_ordered_cpu{0};
void run_ordered(const Work& work) {
    const std::uint64_t now = g_arena_seq.load(std::memory_order_relaxed);
    for (int i = 0; i < work.n; ++i) {
        if (!g_arena || now - work.copies[i].seq >= kArenaBytes) {
            g_arena_lapped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const Copy& c = work.copies[i];
        if (host_gpu_mem_write_ordered(c.dst, g_arena + c.at, c.bytes, nullptr)) {
            g_ordered.fetch_add(1, std::memory_order_relaxed);
        } else if (hle_kernel_va_mapped(c.dst, c.bytes)) {
            // Not on the GPU's timeline (not imported memory): everything
            // recorded before has run, then the CPU writes it.
            host_gpu_flush();
            std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(c.dst)), g_arena + c.at, c.bytes);
            g_ordered_cpu.fetch_add(1, std::memory_order_relaxed);
        }
        g_copy_bytes.fetch_add(c.bytes, std::memory_order_relaxed);
    }
    g_copies.fetch_add(static_cast<std::uint64_t>(work.n), std::memory_order_relaxed);
}

void compare(int stage, const std::uint64_t* mine, const std::uint64_t* theirs, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) {
        g_cmp_checked.fetch_add(1, std::memory_order_relaxed);
        if (mine[i] == theirs[i]) continue;
        g_cmp_bad.fetch_add(1, std::memory_order_relaxed);
        if (g_cmp_logged.fetch_add(1, std::memory_order_relaxed) < 12) {
            host_log("yebis-bind: stage %d descriptor %u resolved to 0x%llx, the builder said 0x%llx", stage, i,
                     static_cast<ull>(mine[i]), static_cast<ull>(theirs[i]));
        }
    }
}

void check_dump(std::uint32_t off, const void* ce, std::uint32_t bytes) {
    g_verify_seen.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_stash_mu);
    // The oldest window of this shape that no dump has taken: GX's own dumps
    // come down the same stream and simply never match.
    for (unsigned i = g_stash_head; i != g_stash_tail; ++i) {
        Stashed& s = g_stashed[i % kStashed];
        if (s.taken || s.ce != off || s.bytes != bytes) continue;
        s.taken = true;
        while (g_stash_head != g_stash_tail && g_stashed[g_stash_head % kStashed].taken) ++g_stash_head;
        g_verify_checked.fetch_add(1, std::memory_order_relaxed);
        g_verify_bytes.fetch_add(bytes, std::memory_order_relaxed);
        if (std::memcmp(ce, g_arena + s.at, bytes) == 0) return;
        g_verify_bad.fetch_add(1, std::memory_order_relaxed);
        if (g_verify_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            std::uint32_t first = 0;
            for (std::uint32_t k = 0; k < bytes; ++k) {
                if (static_cast<const std::uint8_t*>(ce)[k] != g_arena[s.at + k]) {
                    first = k;
                    break;
                }
            }
            host_log("yebis-bind: the window at constant RAM +0x%x (%u bytes) differs from ours at byte %u", off, bytes, first);
        }
        return;
    }
}

void report() {
    if (mode() == 0) return;
    host_log("yebis-bind: %llu stages built natively, %llu draws fell back to the builder%s; %llu windows written (%llu MiB), "
             "%llu record runs refreshed (%llu MiB)",
             static_cast<ull>(g_stages.load()), static_cast<ull>(g_fallback.load()),
             g_overflow.load() ? " (some for want of room)" : "", static_cast<ull>(g_copies.load()),
             static_cast<ull>(g_copy_bytes.load() >> 20), static_cast<ull>(g_refreshed.load()),
             static_cast<ull>(g_refresh_bytes.load() >> 20));
    char types[128] = {};
    int n = 0;
    for (int i = 0; i < 32 && n < static_cast<int>(sizeof(types)) - 16; ++i) {
        if (const std::uint64_t c = g_unknown_type[i].load()) {
            n += std::snprintf(types + n, sizeof(types) - n, " 0x%x:%llu", i, static_cast<ull>(c));
        }
    }
    if (n) host_log("yebis-bind: descriptor types it does not know:%s", types);
    char rings[192] = {};
    n = 0;
    for (int i = 0; i < 32 && n < static_cast<int>(sizeof(rings)) - 28; ++i) {
        if (g_ring_depth[i].load()) {
            n += std::snprintf(rings + n, sizeof(rings) - n, " 0x%x:%u slots of %u dwords", i, g_ring_depth[i].load(),
                               g_ring_stride[i].load());
        }
    }
    if (n) host_log("yebis-bind: rings:%s", rings);
    if (g_handoffs.load() || g_overlapped.load() || g_mirror_resets.load() > 1) {
        host_log("yebis-bind: builds handed from one thread to another %llu times, %llu begun while another thread's was "
                 "running%s; the constant-RAM mirror started blank %llu times (a new resource array)",
                 static_cast<ull>(g_handoffs.load()), static_cast<ull>(g_overlapped.load()),
                 g_overlapped.load() ? " - the one mirror needs to be per queue" : "", static_cast<ull>(g_mirror_resets.load()));
    }
    if (g_ordered.load() || g_ordered_cpu.load()) {
        host_log("yebis-bind: dispatch windows on the GPU's timeline %llu, on the CPU after a flush %llu",
                 static_cast<ull>(g_ordered.load()), static_cast<ull>(g_ordered_cpu.load()));
    }
    if (g_arena_lapped.load()) {
        host_log("yebis-bind: %llu windows were lapped in the arena before the token reached them",
                 static_cast<ull>(g_arena_lapped.load()));
    }
    if (g_cmp_checked.load()) {
        host_log("yebis-bind: %llu addresses checked against the builder, %llu differ", static_cast<ull>(g_cmp_checked.load()),
                 static_cast<ull>(g_cmp_bad.load()));
    }
    if (g_verify_checked.load()) {
        host_log("yebis-bind: %llu constant-RAM dumps walked, %llu of our %llu windows matched one (%llu MiB), %llu differ",
                 static_cast<ull>(g_verify_seen.load()), static_cast<ull>(g_verify_checked.load()),
                 static_cast<ull>(g_verify_stashed.load()), static_cast<ull>(g_verify_bytes.load() >> 20),
                 static_cast<ull>(g_verify_bad.load()));
    }
}

}  // namespace yebis_bind
