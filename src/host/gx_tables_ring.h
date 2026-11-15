// GX's tables ring (render.cpp gx_ring_place): 128 MiB of host memory the GPU
// page table reaches, where a draw or a dispatch places the tables its user
// data points at - the fetch shader's vertex table, the resource tables, the
// context and extended blocks - which the GPU reads when it runs the work.
// The part of it that needs no GPU (tests/gx_tables_ring_test.cpp).
//
// The command processor's placements - its draws in order, its dispatches, a
// run part's chunks (BBHOST_RUN_COMMIT), Scaleform's tables - go round the
// ring from a cursor, and every one is read by the recording open as it is
// placed. Guard keeps, for each 4 MiB segment of its part, the lap the
// cursor last entered it in and the newest recording that placed there: a
// placement entering a segment in a new lap waits for that recording to have
// run (a run part's draw, which cannot wait, stops and is committed in order),
// as the constant-buffer copies' ring has it. A lap takes about a minute of
// fast loading screens (470 flips a second, 4.8 KiB a flip) and two or three
// of the world at 60 (13 to 21 KiB a flip), so the wait comes only with a GPU
// that far behind.
//
// Stage 4b.1b's pool commits draws ahead of the command processor, before the
// recording that will read their tables is known - the splice puts them into
// whichever is open at the draw's token, up to kListMaxAge flips later. They
// place their tables in Arena: the ring's last 16 MiB, in 1 MiB segments a pool
// thread owns while it places in one, so nothing else writes there however
// long it keeps it (the pool rests at a frame cap for minutes and starts
// working again at the next loading screen). A segment its owner left is taken
// again only once every commit placed in it is too old to splice and every
// recording one was spliced into has run: the splice stamps the segment with
// its recording and checks the segment's lap is still the commit's - a take
// claims the lap before it reads the stamp, the splice stamps before it reads
// the lap, so one of the two sees the other. Until 2026-11-13 the pool took its
// tables from the command processor's cursor in 64 KiB chunks it kept until
// they were full, and the ring wrapped by size alone: a pool thread that rested
// with a chunk half used across a lap of the ring wrote its next draws' tables
// over bytes the cursor had handed out since - other draws' and dispatches'
// tables, which the GPU may not have read yet. (ThreadSanitizer's reports on
// the ring were a meeting of the two that did no harm: the command
// processor's copy of a constant buffer GX placed here runs on to the end of
// its last 64-byte line, past the block into the bytes the cursor had handed
// a pool thread next - bytes no draw binds.)
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

namespace gx_ring {

constexpr std::uint64_t kRingBytes = 128ull << 20;

// The command processor's part of the ring (under the ring's lock).
struct Guard {
    static constexpr std::uint64_t kSegment = 4ull << 20;
    static constexpr std::uint32_t kSegments = static_cast<std::uint32_t>(kRingBytes / kSegment);
    static constexpr std::uint64_t kNone = ~0ull;
    std::uint64_t end = kRingBytes;  // where its part ends (the pool's arena after it)
    std::uint64_t cursor = 0, lap = 1;
    std::uint64_t seg_lap[kSegments] = {}, seg_serial[kSegments] = {};
    std::uint64_t laps = 0, waits = 0;
    // `bytes` aligned to `align` (a power of two) for recording `open` (the
    // one being made): their offset in the ring. A segment they enter in a new
    // lap whose last recording has not run - `ran(serial)` - is waited for,
    // `wait(serial)`; a run part's draw (`part`) gets kNone instead wherever a
    // wrap or a wait would come, and nothing moves. kNone too for more than
    // the part holds.
    template <class Ran, class Wait>
    std::uint64_t take(std::uint64_t bytes, std::uint64_t align, std::uint64_t open, bool part, Ran&& ran, Wait&& wait) {
        if (!bytes || bytes > end) return kNone;
        std::uint64_t at = (cursor + align - 1) & ~(align - 1);
        bool wrap = false;
        if (at + bytes > end) {
            if (part) return kNone;
            at = 0;
            wrap = true;
        }
        const std::uint64_t now = lap + (wrap ? 1 : 0);
        const std::uint32_t s0 = static_cast<std::uint32_t>(at / kSegment), s1 = static_cast<std::uint32_t>((at + bytes - 1) / kSegment);
        for (std::uint32_t s = s0; s <= s1; ++s) {
            // (A segment the open recording placed in, a lap before - more than a lap's tables in one recording -
            // cannot be waited for; never met.)
            if (seg_lap[s] && seg_lap[s] != now && seg_serial[s] < open && !ran(seg_serial[s])) {
                if (part) return kNone;
                ++waits;
                wait(seg_serial[s]);
            }
        }
        if (wrap) {
            lap = now;
            ++laps;
        }
        for (std::uint32_t s = s0; s <= s1; ++s) {
            seg_lap[s] = lap;
            seg_serial[s] = open;
        }
        cursor = at + bytes;
        return at;
    }
};

// The pool's part: kBytes at the ring's end, in kSegments segments.
struct Arena {
    static constexpr std::uint64_t kBytes = 16ull << 20, kSegment = 1ull << 20;
    static constexpr std::uint32_t kSegments = static_cast<std::uint32_t>(kBytes / kSegment);
    static constexpr std::uint64_t kNone = ~0ull;
    // A pool thread's own: the segment it owns, the lap it took it at, where its next placement goes, the flip it last
    // stamped as the segment's.
    struct Cursor {
        std::uint32_t seg = ~0u;
        std::uint64_t lap = 0, next = 0, flip = 0;
    };
    std::mutex mu;  // the takes
    bool owned[kSegments] = {};
    std::atomic<std::uint64_t> last_flip[kSegments] = {};   // its owner's last placement's flip (0: never placed in)
    std::atomic<std::uint64_t> max_serial[kSegments] = {};  // the newest recording a commit placed in it was spliced into
    std::atomic<std::uint64_t> lap[kSegments] = {};         // moved by each take (and each look at it that found it busy)
    std::uint32_t next = 0;
    std::atomic<std::uint64_t> takes{0}, full{0};

    // `bytes` aligned to `align` (a power of two, at most 256) for a draw this
    // thread commits at flip `flip`: their offset in the arena, `c` saying in
    // which segment and lap. A new segment when its own is full: one no thread
    // owns, whose last placement is more than `max_age` + 2 flips old (no
    // commit of it may still be spliced) and whose last splice's recording has
    // run - `completed` recordings have. kNone when there is none (the draw
    // is then drawn in order) or the tables are longer than a segment.
    std::uint64_t place(Cursor& c, std::uint64_t bytes, std::uint64_t align, std::uint64_t flip, std::uint64_t completed, std::uint64_t max_age) {
        if (!bytes || bytes > kSegment) return kNone;
        std::uint64_t at = (c.next + align - 1) & ~(align - 1);
        if (c.seg == ~0u || at + bytes > kSegment) {
            std::lock_guard<std::mutex> lk(mu);
            std::uint32_t found = ~0u;
            std::uint64_t found_lap = 0;
            for (std::uint32_t k = 0; k < kSegments && found == ~0u; ++k) {
                const std::uint32_t s = (next + k) % kSegments;
                if (owned[s]) continue;
                const std::uint64_t last = last_flip[s].load(std::memory_order_relaxed);
                if (last && last + max_age + 2 >= flip) continue;
                // Its lap claimed before its stamp is read (the splice stamps, then reads the lap): a commit spliced
                // into a recording not run yet keeps it, or its splice sees the lap moved and draws it in order.
                const std::uint64_t l = lap[s].fetch_add(1, std::memory_order_seq_cst) + 1;
                if (max_serial[s].load(std::memory_order_seq_cst) >= completed + (last ? 0 : 1)) continue;
                found = s;
                found_lap = l;
            }
            if (found == ~0u) {
                full.fetch_add(1, std::memory_order_relaxed);
                return kNone;
            }
            if (c.seg != ~0u) owned[c.seg] = false;  // (its last flip stays: it ages from there)
            owned[found] = true;
            next = (found + 1) % kSegments;
            last_flip[found].store(flip, std::memory_order_relaxed);
            max_serial[found].store(0, std::memory_order_relaxed);
            c.seg = found;
            c.lap = found_lap;
            c.next = 0;
            c.flip = flip;
            at = 0;
            takes.fetch_add(1, std::memory_order_relaxed);
        } else if (c.flip != flip) {
            c.flip = flip;
            last_flip[c.seg].store(flip, std::memory_order_relaxed);
        }
        c.next = at + bytes;
        return static_cast<std::uint64_t>(c.seg) * kSegment + at;
    }
    // Whether segment `seg` is still at lap `l` (a splice's first look).
    bool lap_is(std::uint32_t seg, std::uint64_t l) const { return seg < kSegments && lap[seg].load(std::memory_order_acquire) == l; }
    // A commit that placed in segment `seg` at lap `l`, spliced into recording
    // `open`: the segment stamped with it, then true when it is still that lap
    // (false: taken again since - drawn in order).
    bool splice(std::uint32_t seg, std::uint64_t l, std::uint64_t open) {
        if (seg >= kSegments) return false;
        std::atomic<std::uint64_t>& ms = max_serial[seg];
        std::uint64_t cur = ms.load(std::memory_order_seq_cst);
        while (cur < open && !ms.compare_exchange_weak(cur, open, std::memory_order_seq_cst)) {}
        return lap[seg].load(std::memory_order_seq_cst) == l;
    }
};

}  // namespace gx_ring
