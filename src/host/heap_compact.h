#pragma once
// The image heap's compaction (BBHOST_HEAP_COMPACT; render.cpp
// compact_image_heap_locked): once the world has settled, the heap's emptiest
// block is emptied into the holes of the others, so the memory keeper gives it
// back. A block goes back only when nothing is left in it, and the targets'
// blocks also hold textures that overflowed into their holes: on a Radeon
// 680M at a lamp at 1080p the heap kept 5-6 blocks of 256 MiB with 343-615 MiB
// free between 200-300 holes, the emptiest targets' block holding 88-112 MiB -
// over the 64 MiB the targets' drain takes - while the caches' budget only
// widened the holes. What the block holds is copied (render targets,
// snapshots, surfaces the GPU wrote, textures bound in the last kHotMs: a
// texture made again at its next bind is CPU work on the command processor
// while the game plays) or let go (textures not bound for kHotMs, copies of
// targets' regions, made again if a bind asks), a turn a second, kTurnBytes
// of copies a turn, one block at a time, stopped by a load. The policy here,
// kept apart from Vulkan so a test can drive it: when one begins, which block,
// whether its copies fit the others' holes as the heap would place them
// (host/heap_place.h) and leave room for the largest target the game makes in
// play - a block made on the command processor costs ~190 ms on the 680M,
// more than the memory is worth - the turns' pace, when it stops, and how it
// shares the avoided block with the targets' drain, which goes first after an
// arrival.
#include "host/heap_place.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace heap_compact {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kBlock = 256 * kMiB;     // gpu.cpp kHeapBlock
constexpr std::uint64_t kRoomOver = 64 * kMiB;   // the heap's free room: a block and this much before one is emptied
constexpr std::uint64_t kMostUsed = 160 * kMiB;  // a block holding more stays
constexpr std::uint64_t kSpare = 32 * kMiB;      // the others' room over what moves into it
constexpr std::uint64_t kTurnBytes = 32 * kMiB;  // copied a turn
constexpr std::size_t kTurnItems = 32;           // images copied a turn: each is driver calls on the command processor
constexpr std::uint64_t kTurnMs = 1000;          // a turn a second
constexpr std::uint64_t kGiveUpMs = 30000;       // a block not empty by then is left as it is
constexpr std::uint64_t kQuietMs = 30000;        // after one block's compaction, before the next
constexpr std::uint64_t kHotMs = 20000;          // a texture bound this recently is copied, not let go
constexpr std::uint64_t kAlign = 64 * 1024;      // the alignment a moved image is placed at in the trial (images ask 256 B-64 KiB)
constexpr std::uint64_t kSettleFlips = 900;      // flips since the latest load as well: 30 s at 30 fps
constexpr std::size_t kTries = 4;                // blocks looked at, emptiest first
constexpr std::size_t kMostCopies = kTurnItems * 20;  // images a block may have to copy: 20 turns, well inside kGiveUpMs

// One of the image heap's blocks (gpu.cpp HeapBlock).
struct Block {
    int cls = -1;  // heap_place::kTargets or kTextures; -1 empty
    std::uint64_t used = 0;
    std::map<std::uint64_t, std::uint64_t> free;  // offset -> size, coalesced
    std::uint32_t type = 0;                       // its memory type
    bool live = false;                            // made, in video memory
};

// An image the compaction copies out of a block: its size and the class of
// blocks it may go in (a target, snapshot or GPU-written surface the targets'
// blocks; a texture its own blocks, then the targets' holes).
struct Item {
    std::uint64_t size = 0;
    int cls = heap_place::kTargets;
};

// Why no block was chosen - the emptiest block's reason.
enum Why { kChosen, kNone, kRoom, kFull, kPinned, kMany, kSpareShort, kNoFit, kKeep, kWhyCount };

// A compaction may begin: on, the world settled, no loading screen, none
// under way, the last one's quiet over, and no block lying empty (the keeper
// gives those back first).
inline bool may_begin(bool on, bool settled, bool loading, bool busy, bool empty_block, std::uint64_t now_ms, std::uint64_t quiet_until_ms) {
    return on && settled && !loading && !busy && !empty_block && now_ms >= quiet_until_ms;
}

// The world settled by flips as well as by time: kSettleFlips since the
// latest load ended. On a slow device the 30 s of the settled world are a few
// frames - lavapipe at 0.3 fps had the area still making its images when one
// began, and a block was made on the command processor in play.
inline bool flips_settled(std::uint64_t flip_now, std::uint64_t load_end_flip) { return flip_now - load_end_flip >= kSettleFlips; }

// The block no image goes in while it is emptied is one slot (gpu.cpp
// g_heap_avoid), which the targets' drain and the compaction share: the drain
// empties a nearly empty targets' block within seconds of an arrival, the
// compaction takes what it leaves once the world has settled. Either begins
// only while the slot is free, and goes on only with the block it took - one
// another holds is not its own, and one whose slot a new block reused (gpu.cpp
// heap_add_block_locked) ends its turn.
inline bool slot_free(int avoided) { return avoided < 0; }
inline bool slot_mine(int mine, int avoided) { return mine >= 0 && mine == avoided; }

// One under way stops: a load (a loading screen, or the world not settled
// since one - a death, a lamp) or kGiveUpMs since it began.
inline bool must_stop(bool loading, bool settled, std::uint64_t now_ms, std::uint64_t began_ms) {
    return loading || !settled || now_ms - began_ms >= kGiveUpMs;
}

// A turn is due kTurnMs after the last.
inline bool turn_due(std::uint64_t now_ms, std::uint64_t last_ms) { return now_ms - last_ms >= kTurnMs; }

// The copies' credit at a turn: kTurnBytes more, never over kTurnBytes. An
// image is copied while the credit is above 0 and the turn has copied fewer
// than kTurnItems, and takes its size off the credit, so one larger than a
// turn's leaves a debt the next turns pay first: never more than kTurnBytes a
// second over time, and a target larger than that moves too.
inline std::int64_t turn_credit(std::int64_t credit) {
    return std::min<std::int64_t>(credit + static_cast<std::int64_t>(kTurnBytes), static_cast<std::int64_t>(kTurnBytes));
}
inline bool may_copy(std::int64_t credit, std::size_t copied_this_turn) { return credit > 0 && copied_this_turn < kTurnItems; }

// The free room in the live blocks of memory type `type` that hold anything,
// and whether one of them lies empty.
inline std::uint64_t room(const std::vector<Block>& blocks, std::uint32_t type, bool* empty_block) {
    std::uint64_t r = 0;
    bool empty = false;
    for (const Block& b : blocks) {
        if (!b.live || b.type != type) continue;
        if (!b.used) {
            empty = true;
            continue;
        }
        r += kBlock - b.used;
    }
    if (empty_block) *empty_block = empty;
    return r;
}

// `items` placed, largest first, in the live blocks of `type` other than
// `skip` as heap_place_locked places them (fullest first by class, the first
// free range that takes one, never a new block), on a copy of the blocks:
// false when one finds no room; else, in `largest_after`, the largest free
// range then left in a block that holds anything.
inline bool try_place(std::vector<Block> blocks, int skip, std::uint32_t type, std::vector<Item> items, bool classes, bool overflow,
                      bool fullest, std::uint64_t& largest_after) {
    std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.size > b.size; });
    std::vector<std::size_t> idx;
    for (const Item& it : items) {
        idx.clear();
        for (std::size_t k = 0; k < blocks.size(); ++k) {
            const Block& b = blocks[k];
            if (b.live && b.type == type && static_cast<int>(k) != skip && kBlock - b.used >= it.size &&
                heap_place::class_fits(classes, overflow, b.cls, it.cls))
                idx.push_back(k);
        }
        heap_place::order(blocks, idx, it.cls, fullest);
        bool placed = false;
        for (const std::size_t k : idx) {
            std::uint64_t at = 0;
            if (!heap_place::first_fit(blocks[k].free, it.size, kAlign, at)) continue;
            heap_place::take(blocks[k].free, at, it.size);
            blocks[k].used += it.size;
            if (blocks[k].cls < 0) blocks[k].cls = it.cls;
            placed = true;
            break;
        }
        if (!placed) return false;
    }
    largest_after = 0;
    for (std::size_t k = 0; k < blocks.size(); ++k) {
        const Block& b = blocks[k];
        if (b.live && b.type == type && static_cast<int>(k) != skip && b.used) largest_after = std::max(largest_after, heap_place::largest(b.free));
    }
    return true;
}

// The block to empty, of the live blocks of `type` that hold anything: the
// emptiest whose copies (`items[k]`) fit the others' holes - when the heap's
// free room is a block and kRoomOver, the block holds kMostUsed at most,
// nothing that cannot move (`pinned[k]`) and no more than kMostCopies images
// to copy, the others' room is what it copies and kSpare, and its copies
// placed there leave a free range of `keep` (the largest render target the
// game makes in play) in a block that holds anything. The last block never.
// -1 when none, with the emptiest block's reason in `why`.
inline int choose(const std::vector<Block>& blocks, std::uint32_t type, const std::vector<std::vector<Item>>& items,
                  const std::vector<bool>& pinned, std::uint64_t keep, bool classes, bool overflow, bool fullest, Why* why) {
    const auto say = [why](Why w) {
        if (why) *why = w;
    };
    const std::uint64_t free_room = room(blocks, type, nullptr);
    std::vector<std::size_t> cand;
    for (std::size_t k = 0; k < blocks.size(); ++k)
        if (blocks[k].live && blocks[k].type == type && blocks[k].used) cand.push_back(k);
    if (cand.size() < 2) {
        say(kNone);
        return -1;
    }
    if (free_room < kBlock + kRoomOver) {
        say(kRoom);
        return -1;
    }
    std::stable_sort(cand.begin(), cand.end(), [&blocks](std::size_t a, std::size_t c) { return blocks[a].used < blocks[c].used; });
    Why first = kNone;
    for (std::size_t n = 0; n < cand.size() && n < kTries; ++n) {
        const std::size_t k = cand[n];
        const Block& b = blocks[k];
        std::uint64_t copies = 0;
        for (const Item& it : items[k]) copies += it.size;
        Why w = kChosen;
        std::uint64_t largest_after = 0;
        if (b.used > kMostUsed)
            w = kFull;
        else if (pinned[k])
            w = kPinned;
        else if (items[k].size() > kMostCopies)
            w = kMany;
        else if (free_room - (kBlock - b.used) < copies + kSpare)
            w = kSpareShort;
        else if (!try_place(blocks, static_cast<int>(k), type, items[k], classes, overflow, fullest, largest_after))
            w = kNoFit;
        else if (largest_after < keep)
            w = kKeep;
        if (n == 0) first = w;
        if (w == kChosen) {
            say(kChosen);
            return static_cast<int>(k);
        }
        if (w == kFull) break;  // the rest hold more
    }
    say(first);
    return -1;
}

}  // namespace heap_compact
